import csv
import glob
import hashlib
import json
import os
import pickle
import random
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from construct_prompts import (
    get_imagenet_classes,
    get_prompts_concrete,
    get_prompts_human_related,
    get_prompts_style,
)
from sana_controller import SanaVectorStore, register_vector_control_sana


MODEL_REGISTRY = {
    'small': 'Efficient-Large-Model/Sana_600M_512px_diffusers',
    'large': 'Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers',
}

DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_HOOK_POINTS = ['cross_attn', 'self_attn', 'residual']
ALPHAS_P1 = [0.1, 0.7, 1.4]
ALPHAS_P2_MASTER = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.1, 1.4]
VALIDATION_METRICS = ['clip_score', 'fid']
TEST_METRICS = ['clip_score', 'pick_score', 'image_reward', 'mps', 'vendi']
DEFAULT_VALIDATION_SIZE = 100
DEFAULT_TEST_SIZE = 2000

DEFAULT_SANA_BANK_MODE = 'per_concept_bank'

DEFAULT_BANK_CONCEPTS = [
    ('anime', 'style'),
    ('watercolor', 'style'),
    ('oil painting', 'style'),
    ('pixel art', 'style'),
    ('sketch', 'style'),
    ('photorealistic', 'style'),
    ('Snoopy', 'concrete'),
    ('sunglasses', 'concrete'),
    ('hat', 'concrete'),
    ('flowers', 'concrete'),
    ('snow', 'concrete'),
    ('fire', 'concrete'),
]

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_PICKSCORE_PROCESSOR = None
_PICKSCORE_MODEL = None
_IMAGEREWARD_MODEL = None


def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(path: str, payload: Any) -> None:
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'wb') as f:
        pickle.dump(payload, f)


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def resolve_model_id(model_alias: str = 'small', model_id: Optional[str] = None) -> str:
    if model_id:
        return model_id
    if model_alias not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model_alias={model_alias!r}. Expected one of {sorted(MODEL_REGISTRY)}')
    return MODEL_REGISTRY[model_alias]


def _safe_model_dir_name(model_alias: str, model_id: Optional[str] = None) -> str:
    resolved_model_id = resolve_model_id(model_alias=model_alias, model_id=model_id)
    safe_model_id = re.sub(r'[^A-Za-z0-9._-]+', '_', resolved_model_id)
    return f'{model_alias}__{safe_model_id}'


def ensure_sana_model_on_disk(
    model_alias: str = 'small',
    model_id: Optional[str] = None,
    local_model_root: str = './models_cache',
    cache_dir: str = './cache',
) -> Dict[str, Any]:
    from huggingface_hub import snapshot_download

    resolved_model_id = resolve_model_id(model_alias=model_alias, model_id=model_id)
    local_model_root = ensure_dir(local_model_root)
    local_model_path = os.path.join(local_model_root, _safe_model_dir_name(model_alias, model_id))
    model_index_path = os.path.join(local_model_path, 'model_index.json')

    if os.path.exists(model_index_path):
        return {
            'model_alias': model_alias,
            'model_id': resolved_model_id,
            'local_model_path': local_model_path,
            'downloaded': False,
        }

    ensure_dir(local_model_path)
    snapshot_download(
        repo_id=resolved_model_id,
        local_dir=local_model_path,
        cache_dir=cache_dir,
    )
    return {
        'model_alias': model_alias,
        'model_id': resolved_model_id,
        'local_model_path': local_model_path,
        'downloaded': True,
    }


def load_sana_pipeline(
    model_alias: str = 'small',
    model_id: Optional[str] = None,
    cache_dir: str = './cache',
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    model_path: Optional[str] = None,
    local_model_root: Optional[str] = None,
):
    from diffusers import SanaPipeline

    device = device or get_device()
    if torch_dtype is None:
        torch_dtype = torch.float16 if device == 'cuda' else torch.float32

    resolved_model_id = resolve_model_id(model_alias=model_alias, model_id=model_id)
    local_model_path = model_path
    if local_model_path is None and local_model_root is not None:
        model_info = ensure_sana_model_on_disk(
            model_alias=model_alias,
            model_id=model_id,
            local_model_root=local_model_root,
            cache_dir=cache_dir,
        )
        local_model_path = model_info['local_model_path']

    if local_model_path is not None:
        pipe = SanaPipeline.from_pretrained(
            local_model_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        setattr(pipe, 'local_model_path', local_model_path)
        pipe.to(device)
        return pipe

    try:
        pipe = SanaPipeline.from_pretrained(
            resolved_model_id,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
    except OSError:
        print('Could not fetch from Hub, loading from local cache...')
        pipe = SanaPipeline.from_pretrained(
            resolved_model_id,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    pipe.to(device)
    return pipe


def build_prompts(
    mode: str,
    concept_pos: str,
    num_prompts: int = 50,
    concept_neg: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    if mode == 'concrete':
        return get_prompts_concrete(num=num_prompts, concept_pos=concept_pos, concept_neg=concept_neg)
    if mode == 'style':
        return get_prompts_style(num=num_prompts, concept_pos=concept_pos, concept_neg=concept_neg)
    if mode == 'human-related':
        return get_prompts_human_related(concept_pos=concept_pos, concept_neg=concept_neg)
    raise ValueError(f'Unsupported prompt mode: {mode}')


def _safe_progress_name(value: str) -> str:
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', value).strip('._')
    return safe or 'item'


def _save_partial_steering_vectors(
    save_path: Optional[str],
    pos_vectors: Sequence[Dict[int, Dict[str, List[np.ndarray]]]],
    neg_vectors: Sequence[Dict[int, Dict[str, List[np.ndarray]]]],
    num_denoising_steps: int,
) -> Optional[Dict[int, Dict[str, List[np.ndarray]]]]:
    if not save_path or not pos_vectors or not neg_vectors or len(pos_vectors) != len(neg_vectors):
        return None
    steering_vectors = compute_steering_vectors(pos_vectors, neg_vectors, num_denoising_steps)
    save_pickle(save_path, steering_vectors)
    return steering_vectors


def collect_activations(
    pipe,
    prompts_pos: Sequence[str],
    prompts_neg: Sequence[str],
    num_denoising_steps: int,
    hook_point: str,
    device: str,
    checkpoint_path: Optional[str] = None,
    checkpoint_metadata: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[..., None]] = None,
):
    if len(prompts_pos) != len(prompts_neg):
        raise ValueError('prompts_pos and prompts_neg must have the same length')

    prompts_pos = list(prompts_pos)
    prompts_neg = list(prompts_neg)
    pos_vectors = []
    neg_vectors = []
    seed = 0
    checkpoint_payload = {
        'metadata': {
            **(checkpoint_metadata or {}),
            'prompts_pos': prompts_pos,
            'prompts_neg': prompts_neg,
            'num_denoising_steps': num_denoising_steps,
            'hook_point': hook_point,
        },
        'completed_prompts': 0,
        'pos_vectors': [],
        'neg_vectors': [],
    }

    if checkpoint_path and os.path.exists(checkpoint_path):
        saved_checkpoint = load_pickle(checkpoint_path)
        if saved_checkpoint.get('metadata') == checkpoint_payload['metadata']:
            pos_vectors = saved_checkpoint.get('pos_vectors', [])
            neg_vectors = saved_checkpoint.get('neg_vectors', [])
            if len(pos_vectors) != len(neg_vectors):
                raise ValueError(f'Checkpoint at {checkpoint_path} is corrupted: prompt counts do not match')
            if len(pos_vectors) > len(prompts_pos):
                raise ValueError(f'Checkpoint at {checkpoint_path} has more prompts than the current run')
            checkpoint_payload = saved_checkpoint
            if pos_vectors:
                print(
                    f'  Resuming activation collection from prompt {len(pos_vectors) + 1}/{len(prompts_pos)} '
                    f'using checkpoint {checkpoint_path}'
                )
        else:
            print(f'  Ignoring incompatible activation checkpoint at {checkpoint_path}')

    for i in range(len(pos_vectors), len(prompts_pos)):
        prompt_pos = prompts_pos[i]
        prompt_neg = prompts_neg[i]
        print(f'  Prompt {i + 1}/{len(prompts_pos)}: pos="{prompt_pos}", neg="{prompt_neg}"')

        controller = SanaVectorStore(device=device)
        controller.steer = False
        register_vector_control_sana(pipe.transformer, controller, hook_point=hook_point)
        _ = pipe(
            prompt=prompt_pos,
            num_inference_steps=num_denoising_steps,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        pos_vectors.append(controller.vector_store)

        controller = SanaVectorStore(device=device)
        controller.steer = False
        register_vector_control_sana(pipe.transformer, controller, hook_point=hook_point)
        _ = pipe(
            prompt=prompt_neg,
            num_inference_steps=num_denoising_steps,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        neg_vectors.append(controller.vector_store)

        if progress_callback is not None:
            progress_callback(i, prompt_pos, prompt_neg, pos_vectors, neg_vectors)

        if checkpoint_path:
            checkpoint_payload['completed_prompts'] = len(pos_vectors)
            checkpoint_payload['pos_vectors'] = pos_vectors
            checkpoint_payload['neg_vectors'] = neg_vectors
            save_pickle(checkpoint_path, checkpoint_payload)

    return pos_vectors, neg_vectors


def compute_steering_vectors(
    pos_vectors: Sequence[Dict[int, Dict[str, List[np.ndarray]]]],
    neg_vectors: Sequence[Dict[int, Dict[str, List[np.ndarray]]]],
    num_denoising_steps: int,
) -> Dict[int, Dict[str, List[np.ndarray]]]:
    num_layers = len(pos_vectors[0][0]['layers'])
    steering_vectors: Dict[int, Dict[str, List[np.ndarray]]] = {}

    for denoising_step in range(num_denoising_steps):
        steering_vectors[denoising_step] = {'layers': []}
        for layer_num in range(num_layers):
            pos_avg = np.mean(
                [pos_vectors[i][denoising_step]['layers'][layer_num] for i in range(len(pos_vectors))],
                axis=0,
            )
            neg_avg = np.mean(
                [neg_vectors[i][denoising_step]['layers'][layer_num] for i in range(len(neg_vectors))],
                axis=0,
            )
            sv = pos_avg - neg_avg
            norm = np.linalg.norm(sv)
            if np.isfinite(norm) and norm > 1e-8:
                sv = sv / norm
            else:
                sv = np.zeros_like(sv)
            steering_vectors[denoising_step]['layers'].append(sv.astype(np.float32))
    return steering_vectors


def compute_multi_concept_bank(
    pipe,
    hook_point: str,
    num_concepts: int,
    num_denoising_steps: int,
    device: str,
    aggregate_save_path: Optional[str] = None,
    activation_checkpoint_path: Optional[str] = None,
    partial_bank_path: Optional[str] = None,
) -> Dict[str, Dict[int, Dict[str, List[np.ndarray]]]]:
    imagenet_classes = get_imagenet_classes(num_concepts)
    prompts_pos = [f'a photo of {cls}' for cls in imagenet_classes[:num_concepts]]
    prompts_neg = ['a photo'] * len(prompts_pos)
    partial_bank = load_pickle(partial_bank_path) if partial_bank_path and os.path.exists(partial_bank_path) else {}

    def on_prompt_complete(
        _prompt_index: int,
        _prompt_pos: str,
        _prompt_neg: str,
        pos_vectors: Sequence[Dict[int, Dict[str, List[np.ndarray]]]],
        neg_vectors: Sequence[Dict[int, Dict[str, List[np.ndarray]]]],
    ) -> None:
        _save_partial_steering_vectors(
            aggregate_save_path,
            pos_vectors,
            neg_vectors,
            num_denoising_steps,
        )
        if partial_bank_path:
            concept_name = imagenet_classes[len(pos_vectors) - 1]
            partial_bank[concept_name] = compute_steering_vectors(
                [pos_vectors[-1]],
                [neg_vectors[-1]],
                num_denoising_steps=num_denoising_steps,
            )
            save_pickle(partial_bank_path, partial_bank)

    pos_vectors, neg_vectors = collect_activations(
        pipe=pipe,
        prompts_pos=prompts_pos,
        prompts_neg=prompts_neg,
        num_denoising_steps=num_denoising_steps,
        hook_point=hook_point,
        device=device,
        checkpoint_path=activation_checkpoint_path,
        checkpoint_metadata={
            'bank_mode': 'multi_concept',
            'hook_point': hook_point,
            'num_concepts': num_concepts,
        },
        progress_callback=on_prompt_complete,
    )

    concept_bank: Dict[str, Dict[int, Dict[str, List[np.ndarray]]]] = {}
    for concept_idx, concept_name in enumerate(imagenet_classes[:num_concepts]):
        concept_bank[concept_name] = compute_steering_vectors(
            [pos_vectors[concept_idx]],
            [neg_vectors[concept_idx]],
            num_denoising_steps=num_denoising_steps,
        )
    return concept_bank

def load_or_compute_multi_concept_bank(
    pipe,
    hook_point: str,
    num_concepts: int = 50,
    num_denoising_steps: int = 20,
    bank_dir: str = 'steering_vectors_sana',
    device: Optional[str] = None,
) -> Tuple[Dict[str, Dict[int, Dict[str, List[np.ndarray]]]], str]:
    device = device or get_device()
    ensure_dir(bank_dir)
    bank_path = os.path.join(bank_dir, f'sana_{hook_point}_{num_concepts}concepts_bank.pickle')
    if os.path.exists(bank_path):
        return load_pickle(bank_path), bank_path

    progress_dir = ensure_dir(os.path.join(bank_dir, f'{Path(bank_path).stem}_progress'))
    concept_bank = compute_multi_concept_bank(
        pipe=pipe,
        hook_point=hook_point,
        num_concepts=num_concepts,
        num_denoising_steps=num_denoising_steps,
        device=device,
        aggregate_save_path=os.path.join(progress_dir, f'sana_{hook_point}_{num_concepts}concepts.partial.pickle'),
        activation_checkpoint_path=os.path.join(progress_dir, 'activations.checkpoint.pickle'),
        partial_bank_path=os.path.join(progress_dir, 'concept_bank_partial.pickle'),
    )
    save_pickle(bank_path, concept_bank)
    return concept_bank, bank_path


def load_or_compute_per_concept_bank(
    pipe,
    hook_point: str,
    num_denoising_steps: int = 20,
    bank_dir: str = 'steering_vectors_sana',
    concepts: Optional[Sequence[Tuple[str, str]]] = None,
    device: Optional[str] = None,
) -> Tuple[Dict[str, Dict[int, Dict[str, List[np.ndarray]]]], str]:
    device = device or get_device()
    ensure_dir(bank_dir)
    bank_path = os.path.join(bank_dir, f'sana_{hook_point}_per_concept_bank.pickle')
    if os.path.exists(bank_path):
        return load_pickle(bank_path), bank_path

    concepts = list(concepts or DEFAULT_BANK_CONCEPTS)
    progress_dir = ensure_dir(os.path.join(bank_dir, f'{Path(bank_path).stem}_progress'))
    bank_progress_path = os.path.join(progress_dir, 'concept_bank_partial.pickle')
    progress_state_path = os.path.join(progress_dir, 'progress_state.json')
    progress_signature = {
        'hook_point': hook_point,
        'num_denoising_steps': num_denoising_steps,
        'concepts': [{'concept': concept, 'mode': mode} for concept, mode in concepts],
    }

    concept_bank = {}
    if os.path.exists(bank_progress_path) and os.path.exists(progress_state_path):
        saved_state = load_json(progress_state_path)
        if saved_state.get('signature') == progress_signature:
            concept_bank = load_pickle(bank_progress_path)
            progress_state = saved_state
        else:
            progress_state = {'signature': progress_signature, 'concepts': {}}
    else:
        progress_state = {'signature': progress_signature, 'concepts': {}}

    for idx, (concept, mode) in enumerate(concepts):
        concept_state = progress_state['concepts'].get(concept, {})
        if concept in concept_bank and concept_state.get('status') == 'complete':
            print(f'\n=== [{idx + 1}/{len(concepts)}] Concept: "{concept}" (mode={mode}) already completed, skipping ===')
            continue

        print(f'\n=== [{idx + 1}/{len(concepts)}] Concept: "{concept}" (mode={mode}) ===')
        prompts_pos, prompts_neg = build_prompts(mode, concept)
        concept_checkpoint_path = os.path.join(
            progress_dir,
            f'{idx:02d}_{_safe_progress_name(concept)}.checkpoint.pickle',
        )
        concept_partial_path = os.path.join(
            progress_dir,
            f'{idx:02d}_{_safe_progress_name(concept)}.partial_sv.pickle',
        )

        def on_prompt_complete(
            _prompt_index: int,
            _prompt_pos: str,
            _prompt_neg: str,
            pos_vectors: Sequence[Dict[int, Dict[str, List[np.ndarray]]]],
            neg_vectors: Sequence[Dict[int, Dict[str, List[np.ndarray]]]],
        ) -> None:
            partial_vectors = _save_partial_steering_vectors(
                concept_partial_path,
                pos_vectors,
                neg_vectors,
                num_denoising_steps,
            )
            if partial_vectors is not None:
                concept_bank[concept] = partial_vectors
                save_pickle(bank_progress_path, concept_bank)
            progress_state['concepts'][concept] = {
                'mode': mode,
                'status': 'partial',
                'completed_prompts': len(pos_vectors),
                'num_prompts': len(prompts_pos),
                'checkpoint_path': concept_checkpoint_path,
                'partial_sv_path': concept_partial_path,
            }
            save_json(progress_state_path, progress_state)

        pos_vecs, neg_vecs = collect_activations(
            pipe=pipe,
            prompts_pos=prompts_pos,
            prompts_neg=prompts_neg,
            num_denoising_steps=num_denoising_steps,
            hook_point=hook_point,
            device=device,
            checkpoint_path=concept_checkpoint_path,
            checkpoint_metadata={
                'concept': concept,
                'mode': mode,
                'bank_path': bank_path,
            },
            progress_callback=on_prompt_complete,
        )
        concept_bank[concept] = compute_steering_vectors(pos_vecs, neg_vecs, num_denoising_steps)
        save_pickle(bank_progress_path, concept_bank)
        progress_state['concepts'][concept] = {
            'mode': mode,
            'status': 'complete',
            'completed_prompts': len(prompts_pos),
            'num_prompts': len(prompts_pos),
            'checkpoint_path': concept_checkpoint_path,
            'partial_sv_path': concept_partial_path,
        }
        save_json(progress_state_path, progress_state)

    save_pickle(bank_path, concept_bank)
    return concept_bank, bank_path


def load_or_compute_concept_bank(
    pipe,
    hook_point: str,
    bank_mode: str = DEFAULT_SANA_BANK_MODE,
    num_concepts: int = 50,
    num_denoising_steps: int = 20,
    bank_dir: str = 'steering_vectors_sana',
    concepts: Optional[Sequence[Tuple[str, str]]] = None,
    device: Optional[str] = None,
) -> Tuple[Dict[str, Dict[int, Dict[str, List[np.ndarray]]]], str]:
    if bank_mode == 'per_concept_bank':
        return load_or_compute_per_concept_bank(
            pipe=pipe,
            hook_point=hook_point,
            num_denoising_steps=num_denoising_steps,
            bank_dir=bank_dir,
            concepts=concepts,
            device=device,
        )
    if bank_mode == 'multi_concept':
        return load_or_compute_multi_concept_bank(
            pipe=pipe,
            hook_point=hook_point,
            num_concepts=num_concepts,
            num_denoising_steps=num_denoising_steps,
            bank_dir=bank_dir,
            device=device,
    )
    raise ValueError(f'Unsupported bank_mode={bank_mode!r}')


def load_or_compute_hook_point_banks(
    pipe,
    hook_points: Sequence[str] = tuple(DEFAULT_HOOK_POINTS),
    bank_mode: str = DEFAULT_SANA_BANK_MODE,
    num_concepts: int = 50,
    num_denoising_steps: int = 20,
    bank_dir: str = 'steering_vectors_sana',
    concepts: Optional[Sequence[Tuple[str, str]]] = None,
    device: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    device = device or get_device()
    prepared_banks: Dict[str, Dict[str, Any]] = {}
    for hook_point in hook_points:
        concept_bank, bank_path = load_or_compute_concept_bank(
            pipe=pipe,
            hook_point=hook_point,
            bank_mode=bank_mode,
            num_concepts=num_concepts,
            num_denoising_steps=num_denoising_steps,
            bank_dir=bank_dir,
            concepts=concepts,
            device=device,
        )
        prepared_banks[hook_point] = {
            'concept_bank': concept_bank,
            'bank_path': bank_path,
        }
    return prepared_banks


def get_active_layers(strategy: str, num_layers: int, seed: int = 0):
    if strategy == 'all_layers':
        return None
    if strategy == 'late_layers':
        return set(range(num_layers // 2, num_layers))
    if strategy == 'early_layers':
        return set(range(0, num_layers // 2))
    if strategy == 'random_layers':
        rng = random.Random(seed)
        return set(rng.sample(range(num_layers), num_layers // 2))
    if strategy == 'timestep_scaled':
        return None
    raise ValueError(f'Unsupported strategy={strategy!r}')


def choose_concept_name(concept_names: Sequence[str], prompt_index: int, seed: int, concept_seed: int = 0) -> str:
    rng = random.Random(concept_seed + prompt_index * 1000 + seed)
    return concept_names[rng.randrange(len(concept_names))]


def template_from_concept_bank(
    concept_bank: Dict[str, Dict[int, Dict[str, List[np.ndarray]]]]
) -> Dict[int, Dict[str, List[np.ndarray]]]:
    if not concept_bank:
        raise ValueError('concept_bank is empty')
    first_concept = sorted(concept_bank.keys())[0]
    return concept_bank[first_concept]


def create_random_steering_vectors(
    template_vectors: Dict[int, Dict[str, List[np.ndarray]]],
    seed: int = 12345,
) -> Dict[int, Dict[str, List[np.ndarray]]]:
    rng = np.random.default_rng(seed)
    random_vectors: Dict[int, Dict[str, List[np.ndarray]]] = {}
    for step, payload in template_vectors.items():
        random_vectors[step] = {'layers': []}
        for layer_vector in payload['layers']:
            arr = np.asarray(layer_vector, dtype=np.float32)
            noise = rng.normal(0.0, 1.0, size=arr.shape).astype(np.float32)
            norm = np.linalg.norm(noise)
            if np.isfinite(norm) and norm > 1e-8:
                noise = noise / norm
            else:
                noise = np.zeros_like(arr, dtype=np.float32)
            random_vectors[step]['layers'].append(noise)
    return random_vectors


def save_random_steering_vectors(
    template_vectors: Dict[int, Dict[str, List[np.ndarray]]],
    output_path: str,
    seed: int = 12345,
) -> str:
    save_pickle(output_path, create_random_steering_vectors(template_vectors, seed=seed))
    return output_path


def _download_if_missing(url: str, path: str) -> None:
    if os.path.exists(path):
        return
    ensure_dir(os.path.dirname(path) or '.')
    urllib.request.urlretrieve(url, path)


def ensure_coco_val2017(coco_dir: str) -> Dict[str, str]:
    coco_dir = ensure_dir(coco_dir)
    annotations_path = os.path.join(coco_dir, 'annotations', 'captions_val2017.json')
    val_dir = os.path.join(coco_dir, 'val2017')

    if not os.path.exists(annotations_path):
        annotations_zip = os.path.join(coco_dir, 'annotations_trainval2017.zip')
        _download_if_missing('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', annotations_zip)
        with zipfile.ZipFile(annotations_zip) as zf:
            zf.extractall(coco_dir)

    if not os.path.exists(val_dir):
        val_zip = os.path.join(coco_dir, 'val2017.zip')
        _download_if_missing('http://images.cocodataset.org/zips/val2017.zip', val_zip)
        with zipfile.ZipFile(val_zip) as zf:
            zf.extractall(coco_dir)

    return {'annotations_path': annotations_path, 'val_dir': val_dir}


def download_coco_dataset(coco_dir: str) -> Dict[str, Any]:
    coco_paths = ensure_coco_val2017(coco_dir)
    num_images = len(list(Path(coco_paths['val_dir']).glob('*.jpg')))
    return {
        'coco_dir': coco_dir,
        'annotations_path': coco_paths['annotations_path'],
        'val_dir': coco_paths['val_dir'],
        'num_val_images': num_images,
    }


def load_coco_caption_records(coco_dir: str) -> List[Dict[str, Any]]:
    coco_paths = ensure_coco_val2017(coco_dir)
    coco_data = load_json(coco_paths['annotations_path'])
    image_to_caption = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_caption:
            image_to_caption[image_id] = ann['caption']

    records = []
    for image_id, caption in image_to_caption.items():
        records.append(
            {
                'image_id': int(image_id),
                'caption': caption,
                'real_image_path': os.path.join(coco_paths['val_dir'], f'{int(image_id):012d}.jpg'),
            }
        )
    return records


def _link_or_copy(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def create_reference_dir(records: Sequence[Dict[str, Any]], reference_dir: str) -> str:
    ensure_dir(reference_dir)
    for record in records:
        src = record['real_image_path']
        dst = os.path.join(reference_dir, os.path.basename(src))
        _link_or_copy(src, dst)
    return reference_dir


def create_coco_split_manifests(
    coco_dir: str,
    output_dir: str,
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    split_seed: int = 42,
    seeds: Optional[Sequence[int]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    seeds = list(seeds or DEFAULT_SEEDS)
    split_dir = ensure_dir(os.path.join(output_dir, 'splits'))
    validation_manifest_path = os.path.join(split_dir, 'validation_manifest.json')
    test_manifest_path = os.path.join(split_dir, 'test_manifest.json')

    if not overwrite and os.path.exists(validation_manifest_path) and os.path.exists(test_manifest_path):
        validation_manifest = load_json(validation_manifest_path)
        test_manifest = load_json(test_manifest_path)
    else:
        records = load_coco_caption_records(coco_dir)
        rng = random.Random(split_seed)
        rng.shuffle(records)
        required = validation_size + test_size
        if len(records) < required:
            raise ValueError(f'Need at least {required} COCO records, found {len(records)}')

        selected = records[:required]
        validation_records = selected[:validation_size]
        test_records = selected[validation_size:required]
        validation_manifest = []
        test_manifest = []

        for idx, record in enumerate(validation_records):
            validation_manifest.append(
                {
                    'split': 'validation',
                    'prompt_index': idx,
                    'image_id': record['image_id'],
                    'caption': record['caption'],
                    'real_image_path': record['real_image_path'],
                    'seeds': seeds,
                }
            )
        for idx, record in enumerate(test_records):
            test_manifest.append(
                {
                    'split': 'test',
                    'prompt_index': idx,
                    'image_id': record['image_id'],
                    'caption': record['caption'],
                    'real_image_path': record['real_image_path'],
                    'seeds': seeds,
                }
            )

        save_json(validation_manifest_path, validation_manifest)
        save_json(test_manifest_path, test_manifest)

    validation_reference_dir = create_reference_dir(validation_manifest, os.path.join(split_dir, 'validation_reference'))
    test_reference_dir = create_reference_dir(test_manifest, os.path.join(split_dir, 'test_reference'))
    return {
        'validation_manifest_path': validation_manifest_path,
        'test_manifest_path': test_manifest_path,
        'validation_reference_dir': validation_reference_dir,
        'test_reference_dir': test_reference_dir,
        'validation_manifest': validation_manifest,
        'test_manifest': test_manifest,
    }


def validate_split_manifests(
    validation_manifest: Sequence[Dict[str, Any]],
    test_manifest: Sequence[Dict[str, Any]],
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    images_per_prompt: int = 5,
) -> Dict[str, int]:
    val_ids = [record['image_id'] for record in validation_manifest]
    test_ids = [record['image_id'] for record in test_manifest]
    if len(set(val_ids)) != validation_size:
        raise ValueError('Validation split does not contain the expected number of unique image_id values')
    if len(set(test_ids)) != test_size:
        raise ValueError('Test split does not contain the expected number of unique image_id values')
    if set(val_ids) & set(test_ids):
        raise ValueError('Validation and test splits overlap')

    for manifest in (validation_manifest, test_manifest):
        for record in manifest:
            if len(record['seeds']) != images_per_prompt:
                raise ValueError(
                    f'Expected {images_per_prompt} seeds per prompt, got {len(record["seeds"])} for {record["image_id"]}'
                )

    return {
        'validation_unique_image_ids': len(set(val_ids)),
        'test_unique_image_ids': len(set(test_ids)),
        'images_per_prompt': images_per_prompt,
    }

def build_experiment_tag(split: str, variant: str, hook_point: Optional[str], alpha: Optional[float]) -> str:
    parts = [split, variant]
    if hook_point:
        parts.append(hook_point)
    if alpha is not None:
        parts.append(f'a{alpha}')
    return '_'.join(parts)


def write_experiment_config(experiment_dir: str, config: Dict[str, Any]) -> None:
    save_json(os.path.join(experiment_dir, 'config.json'), config)


def _prompt_dir(experiment_dir: str, prompt_index: int) -> str:
    return os.path.join(experiment_dir, f'prompt_{prompt_index:04d}')


def _prompt_image_path(experiment_dir: str, prompt_index: int, seed: int) -> str:
    return os.path.join(_prompt_dir(experiment_dir, prompt_index), f'seed{seed:02d}.png')


def _is_prompt_complete(experiment_dir: str, prompt_index: int, seeds: Sequence[int]) -> bool:
    return all(os.path.exists(_prompt_image_path(experiment_dir, prompt_index, seed)) for seed in seeds)


def _save_prompt_metadata(
    prompt_dir: str,
    record: Dict[str, Any],
    concept_names: Sequence[str],
    chosen_concepts: Optional[Dict[str, Optional[str]]] = None,
) -> None:
    save_json(
        os.path.join(prompt_dir, 'metadata.json'),
        {
            'prompt_index': record['prompt_index'],
            'image_id': record['image_id'],
            'caption': record['caption'],
            'real_image_path': record['real_image_path'],
            'seeds': record['seeds'],
            'concept_names': list(concept_names),
        },
    )
    if chosen_concepts is not None:
        save_json(os.path.join(prompt_dir, 'chosen_concepts.json'), chosen_concepts)


def generate_images_for_manifest(
    pipe,
    manifest: Sequence[Dict[str, Any]],
    experiment_dir: str,
    hook_point: str = 'cross_attn',
    alpha: Optional[float] = None,
    num_denoising_steps: int = 20,
    strategy: str = 'all_layers',
    steering_vectors: Optional[Dict[int, Dict[str, List[np.ndarray]]]] = None,
    concept_bank: Optional[Dict[str, Dict[int, Dict[str, List[np.ndarray]]]]] = None,
    baseline: bool = False,
    concept_seed: int = 0,
    device: Optional[str] = None,
    save_config: Optional[Dict[str, Any]] = None,
) -> str:
    device = device or get_device()
    ensure_dir(experiment_dir)
    if save_config is not None:
        write_experiment_config(experiment_dir, save_config)

    num_layers = len(pipe.transformer.transformer_blocks)
    concept_names = sorted(concept_bank.keys()) if concept_bank else []

    for record in manifest:
        prompt_dir = ensure_dir(_prompt_dir(experiment_dir, record['prompt_index']))
        chosen_concepts: Dict[str, Optional[str]] = {}
        with open(os.path.join(prompt_dir, 'prompt.txt'), 'w', encoding='utf-8') as f:
            f.write(record['caption'])

        if _is_prompt_complete(experiment_dir, record['prompt_index'], record['seeds']):
            chosen_concepts_path = os.path.join(prompt_dir, 'chosen_concepts.json')
            if os.path.exists(chosen_concepts_path):
                chosen_concepts = load_json(chosen_concepts_path)
            elif concept_bank is not None:
                chosen_concepts = {
                    str(seed): choose_concept_name(
                        concept_names=concept_names,
                        prompt_index=record['prompt_index'],
                        seed=seed,
                        concept_seed=concept_seed,
                    )
                    for seed in record['seeds']
                }
            _save_prompt_metadata(prompt_dir, record, concept_names, chosen_concepts)
            continue

        for seed in record['seeds']:
            image_path = _prompt_image_path(experiment_dir, record['prompt_index'], seed)
            if os.path.exists(image_path):
                chosen_concepts[str(seed)] = (
                    choose_concept_name(
                        concept_names=concept_names,
                        prompt_index=record['prompt_index'],
                        seed=seed,
                        concept_seed=concept_seed,
                    )
                    if concept_bank is not None
                    else None
                )
                continue

            chosen_concept = None
            if baseline:
                controller = SanaVectorStore(device=device)
                controller.steer = False
                chosen_concepts[str(seed)] = None
            else:
                selected_vectors = steering_vectors
                if concept_bank is not None:
                    chosen_concept = choose_concept_name(
                        concept_names=concept_names,
                        prompt_index=record['prompt_index'],
                        seed=seed,
                        concept_seed=concept_seed,
                    )
                    selected_vectors = concept_bank[chosen_concept]
                chosen_concepts[str(seed)] = chosen_concept

                controller = SanaVectorStore(
                    steering_vectors=selected_vectors,
                    steer=True,
                    alpha=alpha if alpha is not None else 0.0,
                    active_layers=get_active_layers(strategy, num_layers, seed=seed),
                    timestep_scaling=(strategy == 'timestep_scaled'),
                    total_steps=num_denoising_steps,
                    device=device,
                )

            register_vector_control_sana(pipe.transformer, controller, hook_point=hook_point)
            image = pipe(
                prompt=record['caption'],
                num_inference_steps=num_denoising_steps,
                generator=torch.Generator(device=device).manual_seed(seed),
            ).images[0]
            image.save(image_path)

            if chosen_concept:
                print(f'[prompt={record["prompt_index"]:04d}] seed={seed:02d} concept={chosen_concept} -> {image_path}')
            else:
                print(f'[prompt={record["prompt_index"]:04d}] seed={seed:02d} -> {image_path}')

        _save_prompt_metadata(prompt_dir, record, concept_names, chosen_concepts)
    return experiment_dir


def load_prompt_images(prompt_dir: str) -> Tuple[List[Image.Image], List[str]]:
    image_paths = sorted(glob.glob(os.path.join(prompt_dir, '*.png')))
    images = [Image.open(path).convert('RGB') for path in image_paths]
    return images, image_paths


def _prompt_dir_index(prompt_dir: str) -> int:
    match = re.search(r'prompt_(\d+)$', os.path.basename(prompt_dir))
    if not match:
        raise ValueError(f'Could not parse prompt index from {prompt_dir}')
    return int(match.group(1))


def _default_prompt_limit_for_split(split: Optional[str]) -> Optional[int]:
    if split == 'validation':
        return DEFAULT_VALIDATION_SIZE
    if split == 'test':
        return DEFAULT_TEST_SIZE
    return None


def _infer_experiment_split(experiment_dir: str) -> Optional[str]:
    config_path = os.path.join(experiment_dir, 'config.json')
    if os.path.exists(config_path):
        config = load_json(config_path)
        split = config.get('split')
        if split in {'validation', 'test'}:
            return split

    experiment_name = os.path.basename(os.path.normpath(experiment_dir))
    if experiment_name.startswith('validation_'):
        return 'validation'
    if experiment_name.startswith('test_'):
        return 'test'
    return None


def _resolve_evaluation_prompt_limit(experiment_dir: str, prompt_limit: Optional[int] = None) -> Optional[int]:
    if prompt_limit is not None:
        return int(prompt_limit)

    config_path = os.path.join(experiment_dir, 'config.json')
    if os.path.exists(config_path):
        config = load_json(config_path)
        stored_limit = config.get('evaluation_prompt_limit')
        if stored_limit is not None:
            return int(stored_limit)

    split = _infer_experiment_split(experiment_dir)
    return _default_prompt_limit_for_split(split)


def _collect_prompt_dirs(experiment_dir: str, prompt_limit: Optional[int] = None) -> List[str]:
    prompt_dirs = sorted(
        glob.glob(os.path.join(experiment_dir, 'prompt_*')),
        key=_prompt_dir_index,
    )
    if prompt_limit is None:
        return prompt_dirs
    return [prompt_dir for prompt_dir in prompt_dirs if _prompt_dir_index(prompt_dir) < prompt_limit]


def _load_clip(device: str):
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS
    import open_clip

    _CLIP_MODEL, _, _CLIP_PREPROCESS = open_clip.create_model_and_transforms(
        'ViT-B-32',
        pretrained='laion2b_s34b_b79k',
    )
    _CLIP_MODEL = _CLIP_MODEL.to(device).eval()
    return _CLIP_MODEL, _CLIP_PREPROCESS


def get_clip_image_features(images: Sequence[Image.Image], device: Optional[str] = None) -> torch.Tensor:
    device = device or get_device()
    clip_model, clip_preprocess = _load_clip(device)
    feats = []
    for img in images:
        img_t = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = clip_model.encode_image(img_t)
            feat = F.normalize(feat, dim=-1)
        feats.append(feat)
    return torch.cat(feats, dim=0)


def get_clip_text_features(texts: Sequence[str], device: Optional[str] = None) -> torch.Tensor:
    device = device or get_device()
    clip_model, _ = _load_clip(device)
    import open_clip

    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    tokens = tokenizer(list(texts)).to(device)
    with torch.no_grad():
        feat = clip_model.encode_text(tokens)
        feat = F.normalize(feat, dim=-1)
    return feat


def compute_clip_score(images: Sequence[Image.Image], prompt: str, device: Optional[str] = None) -> List[float]:
    img_feats = get_clip_image_features(images, device=device)
    txt_feats = get_clip_text_features([prompt], device=device)
    return (img_feats @ txt_feats.T).squeeze(-1).cpu().numpy().tolist()


def compute_mps(images: Sequence[Image.Image], device: Optional[str] = None) -> float:
    device = device or get_device()
    if len(images) < 2:
        return 1.0
    img_feats = get_clip_image_features(images, device=device)
    sim_matrix = img_feats @ img_feats.T
    n = len(images)
    mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
    return sim_matrix[mask].mean().item()


def compute_vendi_score(images: Sequence[Image.Image], device: Optional[str] = None) -> float:
    from vendi_score import vendi

    img_feats = get_clip_image_features(images, device=device)
    sim_matrix = (img_feats @ img_feats.T).cpu().numpy()
    return float(vendi.score_K(sim_matrix))


def _load_pickscore(device: str):
    global _PICKSCORE_PROCESSOR, _PICKSCORE_MODEL
    if _PICKSCORE_MODEL is not None:
        return _PICKSCORE_PROCESSOR, _PICKSCORE_MODEL
    from transformers import AutoModel, AutoProcessor

    _PICKSCORE_PROCESSOR = AutoProcessor.from_pretrained(
        'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        use_fast=False,
    )
    _PICKSCORE_MODEL = AutoModel.from_pretrained('yuvalkirstain/PickScore_v1').eval().to(device)
    return _PICKSCORE_PROCESSOR, _PICKSCORE_MODEL


def compute_pick_score(images: Sequence[Image.Image], prompt: str, device: Optional[str] = None) -> List[float]:
    device = device or get_device()
    processor, model = _load_pickscore(device)
    scores = []
    for img in images:
        inputs = processor(text=prompt, images=img, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            scores.append(model(**inputs).logits_per_image.item())
    return scores


def _load_imagereward():
    global _IMAGEREWARD_MODEL
    if _IMAGEREWARD_MODEL is not None:
        return _IMAGEREWARD_MODEL
    import ImageReward as RM

    _IMAGEREWARD_MODEL = RM.load('ImageReward-v1.0')
    return _IMAGEREWARD_MODEL


def compute_image_reward(images: Sequence[Image.Image], prompt: str) -> List[float]:
    reward_model = _load_imagereward()
    return [reward_model.score(prompt, img) for img in images]


def _ensure_subset_reference_dir(
    experiment_dir: str,
    prompt_dirs: Sequence[str],
    fid_reference_dir: str,
) -> str:
    subset_dir = os.path.join(experiment_dir, '_fid_reference_subset')
    if os.path.exists(subset_dir):
        shutil.rmtree(subset_dir)
    ensure_dir(subset_dir)

    copied_any = False
    for prompt_dir in prompt_dirs:
        metadata_path = os.path.join(prompt_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            return fid_reference_dir

        metadata = load_json(metadata_path)
        real_image_path = metadata.get('real_image_path')
        if not real_image_path or not os.path.exists(real_image_path):
            return fid_reference_dir

        dst = os.path.join(subset_dir, os.path.basename(real_image_path))
        _link_or_copy(real_image_path, dst)
        copied_any = True

    return subset_dir if copied_any else fid_reference_dir


def ensure_flat_image_dir(experiment_dir: str, prompt_dirs: Optional[Sequence[str]] = None) -> str:
    flat_dir = os.path.join(experiment_dir, '_fid_images')
    if os.path.exists(flat_dir):
        shutil.rmtree(flat_dir)
    ensure_dir(flat_dir)

    prompt_dirs = list(prompt_dirs) if prompt_dirs is not None else _collect_prompt_dirs(experiment_dir)
    for prompt_dir in prompt_dirs:
        prompt_name = os.path.basename(prompt_dir)
        for image_path in sorted(glob.glob(os.path.join(prompt_dir, '*.png'))):
            flat_name = f'{prompt_name}_{os.path.basename(image_path)}'
            dst = os.path.join(flat_dir, flat_name)
            _link_or_copy(image_path, dst)
    return flat_dir


def compute_fid_score(
    experiment_dir: str,
    fid_reference_dir: str,
    prompt_dirs: Optional[Sequence[str]] = None,
) -> float:
    from cleanfid import fid

    flat_dir = ensure_flat_image_dir(experiment_dir, prompt_dirs=prompt_dirs)
    reference_dir = fid_reference_dir
    if prompt_dirs is not None:
        reference_dir = _ensure_subset_reference_dir(
            experiment_dir=experiment_dir,
            prompt_dirs=prompt_dirs,
            fid_reference_dir=fid_reference_dir,
        )
    return float(fid.compute_fid(flat_dir, reference_dir))


def evaluate_experiment_dir(
    experiment_dir: str,
    metrics: Sequence[str],
    fid_reference_dir: Optional[str] = None,
    device: Optional[str] = None,
    prompt_limit: Optional[int] = None,
) -> Dict[str, Any]:
    device = device or get_device()
    prompt_limit = _resolve_evaluation_prompt_limit(experiment_dir, prompt_limit=prompt_limit)
    prompt_dirs = _collect_prompt_dirs(experiment_dir, prompt_limit=prompt_limit)
    if not prompt_dirs:
        raise ValueError(f'No prompt_* directories found in {experiment_dir}')

    result: Dict[str, Any] = {'per_prompt': {}, 'aggregate': {}}
    for prompt_dir in prompt_dirs:
        prompt_name = os.path.basename(prompt_dir)
        prompt = Path(os.path.join(prompt_dir, 'prompt.txt')).read_text(encoding='utf-8').strip()
        images, image_paths = load_prompt_images(prompt_dir)
        if not images:
            continue

        metrics_for_prompt: Dict[str, Any] = {'prompt': prompt, 'num_images': len(image_paths)}
        if 'clip_score' in metrics:
            scores = compute_clip_score(images, prompt, device=device)
            metrics_for_prompt['clip_score_mean'] = float(np.mean(scores))
            metrics_for_prompt['clip_score_std'] = float(np.std(scores))
        if 'mps' in metrics:
            metrics_for_prompt['mps'] = compute_mps(images, device=device)
        if 'vendi' in metrics:
            metrics_for_prompt['vendi_score'] = compute_vendi_score(images, device=device)
        if 'pick_score' in metrics:
            scores = compute_pick_score(images, prompt, device=device)
            metrics_for_prompt['pick_score_mean'] = float(np.mean(scores))
            metrics_for_prompt['pick_score_std'] = float(np.std(scores))
        if 'image_reward' in metrics:
            scores = compute_image_reward(images, prompt)
            metrics_for_prompt['image_reward_mean'] = float(np.mean(scores))
            metrics_for_prompt['image_reward_std'] = float(np.std(scores))
        result['per_prompt'][prompt_name] = metrics_for_prompt

    aggregate = {}
    for key in ['clip_score_mean', 'mps', 'vendi_score', 'pick_score_mean', 'image_reward_mean']:
        values = [item[key] for item in result['per_prompt'].values() if key in item]
        if values:
            aggregate[key] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}

    if 'fid' in metrics:
        if not fid_reference_dir:
            raise ValueError('fid_reference_dir is required when requesting FID')
        aggregate['fid'] = compute_fid_score(
            experiment_dir,
            fid_reference_dir,
            prompt_dirs=prompt_dirs,
        )

    aggregate['num_prompts'] = len(result['per_prompt'])
    aggregate['num_images'] = sum(item['num_images'] for item in result['per_prompt'].values())
    if prompt_limit is not None:
        aggregate['prompt_limit'] = prompt_limit
    result['aggregate'] = aggregate
    return result


def write_metrics_json(experiment_dir: str, metrics_payload: Dict[str, Any]) -> str:
    metrics_path = os.path.join(experiment_dir, 'metrics.json')
    save_json(metrics_path, metrics_payload)
    return metrics_path


def find_experiment_dirs(root_dir: str) -> List[str]:
    experiment_dirs = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        prompt_subdirs = [d for d in dirnames if d.startswith('prompt_')]
        if prompt_subdirs:
            experiment_dirs.append(dirpath)
            dirnames[:] = [d for d in dirnames if not d.startswith('prompt_')]
    return sorted(experiment_dirs)


def _row_metric_value(row: Dict[str, Any], metric_name: str) -> float:
    value = row.get(metric_name)
    if value is None:
        value = row.get(f'{metric_name}_mean')
    if value is None:
        raise KeyError(metric_name)
    return float(value)


def _dominates(row_a: Dict[str, Any], row_b: Dict[str, Any]) -> bool:
    clip_a = _row_metric_value(row_a, 'clip_score_mean')
    clip_b = _row_metric_value(row_b, 'clip_score_mean')
    fid_a = _row_metric_value(row_a, 'fid')
    fid_b = _row_metric_value(row_b, 'fid')
    return (
        clip_a >= clip_b
        and fid_a <= fid_b
        and (clip_a > clip_b or fid_a < fid_b)
    )


def assign_pareto_ranks(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    remaining = [dict(row) for row in rows]
    ranked = []
    rank = 1
    while remaining:
        front = []
        for row in remaining:
            if not any(_dominates(other, row) for other in remaining if other is not row):
                front.append(row)
        for row in front:
            row['pareto_rank'] = rank
            ranked.append(row)
        remaining = [row for row in remaining if row not in front]
        rank += 1
    return ranked


def add_selection_scores(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in rows]
    clips = np.asarray([_row_metric_value(row, 'clip_score_mean') for row in rows], dtype=np.float32)
    fids = np.asarray([_row_metric_value(row, 'fid') for row in rows], dtype=np.float32)
    clip_min, clip_max = float(clips.min()), float(clips.max())
    fid_min, fid_max = float(fids.min()), float(fids.max())
    for row in rows:
        clip_value = _row_metric_value(row, 'clip_score_mean')
        fid_value = _row_metric_value(row, 'fid')
        clip_norm = (clip_value - clip_min) / max(clip_max - clip_min, 1e-8)
        fid_norm = (fid_value - fid_min) / max(fid_max - fid_min, 1e-8)
        row['selection_score'] = float(clip_norm + (1.0 - fid_norm))
        row['is_selected_best'] = False
    return rows


def select_best_validation_row(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ranked = assign_pareto_ranks(rows)
    scored = add_selection_scores(ranked)
    pareto_rows = [row for row in scored if row['pareto_rank'] == 1]
    best = max(
        pareto_rows,
        key=lambda row: (
            row['selection_score'],
            _row_metric_value(row, 'clip_score_mean'),
            -_row_metric_value(row, 'fid'),
        ),
    )
    for row in scored:
        if row['config_id'] == best['config_id']:
            row['is_selected_best'] = True
    return best, scored


def build_refinement_window(
    best_alpha: float,
    master_alphas: Sequence[float] = ALPHAS_P2_MASTER,
    window_size: int = 5,
) -> List[float]:
    if best_alpha not in master_alphas:
        raise ValueError(f'best_alpha={best_alpha} is not present in master_alphas')
    idx = master_alphas.index(best_alpha)
    left = idx
    right = idx
    selected = {idx}
    while len(selected) < min(window_size, len(master_alphas)):
        expanded = False
        if left - 1 >= 0:
            left -= 1
            selected.add(left)
            expanded = True
        if len(selected) >= window_size:
            break
        if right + 1 < len(master_alphas):
            right += 1
            selected.add(right)
            expanded = True
        if not expanded:
            break
    return [master_alphas[i] for i in sorted(selected)]


def flatten_metrics_for_summary(
    metrics_payload: Dict[str, Any],
    split: str,
    variant: str,
    model_alias: str,
    hook_point: Optional[str],
    alpha: Optional[float],
) -> Dict[str, Any]:
    aggregate = metrics_payload['aggregate']
    row = {
        'split': split,
        'variant': variant,
        'model_alias': model_alias,
        'hook_point': hook_point,
        'alpha': alpha,
        'num_prompts': aggregate.get('num_prompts'),
        'num_images': aggregate.get('num_images'),
    }
    for metric_name in ['clip_score_mean', 'mps', 'vendi_score', 'pick_score_mean', 'image_reward_mean']:
        metric_value = aggregate.get(metric_name)
        if isinstance(metric_value, dict):
            row[metric_name] = metric_value['mean']
            row[f'{metric_name}_mean'] = metric_value['mean']
            row[f'{metric_name}_std'] = metric_value['std']
        elif metric_value is not None:
            row[metric_name] = metric_value
    if 'fid' in aggregate:
        row['fid'] = aggregate['fid']
    return row


def write_summary_csv(rows: Sequence[Dict[str, Any]], output_path: str) -> str:
    ensure_dir(os.path.dirname(output_path) or '.')
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def make_config_fingerprint(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


def _validation_metrics_are_complete(metrics_payload: Dict[str, Any]) -> bool:
    aggregate = metrics_payload.get('aggregate', {})
    return 'clip_score_mean' in aggregate and 'fid' in aggregate


def load_cached_validation_metrics(experiment_dir: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    config_path = os.path.join(experiment_dir, 'config.json')
    metrics_path = os.path.join(experiment_dir, 'metrics.json')
    if not (os.path.exists(config_path) and os.path.exists(metrics_path)):
        return None

    stored_config = load_json(config_path)
    expected_fp = make_config_fingerprint(config)
    stored_fp = stored_config.get('config_fingerprint') or make_config_fingerprint(
        {k: v for k, v in stored_config.items() if k != 'config_fingerprint'}
    )
    if stored_fp != expected_fp:
        return None

    metrics_payload = load_json(metrics_path)
    if not _validation_metrics_are_complete(metrics_payload):
        return None
    return metrics_payload


def _stored_config_fingerprint(config_path: str) -> Optional[str]:
    if not os.path.exists(config_path):
        return None
    stored_config = load_json(config_path)
    return stored_config.get('config_fingerprint') or make_config_fingerprint(
        {k: v for k, v in stored_config.items() if k != 'config_fingerprint'}
    )


def resolve_validation_experiment_dir(
    output_root: str,
    variant: str,
    hook_point: str,
    alpha: float,
    config_base: Dict[str, Any],
) -> Tuple[str, str]:
    config_fingerprint = make_config_fingerprint(config_base)
    experiment_name = build_experiment_tag('validation', variant, hook_point, alpha)
    canonical_dir = os.path.join(output_root, experiment_name)
    canonical_config_path = os.path.join(canonical_dir, 'config.json')

    if os.path.exists(canonical_config_path):
        if _stored_config_fingerprint(canonical_config_path) == config_fingerprint:
            return canonical_dir, config_fingerprint
    else:
        legacy_candidates = sorted(glob.glob(os.path.join(output_root, f'{experiment_name}__*')))
        for candidate in legacy_candidates:
            candidate_config_path = os.path.join(candidate, 'config.json')
            if _stored_config_fingerprint(candidate_config_path) == config_fingerprint:
                os.rename(candidate, canonical_dir)
                return canonical_dir, config_fingerprint
        return canonical_dir, config_fingerprint

    fallback_dir = os.path.join(output_root, f'{experiment_name}__cfg_{config_fingerprint[:10]}')
    return fallback_dir, config_fingerprint


def save_validation_plot(rows: Sequence[Dict[str, Any]], output_path: str, title: str = 'Validation Sweep') -> str:
    import matplotlib.pyplot as plt

    ensure_dir(os.path.dirname(output_path) or '.')
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get('hook_point')), []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for hook_point, hook_rows in sorted(grouped.items()):
        hook_rows = sorted(hook_rows, key=lambda item: float(item['alpha']))
        alphas = [float(item['alpha']) for item in hook_rows]
        clips = [_row_metric_value(item, 'clip_score_mean') for item in hook_rows]
        fids = [_row_metric_value(item, 'fid') for item in hook_rows]
        axes[0].plot(alphas, clips, marker='o', label=hook_point)
        axes[1].plot(alphas, fids, marker='o', label=hook_point)

    axes[0].set_title('Alpha vs CLIPScore')
    axes[0].set_xlabel('alpha')
    axes[0].set_ylabel('CLIPScore')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Alpha vs FID')
    axes[1].set_xlabel('alpha')
    axes[1].set_ylabel('FID')
    axes[1].grid(True, alpha=0.3)

    if grouped:
        axes[0].legend()
        axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def smoke_test_setup(
    model_aliases: Sequence[str] = ('small', 'large'),
    hook_points: Sequence[str] = tuple(DEFAULT_HOOK_POINTS),
    steps: int = 1,
    prompt: str = 'a red apple on a table',
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    device = device or get_device()
    results = []
    for model_alias in model_aliases:
        status = {'model_alias': model_alias, 'loaded': False, 'forward_pass': False, 'hook_points': {}}
        pipe = load_sana_pipeline(model_alias=model_alias, device=device)
        status['loaded'] = True
        for hook_point in hook_points:
            controller = SanaVectorStore(device=device)
            controller.steer = False
            register_vector_control_sana(pipe.transformer, controller, hook_point=hook_point)
            _ = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                generator=torch.Generator(device=device).manual_seed(0),
            )
            status['hook_points'][hook_point] = 'ok'
        status['forward_pass'] = True
        results.append(status)
    return results


def run_validation_sweep(
    pipe,
    validation_manifest: Sequence[Dict[str, Any]],
    validation_reference_dir: str,
    output_root: str,
    model_alias: str,
    num_denoising_steps: int = 20,
    strategy: str = 'all_layers',
    num_concepts: int = 50,
    hook_points: Sequence[str] = tuple(DEFAULT_HOOK_POINTS),
    alphas: Sequence[float] = tuple(ALPHAS_P1),
    bank_mode: str = DEFAULT_SANA_BANK_MODE,
    bank_dir: str = 'steering_vectors_sana',
    prepared_banks: Optional[Dict[str, Dict[str, Any]]] = None,
    device: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    device = device or get_device()
    sweep_rows = []
    ensure_dir(output_root)

    prepared_banks = prepared_banks or load_or_compute_hook_point_banks(
        pipe=pipe,
        hook_points=hook_points,
        bank_mode=bank_mode,
        num_concepts=num_concepts,
        num_denoising_steps=num_denoising_steps,
        bank_dir=bank_dir,
        device=device,
    )

    for hook_point in hook_points:
        if hook_point not in prepared_banks:
            raise ValueError(f'Missing prepared bank for hook_point={hook_point!r}')
        concept_bank = prepared_banks[hook_point]['concept_bank']
        bank_path = prepared_banks[hook_point]['bank_path']
        for alpha in alphas:
            variant = 'best_steering'
            prompt_limit = len(validation_manifest)
            config_base = {
                'split': 'validation',
                'variant': variant,
                'model_alias': model_alias,
                'model_id': resolve_model_id(model_alias=model_alias),
                'hook_point': hook_point,
                'alpha': alpha,
                'num_denoising_steps': num_denoising_steps,
                'strategy': strategy,
                'num_concepts': num_concepts,
                'bank_mode': bank_mode,
                'steering_bank_path': bank_path,
                'evaluation_prompt_limit': prompt_limit,
                'metrics': VALIDATION_METRICS,
            }
            experiment_dir, config_fingerprint = resolve_validation_experiment_dir(
                output_root=output_root,
                variant=variant,
                hook_point=hook_point,
                alpha=alpha,
                config_base=config_base,
            )
            config = dict(config_base)
            config['config_fingerprint'] = config_fingerprint
            config_id = f'{hook_point}_a{alpha}_{config_fingerprint[:10]}'

            metrics_payload = load_cached_validation_metrics(experiment_dir, config_base)
            if metrics_payload is None:
                generate_images_for_manifest(
                    pipe=pipe,
                    manifest=validation_manifest,
                    experiment_dir=experiment_dir,
                    hook_point=hook_point,
                    alpha=alpha,
                    num_denoising_steps=num_denoising_steps,
                    strategy=strategy,
                    concept_bank=concept_bank,
                    device=device,
                    save_config=config,
                )
                metrics_payload = evaluate_experiment_dir(
                    experiment_dir=experiment_dir,
                    metrics=VALIDATION_METRICS,
                    fid_reference_dir=validation_reference_dir,
                    device=device,
                    prompt_limit=prompt_limit,
                )
                write_metrics_json(experiment_dir, metrics_payload)
            else:
                print(f'Using cached validation metrics for {hook_point} alpha={alpha} from {experiment_dir}')

            row = flatten_metrics_for_summary(
                metrics_payload=metrics_payload,
                split='validation',
                variant=variant,
                model_alias=model_alias,
                hook_point=hook_point,
                alpha=alpha,
            )
            row['config_id'] = config_id
            row['experiment_dir'] = experiment_dir
            row['steering_bank_path'] = bank_path
            sweep_rows.append(row)

    best_row, annotated_rows = select_best_validation_row(sweep_rows)
    write_summary_csv(annotated_rows, os.path.join(output_root, 'summary.csv'))
    save_json(os.path.join(output_root, 'summary.json'), annotated_rows)
    save_json(os.path.join(output_root, 'best_config.json'), best_row)
    save_validation_plot(
        annotated_rows,
        os.path.join(output_root, 'validation_alpha_clip_fid.png'),
        title=os.path.basename(output_root) or 'Validation Sweep',
    )
    return annotated_rows, best_row


def run_test_evaluation(
    pipe,
    test_manifest: Sequence[Dict[str, Any]],
    output_root: str,
    model_alias: str,
    best_hook_point: str,
    best_alpha: float,
    num_denoising_steps: int = 20,
    strategy: str = 'all_layers',
    num_concepts: int = 50,
    bank_mode: str = DEFAULT_SANA_BANK_MODE,
    bank_dir: str = 'steering_vectors_sana',
    random_sv_seed: int = 12345,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    device = device or get_device()
    ensure_dir(output_root)
    concept_bank, bank_path = load_or_compute_concept_bank(
        pipe=pipe,
        hook_point=best_hook_point,
        bank_mode=bank_mode,
        num_concepts=num_concepts,
        num_denoising_steps=num_denoising_steps,
        bank_dir=bank_dir,
        device=device,
    )
    template_vectors = template_from_concept_bank(concept_bank)
    random_vectors = create_random_steering_vectors(template_vectors, seed=random_sv_seed)
    random_sv_path = os.path.join(output_root, 'random_steering_vector.pickle')
    save_pickle(random_sv_path, random_vectors)

    variants = [
        {
            'variant': 'best_steering',
            'baseline': False,
            'concept_bank': concept_bank,
            'steering_vectors': None,
            'hook_point': best_hook_point,
            'alpha': best_alpha,
        },
        {
            'variant': 'baseline',
            'baseline': True,
            'concept_bank': None,
            'steering_vectors': None,
            'hook_point': best_hook_point,
            'alpha': None,
        },
        {
            'variant': 'random_steering',
            'baseline': False,
            'concept_bank': None,
            'steering_vectors': random_vectors,
            'hook_point': best_hook_point,
            'alpha': best_alpha,
        },
    ]

    summary_rows = []
    prompt_limit = len(test_manifest)
    for variant in variants:
        experiment_dir = os.path.join(
            output_root,
            build_experiment_tag('test', variant['variant'], variant['hook_point'], variant['alpha']),
        )
        config = {
            'split': 'test',
            'variant': variant['variant'],
            'model_alias': model_alias,
            'model_id': resolve_model_id(model_alias=model_alias),
            'hook_point': variant['hook_point'],
            'alpha': variant['alpha'],
            'num_denoising_steps': num_denoising_steps,
            'strategy': strategy,
            'num_concepts': num_concepts,
            'bank_mode': bank_mode,
            'steering_bank_path': bank_path if variant['variant'] == 'best_steering' else None,
            'random_sv_path': random_sv_path if variant['variant'] == 'random_steering' else None,
            'evaluation_prompt_limit': prompt_limit,
            'metrics': TEST_METRICS,
        }
        generate_images_for_manifest(
            pipe=pipe,
            manifest=test_manifest,
            experiment_dir=experiment_dir,
            hook_point=variant['hook_point'],
            alpha=variant['alpha'],
            num_denoising_steps=num_denoising_steps,
            strategy=strategy,
            concept_bank=variant['concept_bank'],
            steering_vectors=variant['steering_vectors'],
            baseline=variant['baseline'],
            device=device,
            save_config=config,
        )
        metrics_payload = evaluate_experiment_dir(
            experiment_dir=experiment_dir,
            metrics=TEST_METRICS,
            device=device,
            prompt_limit=prompt_limit,
        )
        write_metrics_json(experiment_dir, metrics_payload)
        summary_rows.append(
            flatten_metrics_for_summary(
                metrics_payload=metrics_payload,
                split='test',
                variant=variant['variant'],
                model_alias=model_alias,
                hook_point=variant['hook_point'],
                alpha=variant['alpha'],
            )
        )

    write_summary_csv(summary_rows, os.path.join(output_root, 'summary.csv'))
    save_json(os.path.join(output_root, 'summary.json'), summary_rows)
    return summary_rows
