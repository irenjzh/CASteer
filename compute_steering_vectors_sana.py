import os
import numpy as np
import pickle
from collections import defaultdict

import torch
from diffusers import SanaPipeline

from construct_prompts import (get_imagenet_classes, get_prompts_concrete,
                               get_prompts_style, get_prompts_human_related)
from sana_controller import SanaVectorStore, register_vector_control_sana

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_concepts', type=int, default=50)
parser.add_argument('--num_denoising_steps', type=int, default=20)
parser.add_argument('--save_dir', type=str, default='steering_vectors_sana')
parser.add_argument('--hook_point', type=str, default='cross_attn',
                    choices=['cross_attn', 'self_attn', 'residual'])
parser.add_argument('--model_id', type=str,
                    default='Efficient-Large-Model/Sana_600M_512px_diffusers')
parser.add_argument('--averaging', type=str, default='multi_concept',
                    choices=['multi_concept', 'per_concept', 'per_concept_bank'],
                    help='multi_concept: many concepts, one prompt each (for diversity bank). '
                         'per_concept: one concept, many prompts averaged (like SD). '
                         'per_concept_bank: multiple concepts from different modes, each averaged across many prompts.')
parser.add_argument('--concept_pos', type=str, default='anime',
                    help='Positive concept for per_concept averaging')
parser.add_argument('--concept_neg', type=str, default=None,
                    help='Negative concept for per_concept averaging')
parser.add_argument('--prompt_mode', type=str, default='style',
                    choices=['concrete', 'human-related', 'style'],
                    help='Prompt generation mode for per_concept averaging')
parser.add_argument('--concepts_file', type=str, default=None,
                    help='Path to concepts file for per_concept_bank mode (format: concept,mode per line)')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load SANA pipeline
try:
    pipe = SanaPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        cache_dir='./cache'
    )
except OSError:
    print("Could not fetch from Hub, loading from local cache...")
    pipe = SanaPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        cache_dir='./cache',
        local_files_only=True
    )
pipe.to(device)

# Default concepts for per_concept_bank mode
DEFAULT_BANK_CONCEPTS = [
    ("anime", "style"),
    ("watercolor", "style"),
    ("oil painting", "style"),
    ("pixel art", "style"),
    ("sketch", "style"),
    ("photorealistic", "style"),
    ("Snoopy", "concrete"),
    ("sunglasses", "concrete"),
    ("hat", "concrete"),
    ("flowers", "concrete"),
    ("snow", "concrete"),
    ("fire", "concrete"),
]


def build_prompts(mode, concept_pos, num_prompts, concept_neg=None):
    if mode == 'concrete':
        return get_prompts_concrete(num=num_prompts, concept_pos=concept_pos, concept_neg=concept_neg)
    elif mode == 'style':
        return get_prompts_style(num=num_prompts, concept_pos=concept_pos, concept_neg=concept_neg)
    elif mode == 'human-related':
        return get_prompts_human_related(concept_pos=concept_pos, concept_neg=concept_neg)


def collect_activations(pipe, prompts_pos, prompts_neg, num_denoising_steps, hook_point, device):
    pos_vectors = []
    neg_vectors = []
    seed = 0

    for i, (prompt_pos, prompt_neg) in enumerate(zip(prompts_pos, prompts_neg)):
        print(f'  Prompt {i}/{len(prompts_pos)}: pos="{prompt_pos}", neg="{prompt_neg}"')

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

    return pos_vectors, neg_vectors


def compute_sv(pos_vectors, neg_vectors, num_denoising_steps):
    num_layers = len(pos_vectors[0][0]["layers"])
    steering_vectors = {}

    for denoising_step in range(num_denoising_steps):
        steering_vectors[denoising_step] = {"layers": []}
        for layer_num in range(num_layers):
            pos_avg = np.mean([
                pos_vectors[i][denoising_step]["layers"][layer_num]
                for i in range(len(pos_vectors))
            ], axis=0)
            neg_avg = np.mean([
                neg_vectors[i][denoising_step]["layers"][layer_num]
                for i in range(len(neg_vectors))
            ], axis=0)
            sv = pos_avg - neg_avg
            norm = np.linalg.norm(sv)
            if np.isfinite(norm) and norm > 1e-8:
                sv = sv / norm
            else:
                sv = np.zeros_like(sv)
            steering_vectors[denoising_step]["layers"].append(sv)

    return steering_vectors


os.makedirs(args.save_dir, exist_ok=True)

if args.averaging == 'per_concept_bank':
    # Bank of per-concept averaged steering vectors from different modes
    if args.concepts_file:
        concepts = []
        with open(args.concepts_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    concepts.append((parts[0].strip(), parts[1].strip()))
    else:
        concepts = DEFAULT_BANK_CONCEPTS

    concept_bank = {}
    for idx, (concept, mode) in enumerate(concepts):
        print(f'\n=== [{idx+1}/{len(concepts)}] Concept: "{concept}" (mode={mode}) ===')
        prompts_pos, prompts_neg = build_prompts(mode, concept, args.num_concepts)
        pos_vecs, neg_vecs = collect_activations(
            pipe, prompts_pos, prompts_neg, args.num_denoising_steps, args.hook_point, device)
        concept_bank[concept] = compute_sv(pos_vecs, neg_vecs, args.num_denoising_steps)

    bank_path = os.path.join(
        args.save_dir,
        f'sana_{args.hook_point}_per_concept_bank.pickle'
    )
    with open(bank_path, 'wb') as f:
        pickle.dump(concept_bank, f)
    print(f'\nSaved per-concept bank ({len(concept_bank)} concepts) to {bank_path}')

else:
    # Build prompts
    if args.averaging == 'multi_concept':
        imagenet_classes = get_imagenet_classes(args.num_concepts)
        prompts_pos = [f"a photo of {cls}" for cls in imagenet_classes[:args.num_concepts]]
        prompts_neg = ["a photo"] * len(prompts_pos)
    elif args.averaging == 'per_concept':
        prompts_pos, prompts_neg = build_prompts(
            args.prompt_mode, args.concept_pos, args.num_concepts, args.concept_neg)

    # Collect activations
    pos_vectors, neg_vectors = collect_activations(
        pipe, prompts_pos, prompts_neg, args.num_denoising_steps, args.hook_point, device)

    # Compute steering vectors (averaged across all prompts)
    steering_vectors = compute_sv(pos_vectors, neg_vectors, args.num_denoising_steps)

    # Save
    if args.averaging == 'per_concept':
        save_path = os.path.join(
            args.save_dir,
            f'sana_{args.hook_point}_{args.concept_pos}_{args.concept_neg}.pickle'
        )
        with open(save_path, 'wb') as f:
            pickle.dump(steering_vectors, f)
        print(f'Saved per-concept averaged steering vectors to {save_path}')

    elif args.averaging == 'multi_concept':
        save_path = os.path.join(
            args.save_dir,
            f'sana_{args.hook_point}_{args.num_concepts}concepts.pickle'
        )
        with open(save_path, 'wb') as f:
            pickle.dump(steering_vectors, f)
        print(f'Saved steering vectors to {save_path}')

        # Also compute per-concept steering vectors bank
        print("\nComputing per-concept steering vector bank...")
        concept_bank = {}
        for concept_idx in range(len(pos_vectors)):
            concept_name = imagenet_classes[concept_idx]
            concept_bank[concept_name] = compute_sv(
                [pos_vectors[concept_idx]], [neg_vectors[concept_idx]], args.num_denoising_steps)

        bank_path = os.path.join(
            args.save_dir,
            f'sana_{args.hook_point}_{args.num_concepts}concepts_bank.pickle'
        )
        with open(bank_path, 'wb') as f:
            pickle.dump(concept_bank, f)
        print(f'Saved concept bank ({len(concept_bank)} concepts) to {bank_path}')
