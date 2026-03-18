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
                    default='Efficient-Large-Model/Sana_600M_512px_BF16_diffusers')
parser.add_argument('--averaging', type=str, default='multi_concept',
                    choices=['multi_concept', 'per_concept'],
                    help='multi_concept: many concepts, one prompt each (for diversity bank). '
                         'per_concept: one concept, many prompts averaged (like SD)')
parser.add_argument('--concept_pos', type=str, default='anime',
                    help='Positive concept for per_concept averaging')
parser.add_argument('--concept_neg', type=str, default=None,
                    help='Negative concept for per_concept averaging')
parser.add_argument('--prompt_mode', type=str, default='style',
                    choices=['concrete', 'human-related', 'style'],
                    help='Prompt generation mode for per_concept averaging')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load SANA pipeline
pipe = SanaPipeline.from_pretrained(
    args.model_id,
    torch_dtype=torch.float16,
    cache_dir='./cache'
)
pipe.to(device)

# Build positive/negative prompt pairs depending on averaging mode
if args.averaging == 'multi_concept':
    # Many concepts, one prompt each (for diversity bank)
    imagenet_classes = get_imagenet_classes(args.num_concepts)
    prompts_pos = []
    prompts_neg = []
    for cls in imagenet_classes[:args.num_concepts]:
        prompts_pos.append(f"a photo of {cls}")
        prompts_neg.append("a photo")
elif args.averaging == 'per_concept':
    # One concept, many prompts averaged (like SD)
    if args.prompt_mode == 'concrete':
        prompts_pos, prompts_neg = get_prompts_concrete(
            num=args.num_concepts, concept_pos=args.concept_pos, concept_neg=args.concept_neg)
    elif args.prompt_mode == 'human-related':
        prompts_pos, prompts_neg = get_prompts_human_related(
            concept_pos=args.concept_pos, concept_neg=args.concept_neg)
    elif args.prompt_mode == 'style':
        prompts_pos, prompts_neg = get_prompts_style(
            num=args.num_concepts, concept_pos=args.concept_pos, concept_neg=args.concept_neg)

# Collect activations
pos_vectors = []
neg_vectors = []
seed = 0

for i, (prompt_pos, prompt_neg) in enumerate(zip(prompts_pos, prompts_neg)):
    print(f'Concept {i}/{len(prompts_pos)}: pos="{prompt_pos}", neg="{prompt_neg}"')

    # Positive prompt
    controller = SanaVectorStore(device=device)
    controller.steer = False
    register_vector_control_sana(pipe.transformer, controller, hook_point=args.hook_point)

    _ = pipe(
        prompt=prompt_pos,
        num_inference_steps=args.num_denoising_steps,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    pos_vectors.append(controller.vector_store)

    # Negative prompt
    controller = SanaVectorStore(device=device)
    controller.steer = False
    register_vector_control_sana(pipe.transformer, controller, hook_point=args.hook_point)

    _ = pipe(
        prompt=prompt_neg,
        num_inference_steps=args.num_denoising_steps,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    neg_vectors.append(controller.vector_store)


# Compute steering vectors
steering_vectors = {}
num_layers = len(pos_vectors[0][0]["layers"])

for denoising_step in range(args.num_denoising_steps):
    steering_vectors[denoising_step] = {"layers": []}

    for layer_num in range(num_layers):
        pos_vectors_layer = [
            pos_vectors[i][denoising_step]["layers"][layer_num]
            for i in range(len(pos_vectors))
        ]
        pos_avg = np.mean(pos_vectors_layer, axis=0)

        neg_vectors_layer = [
            neg_vectors[i][denoising_step]["layers"][layer_num]
            for i in range(len(neg_vectors))
        ]
        neg_avg = np.mean(neg_vectors_layer, axis=0)

        sv = pos_avg - neg_avg
        sv = sv / np.linalg.norm(sv)
        steering_vectors[denoising_step]["layers"].append(sv)

# Save
os.makedirs(args.save_dir, exist_ok=True)

if args.averaging == 'per_concept':
    # Single concept, averaged across prompts (like SD)
    save_path = os.path.join(
        args.save_dir,
        f'sana_{args.hook_point}_{args.concept_pos}_{args.concept_neg}.pickle'
    )
    with open(save_path, 'wb') as f:
        pickle.dump(steering_vectors, f)
    print(f'Saved per-concept averaged steering vectors to {save_path}')

elif args.averaging == 'multi_concept':
    # Averaged across all concepts
    save_path = os.path.join(
        args.save_dir,
        f'sana_{args.hook_point}_{args.num_concepts}concepts.pickle'
    )
    with open(save_path, 'wb') as f:
        pickle.dump(steering_vectors, f)
    print(f'Saved steering vectors to {save_path}')

    # Also compute per-concept steering vectors (for diversity bank)
    print("\nComputing per-concept steering vector bank...")
    concept_bank = {}

    for concept_idx in range(len(pos_vectors)):
        concept_name = imagenet_classes[concept_idx]
        concept_sv = {}

        for denoising_step in range(args.num_denoising_steps):
            concept_sv[denoising_step] = {"layers": []}

            for layer_num in range(num_layers):
                pos_v = pos_vectors[concept_idx][denoising_step]["layers"][layer_num]
                neg_v = neg_vectors[concept_idx][denoising_step]["layers"][layer_num]

                sv = pos_v - neg_v
                sv = sv / np.linalg.norm(sv)
                concept_sv[denoising_step]["layers"].append(sv)

        concept_bank[concept_name] = concept_sv

    bank_path = os.path.join(
        args.save_dir,
        f'sana_{args.hook_point}_{args.num_concepts}concepts_bank.pickle'
    )
    with open(bank_path, 'wb') as f:
        pickle.dump(concept_bank, f)
    print(f'Saved concept bank ({len(concept_bank)} concepts) to {bank_path}')
