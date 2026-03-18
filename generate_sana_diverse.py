import os
import random
import pickle
import numpy as np

import torch
from diffusers import SanaPipeline
from PIL import Image

from sana_controller import SanaVectorStore, register_vector_control_sana

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str,
                    default='Efficient-Large-Model/Sana_600M_512px_diffusers')
parser.add_argument('--sv_bank_path', type=str, default=None,
                    help='Path to concept bank pickle (multi_concept mode)')
parser.add_argument('--sv_path', type=str, default=None,
                    help='Path to single averaged steering vector pickle (per_concept mode)')
parser.add_argument('--num_denoising_steps', type=int, default=20)
parser.add_argument('--num_seeds', type=int, default=10)
parser.add_argument('--alpha', type=float, default=10.0)
parser.add_argument('--hook_point', type=str, default='cross_attn',
                    choices=['cross_attn', 'self_attn', 'residual'])
parser.add_argument('--strategy', type=str, default='all_layers',
                    choices=['all_layers', 'late_layers', 'early_layers',
                             'timestep_scaled', 'random_layers'])
parser.add_argument('--save_dir', type=str, default='sana_diverse_outputs')
parser.add_argument('--baseline', action='store_true',
                    help='Generate baseline images without steering')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test prompts
TEST_PROMPTS = [
    "a photo of a girl",
    "a cat sitting on a windowsill",
    "a mountain landscape at sunset",
    "a red sports car on a highway",
    "a bouquet of flowers in a vase",
    "an astronaut floating in space",
    "a cozy coffee shop interior",
    "a dog playing in the park",
    "a futuristic city skyline",
    "a plate of sushi on a wooden table",
]

# Load pipeline
pipe = SanaPipeline.from_pretrained(
    args.model_id,
    torch_dtype=torch.float16,
    cache_dir='./cache'
)
pipe.to(device)

# Load steering vectors
concept_bank = None
single_sv = None
if not args.baseline:
    if args.sv_path:
        # Single averaged SV (per_concept mode)
        with open(args.sv_path, 'rb') as f:
            single_sv = pickle.load(f)
        print(f"Loaded single averaged steering vector from {args.sv_path}")
    elif args.sv_bank_path:
        # Concept bank (multi_concept mode)
        with open(args.sv_bank_path, 'rb') as f:
            concept_bank = pickle.load(f)
        concept_names = list(concept_bank.keys())
        print(f"Loaded {len(concept_names)} concepts from bank")
    else:
        # Default bank path
        args.sv_bank_path = 'steering_vectors_sana/sana_cross_attn_50concepts_bank.pickle'
        with open(args.sv_bank_path, 'rb') as f:
            concept_bank = pickle.load(f)
        concept_names = list(concept_bank.keys())
        print(f"Loaded {len(concept_names)} concepts from bank")

num_layers = len(pipe.transformer.transformer_blocks)


def get_active_layers(strategy, num_layers):
    if strategy == 'all_layers':
        return None  # all layers
    elif strategy == 'late_layers':
        return set(range(num_layers // 2, num_layers))
    elif strategy == 'early_layers':
        return set(range(0, num_layers // 2))
    elif strategy == 'random_layers':
        k = num_layers // 2
        return set(random.sample(range(num_layers), k))
    elif strategy == 'timestep_scaled':
        return None  # all layers, scaling handled by controller
    return None


# Output directory
tag = "baseline" if args.baseline else f"{args.strategy}_{args.hook_point}_a{args.alpha}"
save_dir = os.path.join(args.save_dir, tag)
os.makedirs(save_dir, exist_ok=True)

for prompt_idx, prompt in enumerate(TEST_PROMPTS):
    prompt_dir = os.path.join(save_dir, f"prompt_{prompt_idx:02d}")
    os.makedirs(prompt_dir, exist_ok=True)

    for seed in range(args.num_seeds):
        concept_name = None
        if args.baseline:
            controller = SanaVectorStore(device=device)
            controller.steer = False
            register_vector_control_sana(pipe.transformer, controller, hook_point=args.hook_point)
        else:
            if single_sv is not None:
                # Single averaged SV (per_concept mode)
                steering_vectors = single_sv
                concept_name = None
            else:
                # Pick random concept from bank
                concept_name = random.choice(concept_names)
                steering_vectors = concept_bank[concept_name]

            active_layers = get_active_layers(args.strategy, num_layers)
            timestep_scaling = (args.strategy == 'timestep_scaled')

            controller = SanaVectorStore(
                steering_vectors=steering_vectors,
                steer=True,
                alpha=args.alpha,
                active_layers=active_layers,
                timestep_scaling=timestep_scaling,
                total_steps=args.num_denoising_steps,
                device=device,
            )
            register_vector_control_sana(pipe.transformer, controller, hook_point=args.hook_point)

        image = pipe(
            prompt=prompt,
            num_inference_steps=args.num_denoising_steps,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).images[0]

        img_name = f"seed{seed:02d}.png"
        if not args.baseline and concept_name is not None:
            img_name = f"seed{seed:02d}_{concept_name.replace(' ', '_')}.png"
        image.save(os.path.join(prompt_dir, img_name))

        status = "baseline" if args.baseline else (f"concept={concept_name}" if concept_name else "single_sv")
        print(f"[{prompt_idx}/{len(TEST_PROMPTS)}] seed={seed} {status}")

    # Save prompt text
    with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

print(f"\nSaved all images to {save_dir}")
