# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CASteer implements **steering vectors for controllable diffusion model generation**. The core idea: extract directional vectors from cross-attention activations that represent specific concepts (styles, objects, attributes), then use these vectors to steer generation toward or away from those concepts.

Two pipelines exist:
- **Stable Diffusion pipeline** (original): UNet-based models (SD 1.4, SD 2.1, SDXL, turbo variants)
- **SANA pipeline** (extension): flat transformer architecture (SANA 600M), focused on diversity experiments

## Environment Setup

```bash
conda env create -f environment.yml
conda activate paper
```

Key dependencies: PyTorch (CUDA), diffusers, transformers, safetensors, open_clip, vendi-score, image-reward.

## Common Commands

### Stable Diffusion Pipeline

Compute steering vectors:
```bash
python compute_steering_vectors.py --model sd14 --mode style --concept_pos anime --concept_neg None --num_denoising_steps 50 --save_dir steering_vectors
```

Generate steered images:
```bash
python generate_casteer.py --model sd14 --prompt "a girl with a kitty" --steering_vectors steering_vectors/sd14_anime_None.pickle --alpha 10 --seed 0 --num_denoising_steps 50 --save_dir images
```

### SANA Pipeline

Compute steering vector bank:
```bash
python compute_steering_vectors_sana.py --num_concepts 50 --num_denoising_steps 20 --hook_point cross_attn --save_dir steering_vectors_sana
```

Generate with diversity steering:
```bash
python generate_sana_diverse.py --strategy all_layers --hook_point cross_attn --alpha 10 --sv_bank_path steering_vectors_sana/sana_cross_attn_50concepts_bank.pickle --num_seeds 10 --save_dir sana_diverse_outputs
```

Evaluate (quality + diversity metrics):
```bash
python evaluate_sana.py --results_dir sana_diverse_outputs/baseline --metrics clip_score mps vendi
```

## Architecture

### Steering Mechanism

The controller classes hook into transformer blocks during the diffusion denoising loop, intercepting cross-attention outputs to either **capture** activations (for computing steering vectors) or **inject** steering vectors (during generation).

**`controller.py`** — `VectorStore` for SD's UNet (tracks `down`/`mid`/`up` block regions via `register_vector_control` which recursively hooks into `BasicTransformerBlock`).

**`sana_controller.py`** — `SanaVectorStore` for SANA's flat transformer (flat layer indexing, timestep-aware alpha scaling, configurable hook points: `cross_attn`, `self_attn`, `residual`). Uses `register_vector_control_sana` hooking into `SanaTransformerBlock`.

### Steering Vector Data Format

SD vectors (pickle): `{denoising_step: {'down': [tensors...], 'mid': [tensors...], 'up': [tensors...]}}`

SANA vectors (pickle): `{denoising_step: {'layers': [tensors...]}}` — bank variant stores per-concept vectors.

### Pipeline Flow

`construct_prompts.py` generates positive/negative prompt pairs → `compute_steering_vectors*.py` runs the model capturing activations, computes mean difference vectors → `generate_*.py` loads vectors and applies them during inference via the controller → `evaluate_sana.py` measures quality (CLIPScore, PickScore, ImageReward) and diversity (MPS, Vendi score).

### Key Parameters

- `alpha`: forward steering strength
- `beta`: backward steering strength (with `--steer_back`)
- `--steer_only_up`: restrict steering to upsampling blocks only
- SANA strategies: `all_layers`, `late_layers`, `early_layers`, `timestep_scaled`, `random_layers`

## Models

Pre-trained models are cached in `./cache/`. Supported: `sd14`, `sd21`, `sd21-turbo`, `sdxl`, `sdxl-turbo`, SANA 600M.
