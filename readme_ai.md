# Steering Vectors for SANA Diversity Improvement

## Problem
SANA (Linear Diffusion Transformer) generates low-diversity images — different seeds with the same prompt produce nearly identical results. We adapt the CASteer approach (steering vectors from cross-attention activations) to SANA's architecture to inject diversity at generation time.

## Architecture Differences: SD vs SANA

| Feature | Stable Diffusion | SANA |
|---------|-----------------|------|
| Backbone | UNet (down/mid/up blocks) | Flat transformer (20 `SanaTransformerBlock`) |
| Self-attention | Standard attention | Linear attention (`SanaLinearAttnProcessor2_0`) |
| Cross-attention | Inside `BasicTransformerBlock` | Inside `SanaTransformerBlock` |
| FFN | Standard MLP | GLUMBConv (conv-based) |
| Text encoder | CLIP | Gemma2 |
| Hook point (original) | `attn_output` in `BasicTransformerBlock` | `attn_output` in `SanaTransformerBlock` |

Key change: instead of `down/up/mid` keys we use a flat `layers` list indexed 0..N-1.

## Setup (Colab / Local)

```bash
# Install dependencies
pip install diffusers transformers accelerate torch
pip install open_clip_torch vendi-score image-reward
pip install sentencepiece  # for Gemma tokenizer
```

## Files

| File | Description |
|------|-------------|
| `sana_controller.py` | Controller adapted for SANA (flat transformer blocks) |
| `compute_steering_vectors_sana.py` | Compute steering vectors: multi-concept bank, per-concept averaged (like SD), or per-concept bank (mixed modes) |
| `generate_sana_diverse.py` | Generate images with random steering for diversity |
| `evaluate_sana.py` | Evaluate quality (CLIPScore, PickScore, ImageReward) and diversity (MPS, Vendi) |

## Step-by-Step Commands

### Step 1: Compute Steering Vectors (~30 min on T4)

```bash
cd /path/to/CASteer

# Mode A: Multi-concept bank (default) — many ImageNet concepts, one prompt each
python compute_steering_vectors_sana.py \
    --averaging multi_concept \
    --num_concepts 50 \
    --num_denoising_steps 20 \
    --hook_point cross_attn \
    --save_dir steering_vectors_sana

# Output:
#   steering_vectors_sana/sana_cross_attn_50concepts.pickle     (averaged SV)
#   steering_vectors_sana/sana_cross_attn_50concepts_bank.pickle (per-concept bank)

# Mode B: Per-concept averaged (like SD) — one concept, many prompts averaged
python compute_steering_vectors_sana.py \
    --averaging per_concept \
    --concept_pos anime \
    --prompt_mode style \
    --num_concepts 50 \
    --num_denoising_steps 20 \
    --hook_point cross_attn \
    --save_dir steering_vectors_sana

# Output:
#   steering_vectors_sana/sana_cross_attn_anime_None.pickle

# Mode C: Per-concept bank — multiple concepts from different modes, each averaged across many prompts
python compute_steering_vectors_sana.py \
    --averaging per_concept_bank \
    --num_concepts 10 \
    --num_denoising_steps 20 \
    --hook_point cross_attn \
    --save_dir steering_vectors_sana

# Output:
#   steering_vectors_sana/sana_cross_attn_per_concept_bank.pickle
# Bank format: {"concept_name": {step: {"layers": [...]}}}
# Default concepts: anime, watercolor, oil painting, pixel art, sketch, photorealistic,
#                   Snoopy, sunglasses, hat, flowers, snow, fire

# Mode C with custom concepts file:
python compute_steering_vectors_sana.py \
    --averaging per_concept_bank \
    --concepts_file concepts.txt \
    --num_concepts 20 \
    --hook_point cross_attn \
    --save_dir steering_vectors_sana

# concepts.txt format (one per line: concept,mode):
#   anime,style
#   watercolor,style
#   Snoopy,concrete
#   smiling,human-related
```

### Step 2: Generate Baseline Images

```bash
python generate_sana_diverse.py \
    --baseline \
    --num_seeds 10 \
    --save_dir sana_diverse_outputs
```

### Step 3: Generate with Steering Vectors (test all strategies)

```bash
# Using multi-concept bank (random concept per seed):
# Strategy 1: All layers, cross-attn
python generate_sana_diverse.py \
    --strategy all_layers --hook_point cross_attn --alpha 10 \
    --sv_bank_path steering_vectors_sana/sana_cross_attn_50concepts_bank.pickle \
    --save_dir sana_diverse_outputs

# Strategy 2: Late layers only (10-19)
python generate_sana_diverse.py \
    --strategy late_layers --hook_point cross_attn --alpha 10 \
    --sv_bank_path steering_vectors_sana/sana_cross_attn_50concepts_bank.pickle \
    --save_dir sana_diverse_outputs

# Strategy 3: Early layers only (0-9)
python generate_sana_diverse.py \
    --strategy early_layers --hook_point cross_attn --alpha 10 \
    --sv_bank_path steering_vectors_sana/sana_cross_attn_50concepts_bank.pickle \
    --save_dir sana_diverse_outputs

# Strategy 4: Self-attention hook
python compute_steering_vectors_sana.py \
    --num_concepts 50 --hook_point self_attn --save_dir steering_vectors_sana
python generate_sana_diverse.py \
    --strategy all_layers --hook_point self_attn --alpha 10 \
    --sv_bank_path steering_vectors_sana/sana_self_attn_50concepts_bank.pickle \
    --save_dir sana_diverse_outputs

# Strategy 5: Residual stream hook
python compute_steering_vectors_sana.py \
    --num_concepts 50 --hook_point residual --save_dir steering_vectors_sana
python generate_sana_diverse.py \
    --strategy all_layers --hook_point residual --alpha 10 \
    --sv_bank_path steering_vectors_sana/sana_residual_50concepts_bank.pickle \
    --save_dir sana_diverse_outputs

# Strategy 6: Timestep-scaled (stronger early, weaker late)
python generate_sana_diverse.py \
    --strategy timestep_scaled --hook_point cross_attn --alpha 10 \
    --sv_bank_path steering_vectors_sana/sana_cross_attn_50concepts_bank.pickle \
    --save_dir sana_diverse_outputs

# Strategy 7: Random subset of layers
python generate_sana_diverse.py \
    --strategy random_layers --hook_point cross_attn --alpha 10 \
    --sv_bank_path steering_vectors_sana/sana_cross_attn_50concepts_bank.pickle \
    --save_dir sana_diverse_outputs
```

```bash
# Using per-concept averaged SV (single concept, like SD):
python generate_sana_diverse.py \
    --strategy all_layers --hook_point cross_attn --alpha 10 \
    --sv_path steering_vectors_sana/sana_cross_attn_anime_None.pickle \
    --save_dir sana_diverse_outputs

# Using per-concept bank (random concept per seed, each concept averaged across many prompts):
python generate_sana_diverse.py \
    --strategy all_layers --hook_point cross_attn --alpha 10 \
    --sv_bank_path steering_vectors_sana/sana_cross_attn_per_concept_bank.pickle \
    --save_dir sana_diverse_outputs
```

### Alpha sweep (for best strategy):

```bash
for alpha in 0.5 1 2 5 10; do
    python generate_sana_diverse.py \
        --strategy all_layers --hook_point cross_attn --alpha $alpha \
        --sv_bank_path steering_vectors_sana/sana_cross_attn_50concepts_bank.pickle \
        --save_dir sana_diverse_outputs
done
```

### Step 4: Evaluate

```bash
# Evaluate baseline
python evaluate_sana.py \
    --results_dir sana_diverse_outputs/baseline \
    --metrics clip_score mps vendi

# Evaluate each strategy
for dir in sana_diverse_outputs/*/; do
    echo "=== Evaluating: $dir ==="
    python evaluate_sana.py \
        --results_dir "$dir" \
        --metrics clip_score mps vendi
done

# Full metrics (including PickScore and ImageReward, slower)
python evaluate_sana.py \
    --results_dir sana_diverse_outputs/all_layers_cross_attn_a10 \
    --metrics clip_score pick_score image_reward mps vendi
```

## Strategies Summary

| # | Strategy | Description | Command flag |
|---|----------|-------------|--------------|
| 1 | All layers, cross-attn | Hook after cross-attn in all 20 blocks | `--strategy all_layers --hook_point cross_attn` |
| 2 | Late layers only | Only blocks 10-19 | `--strategy late_layers` |
| 3 | Early layers only | Only blocks 0-9 | `--strategy early_layers` |
| 4 | Self-attn hook | Hook after self-attention | `--hook_point self_attn` |
| 5 | Residual stream | Hook after entire block | `--hook_point residual` |
| 6 | Timestep-scaled | alpha decays linearly over denoising steps | `--strategy timestep_scaled` |
| 7 | Random layers | Random half of layers per generation | `--strategy random_layers` |

## Averaging Modes

| Mode | Description | Command |
|------|-------------|---------|
| `multi_concept` (default) | Many ImageNet concepts, one prompt each → diversity bank | `--averaging multi_concept` |
| `per_concept` | One concept, many prompts averaged (like SD) → single SV | `--averaging per_concept --concept_pos anime --prompt_mode style` |
| `per_concept_bank` | Multiple concepts from different modes (style, concrete, human-related), each averaged across many prompts → named concept bank | `--averaging per_concept_bank` |

**`per_concept`** reuses the same prompt templates as the SD pipeline (`style`, `concrete`, `human-related` from `construct_prompts.py`). The `--num_concepts` parameter controls how many ImageNet subjects are used as prompt variations (default 50). In generation, use `--sv_path` instead of `--sv_bank_path` to load the single averaged SV.

**`per_concept_bank`** combines the best of both modes: each concept gets a high-quality averaged steering vector (like `per_concept`), but the result is a bank of multiple concepts (like `multi_concept`). Concepts span different modes — style (anime, watercolor, ...), concrete (Snoopy, sunglasses, ...), and human-related. During generation, a random concept is picked per seed. The default concept list can be overridden with `--concepts_file`. In generation, use `--sv_bank_path` to load the bank.

## Test Prompts (10)

1. "a photo of a girl"
2. "a cat sitting on a windowsill"
3. "a mountain landscape at sunset"
4. "a red sports car on a highway"
5. "a bouquet of flowers in a vase"
6. "an astronaut floating in space"
7. "a cozy coffee shop interior"
8. "a dog playing in the park"
9. "a futuristic city skyline"
10. "a plate of sushi on a wooden table"

## Evaluation Metrics

| Metric | Type | Library | Good direction |
|--------|------|---------|---------------|
| CLIPScore | Quality | `open_clip` | Higher = better |
| PickScore | Quality | `transformers` (yuvalkirstain/PickScore_v1) | Higher = better |
| ImageReward | Quality | `image-reward` | Higher = better |
| MPS (Mean Pairwise Similarity) | Diversity | `open_clip` | Lower = more diverse |
| Vendi Score | Diversity | `vendi-score` | Higher = more diverse |

## Expected Results

- **Diversity**: MPS should decrease (more diverse), Vendi Score should increase
- **Quality**: CLIPScore, PickScore, ImageReward should not degrade >5%
- **Best strategy**: likely late_layers or timestep_scaled (preserve structure while adding variation)

## Model Info

- **SANA 600M 512px**: `Efficient-Large-Model/Sana_600M_512px_diffusers`
- Fits in T4 16GB (Colab free tier)
- 20 transformer blocks, inner_dim=2240
- Text encoder: Gemma2 (2304 caption channels)
- Default 20 denoising steps
