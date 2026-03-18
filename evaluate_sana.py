import os
import glob
import json
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchvision import transforms

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, required=True,
                    help='Directory with generated images (e.g., sana_diverse_outputs/baseline)')
parser.add_argument('--output_json', type=str, default=None,
                    help='Path to save metrics JSON')
parser.add_argument('--metrics', type=str, nargs='+',
                    default=['clip_score', 'mps', 'vendi'],
                    choices=['clip_score', 'pick_score', 'image_reward', 'mps', 'vendi'])
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---- CLIP model (shared for CLIPScore and MPS) ----
clip_model = None
clip_preprocess = None

def load_clip():
    global clip_model, clip_preprocess
    if clip_model is not None:
        return
    import open_clip
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    clip_model = clip_model.to(device).eval()

def get_clip_image_features(images):
    load_clip()
    import open_clip
    feats = []
    for img in images:
        img_t = clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            f = clip_model.encode_image(img_t)
            f = F.normalize(f, dim=-1)
        feats.append(f)
    return torch.cat(feats, dim=0)

def get_clip_text_features(texts):
    load_clip()
    import open_clip
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        f = clip_model.encode_text(tokens)
        f = F.normalize(f, dim=-1)
    return f


# ---- Metrics ----

def compute_clip_score(images, prompt):
    img_feats = get_clip_image_features(images)
    txt_feats = get_clip_text_features([prompt])
    scores = (img_feats @ txt_feats.T).squeeze(-1)
    return scores.cpu().numpy().tolist()


def compute_mps(images):
    """Mean Pairwise Similarity — average cosine similarity between all pairs."""
    if len(images) < 2:
        return 1.0
    img_feats = get_clip_image_features(images)
    sim_matrix = img_feats @ img_feats.T
    n = len(images)
    # Extract upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
    pairwise_sims = sim_matrix[mask]
    return pairwise_sims.mean().item()


def compute_vendi_score(images):
    """Vendi Score — diversity metric based on eigenvalues of similarity matrix."""
    from vendi_score import vendi
    img_feats = get_clip_image_features(images)
    sim_matrix = (img_feats @ img_feats.T).cpu().numpy()
    # Vendi score from similarity matrix
    return vendi.score_K(sim_matrix)


def compute_pick_score(images, prompt):
    """PickScore using yuvalkirstain/PickScore_v1."""
    from transformers import AutoProcessor, AutoModel
    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)

    scores = []
    for img in images:
        inputs = processor(
            text=prompt, images=img, return_tensors="pt",
            padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            scores.append(logits.item())
    return scores


def compute_image_reward(images, prompt):
    """ImageReward score."""
    import ImageReward as RM
    model = RM.load("ImageReward-v1.0")
    scores = []
    for img in images:
        score = model.score(prompt, img)
        scores.append(score)
    return scores


# ---- Main ----

def load_prompt_images(prompt_dir):
    """Load all PNG images from a prompt directory."""
    image_paths = sorted(glob.glob(os.path.join(prompt_dir, "*.png")))
    images = [Image.open(p).convert("RGB") for p in image_paths]
    return images, image_paths


def main():
    prompt_dirs = sorted(glob.glob(os.path.join(args.results_dir, "prompt_*")))
    if not prompt_dirs:
        print(f"No prompt directories found in {args.results_dir}")
        return

    all_metrics = {}

    for prompt_dir in prompt_dirs:
        prompt_name = os.path.basename(prompt_dir)
        prompt_file = os.path.join(prompt_dir, "prompt.txt")
        if os.path.exists(prompt_file):
            with open(prompt_file) as f:
                prompt = f.read().strip()
        else:
            prompt = prompt_name

        images, paths = load_prompt_images(prompt_dir)
        if not images:
            continue

        print(f"\n--- {prompt_name}: \"{prompt}\" ({len(images)} images) ---")
        metrics = {"prompt": prompt, "num_images": len(images)}

        if 'clip_score' in args.metrics:
            scores = compute_clip_score(images, prompt)
            metrics['clip_score_mean'] = float(np.mean(scores))
            metrics['clip_score_std'] = float(np.std(scores))
            print(f"  CLIPScore: {metrics['clip_score_mean']:.4f} +/- {metrics['clip_score_std']:.4f}")

        if 'mps' in args.metrics:
            mps = compute_mps(images)
            metrics['mps'] = mps
            print(f"  MPS (lower=more diverse): {mps:.4f}")

        if 'vendi' in args.metrics:
            vs = compute_vendi_score(images)
            metrics['vendi_score'] = vs
            print(f"  Vendi Score (higher=more diverse): {vs:.4f}")

        if 'pick_score' in args.metrics:
            scores = compute_pick_score(images, prompt)
            metrics['pick_score_mean'] = float(np.mean(scores))
            metrics['pick_score_std'] = float(np.std(scores))
            print(f"  PickScore: {metrics['pick_score_mean']:.4f} +/- {metrics['pick_score_std']:.4f}")

        if 'image_reward' in args.metrics:
            scores = compute_image_reward(images, prompt)
            metrics['image_reward_mean'] = float(np.mean(scores))
            metrics['image_reward_std'] = float(np.std(scores))
            print(f"  ImageReward: {metrics['image_reward_mean']:.4f} +/- {metrics['image_reward_std']:.4f}")

        all_metrics[prompt_name] = metrics

    # Aggregate
    print("\n=== AGGREGATE ===")
    for metric_key in ['clip_score_mean', 'mps', 'vendi_score', 'pick_score_mean', 'image_reward_mean']:
        values = [m[metric_key] for m in all_metrics.values() if metric_key in m]
        if values:
            print(f"  {metric_key}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    # Save
    output_path = args.output_json or os.path.join(args.results_dir, "metrics.json")
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {output_path}")


if __name__ == '__main__':
    main()
