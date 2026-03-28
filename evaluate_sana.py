import os
import glob
import json
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, required=False,
                    default="/content/drive/MyDrive/diplom/data/",
                    help='Top-level data directory (e.g., /content/drive/MyDrive/diplom/data/)')
parser.add_argument('--metrics', type=str, nargs='+',
                    default=['clip_score', 'mps', 'vendi'],
                    choices=['clip_score', 'pick_score', 'image_reward', 'mps', 'vendi'])
parser.add_argument('--download', action='store_true',
                    help='Download JSON files (for Google Colab)')
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
    mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
    pairwise_sims = sim_matrix[mask]
    return pairwise_sims.mean().item()


def compute_vendi_score(images):
    """Vendi Score — diversity metric based on eigenvalues of similarity matrix."""
    from vendi_score import vendi
    img_feats = get_clip_image_features(images)
    sim_matrix = (img_feats @ img_feats.T).cpu().numpy()
    return float(vendi.score_K(sim_matrix))


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


# ---- Helpers ----

def load_prompt_images(prompt_dir):
    """Load all PNG images from a prompt directory."""
    image_paths = sorted(glob.glob(os.path.join(prompt_dir, "*.png")))
    images = [Image.open(p).convert("RGB") for p in image_paths]
    return images, image_paths


def find_experiment_dirs(root_dir):
    """Find all directories that directly contain prompt_* subdirectories."""
    experiment_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        prompt_subdirs = [d for d in dirnames if d.startswith("prompt_")]
        if prompt_subdirs:
            experiment_dirs.append(dirpath)
            # Don't recurse into prompt_* dirs
            dirnames[:] = [d for d in dirnames if not d.startswith("prompt_")]
    return sorted(experiment_dirs)


def evaluate_experiment(experiment_dir):
    """Evaluate all prompts in one experiment directory. Returns dict with per-prompt and aggregate metrics."""
    prompt_dirs = sorted(glob.glob(os.path.join(experiment_dir, "prompt_*")))
    if not prompt_dirs:
        return None

    result = {"per_prompt": {}, "aggregate": {}}

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

        print(f"\n  --- {prompt_name}: \"{prompt}\" ({len(images)} images) ---")
        metrics = {"prompt": prompt, "num_images": len(images)}

        if 'clip_score' in args.metrics:
            scores = compute_clip_score(images, prompt)
            metrics['clip_score_mean'] = float(np.mean(scores))
            metrics['clip_score_std'] = float(np.std(scores))
            print(f"    CLIPScore: {metrics['clip_score_mean']:.4f} +/- {metrics['clip_score_std']:.4f}")

        if 'mps' in args.metrics:
            mps = compute_mps(images)
            metrics['mps'] = mps
            print(f"    MPS (lower=more diverse): {mps:.4f}")

        if 'vendi' in args.metrics:
            vs = compute_vendi_score(images)
            metrics['vendi_score'] = vs
            print(f"    Vendi Score (higher=more diverse): {vs:.4f}")

        if 'pick_score' in args.metrics:
            try:
                scores = compute_pick_score(images, prompt)
                metrics['pick_score_mean'] = float(np.mean(scores))
                metrics['pick_score_std'] = float(np.std(scores))
                print(f"    PickScore: {metrics['pick_score_mean']:.4f} +/- {metrics['pick_score_std']:.4f}")
            except Exception as e:
                print(f"    PickScore: Failed ({type(e).__name__})")

        if 'image_reward' in args.metrics:
            try:
                scores = compute_image_reward(images, prompt)
                metrics['image_reward_mean'] = float(np.mean(scores))
                metrics['image_reward_std'] = float(np.std(scores))
                print(f"    ImageReward: {metrics['image_reward_mean']:.4f} +/- {metrics['image_reward_std']:.4f}")
            except Exception as e:
                print(f"    ImageReward: Failed ({type(e).__name__})")

        result["per_prompt"][prompt_name] = metrics

    # Aggregate across all prompts
    aggregate = {}
    metric_keys = ['clip_score_mean', 'mps', 'vendi_score', 'pick_score_mean', 'image_reward_mean']
    for key in metric_keys:
        values = [m[key] for m in result["per_prompt"].values() if key in m]
        if values:
            aggregate[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

    result["aggregate"] = aggregate
    return result


def main():
    experiment_dirs = find_experiment_dirs(args.results_dir)

    if not experiment_dirs:
        print(f"No experiment directories with prompt_* folders found in {args.results_dir}")
        return

    print(f"Found {len(experiment_dirs)} experiment(s):")
    for d in experiment_dirs:
        rel = os.path.relpath(d, args.results_dir)
        print(f"  - {rel}")

    saved_jsons = []

    for experiment_dir in experiment_dirs:
        rel_path = os.path.relpath(experiment_dir, args.results_dir)
        # Use the relative path as experiment name (replace / with _)
        experiment_name = rel_path.replace(os.sep, "_")

        print(f"\n{'='*60}")
        print(f"Evaluating: {rel_path}")
        print(f"{'='*60}")

        result = evaluate_experiment(experiment_dir)
        if result is None:
            continue

        # Print aggregate
        print(f"\n  === AGGREGATE for {rel_path} ===")
        for key, vals in result["aggregate"].items():
            print(f"    {key}: {vals['mean']:.4f} +/- {vals['std']:.4f}")

        # Save JSON
        json_path = os.path.join(args.results_dir, f"{experiment_name}.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved: {json_path}")
        saved_jsons.append(json_path)

    # Download in Colab
    if args.download and saved_jsons:
        try:
            from google.colab import files
            for path in saved_jsons:
                files.download(path)
                print(f"  Downloaded: {os.path.basename(path)}")
        except ImportError:
            print("\n  --download flag ignored (not running in Google Colab)")


if __name__ == '__main__':
    main()
