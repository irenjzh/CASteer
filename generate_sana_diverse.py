import argparse
import os

from sana_experiment_utils import (
    DEFAULT_SANA_BANK_MODE,
    generate_images_for_manifest,
    get_device,
    load_or_compute_concept_bank,
    load_pickle,
    load_sana_pipeline,
)

TEST_PROMPTS = [
    'a photo of a girl',
    'a cat sitting on a windowsill',
    'a mountain landscape at sunset',
    'a red sports car on a highway',
    'a bouquet of flowers in a vase',
    'an astronaut floating in space',
    'a cozy coffee shop interior',
    'a dog playing in the park',
    'a futuristic city skyline',
    'a plate of sushi on a wooden table',
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_alias', type=str, default='small', choices=['small', 'large'])
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--local_model_root', type=str, default=None)
    parser.add_argument('--sv_bank_path', type=str, default=None)
    parser.add_argument('--sv_path', type=str, default=None)
    parser.add_argument('--num_denoising_steps', type=int, default=20)
    parser.add_argument('--num_seeds', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--hook_point', type=str, default='cross_attn', choices=['cross_attn', 'self_attn', 'residual'])
    parser.add_argument('--strategy', type=str, default='all_layers', choices=['all_layers', 'late_layers', 'early_layers', 'timestep_scaled', 'random_layers'])
    parser.add_argument('--bank_mode', type=str, default=DEFAULT_SANA_BANK_MODE, choices=['per_concept_bank', 'multi_concept'])
    parser.add_argument('--num_concepts', type=int, default=50)
    parser.add_argument('--save_dir', type=str, default='sana_diverse_outputs')
    parser.add_argument('--baseline', action='store_true')
    return parser.parse_args()


def build_manifest(num_seeds):
    seeds = list(range(num_seeds))
    return [
        {
            'split': 'demo',
            'prompt_index': idx,
            'image_id': idx,
            'caption': prompt,
            'real_image_path': '',
            'seeds': seeds,
        }
        for idx, prompt in enumerate(TEST_PROMPTS)
    ]


def main():
    args = parse_args()
    device = get_device()
    pipe = load_sana_pipeline(
        model_alias=args.model_alias,
        model_id=args.model_id,
        device=device,
        local_model_root=args.local_model_root,
    )

    concept_bank = None
    steering_vectors = None
    steering_bank_path = args.sv_bank_path

    if not args.baseline:
        if args.sv_path:
            steering_vectors = load_pickle(args.sv_path)
            print(f'Loaded single averaged steering vector from {args.sv_path}')
        elif args.sv_bank_path:
            concept_bank = load_pickle(args.sv_bank_path)
            print(f'Loaded {len(concept_bank)} concepts from bank {args.sv_bank_path}')
        else:
            concept_bank, steering_bank_path = load_or_compute_concept_bank(
                pipe=pipe,
                hook_point=args.hook_point,
                bank_mode=args.bank_mode,
                num_concepts=args.num_concepts,
                device=device,
            )
            print(f'Loaded {len(concept_bank)} concepts from bank {steering_bank_path}')

    variant = 'baseline' if args.baseline else ('single_sv' if steering_vectors is not None else args.bank_mode)
    tag = f'{variant}_{args.strategy}_{args.hook_point}_a{args.alpha}' if not args.baseline else 'baseline'
    save_dir = os.path.join(args.save_dir, tag)
    manifest = build_manifest(args.num_seeds)
    config = {
        'split': 'demo',
        'variant': variant,
        'model_alias': args.model_alias,
        'hook_point': args.hook_point,
        'alpha': None if args.baseline else args.alpha,
        'strategy': args.strategy,
        'bank_mode': args.bank_mode,
        'steering_bank_path': steering_bank_path,
        'steering_vector_path': args.sv_path,
        'num_denoising_steps': args.num_denoising_steps,
    }
    generate_images_for_manifest(
        pipe=pipe,
        manifest=manifest,
        experiment_dir=save_dir,
        hook_point=args.hook_point,
        alpha=None if args.baseline else args.alpha,
        num_denoising_steps=args.num_denoising_steps,
        strategy=args.strategy,
        steering_vectors=steering_vectors,
        concept_bank=concept_bank,
        baseline=args.baseline,
        device=device,
        save_config=config,
    )
    print(f'\nSaved all images to {save_dir}')


if __name__ == '__main__':
    main()
