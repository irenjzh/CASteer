import argparse
import os

from sana_experiment_utils import (
    DEFAULT_BANK_CONCEPTS,
    DEFAULT_SANA_BANK_MODE,
    build_prompts,
    collect_activations,
    compute_steering_vectors,
    get_device,
    get_imagenet_classes,
    load_pickle,
    load_or_compute_per_concept_bank,
    load_sana_pipeline,
    save_pickle,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_concepts', type=int, default=50)
    parser.add_argument('--num_denoising_steps', type=int, default=20)
    parser.add_argument('--save_dir', type=str, default='steering_vectors_sana')
    parser.add_argument('--hook_point', type=str, default='cross_attn', choices=['cross_attn', 'self_attn', 'residual'])
    parser.add_argument('--model_alias', type=str, default='small', choices=['small', 'large'])
    parser.add_argument('--model_id', type=str, default=None)
    parser.add_argument('--local_model_root', type=str, default=None)
    parser.add_argument(
        '--averaging',
        type=str,
        default=DEFAULT_SANA_BANK_MODE,
        choices=['multi_concept', 'per_concept', 'per_concept_bank'],
        help='multi_concept: many concepts, one prompt each. per_concept: one concept, many prompts averaged. '
             'per_concept_bank: multiple concepts from different modes, each averaged across many prompts.',
    )
    parser.add_argument('--concept_pos', type=str, default='anime')
    parser.add_argument('--concept_neg', type=str, default=None)
    parser.add_argument('--prompt_mode', type=str, default='style', choices=['concrete', 'human-related', 'style'])
    parser.add_argument('--concepts_file', type=str, default=None)
    return parser.parse_args()


def load_concepts(args):
    if args.concepts_file:
        concepts = []
        with open(args.concepts_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    concepts.append((parts[0].strip(), parts[1].strip()))
        return concepts
    return list(DEFAULT_BANK_CONCEPTS)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = get_device()
    pipe = load_sana_pipeline(
        model_alias=args.model_alias,
        model_id=args.model_id,
        device=device,
        local_model_root=args.local_model_root,
    )

    if args.averaging == 'per_concept_bank':
        concepts = load_concepts(args)
        concept_bank, bank_path = load_or_compute_per_concept_bank(
            pipe=pipe,
            hook_point=args.hook_point,
            num_denoising_steps=args.num_denoising_steps,
            bank_dir=args.save_dir,
            concepts=concepts,
            device=device,
        )
        print(f'Saved per-concept bank ({len(concept_bank)} concepts) to {bank_path}')
        return

    if args.averaging == 'multi_concept':
        imagenet_classes = get_imagenet_classes(args.num_concepts)
        prompts_pos = [f'a photo of {cls}' for cls in imagenet_classes[:args.num_concepts]]
        prompts_neg = ['a photo'] * len(prompts_pos)
    else:
        prompts_pos, prompts_neg = build_prompts(
            args.prompt_mode,
            args.concept_pos,
            args.num_concepts,
            args.concept_neg,
        )

    if args.averaging == 'per_concept':
        save_path = os.path.join(args.save_dir, f'sana_{args.hook_point}_{args.concept_pos}_{args.concept_neg}.pickle')
    else:
        save_path = os.path.join(args.save_dir, f'sana_{args.hook_point}_{args.num_concepts}concepts.pickle')

    checkpoint_path = f'{save_path}.checkpoint.pickle'
    multi_concept_partial_bank = {}
    multi_concept_partial_bank_path = None
    if args.averaging == 'multi_concept':
        multi_concept_partial_bank_path = os.path.join(
            args.save_dir,
            f'sana_{args.hook_point}_{args.num_concepts}concepts_bank.partial.pickle',
        )
        if os.path.exists(multi_concept_partial_bank_path):
            multi_concept_partial_bank = load_pickle(multi_concept_partial_bank_path)

    def on_prompt_complete(
        _prompt_index,
        _prompt_pos,
        _prompt_neg,
        pos_vectors,
        neg_vectors,
    ):
        partial_vectors = compute_steering_vectors(pos_vectors, neg_vectors, args.num_denoising_steps)
        save_pickle(save_path, partial_vectors)

        if args.averaging == 'multi_concept':
            concept_name = imagenet_classes[len(pos_vectors) - 1]
            multi_concept_partial_bank[concept_name] = compute_steering_vectors(
                [pos_vectors[-1]],
                [neg_vectors[-1]],
                args.num_denoising_steps,
            )
            save_pickle(multi_concept_partial_bank_path, multi_concept_partial_bank)

    pos_vectors, neg_vectors = collect_activations(
        pipe=pipe,
        prompts_pos=prompts_pos,
        prompts_neg=prompts_neg,
        num_denoising_steps=args.num_denoising_steps,
        hook_point=args.hook_point,
        device=device,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata={
            'averaging': args.averaging,
            'concept_pos': args.concept_pos,
            'concept_neg': args.concept_neg,
            'prompt_mode': args.prompt_mode,
            'num_concepts': args.num_concepts,
            'save_path': save_path,
        },
        progress_callback=on_prompt_complete,
    )
    steering_vectors = compute_steering_vectors(pos_vectors, neg_vectors, args.num_denoising_steps)

    if args.averaging == 'per_concept':
        save_pickle(save_path, steering_vectors)
        print(f'Saved per-concept averaged steering vectors to {save_path}')
        return

    save_pickle(save_path, steering_vectors)
    print(f'Saved steering vectors to {save_path}')

    print('\nComputing per-concept steering vector bank...')
    concept_bank = {}
    for concept_idx, concept_name in enumerate(imagenet_classes[:args.num_concepts]):
        concept_bank[concept_name] = compute_steering_vectors(
            [pos_vectors[concept_idx]],
            [neg_vectors[concept_idx]],
            args.num_denoising_steps,
        )
    bank_path = os.path.join(args.save_dir, f'sana_{args.hook_point}_{args.num_concepts}concepts_bank.pickle')
    save_pickle(bank_path, concept_bank)
    print(f'Saved concept bank ({len(concept_bank)} concepts) to {bank_path}')


if __name__ == '__main__':
    main()
