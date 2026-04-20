import argparse
import json
import os
from typing import Optional

from sana_experiment_utils import (
    TEST_METRICS,
    VALIDATION_METRICS,
    evaluate_experiment_dir,
    find_experiment_dirs,
    get_device,
    write_metrics_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True, help='Root directory with experiment outputs')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=list(TEST_METRICS),
        choices=sorted(set(TEST_METRICS + VALIDATION_METRICS)),
    )
    parser.add_argument('--fid_reference_dir', type=str, default=None, help='Reference image directory for FID')
    parser.add_argument('--prompt_limit', type=int, default=None, help='Evaluate only prompt_<idx> with idx < limit')
    parser.add_argument(
        '--validation_prompt_limit',
        type=int,
        default=None,
        help='Validation-only prompt limit override',
    )
    parser.add_argument(
        '--test_prompt_limit',
        type=int,
        default=None,
        help='Test-only prompt limit override',
    )
    return parser.parse_args()


def resolve_prompt_limit(experiment_dir: str, args) -> Optional[int]:
    config_path = os.path.join(experiment_dir, 'config.json')
    split = None
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            split = json.load(f).get('split')

    if split == 'validation' and args.validation_prompt_limit is not None:
        return args.validation_prompt_limit
    if split == 'test' and args.test_prompt_limit is not None:
        return args.test_prompt_limit
    return args.prompt_limit


def main():
    args = parse_args()
    device = get_device()
    experiment_dirs = find_experiment_dirs(args.results_dir)

    if not experiment_dirs:
        print(f'No experiment directories with prompt_* folders found in {args.results_dir}')
        return

    print(f'Found {len(experiment_dirs)} experiment(s):')
    for experiment_dir in experiment_dirs:
        print(f'  - {os.path.relpath(experiment_dir, args.results_dir)}')

    for experiment_dir in experiment_dirs:
        rel_path = os.path.relpath(experiment_dir, args.results_dir)
        print(f'\n{"=" * 60}')
        print(f'Evaluating: {rel_path}')
        print(f'{"=" * 60}')
        prompt_limit = resolve_prompt_limit(experiment_dir, args)
        metrics_payload = evaluate_experiment_dir(
            experiment_dir=experiment_dir,
            metrics=args.metrics,
            fid_reference_dir=args.fid_reference_dir,
            device=device,
            prompt_limit=prompt_limit,
        )
        write_metrics_json(experiment_dir, metrics_payload)
        for key, value in metrics_payload['aggregate'].items():
            if isinstance(value, dict):
                print(f'  {key}: {value["mean"]:.4f} +/- {value["std"]:.4f}')
            else:
                print(f'  {key}: {value}')


if __name__ == '__main__':
    main()
