import argparse
import os

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
    return parser.parse_args()


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
        metrics_payload = evaluate_experiment_dir(
            experiment_dir=experiment_dir,
            metrics=args.metrics,
            fid_reference_dir=args.fid_reference_dir,
            device=device,
        )
        write_metrics_json(experiment_dir, metrics_payload)
        for key, value in metrics_payload['aggregate'].items():
            if isinstance(value, dict):
                print(f'  {key}: {value["mean"]:.4f} +/- {value["std"]:.4f}')
            else:
                print(f'  {key}: {value}')


if __name__ == '__main__':
    main()
