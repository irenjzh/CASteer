import argparse

from sana_experiment_utils import download_coco_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_dir', type=str, default='coco')
    return parser.parse_args()


def main():
    args = parse_args()
    info = download_coco_dataset(args.coco_dir)
    print(f"Annotations: {info['annotations_path']}")
    print(f"Images dir:   {info['val_dir']}")
    print(f"val2017 size: {info['num_val_images']}")


if __name__ == '__main__':
    main()
