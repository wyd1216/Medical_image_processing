import argparse
import pandas as pd
from pathlib import Path
from natsort import os_sorted
from utils import str2bool

def get_args_parser():
    parser = argparse.ArgumentParser('Image preprocessing for machine learning', add_help=False)
    # ------------------------------------------ Path info statistics ------------------------------------------------
    # It is better to set the global parameters here since each script will use the parameters.
    parser.add_argument('--images', '-i', type=str, help='The directory path of the image of the dataset')
    parser.add_argument('--labels', '-l', type=str, default='',
                        help='Path to labels. For a classification task, this should be a CSV file. \
                        For a segmentation task, this should be a directory with mask images.')
    parser.add_argument('--dataset_file', type=str, default='', help='The path of dataset information file.')
    return parser


def main(args):
    # ------------------------------ Define the traversal rules of image search ------------------------------
    image_dir = Path(args.images)
    image_names = os_sorted([x.name for x in image_dir.iterdir()])
    image_pids = ['_'.join(x.split('_')[:2]) for x in image_names]
    image_paths = [str(image_dir / name) for name in image_names]
    path_info = pd.DataFrame({'pid': image_pids, 'image': image_paths})

    if args.labels:
        label_dir = Path(args.labels)
        label_names = [x.name for x in label_dir.iterdir()]
        label_pids = ['.'.join(x.split('.')[:-2]) for x in label_names]
        label_keys = {x: y for x, y in zip(label_pids, label_names)}
        label_names = [label_keys[id] for id in image_pids]
        label_paths = [str(label_dir / name) for name in label_names]
        path_info['label'] = label_paths

    if args.dataset_file:
        set_df = pd.read_csv(args.paths_dataset)
        cols = ['pid', 'dataset']
        sub_df = set_df[cols]
        path_info = pd.merge(left=path_info, right=sub_df, on='pid', how='left')
        path_info = path_info.fillna('nonused')

    outdir = Path('./output/dataset_statistics').absolute()
    path_info.to_csv(outdir / 'path_statistics.csv', index=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image format convert', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
