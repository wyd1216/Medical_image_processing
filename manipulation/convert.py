import os
import argparse
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union
from tqdm import tqdm

from config import get_args_parser
from image_processing import MedicalImageProcessor
from report_generation import Graphs


def create_path_info(data_path: str, saved_path: str, data_depth=1):
    path = Path(data_path)
    if path.suffix == '.csv':
        df = pd.read_csv(str(path))
    else:
        path_list = list_entries_at_depth(data_path, data_depth)
        df = pd.DataFrame({'in_path': path_list})
        df['out_path'] = df['in_path'].apply(lambda x: str(Path(saved_path) / path_of_last_n(x, data_depth)))
    return df


def path_of_last_n(path: str, n: int) -> Union[Path, None]:
    """
    Create a new pathlib.Path object containing the last n components of the input path.

    Args:
        path (str): The input path string.
        n (int): The number of last components to retrieve from the input path.

    Returns:
        pathlib.Path or None: A new pathlib.Path object containing the last n components
                              of the input path, or None if the input 'n' is not valid.
    """
    # Create a Path object from the input path string
    path = Path(path)
    # Check if the input 'n' is valid, return None if not valid
    if n <= 0 or n > len(path.parts):
        return None

    # Get the last n parts of the path
    path_names = path.parts[-n:]
    # Create a new Path object using the first path name from the last n parts
    path = Path(path_names[0])
    # Iterate through the remaining path names and join them using the '/' operator
    for name in path_names[1:]:
        path = path / name
    return path


def list_files_in_hierarchy(directory, max_depth, current_depth=1):
    all_files = []
    if current_depth > max_depth:
        return all_files

    for entry in os.listdir(directory):
        entry_path = os.path.join(directory, entry)

        if os.path.isfile(entry_path):
            all_files.append(entry_path)
        elif os.path.isdir(entry_path):
            all_files.append(entry_path)
            all_files.extend(list_files_in_hierarchy(entry_path, max_depth, current_depth + 1))
    return all_files


def list_entries_at_depth(directory, target_depth, current_depth=1):
    entries_at_depth = []

    if current_depth == target_depth:
        for entry in os.listdir(directory):
            entry_path = os.path.join(directory, entry)
            entries_at_depth.append(entry_path)
        return entries_at_depth

    if current_depth < target_depth:
        for entry in os.listdir(directory):
            entry_path = os.path.join(directory, entry)
            if os.path.isdir(entry_path):
                entries_at_depth.extend(list_entries_at_depth(entry_path, target_depth, current_depth + 1))

    return entries_at_depth


# def convert_nifti_batch_to_dicom(info_table, nifti_col, dicom_col, overwrite=True):
#     for ind, row in tqdm(info_table.iterrows()):
#         nii_file_path = pathlib.Path(row[nifti_col])
#         dicom_dir_path = pathlib.Path(row[dicom_col])
#         if not overwrite:
#             if dicom_dir_path.exists():
#                 continue
#         else:
#             if dicom_dir_path.exists():
#                 shutil.rmtree(dicom_dir_path, ignore_errors=True)
#         imgprs = MedicalImageProcessor(nii_file_path)
#         imgprs.saveimg(savepath=dicom_dir_path, format='dcm', meta_data=False)

def image_copy(data_info, image_col, image_new_col, overwirte=False):
    for ind, row in tqdm(data_info.iterrows()):
        shutil.copy(row[image_col], row[image_new_col])

def convert_dicom_nii(data_info, image_col, image_new_col, overwrite=False):
    # Rename the suffix
    for ind, row in tqdm(data_info.iterrows()):
        in_path = Path(row[image_col])
        # If is dicom
        if in_path.is_file() and in_path.suffix != '.dcm':
            continue
        out_path = Path(row[image_new_col])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite:
            if out_path.exists():
                continue
        else:
            if out_path.exists():
                shutil.rmtree(out_path, ignore_errors=True)
        imgprs = MedicalImageProcessor(in_path)
        imgprs.saveimg(savepath=out_path, format='nii', meta_data=False)
    return


def convert_nii_dicom(data_info, image_col, image_new_col, overwrite=False):
    for ind, row in tqdm(data_info.iterrows()):
        in_path = Path(row[image_col])
        # If is dicom
        if in_path.suffix not in ['.nii', '.nii.gz']:
            continue
        out_path = Path(row[image_new_col])
        if not overwrite:
            if out_path.exists():
                continue
        else:
            if out_path.exists():
                shutil.rmtree(out_path, ignore_errors=True)
        imgprs = MedicalImageProcessor(in_path)
        imgprs.saveimg(savepath=out_path, format='dcm', meta_data=False)
    return


def add_output_path(path_df: pd.DataFrame, output_path: str, out_format: str, image_col: str = 'image',
                    label_col: str = 'label') -> pd.DataFrame:
    """
    This function takes a DataFrame and adds new paths for image and label files. The new paths are
    based on the provided output path and format.

    Parameters:
    - path_df (pd.DataFrame): The input DataFrame. It should contain columns for image and label paths.
    - output_path (str): The base path for the new image and label files.
    - out_format (str): The desired output format for the image files.
    - image_col (str): The name of the column in the DataFrame that contains the image paths.
    - label_col (str): The name of the column in the DataFrame that contains the label paths or numerical labels.

    Returns:
    - pd.DataFrame: A DataFrame with added columns for the new image and label paths.
    """

    # Determine file suffix based on the output format
    suffix = '.nii' if out_format == 'nii' else ''

    # Create new image paths and add them to the DataFrame
    output_path_img = Path(output_path) / 'IMG'
    output_path_img.mkdir(exist_ok=True, parents=True)
    path_df['image_new'] = path_df[image_col].apply(lambda x: str(output_path_img / (Path(x).name + suffix)))

    # Check if the first element in the label column is numerical
    if isinstance(path_df.loc[0, label_col], (int, float, np.number)):
        # If the labels are numerical, just copy them to the new column
        path_df['label_new'] = path_df[label_col]
    else:
        # If the labels are not numerical, assume they are paths and create new paths
        output_path_label = Path(output_path) / 'ANNO'
        output_path_label.mkdir(exist_ok=True, parents=True)
        path_df['label_new'] = path_df[image_col].apply(lambda x: str(output_path_label / (Path(x).name + suffix + '.gz')))

    return path_df


def main(args):
    # mkdir a template directory
    tmp_dir = Path('./tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    infort = args.convert_input_format
    outfort = args.convert_out_format

    path_df = pd.read_csv(args.convert_csv_path)
    label_ele = path_df.loc[0, 'label']
    if isinstance(label_ele, str) and len(label_ele) < 4:
        path_df['label'] = path_df['label'].astype('int')
    out_path = args.convert_dataset_saved
    # Need to add label information if exist
    info_table = add_output_path(path_df, out_path, outfort, image_col='image', label_col='label')
    info_table.to_csv(args.convert_out_csv_path, index=0)

    if infort == 'dicom' and outfort == 'nii':
        convert_dicom_nii(info_table, 'image', 'image_new', overwrite=args.convert_overwrite)

    if isinstance(info_table.loc[0, 'label'], str):
        image_copy(info_table, 'label', 'label_new', overwirte=args.convert_overwrite)

    if not args.convert_tmp_keep:
        shutil.rmtree(str(tmp_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image format convert', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.convert_dataset_saved:
        Path(args.convert_dataset_saved).mkdir(parents=True, exist_ok=True)
    main(args)
