import os
import argparse
import shutil
import pandas as pd
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


def convert_dicom_nii(data_info, overwrite=False):
    # Rename the suffix
    data_info.iloc[:, 1] = data_info.iloc[:, 1].apply(lambda x: str(Path(x).with_suffix('.nii.gz')))
    for ind, row in tqdm(data_info.iterrows()):
        in_path = Path(row[0])
        # If is dicom
        if in_path.is_file() and in_path.suffix != '.dcm':
            continue
        out_path = Path(row[1])
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


def convert_nii_dicom(data_info, overwrite=False):
    for ind, row in tqdm(data_info.iterrows()):
        in_path = Path(row[0])
        # If is dicom
        if in_path.suffix not in ['.nii', '.nii.gz']:
            continue
        out_path = Path(row[1])
        out_path.mkdir(parents=True, exist_ok=True)
        if not overwrite:
            if out_path.exists():
                continue
        else:
            if out_path.exists():
                shutil.rmtree(out_path, ignore_errors=True)
        imgprs = MedicalImageProcessor(in_path)
        imgprs.saveimg(savepath=out_path, format='dcm', meta_data=False)
    return


def main(args):
    # mkdir a template directory
    tmp_dir = Path('./tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    infort = args.convert_in_format
    outfort = args.convert_out_format
    in_path = args.convert_dataset_path
    out_path = args.convert_dataset_saved
    # Need to add label information if exist
    info_table = create_path_info(in_path, out_path, data_depth=args.convert_dataset_depth)
    info_table.to_csv(tmp_dir/'data_info.csv', index=0)

    if infort == 'dicom' and outfort == 'nii':
        convert_dicom_nii(info_table, args.convert_overwrite)

    # Delete the template files in the process
    if not args.convert_tmp_keep:
        shutil.rmtree(str(tmp_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image format convert', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.convert_dataset_saved:
        Path(args.convert_dataset_saved).mkdir(parents=True, exist_ok=True)
    main(args)
