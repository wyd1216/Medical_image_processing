import os
import argparse
import shutil
import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from typing import Union
from tqdm import tqdm

from config import get_args_parser
from image_processing import MedicalImageProcessor
from report_generation import Graphs


# To list the files in two folders, match them based on their identical file names, and save the matched file paths
# as a DataFrame using Python's pathlib library
def list_files(folder_path):
    folder = Path(folder_path)
    return [file for file in folder.iterdir()]


def match_files(folder1_files, folder2_files):
    matched_files = []
    for file1 in folder1_files:
        for file2 in folder2_files:
            if file1.name in file2.name or file2.name in file1.name:
                matched_files.append((str(file1), str(file2)))
                break
    return matched_files


def create_table_info(image_dir, mask_dir, out_image_dir=None, out_mask_dir=None):
    # List files in each folder
    image_files = list_files(image_dir)
    mask_files = list_files(mask_dir)

    # Match files based on their identical names
    matched_files = match_files(image_files, mask_files)

    # Create a DataFrame from the matched file paths
    df = pd.DataFrame(matched_files, columns=['image_path', 'mask_path'])

    if out_image_dir:
        df['out_image_path'] = df['image_path'].apply(lambda x: str(out_image_dir / Path(x).name))
    if out_mask_dir:
        df['out_mask_path'] = df['mask_path'].apply(lambda x: str(out_mask_dir / Path(x).name))
    return df


# Extract the slice with the largest area based on DICOM sequence images and NIfTI masks, 
# and save the slice and corresponding label
def read_image_series(image_path):
    if Path(image_path).is_dir():
        reader = sitk.ImageSeriesReader()
        dicom_series = reader.GetGDCMSeriesFileNames(image_path)
        reader.SetFileNames(dicom_series)
        image_3d = reader.Execute()
    else:
        image_3d = sitk.ReadImage(image_path)
    return image_3d


def read_nifti_mask(mask_path):
    mask_3d = sitk.ReadImage(mask_path)
    return mask_3d


def find_largest_area_slice(mask_3d):
    mask_array = sitk.GetArrayFromImage(mask_3d)
    areas = np.sum(mask_array, axis=(1, 2))
    largest_area_slice = np.argmax(areas)
    return largest_area_slice


def save_nifti(image_2d, output_file):
    sitk.WriteImage(image_2d, output_file)


def extract_largest_slice(data_info, overwrite=False):
    data_info.iloc[:, 2] = data_info.iloc[:, 2].apply(
        lambda x: x + '.nii.gz' if 'nii' not in Path(x).suffixes else x + '.gz' if '.gz' not in Path(x).suffixes else x)
    data_info.iloc[:, 3] = data_info.iloc[:, 3].apply(
        lambda x: x + '.nii.gz' if 'nii' not in Path(x).suffixes else x + '.gz' if '.gz' not in Path(x).suffixes else x)

    for ind, row in tqdm(data_info.iterrows()):
        # Read the DICOM series and NIfTI label
        image_3d = read_image_series(row[0])
        mask_3d = read_nifti_mask(row[1])
        print('label size=', image_3d.GetSize())

        # Find the largest area slice index
        largest_area_slice = int(find_largest_area_slice(mask_3d))
        print('slice=', largest_area_slice)

        # Extract the largest area slice and corresponding label
        extracted_slice = image_3d[:, :, largest_area_slice]
        extracted_mask = mask_3d[:, :, largest_area_slice]

        # Save the extracted slice and corresponding label in NIfTI format
        print(row[2])
        print(row[3])
        save_nifti(extracted_slice, row[2])
        save_nifti(extracted_mask, row[3])


def main(args):
    # mkdir a template directory
    tmp_dir = Path('./tmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    image_root = Path(args.extract_image_path)
    mask_root = Path(args.extract_mask_path)
    image_out_root = Path(args.extract_image_out)
    mask_out_root = Path(args.extract_mask_out)
    image_out_root.mkdir(parents=True, exist_ok=True)
    mask_out_root.mkdir(parents=True, exist_ok=True)
    # Need to add label information if exist
    info_table = create_table_info(image_root, mask_root, image_out_root, mask_out_root)
    # print(info_table.head(3))
    info_table.to_csv(tmp_dir / 'data_info.csv', index=0)

    if args.extract_save_format == 'nii':
        extract_largest_slice(info_table, args.extract_overwrite)

    # Delete the template files in the process
    if not args.tmp_keep:
        shutil.rmtree(str(tmp_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image format extract', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
