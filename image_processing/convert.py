import pathlib
import shutil
from tqdm import tqdm
from image_processing import ImageProcess


def convert_nifti_batch_to_dicom(info_table, nifti_col, dicom_col, overwrite=True):
    for ind, row in tqdm(info_table.iterrows()):
        nii_file_path = pathlib.Path(row[nifti_col])
        dicom_dir_path = pathlib.Path(row[dicom_col])
        if not overwrite:
            if dicom_dir_path.exists():
                continue
        else:
            if dicom_dir_path.exists():
                shutil.rmtree(dicom_dir_path, ignore_errors=True)
        imgprs = ImageProcess(nii_file_path)
        imgprs.saveimg(savepath=dicom_dir_path, format='dcm', meta_data=False)
