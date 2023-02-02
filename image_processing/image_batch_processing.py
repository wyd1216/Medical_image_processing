import scipy
import cv2
import pathlib
from cv2 import threshold
import pandas as pd
import numpy as np
from sklearn.utils import column_or_1d
from tqdm import tqdm, trange
import SimpleITK as sitk
from natsort import os_sorted

from image_processing.util import wprint
from image_processing import ImageProcess
from image_processing import get_meta_reader
#
'''
Author: wyd1216.git&&yudongwang117@icloud.com
Date: 2022-10-14 11:03:56
LastEditors: Yudong Wang yudongwang117@icloud.com
LastEditTime: 2022-10-17 15:38:46
FilePath: /image_processing/image_processing/image_batch_processing.py
Description: 

Copyright (c) 2022 by Yudong Wang yudongwang117@icloud.com, All Rights Reserved. 
'''


def select_array_by_sum(array):
    """_summary_

    Args:
        array (np.array): 3-dimension

    Returns:
        1d np.array: the indices range by 2d-sum
    """
    array_sum = array.sum(axis=(1, 2))
    index_val = np.argsort(array_sum)
    return index_val


def adjust_box_range(box_array, low=0, high=512):
    box_l = box_array[0]
    box_h = box_array[1]
    if box_h[0] - box_l[0] > high - low:
        print('box range larger than box boundary setting')
    if box_h[1] - box_l[1] > high - low:
        print('box range larger than box boundary setting')
    for ind, val in enumerate(box_l):
        if val < low:
            box_h[ind] = low - box_l[ind] + box_h[ind]
            box_l[ind] = low
    for ind, val in enumerate(box_h):
        if val > high:
            box_l[ind] = box_l[ind] - box_h[ind] + high
            box_h[ind] = high
    return np.array([box_l, box_h])


def replace_suffix(filename, remove_suffix='', add_suffix=''):
    if not remove_suffix and not add_suffix:
        return filename
    path = pathlib.Path(filename)
    if remove_suffix and path.suffix == remove_suffix:
        path = path.with_suffix('')
    if add_suffix:
        path = path.with_suffix(path.suffix + add_suffix)
    return str(path)


def expand_dilation(mask_img_arr, iterations=5):
    shape_nrrd = mask_img_arr.shape
    mask_img_arr_expand = np.zeros(shape_nrrd)
    for index in range(shape_nrrd[0]):
        mask_img_arr_expand[index, :, :] = scipy.ndimage.binary_dilation(
            mask_img_arr[index, :, :], iterations=iterations).astype('uint16')
    return mask_img_arr_expand

def get_img_anno_path(img_dir, anno_dir, anno_suffix='.nii.gz'):
    """
    Get the paths of the example_image and annotation files in the given directories.

    Parameters:
    - img_dir (str or pathlib.PosixPath or list): directory of the example_image files or a list of paths to the example_image files
    - anno_dir (str or pathlib.PosixPath or list): directory of the annotation files or a list of paths to the annotation files
    - anno_suffix (str, optional): suffix added to the example_image name to get the annotation name
      (default: '.nii.gz')

    Returns:
    - pd.DataFrame: a dataframe with two columns: 'image_path' and 'anno_path' containing the corresponding paths of the example_image and annotation files
    """
    # Convert the input directories to pathlib objects
    if isinstance(img_dir, (str, pathlib.PosixPath)):
        img_dir = pathlib.Path(img_dir)
        # Get all directories in the example_image directory and sort them
        img_path_list = os_sorted(
            [x for x in img_dir.iterdir() if x.is_dir()])
    else:
        img_path_list = img_dir
        # Get the parent directory of the first example_image path
        img_dir = pathlib.Path(img_path_list[0]).parent.absolute()
    if isinstance(anno_dir, (str, pathlib.PosixPath)):
        anno_dir = pathlib.Path(anno_dir)
        # Get all files in the annotation directory
        anno_path_list = [x for x in anno_dir.iterdir() if x.is_file()]
    else:
        anno_path_list = anno_dir
        # Get the parent directory of the first annotation path
        anno_dir = pathlib.Path(anno_path_list[0]).parent.absolute()

    # Convert the paths to absolute paths
    img_path_list = [x.absolute() for x in img_path_list]
    anno_path_list = [x.absolute() for x in anno_path_list]

    # Create a dictionary to store the example_image and annotation paths
    df_info = {'image_path': [], 'anno_path': []}

    # For each example_image path
    for img_path in img_path_list:
        # Get the name of the example_image file
        img_name = pathlib.Path(img_path).name
        # Compute the name of the corresponding annotation file
        anno_name = img_name + anno_suffix
        anno_path = anno_dir / anno_name
        # Check if the annotation file exists in the annotation directory
        if anno_path in anno_path_list:
            # Add the example_image and annotation paths to the dictionary
            df_info['image_path'].append(str(img_path))
            df_info['anno_path'].append(str(anno_path))

    # Convert the dictionary to a dataframe
    df_info = pd.Data
    return df_info

def mark_number_check(img_anno_path, anno_path_name):
    '''
    Check the number of mark for each anno.
    :param img_anno_path: 'pd.Dataframe'
    :return: 'dict', the count of mark unique number.
    '''
    mark_num = []
    info_df = img_anno_path.copy()
    for ind_, row in tqdm(img_anno_path.iterrows()):
        anno_path = row[anno_path_name]
        label_mask = sitk.ReadImage(anno_path)
        label_mask_arr = sitk.GetArrayFromImage(label_mask)
        num = np.unique(label_mask_arr)
        mark_num.append(num)
        # Normal situation: print nothing
        if len(num) == 2:
            continue
        # Abnormal situation
        elif len(num) < 2:
            print('===== <2 =====', anno_path)
            print(' ' * 10, num)
        elif len(num) > 2:
            print('===== >2 =====', anno_path)
            print(' ' * 10, num)
    info_df['marks'] = mark_num
    return info_df


def get_metainfo(filename, tag):
    file_reader = get_meta_reader(filename)
    value = file_reader.GetMetaData(tag) if file_reader.HasMetaDataKey(
        tag) else 'none'
    return value


def get_pid(path):
    filename = pathlib.Path(path).name
    pid = filename.split('_')[0]
    print(filename, 'has pid:', pid)
    return str(pid)


def select_idx_max(df, group_col, compare_col):
    idx = df.groupby(group_col)[compare_col].idxmax()
    return df.loc[idx]


def select_colval_topn(df, topn, group_col, compare_col):
    """ Group the df by column "group_col", and then select the top N largest value rows.

    Args:
        df (DataFrame): 
        topn (int): 
        group_col (str): Group the DataFrame by this col
        compare_col (str): This col should be the number columns.

    Returns:
        pd.DataFrame: _description_
    """
    tmp_df = df.groupby([group_col]).apply(
        lambda x: x.sort_values([compare_col], ascending=False)).reset_index(
            drop=True).groupby(group_col).head(topn)
    return tmp_df


def set_ranknum_by_group(df, group_col, rank_col, add_col_name=None, ascending=False):
    if add_col_name is None:
        add_col_name = rank_col + '_sortid'
    tmp_df = df.copy()
    tmp_df[add_col_name] = tmp_df[rank_col].groupby(
        tmp_df[group_col]).rank(method='first', ascending=ascending)
    return tmp_df


class DatasetManipulation:

    def __init__(self, image_root, anno_root, savepath=None):
        self._image_root = image_root
        self._anno_root = anno_root
        self._info = get_img_anno_path(image_root, anno_root)
        if savepath is not None:
            self._savepath = pathlib.Path(savepath)
        else:
            self._savepath = pathlib.Path('./output')
        self._savepath.mkdir(parents=True, exist_ok=True)
        self._current_info = self._info.copy()

    def load_info(self, info_path):
        self.update(pd.read_csv(info_path))

    def save_info(self, savepath):
        self.current_info.to_csv(savepath, index=0)

    def anno_number_check(self):
        mark_num_info = mark_number_check(self._info, 'anno_path')
        mark_num_info.to_csv(self._savepath / 'mark_number.csv', index=0)

    def set_image_info(self):
        exp_img = ImageProcess(self._info.loc[0, 'image_path'], self._info.loc[0, 'anno_path'])
        exp_img_info = exp_img.get_info()[0]
        cols = list(exp_img_info.keys())
        cols = ['image_path', 'anno_path'] + cols
        tmp_info = pd.DataFrame(columns=cols)
        for ind_, row in tqdm(self._info.iterrows()):
            imgprs = ImageProcess(row['image_path'], row['anno_path'])
            infos = imgprs.get_info()
            for info in infos:
                sinfo = {
                    'image_path': row['image_path'],
                    'anno_path': row['anno_path']
                }
                sinfo.update(info)
                tmp_info = tmp_info.append(sinfo, ignore_index=True)
        self.update(tmp_info)

    def set_meta_info(self, name_tag_dicts):
        info_df = self.current_info.copy()
        for name, tag in name_tag_dicts.items():
            info_df[name] = info_df['image_path'].apply(
                lambda x: get_metainfo(x, tag))
        self.update(info_df)
        
    def get_base_info(self, image_col, anno_col):
        """
        info = [size, spacing, image_low, image_bias, mask_low, mask_bias, vol]
        """
        tmp_info = pd.DataFrame()
        for ind_, row in tqdm(self.current_info.iterrows()):
            imgprs = ImageProcess(row[image_col], row[anno_col])
            if 'pixel_value' in list(row.keys()):
                mask_val = row['pixel_value']
            else:
                mask_val = -1
            infos = imgprs.get_base_info(mask_val=mask_val)
            path_info = {
                'pid1': row['pid1'],
                image_col: row[image_col],
                anno_col: row[anno_col]
                }
            infos.update(path_info)
            tmp_info = tmp_info.append(infos, ignore_index=True)
        return tmp_info
        
    def set_pid(self, pid='pid'):
        info_df = self.current_info
        info_df[pid] = info_df['anno_path'].apply(lambda x: get_pid(x))
        self.update(info_df) 

    def set_unique_pid(self, pid='pid', roi_id='roi_id'):
        """_summary_
        Give the roi id for the ID+roi_label

        Args:
            pid (_str_): The pid of dataset
            roi_id (_str_): The roi id 
        """
        if pid not in list(self.current_info.columns):
            self.set_pid(pid=pid)
        info_df = self.current_info
        info_df['pid1'] = info_df.apply(
            lambda row: str(row[pid]) + '-' + str(row[roi_id]), axis=1)
        self.update(info_df)

    def set_phaseid(self, key, pid='pid1'):
        """_summary_
        Give the phase id for the multi-phase

        Args:
            key (_type_): _description_
            id (_type_): _description_
        """
        info_df = self.current_info
        info_df['PhaseID'] = info_df[key].groupby(
            info_df[pid]).rank(ascending=True).astype(int)
        info_df = info_df.sort_values([pid, 'PhaseID'],
                                      ascending=[True, True])
        self.update(info_df)

    def add_clinical_info(self, filename, cols, on_col='pid1'):
        gold_df = pd.read_csv(filename)
        merge_df = gold_df[cols]
        merge_df[on_col] = merge_df[on_col].astype(str)
        info_df = self.current_info
        new_df = pd.merge(info_df, merge_df, how='left', on='pid1')
        for col in cols:
            if col != on_col:
                new_df = new_df[~new_df[col].isna()]
        self.update(new_df)

    def convert2nrrd(self, savepath=None, overwrite=False):
        if savepath is None:
            tmp_path = self.current_info.loc[0, 'image_path']
            tmp_path = pathlib.Path(tmp_path).parent.parent.parent
            savepath = tmp_path / 'NRRD'
        image_root = savepath / 'IMG'
        anno_root = savepath / 'ANNO'
        image_root.mkdir(parents=True, exist_ok=True)
        anno_root.mkdir(parents=True, exist_ok=True)
        for ind, row in tqdm(self.current_info.iterrows()):
            # The multi-roi unit share the same example_image series.
            if row['roi_id'] > 1:
                continue
            image_path = row['image_path']
            anno_path = row['anno_path']
            image_name = pathlib.Path(image_path).name + '.nrrd'
            image_saved_path = image_root / image_name
            anno_saved_path = anno_root / image_name
            if not overwrite:
                if image_saved_path.exists() and anno_saved_path.exists():
                    continue
            else:
                if image_saved_path.exists() and image_saved_path.is_file():
                    image_saved_path.unlink()
                if anno_saved_path.exists() and anno_saved_path.is_file():
                    anno_saved_path.unlink()
            imgprs = ImageProcess(image_path, anno_path)
            imgprs.reset_window(win_center_width=(80, 250))
            imgprs.remove_machine_bed()
            imgprs.saveimg(savepath=image_root / image_name, format='nrrd')
            imgprs.savemask(saved_path=anno_root / image_name, format='nrrd')
            
    def cvt2nii(self, savepath, out_spacing=None, overwrite=False):
        savepath = pathlib.Path(savepath)
        image_root = savepath / 'IMG'
        anno_root = savepath / 'ANNO'
        image_root.mkdir(parents=True, exist_ok=True)
        anno_root.mkdir(parents=True, exist_ok=True)
        for ind, row in tqdm(self.current_info.iterrows()):
            # The multi-roi unit share the same example_image series.
            if row['roi_id'] > 1:
                continue
            image_path = row['image_path']
            anno_path = row['anno_path']
            image_name = pathlib.Path(image_path).name + '.nii.gz'
            image_saved_path = image_root / image_name
            anno_saved_path = anno_root / image_name
            if not overwrite:
                if image_saved_path.exists() and anno_saved_path.exists():
                    continue
            else:
                if image_saved_path.exists() and image_saved_path.is_file():
                    image_saved_path.unlink()
                if anno_saved_path.exists() and anno_saved_path.is_file():
                    anno_saved_path.unlink()
            imgprs = ImageProcess(image_path, anno_path)
            # imgprs.reset_window(win_center_width=(80, 250))
            imgprs.get_largest_connect_area()
            if out_spacing is not None:
                imgprs.resample_spacing(out_spacing)
            # imgprs.remove_machine_bed()
            imgprs.saveimg(savepath=image_root / image_name, format='nii')
            imgprs.savemask(savepath=anno_root / image_name, format='nii')
        new_info = self._current_info.copy()
        new_info['nii_image_path'] = new_info['anno_path'].apply(lambda x: str(pathlib.Path(image_root)/(x.split('/')[-1])))
        new_info['nii_anno_path'] = new_info['anno_path'].apply(lambda x: str(pathlib.Path(anno_root)/(x.split('/')[-1])))
        self.update(new_info)
        
    def cvt2dcm(self, savepath, out_spacing=None, overwrite=False, savemask=False):
        savepath = pathlib.Path(savepath)
        image_root = savepath / 'IMG'
        anno_root = savepath / 'ANNO'
        image_root.mkdir(parents=True, exist_ok=True)
        anno_root.mkdir(parents=True, exist_ok=True)
        for ind, row in tqdm(self.current_info.iterrows()):
            # The multi-roi unit share the same example_image series.
            if row['roi_id'] > 1:
                continue
            image_path = row['image_path']
            anno_path = row['anno_path']
            image_name = pathlib.Path(image_path).name
            image_saved_path = image_root / image_name
            anno_saved_path = anno_root / image_name
            if not overwrite:
                if image_saved_path.exists() and anno_saved_path.exists():
                    continue
            else:
                if image_saved_path.exists() and image_saved_path.is_file():
                    image_saved_path.unlink()
                if anno_saved_path.exists() and anno_saved_path.is_file():
                    anno_saved_path.unlink()
            imgprs = ImageProcess(image_path, anno_path)
            # imgprs.reset_window(win_center_width=(80, 250))
            imgprs.get_largest_connect_area()
            if out_spacing is not None:
                imgprs.resample_spacing(out_spacing)
            # imgprs.remove_machine_bed()
            imgprs.saveimg(savepath=image_root / image_name, format='dcm')
            if savemask:
                imgprs.savemask(savepath=anno_root / image_name, format='dcm')
        new_info = self._current_info.copy()
        new_info['dcm_image_path'] = new_info['image_path'].apply(lambda x: str(pathlib.Path(image_root)/(x.split('/')[-1])))
        if savemask:
            new_info['dcm_anno_path'] = new_info['image_path'].apply(lambda x: str(pathlib.Path(anno_root)/(x.split('/')[-1])))
        self.update(new_info)

    def convert2png(self,
                    savepath=None,
                    overwrite=False,
                    perigma=None,
                    only_mask=True,
                    box_size=None,
                    image_type=None,
                    reset_window=None):
        if image_type is not None:
            image_name = image_type + '_' + 'image_path'
            anno_name = image_type + '_' + 'anno_path'
        else:
            image_name = 'image_path'
            anno_name = 'anno_path'
        if savepath is None:
            tmp_path = self.current_info.loc[0, image_name]
            tmp_path = pathlib.Path(tmp_path).parent.parent.parent
            savepath = tmp_path / 'PNG'
        image_root = savepath / 'IMG'
        anno_root = savepath / 'ANNO'
        image_root.mkdir(parents=True, exist_ok=True)
        anno_root.mkdir(parents=True, exist_ok=True)
        info0 = list(self.current_info.columns)
        info1 = [
            'area_ratio', 'pixel_ratio', 'border', 'slice_num',
            'png_image_path'
        ]
        df_info = pd.DataFrame(columns=info0 + info1)

        for ind, row in tqdm(self.current_info.iterrows()):
            image_path = row[image_name]
            anno_path = row[anno_name]
            png_dir_name = pathlib.Path(image_path).name
            png_dir_name = replace_suffix(png_dir_name,
                                          remove_suffix='.' + image_type)
            png_path = image_root / png_dir_name
            png_path.mkdir(parents=True, exist_ok=True)
            imgprs = ImageProcess(image_path, anno_path)
            if reset_window is not None:
                imgprs.reset_window(win_center_width=reset_window)
            image_array, mask_array, indices = imgprs.get_masked_array(
                pixel_value=row['pixel_value'])
            mask_array = mask_array / row['pixel_value']
            masked_image_array = image_array * mask_array
            sum_range_index = select_array_by_sum(mask_array)
            image_range_index = select_array_by_sum(masked_image_array)
            largest_area = mask_array.sum(axis=(1, 2))[sum_range_index[-1]]
            largest_pixel = masked_image_array.sum(
                axis=(1, 2))[image_range_index[-1]]
            if perigma is not None:
                iteration = round(perigma / row['spacing_w'])
                if iteration != 0:
                    mask_perigma = expand_dilation(mask_array,
                                                   iterations=iteration)
                else:
                    mask_perigma = mask_array
            else:
                mask_perigma = mask_array
            image_center = [
                row['h_l'] + row['h_bias'] // 2,
                row['w_l'] + row['w_bias'] // 2,
            ]
            for i in range(len(indices)):
                tmp_img = image_array[i]
                tmp_mask = mask_perigma[i]
                if only_mask:
                    final_img = tmp_img * tmp_mask
                else:
                    final_img = tmp_img
                # Extraction
                if box_size is not None:
                    box_l = [
                        image_center[0] - box_size[0] // 2,
                        image_center[1] - box_size[1] // 2
                    ]
                    box_h = [
                        image_center[0] + box_size[0] // 2,
                        image_center[1] + box_size[1] // 2
                    ]
                    box_l, box_h = adjust_box_range(np.array([box_l, box_h]))
                    final_img = final_img[box_l[0]:box_h[0], box_l[1]:box_h[1]]

                img_path = png_path / (str(indices[i]) + '.png')
                info_dict = dict(row)
                info_dict['slice_num'] = indices[i]
                if i == 0 or i == len(indices) - 1:
                    info_dict['border'] = 1
                else:
                    info_dict['border'] = 0
                info_dict['area_ratio'] = round(
                    np.sum(mask_array[i]) / largest_area, 2)
                info_dict['pixel_ratio'] = round(
                    np.sum(masked_image_array[i]) / largest_pixel, 2)
                info_dict['png_image_path'] = str(img_path)
                df_info = df_info.append(info_dict, ignore_index=True)
                # If exsit *.png files in the image_png_path.
                if not overwrite and img_path.exists():
                    continue
                cv2.imwrite(str(img_path), final_img,
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
        self.update(df_info)

    def generate_single_png(self, low_window, high_window, how='middle'):
        png_index = []
        png_paths = []
        for ind, row in tqdm(self.current_info.iterrows()):
            image_path = row['image_path']
            anno_path = row['anno_path']
            png_dir_name = pathlib.Path(image_path).name
            png_dir = self._savepath / png_dir_name
            png_dir.mkdir(parents=True, exist_ok=True)
            imgprs = ImageProcess(image_path, anno_path, window_center_width=((low_window + high_window) // 2, high_window - low_window))
            imgprs.remove_machine_bed()
            index, png_path = imgprs.generate_single_png(savepath=png_dir)
            png_index.append(index)
            png_paths.append(str(png_path))
        new_df = self.current_info.copy()
        new_df = new_df.assign(
            png_index = png_index,
            max_img_path = png_paths
        )
        self.update(new_df)

        
    def tmp_generate_hcc_png(self, low_window, high_window):
        max_roi_paths = []
        max_rect1_paths = []
        for ind, row in self.current_info.iterrows():
            index = row['png_index']
            image_path = row['image_path']
            anno_path = row['anno_path']
            png_dir_name = pathlib.Path(image_path).name
            png_dir = self._savepath / png_dir_name
            png_dir.mkdir(parents=True, exist_ok=True)
            imgprs = ImageProcess(image_path, anno_path, window_center_width=((low_window + high_window) // 2, high_window - low_window))
            imgprs.remove_machine_bed()
            max_roi_path, max_rect1_path = imgprs.generate_hcc_tmp(index=index, savepath=png_dir)
            max_roi_paths.append(str(max_roi_path))
            max_rect1_paths.append(str(max_rect1_path))
        new_df = self.current_info.copy()
        new_df = new_df.assign(
            max_roi_path = max_roi_paths,
            max_rect1_path = max_rect1_paths,
        )
        self.update(new_df)
            

        

    def add_path_info(self,
                      col_name,
                      root_path,
                      origin_suffix='',
                      new_suffix='.nii.gz'):
        self._current_info[col_name] = self._current_info['image_path'].apply(
            lambda x: str(
                pathlib.Path(root_path) / replace_suffix(
                    pathlib.Path(x).name, origin_suffix, new_suffix)))

    def dataset_split(self, refs_file=None):
        if refs_file is not None:
            ref_df = pd.read_csv(refs_file)
            if 'Phase_ID' in list(ref_df.columns):
                ref_df = ref_df[ref_df['Phase_ID'] == 1]
            ref_df = ref_df[['pid1', 'dataset']]
            df = self.current_info
            print(len(df))
            df = pd.merge(df, ref_df, how='left', on='pid1')
            print(len(df))
            self.update(df)

    def info_multi_phase_merge(self, on, add_cols, phase_name='PhaseID'):
        info_df = self.current_info
        phase_id_list = info_df[phase_name].value_counts().index.to_list()
        phase_id_list = os_sorted(phase_id_list)
        phase_df_list = [
            info_df[info_df[phase_name] == id] for id in phase_id_list
        ]
        for i in range(len(phase_df_list)):
            col_dict = {
                x: str(x) + '_' + str(phase_id_list[i])
                for x in add_cols
            }
            phase_df_list[i].rename(columns=col_dict, inplace=True)
        base_df = phase_df_list[0]
        for i in range(1, len(phase_df_list)):
            add_cols_new = [
                str(x) + '_' + str(phase_id_list[i]) for x in add_cols
            ]
            tmp_df = phase_df_list[i][on + add_cols_new]
            base_df = pd.merge(base_df, tmp_df, how='inner', on=on)
        return base_df

    def info_sel_topn_largest(self,
                              topn=1,
                              groupby='image_path',
                              target_col='score'):
        df = select_colval_topn(self.current_info,
                                topn,
                                group_col=groupby,
                                compare_col=target_col)
        self.update(df)

    def info_add_rank_col(self, group_col, rank_col, add_col_name=None):
        if add_col_name is None:
            add_col_name = rank_col + '_sortid'
        rank_df = set_ranknum_by_group(self.current_info,
                                       group_col=group_col,
                                       rank_col=rank_col,
                                       add_col_name=add_col_name)
        self._current_info = rank_df

    def info_add_new_col(self, col1, col2, new_col_name, func):
        df = self.current_info
        df[new_col_name] = df.apply(lambda row: func(row[col1], row[col2]), axis=1)
        self.update(df)

    def update(self, new_info):
        self._current_info = new_info

    @property
    def current_info(self):
        return self._current_info

    @property
    def current_info_path(self):
        return self._current_info_path
