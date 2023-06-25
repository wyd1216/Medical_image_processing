import argparse
import ast
import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm
from image_processing import MedicalImageProcessor
from config import get_args_parser



def main(args):
    # input params
    filename = args.preprocess_info_input
    img_col = args.preprocess_image_col
    label_col = args.preprocess_label_col
    target_spacing = tuple(args.preprocess_resample)

    df = pd.read_csv(filename)
    replace_str = args.mt_save_root_replace
    info_df = pd.DataFrame(
        columns=['origin_image', 'origin_spacing', 'origin_label', 'origin_size', 'image', 'label', 'spacing', 'size',
                 'direction', 'crop_bbox'])
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        image_path = row[img_col]
        # Here is an int value.
        label = row[label_col]
        imgprs = MedicalImageProcessor(image=image_path, label=label)
        sitk_image = imgprs.get_sitk_image()
        origin_spacing = sitk_image.GetSpacing()
        origin_size = sitk_image.GetSize()
        # origin_direction = sitk_image.GetDirection()
        # print('Origin spacing = ', origin_spacing)
        # print('Origin size = ', origin_size)

        # Manipulation in the preprocess
        # 1.Spacing resample
        imgprs.resample_spacing(out_spacing=target_spacing)

        # 2. Normalize
        imgprs.normalize()
        # imgprs.recover()

        # 3. Get largest connectivity
        imgprs.get_largest_connectivity(n=1, exclude_value=25, extract_box=True, key='mean')

        # 4. crop
        bbox = imgprs.crop(padding=5)

        # 5. padding that make its width and height equality
        imgprs.square_pad()

        # imgprs.plot_multi_slices(num=10, savefig='./test.png')

        sitk_image1 = imgprs.get_sitk_image()
        new_spacing = sitk_image1.GetSpacing()
        new_size = sitk_image1.GetSize()
        new_direction = sitk_image1.GetDirection()
        if 'bbox' not in locals():
            bbox = None
        # print('New spacing = ', new_spacing)
        # print('New size = ', new_size)
        new_image_path = image_path.replace(replace_str[0], replace_str[1])
        imgprs.saveimg(savepath=new_image_path, format='nii')
        # break

        # Information update
        info_dict = {
            'origin_image': image_path,
            'origin_label': label,
            'origin_spacing': origin_spacing,
            'origin_size': origin_size,
            'image': new_image_path,
            'label': label,
            'spacing': new_spacing,
            'size': new_size,
            'direction': new_direction,
            'crop_bbox': bbox,
        }
        # info_df.append(info_dict, ignore_index=True)
        info_df.loc[index] = info_dict

    info_df.to_csv(args.preprocess_info_savepath, index=0)

    # test array
    info1 = pd.read_csv(args.preprocess_info_savepath)
    # info1.loc[:, 'spacing'] = info1['spacing'].astype('float')
    # a1 = info1.loc[0, 'spacing']
    # a1 = ast.literal_eval(a1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image format convert', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
