import os
import argparse
import numpy as np
import SimpleITK as sitk
import monai
import pandas as pd
import monai.transforms as mt
# from monai.transforms import (
#     Compose, LoadImaged, AddChanneld, ScaleIntensityRanged,
#     RandCropByPosNegLabeld, RandRotate90d, ToTensord
# )
from monai.data import Dataset, CSVDataset, DataLoader, NibabelWriter
from monai.utils import set_determinism, first
from config import get_args_parser
from pathlib import Path


def main(args):
    # image_saved_dir = Path(args.mt_image_saved_dir)
    # label_saved_dir = Path(args.mt_label_saved_dir)

    data_df = pd.read_csv(args.mt_csv_path)
    data_df = data_df[['image', 'label', 'ID']]
    # if args.mt_save_root_replace:
    if args.mt_label_type == 'image':
        keys = ['image', 'label']
    else:
        keys = ['image']

    target_size = (256, 256, 24)
    transforms_list = [
        mt.LoadImaged(keys=keys),
        mt.EnsureChannelFirstd(keys=keys),
    ]

    # Spacing resample
    transforms_list.append(mt.Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0)))

    # Scale value to 0 - 255.
    # transforms_list.append(mt.ScaleIntensityd(keys=keys, minv=0, maxv=255, dtype=np.int8))

    # Keep largest connectivity region.
    # transforms_list.append(mt.KeepLargestConnectedComponentd(keys=keys, connectivity=1))

    # keep same size
    # transforms_list.append(mt.SpatialPadd(keys=keys, spatial_size=(256, 256, 20)))

    transforms_list.append(mt.Resized(keys='image', spatial_size=target_size))

    # crop the retangular region
    # transforms_list.append(mt.CropForegroundd(keys=keys, source_key='image', margin=4))

    transforms = mt.Compose(transforms_list)

    dataset = CSVDataset(src=data_df, transform=transforms)
    dataloader = DataLoader(dataset=dataset)

    test0 = dataset[0]
    # print(test0)
    # print('image shape = ', test0['image'].shape)
    # image_meta = test0['image_meta_dict']
    # for key, value in image_meta.items():
    #     print(key, '   =   ', value)

    # print('image spacing = ', test0['image_meta_dict']['spacing'])
    # output_dir = args.mt_
    '''
    for index, row in data_df.iterrows():
        sample = {"image": row["image"], "label": row["label"]}
        processed_sample = transforms(sample)

        image_data = processed_sample["image"]
        pid = row["ID"]
        # print('type image ', type(image_data))
        # print(image_data.get_array())
        print(image_data.get_default_meta())

        replace_str = args.mt_save_root_replace
        image_path = data_df.loc[index, 'image']
        image_path = image_path.replace(replace_str[0], replace_str[1])
        # Save processed images
        # sitk.WriteImage(image_sitk, image_path)

        if args.mt_label_type == 'image':
            label_data = processed_sample["label"]

        # Save the processed image
        # output_image_path = os.path.join(output_image_dir, f"patient_{pid}_image_processed.nii.gz")
        # processed_image = nib.Nifti1Image(image_data, original_image.affine, original_image.header)
        # nib.save(processed_image, output_image_path)

        # # Save the processed label
        # output_label_path = os.path.join(output_label_dir, f"patient_{pid}_label_processed.nii.gz")
        # processed_label = nib.Nifti1Image(label_data, original_label.affine, original_label.header)
        # nib.save(processed_label, output_label_path)

    '''
    for idx, batch_data in enumerate(dataloader):

        replace_str = args.mt_save_root_replace
        image_path = data_df.loc[idx, 'image']
        image_path = image_path.replace(replace_str[0], replace_str[1])

        # print(batch_data)
        print(batch_data['image'].shape)
        print(batch_data['image_meta_dict'])
        for item in dir(batch_data):
            print(item)

        # itkwriter = NibabelWriter()
        # itkwriter.set_data_array(batch_data['image'])
        # itkwriter.set_metadata(batch_data['image_meta_dict'])
        # itkwriter.write(image_path)



        # image_data = batch_data["image"].numpy()
        # print(image_data.shape)
        # # Convert back to SimpleITK Image format
        # image_sitk = sitk.GetImageFromArray(np.transpose(image_data[0], (0, 3, 1, 2)))
        # # Save processed images
        # sitk.WriteImage(image_sitk, image_path)

        # if args.mt_label_type == 'image':
        #     label_data = batch_data["label"].numpy()
        #     label_sitk = sitk.GetImageFromArray(label_data[0])
        #     label_path = data_df.loc[idx, 'label']
        #     label_path = image_path.replace(replace_str[0], replace_str[1])
        #     # Save processed images
        #     sitk.WriteImage(label_sitk, label_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Image format convert', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
