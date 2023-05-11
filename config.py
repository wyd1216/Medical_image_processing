import argparse


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    parser = argparse.ArgumentParser('Image preprocessing for machine learning', add_help=False)
    parser.add_argument('--tmp_keep', action='store_true', default=False)
    # ------------------------------------------ information report_saved  ------------------------------------------------
    parser.add_argument('--report_title', default='', type=str, help='The title of report_saved')
    parser.add_argument('--report_describe', default='MRI dataset', type=str, help='Description about the dataset')
    parser.add_argument('--image_path', type=str, help='If single then the path of image, if multi then csv info path')
    parser.add_argument('--report_image_col', type=str, default='image', help='The image path of multi info df')
    parser.add_argument('--image_type', default='single', type=str, help='Image format')
    parser.add_argument('--report_saved_dir', default='./output/report_saved', type=str,
                        help='The path of report_saved saved')
    parser.add_argument('--report_pdf', default='report.pdf', type=str, help='The name of saved pdf file')
    parser.add_argument('--report_tmp_keep', action='store_true', default=False)


    # ------------------------------------------ Format convert ------------------------------------------------
    parser.add_argument('--convert_dataset_path', type=str, help='The path of all image used in dataset')
    parser.add_argument('--convert_dataset_depth', type=int, default=1,
                        help='The depth of data located')
    parser.add_argument('--convert_dataset_saved', type=str, help='The path of converted data saved')
    parser.add_argument('--convert_overwrite', action='store_true', default=False)
    parser.add_argument('--convert_dim', type=int, default=2, help='The type of input format')
    # Format key word: 'dicom', 'nii', 'png', 'npy', 'jpeg', 
    parser.add_argument('--convert_in_format', type=str, help='The type of input format')
    parser.add_argument('--convert_mask_path', type=str, default='', help='The type of input format')
    parser.add_argument('--convert_out_format', type=str, help='The type of output format')
    parser.add_argument('--convert_tmp_keep', action='store_true', default=False)
    parser.add_argument('--convert_extract', action='store_true', default=False)

    # ------------------------------------------ slice extract ------------------------------------------------
    parser.add_argument('--extract_image_path', type=str, help='The path of image root')
    parser.add_argument('--extract_mask_path', type=str, help='The path of label root')
    parser.add_argument('--extract_image_out', type=str, help='The path of image output')
    parser.add_argument('--extract_mask_out', type=str, help='The path of label output')
    parser.add_argument('--extract_save_format', type=str, default='nii', help='The format of output slice')
    parser.add_argument('--extract_overwrite', action='store_true', default=False)

    # ------------------------------------------ Image preprocess by monai ------------------------------------------------
    parser.add_argument('--mt_csv_path', type=str, help='The path of csv file')
    parser.add_argument('--mt_save_root_replace', nargs='+', type=str, default=[])
    parser.add_argument('--mt_image_saved_dir', type=str, help='The output dir of image.')
    parser.add_argument('--mt_mask_saved_dir', type=str, help='The output dir of label.')
    parser.add_argument('--mt_label_type', type=str, default='image', help='The output dir of label.')
    # transform option
    parser.add_argument('--mt_resample', nargs='+', type=str, default=[])
    parser.add_argument('--mt_scale255', nargs='+', type=str, default=[])

    # ------------------------------------------ Image preprocess ------------------------------------------------
    parser.add_argument('--preprocess_info_savepath', type=str, default='./preprocess_info.csv',
                        help='The general information for preprocessing')
    parser.add_argument('--preprocess_info_input', type=str, help='The input information of image.')
    parser.add_argument('--preprocess_image_col', type=str, default='image', help='The input information of image.')
    parser.add_argument('--preprocess_label_col', type=str, default='label', help='The input information of image.')
    parser.add_argument('--preprocess_resample', nargs='+', type=float, default=[1, 1, 1])

    return parser
