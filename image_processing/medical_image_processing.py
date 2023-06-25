import filetype
import os
import sys
import cv2
import pathlib
import time
import shutil
import copy
import pydicom
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
from natsort import os_sorted
from skimage import morphology
from tqdm import tqdm, trange
from PIL import Image
from natsort import os_sorted
from scipy import ndimage
from .util import multi_image_show

'''
Author: wyd1216.git&&yudongwang117@icloud.com
Date: 2022-10-11 16:19:20
LastEditors: Yudong Wang yudongwang117@icloud.com
LastEditTime: 2022-11-02 16:15:47
FilePath: /image_processing/image_processing/medical_image_processing.py
Description: 

Copyright (c) 2022 by Yudong Wang yudongwang117@icloud.com, All Rights Reserved. 
'''

plt.clf()
plt.style.use(pathlib.Path(__file__).parent / 'mplstyle' / 'wydplot.mplstyle')


# --------------------------------------------------------------------------------------------------------------------
# ------------------------|   medical Image Processing by SimpleITK                  |--------------------------------
# --------------------------------------------------------------------------------------------------------------------
def sitk_normalize_image(image, vmin=None, vmax=None):
    """
    Normalize a 3D image to the key 0-255.

    Args:
        image (sitk.Image): 3D image to normalize.

    Returns:
        sitk.Image: Normalized 3D image.
        float: The original minimum pixel value of the image.
        float: The original key (maximum - minimum) of the image.
    """
    image_min = vmin if vmin else sitk.GetArrayViewFromImage(image).min()
    image_max = vmax if vmax else sitk.GetArrayViewFromImage(image).max()
    normalized_image = sitk.Cast(sitk.RescaleIntensity(image, outputMinimum=0, outputMaximum=255), sitk.sitkUInt8)
    return normalized_image, image_min, image_max


def sitk_recover_image(normalized_image, original_min, original_max, pixel_type=sitk.sitkInt16):
    """
    Recover the original image values from a normalized image.

    Args:
        normalized_image (sitk.Image): Normalized 3D image.
        original_min (float): The original minimum pixel value of the image.
        original_range (float): The original key (maximum - minimum) of the image.

    Returns:
        sitk.Image: Recovered 3D image with the original pixel values.
    """
    recovered_image = sitk.Cast(sitk.RescaleIntensity(normalized_image, outputMinimum=original_min,
                                                      outputMaximum=original_max), pixel_type)
    return recovered_image


def sitk_get_nth_largest_connected_component(image, n=1, exclude_value=None, full_connected=True, extract_box=False,
                                             key='size'):
    """
    Extract the nth largest connected component from a 3D image, excluding a specific pixel value.

    Args:
        image (sitk.Image): 3D image to process.
        n (int): The rank of the connected component to extract (1 for the largest, 2 for the second largest, etc.).
        exclude_value (int, optional): Pixel value to exclude from the analysis.
        full_connected (bool) : True: Use 26-connectivity of 3D and 8-connectivity of 2D. False: use 6-connectivity of
            3D and 4-connectivity of 2D.
        size_nth: if we choose key = mean, median .. to guard the size is not too small, the rank of size is demand.

    Returns:
        sitk.Image: 3D mask containing only the nth largest connected component.
    """

    # Threshold the image to exclude the specific pixel value, if provided
    if exclude_value is not None:
        binary_image = sitk.BinaryThreshold(image, lowerThreshold=exclude_value + 1, upperThreshold=255)
    else:
        binary_image = sitk.BinaryThreshold(image, lowerThreshold=1, upperThreshold=255)

    ccfilter = sitk.ConnectedComponentImageFilter()
    ccfilter.SetFullyConnected(full_connected)
    connected_components = ccfilter.Execute(binary_image)

    # Apply label shape statistics filter to get information about each connected component
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(connected_components)

    if key == 'size':
        # Get a list of labels sorted by the size of their connected components (largest to smallest)
        sorted_labels = sorted(label_shape_filter.GetLabels(),
                               key=lambda label: label_shape_filter.GetNumberOfPixels(label), reverse=True)

    elif key == 'mean':
        # The brightness is equal to the sum of pixel value.
        intensity_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
        intensity_stats_filter.Execute(connected_components, image)
        # Get a list of labels sorted by the size of their connected components (largest to smallest)

        sorted_labels = sorted(intensity_stats_filter.GetLabels(),
                               key=lambda label: intensity_stats_filter.GetMean(label), reverse=True)
        sum_sorted_labels = sorted(intensity_stats_filter.GetLabels(),
                                   key=lambda label: intensity_stats_filter.GetSum(label), reverse=True)
        sum_maximum = intensity_stats_filter.GetSum(sum_sorted_labels[0])
        sorted_sum = [intensity_stats_filter.GetSum(label) for label in sorted_labels]
        sorted_labels = [label for label in sorted_labels if intensity_stats_filter.GetSum(label) / sum_maximum > 0.3]

    # Find the label of the nth largest connected component
    if n <= len(sorted_labels):
        target_label = sorted_labels[n - 1]
    else:
        print(
            f'Warning: Only {len(sorted_labels)} connected components exist, but {n}-th is chosen. The smallest one will be extracted')
        target_label = sorted_labels[-1]

    # Create a binary image containing only the nth largest connected component
    target_component = sitk.BinaryThreshold(connected_components, lowerThreshold=target_label,
                                            upperThreshold=target_label)
    if extract_box:
        # shape_filter = sitk.LabelShapeStatisticsImageFilter()
        # shape_filter.Execute(target_component)
        # bbox = shape_filter.GetBoundingBox(1)
        bbox = label_shape_filter.GetBoundingBox(target_label)
        target_component = create_image_with_bbox(target_component.GetSize(), bbox)
        target_component.CopyInformation(image)

    return target_component


def create_image_with_bbox(shape, bbox):
    # Create a black image of the desired size
    image = sitk.Image(shape, sitk.sitkUInt8, 1)

    # Create a box image
    box_size = [bbox[i] for i in range(3, 6)]
    box_image = sitk.Image(box_size, sitk.sitkUInt8, 1)
    box_image[:, :, :] = 1

    # Paste the box image into the original image
    paste_filter = sitk.PasteImageFilter()
    paste_filter.SetSourceSize(box_size)
    paste_filter.SetSourceIndex((0, 0, 0))
    paste_filter.SetDestinationIndex(bbox[:3])
    pasted_image = paste_filter.Execute(image, box_image)
    return pasted_image


def sitk_extract_box(image: sitk.Image, bbox: tuple, crop=False):
    if crop:
        extract_image = sitk_crop_bbox(image, bbox)
    else:
        dimension = image.GetDimension()
        extract_image = sitk_extract_box_2d(image, bbox) if dimension == 2 \
            else sitk_extract_box_3d(image, bbox)

    return extract_image


def sitk_extract_box_2d(input_image: sitk.Image, bbox: tuple) -> sitk.Image:
    """
    Extract a 2D region from a SimpleITK image using a bounding box.

    Args:
        input_image (sitk.Image): The input 2D SimpleITK image.
        bbox (tuple): A tuple with four elements (min_x, min_y, size_x, size_y) representing
                              the bounding box.

    Returns:
        sitk.Image: The extracted 2D region as a SimpleITK image.
    """

    # Convert the input SimpleITK image to a NumPy array
    image_array = sitk.GetArrayFromImage(input_image)

    # Create a binary mask of the same size as the input image with all elements set to 0
    mask_array = np.zeros(image_array.shape, dtype=np.uint8)

    # Set the elements inside the bounding box to 1
    mask_array[bbox[1]:bbox[1] + bbox[3],
    bbox[0]:bbox[0] + bbox[2]] = 1

    # Multiply the input image array by the mask array to extract the region
    extract_array = image_array * mask_array

    # Convert the resulting NumPy array back to a SimpleITK image and match the input image's metadata
    extract_image = sitk.GetImageFromArray(extract_array)
    extract_image.CopyInformation(input_image)

    # Return the extracted image
    return extract_image


def sitk_extract_box_3d(input_image: sitk.Image, bbox: tuple) -> sitk.Image:
    """
    Extract a 3D region from a SimpleITK image using a bounding box.

    Args:
        input_image (sitk.Image): The input 3D SimpleITK image.
        bbox (tuple): A tuple with six elements (min_x, min_y, min_z, size_x, size_y, size_z) representing
                              the bounding box.

    Returns:
        sitk.Image: The extracted 3D region as a SimpleITK image.
    """

    # Convert the input SimpleITK image to a NumPy array
    image_array = sitk.GetArrayFromImage(input_image)

    # Create a binary mask of the same size as the input image with all elements set to 0
    mask_array = np.zeros(image_array.shape, dtype=np.uint8)

    # Set the elements inside the bounding box to 1
    mask_array[bbox[2]:bbox[2] + bbox[5],
    bbox[1]:bbox[1] + bbox[4],
    bbox[0]:bbox[0] + bbox[3]] = 1

    # Multiply the input image array by the mask array to extract the region
    extract_array = image_array * mask_array

    # Convert the resulting NumPy array back to a SimpleITK image and match the input image's metadata
    extract_image = sitk.GetImageFromArray(extract_array)
    extract_image.CopyInformation(input_image)

    # Return the extracted image
    return extract_image


def sitk_get_fg_bbox(image, bg_value=None):
    """
    Get the bounding box of the foreground in a SimpleITK image.

    Args:
        image (SimpleITK.Image): The input image.
        bg_value (float, optional): The background value. Defaults to the minimum value in the image.

    Returns:
        tuple: The bounding box containing all non-background pixels.
               For 3D images: (start_x, start_y, start_z, size_x, size_y, size_z)
               For 2D images: (start_x, start_y, size_x, size_y)
    """
    # Convert the SimpleITK image to a NumPy array
    image_np = sitk.GetArrayFromImage(image)

    # Set the background value to the minimum pixel value in the image if not provided
    if bg_value is None:
        bg_value = np.min(image_np)

    # Find the indices of non-background pixels
    non_bg_indices = np.nonzero(image_np != bg_value)

    # Get the minimum and maximum indices for each dimension
    min_z, max_z = (0, 0) if image.GetDimension() == 2 else (np.min(non_bg_indices[0]), np.max(non_bg_indices[0]))
    min_y, max_y = np.min(non_bg_indices[-2]), np.max(non_bg_indices[-2])
    min_x, max_x = np.min(non_bg_indices[-1]), np.max(non_bg_indices[-1])

    # Calculate the size of the bounding box in each dimension
    size_z = max_z - min_z + 1 if image.GetDimension() == 3 else 0
    size_y = max_y - min_y + 1
    size_x = max_x - min_x + 1

    # Create the bounding box as a tuple
    if image.GetDimension() == 3:
        bbox = (min_x, min_y, min_z, size_x, size_y, size_z)
    else:
        bbox = (min_x, min_y, size_x, size_y)

    # Convert the bounding box elements to integers
    bbox = tuple(int(x) for x in bbox)

    return bbox


def sitk_crop_bbox(image: sitk.Image, bbox: tuple) -> sitk.Image:
    """
    Crop a SimpleITK image using the given bounding box.

    Args:
        image (sitk.Image): The input image.
        bbox (tuple): The bounding box used to crop the image.
                      For 3D images: (start_x, start_y, start_z, size_x, size_y, size_z)
                      For 2D images: (start_x, start_y, size_x, size_y)

    Returns:
        sitk.Image: The cropped image.
    """
    # Create an instance of the RegionOfInterestImageFilter
    roi_filter = sitk.RegionOfInterestImageFilter()

    # Set the starting index of the region of interest
    roi_filter.SetIndex(bbox[:image.GetDimension()])

    # Set the size of the region of interest
    roi_filter.SetSize(bbox[image.GetDimension():])

    # Apply the filter to the input image to extract the region of interest
    cropped_image = roi_filter.Execute(image)

    # Return the cropped image
    return cropped_image


def sitk_pad_image(image, pad_width=(0, 0), pad_height=(0, 0), pad_depth=None, constant_value=0):
    """
    Pads a 2D or 3D medical image with a constant value along specified dimensions.

    Args:
        image (SimpleITK.Image): The input image.
        pad_width (tuple): The padding size (before, after) along the width (x-axis).
        pad_height (tuple): The padding size (before, after) along the height (y-axis).
        pad_depth (tuple, optional): The padding size (before, after) along the depth (z-axis) for 3D images.
        constant_value (int, float): The constant value used for padding.

    Returns:
        SimpleITK.Image: The padded image.
    """
    pad_width = (pad_width, pad_width) if isinstance(pad_width, (int, float)) else pad_width
    pad_height = (pad_height, pad_height) if isinstance(pad_height, (int, float)) else pad_height
    pad_depth = (0, 0) if pad_depth is None else pad_depth
    pad_depth = (pad_depth, pad_depth) if isinstance(pad_depth, (int, float)) else pad_depth
    pad_filter = sitk.ConstantPadImageFilter()

    if image.GetDimension() == 2:
        pad_filter.SetPadLowerBound([pad_width[0], pad_height[0]])
        pad_filter.SetPadUpperBound([pad_width[1], pad_height[1]])
    elif image.GetDimension() == 3:
        if pad_depth is None:
            raise ValueError("pad_depth must be provided for 3D images")
        pad_filter.SetPadLowerBound([pad_width[0], pad_height[0], pad_depth[0]])
        pad_filter.SetPadUpperBound([pad_width[1], pad_height[1], pad_depth[1]])
    else:
        raise ValueError("Unsupported image dimension")

    pad_filter.SetConstant(constant_value)
    return pad_filter.Execute(image)


# -----
def dicom_window_adjust(sitk_image, wincenter, winwidth):
    # Set the window center and window width
    # 设置窗宽窗位，设置窗宽窗位的同时会把图像缩放到【0~255】之间
    # 由于是增强期肝脏，所有窗位可以设置高一点
    # IntensityWindowingImageFilter即可进行窗宽窗位的调节
    win_min = int(wincenter - winwidth / 2.0)
    win_max = int(wincenter + winwidth / 2.0)
    intensityWindow = sitk.IntensityWindowingImageFilter()
    intensityWindow.SetWindowMaximum(win_max)
    intensityWindow.SetWindowMinimum(win_min)
    window_img_sitkImage = intensityWindow.Execute(sitk_image)
    return window_img_sitkImage


def load_image(image):
    """
    Get the sitk example_image.

    Parameters
    ----------
    image: str or sitk.Image
        input path of example_image or sitk.Image.

    Return:
    ----------
        sitk.Image
    """
    if isinstance(image, (pathlib.PosixPath, str)):
        if is_dicom_series(str(image)):
            sitk_image = read_dicom_series(image)
        elif is_nifti(str(image)):
            sitk_image = read_nifti(image)
        elif is_nrrd(str(image)):
            sitk_image = read_nifti(image)
    elif isinstance(image, sitk.SimpleITK.Image):
        sitk_image = image
    else:
        sys.stderr.write('Wrong type of input params\n')
        raise SystemExit(1)
    return sitk_image


def load_metadata(sitk_image):
    meta_dicts = {
        key: sitk_image.GetMetaData(key)
        for key in sitk_image.GetMetaDataKeys()
    }
    return meta_dicts


def write_metadata(sitk_image, meta_info):
    for key, value in meta_info.items():
        sitk_image.SetMetaData(key, value)
    return sitk_image


def read_nifti(nii_path):
    sitk_image = sitk.ReadImage(str(nii_path))
    return sitk_image


def read_dicom_series(dicoms_path):
    '''
    description: Get sitk_image by the instance-number order.
    import pathlib, SimpleITK as sitk
    param {*} dicoms_path
    return {*} sitk_image
    '''
    dicoms_path = pathlib.Path(dicoms_path)
    reader = sitk.ImageSeriesReader()
    dicomName = reader.GetGDCMSeriesFileNames(str(dicoms_path))
    reader.SetFileNames(dicomName)
    origin_sitk_image = reader.Execute()
    meta_info = load_metadata(origin_sitk_image)
    # 按照instance num排序
    pair_info = {}
    dcm_files = [x for x in dicoms_path.iterdir() if x.is_file()]
    for dcm_file in dcm_files:
        try:
            instance_num = sitk.ReadImage(
                str(dcm_file)).GetMetaData('0020|0013')
            # Delete the space at the end of instance number
            instance_num = ''.join(str(instance_num).split(' '))
            instance_num = instance_num.zfill(5)
            # instance_num = instance_num + dcm_file.name.split('.')[0]
            pair_info[instance_num] = str(dcm_file)
        except:
            print(dcm_file)
    # 按照instance num从小到大组合图像
    ranged_dcms = [pair_info[key] for key in os_sorted(pair_info)]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(ranged_dcms)
    sitk_image = reader.Execute()
    if sitk_image.GetSize() != origin_sitk_image.GetSize():
        print(
            'Detecting multi-series, choose one and resample the size as anno_sitk to resolve nonuniformity.'
        )
        channel_num = origin_sitk_image.GetSize()[2]
        sitk_image = sitk_resample_size(sitk_image, channel_num)
    sitk_image = write_metadata(sitk_image, meta_info)
    return sitk_image


def sitk_resample_spacing(sitk_image,
                          is_label=False,
                          out_spacing=[1.0, 1.0, 1.0]):
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    out_size = [
        int(np.round(original_size[0] *
                     (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] *
                     (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] *
                     (original_spacing[2] / out_spacing[2])))
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(sitk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)  ### sitk.sitkBSpline

    return resample.Execute(sitk_image)


def sitk_resample_size(sitkImage, depth):
    """
    Resampling function.
    X轴和Y轴的Size和Spacing没有变化，
    Z轴的Size和Spacing有变化
    """
    euler3d = sitk.Euler3DTransform()

    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    new_spacing_z = zspacing / (depth / float(zsize))

    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()
    # 根据新的spacing 计算新的size
    newsize = (xsize, ysize, int(zsize * zspacing / new_spacing_z))
    newspace = (xspacing, yspacing, new_spacing_z)
    sitkImage = sitk.Resample(sitkImage, newsize, euler3d,
                              sitk.sitkNearestNeighbor, origin, newspace,
                              direction)
    return sitkImage


def is_nifti(file):
    file_path = pathlib.Path(file)
    suffixes = file_path.suffixes
    if len(suffixes) == 0:
        return False
    elif suffixes[-1] == '.nii':
        return True
    elif len(suffixes) > 1 and suffixes[-2] == '.nii':
        return True
    else:
        return False


def is_nrrd(file):
    file_path = pathlib.Path(file)
    suffixes = file_path.suffixes
    if len(suffixes) == 0:
        return False
    elif suffixes[-1] == '.nrrd':
        return True
    else:
        return False


# ==================================================  File type manipulation ===========================================
def get_file_extension(file):
    """
    Get the extension of the file in true type (not the extension of file name). For all the names of extension, see
    website at "https://pypi.org/project/filetype/".

    Parameters
    ----------
    file: str
        File path

    Returns:
    ----------
    extension: str
        Extension of file in true type.
    """
    # filetyle can not recognize the directory.
    if pathlib.Path(file).is_dir():
        return ''
    kind = filetype.guess(file)
    if kind is None:
        extension = ''
    else:
        extension = kind.extension
    return extension


def get_file_mime(file):
    """
    Get the mime type of the file. For all the names of mime type, see website at "https://pypi.org/project/filetype/".

    Parameters
    ----------
    file: str
        File path

    Return:
    ----------
    mime_type: str
        mime type of file.
    """
    kind = filetype.guess(file)
    if kind is None:
        mime_type = ''
    else:
        mime_type = kind.mime
    return mime_type


def is_dcm(file):
    """
    If is the dicom file.

    Parameters
    ----------
    file: str
        File path

    Return:
    ----------
    bool
        If is the dicom file.
    """
    extension = get_file_extension(file)
    if extension == 'dcm':
        return True
    else:
        return False


def is_dicom_series(file):
    """_summary_

    Args:
        file (_type_): _description_

    Raises:
        SystemExit: _description_
    """
    file_path = pathlib.Path(file)
    if not file_path.is_dir():
        return False
    subfile = [x for x in file_path.iterdir()][0]
    return is_dcm(subfile)


# ==================================================  numpy manipulation ===========================================
def get_nonzero_indices(array):
    # Get the index of the 3D-array slices with non-zero sum of 2D-array.
    array_sum = array.sum(axis=(1, 2))
    nonzero_indices = np.where(array_sum > 0)[0]
    return nonzero_indices


def get_masked_slices(img_array, anno_array):
    '''
    description: 
    param img_array: 3D np.array 
    param anno_array: 3D np.array with same shape as img_array
    return 
    '''
    high_window = np.max(img_array)
    low_window = np.min(img_array)

    # Get the index of the slices with non-zero sum of pixels.
    masked_index = get_nonzero_indices(anno_array)

    # Get the labeled sub-array
    img_array = img_array.take(masked_index, axis=0)
    anno_array = anno_array.take(masked_index, axis=0)
    img_array = image_convert_uint8(
        img_array, low_window,
        high_window)  # Convert the example_image dtype to uint8.

    # Tag the labeled region with different color for the example_image.
    fusion_array = np.array([
        fusion_image_mask(img, anno)
        for img, anno in zip(img_array, anno_array)
    ])

    return img_array, anno_array, fusion_array


# =============================================================================================

# =============================================== Image convert ==============================================
def cvt_nifti2dicom(new_img, out_dir, meta_data=None):
    """
    new_img: sitk.Image
    """
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    direction = new_img.GetDirection()
    if meta_data is not None:
        series_tag_values = [(key, value) for key, value in meta_data.items()]
    else:
        series_tag_values = [("0008|0031", modification_time),  # Series Time
                             ("0008|0021", modification_date),  # Series Date
                             ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
                             (
                                 "0020|000e",
                                 "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
                             # Series Instance UID
                             ("0020|0037", '\\'.join(
                                 map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                           direction[1], direction[4], direction[7])))),
                             ("0008|103e", "Created-SimpleITK")]  # Series Description

    # Write slices to output directory
    list(map(lambda i: writeSlices(series_tag_values, new_img, i, out_dir), range(new_img.GetDepth())))


def writeSlices(series_tag_values, new_img, i, out_dir):
    image_slice = new_img[:, :, i]
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    # Tags shared by the series.
    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))

    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time

    # Setting the type to CT preserves the slice location.
    image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over

    # (0020, 0032) example_image position patient determines the 3D spacing between slices.
    image_slice.SetMetaData("0020|0032", '\\'.join(
        map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
    image_slice.SetMetaData("0020|0013", str(i))  # Instance Number

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer.SetFileName(os.path.join(out_dir, str(i).zfill(4) + '.dcm'))
    writer.Execute(image_slice)


# =============================================================================================
def fusion_image_mask(image, mask):
    '''
    param example_image: np.array dtypes=uint8, np.array 2D
    param label: np.array dtypes=uint8, np.array 2D
    param color_map: int, 0-12
    return fusion_image.
    '''
    pixel_values = [x for x in np.unique(mask) if x != 0]
    # Uniform gray label. From shape=(w, h) to (w, h, 3)
    mask_uniform = np.select([mask > 0], [1], default=0).astype('uint8')
    mask_gray = cv2.cvtColor(mask_uniform, cv2.COLOR_BGR2RGB)

    # Gray example_image and gray example_image with roi region set 0
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_background_image = (1 - mask_gray) * image_gray

    # Get the roi with different colormap.
    roi_list = []
    for pixel in pixel_values:
        single_mask = np.select([mask == pixel], [pixel],
                                default=0).astype('uint8')
        # Extract the label in example_image
        single_mask = single_mask // pixel
        color_ind = pixel % 12
        single_roi = image * single_mask
        # Convert 3D-roi by apply color map
        single_rgb_roi = cv2.applyColorMap(single_roi, color_ind)
        # Zero-region of roi maybe get non-zero value after convert to
        # RGB. multiply it by converted RGB-label to reset it zero.
        single_mask_gray = cv2.cvtColor(single_mask, cv2.COLOR_BGR2RGB)
        single_rgb_roi = single_rgb_roi * single_mask_gray
        roi_list.append(single_rgb_roi)
    colored_roi_image = gray_background_image
    for roi in roi_list:
        colored_roi_image = colored_roi_image + roi
    return colored_roi_image


def image_convert_uint8(img_array, low_window, high_window):
    '''
    description: 
    param {*} img_array
    param {*} low_window
    param {*} high_window
    return {*}
    '''
    lungwin = np.array([low_window * 1., high_window * 1.])
    new_array = np.select([img_array < lungwin[0], img_array > lungwin[1]],
                          [lungwin[0], lungwin[1]],
                          default=img_array)
    new_array = (new_array - lungwin[0]) / (lungwin[1] - lungwin[0])
    new_array = (new_array * 255).astype('uint8')
    return new_array


def array2png(array, savepath, low_window, high_window):
    """
    array: np.array,
    savepath: str, the name of saved new img.
    low_window: vmin value of img
    high_window: vmax value of img
    """
    new_array = image_convert_uint8(array, low_window=low_window, high_window=high_window)
    # 其中0代表图片保存时的压缩程度，有0-9这个范围的10个等级，数字越大表示压缩程度越高。
    cv2.imwrite(str(savepath), new_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def loop_add(num, loop_min=0, loop_max=100):
    if num < loop_max:
        num_add = num + 1
    elif num == loop_max:
        num_add = loop_min
    return num_add


def remove_noise(image):
    segmentation = morphology.dilation(image, np.ones((1, 1, 1)))
    labels, label_nb = ndimage.label(segmentation)

    label_count = np.bincount(labels.ravel().astype(int))  #
    label_count[0] = 0

    mask = labels == label_count.argmax()

    mask = morphology.dilation(mask, np.ones((1, 1, 1)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((1, 1, 1)))
    masked_image = mask * image
    return masked_image


def get_meta_reader(filename):
    filepath = pathlib.Path(filename)
    if filepath.is_dir():
        dcms = os_sorted([
            dcm for dcm in pathlib.Path(filename).iterdir()
            if dcm.with_suffix('.dcm')
        ])
        single_dcm = dcms[0]
    else:
        single_dcm = filepath
    if single_dcm.suffix == '.dcm':
        file_reader = sitk.ImageFileReader()
        file_reader.SetImageIO('GDCMImageIO')
        file_reader.SetFileName(str(single_dcm))
        file_reader.ReadImageInformation()
    else:
        file_reader = sitk.ReadImage(single_dcm)
    return file_reader


def array3d2png(array3d, slice_indices, savepath, low_window, high_window):
    """ Convert the 3D-array to png format

    Args:
        array3d (np.array): The 3D-array to  be converted.
        slice_indices (list(int) or -1): if list, then select the given index to convert;
            if -1, select all.
        savepath (_type_): The saved png directory.
        low_window (int): low window
        high_window (int): high window

    Returns:
        list (pahtlib.Posixpath): The saved paths of png.
    """
    savepath = pathlib.Path(savepath)
    savepath.mkdir(parents=True, exist_ok=True)
    if slice_indices == -1:
        slice_indices = range(len(array3d))
    png_paths = []
    for index in tqdm(slice_indices):
        array2d = array3d[index]
        png_path = savepath / (str(index) + '.png')
        array2png(array2d, savepath=png_path, low_window=low_window, high_window=high_window)
        png_paths.append(png_path)
    return png_paths


##method1 --opencv
def get_largest_connect_component1(img):
    """
    Get the label of largest connect component.

    Parameters
    ----------
    img: np.array
        The input img array in 2D or 3D. If 2D, shape is (h, w), if 3D, shape is (h, w, t).

    Returns:
        tuple = (float, np.array)
        The area of largest connect component and the label.
    """
    # rgb->gray
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    # img_gray = cv2.cvtColor(img, 1)
    # gaussian filter
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # binary exp-threshold=0
    _, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)  # ret==threshold
    # find contour
    contours, _ = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img_gray, contours, -1, 255, 3)
    # find the area_max region
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    if len(area) >= 1:
        max_idx = np.argmax(area)
        max_contour_area = area[max_idx]

        for k in range(len(contours)):
            if k != max_idx:
                cv2.fillPoly(img_gray, [contours[k]], 0)
    else:
        max_contour_area = 0
    return max_contour_area, img_gray


def getmaxcomponent(mask_array, num_limit=10):
    # sitk方法获取连通域
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    cca.FullyConnectedOff()
    _input = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    output_ex = cca.Execute(_input)
    labeled_img = sitk.GetArrayFromImage(output_ex)
    num = cca.GetObjectCount()
    max_label = 0
    max_num = 0
    # 不必遍历全部连通域，一般在前面就有对应全身mask的label，减少计算时间
    for i in range(1, num_limit):
        if np.sum(labeled_img == i) < 1e5:  # 全身mask的体素数量必然很大，小于设定值的不考虑
            continue
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    maxcomponent = np.array((labeled_img == max_label)).astype(np.uint8)
    # print(str(max_label) + ' num:' + str(max_num))  # 看第几个是最大的
    return maxcomponent


def get_largest_connected_domain(image):
    """
    Returns the largest connected domain of a 3D binary image.

    Args:
        image (numpy.array): A 3D binary image.

    Returns:
        numpy.array: A 3D binary image containing the largest connected domain.
    """
    # Label connected components in the 3D image
    labeled_image, num_features = ndimage.label(image)

    # Measure the size of each connected component
    sizes = ndimage.sum(image, labeled_image, range(num_features + 1))

    # Find the label of the largest connected domain
    max_label = np.argmax(sizes)

    # Create a binary image with the largest connected domain
    largest_connected_domain = (labeled_image == max_label)

    return largest_connected_domain


def get_body(CT_nii_array):
    """
    卡CT阈值获取身体（理想情况下够用了，不过多数情况下会包括到机床部分）
    """
    # 阈值二值化，获得最大的3d的连通域
    CT_array = np.copy(CT_nii_array)
    print(CT_array)
    arr_min = np.min(CT_array)
    if arr_min < -500:
        # For remove the bed by this value.
        threshold_all = -300
    else:
        threshold_all = int(np.min(CT_array) + (np.max(CT_array) - np.min(CT_array)) * 0.55)  # 卡的阈值，卡出整个身体以及机床部分
    CT_array[CT_nii_array >= threshold_all] = 1
    CT_array[CT_nii_array < threshold_all] = 0
    body_mask1 = getmaxcomponent(CT_array, 10)
    return body_mask1.astype(np.uint8)


def remove_shell(sitk_image, num=1):
    array = sitk.GetArrayFromImage(sitk_image)
    array = array[num:-num, num:-num, num:-num]
    new_sitk_image = sitk.GetImageFromArray(array)
    new_sitk_image.SetSpacing(sitk_image.GetSpacing())
    new_sitk_image.SetOrigin(sitk_image.GetOrigin())
    return new_sitk_image


def get_largest_contour_region(array2d):
    mask = array2d.copy()
    # 二值化，100为阈值，小于100的变为255，大于100的变为0
    # 也可以根据自己的要求，改变参数：
    # cv2.THRESH_BINARY
    # cv2.THRESH_BINARY_INV
    # cv2.THRESH_TRUNC
    # cv2.THRESH_TOZERO_INV
    # cv2.THRESH_TOZERO
    gray = array2d.copy()
    etVal, threshold = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    # OpenCV定义的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 腐蚀图像
    eroded = cv2.erode(threshold, kernel)
    # 膨胀图像
    dilated = cv2.dilate(eroded, kernel)
    # 高斯滤波
    Gaussian = cv2.GaussianBlur(dilated, (5, 5), 0)
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(Gaussian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return mask
    area = []
    # 找到最大的轮廓
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    # 填充最大的轮廓, 填充数值1
    mask = cv2.drawContours(mask, contours, max_idx, 1, cv2.FILLED)
    return mask


class MedicalImageProcessor:

    def __init__(self, image, label=None):
        """
        Class to read sitk.Image and make processing.

        Parameters
        ----------
        image: str or sitk.Image
            The input img array in 2D or 3D. If 2D, shape is (h, w), if 3D, shape is (h, w, c).
        label: str, sitk.Image or None, default None

        window_center_width: tuple(center, width), default None
            If None, then set the pixel value key by the original.
            else, set by (center - width//2, center + width // 2)
        """
        # Load base information
        self._sitk_image = load_image(image)
        image_reader = get_meta_reader(image)
        self._meta_data = load_metadata(image_reader)

        # label_type: 0:None, 1: int, float, 2: sitk_image
        if label is None:
            self._label = None
            self._label_type = 0
        elif isinstance(label, (int, float)):
            self._label = label
            self._label_type = 1
        else:
            self._label = load_image(label)
            self._label.CopyInformation(self._sitk_image)
            self._label_type = 2

        # Normalize
        self._origin_min = None
        self._origin_max = None
        self._origin_type = self._sitk_image.GetPixelIDValue()

        # Get number of series slices.
        image_array = sitk.GetArrayFromImage(self._sitk_image)
        self._low_window = np.min(image_array)
        self._high_window = np.max(image_array)

    def plot_single_slice(self,
                          slice_num=-1,
                          is_masked=False,
                          savefig=None):
        """
        Plots a single slice of the image, with or without label, and saves the plot to file if specified.

        Args:
            slice_num (int): the index of the slice to plot, default -1 (middle slice).
            is_masked (bool): whether to plot the label overlay on the image, default False.
            savefig (str): the file name to save the plot to, default None.

        Returns:
            ax: the plot axis object.
        """

        # Get the number of channels in the image
        channel_num = self.size[-1]

        # If slice_num is out of bounds, set it to the middle slice
        if slice_num < 0 or slice_num > channel_num - 1:
            print(
                f'Total {channel_num} slices in the dicoms sequences, the outer value will set the slice_num to be median one.')
            slice_num = channel_num // 2

        # Get the image array and the slice to plot
        image_array = sitk.GetArrayFromImage(self._sitk_image)
        image_slice = image_array[slice_num]

        # Convert the image to uint8
        image_slice = image_convert_uint8(image_slice, self._low_window, self._high_window)

        # If masked, get the label array and the masked slice
        if not self._label:
            is_masked = False
        if is_masked:
            mask_array = sitk.GetArrayFromImage(self._label)
            mask_slice = mask_array[slice_num].astype('uint8')
            fusion_slice = fusion_image_mask(image_slice, mask_slice)
        else:
            fusion_slice = image_slice

        fig, ax = multi_image_show(np.array([fusion_slice]))

        # # Save the plot to file (if applicable) and return the plot axis object
        if savefig:
            plt.savefig(savefig)
        return ax

    def plot_multi_slices(self, num=None, savefig=None):
        img_array = sitk.GetArrayFromImage(self._sitk_image)
        if self._label_type == 2:
            mask_array = sitk.GetArrayFromImage(self._label)
            img_array, mask_array, fusion_array = get_masked_slices(img_array, mask_array)
            plot_array = fusion_array
        else:
            plot_array = img_array
        if num:
            if len(plot_array) > num:
                plot_array = plot_array[(len(plot_array) - num) // 2: (len(plot_array) + num) // 2]
        multi_image_show(plot_array, cols=4)
        if savefig:
            plt.savefig(savefig)

    # Optimized above

    def saveimg(self, savepath, format='nii', meta_data=True):
        """_summary_

        Args:
            savepath (_type_): _description_
            format (str, optional): _description_. Defaults to 'nii'.
        """
        # image = write_metadata(self._sitk_image, self._meta_data)
        image = self._sitk_image
        if format == 'nii':
            sitk.WriteImage(image, savepath, useCompression=True)
        elif format == 'nrrd':
            sitk.WriteImage(image, savepath, useCompression=True)
        elif format == 'npy':
            img_array = sitk.GetArrayFromImage(image)
            np.save(savepath, img_array)
        elif format == 'dcm':
            savepath = pathlib.Path(savepath)
            if savepath.exists():
                shutil.rmtree(savepath)
            savepath.mkdir(parents=True, exist_ok=True)
            if meta_data:
                cvt_nifti2dicom(image, savepath, meta_data=self._meta_data)
            else:
                cvt_nifti2dicom(image, savepath)

    def savemask(self, savepath, format='nii'):
        """_summary_

        Args:
            saved_path (_type_): _description_
            format (str, optional): _description_. Defaults to 'nii'.
        """
        if format == 'nii':
            sitk.WriteImage(self._label, savepath, useCompression=True)
        elif format == 'nrrd':
            sitk.WriteImage(self._label, savepath, useCompression=True)
        elif format == 'npy':
            img_array = sitk.GetArrayFromImage(self._label)
            np.save(savepath, img_array)
        elif format == 'dcm':
            savepath = pathlib.Path(savepath)
            if savepath.exists():
                shutil.rmtree(savepath)
            savepath.mkdir(parents=True, exist_ok=True)
            cvt_nifti2dicom(self._label, savepath, meta_data=self._meta_data)

    def get_masked_indices(self, pixel_value=None):
        if not isinstance(self._label, (int, float)):
            mask_array = sitk.GetArrayFromImage(self._label)
            if pixel_value is not None:
                mask_array = np.select([mask_array == pixel_value],
                                       [pixel_value],
                                       default=0)
            indices = get_nonzero_indices(mask_array)
        else:
            indices = np.array([], dtype=int)
        return indices

    def generate_single_png(self, savepath, how='middle'):
        indices = self.get_masked_indices()
        if how == 'middle':
            index = indices[len(indices) // 2]
        else:
            index = -1
        image_array = sitk.GetArrayFromImage(self._sitk_image)
        png_paths = array3d2png(image_array, slice_indices=[index], savepath=savepath, low_window=self._low_window,
                                high_window=self._high_window)
        png_path = png_paths[0]
        return index, png_path

    def generate_hcc_tmp(self, index, savepath):
        image_array = sitk.GetArrayFromImage(self._sitk_image)
        mask_array = sitk.GetArrayFromImage(self._label)
        image2d = image_array[index]
        mask2d = mask_array[index]
        mask2d = np.select([mask2d > 0], [1])
        image2d = image2d * mask2d
        save_png_path = savepath / 'max_roi.png'
        array2png(image2d, str(save_png_path), low_window=self._low_window, high_window=self._high_window)

        # img_array = np.array(Image.open(img_path))
        # img_array = img_array * mask2d
        nonzero = np.nonzero(image2d)
        x_min, x_max, y_min, y_max = np.min(nonzero[0]), np.max(nonzero[0]), np.min(nonzero[1]), np.max(nonzero[1])
        x_min, x_max, y_min, y_max = x_min - 10, x_max + 10, y_min - 10, y_max + 10
        extract_array = image2d[x_min:x_max, y_min:y_max]
        new_img = Image.fromarray(extract_array.astype(np.uint8))
        new_img = new_img.resize((224, 224), Image.BILINEAR)
        new_image2d = np.array(new_img)
        save_rect_path = savepath / 'rect1_roi.png'
        array2png(new_image2d, str(save_rect_path), low_window=self._low_window, high_window=self._high_window)
        return save_png_path, save_rect_path

    def reset_window(self, win_center_width):
        # Adjust the image window using dicom_window_adjust with win_center and win_width values
        # Update the _sitk_image variable with the adjusted image
        # Pass _meta_data as the metadata parameter to write_metadata function
        win_center, win_width = win_center_width
        self._sitk_image = write_metadata(dicom_window_adjust(self._sitk_image,
                                                              wincenter=win_center,
                                                              winwidth=win_width),
                                          self._meta_data)

    def normalize(self, vmin=None, vmax=None):
        '''Normalize the image into 0-255'''
        if not self._origin_max:
            sitk_image, image_min, image_max = sitk_normalize_image(self._sitk_image, vmin, vmax)
            self._sitk_image = sitk_image
            self._origin_min = image_min
            self._origin_max = image_max

    def recover(self):
        if self._origin_min:
            self._sitk_image = sitk_recover_image(self._sitk_image, original_min=self._origin_min,
                                                  original_max=self._origin_max, type=self._origin_type)
            self._origin_max = None
            self._origin_min = None

    def get_largest_connectivity(self, n=1, exclude_value=None, method='sitk', extract_box=False, key='size'):
        sitk_image = self._sitk_image
        if method == 'sitk':
            # Normalize first
            # sitk_image, _, _ = sitk_normalize_image(sitk_image)
            sitk_mask = sitk_get_nth_largest_connected_component(sitk_image, n, exclude_value, extract_box=extract_box,
                                                                 key=key)
        # Apply the binary image as a mask to the original image
        extract_sitk_image = sitk.Mask(sitk_image, sitk_mask)
        # extract_sitk_image = sitk_mask

        # extract_sitk_image.CopyInformation(sitk_image)
        self._sitk_image = extract_sitk_image

    def crop(self, padding=None):
        '''

        :param size: first, crop the box where the background is not zero. if None, then complete. If int, the crop
                     the center value base on the size, if the crop size is larger than the figure size, then do nothing
                     or padding the bounding according to the padding.
        :param padding: if True, then padding the boudary if the crop size if larger than figure size.
        :return:
        '''
        bbox = sitk_get_fg_bbox(self._sitk_image)
        self._sitk_image = sitk_crop_bbox(self._sitk_image, bbox)
        if self._label_type == 2:
            self._label = sitk_crop_bbox(self._label, bbox)
        return bbox

    def square_pad(self, size=None):
        sitk_size = self._sitk_image.GetSize()
        max_dim = max(sitk_size[:2])
        if size is None or size <= max_dim:
            size = max_dim
        pad_width = (size - sitk_size[0]) // 2
        pad_height = (size - sitk_size[1]) // 2
        pad_width = (pad_width, size - sitk_size[0] - pad_width)
        pad_height = (pad_height, size - sitk_size[1] - pad_height)
        self._sitk_image = sitk_pad_image(self._sitk_image, pad_width=pad_width, pad_height=pad_height)
        if self._label_type == 2:
            self.label = sitk_pad_image(self.label, pad_width=pad_width, pad_height=pad_height)

    def resample_spacing(self, out_spacing=(1, 1, 1)):
        self._sitk_image = sitk_resample_spacing(self._sitk_image, is_label=False, out_spacing=out_spacing)
        self._sitk_image = write_metadata(self._sitk_image, self._meta_data)
        if self._label_type == 2:
            self._label = sitk_resample_spacing(self._label, is_label=True,
                                                out_spacing=out_spacing)
            # self._label = write_metadata(self._label, self._meta_data)

        # # Check addtional slices
        # masked_indices = self.get_masked_indices()
        # if masked_indices[-1] - masked_indices[0] + 1 != len(masked_indices):
        #     image = remove_shell(self._sitk_image)
        #     self._sitk_image = write_metadata(image, self._meta_data)
        #     if self._label
        #         mask = remove_shell(self._label)
        #         self._label = write_metadata(mask, self._meta_data)

    def get_info(self):
        if self._label is None:
            print('None label information')
            tmp_sitk = self._sitk_image
        else:
            tmp_sitk = self._label
        # The slice_end is used to store the multi-roi label infomation for inferscholar.
        keys = [key for key in load_metadata(tmp_sitk).keys()]
        if 'slice_end' in keys:
            slice_end = int(tmp_sitk.GetMetaData('slice_end'))
        else:
            slice_end = 0
        spacing_w, spacing_h, spacing_c = tmp_sitk.GetSpacing()
        # slice_dim = int(tmp_sitk.GetMetaData('dim_info'))
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(tmp_sitk)
        # num_ins = stats.GetLabels()
        info_list = []
        if self._label is None:
            labels = [1]
        else:
            labels = stats.GetLabels()
        for l in labels:
            info_dict = {}
            # this sentence fit for inferscholar data, but not external
            if slice_end == 0:
                label_id = -1
                roi_id = -1
            else:
                label_id = int(l % slice_end)
                roi_id = l // slice_end
            info_dict['label'] = label_id
            info_dict['roi_id'] = roi_id
            info_dict['pixel_value'] = int(l)
            info_dict['spacing_w'] = spacing_w
            info_dict['spacing_h'] = spacing_h
            info_dict['spacing_c'] = spacing_c
            info_dict['origin'] = tmp_sitk.GetOrigin()
            info_dict['direction'] = tmp_sitk.GetDirection()
            # info_dict['spacing'] = tmp_sitk.GetSpacing()
            w, h, c = tmp_sitk.GetSize()
            info_dict['c'] = c
            info_dict['h'] = h
            info_dict['w'] = w
            # Get roi info
            bbox = stats.GetBoundingBox(l)
            w_l, h_l, c_l, w_bias, h_bias, c_bias = bbox
            info_dict['w_l'] = w_l
            info_dict['h_l'] = h_l
            info_dict['c_l'] = c_l
            info_dict['w_bias'] = w_bias
            info_dict['h_bias'] = h_bias
            info_dict['c_bias'] = c_bias
            vol_lesion = stats.GetPhysicalSize(l)
            info_dict['vol'] = round(vol_lesion)
            info_list.append(info_dict)
        return info_list

    def get_mask_base_info(self, mask_val=-1):
        info_dict = {}
        info_dict['size'] = tuple(reversed(self._sitk_image.GetSize()))
        info_dict['spacing'] = tuple(reversed([round(x, 2) for x in (self._sitk_image).GetSpacing()]))

        image_array = sitk.GetArrayFromImage(self._sitk_image)
        img_min = np.min(image_array)
        image_array = np.select([image_array > img_min + 10], [1], default=0)
        image_nonzero = np.nonzero(image_array)
        image_low = (np.min(image_nonzero[0]), np.min(image_nonzero[1]), np.min(image_nonzero[2]))
        image_high = (np.max(image_nonzero[0]), np.max(image_nonzero[1]), np.max(image_nonzero[2]))
        image_bias = tuple((image_high[i] - image_low[i] + 1) for i in range(len(image_low)))
        info_dict['image_low'] = image_low
        info_dict['image_bias'] = image_bias

        mask_array = sitk.GetArrayFromImage(self._label)
        if mask_val != -1:
            mask_array = np.select([mask_array == mask_val], [1], default=0)
        else:
            mask_array = np.select([mask_array > 0], [1], default=0)
        mask_nonzero = np.nonzero(mask_array)
        mask_low = (np.min(mask_nonzero[0]), np.min(mask_nonzero[1]), np.min(mask_nonzero[2]))
        mask_high = (np.max(mask_nonzero[0]), np.max(mask_nonzero[1]), np.max(mask_nonzero[2]))
        mask_bias = tuple(mask_high[i] - mask_low[i] + 1 for i in range(len(mask_low)))
        info_dict['mask_low'] = mask_low
        info_dict['mask_bias'] = mask_bias
        unit_vol = np.prod(info_dict['spacing'])
        mask_vol = unit_vol * np.sum(mask_array == 1)
        info_dict['mask_vol'] = int(mask_vol)
        return info_dict

    def get_metainfo(self, tag):
        if tag in self._meta_data.keys():
            value = self._meta_data[tag]
        else:
            value = None
        return value

    def get_masked_array(self, pixel_value=None):
        image_array = sitk.GetArrayFromImage(self._sitk_image)
        mask_array = sitk.GetArrayFromImage(self._label)
        indices = self.get_masked_indices(pixel_value=pixel_value)
        if indices[-1] - indices[0] != len(indices) - 1:
            print('Not a contiguous array for the masked indices')
        masked_image_array = image_array[indices]
        masked_mask_array = mask_array[indices]
        masked_mask_array = np.select([masked_mask_array == pixel_value],
                                      [pixel_value],
                                      default=0)
        return masked_image_array, masked_mask_array, indices

    def metadata_info(self):
        meta_dict = {'Tag': [], 'Value': []}
        for key, value in self._meta_data.items():
            meta_dict['Tag'].append(key)
            meta_dict['Value'].append(value)
        return pd.DataFrame(meta_dict)

    def get_base_info(self):
        array = sitk.GetArrayFromImage(self._sitk_image)
        pixel_min, pixel_max = np.min(array), np.max(array)
        # print('spacing : ', self.spacing)
        # print('size : ', self.size)
        # print('direction : ', self._sitk_image.GetDirection())
        # print('origin: ', self._sitk_image.GetOrigin())
        # print('pixel value key : ', (pixel_min, pixel_max))
        direction = tuple(round(x, 3) for x in self._sitk_image.GetDirection())
        info_dict = {'spacing': self.spacing,
                     'size': self.size,
                     'direction': direction,
                     'origin': self._sitk_image.GetOrigin(),
                     'pixel value key': (pixel_min, pixel_max),
                     }
        index = info_dict.keys()
        value = info_dict.values()
        info_df = pd.DataFrame({'Property': index, 'Value': value})
        return info_df

    def get_metadata(self):
        return self._meta_data

    def get_sitk_image(self):
        return self._sitk_image

    @property
    def image(self):
        return self._sitk_image

    @property
    def label(self):
        return self._label

    @property
    def size(self):
        return self._sitk_image.GetSize()

    @property
    def spacing(self):
        return self._sitk_image.GetSpacing()

    @property
    def mask_size(self):
        if self._label:
            return self._label.GetSize()
        else:
            return (0, 0, 0)
