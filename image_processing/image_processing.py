import filetype
import os
import sys
import cv2
import pathlib
import time
import shutil
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import ndimage
from natsort import os_sorted
from skimage import morphology
from tqdm import tqdm, trange
from PIL import Image
from natsort import os_sorted

'''
Author: wyd1216.git&&yudongwang117@icloud.com
Date: 2022-10-11 16:19:20
LastEditors: Yudong Wang yudongwang117@icloud.com
LastEditTime: 2022-11-02 16:15:47
FilePath: /image_processing/image_processing/image_processing.py
Description: 

Copyright (c) 2022 by Yudong Wang yudongwang117@icloud.com, All Rights Reserved. 
'''

plt.clf()
plt.style.use(pathlib.Path(__file__).parent / 'mplstyle' / 'wydplot.mplstyle')


# plt func
def multi_image_show(array, cols=None):
    # Set up the plot
    subfig_num = len(array)
    if cols:
        cols = cols
    elif len(array) < 4:
        cols = subfig_num
    else:
        cols = 4
    rows, fig, ax = subplots_extension(subfig_num, cols)
    for i in range(subfig_num):
        _ = ax[i // cols, i % cols].imshow(array[i], 'gray')
        _ = ax[i // cols, i % cols].axis('off')  # 不显示坐标尺寸
        _ = ax[i // cols, i % cols].set_xticks([])
        _ = ax[i // cols, i % cols].set_yticks([])
    subplots_remove_residual(ax, subfig_num, rows, cols)
    # Save the plot to file (if applicable) and return the plot axis object
    return fig, ax

def subplots_remove_residual(ax, subfig_num, rows, cols):
    for i in range(subfig_num, cols * rows):
        ax[i // cols, i % cols].set_xticks([])
        ax[i // cols, i % cols].set_yticks([])
        ax[i // cols, i % cols].spines['left'].set_linewidth(0)
        ax[i // cols, i % cols].spines['right'].set_linewidth(0)
        ax[i // cols, i % cols].spines['top'].set_linewidth(0)
        ax[i // cols, i % cols].spines['bottom'].set_linewidth(0)


def subplots_extension(subfig_num, cols):
    rows = subfig_num // cols
    if subfig_num % cols > 0:
        rows += 1
    fig, ax = plt.subplots(
        rows, cols, sharex='col', sharey='row',
        figsize=(cols * 3,
                 rows * 3))  # 通过"sharex='col', sharey='row'"参数设置共享行列坐标轴
    if rows == 1:
        ax = np.array([ax])
    if cols == 1:
        ax = np.array([[x] for x in ax])
    return rows, fig, ax


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
                             "0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
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
    param mask: np.array dtypes=uint8, np.array 2D
    param color_map: int, 0-12
    return fusion_image.
    '''
    pixel_values = [x for x in np.unique(mask) if x != 0]
    # Uniform gray mask. From shape=(w, h) to (w, h, 3)
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
        # Extract the mask in example_image
        single_mask = single_mask // pixel
        color_ind = pixel % 12
        single_roi = image * single_mask
        # Convert 3D-roi by apply color map
        single_rgb_roi = cv2.applyColorMap(single_roi, color_ind)
        # Zero-region of roi maybe get non-zero value after convert to
        # RGB. multiply it by converted RGB-mask to reset it zero.
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
    low_window: min value of img
    high_window: max value of img
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
    Get the mask of largest connect component.

    Parameters
    ----------
    img: np.array
        The input img array in 2D or 3D. If 2D, shape is (h, w), if 3D, shape is (h, w, t).

    Returns:
        tuple = (float, np.array)
        The area of largest connect component and the mask.
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


class ImageProcess:

    def __init__(self, image, mask=None):
        """
        Class to read sitk.Image and make processing.

        Parameters
        ----------
        image: str or sitk.Image
            The input img array in 2D or 3D. If 2D, shape is (h, w), if 3D, shape is (h, w, c).
        mask: str, sitk.Image or None, default None

        window_center_width: tuple(center, width), default None
            If None, then set the pixel value range by the original.
            else, set by (center - width//2, center + width // 2)
        """
        # Load sitk example_image
        self._image = load_image(image)
        image_reader = get_meta_reader(image)
        self._image_meta = load_metadata(image_reader)
        if mask:
            self._mask = load_image(mask)
            self._mask_meta = load_metadata(self._mask)
            self._mask.CopyInformation(self._image)
        else:
            self._mask = None
            self._mask_meta = {}

        # Get number of series slices.
        image_array = sitk.GetArrayFromImage(self._image)
        self._low_window = np.min(image_array)
        self._high_window = np.max(image_array)

    def plot_single_slice(self,
                          slice_num=-1,
                          is_masked=False,
                          savefig=None):
        """
        Plots a single slice of the image, with or without mask, and saves the plot to file if specified.

        Args:
            slice_num (int): the index of the slice to plot, default -1 (middle slice).
            is_masked (bool): whether to plot the mask overlay on the image, default False.
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
        image_array = sitk.GetArrayFromImage(self._image)
        image_slice = image_array[slice_num]

        # Convert the image to uint8
        image_slice = image_convert_uint8(image_slice, self._low_window, self._high_window)

        # If masked, get the mask array and the masked slice
        if not self._mask:
            is_masked = False
        if is_masked:
            mask_array = sitk.GetArrayFromImage(self._mask)
            mask_slice = mask_array[slice_num].astype('uint8')
            fusion_slice = fusion_image_mask(image_slice, mask_slice)
        else:
            fusion_slice = image_slice

        fig, ax = multi_image_show(np.array([image_slice, fusion_slice]))

        # # Save the plot to file (if applicable) and return the plot axis object
        if savefig:
            plt.savefig(savefig)
        return ax

    # Optimized above

    def saveimg(self, savepath, format='nii', meta_data=True):
        """_summary_

        Args:
            savepath (_type_): _description_
            format (str, optional): _description_. Defaults to 'nii'.
        """
        image = write_metadata(self._image, self._image_meta)
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
                cvt_nifti2dicom(image, savepath, meta_data=self._image_meta)
            else:
                cvt_nifti2dicom(image, savepath)

    def savemask(self, savepath, format='nii'):
        """_summary_

        Args:
            saved_path (_type_): _description_
            format (str, optional): _description_. Defaults to 'nii'.
        """
        # self._mask = write_metadata(self._mask, self._mask_meta)
        if format == 'nii':
            sitk.WriteImage(self._mask, savepath, useCompression=True)
        elif format == 'nrrd':
            sitk.WriteImage(self._mask, savepath, useCompression=True)
        elif format == 'npy':
            img_array = sitk.GetArrayFromImage(self._mask)
            np.save(savepath, img_array)
        elif format == 'dcm':
            savepath = pathlib.Path(savepath)
            if savepath.exists():
                shutil.rmtree(savepath)
            savepath.mkdir(parents=True, exist_ok=True)
            cvt_nifti2dicom(self._mask, savepath, meta_data=self._image_meta)


    def plot_masked_slices(self, is_masked=True, savefig=None):
        image_array = sitk.GetArrayFromImage(self._image)
        mask_array = sitk.GetArrayFromImage(self._mask)
        img_array, mask_array, fusion_array = get_masked_slices(
            image_array, mask_array)
        subfig_num = len(fusion_array)
        if is_masked:
            plot_array = fusion_array
        else:
            plot_array = img_array
        cols = 4
        rows, fig, ax = subplots_extension(subfig_num, cols)
        for i in range(subfig_num):
            _ = ax[i // cols, i % cols].imshow(plot_array[i], 'gray')
            _ = ax[i // cols, i % cols].axis('off')  # 不显示坐标尺寸
            _ = ax[i // cols, i % cols].set_xticks([])
            _ = ax[i // cols, i % cols].set_yticks([])
        subplots_remove_residual(ax, subfig_num, rows, cols)
        if savefig:
            plt.savefig(savefig)

    def get_masked_indices(self, pixel_value=None):
        if self._mask:
            mask_array = sitk.GetArrayFromImage(self._mask)
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
        image_array = sitk.GetArrayFromImage(self._image)
        png_paths = array3d2png(image_array, slice_indices=[index], savepath=savepath, low_window=self._low_window,
                                high_window=self._high_window)
        png_path = png_paths[0]
        return index, png_path

    def generate_hcc_tmp(self, index, savepath):
        image_array = sitk.GetArrayFromImage(self._image)
        mask_array = sitk.GetArrayFromImage(self._mask)
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
        # Update the _image variable with the adjusted image
        # Pass _image_meta as the metadata parameter to write_metadata function
        win_center, win_width = win_center_width
        self._image = write_metadata(dicom_window_adjust(self._image,
                                                         wincenter=win_center,
                                                         winwidth=win_width),
                                     self._image_meta)

    #     def get_largest_connect_area(self):
    #         array = sitk.GetArrayFromImage(self._image)
    #         masked_image_array, masked_mask_array, indices = self.get_masked_array()
    #         array_mask = get_body(masked_image_array)
    #         array2d = array_mask[0]
    #         max_area, array2d = get_largest_connect_component1(array2d)
    #         nonzero = np.nonzero(array2d)
    #         x_min, x_max, y_min, y_max = np.min(nonzero[0]), np.max(nonzero[0]), np.min(nonzero[1]), np.max(nonzero[1])
    #         # x_min, x_max, y_min, y_max = x_min - 5, x_max + 5, y_min - 5, y_max + 5
    #
    #         array3d = np.array([array2d for x in range(len(array))])
    #
    #         # 这里注意不能直接乘上mask，因为dicom image最小元素为-1024而不是0.
    #         array_new = np.select([array3d==0],[np.min(array)], default=array)
    #
    #         array_new1 = array.copy()
    #         array_new1[:, :x_min, :] = np.min(array)
    #         array_new1[:, x_max:, :] = np.min(array)
    #         array_new1[:, :, :y_min] = np.min(array)
    #         array_new1[:, :, y_max:] = np.min(array)
    #
    #         # extract_array = array[x_min:x_max, y_min:y_max]
    #
    #         sitk_image = sitk.GetImageFromArray(array_new1)
    #         sitk_image.CopyInformation(self._image)
    #         self._image = sitk_image

    def get_largest_connect_area(self):
        image_array = sitk.GetArrayFromImage(self._image)

        # Select the masked slices to get connect region.
        indices = self.get_masked_indices()
        if indices:
            masked_image_array = image_array[indices]
        else:
            masked_image_array = image_array

        # Get the largest masked region
        array_mask = get_body(masked_image_array)
        # array3d = np.array([get_largest_connect_component1(x)[1] for x in array_mask])
        # array2d = np.sum(array3d, axis=0)

        # Again Get the largest region of the mask.
        # array2d = get_largest_connect_component1(array_mask[0])[1]
        array2d = array_mask[0]
        array2d = image_convert_uint8(array2d, np.min(array2d), np.max(array2d))

        # Fill the region of largest contour and return the final mask.
        array2d = get_largest_contour_region(array2d)
        array3d = np.array([array2d for x in range(len(image_array))])

        # 这里注意不能直接乘上mask，因为dicom image最小元素为-1024而不是0.
        array_new = np.select([array3d == 0], [np.min(image_array)], default=image_array)
        sitk_image = sitk.GetImageFromArray(array_new)
        sitk_image.CopyInformation(self._image)
        self._image = sitk_image

    def resample_spacing(self, out_spacing=[1, 1, 1]):
        self._image = sitk_resample_spacing(self._image, is_label=False, out_spacing=out_spacing)
        self._image = write_metadata(self._image, self._image_meta)
        if self._mask:
            self._mask = sitk_resample_spacing(self._mask, is_label=True,
                                               out_spacing=out_spacing)
            self._mask = write_metadata(self._mask, self._image_meta)

        # Check addtional slices
        masked_indices = self.get_masked_indices()
        if masked_indices[-1] - masked_indices[0] + 1 != len(masked_indices):
            image = remove_shell(self._image)
            self._image = write_metadata(image, self._image_meta)
            if self._mask:
                mask = remove_shell(self._mask)
                self._mask = write_metadata(mask, self._image_meta)

    def get_info(self):
        if self._mask is None:
            print('None mask information')
            tmp_sitk = self._image
        else:
            tmp_sitk = self._mask
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
        if self._mask is None:
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

    def get_base_info(self, mask_val=-1):
        info_dict = {}
        info_dict['size'] = tuple(reversed(self._image.GetSize()))
        info_dict['spacing'] = tuple(reversed([round(x, 2) for x in (self._image).GetSpacing()]))

        image_array = sitk.GetArrayFromImage(self._image)
        img_min = np.min(image_array)
        image_array = np.select([image_array > img_min + 10], [1], default=0)
        image_nonzero = np.nonzero(image_array)
        image_low = (np.min(image_nonzero[0]), np.min(image_nonzero[1]), np.min(image_nonzero[2]))
        image_high = (np.max(image_nonzero[0]), np.max(image_nonzero[1]), np.max(image_nonzero[2]))
        image_bias = tuple((image_high[i] - image_low[i] + 1) for i in range(len(image_low)))
        info_dict['image_low'] = image_low
        info_dict['image_bias'] = image_bias

        mask_array = sitk.GetArrayFromImage(self._mask)
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
        if tag in self._image_meta.keys():
            value = self._image_meta[tag]
        else:
            value = None
        return value

    def get_masked_array(self, pixel_value=None):
        image_array = sitk.GetArrayFromImage(self._image)
        mask_array = sitk.GetArrayFromImage(self._mask)
        indices = self.get_masked_indices(pixel_value=pixel_value)
        if indices[-1] - indices[0] != len(indices) - 1:
            print('Not a contiguous array for the masked indices')
        masked_image_array = image_array[indices]
        masked_mask_array = mask_array[indices]
        masked_mask_array = np.select([masked_mask_array == pixel_value],
                                      [pixel_value],
                                      default=0)
        return masked_image_array, masked_mask_array, indices

    def print_meta_info(self):
        for key, value in self._image_meta.items():
            print(key, ':', value)

    def print_info(self):
        print('spacing : ', self.spacing)
        print('size : ', self.size)
        print('direction : ', self._image.GetDirection())
        print('origin: ', self._image.GetOrigin())
        array = sitk.GetArrayFromImage(self._image)
        pixel_min, pixel_max = np.min(array), np.max(array)
        print('pixel value range : ', (pixel_min, pixel_max))

    @property
    def image(self):
        return self._image

    @property
    def mask(self):
        return self._mask
    @property
    def size(self):
        return self._image.GetSize()

    @property
    def spacing(self):
        return self._image.GetSpacing()
    @property
    def mask_size(self):
        if self._mask:
            return self._mask.GetSize()
        else:
            return (0, 0, 0)
