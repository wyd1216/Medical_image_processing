from distutils.log import info
import sys
import cv2
import pathlib
import filetype
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from scipy import ndimage
from natsort import os_sorted
from skimage import morphology
from tqdm import tqdm, trange
from PIL import Image
from skimage.measure import label
from .util import multi_image_show

'''
Author: wyd1216.git&&yudongwang117@icloud.com
Date: 2022-10-11 16:19:20
LastEditors: Yudong Wang yudongwang117@icloud.com
LastEditTime: 2022-11-02 16:15:47
FilePath: /image_preprocess/image_preprocess/image_preprocess.py
Description: 

Copyright (c) 2022 by Yudong Wang yudongwang117@icloud.com, All Rights Reserved. 
'''

plt.clf()
plt.style.use(pathlib.Path(__file__).parent / 'mplstyle' / 'wydplot.mplstyle')


def load_image2d(image, style='gray'):
    file_extension = get_file_extension(image)
    if file_extension == 'dcm':
        sitk_image = sitk.ReadImage(image)
        image_array = sitk.GetArrayFromImage(sitk_image)
        image_array = image_array[0]
    elif file_extension in ['bmp', 'png']:
        image_array = cv2.imread(str(image))
    else:
        print(f'input image type is: {file_extension}, which not included yet')
    image_array = image_convert_uint8(image_array)
    if style == 'gray':
        if len(image_array.shape) == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    return image_array


def image_convert_uint8(img_array, low_window=None, high_window=None):
    '''
    description:
    param {*} img_array
    param {*} low_window
    param {*} high_window
    return {*}
    '''
    if not low_window:
        low_window = np.min(img_array)
    if not high_window:
        high_window = np.max(img_array)
    lungwin = np.array([low_window * 1., high_window * 1.])
    new_array = np.select([img_array < lungwin[0], img_array > lungwin[1]],
                          [lungwin[0], lungwin[1]],
                          default=img_array)
    new_array = (new_array - lungwin[0]) / (lungwin[1] - lungwin[0])
    new_array = (new_array * 255).astype('uint8')
    return new_array


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


def array2png(array, savepath, low_window=None, high_window=None):
    """
    array: np.array,
    savepath: str, the name of saved new img.
    low_window: min value of img
    high_window: max value of img
    """
    if not low_window:
        low_window = min(array)
    if not high_window:
        high_window = max(array)
    new_array = image_convert_uint8(array, low_window=low_window, high_window=high_window)
    # 其中0代表图片保存时的压缩程度，有0-9这个范围的10个等级，数字越大表示压缩程度越高。
    cv2.imwrite(str(savepath), new_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def get_largest_connect_region(img_ori, low_pixel_threshold):
    # img_ori: np.array, 0-255
    img = remove_low_pixel_areas(img_ori, low_pixel_threshold)
    max_area1, img_gray1 = get_largest_connect_component1(img)
    max_area2, img_gray2 = get_largest_connect_component2(img)
    img_gray = (img_gray1 / np.max(img_gray1)) * (img_gray2 / np.max(img_gray2))
    if len(img.shape) == 3:
        img_gray = np.array([img_gray, img_gray, img_gray])
        img_gray = img_gray.transpose(1, 2, 0)
    new_img = img_ori * (img_gray / np.max(img_gray))
    return new_img


def remove_low_pixel_areas(img, threshold):
    img_new = np.select([img <= threshold], [0], default=img).astype('uint8')
    return img_new


# method1 --opencv
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


##method2 --opencv+skimage
def get_largest_connect_component2(img):
    """
    Get the mask of largest connect component.

    Parameters
    ----------
    img: np.array
        The input img array in 2D or 3D. If 2D, shape is (h, w), if 3D, shape is (h, w, t).

    Returns:
        tuple = (float, numpy.array)
        The area of largest connect component and the mask.
    """
    # rgb->gray
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    # gaussian filter
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # binary
    _, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)  # ret==threshold
    # find contour
    labeled_img, num = label(img_gray, background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        sub_num = np.sum(labeled_img == i)
        if sub_num > max_num:
            max_num = sub_num
            max_label = i
    if max_label > 0:
        max_area = np.sum(labeled_img == max_label)
        img_gray[labeled_img != max_label] = 0
    else:
        max_area = 0
    return max_area, img_gray


# ===================================================================================================
def extract_largest_nonzero_region(array, crop_size=0):
    # Find the non-zero elements in the array
    nonzero = np.nonzero(array)
    # Determine the bounds of the non-zero region
    x_min, x_max, y_min, y_max = np.min(nonzero[0]), np.max(nonzero[0]), np.min(nonzero[1]), np.max(nonzero[1])
    # Shrink the bounds of the non-zero region by `crop_size` on each side
    x_min, x_max, y_min, y_max = x_min + crop_size, x_max - crop_size, y_min + crop_size, y_max - crop_size
    # Extract the non-zero region from the original array
    extract_array = array[x_min:x_max, y_min:y_max]
    return extract_array


def crop_minimize_zero_ratio(arr, threshold=0, test_len=5, step=2):
    """
    Crop a 2D NumPy array so that the ratio of the number of zero pixels is minimized.

    Parameters:
        arr (numpy.ndarray): A 2D NumPy array.
        threshold: the value that less than threshold will be regard as 0
        test_len: The maximum search step which could not find smaller ratio will stop
        step: The search step for the pixel

    Returns:
        numpy.ndarray: A 2D NumPy array that has been cropped to minimize the ratio of the number of zero pixels.
    """
    # Find the minimum and maximum non-zero indices of the input array along each axis
    nonzero = np.nonzero(arr)
    x_min, x_max, y_min, y_max = np.min(nonzero[0]), np.max(nonzero[0]), np.min(nonzero[1]), np.max(nonzero[1])
    ratio = 1.0
    count = 0
    for i in range(x_min, (x_max + x_min) // 2, step):
        crop_arr = arr[i:x_max, y_min:y_max]
        tmp_ratio = np.sum(crop_arr <= threshold) / crop_arr.size
        if tmp_ratio >= ratio:
            count += 1
            if count > test_len:
                count = 0
                break
        if tmp_ratio < ratio:
            ratio = tmp_ratio
            x_min = i

    for i in range(x_max, (x_max + x_min) // 2, -step):
        crop_arr = arr[x_min:x_max, y_min:y_max]
        tmp_ratio = np.sum(crop_arr <= threshold) / crop_arr.size
        if tmp_ratio >= ratio:
            count += 1
            if count > test_len:
                count = 0
                break
        if tmp_ratio < ratio:
            x_max = i
            ratio = tmp_ratio

    for i in range(y_min, (y_max + y_min) // 2, step):
        crop_arr = arr[x_min:x_max, i:y_max]
        tmp_ratio = np.sum(crop_arr <= threshold) / crop_arr.size
        if tmp_ratio >= ratio:
            count += 1
            if count > test_len:
                count = 0
                break
        if tmp_ratio < ratio:
            y_min = i
            ratio = tmp_ratio

    for i in range(y_max, (y_max + y_min) // 2, -step):
        crop_arr = arr[x_min:x_max, y_min:i]
        tmp_ratio = np.sum(crop_arr <= threshold) / crop_arr.size
        if tmp_ratio >= ratio:
            count += 1
            if count > test_len:
                count = 0
                break
        if tmp_ratio < ratio:
            y_max = i
            ratio = tmp_ratio
    return arr[x_min:x_max, y_min:y_max]


def npy2png(npyarray, savepath, low_window=None, high_window=None, names=None):
    """
    Convert the np.array into png format.

    Parameters
    ----------
    npyarray : np.array
        2D or 3D array
    savepath : str
    names : list
        if dimension of npyarray == 3, then the names of the converted png files.
    """
    # Set range of pixel value.
    if isinstance(npyarray, str):
        npyarray = np.load(npyarray)
    if low_window is None:
        low_window = np.min(npyarray)
    if high_window is None:
        high_window = np.max(npyarray)
    dim = len(npyarray.shape)
    if dim == 2:
        pathlib.Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        npy2png0(npyarray, savepath, low_window, high_window)
    elif dim == 3 and npyarray.shape[2] == 3:
        pathlib.Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        npy2png0(npyarray, savepath, low_window, high_window)
    else:
        savedir = pathlib.Path(savepath)
        savedir.mkdir(parents=True, exist_ok=True)
        for i in range(len(npyarray)):
            slice_array = npyarray[i]
            if names is not None:
                slice_name = names[i]
            else:
                slice_name = str(i)
            slice_path = savedir / (str(i) + '.png')
            npy2png0(slice_array, str(slice_path), low_window, high_window)


def npy2png0(array, save_path, low_window, high_window):
    """ 
    array: np.array
    low_window: min value of img
    high_window: max value of img
    save_path: the name of saved new img.
    """
    lungwin = np.array([low_window * 1., high_window * 1.])
    newimg = (array - lungwin[0]) / (lungwin[1] - lungwin[0])  # 归一化
    newimg = (newimg * 255).astype('uint8')  # 将像素值扩展到[0,255]
    # 其中0代表图片保存时的压缩程度，有0-9这个范围的10个等级，数字越大表示压缩程度越高。
    cv2.imwrite(str(save_path), newimg, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def get_nonzero_indices(array):
    # Get the index of the 3D-array slices with non-zero sum of 2D-array.
    array_sum = array.sum(axis=(1, 2))
    nonzero_indices = np.where(array_sum > 0)[0]
    return nonzero_indices


def fusion_image_mask(image, mask):
    '''
    param image: np.array dtypes=uint8, np.array 2D
    param mask: np.array dtypes=uint8, np.array 2D
    param color_map: int, 0-12
    return fusion_image.
    '''
    pixel_values = [x for x in np.unique(mask) if x != 0]
    # Uniform gray mask. From shape=(w, h) to (w, h, 3)
    mask_uniform = np.select([mask > 0], [1], default=0).astype('uint8')
    mask_gray = cv2.cvtColor(mask_uniform, cv2.COLOR_BGR2RGB)

    # Gray image and gray image with roi region set 0
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_background_image = (1 - mask_gray) * image_gray

    # Get the roi with different colormap.
    roi_list = []
    for pixel in pixel_values:
        single_mask = np.select([mask == pixel], [pixel],
                                default=0).astype('uint8')
        # Extract the mask in image
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


def loop_add(num, loop_min=0, loop_max=100):
    if num < loop_max:
        num_add = num + 1
    elif num == loop_max:
        num_add = loop_min
    return num_add


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


# ================================================== np.array process ================================================
class Image2dProcessing:

    def __init__(self, image, mask=None, image_style='gray'):
        """  class to read dicoms or nifti images and make processing.

        Args:
            image (_type_): str of path, PosixPath or SimpleITK.Image.

        Raises:
            SystemExit: _description_
        """
        # Load sitk image
        self._image_array = load_image2d(image, image_style)
        if mask:
            self._mask = load_image2d(mask, image_style)
        else:
            self._mask = None

    def saveimg(self, savepath, format='png'):
        if format == 'png':
            npy2png(self._image_array, savepath)

    def savemask(self, savepath, format='png'):
        if format == 'png':
            npy2png(self._mask_array, savepath)

    def image_show(self, is_masked=False, savefig=None):
        """
        Plots a single slice of the image, with or without mask, and saves the plot to file if specified.

        Args:
            is_masked (bool): Whether to plot the mask overlay on the image, default False.
            savefig (str): The file name to save the plot to, default None.

        Returns:
            ax: The plot axis object.
        """

        # Check if mask exists and set is_masked accordingly
        is_masked = is_masked and bool(self._mask)

        # Get the image to plot
        if is_masked:
            fusion_array = fusion_image_mask(self._image_array, self._mask_array)
        else:
            fusion_array = self._image_array

        # Plot the image
        fig, ax = multi_image_show(np.array([fusion_array]))

        # Save the plot to file (if applicable) and return the plot axis object
        if savefig:
            plt.savefig(savefig)
        return ax

    def keep_maximum_connectivity_area(self, low_pixel_threshold, crop_size=0, zero_ratio=0.25):
        """
        Applies image processing techniques to keep only the maximum connectivity area in the image.

        Args:
            low_pixel_threshold (int): The minimum pixel intensity value for the region to be considered as connected.
            crop_size (int): The size to crop the image around the connected region, default 0.

        Returns:
            None.
        """

        # Get the largest connected region and crop the image
        keep_array = get_largest_connect_region(self._image_array, low_pixel_threshold)
        keep_array = extract_largest_nonzero_region(keep_array, crop_size)

        # Print the ratio of non-zero values in the original image and apply further processing if necessary
        original_ratio = np.sum(keep_array <= 1) / keep_array.size
        if original_ratio > zero_ratio:
            keep_array = crop_minimize_zero_ratio(keep_array, threshold=1, test_len=5)

        # Update the image array with the processed image
        self._image_array = keep_array

    def crop(self, height=500, width=500):
        h, w = self._image_array.shape
        if h > height:
            diff = h - height
            self._image_array = self._image_array[diff // 2:h - diff // 2, :]
        if w > width:
            diff = w - width
            self._image_array = self._image_array[:, diff // 2:w - diff // 2]

    def resize(self, height=500, width=500):
        # h, w = self._image_array.shape
        self._image_array = cv2.resize(self._image_array, (height, width), interpolation=cv2.INTER_CUBIC)

    def numerical_smoothing(self):
        self._image_array = cv2.blur(self._image_array, (5, 5))

    def remove_straight_lines(self):
        # 有点作用，但总体不成功
        # Convert the image to grayscale
        gray = self._image_array.astype(np.uint8)
        image = cv2.merge((gray, gray, gray))

        # Apply Canny edge detection to detect edges in the image
        edges = cv2.Canny(gray, 150, 200, apertureSize=3)

        # Apply probabilistic Hough transform to detect straight lines in the image
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=200, maxLineGap=1)

        # Create a mask to store the detected lines
        mask = np.zeros_like(gray)

        # Draw the detected lines on the mask
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 3)

        # Apply bitwise not to invert the mask
        mask = cv2.bitwise_not(mask)

        # Apply the mask to the original image to remove the detected lines
        result = cv2.bitwise_and(image, image, mask=mask)

        # Return the image with straight lines removed
        self._image_array = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    @property
    def image_array(self):
        return self._image_array

    @property
    def shape(self):
        return self._image_array.shape
