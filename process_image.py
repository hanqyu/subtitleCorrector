# -*- coding: utf-8 -*-

import cv2
import numpy as np
import config
from matplotlib import pyplot as plt


# configurations
configs = config.CONFIG
max_size_of_axis = 300
resize = configs['resize']


# 이미지 불러오기
def open_image(file_path):
    image = cv2.imread(file_path)
    return image


def show_image(cv2_image):
    b, g, r = cv2.split(cv2_image)  # img파일을 b,g,r로 분리
    img2 = cv2.merge([r, g, b])  # b, r을 바꿔서 Merge
    plt.imshow(img2)


def show_image_cv2(cv2_image):
    cv2.imshow('img',cv2_image)
    cv2.waitKey(10)


def resize(cv2_image, max_size_of_axis=300):
    height, width = cv2_image.shape[:2]
    resize_ratio = max_size_of_axis / max(height, width)
    height *= resize_ratio
    width *= resize_ratio
    cv2_image = cv2.resize(cv2_image, dsize=(int(width), int(height)), interpolation=cv2.INTER_AREA)
    return cv2_image


def get_gray(cv2_image):
    copy = cv2_image.copy()
    image_gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    return image_gray


def get_canny(image_gray):
    copy = image_gray.copy()
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(copy, (kernel_size, kernel_size), 0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    return edges


def get_gradient(image_gray):
    copy = image_gray.copy()  # copy the image to be processed
    # get configs
    global configs
    kernel_size_row = configs['gradient']['kernel_size_row']
    kernel_size_col = configs['gradient']['kernel_size_col']
    # make kernel matrix for dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_row, kernel_size_col))
    # morph gradient
    image_gradient = cv2.morphologyEx(copy, cv2.MORPH_GRADIENT, kernel)
    return image_gradient


def get_threshold(image_gray):
    copy = image_gray.copy()  # copy the image to be processed
    # get configs
    global configs
    mode = configs['threshold']['mode']  # get threshold mode (mean or gaussian or global)
    block_size = configs['threshold']['block_size']
    subtract_val = configs['threshold']['subtract_val']

    if mode == 'mean':  # adaptive threshold - mean
        image_threshold = cv2.adaptiveThreshold(copy, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY_INV, block_size, subtract_val)
    elif mode == 'gaussian':  # adaptive threshold - gaussian
        image_threshold = cv2.adaptiveThreshold(copy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, block_size, subtract_val)
    else:  # (mode == 'global') global threshold - otsu's binary operation
        image_threshold = get_otsu_threshold(copy)

    return image_threshold  # Returns the image with the threshold applied.


def remove_long_line(image_binary):
    copy = image_binary.copy()  # copy the image to be processed
    # get configs
    global configs
    threshold = configs['remove_line']['threshold']
    min_line_length = configs['remove_line']['min_line_length']
    max_line_gap = configs['remove_line']['max_line_gap']

    # find and remove lines
    lines = cv2.HoughLinesP(copy, 1, np.pi / 180, threshold, np.array([]), min_line_length, max_line_gap)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(copy, (x1, y1), (x2, y2), (0, 0, 0), 2)
    return copy


def get_closing(image_gray):
    copy = image_gray.copy()  # copy the image to be processed
    # get configs
    global configs
    kernel_size_row = configs['close']['kernel_size_row']
    kernel_size_col = configs['close']['kernel_size_col']
    # make kernel matrix for dilation and erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size_row, kernel_size_col))
    # closing (dilation and erosion)
    image_close = cv2.morphologyEx(copy, cv2.MORPH_CLOSE, kernel)
    return image_close


def get_contours(image):
    # get configs
    global configs, resize
    retrieve_mode = configs['contour']['retrieve_mode']  # integer value
    approx_method = configs['contour']['approx_method']  # integer value

    if resize:
        min_width = configs['contour']['min_width_for_resize']
        min_height = configs['contour']['min_height_for_resize']
    else:
        min_width = configs['contour']['min_width']
        min_height = configs['contour']['min_height']

    # find contours from the image
    contours, _ = cv2.findContours(image, retrieve_mode, approx_method)
    result = list()
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width > min_width and height > min_height:
            result.append(contour)
    return result


def get_cropped_images(image_origin, contours):
    image_copy = image_origin.copy()  # copy the image to be processed
    # get configs
    global configs

    padding = 8  # to give the padding when cropping the screenshot
    origin_height, origin_width = image_copy.shape[:2]  # get image size
    cropped_images = []  # list to save the crop image.

    for contour in contours:  # Crop the screenshot with on bounding rectangles of contours
        x, y, width, height = cv2.boundingRect(contour)  # top-left vertex coordinates (x,y) , width, height
        # screenshot that are larger than the standard size
        # The range of row to crop (with padding)
        row_from = (y - padding) if (y - padding) > 0 else y
        row_to = (y + height + padding) if (y + height + padding) < origin_height else y + height
        # The range of column to crop (with padding)
        col_from = (x - padding) if (x - padding) > 0 else x
        col_to = (x + width + padding) if (x + width + padding) < origin_width else x + width
        # Crop the image with Numpy Array
        cropped = image_copy[row_from: row_to, col_from: col_to]
        cropped_images.append(cropped)  # add to the list
    return cropped_images


def process_image(cv2_image, line_remove=False, no_contours=False):
    # Grey-Scale
    image_gray = get_gray(cv2_image)
    # Canny
    image_canny = get_canny(image_gray)
    # Morph Gradient
    image_gradient = get_gradient(image_canny)
    if no_contours:
        return image_gradient
    else:
        # Threshold
        image_threshold = get_threshold(image_gradient)
        # Long line remove
        if line_remove:
            image_line_removed = remove_long_line(image_threshold)
        else:
            image_line_removed = image_threshold
        # Morph Close
        image_close = get_closing(image_line_removed)
        contours = get_contours(image_close)
        return contours


def draw_contour_rect(image_origin, contours, show_original=True, resize_ratio=None):
    if show_original:
        global max_size_of_axis
        if resize_ratio is None:
            resize_ratio = max_size_of_axis / max(image_origin.shape[:2])
    else:
        resize_ratio = 1
    rgb_copy = image_origin.copy()  # copy the image to be processed
    # get configs
    global configs
    min_width = configs['contour']['min_width']
    min_height = configs['contour']['min_height']
    # Draw bounding rectangles
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)  # top-left vertex coordinates (x,y) , width, height
        # Draw screenshot that are larger than the standard size
        if width > min_width and height > min_height:
            x = int(x / resize_ratio)
            y = int(y / resize_ratio)
            width = int(width / resize_ratio)
            height = int(height / resize_ratio)
            cv2.rectangle(rgb_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return rgb_copy


# 아래는 자막 영역과 관련
subtitle_standard = (509, 812, 900, 267)  # 1920px 기준, x, y, w, h


def get_standard_subtitle_region(image_origin):
    global subtitle_standard
    image_copy = image_origin.copy()
    _, og_width = image_copy.shape[:2]
    ratio_to_original = og_width / 1920
    x, y, width, height = [x * ratio_to_original for x in subtitle_standard]

    return [int(x), int(y), int(width), int(height)]


def crop_image(image, region):
    x, y, width, height = region
    image_copy = image.copy()
    crop = image_copy[y:y + height, x:x + width]

    return crop


def is_text_or_not(file_path):
    image_origin = open_image(file_path)
    # image_resize = resize(image_origin)
    # image_crop = crop_image(image_resize, get_subtitle_standard(image_resize))
    contours = process_image(image_origin)
    return contours is not None


def is_contour_in_subtitle_region(contour, subtitle_region):
    x, y, w, h = [cv2.boundingRect(contour)[i] - subtitle_region[i] for i in range(0, 4)]
    if x * y >= 0:
        if w * h >= 0:
            return True
        else:
            return False
    else:
        if w * h < 0:
            return True
        else:
            return False
