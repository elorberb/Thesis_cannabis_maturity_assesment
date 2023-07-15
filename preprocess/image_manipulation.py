import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def resize_image(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image


def is_monochromatic(image, tolerance=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_std = np.std(h)
    s_std = np.std(s)
    v_std = np.std(v)
    if h_std > tolerance or s_std > tolerance or v_std > tolerance:
        return False
    else:
        return True


def is_blurry(image, threshold=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    return rotated


def rescale_image(image, scaling):
    width, height = int(image.shape[0] * scaling), int(image.shape[1] * scaling)
    resized_image = cv2.resize(image, (width, height))
    return resized_image


def flip_image(image, direction):
    if direction == "horizontal":
        flip_code = 0
    elif direction == "vertical":
        flip_code = 1
    elif direction == "both":
        flip_code = -1
    flipped_img = cv2.flip(image, flip_code)
    return flipped_img


def rgb_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image


def rgb_to_lab(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return lab_image


def dilation(image):
    kernel_size = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated_image = cv2.dilate(image, kernel)
    return dilated_image
