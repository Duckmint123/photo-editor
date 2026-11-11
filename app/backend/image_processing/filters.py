import cv2
import numpy as np

from .alpha import on_premultiplied

def blur(img, kernel_size):
    """Gaussian Blur với nhân vuông kernel_size (số lẻ). Ảnh trong suốt được làm mờ theo alpha."""
    return on_premultiplied(img, lambda a: cv2.GaussianBlur(a, (kernel_size, kernel_size), 0))
