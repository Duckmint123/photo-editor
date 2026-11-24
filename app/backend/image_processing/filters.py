import cv2
import numpy as np

from .alpha import on_premultiplied

def blur(img, kernel_size):
    """Gaussian Blur với nhân vuông kernel_size (số lẻ). Ảnh trong suốt được làm mờ theo alpha."""
    return on_premultiplied(img, lambda a: cv2.GaussianBlur(a, (kernel_size, kernel_size), 0))


def sharpen(img, amount):
    """Làm nét cơ bản bằng Unsharp Mask; giữ nguyên kích thước và kênh alpha."""
    if amount == 0:
        return img.copy()

    def apply(a):
        color = a[:, :, :3]
        blurred = cv2.GaussianBlur(color, (0, 0), 1.0)
        sharp = cv2.addWeighted(color, 1.0 + amount, blurred, -amount, 0)
        return np.dstack([sharp, a[:, :, 3]]) if a.shape[2] == 4 else sharp

    return on_premultiplied(img, apply)
