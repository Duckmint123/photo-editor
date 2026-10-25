import cv2
import numpy as np

from .alpha import on_premultiplied

# Ba mức Sharpen bằng Unsharp Mask: out = img + amount * (img - blur(img))
SHARPEN_AMOUNTS = {1: 0.5, 2: 1.0, 3: 1.5}
SHARPEN_SIGMA = 1.2


def blur(img, kernel_size):
    """Gaussian Blur với nhân vuông kernel_size (số lẻ). Ảnh trong suốt được làm mờ theo alpha."""
    return on_premultiplied(img, lambda a: cv2.GaussianBlur(a, (kernel_size, kernel_size), 0))


def sharpen(img, level):
    """Làm nét theo ba mức 1 (nhẹ), 2 (vừa), 3 (mạnh); alpha giữ nguyên."""
    amount = SHARPEN_AMOUNTS[level]
    bgr = np.ascontiguousarray(img[:, :, :3])
    blurred = cv2.GaussianBlur(bgr, (0, 0), SHARPEN_SIGMA)
    out = cv2.addWeighted(bgr, 1.0 + amount, blurred, -amount, 0)
    return out if img.shape[2] == 3 else np.dstack([out, img[:, :, 3]])