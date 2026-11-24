import cv2
import numpy as np

from .alpha import keep_alpha


def _apply_linear(img, scale, offset):
    """Áp dụng scale * màu + offset bằng OpenCV, giữ nguyên alpha."""
    bgr = np.ascontiguousarray(img[:, :, :3])
    out = cv2.addWeighted(bgr, scale, bgr, 0, offset)
    return keep_alpha(out, img)


def brightness(img, value):
    """Chỉnh sáng từ -100 đến 100 bằng phép cộng có giới hạn của OpenCV."""
    return _apply_linear(img, 1.0, value * 255.0 / 100.0)


def contrast(img, factor):
    """Chỉnh tương phản quanh mức giữa 128 bằng cv2.addWeighted."""
    return _apply_linear(img, factor, 128.0 * (1.0 - factor))
