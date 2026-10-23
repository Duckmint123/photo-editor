import cv2
import numpy as np


def to_bgr_u8(img):
    """Chuẩn hóa ảnh vừa giải mã về uint8 với 3 kênh BGR hoặc 4 kênh BGRA."""
    if img.dtype == np.uint16:
        img = (img // 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if img.ndim == 2:                       # ảnh xám một kênh
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 2:                   # xám + alpha
        bgr = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        return np.dstack([bgr, img[:, :, 1]])
    return img


def grayscale(img):
    """Chuyển xám nhưng giữ 3 kênh (và alpha nếu có) để các bước sau dùng chung định dạng."""
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        out = np.dstack([out, img[:, :, 3]])
    return out