import cv2
import numpy as np


def _apply_table(img, table):
    """Áp bảng tra lên phần màu; kênh alpha (nếu có) giữ nguyên."""
    out = cv2.LUT(np.ascontiguousarray(img[:, :, :3]), table)
    return out if img.shape[2] == 3 else np.dstack([out, img[:, :, 3]])


def brightness(img, value):
    """Cộng value*255/100 vào mỗi kênh màu (value từ -100 đến 100), kẹp trong 0..255.

    Dùng bảng tra có kẹp giá trị thay vì cv2.convertScaleAbs, vì hàm đó lấy trị tuyệt đối
    nên điểm ảnh tối khi giảm sáng sẽ bị bật thành sáng.
    """
    delta = (abs(value) * 255 + 50) // 100
    if value < 0:
        delta = -delta
    table = np.clip(np.arange(256) + delta, 0, 255).astype(np.uint8)
    return _apply_table(img, table)


def contrast(img, factor):
    """Kéo giãn quanh mức xám giữa: out = (in - 128) * factor + 128, kẹp trong 0..255."""
    values = (np.arange(256) - 128.0) * factor + 128.0
    table = np.clip(np.floor(values + 0.5), 0, 255).astype(np.uint8)
    return _apply_table(img, table)