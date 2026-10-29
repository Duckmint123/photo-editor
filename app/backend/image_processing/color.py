import cv2
import numpy as np

from .alpha import keep_alpha

# Ma trận sepia chuẩn, viết lại theo thứ tự kênh của OpenCV (B, G, R): mỗi hàng cho một kênh đầu ra
_SEPIA = np.array([
    [0.131, 0.534, 0.272],   # B' = 0.131 B + 0.534 G + 0.272 R
    [0.168, 0.686, 0.349],   # G'
    [0.189, 0.769, 0.393],   # R'
], np.float32)

# Hệ số nhân từng kênh (B, G, R) cho tông ấm và tông lạnh
_WARM = (0.88, 1.00, 1.12)
_COOL = (1.12, 1.00, 0.88)


def invert(img):
    """Âm bản: 255 - giá trị mỗi kênh màu; alpha giữ nguyên."""
    return keep_alpha(cv2.bitwise_not(np.ascontiguousarray(img[:, :, :3])), img)


def _sepia(bgr):
    return cv2.transform(bgr, _SEPIA)   # kết quả uint8, tự kẹp trong 0..255


def sepia(img):
    """Tông nâu cổ điển bằng ma trận sepia chuẩn; alpha giữ nguyên."""
    return keep_alpha(_sepia(np.ascontiguousarray(img[:, :, :3])), img)


def _gain(img, gains):
    """Nhân từng kênh B, G, R với hệ số riêng (bảng tra), kẹp trong 0..255; alpha giữ nguyên."""
    out = np.empty_like(img)
    values = np.arange(256, dtype=np.float32)
    for channel, gain in enumerate(gains):
        table = np.clip(np.floor(values * gain + 0.5), 0, 255).astype(np.uint8)
        out[:, :, channel] = cv2.LUT(np.ascontiguousarray(img[:, :, channel]), table)
    if img.shape[2] == 4:
        out[:, :, 3] = img[:, :, 3]
    return out


def warm(img):
    """Tông ấm: tăng đỏ, giảm xanh dương."""
    return _gain(img, _WARM)


def cool(img):
    """Tông lạnh: tăng xanh dương, giảm đỏ."""
    return _gain(img, _COOL)


def vintage(img):
    """Cổ điển: trộn 60% sepia với 40% ảnh gốc, rồi làm phai (giảm tương phản, nâng vùng tối).

    out = 0.85 * trộn + 25, nên đen không còn đen hẳn và trắng không còn trắng hẳn.
    """
    bgr = np.ascontiguousarray(img[:, :, :3])
    mixed = cv2.addWeighted(_sepia(bgr), 0.6, bgr, 0.4, 0)
    faded = cv2.convertScaleAbs(mixed, alpha=0.85, beta=25)
    return keep_alpha(faded, img)