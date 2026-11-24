import cv2
import numpy as np

from .alpha import keep_alpha

# Ma trận sepia chuẩn, viết lại theo thứ tự kênh của OpenCV (B, G, R): mỗi hàng cho một kênh đầu ra
_SEPIA = np.array([
    [0.131, 0.534, 0.272],   # B' = 0.131 B + 0.534 G + 0.272 R
    [0.168, 0.686, 0.349],   # G'
    [0.189, 0.769, 0.393],   # R'
], np.float32)

# Ma trận chéo: nhân riêng ba kênh B, G, R bằng cv2.transform
_WARM_MATRIX = np.diag((0.88, 1.00, 1.12)).astype(np.float32)
_COOL_MATRIX = np.diag((1.12, 1.00, 0.88)).astype(np.float32)


def invert(img):
    """Âm bản: 255 - giá trị mỗi kênh màu; alpha giữ nguyên."""
    return keep_alpha(cv2.bitwise_not(np.ascontiguousarray(img[:, :, :3])), img)


def _sepia(bgr):
    return cv2.transform(bgr, _SEPIA)   # kết quả uint8, tự kẹp trong 0..255


def sepia(img):
    """Tông nâu cổ điển bằng ma trận sepia chuẩn; alpha giữ nguyên."""
    return keep_alpha(_sepia(np.ascontiguousarray(img[:, :, :3])), img)


def _gain(img, matrix):
    """Nhân mỗi kênh màu bằng ma trận OpenCV và giữ nguyên alpha."""
    bgr = np.ascontiguousarray(img[:, :, :3])
    return keep_alpha(cv2.transform(bgr, matrix), img)


def warm(img):
    """Tông ấm: tăng đỏ, giảm xanh dương."""
    return _gain(img, _WARM_MATRIX)


def cool(img):
    """Tông lạnh: tăng xanh dương, giảm đỏ."""
    return _gain(img, _COOL_MATRIX)


def vintage(img):
    """Cổ điển: trộn 60% sepia với 40% ảnh gốc, rồi làm phai (giảm tương phản, nâng vùng tối).

    out = 0.85 * trộn + 25, nên đen không còn đen hẳn và trắng không còn trắng hẳn.
    """
    bgr = np.ascontiguousarray(img[:, :, :3])
    mixed = cv2.addWeighted(_sepia(bgr), 0.6, bgr, 0.4, 0)
    faded = cv2.convertScaleAbs(mixed, alpha=0.85, beta=25)
    return keep_alpha(faded, img)