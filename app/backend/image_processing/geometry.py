import cv2

from .alpha import on_premultiplied

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
_FLIPS = {"horizontal": 1, "vertical": 0, "both": -1}


def rotate(img, angle):
    """Xoay theo chiều kim đồng hồ 90, 180 hoặc 270 độ."""
    return cv2.rotate(img, _ROTATIONS[angle])


def flip(img, direction):
    """Lật ngang (horizontal), dọc (vertical) hoặc cả hai (both)."""
    return cv2.flip(img, _FLIPS[direction])


def resize(img, width, height):
    """Đổi kích thước: thu nhỏ dùng INTER_AREA, phóng to dùng INTER_CUBIC."""
    old_h, old_w = img.shape[:2]
    interpolation = cv2.INTER_AREA if width * height < old_w * old_h else cv2.INTER_CUBIC
    return on_premultiplied(img, lambda a: cv2.resize(a, (width, height), interpolation=interpolation))


def crop(img, x, y, width, height):
    """Cắt vùng chữ nhật theo tọa độ điểm ảnh của ảnh tại bước này (gốc tọa độ ở góc trên trái)."""
    h, w = img.shape[:2]
    if x < 0 or y < 0 or width < 1 or height < 1 or x + width > w or y + height > h:
        raise ValueError("Vùng cắt nằm ngoài ảnh.")
    return img[y:y + height, x:x + width].copy()