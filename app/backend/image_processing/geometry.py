import math

import cv2
import numpy as np

from .alpha import on_premultiplied

_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
_FLIPS = {"horizontal": 1, "vertical": 0, "both": -1}

# Tỷ lệ rộng:cao cho crop_ratio
RATIOS = {"1:1": (1, 1), "4:3": (4, 3), "16:9": (16, 9)}


def rotate(img, angle):
    """Xoay theo chiều kim đồng hồ 90, 180 hoặc 270 độ."""
    return cv2.rotate(img, _ROTATIONS[angle])


def flip(img, direction):
    """Lật ngang (horizontal), dọc (vertical) hoặc cả hai (both)."""
    return cv2.flip(img, _FLIPS[direction])


def _interpolation(old_w, old_h, width, height):
    return cv2.INTER_AREA if width * height < old_w * old_h else cv2.INTER_CUBIC


def resize(img, width, height):
    """Đổi kích thước: thu nhỏ dùng INTER_AREA, phóng to dùng INTER_CUBIC."""
    old_h, old_w = img.shape[:2]
    interpolation = _interpolation(old_w, old_h, width, height)
    return on_premultiplied(img, lambda a: cv2.resize(a, (width, height), interpolation=interpolation))


def crop(img, x, y, width, height):
    """Cắt vùng chữ nhật theo tọa độ điểm ảnh của ảnh tại bước này (gốc tọa độ ở góc trên trái)."""
    h, w = img.shape[:2]
    if x < 0 or y < 0 or width < 1 or height < 1 or x + width > w or y + height > h:
        raise ValueError("Vùng cắt nằm ngoài ảnh.")
    return img[y:y + height, x:x + width].copy()


# ---------- Tuần 5-6: xoay tự do, resize theo %, crop theo tỷ lệ ----------

def percent_size(width, height, percent):
    """Kích thước mới khi đổi cỡ theo phần trăm (làm tròn, tối thiểu 1 điểm ảnh mỗi cạnh)."""
    return max(1, round(width * percent / 100)), max(1, round(height * percent / 100))


def resize_percent(img, percent):
    """Đổi kích thước theo phần trăm so với ảnh hiện tại (100 = giữ nguyên)."""
    h, w = img.shape[:2]
    new_w, new_h = percent_size(w, h, percent)
    return resize(img, new_w, new_h)


def rotated_size(width, height, angle):
    """Kích thước hộp bao nhỏ nhất chứa ảnh sau khi xoay `angle` độ."""
    theta = math.radians(angle)
    cos, sin = abs(math.cos(theta)), abs(math.sin(theta))
    # round(..., 6) bỏ sai số dấu phẩy động (ví dụ cos 90° ≈ 6e-17) trước khi làm tròn lên
    new_w = math.ceil(round(width * cos + height * sin, 6))
    new_h = math.ceil(round(width * sin + height * cos, 6))
    return max(1, new_w), max(1, new_h)


def rotate_free(img, angle):
    """Xoay góc bất kỳ theo chiều kim đồng hồ (angle âm là ngược chiều), mở rộng khung để không mất góc.

    - Bội số của 90 độ: xoay chính xác bằng cv2.rotate, không nội suy, không thêm alpha.
    - Góc khác: nội suy tuyến tính; góc trống ở bốn phía trong suốt hoàn toàn, nên kết quả luôn có
      kênh alpha (xuất JPG sẽ được trộn lên nền trắng như mọi ảnh trong suốt).
    """
    angle = float(angle)
    quarter = angle / 90.0
    if quarter == round(quarter):
        turns = int(round(quarter)) % 4
        if turns == 0:
            return img.copy()
        return cv2.rotate(img, (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)[turns])

    h, w = img.shape[:2]
    new_w, new_h = rotated_size(w, h, angle)
    # OpenCV coi góc dương là ngược chiều kim đồng hồ, nên đổi dấu để khớp với rotate()
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
    matrix[0, 2] += new_w / 2.0 - w / 2.0   # dời tâm ảnh về tâm khung mới
    matrix[1, 2] += new_h / 2.0 - h / 2.0

    def warp(a, border, value=0):
        return cv2.warpAffine(a, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR,
                              borderMode=border, borderValue=value)

    if img.shape[2] == 4 and (img[:, :, 3] != 255).any():
        # Ảnh đã có vùng trong suốt: nhân trước alpha để màu ẩn của điểm ảnh trong suốt không lem ra viền
        return on_premultiplied(img, lambda a: warp(a, cv2.BORDER_CONSTANT), force_alpha=True)

    # Ảnh đặc (phổ biến nhất): xoay màu và alpha riêng bằng uint8, nhanh và ít tốn bộ nhớ hơn float32.
    # Màu lấy theo điểm viền gần nhất (không bị đen lem), alpha về 0 ở ngoài ảnh nên viền được khử răng cưa.
    color = warp(np.ascontiguousarray(img[:, :, :3]), cv2.BORDER_REPLICATE)
    alpha = warp(np.full((h, w), 255, np.uint8), cv2.BORDER_CONSTANT)
    return np.dstack([color, alpha])


def ratio_crop_box(width, height, ratio):
    """Vùng cắt lớn nhất, đúng tỷ lệ, nằm giữa ảnh: (x, y, rộng, cao); None nếu ảnh quá nhỏ."""
    rw, rh = RATIOS[ratio]
    k = min(width // rw, height // rh)   # chia nguyên để tỷ lệ chính xác, không lệch 1 điểm ảnh
    if k < 1:
        return None
    crop_w, crop_h = rw * k, rh * k
    return (width - crop_w) // 2, (height - crop_h) // 2, crop_w, crop_h


def crop_ratio(img, ratio):
    """Cắt vùng lớn nhất theo tỷ lệ 1:1, 4:3 hoặc 16:9 (rộng:cao), căn giữa ảnh."""
    h, w = img.shape[:2]
    box = ratio_crop_box(w, h, ratio)
    if box is None:
        raise ValueError("Ảnh quá nhỏ để cắt theo tỷ lệ này.")
    return crop(img, *box)