import io

import cv2
import numpy as np
from PIL import Image

from .basic import to_bgr_u8

_EXIF_ORIENTATION = 0x0112


def read_orientation(data):
    """Đọc thẻ hướng EXIF (1–8); không có hoặc lỗi thì coi như 1."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            value = int(im.getexif().get(_EXIF_ORIENTATION, 1))
    except Exception:
        return 1
    return value if 1 <= value <= 8 else 1


def apply_orientation(img, orientation):
    """Xoay/lật ảnh để đúng hướng hiển thị, giống ImageOps.exif_transpose của Pillow."""
    if orientation == 2:
        out = img[:, ::-1]
    elif orientation == 3:
        out = img[::-1, ::-1]
    elif orientation == 4:
        out = img[::-1]
    elif orientation == 5:
        out = img.transpose(1, 0, 2)
    elif orientation == 6:
        out = np.rot90(img, 3)
    elif orientation == 7:
        out = np.rot90(img.transpose(1, 0, 2), 2)
    elif orientation == 8:
        out = np.rot90(img, 1)
    else:
        return img
    return np.ascontiguousarray(out)


def decode_image(data):
    """Giải mã byte của tệp ảnh thành mảng uint8 BGR/BGRA đã chuẩn hóa hướng EXIF.

    IMREAD_UNCHANGED giữ alpha và không tự xoay theo EXIF, nên việc xoay chỉ làm một lần ở đây.
    """
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        raise ValueError("Không giải mã được ảnh.")
    return apply_orientation(to_bgr_u8(img), read_orientation(data))