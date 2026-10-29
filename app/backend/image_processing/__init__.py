from .adjust import brightness, contrast
from .analysis import canny, otsu, threshold
from .basic import grayscale, to_bgr_u8
from .color import cool, invert, sepia, vintage, warm
from .enhance import clahe
from .filters import blur, sharpen
from .geometry import crop, crop_ratio, flip, resize, resize_percent, rotate, rotate_free
from .loader import decode_image

# Tên thao tác -> hàm xử lý: nhận ảnh và tham số đã kiểm tra, trả về ảnh mới.
# Ảnh luôn là uint8, 3 kênh (BGR) hoặc 4 kênh (BGRA).
OPERATIONS = {
    "grayscale": grayscale,
    "brightness": brightness,
    "contrast": contrast,
    "blur": blur,
    "sharpen": sharpen,
    "rotate": rotate,
    "flip": flip,
    "resize": resize,
    "crop": crop,
    # Tuần 5-6
    "clahe": clahe,
    "invert": invert,
    "sepia": sepia,
    "warm": warm,
    "cool": cool,
    "vintage": vintage,
    "canny": canny,
    "threshold": threshold,
    "otsu": otsu,
    "rotate_free": rotate_free,
    "resize_percent": resize_percent,
    "crop_ratio": crop_ratio,
}

__all__ = ["OPERATIONS", "decode_image", "to_bgr_u8"]