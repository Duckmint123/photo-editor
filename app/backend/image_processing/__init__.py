from .basic import grayscale, to_bgr_u8
from .loader import decode_image

# Tên thao tác -> hàm xử lý: nhận ảnh và tham số đã kiểm tra, trả về ảnh mới.
# Ảnh luôn là uint8, 3 kênh (BGR) hoặc 4 kênh (BGRA).
OPERATIONS = {
    "grayscale": grayscale,
}

__all__ = ["OPERATIONS", "decode_image", "to_bgr_u8"]