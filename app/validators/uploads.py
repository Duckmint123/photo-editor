import io
import os

from PIL import Image

from ..image_processing import decode_image
from .errors import ApiError

# Định dạng thật (theo nội dung tệp) -> tên dùng trong API
FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
MIB = 1024 * 1024


def _human(n):
    return f"{n // MIB} MiB" if n >= MIB else f"{n} byte"


def validate_upload(filename, data, max_bytes, max_pixels):
    """Kiểm tra theo nội dung tệp. Trả về dict {name, fmt, size_bytes, img}.

    img là ảnh đã giải mã, đã chuẩn hóa hướng EXIF, uint8 BGR/BGRA.
    """
    if not data:
        raise ApiError("EMPTY_IMAGE", "Tệp ảnh rỗng.", 400, "image")
    if len(data) > max_bytes:
        raise ApiError("FILE_TOO_LARGE", f"Tệp vượt quá {_human(max_bytes)}.", 413, "image")

    # Chỉ đọc phần đầu tệp để biết định dạng và kích thước, chưa giải mã toàn bộ ảnh
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt_name, (width, height) = im.format, im.size
            animated = bool(getattr(im, "is_animated", False))
    except Image.DecompressionBombError:
        raise ApiError("IMAGE_DIMENSIONS_EXCEEDED", "Độ phân giải ảnh quá lớn.", 422, "image")
    except Exception:
        raise ApiError("INVALID_IMAGE", "Tệp không phải ảnh hợp lệ hoặc đã bị hỏng.", 422, "image")

    if fmt_name not in FORMATS:
        raise ApiError("UNSUPPORTED_MEDIA_TYPE", "Chỉ hỗ trợ ảnh JPG, PNG hoặc WebP.", 415, "image")
    if animated:
        raise ApiError("INVALID_IMAGE", "Ảnh động chưa được hỗ trợ.", 422, "image")
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise ApiError(
            "IMAGE_DIMENSIONS_EXCEEDED",
            f"Độ phân giải ảnh vượt quá {max_pixels // 1_000_000} triệu pixel.", 422, "image",
        )

    try:
        img = decode_image(data)
    except Exception:
        raise ApiError("INVALID_IMAGE", "Không đọc được ảnh, tệp có thể bị hỏng.", 422, "image")

    name = os.path.basename((filename or "").replace("\\", "/")).strip()[:255] or "image"
    return {"name": name, "fmt": FORMATS[fmt_name], "size_bytes": len(data), "img": img}