import cv2
import numpy as np

from ..image_processing import OPERATIONS, decode_image

EXPORT_FORMATS = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "webp": "image/webp",
}
PNG_COMPRESSION = 3   # mức nén PNG cố định theo hợp đồng


def render(original_path, ops):
    """Dựng ảnh kết quả từ ảnh gốc (đã chuẩn hóa hướng) theo đúng chuỗi thao tác."""
    img = decode_image(original_path.read_bytes())
    for op in ops:
        img = OPERATIONS[op["name"]](img, **op["params"])
    return img


def describe_image(img):
    """Mô tả ảnh cho trường `current` trong JSON."""
    height, width = img.shape[:2]
    is_gray = np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(img[:, :, 1], img[:, :, 2])
    return {
        "width": int(width),
        "height": int(height),
        "color_mode": "grayscale" if is_gray else "color",
        "has_alpha": bool(img.shape[2] == 4),
    }


def _encode(ext, img, params=()):
    ok, buf = cv2.imencode(ext, img, list(params))
    if not ok:
        raise ValueError("Không mã hóa được ảnh.")
    return buf.tobytes()


def encode_png(img):
    return _encode(".png", img, (cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION))


def decode_png(data):
    """Đọc lại PNG do chính server ghi (đã là BGR/BGRA uint8)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Không đọc được ảnh hiện tại.")
    return img


def flatten_on_white(img):
    """JPEG không có alpha: trộn ảnh trong suốt lên nền trắng."""
    if img.shape[2] != 4:
        return img
    alpha = img[:, :, 3:4].astype(np.float32) / 255.0
    bgr = img[:, :, :3].astype(np.float32)
    return (bgr * alpha + 255.0 * (1.0 - alpha)).round().astype(np.uint8)


def encode_export(img, fmt, quality):
    if fmt == "png":
        return encode_png(img)
    if fmt == "jpg":
        return _encode(".jpg", flatten_on_white(img), (cv2.IMWRITE_JPEG_QUALITY, quality))
    return _encode(".webp", img, (cv2.IMWRITE_WEBP_QUALITY, quality))