import re

from .common import check_fields, parse_expected_version
from .errors import invalid

FORMATS = ("jpg", "png", "webp")
DEFAULT_STEM = "edited-image"
DEFAULT_QUALITY = 90

_FIELDS = {"expected_version", "format", "filename", "quality"}
_FORBIDDEN = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]')
_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}


def validate_export_body(body):
    """Trả về (expected_version, định dạng, tên tệp không đuôi, quality hoặc None)."""
    check_fields(body, _FIELDS)
    version = parse_expected_version(body)

    fmt = body.get("format", "png")
    if not isinstance(fmt, str) or fmt not in FORMATS:
        raise invalid("format", "Định dạng phải là jpg, png hoặc webp.")

    if "filename" in body:
        raw = body["filename"]
        if not isinstance(raw, str):
            raise invalid("filename", "filename phải là chuỗi.")
        stem = raw.strip()
        if not 1 <= len(stem) <= 100:
            raise invalid("filename", "Tên tệp phải dài từ 1 đến 100 ký tự.")
        if _FORBIDDEN.search(stem):
            raise invalid("filename", 'Tên tệp không được chứa ký tự điều khiển hoặc < > : " / \\ | ? *.')
    else:
        stem = DEFAULT_STEM
    if stem.split(".")[0].strip().upper() in _RESERVED:   # tên dành riêng của Windows
        stem = DEFAULT_STEM

    if fmt == "png":
        if "quality" in body:
            raise invalid("quality", "PNG không dùng trường quality.")
        quality = None
    else:
        quality = body.get("quality", DEFAULT_QUALITY)
        if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
            raise invalid("quality", "Chất lượng ảnh phải là số nguyên từ 1 đến 100.")
    return version, fmt, stem, quality