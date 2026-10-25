import math
from functools import partial

from .common import check_fields, parse_expected_version
from .errors import ApiError, invalid

MAX_SIDE = 100_000   # chặn số quá lớn; giới hạn thật là tổng số pixel (kiểm tra theo ảnh hiện tại)


# ---------- Bộ kiểm tra từng tham số: trả về giá trị đã làm sạch hoặc ném ValueError ----------

def _int_in(lo, hi):
    def check(value):
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
            raise ValueError(f"phải là số nguyên từ {lo} đến {hi}")
        return value
    return check


def _number_in(lo, hi):
    def check(value):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not lo <= value <= hi):
            raise ValueError(f"phải là số từ {lo} đến {hi}")
        return float(value)
    return check


def _odd_in(lo, hi):
    def check(value):
        if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi or value % 2 == 0:
            raise ValueError(f"phải là số nguyên lẻ từ {lo} đến {hi}")
        return value
    return check


def _one_of(options):
    def check(value):
        if isinstance(value, bool) or type(value) not in (int, str) or value not in options:
            raise ValueError("phải là một trong: " + ", ".join(str(o) for o in options))
        return value
    return check


# Tên thao tác -> {tên tham số: bộ kiểm tra}. Mọi tham số đều bắt buộc.
SCHEMAS = {
    "grayscale": {},
    "brightness": {"value": _int_in(-100, 100)},
    "contrast": {"factor": _number_in(0.5, 3.0)},
    "blur": {"kernel_size": _odd_in(3, 31)},
    "sharpen": {"level": _int_in(1, 3)},
    "rotate": {"angle": _one_of((90, 180, 270))},   # theo chiều kim đồng hồ
    "flip": {"direction": _one_of(("horizontal", "vertical", "both"))},
    "resize": {"width": _int_in(1, MAX_SIDE), "height": _int_in(1, MAX_SIDE)},
    "crop": {"x": _int_in(0, MAX_SIDE), "y": _int_in(0, MAX_SIDE),
             "width": _int_in(1, MAX_SIDE), "height": _int_in(1, MAX_SIDE)},
}

OPERATION_MESSAGES = {
    "grayscale": "Đã chuyển ảnh sang ảnh xám.",
    "brightness": "Đã chỉnh độ sáng.",
    "contrast": "Đã chỉnh độ tương phản.",
    "blur": "Đã làm mờ ảnh.",
    "sharpen": "Đã làm nét ảnh.",
    "rotate": "Đã xoay ảnh.",
    "flip": "Đã lật ảnh.",
    "resize": "Đã đổi kích thước ảnh.",
    "crop": "Đã cắt ảnh.",
}


def validate_params(name, params):
    """Kiểm tra kiểu và phạm vi tham số; trả về dict đã làm sạch."""
    schema = SCHEMAS[name]
    for key in params:
        if key not in schema:
            raise invalid(f"parameters.{key}", f"Tham số {key!r} không được khai báo cho thao tác {name}.")
    clean = {}
    for key, check in schema.items():
        if key not in params:
            raise invalid(f"parameters.{key}", f"Thiếu tham số {key}.")
        try:
            clean[key] = check(params[key])
        except ValueError as e:
            raise invalid(f"parameters.{key}", f"{key} {e}.")
    return clean


# Giữ tên cũ: test kiểm tra khóa trùng với image_processing.OPERATIONS
PARAM_VALIDATORS = {name: partial(validate_params, name) for name in SCHEMAS}

_FIELDS = {"operation", "parameters", "expected_version"}


def validate_operation_body(body):
    """Trả về (tên thao tác, tham số, expected_version)."""
    check_fields(body, _FIELDS)
    for key in ("operation", "parameters", "expected_version"):
        if key not in body:
            raise invalid(key, f"Thiếu trường {key}.")

    name = body["operation"]
    if not isinstance(name, str):
        raise invalid("operation", "operation phải là chuỗi.")
    if name not in SCHEMAS:
        raise ApiError("UNSUPPORTED_OPERATION", "Thao tác chưa được hỗ trợ.", 400, "operation")

    params = body["parameters"]
    if not isinstance(params, dict):
        raise invalid("parameters", "parameters phải là một đối tượng JSON.")
    return name, PARAM_VALIDATORS[name](params), parse_expected_version(body)


def check_operation_context(name, params, current, max_pixels):
    """Kiểm tra các điều kiện phụ thuộc ảnh hiện tại (gọi khi đã giữ khóa và khớp version).

    Tọa độ crop là tọa độ ảnh tại đúng bước này trong chuỗi, không phải tọa độ màn hình.
    """
    if name == "crop":
        width, height = current["width"], current["height"]
        if params["x"] + params["width"] > width:
            raise invalid("parameters.width", f"Vùng cắt vượt ra ngoài ảnh (ảnh rộng {width}px).")
        if params["y"] + params["height"] > height:
            raise invalid("parameters.height", f"Vùng cắt vượt ra ngoài ảnh (ảnh cao {height}px).")
    elif name == "resize":
        if params["width"] * params["height"] > max_pixels:
            raise ApiError(
                "IMAGE_DIMENSIONS_EXCEEDED",
                f"Kích thước mới vượt quá {max_pixels // 1_000_000} triệu pixel.", 422, "parameters",
            )


def validate_version_body(body):
    """Thân request của undo, redo, reset: chỉ có expected_version."""
    check_fields(body, {"expected_version"})
    return parse_expected_version(body)