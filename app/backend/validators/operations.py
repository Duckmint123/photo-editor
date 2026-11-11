import math
from functools import partial

from ..image_processing.geometry import RATIOS, percent_size, ratio_crop_box, rotated_size
from ..image_processing.upscale import DENOISE_MAX_PIXELS, upscale_size
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
    "rotate": {"angle": _one_of((90, 180, 270))},   # theo chiều kim đồng hồ
    "flip": {"direction": _one_of(("horizontal", "vertical", "both"))},
    "resize": {"width": _int_in(1, MAX_SIDE), "height": _int_in(1, MAX_SIDE)},
    "crop": {"x": _int_in(0, MAX_SIDE), "y": _int_in(0, MAX_SIDE),
             "width": _int_in(1, MAX_SIDE), "height": _int_in(1, MAX_SIDE)},
    "clahe": {"clip_limit": _number_in(1.0, 10.0), "tile_grid_size": _int_in(2, 16)},
    "invert": {},
    "sepia": {},
    "warm": {},
    "cool": {},
    "vintage": {},
    "canny": {"threshold1": _int_in(0, 255), "threshold2": _int_in(0, 255)},
    "threshold": {"value": _int_in(0, 255)},
    "otsu": {},
    "rotate_free": {"angle": _number_in(-360, 360)},   # độ, theo chiều kim đồng hồ; âm là ngược chiều
    "resize_percent": {"percent": _number_in(1, 1000)},
    "crop_ratio": {"ratio": _one_of(tuple(RATIOS))},
        # Tuần 7-8: phóng to + Unsharp Mask + giảm nhiễu tùy chọn
    "upscale": {
        "mode": _one_of(("percent", "pixels")),   # percent: value là %; pixels: value là chiều rộng mới (px)
        "value": _number_in(1, MAX_SIDE),
        "amount": _number_in(0, 5),               # độ mạnh Unsharp Mask, 0 = không làm nét
        "radius": _number_in(0.1, 10),            # sigma Gaussian (điểm ảnh của ảnh đầu ra)
        "denoise": _int_in(0, 20),                # độ mạnh giảm nhiễu, 0 = tắt
    },
}

OPERATION_MESSAGES = {
    "grayscale": "Đã chuyển ảnh sang ảnh xám.",
    "brightness": "Đã chỉnh độ sáng.",
    "contrast": "Đã chỉnh độ tương phản.",
    "blur": "Đã làm mờ ảnh.",
    "rotate": "Đã xoay ảnh.",
    "flip": "Đã lật ảnh.",
    "resize": "Đã đổi kích thước ảnh.",
    "crop": "Đã cắt ảnh.",
    "clahe": "Đã tăng cường độ tương phản cục bộ (CLAHE).",
    "invert": "Đã chuyển ảnh sang âm bản.",
    "sepia": "Đã áp dụng tông sepia.",
    "warm": "Đã áp dụng tông ấm.",
    "cool": "Đã áp dụng tông lạnh.",
    "vintage": "Đã áp dụng bộ lọc cổ điển.",
    "canny": "Đã dò biên Canny.",
    "threshold": "Đã áp dụng ngưỡng cố định.",
    "otsu": "Đã áp dụng ngưỡng Otsu.",
    "rotate_free": "Đã xoay ảnh theo góc tùy chỉnh.",
    "resize_percent": "Đã đổi kích thước theo phần trăm.",
    "crop_ratio": "Đã cắt ảnh theo tỷ lệ.",
    "upscale": "Đã phóng to và làm nét ảnh.",
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
    if name == "canny" and clean["threshold1"] > clean["threshold2"]:   # ngưỡng thấp không được lớn hơn ngưỡng cao
            raise invalid("parameters.threshold1", "threshold1 phải nhỏ hơn hoặc bằng threshold2.")
    if name == "upscale":
        if clean["mode"] == "percent":
            if not 100 <= clean["value"] <= 1000:
                    raise invalid("parameters.value", "value phải từ 100 đến 1000 khi mode là percent.")
        else:
            if clean["value"] != int(clean["value"]):
                raise invalid("parameters.value", "value phải là số nguyên (điểm ảnh) khi mode là pixels.")
            clean["value"] = int(clean["value"])
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
    elif name in ("resize", "resize_percent", "rotate_free"):
        if name == "resize":
                new_w, new_h = params["width"], params["height"]
        elif name == "resize_percent":
                new_w, new_h = percent_size(current["width"], current["height"], params["percent"])
        else:   # xoay tự do mở rộng khung, ảnh có thể lớn hơn ảnh hiện tại
                new_w, new_h = rotated_size(current["width"], current["height"], params["angle"])
        if new_w * new_h > max_pixels:
            raise ApiError(
                "IMAGE_DIMENSIONS_EXCEEDED",
                f"Kích thước mới vượt quá {max_pixels // 1_000_000} triệu pixel.", 422, "parameters",
            )
    elif name == "upscale":
        width, height = current["width"], current["height"]
        if params["mode"] == "pixels" and params["value"] < width:
            raise invalid("parameters.value",
                          f"Chiều rộng mới phải từ {width}px trở lên. Dùng Đổi kích thước để thu nhỏ ảnh.")
        new_w, new_h = upscale_size(width, height, params["mode"], params["value"])
        if new_w * new_h > max_pixels:
            raise ApiError(
                "IMAGE_DIMENSIONS_EXCEEDED",
                f"Kích thước mới vượt quá {max_pixels // 1_000_000} triệu pixel.", 422, "parameters",
            )
        if params["denoise"] > 0 and width * height > DENOISE_MAX_PIXELS:
            raise invalid("parameters.denoise",
                          f"Giảm nhiễu chỉ dùng cho ảnh đến {DENOISE_MAX_PIXELS // 1_000_000} triệu pixel "
                          f"(ảnh hiện tại {width}x{height}). Hãy cắt hoặc thu nhỏ ảnh trước, hoặc đặt denoise = 0.")
    elif name == "crop_ratio":
        if ratio_crop_box(current["width"], current["height"], params["ratio"]) is None:
            raise invalid("parameters.ratio", f"Ảnh quá nhỏ để cắt theo tỷ lệ {params['ratio']}.")


def validate_version_body(body):
    """Thân request của undo, redo, reset: chỉ có expected_version."""
    check_fields(body, {"expected_version"})
    return parse_expected_version(body)