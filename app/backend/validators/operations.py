from .common import check_fields, parse_expected_version
from .errors import ApiError, invalid


def _no_params(params):
    if params:
        raise invalid("parameters", "Thao tác này không nhận tham số, parameters phải là {}.")
    return {}


# Tên thao tác -> hàm kiểm tra tham số, trả về tham số đã làm sạch.
# Khóa phải trùng với image_processing.OPERATIONS (có test kiểm tra).
PARAM_VALIDATORS = {
    "grayscale": _no_params,
}

OPERATION_MESSAGES = {
    "grayscale": "Đã chuyển ảnh sang ảnh xám.",
}

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
    if name not in PARAM_VALIDATORS:
        raise ApiError("UNSUPPORTED_OPERATION", "Thao tác chưa được hỗ trợ.", 400, "operation")

    params = body["parameters"]
    if not isinstance(params, dict):
        raise invalid("parameters", "parameters phải là một đối tượng JSON.")
    return name, PARAM_VALIDATORS[name](params), parse_expected_version(body)