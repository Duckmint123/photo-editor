from .errors import invalid


def check_fields(body, allowed):
    """Từ chối trường không khai báo trong hợp đồng."""
    for key in body:
        if key not in allowed:
            raise invalid(str(key), f"Trường {key!r} không được khai báo.")


def parse_expected_version(body):
    if "expected_version" not in body:
        raise invalid("expected_version", "Thiếu trường expected_version.")
    value = body["expected_version"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise invalid("expected_version", "expected_version phải là số nguyên không âm.")
    return value