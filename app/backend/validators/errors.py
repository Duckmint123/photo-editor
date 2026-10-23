class ApiError(Exception):
    """Lỗi API theo hợp đồng: có mã lỗi, trường sai (hoặc None) và mã HTTP."""

    def __init__(self, code, message, status=400, field=None, extra=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.field = field
        self.extra = extra or {}


def invalid(field, message):
    """Lỗi INVALID_PARAMETER: thiếu, sai kiểu, sai phạm vi hoặc trường không khai báo."""
    return ApiError("INVALID_PARAMETER", message, 400, field)