import json
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from urllib.parse import quote

from flask import Blueprint, Response, current_app, jsonify, request, session
from werkzeug.exceptions import HTTPException

from ..services import pipeline
from ..services.store import NotFound
from ..validators import (
    OPERATION_MESSAGES, ApiError, check_operation_context, invalid,
    validate_export_body, validate_operation_body, validate_upload, validate_version_body,
)
from ..validators.export import DEFAULT_STEM

bp = Blueprint("api", __name__)

NOT_FOUND_MESSAGE = "Ảnh không khả dụng hoặc đã hết hạn. Hãy tải ảnh lên lại."

# Lỗi do Flask/Werkzeug chặn trước khi vào route cũng trả JSON theo mẫu chung
HTTP_ERRORS = {
    400: ("INVALID_PARAMETER", "Yêu cầu không hợp lệ."),
    404: ("NOT_FOUND", "Không tìm thấy đường dẫn."),
    405: ("METHOD_NOT_ALLOWED", "Phương thức không được hỗ trợ."),
    413: ("REQUEST_TOO_LARGE", "Tổng dung lượng request vượt quá giới hạn cho phép."),
    415: ("UNSUPPORTED_MEDIA_TYPE", "Content-Type không được hỗ trợ."),
    500: ("INTERNAL_ERROR", "Lỗi xử lý nội bộ. Vui lòng thử lại."),
}


# ---------- Tiện ích dùng chung ----------

def store():
    return current_app.extensions["image_store"]


def ok(data, message, status=200):
    return jsonify(success=True, message=message, data=data), status


def error_response(code, message, status, field=None, extra=None):
    data = {"code": code, "field": field}
    data.update(extra or {})
    return jsonify(success=False, message=message, data=data), status


def current_sid():
    """Mã phiên nằm trong cookie đã ký; ảnh không bao giờ nằm trong cookie."""
    sid = session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid
    return sid


def get_record(image_id):
    try:
        return store().get(image_id, current_sid())
    except NotFound:
        # Cùng một lỗi cho: mã sai, không tồn tại, hết hạn, thuộc phiên khác
        raise ApiError("IMAGE_NOT_FOUND", NOT_FOUND_MESSAGE, 404)


def json_body():
    if request.mimetype != "application/json":
        raise ApiError("UNSUPPORTED_MEDIA_TYPE", "Content-Type phải là application/json.", 415)
    try:
        body = json.loads(request.get_data(as_text=True))
    except ValueError:
        raise ApiError("INVALID_JSON", "JSON sai cú pháp.", 400)
    if not isinstance(body, dict):
        raise invalid(None, "Body phải là một đối tượng JSON.")
    return body


def preview_url(rec):
    return f"/api/images/{rec.id}/preview?source=current&version={rec.version}"


def state_data(rec):
    return {
        "image_id": rec.id,
        "version": rec.version,
        "history_length": rec.cursor,
        "can_undo": rec.cursor > 0,
        "can_redo": rec.cursor < len(rec.ops),
        "steps": [op["name"] for op in rec.ops[: rec.cursor]],
        "current": dict(rec.current),
        "preview_url": preview_url(rec),
    }


def version_conflict(rec, field):
    """Gọi khi đang giữ rec.lock để trạng thái trả về nhất quán."""
    extra = {k: v for k, v in state_data(rec).items() if k not in ("image_id", "version")}
    extra["current_version"] = rec.version
    return ApiError(
        "VERSION_CONFLICT",
        "Trạng thái ảnh đã thay đổi. Hãy kiểm tra ảnh hiện tại trước khi tiếp tục.",
        409, field, extra=extra,
    )


def check_version(rec, expected):
    if expected != rec.version:
        raise version_conflict(rec, "expected_version")


def commit(rec, ops, cursor):
    """Đặt trạng thái mới: ops[:cursor] là chuỗi đang áp dụng, phần sau cursor là nhánh Redo.

    Dựng ảnh và ghi tệp trước, cập nhật trạng thái sau cùng, nên lỗi ở bất kỳ bước nào
    cũng để lại trạng thái cũ nguyên vẹn. Gọi khi đang giữ rec.lock.
    """
    img = pipeline.render(store().original_path(rec), ops[:cursor])
    state = pipeline.describe_image(img)
    if cursor > 0:   # cursor == 0 thì ảnh hiện tại chính là ảnh gốc đã lưu sẵn
        store().write_current(rec, pipeline.encode_png(img))
    rec.ops, rec.cursor, rec.current, rec.version = ops, cursor, state, rec.version + 1


def read_file(path):
    try:
        return path.read_bytes()
    except FileNotFoundError:   # vừa bị dọn vì hết hạn
        raise ApiError("IMAGE_NOT_FOUND", NOT_FOUND_MESSAGE, 404)


def image_response(data, mimetype, version, extra_headers=None):
    headers = {"Cache-Control": "no-store", "X-Image-Version": str(version)}
    headers.update(extra_headers or {})
    return Response(data, mimetype=mimetype, headers=headers)


def content_disposition(stem, ext):
    """Tên ASCII dự phòng trong filename, tên Unicode đầy đủ trong filename*."""
    name = f"{stem}.{ext}"
    ascii_stem = unicodedata.normalize("NFKD", stem.replace("đ", "d").replace("Đ", "D"))
    ascii_stem = ascii_stem.encode("ascii", "ignore").decode().strip() or DEFAULT_STEM
    ascii_name = f"{ascii_stem}.{ext}"
    value = f'attachment; filename="{ascii_name}"'
    if ascii_name != name:
        value += f"; filename*=UTF-8''{quote(name, safe='')}"
    return value


def parse_version_arg(text):
    if not re.fullmatch(r"\d{1,9}", text):
        raise invalid("version", "version phải là số nguyên không âm.")
    return int(text)


# ---------- Xử lý lỗi: luôn trả JSON cho /api ----------

@bp.errorhandler(ApiError)
def on_api_error(e):
    return error_response(e.code, e.message, e.status, e.field, e.extra)


@bp.app_errorhandler(HTTPException)
def on_http_error(e):
    if not request.path.startswith("/api/"):
        return e
    code, message = HTTP_ERRORS.get(e.code, ("HTTP_ERROR", e.description))
    return error_response(code, message, e.code)


# ---------- Route ----------

@bp.post("/images")
def upload():
    sid = current_sid()
    if request.mimetype != "multipart/form-data":
        raise ApiError("UNSUPPORTED_MEDIA_TYPE", "Content-Type phải là multipart/form-data.", 415)
    files = request.files   # vượt MAX_CONTENT_LENGTH sẽ ném 413 ngay ở đây
    if "image" not in files:
        raise ApiError("MISSING_IMAGE", "Thiếu tệp ảnh ở trường image.", 400, "image")
    if sum(len(v) for v in files.listvalues()) > 1:
        raise invalid("image", "Chỉ được tải lên một tệp ảnh.")

    max_bytes = current_app.config["MAX_FILE_BYTES"]
    f = files["image"]
    data = f.read(max_bytes + 1)   # đọc dư 1 byte để biết có vượt giới hạn không
    info = validate_upload(f.filename, data, max_bytes, current_app.config["MAX_PIXELS"])

    img = info.pop("img")
    state = pipeline.describe_image(img)
    rec = store().add(sid, info, data, pipeline.encode_png(img), state)
    expires = datetime.fromtimestamp(rec.created + store().ttl, timezone.utc)
    return ok({
        **state_data(rec),
        "metadata": {
            "original_name": rec.original_name,
            "format": rec.fmt,
            "size_bytes": rec.size_bytes,
            **rec.metadata,
        },
        "original_preview_url": f"/api/images/{rec.id}/preview?source=original",
        "expires_at": expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, "Tải ảnh thành công.", 201)


@bp.post("/images/<image_id>/operations")
def apply_operation(image_id):
    rec = get_record(image_id)   # kiểm tra phiên và hạn trước khi lộ bất kỳ trạng thái nào
    name, params, expected = validate_operation_body(json_body())

    with rec.lock:   # kiểm tra version và ghi trạng thái là một khối nguyên tử theo từng ảnh
        check_version(rec, expected)
        if rec.cursor >= store().max_history:
            raise ApiError(
                "HISTORY_LIMIT_REACHED",
                f"Đã đạt giới hạn {store().max_history} bước. Hãy hoàn tác bớt hoặc tải ảnh lên lại.",
                409,
            )
        check_operation_context(name, params, rec.current, current_app.config["MAX_PIXELS"])
        ops = rec.ops[: rec.cursor] + [{"name": name, "params": params}]   # thao tác mới bỏ nhánh Redo
        commit(rec, ops, len(ops))
        return ok(state_data(rec), OPERATION_MESSAGES[name])


@bp.post("/images/<image_id>/undo")
def undo(image_id):
    rec = get_record(image_id)
    expected = validate_version_body(json_body())
    with rec.lock:
        check_version(rec, expected)
        if rec.cursor == 0:
            raise ApiError("NOTHING_TO_UNDO", "Không còn bước nào để hoàn tác.", 409)
        commit(rec, rec.ops, rec.cursor - 1)
        return ok(state_data(rec), "Đã hoàn tác.")


@bp.post("/images/<image_id>/redo")
def redo(image_id):
    rec = get_record(image_id)
    expected = validate_version_body(json_body())
    with rec.lock:
        check_version(rec, expected)
        if rec.cursor >= len(rec.ops):
            raise ApiError("NOTHING_TO_REDO", "Không còn bước nào để làm lại.", 409)
        commit(rec, rec.ops, rec.cursor + 1)
        return ok(state_data(rec), "Đã làm lại.")


@bp.post("/images/<image_id>/reset")
def reset(image_id):
    rec = get_record(image_id)
    expected = validate_version_body(json_body())
    with rec.lock:
        check_version(rec, expected)
        if not rec.ops:   # chưa có gì để đặt lại: giữ nguyên trạng thái và version
            return ok(state_data(rec), "Ảnh đang ở trạng thái gốc.")
        commit(rec, [], 0)
        return ok(state_data(rec), "Đã đặt lại về ảnh gốc.")


@bp.get("/images/<image_id>/preview")
def preview(image_id):
    rec = get_record(image_id)
    args = request.args
    for key in args:
        if key not in ("source", "version"):
            raise invalid(key, f"Tham số query {key!r} không hợp lệ.")
    source = args.get("source", "current")
    if source not in ("original", "current"):
        raise invalid("source", "source phải là original hoặc current.")

    if source == "original":
        if "version" in args:
            raise invalid("version", "Không gửi version khi source=original.")
        return image_response(read_file(store().source_png_path(rec)), "image/png", 0)

    if "version" not in args:
        raise invalid("version", "Thiếu version khi source=current.")
    version = parse_version_arg(args["version"])
    with rec.lock:   # bản chụp nhất quán: version và tệp ảnh thuộc cùng một trạng thái
        if version != rec.version:
            raise version_conflict(rec, "version")
        data = read_file(store().current_view_path(rec))
    return image_response(data, "image/png", version)


@bp.post("/images/<image_id>/export")
def export(image_id):
    rec = get_record(image_id)
    expected, fmt, stem, quality = validate_export_body(json_body())

    with rec.lock:   # chụp trạng thái của đúng version đã kiểm tra, mã hóa ngoài khóa
        check_version(rec, expected)
        png, version = read_file(store().current_view_path(rec)), rec.version

    data = pipeline.encode_export(pipeline.decode_png(png), fmt, quality)
    return image_response(
        data, pipeline.EXPORT_FORMATS[fmt], version,
        {"Content-Disposition": content_disposition(stem, fmt)},
    )