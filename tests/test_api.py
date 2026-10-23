import io
import re
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from imageeditor.app.backend import create_app
from imageeditor.app.backend.image_processing import OPERATIONS
from app.backend.validators.operations import PARAM_VALIDATORS

from .conftest import decode, export, make_image, new_image, op, upload

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def error_of(res):
    body = res.get_json()
    assert body["success"] is False and set(body["data"]) >= {"code", "field"}
    return body["data"]["code"], body["data"]["field"]


# ======================= Upload =======================

def test_upload_matches_contract(client):
    res = upload(client, make_image(size=(60, 80), alpha=255), "anh-mau.png")
    assert res.status_code == 201 and res.mimetype == "application/json"
    body = res.get_json()
    assert body["success"] is True and body["message"] == "Tải ảnh thành công."
    d = body["data"]
    assert UUID_RE.match(d["image_id"])
    assert (d["version"], d["history_length"]) == (0, 0)
    assert d["metadata"] == {"original_name": "anh-mau.png", "format": "png",
                             "size_bytes": len(make_image(size=(60, 80), alpha=255)),
                             "width": 80, "height": 60, "has_alpha": True}
    assert d["current"] == {"width": 80, "height": 60, "color_mode": "color", "has_alpha": True}
    assert d["original_preview_url"] == f"/api/images/{d['image_id']}/preview?source=original"
    assert d["preview_url"] == f"/api/images/{d['image_id']}/preview?source=current&version=0"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", d["expires_at"])


def test_upload_decides_format_by_content(client):
    assert upload(client, make_image(".jpg"), "a.jpg").get_json()["data"]["metadata"]["format"] == "jpg"
    assert upload(client, make_image(".webp"), "a.webp").get_json()["data"]["metadata"]["format"] == "webp"
    res = upload(client, make_image(".png"), "ten-sai-duoi.jpg")   # nội dung là PNG
    assert res.status_code == 201 and res.get_json()["data"]["metadata"]["format"] == "png"


def test_upload_missing_empty_and_multiple(client):
    assert error_of(upload(client, make_image(), field="file")) == ("MISSING_IMAGE", "image")
    assert error_of(client.post("/api/images", data={}, content_type="multipart/form-data")) == ("MISSING_IMAGE", "image")
    res = upload(client, b"", "a.png")
    assert res.status_code == 400 and error_of(res) == ("EMPTY_IMAGE", "image")
    res = client.post("/api/images", content_type="multipart/form-data",
                      data={"image": [(io.BytesIO(make_image()), "a.png"), (io.BytesIO(make_image()), "b.png")]})
    assert res.status_code == 400 and error_of(res) == ("INVALID_PARAMETER", "image")


def test_upload_rejects_wrong_content_type(client):
    res = client.post("/api/images", json={"image": "x"})
    assert res.status_code == 415 and error_of(res)[0] == "UNSUPPORTED_MEDIA_TYPE"


def test_upload_rejects_bad_images(client):
    res = upload(client, b"khong phai anh", "a.png")
    assert res.status_code == 422 and error_of(res)[0] == "INVALID_IMAGE"
    res = upload(client, make_image(size=(200, 200))[:-30], "a.png")   # PNG bị cắt cụt
    assert res.status_code == 422 and error_of(res)[0] == "INVALID_IMAGE"

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buf, "GIF")
    res = upload(client, buf.getvalue(), "a.gif")                      # GIF không hỗ trợ
    assert res.status_code == 415 and error_of(res)[0] == "UNSUPPORTED_MEDIA_TYPE"

    frames = [Image.new("RGB", (8, 8), c) for c in ("red", "blue")]
    buf = io.BytesIO()
    frames[0].save(buf, "PNG", save_all=True, append_images=frames[1:])   # APNG: ảnh động
    res = upload(client, buf.getvalue(), "a.png")
    assert res.status_code == 422 and error_of(res)[0] == "INVALID_IMAGE"


def test_upload_size_limits(tmp_path):
    def app_with(**cfg):
        return create_app({"TESTING": True, "INSTANCE_DIR": str(tmp_path / str(len(cfg))), **cfg}).test_client()

    big = make_image(size=(200, 200), color=(1, 2, 3))
    res = upload(app_with(MAX_FILE_BYTES=100), big)                          # vượt giới hạn tệp
    assert res.status_code == 413 and error_of(res) == ("FILE_TOO_LARGE", "image")
    res = upload(app_with(MAX_CONTENT_LENGTH=100, MAX_FILE_BYTES=100_000), big)   # vượt giới hạn request
    assert res.status_code == 413 and error_of(res)[0] == "REQUEST_TOO_LARGE"
    res = upload(app_with(MAX_PIXELS=1000, MAX_FILE_BYTES=100_000, X=1), big)      # vượt số pixel
    assert res.status_code == 422 and error_of(res) == ("IMAGE_DIMENSIONS_EXCEEDED", "image")


def test_exif_orientation_matches_pillow(client):
    base = np.zeros((20, 30, 3), np.uint8)       # ảnh không đối xứng để lộ lỗi hướng
    base[:, :, 2] = np.arange(30, dtype=np.uint8) * 8
    base[:, :, 1] = (np.arange(20, dtype=np.uint8) * 12)[:, None]
    for orientation in range(1, 9):
        im = Image.fromarray(base[:, :, ::-1])
        exif = Image.Exif()
        exif[0x0112] = orientation
        buf = io.BytesIO()
        im.save(buf, "PNG", exif=exif)
        data = buf.getvalue()

        expected = np.array(ImageOps.exif_transpose(Image.open(io.BytesIO(data))))[:, :, ::-1]
        d = upload(client, data).get_json()["data"]
        got = decode(client.get(d["original_preview_url"]).data)
        assert np.array_equal(got, expected), f"orientation {orientation}"
        assert (d["metadata"]["width"], d["metadata"]["height"]) == (expected.shape[1], expected.shape[0])


# ======================= Grayscale =======================

def test_grayscale_matches_contract(client):
    img = new_image(client)
    res = op(client, img["image_id"], 0)
    assert res.status_code == 200
    body = res.get_json()
    assert body["message"] == "Đã chuyển ảnh sang ảnh xám."
    d = body["data"]
    assert d["image_id"] == img["image_id"] and (d["version"], d["history_length"]) == (1, 1)
    assert d["current"] == {"width": 60, "height": 40, "color_mode": "grayscale", "has_alpha": False}
    assert d["preview_url"] == f"/api/images/{img['image_id']}/preview?source=current&version=1"

    cur = client.get(d["preview_url"])
    assert cur.status_code == 200 and cur.mimetype == "image/png"
    assert cur.headers["X-Image-Version"] == "1" and cur.headers["Cache-Control"] == "no-store"
    out = decode(cur.data)
    assert np.array_equal(out[:, :, 0], out[:, :, 1]) and np.array_equal(out[:, :, 1], out[:, :, 2])

    orig = client.get(img["original_preview_url"])                  # ảnh gốc không bị ghi đè
    assert orig.headers["X-Image-Version"] == "0"
    assert not np.array_equal(decode(orig.data)[:, :, 0], decode(orig.data)[:, :, 2])


def test_grayscale_keeps_size_and_alpha_and_counts_repeats(client):
    img = new_image(client, alpha=128)
    d = op(client, img["image_id"], 0).get_json()["data"]
    assert d["current"] == {"width": 60, "height": 40, "color_mode": "grayscale", "has_alpha": True}
    out = decode(client.get(d["preview_url"]).data)
    assert out.shape == (40, 60, 4) and (out[:, :, 3] == 128).all()
    d2 = op(client, img["image_id"], 1).get_json()["data"]          # áp dụng lại vẫn ghi một bước
    assert (d2["version"], d2["history_length"]) == (2, 2)


def test_operation_validation(client):
    i = new_image(client)["image_id"]
    url = f"/api/images/{i}/operations"
    assert error_of(op(client, i, 0, operation="blur")) == ("UNSUPPORTED_OPERATION", "operation")
    assert error_of(op(client, i, 0, operation=5)) == ("INVALID_PARAMETER", "operation")
    assert error_of(op(client, i, 0, parameters={"x": 1})) == ("INVALID_PARAMETER", "parameters")
    assert error_of(op(client, i, 0, parameters=[])) == ("INVALID_PARAMETER", "parameters")
    assert error_of(op(client, i, 0, extra=1)) == ("INVALID_PARAMETER", "extra")
    for bad in (-1, "0", True, 1.5, None):
        res = op(client, i, bad)
        assert res.status_code == 400 and error_of(res) == ("INVALID_PARAMETER", "expected_version"), bad
    for missing in ("operation", "parameters", "expected_version"):
        body = {"operation": "grayscale", "parameters": {}, "expected_version": 0}
        del body[missing]
        assert error_of(client.post(url, json=body)) == ("INVALID_PARAMETER", missing)
    assert error_of(client.post(url, data="{sai", content_type="application/json")) == ("INVALID_JSON", None)
    assert error_of(client.post(url, json=[1])) == ("INVALID_PARAMETER", None)
    res = client.post(url, data="{}", content_type="text/plain")
    assert res.status_code == 415 and error_of(res)[0] == "UNSUPPORTED_MEDIA_TYPE"
    assert op(client, i, 0).status_code == 200      # các request lỗi không đổi trạng thái


def test_version_conflict_shape_and_state_unchanged(client):
    i = new_image(client)["image_id"]
    assert op(client, i, 0).status_code == 200
    res = op(client, i, 0)                                          # version đã là 1
    assert res.status_code == 409
    d = res.get_json()["data"]
    assert (d["code"], d["field"], d["current_version"], d["history_length"]) == ("VERSION_CONFLICT", "expected_version", 1, 1)
    assert d["current"]["color_mode"] == "grayscale"
    assert d["preview_url"] == f"/api/images/{i}/preview?source=current&version=1"
    assert op(client, i, 1).get_json()["data"]["version"] == 2     # vẫn áp dụng tiếp được


def test_concurrent_requests_same_version_write_one_step(app):
    c1 = app.test_client()
    i = new_image(c1)["image_id"]
    with c1.session_transaction() as s:
        sid = s["sid"]
    results, barrier = [], threading.Barrier(2)

    def worker():
        c = app.test_client()
        with c.session_transaction() as s:
            s["sid"] = sid
        barrier.wait()
        results.append(op(c, i, 0).status_code)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results) == [200, 409]
    d = op(c1, i, 1).get_json()["data"]
    assert (d["version"], d["history_length"]) == (2, 2)            # chỉ có đúng một bước từ hai request


def test_history_limit(tmp_path):
    app = create_app({"TESTING": True, "INSTANCE_DIR": str(tmp_path), "MAX_HISTORY": 3})
    c = app.test_client()
    i = new_image(c)["image_id"]
    codes = [op(c, i, v).status_code for v in range(4)]
    assert codes == [200, 200, 200, 409]
    res = op(c, i, 3)
    assert error_of(res) == ("HISTORY_LIMIT_REACHED", None)
    assert export(c, i, 3, format="png").headers["X-Image-Version"] == "3"   # trạng thái không đổi


def test_operation_registries_match():
    assert set(OPERATIONS) == set(PARAM_VALIDATORS)


# ======================= Preview =======================

def test_preview_rules(client):
    i = new_image(client)["image_id"]
    base = f"/api/images/{i}/preview"
    assert client.get(base + "?source=current&version=0").status_code == 200
    assert client.get(base + "?version=0").status_code == 200       # source mặc định là current
    assert error_of(client.get(base + "?source=current")) == ("INVALID_PARAMETER", "version")
    assert error_of(client.get(base)) == ("INVALID_PARAMETER", "version")
    assert error_of(client.get(base + "?source=original&version=0")) == ("INVALID_PARAMETER", "version")
    assert error_of(client.get(base + "?source=xyz")) == ("INVALID_PARAMETER", "source")
    assert error_of(client.get(base + "?source=current&version=abc")) == ("INVALID_PARAMETER", "version")
    assert error_of(client.get(base + "?source=current&version=-1")) == ("INVALID_PARAMETER", "version")
    assert error_of(client.get(base + "?source=current&version=0&x=1")) == ("INVALID_PARAMETER", "x")


def test_preview_stale_version_returns_conflict(client):
    i = new_image(client)["image_id"]
    op(client, i, 0)
    res = client.get(f"/api/images/{i}/preview?source=current&version=0")   # URL cũ
    assert res.status_code == 409 and res.mimetype == "application/json"
    d = res.get_json()["data"]
    assert (d["code"], d["field"], d["current_version"]) == ("VERSION_CONFLICT", "version", 1)
    assert client.get(f"/api/images/{i}/preview?source=current&version=2").status_code == 409


# ======================= Export =======================

def test_export_formats_headers_and_no_state_change(client):
    i = new_image(client)["image_id"]
    op(client, i, 0)
    for fmt, mime in [("png", "image/png"), ("jpg", "image/jpeg"), ("webp", "image/webp")]:
        res = export(client, i, 1, format=fmt, filename="anh-xam")
        assert res.status_code == 200 and res.mimetype == mime
        assert res.headers["Content-Disposition"] == f'attachment; filename="anh-xam.{fmt}"'
        assert res.headers["Cache-Control"] == "no-store" and res.headers["X-Image-Version"] == "1"
        out = decode(res.data)
        assert out is not None and out.shape[:2] == (40, 60)
    assert op(client, i, 1).get_json()["data"]["history_length"] == 2   # export không ghi bước nào


def test_export_defaults_and_unicode_filename(client):
    i = new_image(client)["image_id"]
    res = export(client, i, 0)                                       # mặc định: png, edited-image
    assert res.mimetype == "image/png"
    assert res.headers["Content-Disposition"] == 'attachment; filename="edited-image.png"'
    res = export(client, i, 0, format="jpg", filename="  ảnh đẹp của Đạt  ")
    cd = res.headers["Content-Disposition"]
    assert 'filename="anh dep cua Dat.jpg"' in cd
    assert "filename*=UTF-8''%E1%BA%A3nh%20%C4%91%E1%BA%B9p%20c%E1%BB%A7a%20%C4%90%E1%BA%A1t.jpg" in cd
    res = export(client, i, 0, filename="日本語")
    assert 'filename="edited-image.png"' in res.headers["Content-Disposition"]   # ASCII dự phòng
    for reserved in ("CON", "nul", "com1.backup"):
        res = export(client, i, 0, filename=reserved)
        assert 'filename="edited-image.png"' in res.headers["Content-Disposition"], reserved


def test_export_validation(client):
    i = new_image(client)["image_id"]
    res = export(client, i, 0, format="jpg", quality=120)
    assert res.status_code == 400 and res.mimetype == "application/json"
    assert res.get_json()["message"] == "Chất lượng ảnh phải là số nguyên từ 1 đến 100."
    assert error_of(res) == ("INVALID_PARAMETER", "quality")
    assert error_of(export(client, i, 0, format="png", quality=90)) == ("INVALID_PARAMETER", "quality")
    assert error_of(export(client, i, 0, format="webp", quality=True)) == ("INVALID_PARAMETER", "quality")
    assert error_of(export(client, i, 0, format="gif")) == ("INVALID_PARAMETER", "format")
    assert error_of(export(client, i, 0, format="jpeg")) == ("INVALID_PARAMETER", "format")
    for bad in ("", "   ", "a" * 101, "a/b", "a:b", 'a"b', "a\x01b", "a?b", 5):
        assert error_of(export(client, i, 0, filename=bad)) == ("INVALID_PARAMETER", "filename"), repr(bad)
    assert error_of(export(client, i, 0, extra=1)) == ("INVALID_PARAMETER", "extra")
    assert error_of(client.post(f"/api/images/{i}/export", json={"format": "png"})) == ("INVALID_PARAMETER", "expected_version")
    assert export(client, i, 0, format="jpg", quality=1).status_code == 200
    assert export(client, i, 0, format="webp", quality=100).status_code == 200


def test_export_version_conflict(client):
    i = new_image(client)["image_id"]
    op(client, i, 0)
    res = export(client, i, 0)
    assert res.status_code == 409 and error_of(res) == ("VERSION_CONFLICT", "expected_version")
    assert res.get_json()["data"]["current_version"] == 1


def test_export_jpg_flattens_transparency_on_white(client):
    i = new_image(client, alpha=0)["image_id"]                       # hoàn toàn trong suốt
    out = decode(export(client, i, 0, format="jpg").data)
    assert out.shape[2] == 3 and out.min() >= 250


def test_export_keeps_alpha_for_png_and_webp(client):
    i = new_image(client, alpha=100)["image_id"]
    png = decode(export(client, i, 0, format="png").data)
    assert png.shape[2] == 4 and (png[:, :, 3] == 100).all()
    webp = decode(export(client, i, 0, format="webp", quality=90).data)
    assert webp.shape[2] == 4


# ======================= Phiên, hết hạn, lỗi chung =======================

def test_other_session_cannot_access(app):
    owner, stranger = app.test_client(), app.test_client()
    img = new_image(owner)
    i = img["image_id"]
    attempts = [
        stranger.get(img["original_preview_url"]),
        stranger.get(img["preview_url"]),
        op(stranger, i, 0),
        export(stranger, i, 0),
    ]
    for res in attempts:
        assert res.status_code == 404 and error_of(res) == ("IMAGE_NOT_FOUND", None)
        assert res.get_json()["message"] == "Ảnh không khả dụng hoặc đã hết hạn. Hãy tải ảnh lên lại."
    assert owner.get(img["preview_url"]).status_code == 200


def test_unknown_image_id_is_not_found(client):
    new_image(client)
    for bad in ("khong-co", "x" * 300, "57e82aec-bba1-48c9-b917-245a5a745893"):
        res = client.get(f"/api/images/{bad}/preview?source=original")
        assert res.status_code == 404 and error_of(res)[0] == "IMAGE_NOT_FOUND"
    assert error_of(op(client, "khong-co", 0))[0] == "IMAGE_NOT_FOUND"   # 404 trước khi kiểm tra body


def test_expiry_is_fixed_from_upload_and_files_are_removed(app, tmp_path):
    store = app.extensions["image_store"]
    now = [1000.0]
    store.clock = lambda: now[0]
    c = app.test_client()
    img = new_image(c)
    i = img["image_id"]
    assert (tmp_path / "uploads" / i).exists() and (tmp_path / "outputs" / i).exists()

    now[0] = 1000 + 3599                                             # còn hạn, và truy cập không gia hạn
    assert c.get(img["original_preview_url"]).status_code == 200
    now[0] = 1000 + 3601
    res = c.get(img["original_preview_url"])
    assert res.status_code == 404 and error_of(res)[0] == "IMAGE_NOT_FOUND"
    assert not (tmp_path / "uploads" / i).exists() and not (tmp_path / "outputs" / i).exists()


def test_cleanup_removes_expired_and_orphans(app, tmp_path):
    store = app.extensions["image_store"]
    now = [5000.0]
    store.clock = lambda: now[0]
    c = app.test_client()
    i = new_image(c)["image_id"]
    orphan = tmp_path / "uploads" / "mo-coi"
    orphan.mkdir()
    import os
    os.utime(orphan, (1000, 1000))
    now[0] += 4000
    assert store.cleanup() == 1
    assert not orphan.exists() and not (tmp_path / "outputs" / i).exists()


def test_all_api_errors_are_json(client):
    res = client.get("/api/khong-co")
    assert res.status_code == 404 and error_of(res) == ("NOT_FOUND", None)
    res = client.get("/api/images")                                  # sai method
    assert res.status_code == 405 and error_of(res)[0] == "METHOD_NOT_ALLOWED"


def test_index_page(client):
    assert client.get("/").status_code == 200