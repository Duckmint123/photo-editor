import cv2
import numpy as np
import pytest

from imageeditor.app.backend import create_app
from imageeditor.app.backend.image_processing import OPERATIONS
from imageeditor.app.backend.image_processing.upscale import upscale_size

from .conftest import decode, export, new_image, op

UP = OPERATIONS["upscale"]


def error_of(res):
    body = res.get_json()
    assert body["success"] is False
    return body["data"]["code"], body["data"]["field"]


def apply(client, image_id, version, **parameters):
    return op(client, image_id, version, operation="upscale", parameters=parameters)


def params(**overrides):
    base = {"mode": "percent", "value": 200, "amount": 1.5, "radius": 1.5, "denoise": 0}
    base.update(overrides)
    return base


def soft_photo(h=60, w=80, seed=0):
    """Ảnh mềm có nhiều chi tiết (nhiễu ngẫu nhiên đã làm mờ nhẹ rồi kéo giãn tương phản)."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (0, 0), 1.5)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def sharpness(img):
    return cv2.Laplacian(cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


# ======================= Sharpen cũ đã bị xóa =======================

def test_old_sharpen_operation_is_gone(client):
    assert "sharpen" not in OPERATIONS
    i = new_image(client)["image_id"]
    res = op(client, i, 0, operation="sharpen", parameters={"level": 2})
    assert res.status_code == 400 and error_of(res) == ("UNSUPPORTED_OPERATION", "operation")


# ======================= Kích thước đầu ra =======================

@pytest.mark.parametrize("mode,value,size,expected", [
    ("percent", 100, (40, 60), (60, 40)),       # size = (cao, rộng); kết quả (rộng, cao)
    ("percent", 200, (40, 60), (120, 80)),
    ("percent", 150, (41, 61), (92, 62)),       # làm tròn
    ("pixels", 120, (40, 60), (120, 80)),       # giữ tỷ lệ: 60x40 -> 120x80
    ("pixels", 60, (40, 60), (60, 40)),         # bằng chiều rộng hiện tại
    ("pixels", 100, (40, 60), (100, 67)),       # 40 * 100 / 60 = 66.67 -> 67
])
def test_upscale_size_and_output_shape(mode, value, size, expected):
    h, w = size
    assert upscale_size(w, h, mode, value) == expected
    out = UP(soft_photo(h, w), mode, value, 1.0, 1.0, 0)
    assert (out.shape[1], out.shape[0]) == expected and out.dtype == np.uint8 and out.shape[2] == 3


# ======================= Làm nét =======================

def test_amount_zero_equals_plain_lanczos_resize():
    img = soft_photo()
    out = UP(img, "percent", 200, 0, 1.5, 0)
    assert (out == cv2.resize(img, (160, 120), interpolation=cv2.INTER_LANCZOS4)).all()


def test_percent_100_without_sharpen_or_denoise_is_identity():
    img = soft_photo()
    assert (UP(img, "percent", 100, 0, 1.5, 0) == img).all()


def test_sharpness_grows_with_amount():
    img = soft_photo()
    values = [sharpness(UP(img, "percent", 200, a, 1.5, 0)) for a in (0, 1, 2, 4)]
    assert values == sorted(values) and values[-1] > 2 * values[0]


def test_radius_changes_the_result_and_percent_100_is_pure_sharpening():
    img = soft_photo()
    small, large = UP(img, "percent", 100, 2, 0.5, 0), UP(img, "percent", 100, 2, 3.0, 0)
    assert small.shape == img.shape and not (small == large).all()
    assert sharpness(small) > sharpness(img) and sharpness(large) > sharpness(img)


def test_flat_image_stays_flat_after_sharpening():
    flat = np.full((30, 40, 3), (30, 120, 220), np.uint8)
    out = UP(flat, "percent", 300, 5, 2.0, 0)
    assert out.shape == (90, 120, 3) and (out == flat[0, 0]).all()


# ======================= Giảm nhiễu =======================

def test_denoise_reduces_noise_before_upscaling():
    rng = np.random.default_rng(3)
    noisy = np.clip(np.full((60, 80, 3), 128, np.int16) + rng.normal(0, 20, (60, 80, 3)), 0, 255).astype(np.uint8)
    plain = UP(noisy, "percent", 200, 0, 1.5, 0)
    clean = UP(noisy, "percent", 200, 0, 1.5, 15)
    assert clean.shape == plain.shape and clean.std() < 0.5 * plain.std()


# ======================= Alpha =======================

def test_alpha_is_resized_not_sharpened_and_transparent_area_has_no_fringe():
    img = np.zeros((40, 40, 4), np.uint8)
    img[:, :, :3] = (255, 0, 0)          # nền ẩn màu xanh dương (BGR) trong vùng trong suốt
    img[:, 20:, :3] = (0, 0, 255)        # nửa phải màu đỏ
    img[:, :, 3] = 0
    img[:, 20:, 3] = 255                 # chỉ nửa phải đục
    out = UP(img, "percent", 200, 3, 1.5, 0)
    assert out.shape == (80, 80, 4)
    assert (out[:, :30, 3] == 0).all() and (out[:, 52:, 3] == 255).all()      # alpha giữ nguyên xa đường biên
    solid = out[:, 52:, :3]
    assert solid.reshape(-1, 3).std(axis=0).max() < 1                          # vùng đục không lem màu xanh từ vùng ẩn
    assert (solid[:, :, 2] == 255).all() and (solid[:, :, 0] == 0).all()


def test_opaque_rgba_stays_opaque():
    img = np.dstack([soft_photo(), np.full((60, 80), 255, np.uint8)])
    out = UP(img, "percent", 150, 1.5, 1.5, 0)
    assert out.shape == (90, 120, 4) and (out[:, :, 3] == 255).all()


# ======================= API =======================

def test_upscale_through_api_history_and_export(client):
    i = new_image(client, size=(40, 60))["image_id"]
    res = apply(client, i, 0, **params(value=200))
    assert res.status_code == 200
    d = res.get_json()["data"]
    assert d["steps"] == ["upscale"] and d["current"]["width"] == 120 and d["current"]["height"] == 80
    assert apply(client, i, 1, **params(mode="pixels", value=240)).get_json()["data"]["current"]["width"] == 240

    undo = client.post(f"/api/images/{i}/undo", json={"expected_version": 2}).get_json()["data"]
    assert undo["current"]["width"] == 120
    redo = client.post(f"/api/images/{i}/redo", json={"expected_version": undo["version"]}).get_json()["data"]
    assert redo["current"]["width"] == 240
    png = decode(export(client, i, redo["version"], format="png").data)
    assert png.shape[:2] == (160, 240)


def test_upscale_parameter_validation(client):
    i = new_image(client, size=(40, 60))["image_id"]
    bad = [
        (params(mode="width"), "parameters.mode"),
        (params(mode=1), "parameters.mode"),
        (params(value=99), "parameters.value"),
        (params(value=1001), "parameters.value"),
        (params(mode="pixels", value=100.5), "parameters.value"),
        (params(value="200"), "parameters.value"),
        (params(value=True), "parameters.value"),
        (params(amount=-0.1), "parameters.amount"),
        (params(amount=5.1), "parameters.amount"),
        (params(radius=0), "parameters.radius"),
        (params(radius=10.5), "parameters.radius"),
        (params(denoise=21), "parameters.denoise"),
        (params(denoise=-1), "parameters.denoise"),
        (params(denoise=2.5), "parameters.denoise"),
        ({k: v for k, v in params().items() if k != "radius"}, "parameters.radius"),
        ({**params(), "extra": 1}, "parameters.extra"),
        (params(mode="pixels", value=59), "parameters.value"),       # nhỏ hơn chiều rộng hiện tại: không phải phóng to
    ]
    for body, field in bad:
        res = apply(client, i, 0, **body)
        assert res.status_code == 400 and error_of(res) == ("INVALID_PARAMETER", field), (body, res.get_json())
    assert apply(client, i, 0, **params(mode="pixels", value=60)).status_code == 200      # bằng chiều rộng hiện tại: hợp lệ
    assert apply(client, new_image(client)["image_id"], 0, **params(value=100, amount=0)).status_code == 200


def test_upscale_respects_output_pixel_limit(tmp_path):
    app = create_app({"TESTING": True, "INSTANCE_DIR": str(tmp_path), "MAX_PIXELS": 10_000})
    c = app.test_client()
    i = new_image(c, size=(60, 80))["image_id"]                         # 4800 px
    res = apply(c, i, 0, **params(value=150))                           # 90 x 120 = 10800 px
    assert res.status_code == 422 and error_of(res)[0] == "IMAGE_DIMENSIONS_EXCEEDED"
    res = apply(c, i, 0, **params(mode="pixels", value=100))            # 100 x 75 = 7500 px
    assert res.status_code == 200


def test_denoise_is_limited_to_small_images(client, monkeypatch):
    from imageeditor.app.backend.validators import operations
    monkeypatch.setattr(operations, "DENOISE_MAX_PIXELS", 3000)
    i = new_image(client, size=(60, 80))["image_id"]                    # 4800 px > 3000
    res = apply(client, i, 0, **params(denoise=5))
    assert res.status_code == 400 and error_of(res) == ("INVALID_PARAMETER", "parameters.denoise")
    assert apply(client, i, 0, **params(denoise=0)).status_code == 200  # không giảm nhiễu thì vẫn chạy
    j = new_image(client, size=(30, 40))["image_id"]                    # 1200 px
    assert apply(client, j, 0, **params(denoise=5)).status_code == 200


def test_upscale_on_cropped_image_uses_current_size(client):
    i = new_image(client, size=(40, 60))["image_id"]
    r = op(client, i, 0, operation="crop", parameters={"x": 0, "y": 0, "width": 30, "height": 20})
    assert r.status_code == 200
    d = apply(client, i, 1, **params(mode="pixels", value=90)).get_json()["data"]
    assert (d["current"]["width"], d["current"]["height"]) == (90, 60)