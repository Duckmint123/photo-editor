import cv2
import numpy as np
import pytest

from imageeditor.app.backend.image_processing import OPERATIONS

from .conftest import decode, export, make_image, new_image, op


def error_of(res):
    body = res.get_json()
    assert body["success"] is False
    return body["data"]["code"], body["data"]["field"]


def color_img(h=40, w=60, color=(30, 120, 220), alpha=None):
    img = np.full((h, w, 3), color, np.uint8)
    if alpha is not None:
        img = np.dstack([img, np.full((h, w), alpha, np.uint8)])
    return img


def gradient(h=64, w=64):
    """Ảnh xám dốc từ tối sang sáng, 3 kênh."""
    row = np.linspace(0, 255, w).astype(np.uint8)
    return cv2.cvtColor(np.tile(row, (h, 1)), cv2.COLOR_GRAY2BGR)


# ======================= CLAHE =======================

def test_clahe_raises_local_contrast_keeps_chroma_and_alpha():
    rng = np.random.default_rng(0)
    base = rng.integers(100, 121, (64, 64), dtype=np.uint8)           # ảnh tương phản thấp: 100..120
    img = np.dstack([base + 5, base, base - 5]).astype(np.uint8)       # B, G, R hơi lệch nhau
    out = OPERATIONS["clahe"](img, clip_limit=4.0, tile_grid_size=8)
    assert out.shape == img.shape and out.dtype == np.uint8
    lab_in = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_out = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    assert lab_out[:, :, 0].std() > 2 * lab_in[:, :, 0].std()          # kênh sáng được kéo giãn
    assert np.abs(lab_out[:, :, 1].astype(int) - lab_in[:, :, 1]).max() <= 3   # a, b gần như giữ nguyên
    assert np.abs(lab_out[:, :, 2].astype(int) - lab_in[:, :, 2]).max() <= 3

    rgba = np.dstack([img, np.full(img.shape[:2], 77, np.uint8)])
    out = OPERATIONS["clahe"](rgba, clip_limit=2.0, tile_grid_size=4)
    assert out.shape[2] == 4 and (out[:, :, 3] == 77).all()


def test_clahe_tiny_image_does_not_crash():
    out = OPERATIONS["clahe"](color_img(3, 3), clip_limit=10.0, tile_grid_size=16)
    assert out.shape == (3, 3, 3)


# ======================= Bộ lọc màu =======================

def test_invert_is_involution_and_keeps_alpha():
    img = np.dstack([gradient(), np.full((64, 64), 200, np.uint8)])
    once = OPERATIONS["invert"](img)
    assert (once[:, :, :3] == 255 - img[:, :, :3]).all() and (once[:, :, 3] == 200).all()
    assert (OPERATIONS["invert"](once) == img).all()


def test_sepia_matches_standard_matrix():
    img = color_img(2, 2, (10, 100, 200))    # B=10 G=100 R=200
    out = OPERATIONS["sepia"](img)[0, 0]
    b = 0.131 * 10 + 0.534 * 100 + 0.272 * 200
    g = 0.168 * 10 + 0.686 * 100 + 0.349 * 200
    r = 0.189 * 10 + 0.769 * 100 + 0.393 * 200
    assert np.allclose(out, [round(b), round(g), min(255, round(r))], atol=1)


def test_warm_and_cool_shift_red_blue():
    img = color_img(4, 4, (100, 100, 100))
    warm, cool = OPERATIONS["warm"](img)[0, 0], OPERATIONS["cool"](img)[0, 0]
    assert warm[2] > 100 > warm[0] and warm[1] == 100       # B, G, R
    assert cool[0] > 100 > cool[2] and cool[1] == 100
    white = color_img(4, 4, (250, 250, 250))
    assert OPERATIONS["warm"](white)[0, 0][2] == 255        # kẹp trong 0..255, không tràn số


def test_vintage_fades_blacks_and_whites():
    black, white = color_img(4, 4, (0, 0, 0)), color_img(4, 4, (255, 255, 255))
    assert OPERATIONS["vintage"](black).min() >= 25
    assert OPERATIONS["vintage"](white).max() < 255


@pytest.mark.parametrize("name", ["invert", "sepia", "warm", "cool", "vintage"])
def test_color_filters_keep_shape_dtype_and_alpha(name):
    rgba = color_img(10, 12, alpha=90)
    out = OPERATIONS[name](rgba)
    assert out.shape == rgba.shape and out.dtype == np.uint8 and (out[:, :, 3] == 90).all()
    assert OPERATIONS[name](color_img(10, 12)).shape == (10, 12, 3)


# ======================= Canny, Threshold, Otsu =======================

def square_image():
    img = np.zeros((64, 64, 3), np.uint8)
    img[16:48, 16:48] = 255
    return img


def test_canny_finds_edges_of_square_only():
    out = OPERATIONS["canny"](square_image(), threshold1=50, threshold2=150)
    assert out.shape == (64, 64, 3) and set(np.unique(out)) <= {0, 255}
    assert (out[:, :, 0] == out[:, :, 1]).all()
    assert out[16, 30:34].max() == 255 or out[15, 30:34].max() == 255   # biên trên của hình vuông
    assert out[30:34, 30:34].max() == 0                                    # trong lòng hình vuông không có biên
    assert out[:8].max() == 0                                              # nền phẳng không có biên
    flat = OPERATIONS["canny"](color_img(32, 32), 50, 150)
    assert flat.max() == 0


def test_threshold_fixed_boundary_is_strictly_greater():
    img = gradient()
    out = OPERATIONS["threshold"](img, value=127)
    gray = img[:, :, 0]
    assert ((out[:, :, 0] == 255) == (gray > 127)).all()
    assert set(np.unique(out)) <= {0, 255}
    assert OPERATIONS["threshold"](img, 255).max() == 0       # không điểm nào > 255
    assert OPERATIONS["threshold"](img, 0)[:, 5:].min() == 255


def test_otsu_splits_two_clusters():
    img = np.full((32, 64, 3), 40, np.uint8)
    img[:, 32:] = 200
    out = OPERATIONS["otsu"](img)
    assert (out[:, :32] == 0).all() and (out[:, 32:] == 255).all()


@pytest.mark.parametrize("name,args", [("canny", (50, 150)), ("threshold", (100,)), ("otsu", ())])
def test_analysis_keeps_alpha(name, args):
    rgba = np.dstack([gradient(), np.full((64, 64), 33, np.uint8)])
    out = OPERATIONS[name](rgba, *args)
    assert out.shape == rgba.shape and (out[:, :, 3] == 33).all()


# ======================= Xoay tự do =======================

def test_rotate_free_multiples_of_90_are_exact_and_lossless():
    img = np.random.default_rng(1).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    cw = OPERATIONS["rotate"](img, 90)
    assert (OPERATIONS["rotate_free"](img, 90) == cw).all()
    assert (OPERATIONS["rotate_free"](img, -90) == OPERATIONS["rotate"](img, 270)).all()
    assert (OPERATIONS["rotate_free"](img, 180) == OPERATIONS["rotate"](img, 180)).all()
    assert (OPERATIONS["rotate_free"](img, 360) == img).all() and (OPERATIONS["rotate_free"](img, 0) == img).all()
    assert OPERATIONS["rotate_free"](img, 90).shape == (30, 20, 3)     # không thêm alpha


def test_rotate_free_45_expands_canvas_with_transparent_corners():
    img = color_img(100, 100, (10, 20, 30))
    out = OPERATIONS["rotate_free"](img, 45)
    assert out.shape == (142, 142, 4)                                   # ceil(100 * sqrt 2)
    assert out[0, 0, 3] == 0 and out[-1, -1, 3] == 0                    # góc trống trong suốt
    assert out[71, 71, 3] == 255 and tuple(out[71, 71, :3]) == (10, 20, 30)


def test_rotate_free_direction_is_clockwise():
    img = np.zeros((101, 101, 3), np.uint8)
    img[0:10, 45:56] = 255          # mảng sáng ở giữa cạnh trên
    out = OPERATIONS["rotate_free"](img, 30)
    ys, xs = np.nonzero((out[:, :, 3] > 0) & (out[:, :, 0] > 128))
    cx = out.shape[1] / 2
    assert xs.mean() > cx           # xoay thuận chiều kim đồng hồ: cạnh trên đi sang phải


def test_rotate_free_keeps_existing_transparency():
    img = color_img(40, 40, alpha=0)
    img[10:30, 10:30, 3] = 255
    out = OPERATIONS["rotate_free"](img, 20)
    assert out.shape[2] == 4 and out[0, 0, 3] == 0 and out[:, :, 3].max() == 255


def test_rotated_size_has_no_float_noise():
    from imageeditor.app.backend.image_processing.geometry import rotated_size
    assert rotated_size(30, 20, 90) == (20, 30)
    assert rotated_size(30, 20, 0) == (30, 20)
    assert rotated_size(30, 20, 180) == (30, 20)
    assert rotated_size(1, 1, 33) == (2, 2)


# ======================= Resize % và crop tỷ lệ =======================

@pytest.mark.parametrize("percent,expected", [(100, (40, 60)), (50, (20, 30)), (200, (80, 120)), (33.3, (13, 20)), (1, (1, 1))])
def test_resize_percent_sizes(percent, expected):
    out = OPERATIONS["resize_percent"](color_img(40, 60), percent)
    assert out.shape[:2] == expected


def test_resize_percent_keeps_alpha_channel():
    out = OPERATIONS["resize_percent"](color_img(40, 60, alpha=255), 150)
    assert out.shape == (60, 90, 4)


@pytest.mark.parametrize("ratio,rw,rh", [("1:1", 1, 1), ("4:3", 4, 3), ("16:9", 16, 9)])
@pytest.mark.parametrize("size", [(100, 200), (200, 100), (77, 123), (9, 16), (1000, 1001)])
def test_crop_ratio_exact_ratio_inside_image_and_centered(ratio, rw, rh, size):
    h, w = size
    img = np.random.default_rng(2).integers(0, 256, (h, w, 3), dtype=np.uint8)
    if min(w // rw, h // rh) < 1:
        return
    out = OPERATIONS["crop_ratio"](img, ratio)
    oh, ow = out.shape[:2]
    assert ow * rh == oh * rw and ow <= w and oh <= h
    assert (w - ow) // 2 >= 0 and abs((w - ow) - 2 * ((w - ow) // 2)) <= 1
    # lớn nhất: cạnh còn lại không thể tăng thêm một bậc tỷ lệ
    assert (ow + rw > w) or (oh + rh > h)
    x, y = (w - ow) // 2, (h - oh) // 2
    assert (out == img[y:y + oh, x:x + ow]).all()


def test_crop_ratio_too_small_raises():
    with pytest.raises(ValueError):
        OPERATIONS["crop_ratio"](color_img(8, 15), "16:9")


# ======================= API: tham số và ngữ cảnh =======================

def apply(client, image_id, version, operation, **parameters):
    return op(client, image_id, version, operation=operation, parameters=parameters)


def test_new_operations_apply_through_api_and_history(client):
    i = new_image(client, size=(60, 80))["image_id"]
    steps = [
        ("clahe", {"clip_limit": 2.0, "tile_grid_size": 8}),
        ("sepia", {}), ("warm", {}), ("cool", {}), ("vintage", {}), ("invert", {}),
        ("canny", {"threshold1": 50, "threshold2": 150}),
        ("threshold", {"value": 128}), ("otsu", {}),
        ("rotate_free", {"angle": 30}), ("resize_percent", {"percent": 50}), ("crop_ratio", {"ratio": "1:1"}),
    ]
    version = 0
    for name, params in steps:
        res = apply(client, i, version, name, **params)
        assert res.status_code == 200, (name, res.get_json())
        version = res.get_json()["data"]["version"]
    d = res.get_json()["data"]
    assert d["steps"] == [n for n, _ in steps] and d["current"]["width"] == d["current"]["height"]
    # Undo hết rồi Redo lại cho cùng kích thước: chuỗi dựng lại đúng
    for _ in steps:
        version = client.post(f"/api/images/{i}/undo", json={"expected_version": version}).get_json()["data"]["version"]
    for _ in steps:
        res = client.post(f"/api/images/{i}/redo", json={"expected_version": version})
        version = res.get_json()["data"]["version"]
    assert res.get_json()["data"]["current"] == d["current"]


def test_new_operation_parameter_validation(client):
    i = new_image(client)["image_id"]
    bad = [
        ("clahe", {"clip_limit": 0.5, "tile_grid_size": 8}, "parameters.clip_limit"),
        ("clahe", {"clip_limit": 11, "tile_grid_size": 8}, "parameters.clip_limit"),
        ("clahe", {"clip_limit": 2, "tile_grid_size": 1}, "parameters.tile_grid_size"),
        ("clahe", {"clip_limit": 2, "tile_grid_size": 8.5}, "parameters.tile_grid_size"),
        ("clahe", {"clip_limit": 2}, "parameters.tile_grid_size"),
        ("canny", {"threshold1": 200, "threshold2": 100}, "parameters.threshold1"),
        ("canny", {"threshold1": -1, "threshold2": 100}, "parameters.threshold1"),
        ("canny", {"threshold1": 10, "threshold2": 256}, "parameters.threshold2"),
        ("threshold", {"value": 256}, "parameters.value"),
        ("threshold", {"value": True}, "parameters.value"),
        ("threshold", {"value": "100"}, "parameters.value"),
        ("rotate_free", {"angle": 361}, "parameters.angle"),
        ("rotate_free", {"angle": "45"}, "parameters.angle"),
        ("rotate_free", {"angle": float("nan")}, "parameters.angle"),
        ("resize_percent", {"percent": 0}, "parameters.percent"),
        ("resize_percent", {"percent": 1001}, "parameters.percent"),
        ("crop_ratio", {"ratio": "3:2"}, "parameters.ratio"),
        ("crop_ratio", {"ratio": 1}, "parameters.ratio"),
        ("sepia", {"strength": 1}, "parameters.strength"),
        ("otsu", {"value": 1}, "parameters.value"),
    ]
    for name, params, field in bad:
        res = apply(client, i, 0, name, **params)
        assert res.status_code == 400 and error_of(res) == ("INVALID_PARAMETER", field), (name, params, res.get_json())
    assert apply(client, i, 0, "canny", threshold1=100, threshold2=100).status_code == 200   # bằng nhau thì hợp lệ


def test_percent_and_rotate_respect_pixel_limit(tmp_path):
    from imageeditor.app.backend import create_app
    app = create_app({"TESTING": True, "INSTANCE_DIR": str(tmp_path), "MAX_PIXELS": 10_000})
    c = app.test_client()
    i = new_image(c, size=(60, 80))["image_id"]          # 4800 px
    res = apply(c, i, 0, "resize_percent", percent=150)   # 90 x 120 = 10800 px
    assert res.status_code == 422 and error_of(res)[0] == "IMAGE_DIMENSIONS_EXCEEDED"
    res = apply(c, i, 0, "rotate_free", angle=45)         # hộp bao 99 x 99 = 9801 px: còn trong giới hạn
    assert res.status_code == 200
    j = new_image(c, size=(80, 100))["image_id"]          # 8000 px, xoay 45° -> 128 x 128 > 10000
    res = apply(c, j, 0, "rotate_free", angle=45)
    assert res.status_code == 422 and error_of(res)[0] == "IMAGE_DIMENSIONS_EXCEEDED"
    assert apply(c, j, 0, "rotate_free", angle=90).status_code == 200   # bội số 90 không đổi số pixel
    k = new_image(c, size=(8, 15))["image_id"]
    res = apply(c, k, 0, "crop_ratio", ratio="16:9")
    assert res.status_code == 400 and error_of(res) == ("INVALID_PARAMETER", "parameters.ratio")


def test_failed_new_operation_leaves_state_unchanged(client):
    i = new_image(client, size=(8, 15))["image_id"]
    assert apply(client, i, 0, "crop_ratio", ratio="16:9").status_code == 400
    res = apply(client, i, 0, "sepia")
    assert res.status_code == 200 and res.get_json()["data"]["version"] == 1


# ======================= Xuất ảnh sau xoay tự do =======================

def test_free_rotation_exports_png_alpha_and_jpg_on_white(client):
    i = new_image(client, size=(100, 100), color=(10, 20, 30))["image_id"]
    version = apply(client, i, 0, "rotate_free", angle=45).get_json()["data"]["version"]
    png = decode(export(client, i, version, format="png").data)
    assert png.shape == (142, 142, 4) and png[0, 0, 3] == 0
    jpg = decode(export(client, i, version, format="jpg", quality=95).data)
    assert jpg.shape == (142, 142, 3) and (jpg[0, 0] >= 250).all()      # góc trống thành nền trắng