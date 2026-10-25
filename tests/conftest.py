import io

import cv2
import numpy as np
import pytest

from app.backend import create_app 


@pytest.fixture
def app(tmp_path):
    return create_app({"TESTING": True, "INSTANCE_DIR": str(tmp_path)})


@pytest.fixture
def client(app):
    return app.test_client()


def make_image(ext=".png", size=(40, 60), color=(30, 120, 220), alpha=None):
    """Tạo ảnh màu (BGR) trong bộ nhớ, size = (cao, rộng); có alpha thì tạo BGRA."""
    h, w = size
    img = np.full((h, w, 3), color, np.uint8)
    if alpha is not None:
        img = np.dstack([img, np.full((h, w), alpha, np.uint8)])
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return buf.tobytes()


def upload(client, data, filename="anh.png", field="image"):
    return client.post("/api/images", data={field: (io.BytesIO(data), filename)},
                       content_type="multipart/form-data")


def new_image(client, **kwargs):
    """Upload một ảnh hợp lệ và trả về phần data của response."""
    res = upload(client, make_image(**kwargs))
    assert res.status_code == 201, res.get_json()
    return res.get_json()["data"]


def op(client, image_id, version, **overrides):
    body = {"operation": "grayscale", "parameters": {}, "expected_version": version}
    body.update(overrides)
    return client.post(f"/api/images/{image_id}/operations", json=body)


def export(client, image_id, version, **fields):
    return client.post(f"/api/images/{image_id}/export", json={"expected_version": version, **fields})


def decode(data):
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)