import os
import threading
import time

from flask import Flask

from .routes import api_bp, pages_bp
from .services.store import ImageStore


def create_app(test_config=None):
    secret_key = os.environ.get("SECRET_KEY")
    is_testing = bool((test_config or {}).get("TESTING"))
    if os.environ.get("RENDER") and not secret_key and not is_testing:
        raise RuntimeError("Set SECRET_KEY in the Render environment.")
    # Render dùng giới hạn thấp hơn để giảm nguy cơ hết RAM; có thể đặt MAX_PIXELS để ghi đè.
    max_pixels_default = 2_000_000 if os.environ.get("RENDER") else 20_000_000

    app = Flask(
        __name__,
        template_folder="../frontend/templates",   # tính từ thư mục app/backend
        static_folder="../frontend/static",
    )
    app.config.update(
        SECRET_KEY=secret_key or "dev-only-change-me",
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SECURE=bool(os.environ.get("RENDER")),
        SESSION_COOKIE_SAMESITE="Lax",
        INSTANCE_DIR=None,                    # mặc định: thư mục instance/
        MAX_FILE_BYTES=10 * 1024 * 1024,      # một tệp ảnh: 10 MiB
        MAX_CONTENT_LENGTH=11 * 1024 * 1024,  # cả request upload: 11 MiB
        MAX_PIXELS=int(os.environ.get("MAX_PIXELS", max_pixels_default)),
        IMAGE_TTL_SECONDS=60 * 60,            # ảnh tạm tồn tại 60 phút từ lúc tải lên
        MAX_HISTORY=20,                       # số bước thao tác tối đa
        CLEANUP_INTERVAL_SECONDS=300,
    )
    if test_config:
        app.config.update(test_config)
    app.json.ensure_ascii = False

    base = app.config["INSTANCE_DIR"] or app.instance_path
    store = ImageStore(
        upload_dir=f"{base}/uploads",
        output_dir=f"{base}/outputs",
        ttl_seconds=app.config["IMAGE_TTL_SECONDS"],
        max_history=app.config["MAX_HISTORY"],
    )
    app.extensions["image_store"] = store

    app.register_blueprint(pages_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    if not app.config.get("TESTING"):
        _start_cleanup_thread(store, app.config["CLEANUP_INTERVAL_SECONDS"])
    return app


def _start_cleanup_thread(store, interval):
    def loop():
        while True:
            time.sleep(interval)
            store.cleanup()

    threading.Thread(target=loop, daemon=True, name="image-cleanup").start()