import threading
import time

from flask import Flask

from .routes import api_bp, pages_bp
from .services.store import ImageStore


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY="dev-only-change-me",      # đổi khi triển khai thật
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        INSTANCE_DIR=None,                    # mặc định: thư mục instance/
        MAX_FILE_BYTES=10 * 1024 * 1024,      # một tệp ảnh: 10 MiB
        MAX_CONTENT_LENGTH=11 * 1024 * 1024,  # cả request upload: 11 MiB
        MAX_PIXELS=20_000_000,                # 20 triệu pixel
        IMAGE_TTL_SECONDS=60 * 60,            # ảnh tạm tồn tại 60 phút kể từ lần dùng cuối
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