import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


class NotFound(Exception):
    """Ảnh không tồn tại, đã hết hạn, hoặc không thuộc phiên hiện tại."""


@dataclass
class ImageRecord:
    id: str
    owner: str            # mã phiên sở hữu ảnh
    created: float        # thời điểm upload; hạn dùng tính từ đây và không gia hạn
    original_name: str
    fmt: str              # jpg, png hoặc webp (theo nội dung tệp)
    size_bytes: int
    metadata: dict        # ảnh gốc sau chuẩn hóa hướng: width, height, has_alpha
    current: dict         # ảnh đang chỉnh: width, height, color_mode, has_alpha
    ops: list = field(default_factory=list)   # chuỗi thao tác: {"name", "params"}
    cursor: int = 0                           # ops[:cursor] là trạng thái hiện tại
    version: int = 0                          # tăng mỗi khi trạng thái đổi
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


def write_atomic(path, data):
    """Ghi qua tệp tạm rồi đổi tên, để người đọc không bao giờ thấy tệp dở dang."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class ImageStore:
    """Quản lý ảnh tạm. Siêu dữ liệu nằm trong bộ nhớ, ảnh nằm trên đĩa.

    Chạy một tiến trình Flask (cách chạy đồ án). Nếu chạy nhiều worker thì phải
    chuyển siêu dữ liệu sang nơi dùng chung (tệp hoặc CSDL).
    """

    def __init__(self, upload_dir, output_dir, ttl_seconds, max_history, clock=time.time):
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.ttl = ttl_seconds
        self.max_history = max_history
        self.clock = clock   # tách ra để test được việc hết hạn
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._records = {}
        self._lock = threading.Lock()

    # Đường dẫn chỉ dựng từ mã ảnh do server sinh, không dùng tên tệp của người dùng
    def original_path(self, rec):
        """Byte tệp người dùng tải lên."""
        return self.upload_dir / rec.id / f"original.{rec.fmt}"

    def source_png_path(self, rec):
        """Ảnh gốc đã chuẩn hóa hướng, lưu PNG (dùng cho preview source=original)."""
        return self.output_dir / rec.id / "original.png"

    def current_png_path(self, rec):
        """Ảnh hiện tại sau các thao tác (chỉ có khi đã áp dụng ít nhất một bước)."""
        return self.output_dir / rec.id / "current.png"

    def current_view_path(self, rec):
        return self.source_png_path(rec) if rec.cursor == 0 else self.current_png_path(rec)

    def write_current(self, rec, png_bytes):
        write_atomic(self.current_png_path(rec), png_bytes)

    def add(self, owner, info, original_bytes, png_bytes, state):
        self.cleanup()
        rec = ImageRecord(
            id=str(uuid.uuid4()), owner=owner, created=self.clock(),
            original_name=info["name"], fmt=info["fmt"], size_bytes=info["size_bytes"],
            metadata={"width": state["width"], "height": state["height"], "has_alpha": state["has_alpha"]},
            current=dict(state),
        )
        try:
            write_atomic(self.original_path(rec), original_bytes)
            write_atomic(self.source_png_path(rec), png_bytes)
        except Exception:
            self._remove_files(rec)
            raise
        with self._lock:
            self._records[rec.id] = rec
        return rec

    def get(self, image_id, owner):
        """Trả về bản ghi nếu đúng chủ và còn hạn; ngược lại ném NotFound."""
        with self._lock:
            rec = self._records.get(image_id)
            if rec is None or rec.owner != owner:
                raise NotFound(image_id)
            if self.clock() - rec.created >= self.ttl:
                self._remove_locked(rec)
                raise NotFound(image_id)
            return rec

    def remove(self, rec):
        with self._lock:
            self._remove_locked(rec)

    def _remove_files(self, rec):
        shutil.rmtree(self.upload_dir / rec.id, ignore_errors=True)
        shutil.rmtree(self.output_dir / rec.id, ignore_errors=True)

    def _remove_locked(self, rec):
        self._records.pop(rec.id, None)
        self._remove_files(rec)

    def cleanup(self):
        """Xóa ảnh hết hạn và thư mục mồ côi (ví dụ sau khi server khởi động lại)."""
        now = self.clock()
        with self._lock:
            expired = [r for r in self._records.values() if now - r.created >= self.ttl]
            for rec in expired:
                self._remove_locked(rec)
            known = set(self._records)
        for base in (self.upload_dir, self.output_dir):
            for d in base.iterdir():
                if d.is_dir() and d.name not in known and now - d.stat().st_mtime >= self.ttl:
                    shutil.rmtree(d, ignore_errors=True)
        return len(expired)