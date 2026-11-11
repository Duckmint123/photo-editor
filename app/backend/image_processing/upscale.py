import cv2
import numpy as np

from .alpha import keep_alpha, on_premultiplied
from .geometry import percent_size

# Giảm nhiễu (non-local means) rất tốn thời gian: khoảng 1,6 giây cho mỗi triệu điểm ảnh (đo trên máy dev),
# và mỗi lần Hoàn tác/Làm lại backend dựng lại cả chuỗi từ ảnh gốc nên bước này chạy lại. Vì vậy chỉ cho phép
# giảm nhiễu trên ảnh nguồn không quá mức này (ảnh nhỏ cần phóng to thường nằm dưới ngưỡng).
DENOISE_MAX_PIXELS = 2_000_000

DENOISE_TEMPLATE_WINDOW = 7    # cỡ mảnh so sánh của non-local means
DENOISE_SEARCH_WINDOW = 21     # vùng tìm mảnh giống nhau


def upscale_size(width, height, mode, value):
    """Kích thước ảnh đầu ra.

    mode "percent": value là phần trăm so với ảnh hiện tại (100 = giữ nguyên).
    mode "pixels": value là chiều rộng mới (điểm ảnh); chiều cao tính theo đúng tỷ lệ ảnh hiện tại.
    """
    if mode == "percent":
        return percent_size(width, height, value)
    return int(value), max(1, round(height * value / width))


def _denoise(img, strength):
    """Giảm nhiễu trên phần màu (kênh alpha để nguyên). strength là tham số h của OpenCV."""
    bgr = np.ascontiguousarray(img[:, :, :3])
    out = cv2.fastNlMeansDenoisingColored(
        bgr, None, strength, strength, DENOISE_TEMPLATE_WINDOW, DENOISE_SEARCH_WINDOW)
    return keep_alpha(out, img)


def _unsharp(a, amount, radius):
    """Unsharp Mask: out = a + amount * (a - GaussianBlur(a, sigma=radius)), chỉ trên ba kênh màu.

    `a` là ảnh BGR uint8 hoặc mảng float32 4 kênh đã nhân trước alpha (kênh thứ 4 là alpha, không đụng tới).
    """
    color = np.ascontiguousarray(a[:, :, :3])
    blurred = cv2.GaussianBlur(color, (0, 0), radius)
    sharp = cv2.addWeighted(color, 1.0 + amount, blurred, -amount, 0)
    return sharp if a.shape[2] == 3 else np.dstack([sharp, a[:, :, 3]])


def upscale(img, mode, value, amount, radius, denoise):
    """Phóng to rồi làm nét (không dùng AI), theo thứ tự:

    1. Giảm nhiễu (nếu denoise > 0) trên ảnh gốc kích thước nhỏ, để nhiễu không bị phóng to theo ảnh.
    2. Phóng to bằng Lanczos (INTER_LANCZOS4), nếu kích thước đổi.
    3. Unsharp Mask với `amount` (độ mạnh, 0 = tắt) và `radius` (sigma của Gaussian, tính theo điểm ảnh
       của ảnh đầu ra).

    Với ảnh có vùng trong suốt, bước 2 và 3 chạy trên ảnh nhân trước alpha nên màu ẩn của điểm ảnh
    trong suốt không lem ra viền; kênh alpha chỉ được phóng to, không bị làm nét.
    mode = "percent" thì value là phần trăm (100 = giữ nguyên kích thước, chỉ làm nét); mode = "pixels"
    thì value là chiều rộng mới.
    """
    height, width = img.shape[:2]
    new_w, new_h = upscale_size(width, height, mode, value)
    if denoise > 0:
        img = _denoise(img, denoise)

    def process(a):
        if (new_w, new_h) != (width, height):
            a = cv2.resize(a, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return _unsharp(a, amount, radius) if amount > 0 else a

    return on_premultiplied(img, process)