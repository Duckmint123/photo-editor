import cv2

from .alpha import keep_alpha

CANNY_BLUR_SIGMA = 1.4   # làm mờ nhẹ trước khi dò biên để bớt nhiễu


def _gray(img):
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)


def _as_result(mask, img):
    """Ảnh nhị phân/biên một kênh -> BGR 3 kênh (và alpha của ảnh gốc), cùng định dạng với mọi bước khác."""
    return keep_alpha(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), img)


def canny(img, threshold1, threshold2):
    """Dò biên Canny với hai ngưỡng (threshold1 <= threshold2): biên màu trắng trên nền đen.

    Điểm có gradient lớn hơn threshold2 chắc chắn là biên; từ threshold1 đến threshold2 chỉ
    được giữ nếu nối với biên chắc chắn; nhỏ hơn threshold1 bị bỏ.
    """
    blurred = cv2.GaussianBlur(_gray(img), (0, 0), CANNY_BLUR_SIGMA)
    return _as_result(cv2.Canny(blurred, threshold1, threshold2), img)


def threshold(img, value):
    """Ngưỡng cố định trên ảnh xám: điểm sáng hơn value thành trắng, còn lại thành đen."""
    _, mask = cv2.threshold(_gray(img), value, 255, cv2.THRESH_BINARY)
    return _as_result(mask, img)


def otsu(img):
    """Ngưỡng Otsu: tự chọn ngưỡng làm tối thiểu phương sai trong từng lớp (sáng/tối) của ảnh xám."""
    _, mask = cv2.threshold(_gray(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _as_result(mask, img)