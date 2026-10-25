import numpy as np


def on_premultiplied(img, fn):
    """Chạy fn trên ảnh đã nhân trước alpha để màu của điểm ảnh trong suốt không lem vào viền.

    - Ảnh 3 kênh, hoặc ảnh có kênh alpha nhưng đều bằng 255: chạy fn thẳng trên phần màu.
    - Ảnh có điểm ảnh trong suốt: nhân màu với alpha, chạy fn trên mảng float32 4 kênh
      (B*a, G*a, R*a, A), rồi chia lại cho alpha.
    fn nhận một mảng và trả về mảng cùng số kênh (có thể khác kích thước).
    """
    if img.shape[2] == 3:
        return fn(img)
    alpha = img[:, :, 3]
    if (alpha == 255).all():
        out = fn(np.ascontiguousarray(img[:, :, :3]))
        return np.dstack([out, np.full(out.shape[:2], 255, np.uint8)])

    a = alpha.astype(np.float32) / 255.0
    pre = np.dstack([img[:, :, :3].astype(np.float32) * a[:, :, None], alpha.astype(np.float32)])
    out = fn(pre)
    out_alpha = np.clip(out[:, :, 3], 0, 255)
    a_out = out_alpha / 255.0
    visible = a_out > 0
    bgr = np.where(visible[:, :, None], out[:, :, :3] / np.where(visible, a_out, 1.0)[:, :, None], 0)
    result = np.dstack([np.clip(np.floor(bgr + 0.5), 0, 255), np.floor(out_alpha + 0.5)])
    return result.astype(np.uint8)