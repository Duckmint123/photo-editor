import cv2

from .alpha import keep_alpha


def clahe(img, clip_limit, tile_grid_size):
    """CLAHE (cân bằng histogram thích nghi có giới hạn) trên kênh sáng L của không gian LAB.

    Chỉ kênh L được cân bằng, hai kênh màu a, b giữ nguyên nên màu không bị lệch.
    clip_limit: ngưỡng cắt độ tương phản (lớn hơn thì mạnh hơn); tile_grid_size: chia ảnh thành
    tile_grid_size x tile_grid_size ô. Kênh alpha (nếu có) giữ nguyên.
    """
    bgr = img[:, :, :3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    equalizer = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    lab = cv2.merge([equalizer.apply(l_channel), a_channel, b_channel])
    return keep_alpha(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), img)