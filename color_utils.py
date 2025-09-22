import numpy as np
import cv2
from skimage.morphology import remove_small_objects
from skimage.measure import label

def clean_mask(mask: np.ndarray, min_area_px=200, erode_px=2) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    labeled = label(mask)
    cleaned = remove_small_objects(labeled, min_size=min_area_px)
    mask_bin = (cleaned > 0).astype(np.uint8)
    if erode_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erode_px+1, 2*erode_px+1))
        mask_bin = cv2.erode(mask_bin, kernel, iterations=1)
    return (mask_bin * 255).astype(np.uint8)

def gray_world(rgb: np.ndarray, mask: np.ndarray):
    m = mask > 0
    mean = rgb[m].mean(axis=0)
    g = mean.mean()
    gains = g / np.clip(mean, 1e-6, None)
    return (rgb * gains).clip(0, 255).astype(np.uint8)

def rgb_to_lab(rgb: np.ndarray):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[...,0] *= (100.0 / 255.0)
    lab[...,1] -= 128.0
    lab[...,2] -= 128.0
    return lab

def quality_masks(lab: np.ndarray, mask: np.ndarray):
    L = lab[...,0]
    a = lab[...,1]
    b = lab[...,2]
    C = np.sqrt(a**2 + b**2)
    m = mask > 0
    if not np.any(m):
        return m, m, m
    L_vals = L[m]
    p10 = np.percentile(L_vals, 10)
    shadow = (L < p10)
    glare = (L > np.percentile(L_vals, 95)) & (C < 8)
    reliable = m & (~shadow) & (~glare)
    return reliable, shadow & m, glare & m

def classify_ryb(lab: np.ndarray, reliable: np.ndarray, adaptive_black_floor=22.0):
    L = lab[...,0]
    a = lab[...,1]
    b = lab[...,2]
    C = np.sqrt(a**2 + b**2)
    H = (np.degrees(np.arctan2(b, a)) + 360) % 360

    rm = reliable
    if not np.any(rm): return 0, 0, 0, 0

    L_vals = L[rm]
    p20 = float(np.percentile(L_vals, 20))
    L_BLACK = max(adaptive_black_floor, p20)

    black = ((L <= L_BLACK) | ((L <= 48) & (C <= 20))) & rm
    yellow1 = (L >= 78) & (C < 20) & rm
    yellow2 = (C < 12) & (b > 4) & (L >= 60) & (L <= 92) & rm
    undecided = rm & (~black) & (~yellow1) & (~yellow2) & (C >= 5)

    red = (H < 52) & undecided | (H > 115) & undecided
    yellow = ((H >= 52) & (H <= 115) & undecided) | yellow1 | yellow2

    R = int(np.count_nonzero(red))
    Y = int(np.count_nonzero(yellow))
    B = int(np.count_nonzero(black))
    N = int(np.count_nonzero(rm))
    return R, Y, B, N

def percentify(R, Y, B, N):
    if N == 0: return 0.0, 0.0, 0.0
    r = round(100.0 * R / N, 1)
    y = round(100.0 * Y / N, 1)
    b = round(100.0 * B / N, 1)
    total = r + y + b
    diff = round(100.0 - total, 1)
    if diff != 0:
        if r >= y and r >= b: r += diff
        elif y >= r and y >= b: y += diff
        else: b += diff
    return r, y, b

def analyze_color(rgb_crop: np.ndarray, mask: np.ndarray):
    mask = clean_mask(mask)
    rgb_crop = gray_world(rgb_crop, mask)
    lab = rgb_to_lab(rgb_crop)
    reliable, shadow_m, glare_m = quality_masks(lab, mask)
    R, Y, B, N = classify_ryb(lab, reliable)
    r_pct, y_pct, b_pct = percentify(R, Y, B, N)
    roi_px = int(np.count_nonzero(mask))
    rel_px = int(np.count_nonzero(reliable))
    # flags = []
    # if roi_px < 500: flags.append("SMALL_ROI")
    # if rel_px / max(1, roi_px) < 0.6: flags.append("LOW_RELIABILITY")
    # if np.count_nonzero(glare_m) > 0.05 * roi_px: flags.append("GLARE")
    # if np.count_nonzero(shadow_m) > 0.2 * roi_px: flags.append("SHADOWS")

    return {
        "red": r_pct,
        "yellow": y_pct,
        "black": b_pct,
        "sample_count": int(N),
        "reliable_fraction": round(rel_px / max(1, roi_px), 3)
    }