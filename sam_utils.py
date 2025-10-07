import numpy as np
import cv2
import torch
from segment_anything_hq import sam_model_registry, SamPredictor
import random
import time
from datetime import datetime
from typing import Optional
from contextlib import nullcontext

# -------------------------------
# 1. Load SAM model
# -------------------------------
# -------------------------------
# 1. Load SAM model
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_b"
sam_checkpoint = "model/hqsam_finetuned_on_ec2.pth"  # Path to SAM weights
sam = sam_model_registry[model_type]()
state_dict = torch.load(sam_checkpoint, map_location=device)
sam.load_state_dict(state_dict)
sam.to(device)
predictor = SamPredictor(sam)

# ------------------------------
# Utility: safe I/O + overlays
# ------------------------------
def _ensure_uint8(img):
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)

def _overlay_contour(base_bgr, mask, color=(0,255,0), alpha=0.4, thickness=2):
    base = base_bgr.copy()
    overlay = base_bgr.copy()
    if mask is not None and mask.any():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness)
    return cv2.addWeighted(overlay, alpha, base, 1-alpha, 0)

def _save_dbg(image_bgr, path):
    try:
        cv2.imwrite(path, _ensure_uint8(image_bgr))
    except Exception as e:
        print(f"[dbg] Could not save {path}: {e}")

# -----------------------------------------
# Preprocessing
# -----------------------------------------
def _gray_world_white_balance(bgr):
    bgr = bgr.astype(np.float32) + 1e-6
    avg_b, avg_g, avg_r = [np.mean(bgr[:,:,i]) for i in range(3)]
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    bgr[:,:,0] *= (avg_gray / avg_b)
    bgr[:,:,1] *= (avg_gray / avg_g)
    bgr[:,:,2] *= (avg_gray / avg_r)
    return np.clip(bgr, 0, 255).astype(np.uint8)

def _clahe_lab_luminance(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)
    return cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

# ---------------------------------------------------------
# Saliency (redness + saturation emphasis) -> [0..1]
# ---------------------------------------------------------
def _wound_saliency(bgr, roi_mask):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    roi_inds = roi_mask.astype(bool)
    def nz_norm(x):
        vals = x[roi_inds].astype(np.float32)
        if vals.size == 0: 
            return np.zeros_like(x, dtype=np.float32)
        mn, mx = np.percentile(vals, 2), np.percentile(vals, 98)
        return np.clip((x.astype(np.float32)-mn)/(mx-mn+1e-6), 0, 1)

    a_n, S_n, L_n = nz_norm(a), nz_norm(S), nz_norm(L)
    sal = (0.55 * a_n + 0.35 * S_n + 0.15 * (1.0 - L_n))
    sal = cv2.GaussianBlur(sal, (5,5), 0)
    sal = sal * roi_mask.astype(np.float32)
    if sal.max() > 0:
        sal /= (sal.max() + 1e-6)
    return sal

# ---------------------------------------------------------
# Seed generation (adaptive threshold + area filtering)
# -> sure_fg = union of all wound-like blobs
# -> sure_bg = ring + low-saliency
# ---------------------------------------------------------
def _seed_masks_from_saliency(sal, roi_mask, min_area=50):
    vals = sal[roi_mask.astype(bool)]
    if vals.size == 0:
        return None, None

    # Adaptive 75th-percentile, clipped to a safe range
    q = np.quantile(vals, 0.75)
    fg_thresh = np.clip(q, 0.7, 0.85)

    fg_mask = ((sal >= fg_thresh) & (roi_mask > 0)).astype(np.uint8)

    # Remove tiny/huge blobs
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    cleaned = np.zeros_like(fg_mask)
    max_area = 0.5 * roi_mask.sum()  # avoid swallowing most of the ROI
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            cleaned[labels_im == i] = 1

    # Join nearby wound parts
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    sure_fg = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k, iterations=2)
    # Background ring + low-saliency
    ring_px = max(6, min(sal.shape) // 40)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px, ring_px))
    inner = cv2.erode(roi_mask, k, iterations=1)
    ring  = (roi_mask.astype(np.uint8) & (1 - inner)).astype(np.uint8)
    low_sal = ((sal <= 0.25) & (roi_mask > 0)).astype(np.uint8)
    sure_bg = ((ring > 0) | (low_sal > 0)).astype(np.uint8)

    return sure_fg, sure_bg

# ---------------------------------------------------------
# GrabCut
# ---------------------------------------------------------
def _grabcut_refine(image_bgr, roi_mask, sure_fg, sure_bg, iters=5):
    H, W = roi_mask.shape
    gc_mask = np.full((H,W), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[roi_mask == 0] = cv2.GC_BGD
    if sure_bg is not None: gc_mask[sure_bg > 0] = cv2.GC_BGD
    if sure_fg is not None: gc_mask[sure_fg > 0] = cv2.GC_FGD
    bgdModel, fgdModel = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
    cv2.grabCut(image_bgr, gc_mask, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)
    return np.where((gc_mask==cv2.GC_FGD)|(gc_mask==cv2.GC_PR_FGD),1,0).astype(np.uint8) * roi_mask

# ---------------------------------------------------------
# Choose best SAM mask (edge-alignment + IoU to GC)
# ---------------------------------------------------------
def _choose_best_mask(masks, image_gray, gc_mask):
    edges = cv2.Canny(image_gray, 50, 150)
    best_idx, best_score = 0, -1e9

    def boundary_overlap(m, edges):
        cnts,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return 0
        bmask=np.zeros_like(edges)
        cv2.drawContours(bmask,cnts,-1,255,1)
        return np.count_nonzero((bmask>0)&(edges>0))

    def iou(a,b):
        inter=np.logical_and(a>0,b>0).sum()
        union=np.logical_or(a>0,b>0).sum()
        return inter/(union+1e-6)

    for i,m in enumerate(masks):
        m8=m.astype(np.uint8)
        score = boundary_overlap(m8, edges) + 1000.0 * iou(m8, gc_mask)
        if score > best_score:
            best_score, best_idx = score, i

    return masks[best_idx].astype(np.uint8)


# --------------------------------------
# Main function with CONSENSUS fallback
# --------------------------------------
def wound_segmentation(image, roi_polygon, debug=False, debug_dir=None):
    """
    image       : BGR np.ndarray (H,W,3)
    roi_polygon : list[(x,y)] absolute coordinates (same frame as 'image')
    """
    assert image.ndim == 3 and image.shape[2] == 3
    H, W = image.shape[:2]
    t0 = time.perf_counter()  
    last_ts = t0

    # ROI mask
    roi_poly_np = np.array(roi_polygon, dtype=np.int32)
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_poly_np], 1)

    # Preprocess (white balance + CLAHE)
    img_wb = _gray_world_white_balance(image)
    img_pp = _clahe_lab_luminance(img_wb)

    # Saliency + seeds
    sal = _wound_saliency(img_pp, roi_mask)
    sure_fg, sure_bg = _seed_masks_from_saliency(sal, roi_mask)
    # sure_fg, sure_bg = fast_fg_bg_mask(sal, roi_mask)

    t_before_grab_cut = time.perf_counter()
    print(f"[TIMING] Before grab_cut: {t_before_grab_cut - t0:.3f} seconds")

    # GrabCut
    # gc_mask = _grabcut_refine(img_pp, roi_mask, sure_fg, sure_bg, iters=3)
    gc_mask = roi_mask
    t_after_grab_cut = time.perf_counter()
    grab_cut_time = t_after_grab_cut - t_before_grab_cut
    print(f"[TIMING] grab_cut_time: {grab_cut_time:.3f} seconds")

    # SAM (box only, deterministic)
    t_before_set_image = time.perf_counter()

    predictor.set_image(img_pp)
    
    t_after_set_image = time.perf_counter()
    set_image_time = t_after_set_image - t_before_set_image
    print(f"[TIMING] set_image_time: {set_image_time:.3f} seconds")
    
    rx, ry, rw, rh = cv2.boundingRect(roi_poly_np)
    sam_masks, _, _ = predictor.predict(
        point_coords=None, point_labels=None,
        box=np.array([rx, ry, rx+rw, ry+rh], dtype=np.float32),
        multimask_output=True
    )
    t_after_predict = time.perf_counter()
    prediction_time = t_after_predict - t_after_set_image
    print(f"[TIMING] prediction_time: {prediction_time:.3f} seconds")
    
    sam_masks = [m.astype(np.uint8) for m in sam_masks]
    sam_union = np.max(np.stack(sam_masks), axis=0).astype(np.uint8)

    # Best single SAM mask (optional, helps with leakage scoring)
    gray_img  = cv2.cvtColor(img_pp, cv2.COLOR_BGR2GRAY)
    sam_best  = _choose_best_mask(sam_masks, gray_img, gc_mask)

    # -------------------------------
    # Consensus strategies
    # -------------------------------
    # 1) UNION strategy (more recall)
    union_mask = ((sure_fg > 0) | (sam_union > 0)).astype(np.uint8) * roi_mask

    # 2) CONSERVATIVE fallback = (GraphCut ∩ SAM-union)
    conservative = ((gc_mask > 0) & (sam_union > 0)).astype(np.uint8) * roi_mask

    # Heuristics to detect union leakage (too big / too far from GC)
    roi_area = int(roi_mask.sum())
    union_area = int(union_mask.sum())
    gc_area = int(gc_mask.sum())

    # IoU between UNION and GC
    inter = np.logical_and(union_mask>0, gc_mask>0).sum()
    u = np.logical_or(union_mask>0, gc_mask>0).sum()
    iou_union_gc = inter / (u + 1e-6)

    # Conditions to switch to conservative
    leak = (union_area > 0.65 * roi_area) or (iou_union_gc < 0.45)
    final_mask = conservative if leak else union_mask

    # Clean-up and largest component
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, k, iterations=1)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, k, iterations=2)
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    if num_labels > 1:
        best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask = (labels_im == best).astype(np.uint8)

    # Final contour
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contour = max(contours, key=cv2.contourArea) if contours else None
    
    # ---------- Final timestamp ----------
    t_end = time.perf_counter()
    total_time = t_end - t0
    print(f"[TIMING] total_time: {total_time:.3f} seconds")

    return final_mask, final_contour, grab_cut_time, set_image_time, prediction_time, total_time

