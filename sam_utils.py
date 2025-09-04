import numpy as np
import cv2
import torch
from segment_anything_hq import sam_model_registry, SamPredictor
import random

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

# -------------------------------
# 2. Get best mask from ROI or image center
# -------------------------------
# ------------------------------
# Utility: safe I/O + overlays
# ------------------------------
def _ensure_uint8(img):
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

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
# Color constancy + contrast normalization
# -----------------------------------------
def _gray_world_white_balance(bgr):
    bgr = bgr.astype(np.float32) + 1e-6
    avg_b = np.mean(bgr[:,:,0])
    avg_g = np.mean(bgr[:,:,1])
    avg_r = np.mean(bgr[:,:,2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    bgr[:,:,0] *= (avg_gray / avg_b)
    bgr[:,:,1] *= (avg_gray / avg_g)
    bgr[:,:,2] *= (avg_gray / avg_r)
    bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    return bgr

def _clahe_lab_luminance(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

# ---------------------------------------------------------
# Redness / wound-bed saliency inside ROI (returns [0..1])
# ---------------------------------------------------------
def _wound_saliency(bgr, roi_mask):
    # LAB + HSV signals
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Normalize per-ROI to reduce skin tone bias
    roi_inds = roi_mask.astype(bool)
    def nz_norm(x):
        vals = x[roi_inds].astype(np.float32)
        if vals.size == 0:
            return np.zeros_like(x, dtype=np.float32)
        mn, mx = np.percentile(vals, 2), np.percentile(vals, 98)
        x = x.astype(np.float32)
        x = (x - mn) / (mx - mn + 1e-6)
        return np.clip(x, 0, 1)

    a_n = nz_norm(a)      # red-green axis (red -> high)
    S_n = nz_norm(S)      # saturation
    L_n = nz_norm(L)      # luminance (wounds often lower L)
    V_n = nz_norm(V)

    # Combine: emphasize high a*, high S, slightly lower L (darker red)
    sal = (0.55 * a_n + 0.35 * S_n + 0.15 * (1.0 - L_n))
    sal = cv2.GaussianBlur(sal, (5,5), 0)

    # Keep only ROI, null elsewhere
    sal = sal * (roi_mask.astype(np.float32))
    if sal.max() > 0:
        sal /= (sal.max() + 1e-6)
    return sal

# ---------------------------------------------------------
# Seed generation: sure-FG and sure-BG inside ROI
# ---------------------------------------------------------
def _seed_masks_from_saliency(sal, roi_mask, fg_q=0.80, ring_px=8):
    # Foreground = top quantile inside ROI
    vals = sal[roi_mask.astype(bool)]
    if vals.size == 0:
        return None, None
    t = np.quantile(vals, fg_q)
    sure_fg = ((sal >= t) & (roi_mask > 0)).astype(np.uint8)

    # Background ring: ROI minus inner erode, plus low-saliency inside ROI
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_px, ring_px))
    inner = cv2.erode(roi_mask, k, iterations=1)
    ring = (roi_mask.astype(np.uint8) & (1 - inner)).astype(np.uint8)

    low_sal = ((sal <= np.quantile(vals, 0.25)) & (roi_mask > 0)).astype(np.uint8)
    sure_bg = ((ring > 0) | (low_sal > 0)).astype(np.uint8)
    return sure_fg, sure_bg

# ---------------------------------------------------------
# GrabCut with seeds (outside ROI = sure background)
# ---------------------------------------------------------
def _grabcut_refine(image_bgr, roi_mask, sure_fg, sure_bg, iters=1):
    H, W = roi_mask.shape
    gc_mask = np.full((H,W), cv2.GC_PR_BGD, dtype=np.uint8)

    # Outside ROI is definite background
    gc_mask[roi_mask == 0] = cv2.GC_BGD

    # Inside ROI unknown by default, then set seeds
    if sure_bg is not None:
        gc_mask[sure_bg > 0] = cv2.GC_BGD
    if sure_fg is not None:
        gc_mask[sure_fg > 0] = cv2.GC_FGD

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    cv2.grabCut(image_bgr, gc_mask, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)
    print("faster_thing")
    out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    out = out * (roi_mask.astype(np.uint8))
    return out

# ---------------------------------------------------------
# Points for SAM: sample positives at saliency peaks, negatives on BG ring
# ---------------------------------------------------------
def _points_for_sam(sal, roi_mask, sure_fg, sure_bg, max_pos=20, max_neg=20):
    # Positive: top-K peaks
    sal_blur = cv2.GaussianBlur(sal, (9,9), 0)
    peaks = (sal_blur == cv2.dilate(sal_blur, np.ones((9,9), np.uint8)))
    pos_inds = np.argwhere((peaks > 0) & (sure_fg > 0))
    if pos_inds.shape[0] == 0:
        pos_inds = np.argwhere(sure_fg > 0)
    if pos_inds.shape[0] > max_pos:
        sel = np.random.choice(pos_inds.shape[0], max_pos, replace=False)
        pos_inds = pos_inds[sel]

    # Negative: sure background inside ROI
    neg_inds = np.argwhere((sure_bg > 0) & (roi_mask > 0))
    if neg_inds.shape[0] > max_neg:
        sel = np.random.choice(neg_inds.shape[0], max_neg, replace=False)
        neg_inds = neg_inds[sel]

    # SAM wants (x,y); our inds are (y,x)
    pos = np.flip(pos_inds, axis=1) if pos_inds.size else np.empty((0,2), dtype=np.int32)
    neg = np.flip(neg_inds, axis=1) if neg_inds.size else np.empty((0,2), dtype=np.int32)
    return pos.astype(np.int32), neg.astype(np.int32)

# ---------------------------------------------------------
# Choose best SAM mask by edge overlap + IoU to GC
# ---------------------------------------------------------
def _choose_best_mask(masks, image_gray, gc_mask):
    edges = cv2.Canny(image_gray, 50, 150)
    best_idx, best_score = 0, -1e9

    def boundary_overlap(m, edges):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0
        bmask = np.zeros_like(edges)
        cv2.drawContours(bmask, cnts, -1, 255, 1)
        return np.count_nonzero((bmask > 0) & (edges > 0))

    def iou(a, b):
        inter = np.logical_and(a>0, b>0).sum()
        union = np.logical_or(a>0, b>0).sum()
        return inter / (union + 1e-6)

    for i, m in enumerate(masks):
        m8 = m.astype(np.uint8)
        s1 = boundary_overlap(m8, edges)
        s2 = 1000.0 * iou(m8, gc_mask)    # weight IoU strongly
        score = s1 + s2
        if score > best_score:
            best_score, best_idx = score, i
    return masks[best_idx].astype(np.uint8)


#-----------------------------------
# Without bbox 
#-----------------------------------
def wound_segmentation(image, roi_polygon, debug=False, debug_dir=None):
    """
    image       : np.ndarray (H, W, 3)  BGR uint8
    roi_polygon : list[(x, y)] ROI in absolute full image coords (floats allowed)
    """
    assert image.ndim == 3 and image.shape[2] == 3, "image must be HxWx3 BGR"
    H, W = image.shape[:2]

    # Use full image, ROI in absolute coords
    roi_poly_np = np.array(roi_polygon, dtype=np.float32)
    roi_poly_int = np.round(roi_poly_np).astype(np.int32)

    # Full mask
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_poly_int], 1)

    # Preprocessing
    img_wb = _gray_world_white_balance(image)
    img_pp = _clahe_lab_luminance(img_wb)

    # Saliency and seeds
    sal = _wound_saliency(img_pp, roi_mask)
    sure_fg, sure_bg = _seed_masks_from_saliency(sal, roi_mask, fg_q=0.80, ring_px=max(6, min(H, W)//40))

    # GrabCut
    gc_mask = _grabcut_refine(img_pp, roi_mask, sure_fg, sure_bg, iters=5)

    # Points for SAM
    pos_pts, neg_pts = _points_for_sam(sal, roi_mask, sure_fg, sure_bg, max_pos=25, max_neg=25)
    predictor.set_image(img_pp)
    labels = None
    pts = None
    if (len(pos_pts) + len(neg_pts)) > 0:
        labels = np.concatenate([
            np.ones(len(pos_pts), dtype=np.int32), np.zeros(len(neg_pts), dtype=np.int32)
        ])
        pts = np.vstack([pos_pts, neg_pts]).astype(np.float32)

    # Tight ROI bbox for SAM (in full image coords)
    rx, ry, rw, rh = cv2.boundingRect(roi_poly_int)
    sam_masks, sam_scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=labels,
        box=np.array([rx, ry, rx+rw, ry+rh], dtype=np.float32),
        multimask_output=True
    )
    sam_masks = [m.astype(np.uint8) for m in sam_masks]
    gray_img = cv2.cvtColor(img_pp, cv2.COLOR_BGR2GRAY)
    sam_best = _choose_best_mask(sam_masks, gray_img, gc_mask)
    consensus = (gc_mask > 0).astype(np.uint8) & (sam_best > 0).astype(np.uint8)

    # Morph cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    consensus = cv2.morphologyEx(consensus, cv2.MORPH_OPEN, k, iterations=1)
    consensus = cv2.morphologyEx(consensus, cv2.MORPH_CLOSE, k, iterations=2)
    consensus = (consensus * roi_mask).astype(np.uint8)

    # Largest component only
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(consensus, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + np.argmax(areas)
        consensus = (labels_im == best).astype(np.uint8)

    final_mask = consensus

    # Final contour
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contour = max(contours, key=cv2.contourArea) if contours else None
    
    return final_mask, final_contour.astype(float) if final_contour is not None else None