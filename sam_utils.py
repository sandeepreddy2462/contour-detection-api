import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor


# -------------------------------
# 1. Load SAM model
# -------------------------------
sam_checkpoint = "model/sam_vit_b_01ec64.pth"  # Path to SAM weights
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# -------------------------------
# 2. Get best mask from ROI or image center
# -------------------------------
def get_max_contour(image_width, image_height, roi_points,bbox=None):
    roi_copy = np.array(roi_points, dtype=np.float32)

    if bbox is not None:
        roi_copy[:, 0] -= bbox.x
        roi_copy[:, 1] -= bbox.y

    if roi_copy.size > 0:
        cx = np.mean(roi_copy[:, 0])
        cy = np.mean(roi_copy[:, 1])
        input_point = np.array([[cx, cy]])
    else:
        input_point = np.array([[image_width // 2, image_height // 2]])

    input_label = np.array([1])

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True
    )

    best_mask = masks[np.argmax(scores)]
    mask_uint8 = (best_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        return max_contour
    return None