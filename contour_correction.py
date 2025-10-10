import cv2
import numpy as np

def refine_contour_opencv(image_bgr, orig_contour, adjusted_contour, closeness=0.8):
    """
    Refines the wound contour using image edges between the original and adjusted contours.
    Works entirely with OpenCV (mobile compatible).
    Ensures the final contour never goes inside the original (blue) contour.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    H, W = gray.shape[:2]

    # --- Step 1: Masks for regions ---
    mask_orig = np.zeros((H, W), np.uint8)
    mask_user = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask_orig, [orig_contour.astype(np.int32)], 1)
    cv2.fillPoly(mask_user, [adjusted_contour.astype(np.int32)], 1)
    roi_union = cv2.bitwise_or(mask_orig, mask_user)

    # --- Step 2: Edge detection ---
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, None, iterations=1)

    # --- Step 3: Combine contours and make blend mask ---
    blended_mask = (mask_orig * (1 - closeness) + mask_user * closeness).astype(np.uint8)
    blended_mask = cv2.GaussianBlur(blended_mask, (7,7), 0)

    # --- Step 4: Use edges to pull blended contour towards true wound boundaries ---
    guided_edges = cv2.bitwise_and(edges, edges, mask=roi_union)
    combined = cv2.addWeighted(blended_mask*255, 0.5, guided_edges, 0.8, 0)
    _, binary = cv2.threshold(combined, 50, 255, cv2.THRESH_BINARY)

    # --- Step 5: Morphological clean-up ---
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)

    # --- Step 6: Extract contour ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return adjusted_contour
    main_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(main_contour, True)
    refined = cv2.approxPolyDP(main_contour, epsilon, True)
    refined = refined.reshape(-1, 2)

    # ======================================================================
    # --- Step 7: Post-processing: prevent refined contour from going inside original ---
    # ======================================================================

    # Create masks
    mask_refined = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask_refined, [refined.astype(np.int32)], 255)
    mask_orig_255 = (mask_orig * 255).astype(np.uint8)

    # Remove overlapping (inside) regions
    outside_only = cv2.bitwise_and(mask_refined, cv2.bitwise_not(mask_orig_255))

    # Merge with original to ensure continuity (outside OR on the original boundary)
    final_mask = cv2.bitwise_or(mask_orig_255, outside_only)

    # Slightly dilate to ensure it hugs outer boundary (optional, tweakable)
    final_mask = cv2.dilate(final_mask, np.ones((3,3), np.uint8), iterations=1)

    # Extract final contour
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return refined
    final_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(final_contour, True)
    final_contour = cv2.approxPolyDP(final_contour, epsilon, True)

    return final_contour.reshape(-1, 2)
