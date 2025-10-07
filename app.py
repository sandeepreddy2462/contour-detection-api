# app.py (only the changed/added parts shown fully for clarity)
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import numpy as np
import cv2
import logging
from typing import Optional

from sam_utils import wound_segmentation
from contour_correction import fit_between_contours, correct_wound_contour_from_contours

# (existing app initialization & middleware remain unchanged)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# from color_utils import analyze_color

from typing import List

# Initialize FastAPI app
app = FastAPI(
    title="FastCare API",
    description="A FastAPI server with public endpoints for healthcare management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# -----------------------------
# API Endpoint 1: Contour Detection
# -----------------------------
@app.post("/contour-detection", status_code=status.HTTP_200_OK)
async def contour_detection(    
    imageSource: UploadFile = File(...),
    cropBox: str = Form(...),
    roiPoints: str = Form(...)
):
    try:
        # Read the image file
        file_bytes = await imageSource.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # Read ROI
        roi_points = json.loads(roiPoints)
        if not isinstance(roi_points, list):
            raise HTTPException(status_code=400, detail="ROI should be a list of points")
        
        crop_box_dict = json.loads(cropBox)

        # Run segmentation
        final_mask, contour_float, grab_cut_time, set_image_time, prediction_time, total_time = wound_segmentation(img, roi_points)
        if contour_float is None:
            print(f"[RESPONSE] grab_cut_time: {grab_cut_time:.3f} seconds")
            print(f"[RESPONSE] set_image_time: {set_image_time:.3f} seconds")
            print(f"[RESPONSE] prediction_time: {prediction_time:.3f} seconds")
            print(f"[RESPONSE] total_time: {total_time:.3f} seconds")
            logger.warning("No contour found in wound segmentation")
            return {
                'statusCode': 200,
                'result': {
                    "contourPx": [],
                    "cropBox": crop_box_dict,
                    "warning": "No wound contour detected. Please check ROI or image quality."
                }
            }

        final_contour = contour_float.reshape(-1, 2).tolist()
        
        print(f"[RESPONSE] grab_cut_time: {grab_cut_time:.3f} seconds")
        print(f"[RESPONSE] set_image_time: {set_image_time:.3f} seconds")
        print(f"[RESPONSE] prediction_time: {prediction_time:.3f} seconds")
        print(f"[RESPONSE] total_time: {total_time:.3f} seconds")

        return {
            'statusCode': 200,
            'result': {
                "contourPx": final_contour,
                "cropBox": crop_box_dict,
                "grabCutTime": round(grab_cut_time, 3),
                "setImageTime": round(set_image_time, 3),
                "predictionTime": round(prediction_time, 3),
                "totalTime": round(total_time, 3),
            }
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in contour-detection: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON in roi_points: {str(e)}")
    except Exception as e:
        logger.error(f"Error in contour-detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Run the app (for local debugging)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
