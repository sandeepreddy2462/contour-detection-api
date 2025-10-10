from fastapi import FastAPI, HTTPException, status, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn
import json
import requests
import os
import torch
import numpy as np
import cv2
import io
from PIL import Image
import base64
from sam_utils import wound_segmentation
from contour_correction import refine_contour_opencv
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
# API Endpoint
# -----------------------------
@app.post("/contour-detection", status_code=status.HTTP_200_OK)
async def counter_detection(    
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

        #Read ROI
        roi_points = json.loads(roiPoints)
        if not isinstance(roi_points, list):
            raise HTTPException(status_code=400, detail="ROI should be a list of points")
        
        crop_box_dict = json.loads(cropBox)

        # Use the new robust segmentation function
        final_mask, contour_float, set_image_time, prediction_time, total_time = wound_segmentation(img, roi_points)

        if contour_float is None:
            raise HTTPException(status_code=400,detail="No contour Found")
        final_contour = contour_float.reshape(-1,2).tolist()

        #color analysis 
        # color_result = analyze_color(img, final_mask)

        return {
            'statusCode': 200,
            'result':{
                "contourPx": final_contour,
                "cropBox" : crop_box_dict,
                # "colorComposition": color_result,
                "set_image_time":round(set_image_time,3),
                "prediction_time":round(prediction_time,3),
                "totalTime": round(total_time, 3)
            } 
        }
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in roi_points: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")



#-----------------------------
# Contour - Correction Endpoint
#-----------------------------
@app.post("/contour-correction", status_code=status.HTTP_200_OK)
async def contour_correction(
    imageSource: UploadFile = File(...),
    detected_pts: str = Form(...),
    adjusted_pts: str = Form(...)
):
    try:
        # --- Step 1: Read the image ---
        file_bytes = await imageSource.read()
        img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        # --- Step 2: Parse contour points ---
        orig_pts = np.array(json.loads(detected_pts), dtype=np.int32)
        user_pts = np.array(json.loads(adjusted_pts), dtype=np.int32)

        # --- Step 3: Refine contour ---
        from contour_correction import refine_contour_opencv  # ensure your function is imported
        refined_contour = refine_contour_opencv(img, orig_pts, user_pts)

        # Convert to list for JSON response
        refined_list = refined_contour.tolist()

        return {
            "statusCode": 200,
            "result": {
                "refinedContour": refined_list
            }
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in points: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)