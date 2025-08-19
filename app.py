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
from sam_utils import get_max_contour, predictor
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
# Request Models
# -----------------------------
class ROIbbox(BaseModel):
    x: float
    y: float
    w: float
    h: float

class ImageData(BaseModel):
    data: str
    format: str
    width: int
    height: int

class RequestPayload(BaseModel):
    image: ImageData
    roiPx: List[List[float]]   # 2D list instead of objects
    roiBBoxPx: ROIbbox

# ---------------------------
# Response Model
# ---------------------------
# class ProcessResponse(BaseModel):
#     max_contour: List[List[float]]


# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/contour-detection", status_code=status.HTTP_200_OK)
async def counter_detection(req: RequestPayload):
    try:
        # Read the image file
        image_data = base64.b64decode(req.image.data)
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_height, image_width = image_rgb.shape[:2]

        roi_points = req.roiPx

        # If roiBBoxPx is provided, crop image (optional)
        if req.roiBBoxPx:
            x, y, bw, bh = (
                int(req.roiBBoxPx.x),
                int(req.roiBBoxPx.y),
                int(req.roiBBoxPx.w),
                int(req.roiBBoxPx.h),
            )
            cropped_img = image_rgb[y:y+bh, x:x+bw].copy()
            predictor.set_image(cropped_img)
            image_width, image_height = bw, bh
        else:
            predictor.set_image(image_rgb)
            image_height, image_width = image_rgb.shape[:2]
            x, y, bw, bh = 0, 0, image_width, image_height


        max_contour = get_max_contour(image_width, image_height, roi_points, req.roiBBoxPx).reshape(-1,2).astype(float).tolist()
        if max_contour is None:
            raise HTTPException(status_code=400,detail="No contour Found")

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'result':{
                "contourPx":max_contour
            } 
        }
            
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in roi_points: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)