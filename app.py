from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form, Header
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import requests
import numpy as np
import cv2
import os
from sam_utils import wound_segmentation
from contour_correction import refine_contour_opencv

# ==============================================
# ENV LOADING
# ==============================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using system environment variables only.")

# ==============================================
# CONFIG
# ==============================================
FASTIFY_LOGIN_URL = os.getenv("FASTIFY_LOGIN_URL", "https://test.fastcare.uk/api/v1/user/login")
FASTIFY_ORIGIN = os.getenv("FASTIFY_ORIGIN", "https://test.fastcare.uk")

print(f"Fastify Login URL: {FASTIFY_LOGIN_URL}")
print(f"Fastify Origin: {FASTIFY_ORIGIN}")

app = FastAPI(title="FastCare GPU API", version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FASTIFY_ORIGIN],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# ==============================================
# SIMPLE SESSION VALIDATION (no /facility call)
# ==============================================
def validate_fastcare_session(fastcare_session_cookie: str):
    """
    Basic session check — skips calling Fastify facility API.
    You can optionally compare it to a static session for testing.
    """
    if not fastcare_session_cookie:
        raise HTTPException(status_code=401, detail="Missing Fastcare session cookie")
    # Otherwise, assume valid for production use
    return True


# ==============================================
# LOGIN FUNCTION — still uses Fastify login endpoint
# ==============================================
def get_fastcare_session(email: str, password: str):
    """Get session cookie from Fastify login API."""
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": FASTIFY_ORIGIN,
        "User-Agent": "FastCare-GPU-API/1.2.0"
    }

    login_data = {"email": email, "password": password}

    try:
        resp = requests.post(FASTIFY_LOGIN_URL, json=login_data, headers=headers, timeout=10)

        if resp.status_code == 200:
            cookies = resp.cookies
            fastcare_cookie = cookies.get('fastcare_id')
            if fastcare_cookie:
                return fastcare_cookie
            else:
                raise HTTPException(status_code=502, detail="No session cookie received from Fastify")
        elif resp.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        else:
            raise HTTPException(status_code=502, detail=f"Login failed: {resp.status_code}")

    except requests.exceptions.RequestException:
        raise HTTPException(status_code=502, detail="Fastify login endpoint unreachable")


# ==============================================
# HEALTH CHECK
# ==============================================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "fastify_login_url": FASTIFY_LOGIN_URL,
        "fastify_origin": FASTIFY_ORIGIN,
        "version": "1.2.0"
    }


# ==============================================
# LOGIN ENDPOINT
# ==============================================
@app.post("/login")
async def login_endpoint(email: str, password: str):
    try:
        session_cookie = get_fastcare_session(email, password)
        return {
            "status": "success",
            "session_cookie": session_cookie,
            "message": "Use this cookie in X-Fastcare-Session header"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


# ==============================================
# CONTOUR DETECTION
# ==============================================
@app.post("/contour-detection", status_code=status.HTTP_200_OK)
async def contour_detection(
    imageSource: UploadFile = File(...),
    cropBox: str = Form(...),
    roiPoints: str = Form(...),
    x_fastcare_session: str = Header(..., alias="X-Fastcare-Session"),
):
    validate_fastcare_session(x_fastcare_session)

    file_bytes = await imageSource.read()
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    roi_points = json.loads(roiPoints)
    crop_box_dict = json.loads(cropBox)

    final_mask, contour_float, set_image_time, prediction_time, total_time = wound_segmentation(img, roi_points)
    if contour_float is None:
        raise HTTPException(status_code=400, detail="No contour found")

    final_contour = contour_float.reshape(-1, 2).tolist()

    return {
        "statusCode": 200,
        "result": {
            "contourPx": final_contour,
            "cropBox": crop_box_dict,
            "set_image_time": round(set_image_time, 3),
            "prediction_time": round(prediction_time, 3),
            "totalTime": round(total_time, 3),
        },
    }


# ==============================================
# CONTOUR CORRECTION
# ==============================================
@app.post("/contour-correction", status_code=status.HTTP_200_OK)
async def contour_correction(
    imageSource: UploadFile = File(...),
    detected_pts: str = Form(...),
    adjusted_pts: str = Form(...),
    x_fastcare_session: str = Header(..., alias="X-Fastcare-Session"),
):
    validate_fastcare_session(x_fastcare_session)

    file_bytes = await imageSource.read()
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    orig_pts = np.array(json.loads(detected_pts), dtype=np.int32)
    user_pts = np.array(json.loads(adjusted_pts), dtype=np.int32)
    refined_contour = refine_contour_opencv(img, orig_pts, user_pts)

    return {
        "statusCode": 200,
        "result": {"refinedContour": refined_contour.tolist()},
    }


# ==============================================
# RUN
# ==============================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
