## FastCare Wound Analysis API
A FastAPI server for processing wound images and ROI (Region of Interest) data, running SAM (Segment Anything Model) to extract contours and compute wound composition in Red, Yellow, and Black (RYB) values.

## Features
Image Upload: Accepts wound images via POST request.

ROI Handling: Accepts ROI points in JSON format.

SAM Segmentation: Uses a preloaded SAM model for mask prediction.

Contour Extraction: Finds the maximum wound contour.

Color Composition: Calculates RYB wound composition percentages.

CORS Enabled: Allows cross-origin API calls.

Auto API Docs: Swagger & ReDoc available.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- SAM Model Checkpoint (sam_vit_b_01ec64.pth) stored locally in model/

### Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Server

```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`

## API Documentation

Once the server is running, you can access:

- **Interactive API Docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative API Docs (ReDoc)**: http://localhost:8000/redoc

## API Endpoint 
## POST /upload_img_and_roi
## Description
Processes a wound image and ROI points, returning the maximum contour coordinates and wound RYB composition.

## INPUT
| Parameter    | Type   | Location  | Description                  |
| ------------ | ------ | --------- | ---------------------------- |
| `image`      | file   | form-data | Wound image (JPEG/PNG)       |
| `roi_points` | string | form-data | JSON list of ROI coordinates |

## Development

### Project Structure
```
Fastcare/
├── main.py                # FastAPI application
├── sam_utils.py           # SAM mask, contour, and RYB utilities
├── requirements.txt       # Python dependencies
├── model/
│   └── sam_vit_b_01ec64.pth # SAM checkpoint
└── README.md
```

## Notes

The SAM model file is large (357 MB) and is not pushed to GitHub. Store it locally or use cloud storage in deployment.

For Render deployment, store the model in a mounted volume or download it at startup.

Ensure enough RAM/CPU in hosting for SAM inference.
