# AR Viewer Backend

Backend service for 3D product reconstruction with AR visualization.

## Features
- Object detection using YOLOv8
- Depth estimation using DPT
- 3D reconstruction using Open3D
- GLB model generation for AR viewing

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
```

## API Endpoints
- POST `/process`: Process images and generate 3D model
- GET `/health`: Health check endpoint

## Environment Variables
- `PYTHON_VERSION`: Python version (default: 3.8.18) 