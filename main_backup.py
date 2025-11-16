from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np

app = FastAPI(title="Proctoring API - No Face Detection")

# -------------------------
# Root endpoint
# -------------------------
@app.get("/")
async def root():
    return {"detail": "Proctoring API is running! Use /api/proctoring/detect to POST images."}

# -------------------------
# Ping endpoint
# -------------------------
@app.get("/ping")
async def ping():
    return {"detail": "pong"}

# -------------------------
# Proctoring detect endpoint
# -------------------------
@app.post("/api/proctoring/detect")
async def detect_face(frame: UploadFile = File(...)):
    try:
        image_bytes = await frame.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse(status_code=400, content={"detail": "Invalid image file."})

        # Optional simple checks
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 10:
            return {"detail": "Camera might be blocked or image too dark"}

        return {"detail": "Image received, face detection skipped"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"error: {str(e)}"})





