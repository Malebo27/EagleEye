from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import cv2
import threading
import time
import base64
import numpy as np
import io

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import threading
import base64
import time
import numpy as np


app = FastAPI(title="Proctoring API - Live Person Detection with Base64 Frames")

# -------------------------
# Load Haar Cascade for person detection
# -------------------------
cascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
person_cascade = cv2.CascadeClassifier(cascade_path)
if person_cascade.empty():
    raise RuntimeError(f"Failed to load cascade classifier from {cascade_path}")

# -------------------------
# Shared state for JSON endpoint
# -------------------------
live_detection_data = {
    "persons_detected": 0,
    "coordinates": [],
    "frame_b64": None
}

lock = threading.Lock()

# -------------------------
# Root and ping endpoints
# -------------------------
@app.get("/")
async def root():
    return {"detail": "Proctoring API is running! Visit /webcam for live video or /json for live detection data."}

@app.get("/ping")
async def ping():
    return {"detail": "pong"}

# -------------------------
# Generator function for streaming frames
# -------------------------
def webcam_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            persons = person_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )

            coordinates = []
            for (x, y, w, h) in persons:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                coordinates.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})

            # Encode frame as JPEG and then base64
            _, buffer = cv2.imencode(".jpg", frame)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            with lock:
                live_detection_data["persons_detected"] = len(coordinates)
                live_detection_data["coordinates"] = coordinates
                live_detection_data["frame_b64"] = frame_b64

            time.sleep(0.03)
    finally:
        cap.release()

# Start webcam loop in a background thread
threading.Thread(target=webcam_loop, daemon=True).start()

# -------------------------
# Real-time JSON endpoint with optional base64 frame
# -------------------------
@app.get("/json")
async def live_json(include_frame: bool = Query(False, description="Include base64-encoded current frame")):
    with lock:
        data_copy = {
            "persons_detected": live_detection_data["persons_detected"],
            "coordinates": live_detection_data["coordinates"]
        }
        if include_frame and live_detection_data["frame_b64"]:
            data_copy["frame_b64"] = live_detection_data["frame_b64"]
    return JSONResponse(content=data_copy)

# -------------------------
# MJPEG video streaming (optional)
# -------------------------
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            persons = person_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )

            for (x, y, w, h) in persons:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.03)
    finally:
        cap.release()

@app.get("/webcam")
async def webcam_feed():
    return StreamingResponse(gen_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')






