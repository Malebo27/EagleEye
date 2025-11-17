from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io
import os

app = FastAPI(title="NexaAI Full-Body Detection API")

# Enable CORS for testing from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Haar Cascade for full body detection
CASCADE_PATH = os.path.join("XML", "haarcascade_fullbody.xml")
if not os.path.exists(CASCADE_PATH):
    raise FileNotFoundError(f"Haar cascade file not found at {CASCADE_PATH}")

body_cascade = cv2.CascadeClassifier(CASCADE_PATH)

@app.get("/")
async def root():
    return {"message": "NexaAI Full-Body Detection API. Use /detect_person to upload an image."}


@app.post("/detect_person")
async def detect_person(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    # Read uploaded image
    image_bytes = await file.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to read image.")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect bodies
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    # Draw rectangles
    for (x, y, w, h) in bodies:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Prepare annotated image for streaming
    _, img_encoded = cv2.imencode('.jpg', img)
    annotated_image_bytes = io.BytesIO(img_encoded.tobytes())

    # Prepare bounding boxes for JSON
    boxes = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in bodies]

    return JSONResponse(
        content={
            "bounding_boxes": boxes,
            "message": f"{len(boxes)} full bodies detected",
            "annotated_image_url": "Use the /detect_person_image endpoint to get the JPEG"
        }
    )


@app.post("/detect_person_image")
async def detect_person_image(file: UploadFile = File(...)):
    """
    Returns the annotated image (JPEG) for the uploaded image.
    """
    # Validate file type
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    np_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to read image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in bodies:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    _, img_encoded = cv2.imencode('.jpg', img)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")









