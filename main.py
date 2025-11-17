from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os

app = FastAPI(title="NexaAI Proctoring App")

# -----------------------------
# Load Haar Cascade dynamically
# -----------------------------
try:
    cascade_path = os.path.join(os.path.dirname(__file__), "XML", "haarcascade_fullbody.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise ValueError("Failed to load Haar Cascade XML file!")
except Exception as e:
    raise RuntimeError(f"Error loading cascade: {e}")

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
async def root():
    return {"message": "NexaAI Proctoring API is running."}

# -----------------------------
# Test cascade endpoint
# -----------------------------
@app.get("/test_cascade")
async def test_cascade():
    if cascade.empty():
        raise HTTPException(status_code=500, detail="Cascade not loaded")
    return JSONResponse(content={"status": "Cascade loaded successfully", "cascade_path": cascade_path})

# -----------------------------
# Person detection endpoint
# -----------------------------
@app.post("/detect_person")
async def detect_person(file: UploadFile = File(...)):
    # Ensure uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Read image file into numpy array
    file_bytes = await file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    # Convert to grayscale for Haar Cascade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect people
    bodies = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    # Format detection results
    detections = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for (x, y, w, h) in bodies]

    return JSONResponse(content={
        "filename": file.filename,
        "detections": detections,
        "num_people_detected": len(detections)
    })

# -----------------------------
# Run app if executed directly
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Railway assigns $PORT
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)








