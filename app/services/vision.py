# app/services/vision.py
import cv2
import numpy as np
import mediapipe as mp
import math
from typing import Dict, Any

mp_face_mesh = mp.solutions.face_mesh

FACE_MESH_STATIC_IMAGE_MODE = True
MIN_DETECTION_CONFIDENCE = 0.5

# Landmarks indices (MediaPipe Face Mesh)
# select a few landmarks for head-pose and mouth
_LM = {
    "nose_tip": 1,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
    "upper_lip": 13,
    "lower_lip": 14,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "left_iris": 468,   # iris landmarks exist in mediapipe's mesh
    "right_iris": 473,
}

def _to_np(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image bytes")
    return img

def analyze_image_bytes(image_bytes: bytes) -> Dict[str, Any]:
    """
    Analyze a single image (frame) and return detections and scores.
    """
    img = _to_np(image_bytes)
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = {}

    with mp_face_mesh.FaceMesh(
        static_image_mode=FACE_MESH_STATIC_IMAGE_MODE,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE
    ) as face_mesh:
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            # No face found
            return {
                "face_detected": False,
                "gaze_direction": None,
                "head_pose": None,
                "mouth_open": None,
                "phone_detected": False,
                "risk_score": 0.0,
                "alerts": ["no_face_detected"]
            }

        # Use first face
        lm = res.multi_face_landmarks[0]
        pts = []
        for i, l in enumerate(lm.landmark):
            pts.append((int(l.x * w), int(l.y * h)))
        # helper to get landmark by name safely
        def L(name):
            idx = _LM.get(name)
            if idx is None or idx >= len(pts):
                return None
            return pts[idx]

        # Face detection
        results["face_detected"] = True

        # --- Mouth open detection ---
        upper = L("upper_lip")
        lower = L("lower_lip")
        if upper and lower:
            lip_dist = math.hypot(upper[0] - lower[0], upper[1] - lower[1])
            # normalize by face size (distance between eye outers)
            left_eye = L("left_eye_outer")
            right_eye = L("right_eye_outer")
            if left_eye and right_eye:
                eye_dist = math.hypot(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1])
                mouth_ratio = lip_dist / (eye_dist + 1e-6)
            else:
                mouth_ratio = lip_dist / (h + 1e-6)
            # threshold: tuned for typical webcam frames
            mouth_open = mouth_ratio > 0.035
            results["mouth_open"] = {"mouth_ratio": round(mouth_ratio, 4), "mouth_open": bool(mouth_open)}
        else:
            results["mouth_open"] = None

        # --- Gaze estimation (very approximate) ---
        # Use iris center relative to eye bounding box
        def eye_center_and_box(left_inner, left_outer, iris_idx):
            if left_inner is None or left_outer is None or iris_idx is None:
                return None
            x1, y1 = left_inner
            x2, y2 = left_outer
            ex_min, ex_max = min(x1, x2), max(x1, x2)
            ey_min, ey_max = min(y1, y2), max(y1, y2)
            box_w = ex_max - ex_min
            box_h = ey_max - ey_min
            return {"box": (ex_min, ey_min, box_w, box_h), "center": (int((x1 + x2) / 2), int((y1 + y2)/2))}

        left_eye_box = eye_center_and_box(L("left_eye_inner"), L("left_eye_outer"), L("left_iris"))
        right_eye_box = eye_center_and_box(L("right_eye_inner"), L("right_eye_outer"), L("right_iris"))
        gaze = {"left": None, "right": None, "combined": None}
        def iris_offset(eye_box, iris):
            if eye_box is None or iris is None:
                return None
            ex, ey, ew, eh = eye_box["box"]
            ix, iy = iris
            # normalized offset: -1 (left/top) .. +1 (right/bottom)
            nx = ((ix - ex) / (ew + 1e-6) - 0.5) * 2.0
            ny = ((iy - ey) / (eh + 1e-6) - 0.5) * 2.0
            return (nx, ny)
        if left_eye_box and L("left_iris"):
            gaze["left"] = iris_offset(left_eye_box, L("left_iris"))
        if right_eye_box and L("right_iris"):
            gaze["right"] = iris_offset(right_eye_box, L("right_iris"))

        # simple combined logic
        if gaze["left"] and gaze["right"]:
            combined_x = (gaze["left"][0] + gaze["right"][0]) / 2.0
            combined_y = (gaze["left"][1] + gaze["right"][1]) / 2.0
            # thresholds (tunable)
            horiz = "center"
            if combined_x < -0.25:
                horiz = "left"
            elif combined_x > 0.25:
                horiz = "right"
            vert = "center"
            if combined_y < -0.25:
                vert = "up"
            elif combined_y > 0.25:
                vert = "down"
            results["gaze_direction"] = {"x": round(combined_x, 3), "y": round(combined_y, 3), "h": horiz, "v": vert}
        else:
            results["gaze_direction"] = None

        # --- Head pose estimation (solvePnP) ---
        # Choose 2D-3D correspondences:
        # 3D model points approximate in mm for generic face
        image_points = []
        model_points = []
        # Use nose tip, left eye outer, right eye outer, left mouth corner, right mouth corner
        correspondences = [
            ("nose_tip", (0.0, 0.0, 0.0)),
            ("left_eye_outer", (-30.0, 30.0, -30.0)),
            ("right_eye_outer", (30.0, 30.0, -30.0)),
            ("left_mouth", (-25.0, -30.0, -20.0)),
            ("right_mouth", (25.0, -30.0, -20.0)),
        ]
        for name, model_pt in correspondences:
            p = L(name)
            if p is not None:
                image_points.append((float(p[0]), float(p[1])))
                model_points.append(model_pt)

        head_pose = None
        if len(image_points) >= 4:
            image_points_np = np.array(image_points, dtype="double")
            model_points_np = np.array(model_points, dtype="double")
            # camera matrix approximation
            focal_length = w
            center = (w/2, h/2)
            camera_matrix = np.array([[focal_length, 0, center[0]],
                                      [0, focal_length, center[1]],
                                      [0, 0, 1]], dtype="double")
            dist_coeffs = np.zeros((4,1))  # assume no lens distortion
            success, rotation_vector, translation_vector = cv2.solvePnP(model_points_np, image_points_np, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                # convert to euler angles
                rmat, _ = cv2.Rodrigues(rotation_vector)
                sy = math.sqrt(rmat[0,0]*rmat[0,0] + rmat[1,0]*rmat[1,0])
                singular = sy < 1e-6
                if not singular:
                    x = math.atan2(rmat[2,1], rmat[2,2])
                    y = math.atan2(-rmat[2,0], sy)
                    z = math.atan2(rmat[1,0], rmat[0,0])
                else:
                    x = math.atan2(-rmat[1,2], rmat[1,1])
                    y = math.atan2(-rmat[2,0], sy)
                    z = 0
                # convert rad->deg
                pitch = math.degrees(x)
                yaw = math.degrees(y)
                roll = math.degrees(z)
                head_pose = {"pitch": round(pitch, 2), "yaw": round(yaw, 2), "roll": round(roll, 2)}
        results["head_pose"] = head_pose

        # --- Phone detection (naive heuristic) ---
        # Convert to grayscale and look for large rectangular contours with phone-like aspect ratio
        phone_detected = False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edged = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_area = w * h
        possible_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.01 * img_area:  # skip small
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, bw, bh = cv2.boundingRect(approx)
                ar = bw / (bh + 1e-6)
                # phones are often tall rectangles (~0.4-0.7 aspect ratio depending orientation)
                if (0.35 <= ar <= 0.8) or (1.2 <= ar <= 3.0):  # accept both portrait and landscape heuristics
                    # ensure box not covering entire frame (not a board)
                    if area < 0.5 * img_area:
                        possible_boxes.append({"x": x, "y": y, "w": bw, "h": bh, "area": area, "ar": round(ar,2)})
        if possible_boxes:
            phone_detected = True
        results["phone_detected"] = {"detected": bool(phone_detected), "boxes": possible_boxes[:5]}

        # --- Risk scoring (very simple combination of heuristics) ---
        score = 0.0
        alerts = []
        # base: face detected -> low baseline
        if not results["face_detected"]:
            score += 0.9
            alerts.append("no_face")
        else:
            # mouth open increases risk
            if results.get("mouth_open") and results["mouth_open"]["mouth_open"]:
                score += 0.25
                alerts.append("mouth_open")
            # if gaze off-screen (left/right/up/down not center)
            g = results.get("gaze_direction")
            if g:
                if g["h"] != "center":
                    score += 0.25
                    alerts.append("gaze_away")
                if g["v"] != "center":
                    score += 0.1
                    alerts.append("gaze_vertical")
            # head pose large yaw or pitch
            hp = results.get("head_pose")
            if hp:
                # yaw threshold (~looking sideways)
                if abs(hp["yaw"]) > 20:
                    score += 0.3
                    alerts.append("head_turned")
                if abs(hp["pitch"]) > 20:
                    score += 0.2
                    alerts.append("head_tilted")
            # phone detection
            if results.get("phone_detected") and results["phone_detected"]["detected"]:
                score += 0.6
                alerts.append("phone_visible")

        # clamp score to 0..1
        score = min(1.0, score)
        results["risk_score"] = round(score, 3)
        results["alerts"] = list(dict.fromkeys(alerts))  # dedupe
        return results
