"""
engagEYE - Student Engagement Analysis Backend
Stack: OpenCV (video I/O + preprocessing) + MediaPipe (facial landmarks)
Flask API: POST /analyze  →  JSON with events, segments, score, AI summary
"""

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile, os, time, math
from datetime import timedelta
import anthropic

app = Flask(__name__)
CORS(app)

# MediaPipe FaceMesh setup 
mp_face_mesh = mp.solutions.face_mesh

# Eye EAR – 6 points each (vertical pairs + horizontal pair)
L_EAR_IDX = [362, 385, 387, 263, 373, 380]
R_EAR_IDX = [33,  160, 158, 133, 153, 144]

# Mouth MAR – 8 points (corners + top/bottom pairs)
MOUTH_IDX = [61, 291, 39, 181, 0, 17, 269, 405]

# Head-pose reference points (nose tip, eye corners, mouth corners, chin)
POSE_IDX = [1, 33, 263, 61, 291, 199]

# 3-D model coordinates matching POSE_IDX (generic face model, mm)
FACE_3D = np.array([
    [0.0,    0.0,    0.0   ],
    [-225.0, -170.0, -135.0],
    [225.0,  -170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0,  -150.0, -125.0],
    [0.0,    -330.0, -65.0 ],
], dtype=np.float64)

#Thresholds
EAR_THRESH         = 0.22
MAR_THRESH         = 0.60
YAW_THRESH         = 25.0
PITCH_THRESH       = 20.0
INACTIVITY_GAP_S   = 5
EYE_CONSEC_FRAMES  = 6
YAWN_CONSEC_FRAMES = 4
LOOK_AWAY_DEDUP_S  = 3

PENALTY = {
    "yawn":        12,
    "eyes_closed":  8,
    "look_away":   10,
    "inactivity":  15,
}

# Helpers

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calc_ear(landmarks, idx, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in idx]
    A = dist(pts[1], pts[5])
    B = dist(pts[2], pts[4])
    C = dist(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def calc_mar(landmarks, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in MOUTH_IDX]
    v1 = dist(pts[2], pts[6])
    v2 = dist(pts[3], pts[7])
    hz = dist(pts[0], pts[1])
    return (v1 + v2) / (2.0 * hz + 1e-6)

def calc_head_pose(landmarks, w, h):
    """Uses OpenCV solvePnP to estimate yaw and pitch in degrees."""
    pts2d = np.array(
        [[landmarks[i].x * w, landmarks[i].y * h] for i in POSE_IDX],
        dtype=np.float64,
    )
    focal = w
    cam = np.array([[focal, 0, w/2],
                    [0, focal, h/2],
                    [0,     0,   1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        FACE_3D, pts2d, cam, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return angles[1] * 360, angles[0] * 360  # yaw, pitch

def fmt_ts(seconds):
    td = str(timedelta(seconds=int(seconds)))
    return td[2:] if len(td) <= 7 else td

def ts_to_s(ts):
    parts = list(map(int, ts.split(":")))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]

def engagement_label(score):
    if score >= 65: return "HIGH"
    if score >= 40: return "MEDIUM"
    return "LOW"

# Core analysis

def analyze_video(video_path: str, segment_secs: int = 30) -> dict:
    # OpenCV opens and decodes the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_secs   = total_frames / fps
    sample_every = max(1, int(fps / 2))  # process ~2 frames/sec

    events, segments = [], []

    eye_consec     = 0
    yawn_consec    = 0
    last_face_frm  = 0
    last_inact_frm = -(INACTIVITY_GAP_S * fps + 1)
    seg_start      = 0.0
    seg_score      = 100

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        t = frame_idx / fps

        # Flush completed segment
        if t - seg_start >= segment_secs:
            segments.append({
                "start":      fmt_ts(seg_start),
                "end":        fmt_ts(t),
                "engagement": engagement_label(seg_score),
                "score":      max(0, seg_score),
            })
            seg_start = t
            seg_score = 100

        #OpenCV preprocessing
        # Resize so the longer edge = 640px (handles all webcam resolutions)
        h_orig, w_orig = frame.shape[:2]
        scale = 640 / max(w_orig, h_orig)
        if scale < 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        # Auto-gamma brightness correction for poorly-lit webcams
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_lum = gray.mean()
        if mean_lum < 80:
            gamma = math.log(128) / math.log(max(mean_lum, 1))
            lut   = np.array(
                [min(255, int((i / 255.0) ** (1.0 / gamma) * 255)) for i in range(256)],
                dtype=np.uint8,
            )
            frame = cv2.LUT(frame, lut)

        h, w = frame.shape[:2]
        # Convert BGR→RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #MediaPipe FaceMesh
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            gap = frame_idx - last_face_frm
            if (gap > INACTIVITY_GAP_S * fps and
                    frame_idx - last_inact_frm > INACTIVITY_GAP_S * fps):
                events.append({
                    "type":       "inactivity",
                    "timestamp":  fmt_ts(t),
                    "confidence": round(min(1.0, gap / (fps * 10)), 2),
                })
                seg_score      -= PENALTY["inactivity"]
                last_inact_frm  = frame_idx
            continue

        last_face_frm = frame_idx
        lm = result.multi_face_landmarks[0].landmark

        # Eye Aspect Ratio
        avg_ear = (calc_ear(lm, L_EAR_IDX, w, h) + calc_ear(lm, R_EAR_IDX, w, h)) / 2.0
        if avg_ear < EAR_THRESH:
            eye_consec += 1
            if eye_consec == EYE_CONSEC_FRAMES:
                events.append({
                    "type":       "eyes_closed",
                    "timestamp":  fmt_ts(t),
                    "confidence": round(min(1.0, 1.0 - avg_ear / EAR_THRESH), 2),
                })
                seg_score -= PENALTY["eyes_closed"]
        else:
            eye_consec = 0

        # Mouth Aspect Ratio (yawn)
        m_ratio = calc_mar(lm, w, h)
        if m_ratio > MAR_THRESH:
            yawn_consec += 1
            if yawn_consec == YAWN_CONSEC_FRAMES:
                events.append({
                    "type":       "yawn",
                    "timestamp":  fmt_ts(t),
                    "confidence": round(min(1.0, m_ratio / (MAR_THRESH * 1.4)), 2),
                })
                seg_score -= PENALTY["yawn"]
        else:
            yawn_consec = 0

        #  Head pose via OpenCV solvePnP 
        try:
            yaw, pitch = calc_head_pose(lm, w, h)
            if abs(yaw) > YAW_THRESH or pitch < -PITCH_THRESH:
                last_la = next(
                    (e for e in reversed(events) if e["type"] == "look_away"), None
                )
                if not last_la or t - ts_to_s(last_la["timestamp"]) > LOOK_AWAY_DEDUP_S:
                    events.append({
                        "type":       "look_away",
                        "timestamp":  fmt_ts(t),
                        "confidence": round(min(1.0, max(abs(yaw), abs(pitch)) / 45.0), 2),
                    })
                    seg_score -= PENALTY["look_away"]
        except Exception:
            pass

    # Flush final segment
    if seg_start < total_secs:
        segments.append({
            "start":      fmt_ts(seg_start),
            "end":        fmt_ts(total_secs),
            "engagement": engagement_label(seg_score),
            "score":      max(0, seg_score),
        })

    cap.release()
    face_mesh.close()

    overall = int(np.mean([s["score"] for s in segments])) if segments else 50

    return {
        "total_duration":   fmt_ts(total_secs),
        "total_events":     len(events),
        "engagement":       engagement_label(overall),
        "engagement_score": overall,
        "events":           events,
        "segments":         segments,
    }

# AI Summary 

def generate_summary(analysis: dict, filename: str) -> str:
    client = anthropic.Anthropic()
    counts = {}
    for e in analysis["events"]:
        counts[e["type"]] = counts.get(e["type"], 0) + 1

    worst = min(analysis["segments"], key=lambda s: s["score"], default=None)
    worst_str = (f"worst segment: {worst['start']}–{worst['end']} score={worst['score']}"
                 if worst else "no segments")

    prompt = f"""You are an educational engagement analyst reviewing webcam behavioral data.

Video: {filename}
Duration: {analysis['total_duration']}
Overall engagement: {analysis['engagement']} (score {analysis['engagement_score']}/100)
Event counts: {counts}
{worst_str}

Write a 3–4 sentence instructor report covering:
1. Overall engagement assessment
2. Most significant behavioral signals
3. Which part of the session was weakest and a possible reason
4. One concrete actionable recommendation

Direct, specific, plain prose — no bullet points."""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text

# Routes

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "engagEYE"})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "Send video as multipart/form-data field 'video'"}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        t0     = time.time()
        result = analyze_video(tmp_path)
        result["session_id"]        = f.filename
        result["processing_time_s"] = round(time.time() - t0, 2)

        try:
            result["ai_summary"] = generate_summary(result, f.filename)
        except Exception:
            result["ai_summary"] = (
                f"Engagement level: {result['engagement']} "
                f"(score {result['engagement_score']}/100). "
                f"{result['total_events']} behavioral events across {result['total_duration']}."
            )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
