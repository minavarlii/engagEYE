"""
engagEYE - Student Engagement Analysis Backend
Stack: OpenCV (video I/O + preprocessing) + MediaPipe (facial landmarks + iris)

Detectors:
  Negative: yawn, eyes_closed, look_away (iris + head pose), inactivity
  Positive: smile, nodding
"""

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile, os, time, math, base64
from datetime import timedelta
import anthropic

app = Flask(__name__)
CORS(app)

mp_face_mesh = mp.solutions.face_mesh

L_EAR_IDX = [362, 385, 387, 263, 373, 380]
R_EAR_IDX = [33,  160, 158, 133, 153, 144]
MOUTH_IDX = [61, 291, 39, 181, 0, 17, 269, 405]
POSE_IDX  = [1, 33, 263, 61, 291, 199]

# Iris landmarks
L_IRIS_CENTER = 468
R_IRIS_CENTER = 473
L_EYE_INNER = 133
L_EYE_OUTER = 33
R_EYE_INNER = 362
R_EYE_OUTER = 263
L_EYE_TOP = 159
L_EYE_BOT = 145
R_EYE_TOP = 386
R_EYE_BOT = 374

FACE_3D = np.array([
    [0.0,    0.0,    0.0   ],
    [-225.0, -170.0, -135.0],
    [225.0,  -170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0,  -150.0, -125.0],
    [0.0,    -330.0, -65.0 ],
], dtype=np.float64)

# Thresholds - increased for more tolerance
EAR_THRESH          = 0.22
MAR_THRESH          = 0.60
YAW_THRESH          = 35.0   # increased from 28
PITCH_THRESH        = 30.0   # increased from 22
INACTIVITY_GAP_S    = 5
EYE_CONSEC_FRAMES   = 6
YAWN_CONSEC_FRAMES  = 4
LOOK_AWAY_DEDUP_S   = 3
SMILE_THRESH        = 0.28
SMILE_CONSEC_FRAMES = 3
NOD_PITCH_DELTA     = 10.0
NOD_DEDUP_S         = 4
CALIBRATION_N       = 10
GAZE_H_THRESH       = 0.40   # increased from 0.25
GAZE_V_THRESH       = 0.35   # increased from 0.25
GAZE_CONSEC_FRAMES  = 3

PENALTY = {
    "yawn":        12,
    "eyes_closed":  8,
    "look_away":   10,
    "inactivity":  15,
}
BONUS = {
    "smile":   5,
    "nodding": 8,
}

THUMBNAIL_W = 160


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


def calc_smile_ratio(landmarks, w, h):
    left_corner  = (landmarks[61].x * w,  landmarks[61].y * h)
    right_corner = (landmarks[291].x * w, landmarks[291].y * h)
    mouth_width  = dist(left_corner, right_corner)
    left_cheek   = (landmarks[234].x * w, landmarks[234].y * h)
    right_cheek  = (landmarks[454].x * w, landmarks[454].y * h)
    face_width   = dist(left_cheek, right_cheek) + 1e-6
    mar = calc_mar(landmarks, w, h)
    if mar > MAR_THRESH * 0.8:
        return 0.0

    left_corner_y  = landmarks[61].y
    right_corner_y = landmarks[291].y
    nose_y         = landmarks[1].y
    chin_y         = landmarks[152].y
    face_h         = abs(chin_y - nose_y) + 1e-6
    avg_corner_y   = (left_corner_y + right_corner_y) / 2.0
    corner_raise   = (chin_y - avg_corner_y) / face_h

    smile_ratio = mouth_width / face_width
    if corner_raise > 0.6:
        smile_ratio *= 1.3

    return smile_ratio


def calc_gaze_ratio(landmarks, w, h):
    try:
        l_iris  = (landmarks[L_IRIS_CENTER].x * w, landmarks[L_IRIS_CENTER].y * h)
        l_inner = (landmarks[L_EYE_INNER].x * w,   landmarks[L_EYE_INNER].y * h)
        l_outer = (landmarks[L_EYE_OUTER].x * w,   landmarks[L_EYE_OUTER].y * h)
        l_top   = (landmarks[L_EYE_TOP].x * w,     landmarks[L_EYE_TOP].y * h)
        l_bot   = (landmarks[L_EYE_BOT].x * w,     landmarks[L_EYE_BOT].y * h)
        l_eye_w = dist(l_inner, l_outer) + 1e-6
        l_eye_h = dist(l_top,   l_bot)   + 1e-6
        l_h = (l_iris[0] - l_outer[0]) / l_eye_w
        l_v = (l_iris[1] - l_top[1])   / l_eye_h

        r_iris  = (landmarks[R_IRIS_CENTER].x * w, landmarks[R_IRIS_CENTER].y * h)
        r_inner = (landmarks[R_EYE_INNER].x * w,   landmarks[R_EYE_INNER].y * h)
        r_outer = (landmarks[R_EYE_OUTER].x * w,   landmarks[R_EYE_OUTER].y * h)
        r_top   = (landmarks[R_EYE_TOP].x * w,     landmarks[R_EYE_TOP].y * h)
        r_bot   = (landmarks[R_EYE_BOT].x * w,     landmarks[R_EYE_BOT].y * h)
        r_eye_w = dist(r_inner, r_outer) + 1e-6
        r_eye_h = dist(r_top,   r_bot)   + 1e-6
        r_h = (r_iris[0] - r_outer[0]) / r_eye_w
        r_v = (r_iris[1] - r_top[1])   / r_eye_h

        return (l_h + r_h) / 2.0, (l_v + r_v) / 2.0
    except Exception:
        return 0.5, 0.5


def calc_head_pose(landmarks, w, h):
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
    return angles[1] * 360, angles[0] * 360


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


def frame_to_base64(frame):
    h, w = frame.shape[:2]
    scale = THUMBNAIL_W / w
    thumb = cv2.resize(frame, (THUMBNAIL_W, int(h * scale)))
    _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def log_look_away(events, t, confidence, frame, seg_score):
    last_la = next((e for e in reversed(events) if e["type"] == "look_away"), None)
    if not last_la or t - ts_to_s(last_la["timestamp"]) > LOOK_AWAY_DEDUP_S:
        events.append({
            "type":       "look_away",
            "timestamp":  fmt_ts(t),
            "confidence": confidence,
            "thumbnail":  frame_to_base64(frame),
        })
        return seg_score - PENALTY["look_away"]
    return seg_score


def analyze_video(video_path: str, segment_secs: int = 30) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_secs   = total_frames / fps
    sample_every = max(1, int(fps / 2))

    events, segments = [], []

    eye_consec      = 0
    yawn_consec     = 0
    smile_consec    = 0
    gaze_consec     = 0
    last_face_frm   = 0
    last_inact_frm  = -(INACTIVITY_GAP_S * fps + 1)
    seg_start       = 0.0
    seg_score       = 100
    pitch_history   = []
    last_nod_t      = -NOD_DEDUP_S - 1

    baseline_h_gaze   = None
    baseline_v_gaze   = None
    gaze_calib        = []
    baseline_yaw      = None
    baseline_pitch    = None
    pose_calib        = []

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

        if t - seg_start >= segment_secs:
            segments.append({
                "start":      fmt_ts(seg_start),
                "end":        fmt_ts(t),
                "engagement": engagement_label(seg_score),
                "score":      max(0, min(100, seg_score)),
            })
            seg_start = t
            seg_score = 100

        h_orig, w_orig = frame.shape[:2]
        scale = 640 / max(w_orig, h_orig)
        if scale < 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            gap = frame_idx - last_face_frm
            if (gap > INACTIVITY_GAP_S * fps and
                    frame_idx - last_inact_frm > INACTIVITY_GAP_S * fps):
                events.append({
                    "type":       "inactivity",
                    "timestamp":  fmt_ts(t),
                    "confidence": round(min(1.0, gap / (fps * 10)), 2),
                    "thumbnail":  frame_to_base64(frame),
                })
                seg_score      -= PENALTY["inactivity"]
                last_inact_frm  = frame_idx
            continue

        last_face_frm = frame_idx
        lm = result.multi_face_landmarks[0].landmark

        # EAR
        avg_ear = (calc_ear(lm, L_EAR_IDX, w, h) + calc_ear(lm, R_EAR_IDX, w, h)) / 2.0
        if avg_ear < EAR_THRESH:
            eye_consec += 1
            if eye_consec == EYE_CONSEC_FRAMES:
                events.append({
                    "type":       "eyes_closed",
                    "timestamp":  fmt_ts(t),
                    "confidence": round(min(1.0, 1.0 - avg_ear / EAR_THRESH), 2),
                    "thumbnail":  frame_to_base64(frame),
                })
                seg_score -= PENALTY["eyes_closed"]
        else:
            eye_consec = 0

        # MAR
        m_ratio = calc_mar(lm, w, h)
        if m_ratio > MAR_THRESH:
            yawn_consec += 1
            if yawn_consec == YAWN_CONSEC_FRAMES:
                events.append({
                    "type":       "yawn",
                    "timestamp":  fmt_ts(t),
                    "confidence": round(min(1.0, m_ratio / (MAR_THRESH * 1.4)), 2),
                    "thumbnail":  frame_to_base64(frame),
                })
                seg_score -= PENALTY["yawn"]
        else:
            yawn_consec = 0

        # Smile
        smile_ratio = calc_smile_ratio(lm, w, h)
        if smile_ratio > SMILE_THRESH:
            smile_consec += 1
            if smile_consec == SMILE_CONSEC_FRAMES:
                last_sm = next(
                    (e for e in reversed(events) if e["type"] == "smile"), None
                )
                if not last_sm or t - ts_to_s(last_sm["timestamp"]) > 4:
                    events.append({
                        "type":       "smile",
                        "timestamp":  fmt_ts(t),
                        "confidence": round(min(1.0, smile_ratio / (SMILE_THRESH * 1.5)), 2),
                        "thumbnail":  frame_to_base64(frame),
                    })
                    seg_score += BONUS["smile"]
        else:
            smile_consec = 0

        # Iris gaze
        try:
            h_gaze, v_gaze = calc_gaze_ratio(lm, w, h)

            if baseline_h_gaze is None:
                gaze_calib.append((h_gaze, v_gaze))
                if len(gaze_calib) >= CALIBRATION_N:
                    baseline_h_gaze = float(np.mean([f[0] for f in gaze_calib]))
                    baseline_v_gaze = float(np.mean([f[1] for f in gaze_calib]))
            else:
                rel_h = h_gaze - baseline_h_gaze
                rel_v = v_gaze - baseline_v_gaze

                if abs(rel_h) > GAZE_H_THRESH or abs(rel_v) > GAZE_V_THRESH:
                    gaze_consec += 1
                    if gaze_consec == GAZE_CONSEC_FRAMES:
                        conf = round(min(1.0, max(abs(rel_h), abs(rel_v)) / 0.4), 2)
                        seg_score = log_look_away(events, t, conf, frame, seg_score)
                else:
                    gaze_consec = 0
        except Exception:
            pass

        # Head pose
        try:
            yaw, pitch = calc_head_pose(lm, w, h)

            if baseline_yaw is None:
                pose_calib.append((yaw, pitch))
                if len(pose_calib) >= CALIBRATION_N:
                    baseline_yaw   = float(np.mean([f[0] for f in pose_calib]))
                    baseline_pitch = float(np.mean([f[1] for f in pose_calib]))
            else:
                rel_yaw   = yaw   - baseline_yaw
                rel_pitch = pitch - baseline_pitch

                if abs(rel_yaw) > YAW_THRESH or rel_pitch < -PITCH_THRESH:
                    conf = round(min(1.0, max(abs(rel_yaw), abs(rel_pitch)) / 45.0), 2)
                    seg_score = log_look_away(events, t, conf, frame, seg_score)

                if abs(rel_yaw) < 15:
                    pitch_history.append(rel_pitch)
                    if len(pitch_history) > 8:
                        pitch_history.pop(0)

                    if len(pitch_history) >= 5:
                        p = pitch_history
                        went_down = min(p[-3:]) < min(p[:3]) - NOD_PITCH_DELTA
                        came_back = max(p[-2:]) > min(p[-3:]) + NOD_PITCH_DELTA * 0.5
                        if went_down and came_back and t - last_nod_t > NOD_DEDUP_S:
                            events.append({
                                "type":       "nodding",
                                "timestamp":  fmt_ts(t),
                                "confidence": round(min(1.0, abs(min(p) - max(p)) / 15.0), 2),
                                "thumbnail":  frame_to_base64(frame),
                            })
                            seg_score += BONUS["nodding"]
                            last_nod_t = t
                else:
                    pitch_history = []

        except Exception:
            pass

    if seg_start < total_secs:
        segments.append({
            "start":      fmt_ts(seg_start),
            "end":        fmt_ts(total_secs),
            "engagement": engagement_label(seg_score),
            "score":      max(0, min(100, seg_score)),
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


def generate_summary(analysis: dict, filename: str) -> str:
    client = anthropic.Anthropic()
    counts = {}
    for e in analysis["events"]:
        counts[e["type"]] = counts.get(e["type"], 0) + 1

    worst = min(analysis["segments"], key=lambda s: s["score"], default=None)
    best  = max(analysis["segments"], key=lambda s: s["score"], default=None)
    worst_str = (f"{worst['start']}–{worst['end']} (score {worst['score']})" if worst else "N/A")
    best_str  = (f"{best['start']}–{best['end']} (score {best['score']})"  if best  else "N/A")

    prompt = f"""You are an expert educational engagement analyst. Analyze this student engagement data.

Video: {filename}
Duration: {analysis['total_duration']}
Overall engagement: {analysis['engagement']} (score {analysis['engagement_score']}/100)
Events: {counts}
Best segment: {best_str}
Worst segment: {worst_str}

Positive signals: smile, nodding. Negative: yawn, eyes_closed, look_away, inactivity.
Look away uses iris gaze tracking + head pose combined.

Write EXACTLY:

ANALYSIS:
3-4 sentences on overall engagement, positive vs negative signals, best/worst moments.

RECOMMENDATIONS:
1. [Specific recommendation with timestamp]
2. [Pacing or content delivery recommendation]
3. [Build on positives or fix most frequent negative]"""

    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


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
                f"ANALYSIS:\nEngagement level: {result['engagement']} "
                f"(score {result['engagement_score']}/100). "
                f"{result['total_events']} behavioral events detected across {result['total_duration']}.\n\n"
                f"RECOMMENDATIONS:\n1. Review the session recording for disengagement patterns.\n"
                f"2. Consider adding interactive elements every 10 minutes.\n"
                f"3. Monitor student facial cues during complex topic transitions."
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