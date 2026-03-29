# engagEYE 👁️
### AI-Powered Student Engagement Analysis

> Upload a lecture webcam recording → get behavioral event detection, engagement scoring, and Claude AI recommendations.

**🔗 Live Demo:** [engageye.lovable.app](https://engageye.lovable.app)
**🚀 Backend API:** [web-production-6f96.up.railway.app](https://web-production-6f96.up.railway.app)

---

## The Pipeline
```
Video Input (.mp4 / .mov / .avi / .webm)
        ↓
Frame Extraction — OpenCV @ 2fps
        ↓
Preprocessing
  • Resize to 640px
  • Auto gamma correction for low-light webcams
        ↓
Face Detection — MediaPipe FaceMesh (468 landmarks + iris)
        ↓
Per-Session Calibration
  • First 10 frames = neutral gaze baseline
  • First 10 frames = neutral head pose baseline
        ↓
6 Parallel Detectors
  😴 EAR   → drowsiness / eyes closed
  😮 MAR   → yawn detection
  👀 Iris  → gaze direction (landmarks 468 + 473)
  🎯 Pose  → head turn (OpenCV solvePnP)
  😊 Smile → positive engagement signal
  🙋 Nod   → positive engagement signal
        ↓
Event Indexing (timestamp + confidence + frame thumbnail)
        ↓
Segment Scoring — 0 to 100 per 30 seconds
  Penalties: yawn (-12) eyes_closed (-8) look_away (-6) inactivity (-15)
  Bonuses:   smile (+5) nodding (+8)
        ↓
Engagement Classification
  HIGH ≥ 65 | MEDIUM ≥ 40 | LOW < 40
        ↓
Claude AI → ANALYSIS + 3 RECOMMENDATIONS
        ↓
JSON Response → React Dashboard
```

---

## Technical Highlights

### Iris-Based Gaze Tracking
We use MediaPipe iris landmarks 468 and 473 to track actual eye direction — not just head rotation. Gaze ratio is calculated per eye and averaged:
```
h_ratio = (iris_x - eye_outer_x) / eye_width
v_ratio = (iris_y - eye_top_y) / eye_height
```
Per-session calibration means screen-viewing naturally won't trigger look_away.

### Webcam Quality Tolerance
- **Lighting** — auto gamma correction when mean luminance < 80/255
- **Angle** — MediaPipe handles ±45° rotation, solvePnP handles off-center faces  
- **Compression** — OpenCV decodes all standard formats (H.264, HEVC, VP9)

### False Positive Prevention
- Gaze detection skipped when eyes are closed (EAR < threshold)
- 6 consecutive sampled frames (~3s) required before triggering look_away
- 5 second deduplication between look_away events

---

## API

### POST /analyze
**Request:** `multipart/form-data` with field `video`

**Response:**
```json
{
  "session_id": "lecture.mp4",
  "engagement": "MEDIUM",
  "engagement_score": 54,
  "total_events": 12,
  "processing_time_s": 8.4,
  "events": [
    {
      "type": "yawn",
      "timestamp": "0:00:12",
      "confidence": 0.78,
      "thumbnail": "data:image/jpeg;base64,..."
    }
  ],
  "segments": [
    {
      "start": "0:00:00",
      "end": "0:00:30",
      "engagement": "LOW",
      "score": 31
    }
  ],
  "ai_summary": "ANALYSIS:\n...\n\nRECOMMENDATIONS:\n1. ..."
}
```

### GET /health
```json
{"status": "ok", "service": "engagEYE"}
```

---

## Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Web framework | Flask + Flask-CORS |
| Video I/O | OpenCV |
| Facial landmarks | MediaPipe FaceMesh |
| Head pose | OpenCV solvePnP + RQDecomp3x3 |
| AI | Anthropic Claude (claude-sonnet-4) |
| Deployment | Railway (Docker) |

---

## Run Locally
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python app.py
```

---

## Built for GenAI Hackathon 2026 — Track 3: Constructor Tech
