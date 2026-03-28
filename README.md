# engagEYE Backend

**Stack:** Python · Flask · OpenCV · MediaPipe · Claude AI

## What it does
Receives a webcam lecture video → runs behavioral analysis → returns JSON:
- Yawn detection (Mouth Aspect Ratio)
- Drowsiness / eyes-closed detection (Eye Aspect Ratio)
- Look-away detection (head pose via OpenCV solvePnP)
- Inactivity detection (no face in frame)
- Per-30s segment engagement scores (0–100)
- Overall HIGH / MEDIUM / LOW classification
- AI-generated instructor summary (Claude)

## Local setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python app.py
# → running on http://localhost:5000
```

## Test it locally

```bash
curl -X POST http://localhost:5000/analyze \
  -F "video=@/path/to/lecture.mp4"
```

## Deploy to Railway

1. Push this folder to a GitHub repo
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Set environment variable: `ANTHROPIC_API_KEY` = your key
4. Railway auto-detects Python, installs requirements.txt, uses Procfile
5. Your backend URL will be: `https://your-app.up.railway.app`

## API

### GET /health
Returns `{"status": "ok"}`

### POST /analyze
- Body: `multipart/form-data`
- Field: `video` (file — mp4, mov, avi, mkv, webm)

**Response:**
```json
{
  "session_id": "lecture.mp4",
  "total_duration": "1:23:45",
  "engagement": "LOW",
  "engagement_score": 42,
  "total_events": 7,
  "processing_time_s": 18.4,
  "events": [
    {"type": "yawn", "timestamp": "0:02:14", "confidence": 0.78},
    {"type": "look_away", "timestamp": "0:05:30", "confidence": 0.62}
  ],
  "segments": [
    {"start": "0:00:00", "end": "0:00:30", "engagement": "HIGH", "score": 88},
    {"start": "0:00:30", "end": "0:01:00", "engagement": "LOW", "score": 31}
  ],
  "ai_summary": "The student showed moderate engagement overall..."
}
```

## Lovable frontend — connect to this backend

In your Lovable React app, call the backend like this:

```typescript
const analyzeVideo = async (file: File) => {
  const formData = new FormData();
  formData.append("video", file);

  const response = await fetch("https://YOUR-RAILWAY-URL/analyze", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  return data;
};
```

## Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key |
| `PORT` | Port (set automatically by Railway) |
