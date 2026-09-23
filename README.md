# Multi-Agent Online Exam Invigilation System

A multi-agent AI system for automated exam proctoring, coordinating 5 specialized
agents through a central orchestrator that fuses their outputs into a single,
statistically-grounded trust score in real time.

## Architecture

**5 specialized agents, each monitoring a different signal:**
- **Identity verification** — face embeddings (dlib)
- **Gaze tracking** — attention/eye tracking (MediaPipe)
- **Phone detection** — object detection (YOLOv3-tiny)
- **Audio monitoring** — voice activity + transcription (PyAudio + Whisper)
- **Browser integrity** — tab-switch / focus-loss checks

**Orchestrator:**
- Fuses all agent outputs via **Bayesian log-odds** for statistically-grounded
  decision-making
- Auto-escalates alerts (warning → suspension) based on aggregated evidence
- Streams live trust-score telemetry to a **Flask + WebSocket dashboard** for
  proctors

## Tech Stack
Python, OpenCV, dlib, MediaPipe, YOLOv3-tiny, Whisper, PyAudio, Flask, WebSockets

## Repository Structure
```
├── backend/
│   ├── agents/          # identity, gaze, audio, integrity, behaviour agents
│   │                     #   + orchestrator_agent (Bayesian fusion)
│   ├── utils/
│   ├── app.py
│   └── config.py
├── frontend/
│   ├── static/           # CSS + JS (admin dashboard, webcam monitor)
│   └── templates/        # admin.html, exam.html, index.html
├── data/
│   └── models/            # pretrained weights — see Setup below
├── demo_setup.py
├── run.py
├── MULTI_LAPTOP_DEMO.md   # running the demo across multiple devices
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

Pretrained model weights are not tracked in this repo (see `.gitignore`) —
download them separately and place them in `data/models/`:
- `yolov3-tiny.weights` / `yolov3-tiny.cfg` — [YOLOv3-tiny](https://pjreddie.com/darknet/yolo/)
- `res10_300x300_ssd_iter_140000.caffemodel` / `deploy.prototxt` — OpenCV's
  pretrained face detector

```bash
python run.py
```

See `MULTI_LAPTOP_DEMO.md` for running the identity-verification demo across
multiple devices.

> **Note:** registered face embeddings, SSL certs, and user data are excluded
> from this repo for privacy — you'll need to register your own test users
> locally to run the identity-verification flow end to end.
