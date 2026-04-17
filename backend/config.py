"""
config.py — Central configuration for the Online Exam Invigilation System.

All tuneable thresholds and constants are defined here so that each agent
can import them rather than hard-coding magic numbers.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
DATA_DIR           = os.path.join(BASE_DIR, "..", "data")
REGISTERED_FACES   = os.path.join(DATA_DIR, "registered_faces")
EXAM_LOGS_DIR      = os.path.join(DATA_DIR, "exam_logs")

# ─── Identity Verification Agent ──────────────────────────────────────────────
# dlib ResNet-128 face embeddings use Euclidean distance.
# Threshold of 0.6 is recommended by the face_recognition library authors
# (Davis King, 2017) and yields ~99.38% accuracy on the LFW benchmark.
FACE_DISTANCE_THRESHOLD   = 0.6      # Euclidean distance; below = same person
IDENTITY_CHECK_INTERVAL   = 10       # seconds between periodic re-verifications
FACE_CONFIDENCE_THRESHOLD = 0.75     # SSD confidence cutoff for PRESENCE detection.
                                     # 0.88 was too high — a student at slight angle
                                     # or in average lighting scores 0.75-0.85, causing
                                     # false "absent" alerts even when present.
                                     # Area filter + temporal debounce handle false
                                     # positives from posters/reflections instead.
FACE_IDENTITY_THRESHOLD   = 0.88     # Higher threshold used ONLY inside IdentityAgent
                                     # for the initial face crop quality gate.

# ─── Gaze & Attention Tracking Agent ──────────────────────────────────────────
# MediaPipe Face Mesh provides 468 landmarks.  We derive gaze from iris-centre
# displacement relative to the eye bounding box.
GAZE_YAW_THRESHOLD    = 25   # degrees; reduced from 35 — catches deliberate head turns
GAZE_PITCH_THRESHOLD  = 20   # degrees; reduced from 25
GAZE_AWAY_DURATION    = 3    # seconds; reduced from 5 — flags quicker
GAZE_EAR_THRESHOLD    = 0.20 # Eye Aspect Ratio below this = eyes closed

# ─── Suspicious Behaviour Detection Agent ─────────────────────────────────────
MAX_ALLOWED_FACES      = 1
ABSENCE_THRESHOLD      = 10   # seconds without face before FIRST warning
ABSENCE_SUSPEND_COUNT  = 2    # number of confirmed absence events before auto-suspend
HEAD_TURN_YAW_LIMIT    = 30   # degrees yaw — catches clear left/right head turns
HEAD_TURN_SUSTAIN_S    = 2.5  # seconds sustained before alert
HEAD_TURN_WARN_COUNT   = 1    # 1st confirmed turn → MEDIUM banner
HEAD_TURN_FLAG_COUNT   = 2    # 2nd confirmed turn → HIGH modal warning
HEAD_TURN_SUSPEND_COUNT= 4    # 4th confirmed turn → CRITICAL, feeds suspension
PHONE_CONF_THRESHOLD   = 0.82  # Raised from 0.75 — YOLOv3-tiny produces high-confidence
                               # false positives for notebooks, remotes, and dark rectangular
                               # objects.  0.82 requires very clear phone-like features.
PHONE_CONFIRM_FRAMES   = 5    # Raised from 3 → 2.5 s at 2 fps.  Eliminates brief
                               # reflections, cable shadows, and single-frame noise.
PHONE_SUSPEND_COUNT    = 3    # confirmed detections before suspension

# Phone geometry filters (applied after confidence check):
#   • PHONE_MAX_AREA_RATIO : phone bounding-box must be < this fraction of the frame.
#     A detected "phone" covering > 20% of the frame is almost certainly the student's
#     laptop screen, a monitor behind them, or a large book — not a handheld phone.
#   • PHONE_MIN_ASPECT_RATIO / PHONE_MAX_ASPECT_RATIO : phones are rectangular.
#     Aspect ratio = max(w,h) / min(w,h).  Typical phone: 1.7–2.5.
#     We allow 1.3–4.0 to accommodate tilted or partially-occluded phones,
#     while rejecting nearly square objects (notebooks, sticky notes) and
#     very elongated lines (pens, cables, rulers → ratio > 5).
PHONE_MAX_AREA_RATIO   = 0.20  # box area must be < 20% of frame
PHONE_MIN_ASPECT_RATIO = 1.3   # must be clearly rectangular (not square)
PHONE_MAX_ASPECT_RATIO = 4.0   # must not be a thin line / cable

# Multiple-face detection — graduated response (warn → suspend), fully automatic
#
# Root causes of false positives addressed here:
#   (a) Wall posters / framed photos — these have faces but are small and far away.
#       MULTI_FACE_MIN_AREA_RATIO filters them out by minimum face size.
#   (b) The student's own face detected twice at different angles.
#       IOU-based NMS (implemented in BehaviourAgent) suppresses duplicate boxes
#       from the same physical face.
#   (c) Bright-window reflections producing ghost faces.
#       MULTI_FACE_CONFIDENCE (higher than presence threshold) rejects low-quality
#       detections that only a nearby, well-lit face would pass.
#
#   • MULTI_FACE_CONFIDENCE    : dedicated confidence threshold for the second-face
#     check, HIGHER than FACE_CONFIDENCE_THRESHOLD (which handles presence/absence).
#     0.88 — a poster or reflection rarely scores this high on ResNet-10 SSD.
#   • MULTI_FACE_MIN_AREA_RATIO : 0.04 = 4% of frame ≈ 80×60 px in 640×480.
#     A real nearby person would occupy well above this area.
#   • MULTI_FACE_CONFIRM_FRAMES : 5 frames = 2.5 s at 2 fps.
MULTI_FACE_CONFIDENCE      = 0.88   # separate, stricter gate for counting extra faces
MULTI_FACE_MIN_AREA_RATIO  = 0.04   # ignore faces smaller than 4% of frame (raised from 1.5%)
MULTI_FACE_CONFIRM_FRAMES  = 5      # consecutive frames needed to confirm (raised from 3)
MULTI_FACE_WARN_COUNT      = 1      # 1st confirmed event  → WARN student
MULTI_FACE_SUSPEND_COUNT   = 2      # 2nd confirmed event  → SUSPEND (automatic)

# ─── Question Paper Integrity Agent ───────────────────────────────────────────
# Fisher-Yates O(n) shuffle seed is derived from student-id + session-token
# so each student gets a deterministic but unique ordering.
COPY_PASTE_DISABLED    = True
RIGHT_CLICK_DISABLED   = True
QUESTIONS_PER_EXAM     = 10
EXAM_DURATION_MINUTES  = 30

# ─── Audio Monitoring Agent ───────────────────────────────────────────────────
# WebRTC VAD (Google) — processes 30 ms frames of 16 kHz 16-bit mono PCM.
# Aggressiveness 0 (least aggressive) to 3 (most aggressive).
# Mode 2 balances whisper detection vs. false positives from ambient noise.
AUDIO_SAMPLE_RATE        = 16000   # Hz — WebRTC VAD supports 8k/16k/32k/48k
AUDIO_VAD_AGGRESSIVENESS = 2       # 0–3. Mode 2 — strict noise rejection:
                                   #   mode 1 = lets fan/AC noise peaks through as speech
                                   #   mode 2 = rejects broadband noise (fans, AC, hum) reliably;
                                   #            still passes normal conversational speech
                                   #   mode 3 = too strict, misses some real speech
                                   # The adaptive noise-floor gate (SPEECH_SNR_RATIO) handles fan
                                   # noise at the energy level; mode 2 handles spectral ambiguity.
AUDIO_SPEECH_DURATION    = 2.0     # total seconds of speech before alerting.
                                   # Raised from 1.5 → 2.0 s: genuine cheating speech is
                                   # sustained; fan noise rarely fools the VAD for 2 full seconds.
AUDIO_SPEECH_GRACE_S     = 1.0     # seconds of silence allowed within a speech episode.
                                   # Lowered from 1.5 — tighter gap prevents slow talkers
                                   # from accidentally resetting the timer between words.
AUDIO_MIN_RMS            = 0.0005  # RMS gate before VAD — blocks true digital silence/DC.
                                   # Lowered from 0.001: some laptop mics produce speech at
                                   # RMS 0.001–0.003 even with AGC on.  0.0005 still blocks
                                   # 50/60 Hz hum (RMS ≈ 0.0001) and pure silence (≈ 0).
AUDIO_FALLBACK_RMS       = 0.003   # RMS threshold when webrtcvad is NOT installed or says no.
                                   # Middle ground: catches quiet speech without triggering on
                                   # keyboard clicks (transient peaks rarely sustain > 0.003).

# ─── Orchestrator / Risk Scoring ──────────────────────────────────────────────
# Bayesian log-odds aggregation weights.  Higher weight = agent's verdict
# carries more influence on the final risk score (0-1).
#
# Weight rationale:
#   identity  (0.30) : face mismatch is a strong but not infallible signal
#                      (lighting / webcam angle can cause false positives)
#   behaviour (0.30) : multiple faces is a hard override (returns 1.0),
#                      so exact weight matters less; absence/phone also serious
#   gaze      (0.15) : looking away is frequent but ambiguous alone
#   integrity (0.15) : tab switch triggers hard override at count ≥ 2;
#                      minor events (copy, focus-loss) need weight
#   audio     (0.10) : audio evidence is corroborative; weighted lower because
#                      microphone quality varies widely across students
#
# Note: weights are used in the Bayesian log-odds sum, not a simple average.
# They are normalised implicitly via the log-odds accumulation.
# Bayesian log-odds weights — NOT required to sum to 1.
# Each weight scales how much that agent's likelihood-ratio evidence
# shifts the fused log-odds.  Larger = more impactful.
# Calibrated so that a single clear violation (phone, tab-suspend)
# pushes risk into FLAG territory on its own.
AGENT_WEIGHTS = {
    "identity":  1.0,   # face mismatch
    "behaviour": 1.8,   # phone / absence / multi-face  ← highest impact
    "gaze":      1.2,   # repeated gaze-away
    "integrity": 1.5,   # tab switch, fullscreen, copy-paste
    "audio":     0.6,   # corroborative (mic quality varies)
}

# Gaze escalation: how many GAZE_AWAY events before severity escalates
GAZE_WARN_COUNT    = 1   # 1st gaze-away → MEDIUM (banner)
GAZE_FLAG_COUNT    = 2   # 2nd gaze-away → HIGH (warning modal)
GAZE_SUSPEND_COUNT = 5   # 5th gaze-away → CRITICAL (feeds Bayesian suspension)

RISK_WARN_THRESHOLD    = 0.50   # send warning to student (automatic)
RISK_SUSPEND_THRESHOLD = 0.75   # auto-suspend exam session (automatic, no human needed)
# NOTE: FLAG/human-review step is intentionally removed.
# All decisions are taken automatically by the system.

# ─── Flask / SocketIO ─────────────────────────────────────────────────────────
SECRET_KEY  = os.environ.get("SECRET_KEY", "exam_invigilation_secret_2024")
DEBUG       = os.environ.get("DEBUG", "true").lower() == "true"
HOST        = "0.0.0.0"
PORT        = 5000
FRAME_RATE  = 2   # how many webcam frames per second the frontend sends
