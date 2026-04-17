"""
gaze_agent.py — Gaze & Attention Tracking Agent
================================================

ALGORITHM JUSTIFICATION
-----------------------
Landmark Detection : MediaPipe Face Mesh (Lugaresi et al., 2019)
  • Provides 468 3-D facial landmarks in real-time (~30 fps on CPU).
  • Why not Active Appearance Models (AAM)?
    AAMs require per-subject fitting and degrade under lighting variation.
    MediaPipe is a pre-trained deep model—no calibration required.
  • Why not a dedicated CNN gaze estimator (e.g. iTracker, GazeNet)?
    CNN gaze estimators need per-user calibration or large labelled datasets.
    They also demand a GPU for low-latency inference.  For the invigilation
    use case, detecting whether gaze is *broadly off-screen* (not measuring
    exact gaze angle) is sufficient, making geometric methods preferable.

Head Pose Estimation : OpenCV solvePnP (Perspective-n-Point, EPnP algorithm)
  • Selected landmarks are mapped to a generic 3-D face model.
  • solvePnP recovers the rotation vector (pitch, yaw, roll) via the EPnP
    algorithm (Lepetit et al., 2009), which runs in O(n) and is numerically
    stable for n ≥ 4 correspondences.
  • Why EPnP over DLT + SVD?
    EPnP is 100× faster than iterative methods and more accurate than DLT
    for noisy correspondences.  OpenCV's cv2.SOLVEPNP_EPNP flag enables it.
  • Rodrigues rotation → Euler angles via decomposition of the rotation matrix.

Eye Aspect Ratio (EAR) : Soukupova & Cech (2016)
  • EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
  • A value below 0.20 for > 1 s indicates closed eyes (drowsiness / absence).
  • Chosen because it requires no training — purely geometric and interpretable.

Gaze-Away Duration Tracking:
  • A sliding window counts consecutive frames where gaze is flagged.
  • Only alerts after GAZE_AWAY_DURATION continuous seconds to avoid spurious
    flags from natural brief look-aways (blinking, reading captions, etc.).
"""

import time
import math
import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    logging.warning("mediapipe not installed — gaze agent in mock mode.")

from .base_agent import BaseAgent, AgentEvent
from ..config import (
    GAZE_YAW_THRESHOLD,
    GAZE_PITCH_THRESHOLD,
    GAZE_AWAY_DURATION,
    GAZE_EAR_THRESHOLD,
    GAZE_WARN_COUNT,
    GAZE_FLAG_COUNT,
    GAZE_SUSPEND_COUNT,
)

logger = logging.getLogger(__name__)

# ── 3-D model points (generic face, in mm) ────────────────────────────────────
# Correspond to: nose tip, chin, left eye corner, right eye corner,
#                left mouth corner, right mouth corner
_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),    # nose tip
    (0.0,  -330.0, -65.0),   # chin
    (-225.0, 170.0, -135.0), # left eye outer corner
    (225.0,  170.0, -135.0), # right eye outer corner
    (-150.0,-150.0, -125.0), # left mouth corner
    (150.0, -150.0, -125.0), # right mouth corner
], dtype=np.float64)

# MediaPipe Face Mesh landmark indices for the 6 model points above
_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# MediaPipe indices for left/right eye landmarks (EAR computation)
# Left eye:  [p1, p2, p3, p4, p5, p6]
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33,  160, 158,  133, 153, 144]


class GazeAgent(BaseAgent):
    """
    Tracks where the student is looking using head pose estimation and EAR.

    process() accepts payload = { "frame": <base64 JPEG> }
    and returns an AgentEvent when:
      • head yaw  > GAZE_YAW_THRESHOLD   for GAZE_AWAY_DURATION seconds
      • head pitch > GAZE_PITCH_THRESHOLD for GAZE_AWAY_DURATION seconds
      • EAR       < GAZE_EAR_THRESHOLD   for > 2 s  (eyes closed / absent)
    """

    def __init__(self):
        super().__init__("GazeAgent")
        self._face_mesh  = None
        self._gaze_away_since: Optional[float] = None
        self._eyes_closed_since: Optional[float] = None
        self._yaw_history   = deque(maxlen=30)   # last 30 frames
        self._pitch_history = deque(maxlen=30)

    def initialise(self):
        if MP_AVAILABLE:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode   = False,
                max_num_faces       = 1,
                refine_landmarks    = True,   # enables iris landmarks
                min_detection_confidence = 0.5,
                min_tracking_confidence  = 0.5,
            )
            logger.info("[GazeAgent] MediaPipe Face Mesh initialised.")
        else:
            logger.warning("[GazeAgent] Running in mock mode.")

    def process(self, payload: dict) -> Optional[AgentEvent]:
        """
        Analyse one webcam frame for gaze deviation.

        Returns an AgentEvent if the student has been looking away for
        longer than GAZE_AWAY_DURATION seconds, else None.
        """
        import base64
        frame_b64 = payload.get("frame", "")

        if not MP_AVAILABLE:
            return self._mock_process()

        # Decode frame
        try:
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",", 1)[1]
            arr   = np.frombuffer(base64.b64decode(frame_b64), dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

        if frame is None:
            return None

        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None   # face not visible — handled by BehaviourAgent

        landmarks = result.multi_face_landmarks[0].landmark

        # ── EAR check ─────────────────────────────────────────────────────────
        ear = self._compute_ear(landmarks, w, h)
        if ear < GAZE_EAR_THRESHOLD:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = time.time()
            elif time.time() - self._eyes_closed_since > 2.0:
                return self._raise_event(
                    "EYES_CLOSED", "MEDIUM",
                    f"Student's eyes appear closed (EAR={ear:.3f}) for "
                    f"{time.time()-self._eyes_closed_since:.1f}s.",
                    {"ear": round(ear, 3)},
                )
        else:
            self._eyes_closed_since = None

        # ── Head pose via solvePnP (EPnP) ─────────────────────────────────────
        yaw, pitch, roll = self._compute_head_pose(landmarks, w, h)
        self._yaw_history.append(yaw)
        self._pitch_history.append(pitch)

        # Smooth with a 5-frame moving average to reduce jitter
        smooth_yaw   = float(np.mean(list(self._yaw_history)[-5:]))
        smooth_pitch = float(np.mean(list(self._pitch_history)[-5:]))

        gaze_away = (
            abs(smooth_yaw)   > GAZE_YAW_THRESHOLD or
            abs(smooth_pitch) > GAZE_PITCH_THRESHOLD
        )

        now = time.time()
        if gaze_away:
            if self._gaze_away_since is None:
                self._gaze_away_since = now
            elif now - self._gaze_away_since > GAZE_AWAY_DURATION:
                duration = now - self._gaze_away_since
                self._gaze_away_since = now   # reset timer after alert

                # Escalate severity based on how many gaze-away events this session
                gaze_count = self.event_count("GAZE_AWAY") + 1  # +1 for this event

                if gaze_count >= GAZE_SUSPEND_COUNT:
                    severity = "CRITICAL"
                    msg = (f"Gaze away {gaze_count} times total — risk now critical. "
                           f"Exam will be automatically suspended. "
                           f"(yaw={smooth_yaw:.1f}°, pitch={smooth_pitch:.1f}°, {duration:.1f}s)")
                elif gaze_count >= GAZE_FLAG_COUNT:
                    severity = "HIGH"
                    msg = (f"Looked away again ({gaze_count}× total, {duration:.1f}s). "
                           f"Stop looking away — "
                           f"{GAZE_SUSPEND_COUNT - gaze_count} more occurrence(s) will "
                           f"automatically suspend your exam.")
                else:
                    severity = "MEDIUM"
                    msg = (f"Gaze deviated for {duration:.1f}s "
                           f"(yaw={smooth_yaw:.1f}°, pitch={smooth_pitch:.1f}°). "
                           f"Please keep your eyes on the screen.")

                return self._raise_event(
                    "GAZE_AWAY", severity, msg,
                    {
                        "yaw":        round(smooth_yaw, 2),
                        "pitch":      round(smooth_pitch, 2),
                        "duration_s": round(duration, 1),
                        "gaze_count": gaze_count,
                    },
                )
        else:
            self._gaze_away_since = None

        return None

    def get_risk(self) -> float:
        """
        Risk proportional to gaze-away frequency.
        Normalised so that GAZE_SUSPEND_COUNT events → risk 1.0.
        Saturates faster than before (÷3 instead of ÷5) to ensure
        repeated gaze-away meaningfully drives the Bayesian score up.
        """
        gaze_events   = self.event_count("GAZE_AWAY")
        closed_events = self.event_count("EYES_CLOSED")
        combined = gaze_events + 0.5 * closed_events
        return float(min(combined / float(GAZE_SUSPEND_COUNT), 1.0))

    def reset(self):
        self._gaze_away_since    = None
        self._eyes_closed_since  = None
        self._yaw_history.clear()
        self._pitch_history.clear()
        self.events.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_head_pose(self, landmarks, w: int, h: int):
        """
        Estimate head pose using PnP (Perspective-n-Point).

        Steps:
          1. Extract 2-D image points for the 6 canonical landmarks.
          2. Construct camera intrinsic matrix (focal length = image width,
             principal point = image centre — a reasonable approximation for
             webcams with unknown calibration).
          3. Solve for rotation vector via cv2.SOLVEPNP_EPNP.
          4. Convert Rodrigues rotation vector → rotation matrix → Euler angles.

        Returns (yaw, pitch, roll) in degrees.
        """
        img_pts = np.array(
            [(landmarks[idx].x * w, landmarks[idx].y * h) for idx in _LANDMARK_IDS],
            dtype=np.float64,
        )

        focal   = w
        cx, cy  = w / 2.0, h / 2.0
        cam_mat = np.array([
            [focal, 0,     cx],
            [0,     focal, cy],
            [0,     0,     1 ],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS, img_pts, cam_mat, dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success:
            return 0.0, 0.0, 0.0

        rot_mat, _ = cv2.Rodrigues(rvec)

        # Decompose rotation matrix into Euler angles (ZYX convention)
        sy   = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = math.degrees(math.atan2( rot_mat[2, 1], rot_mat[2, 2]))
            yaw   = math.degrees(math.atan2(-rot_mat[2, 0], sy))
            roll  = math.degrees(math.atan2( rot_mat[1, 0], rot_mat[0, 0]))
        else:
            pitch = math.degrees(math.atan2(-rot_mat[1, 2], rot_mat[1, 1]))
            yaw   = math.degrees(math.atan2(-rot_mat[2, 0], sy))
            roll  = 0.0

        return yaw, pitch, roll

    @staticmethod
    def _compute_ear(landmarks, w: int, h: int) -> float:
        """
        Eye Aspect Ratio (EAR) — Soukupova & Cech (2016).

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Average of left and right eye EAR is returned.
        A value below 0.20 indicates the eyes are closed.
        """
        def _pts(ids):
            return np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in ids])

        def _ear(pts):
            A = np.linalg.norm(pts[1] - pts[5])
            B = np.linalg.norm(pts[2] - pts[4])
            C = np.linalg.norm(pts[0] - pts[3])
            return (A + B) / (2.0 * C + 1e-6)

        left_ear  = _ear(_pts(_LEFT_EYE))
        right_ear = _ear(_pts(_RIGHT_EYE))
        return float((left_ear + right_ear) / 2.0)

    def _mock_process(self) -> Optional[AgentEvent]:
        """Mock: randomly raise a gaze-away event ~10% of the time."""
        if np.random.rand() < 0.10:
            return self._raise_event(
                "GAZE_AWAY", "MEDIUM",
                "Mock: student gaze deviated (demo mode — mediapipe not installed).",
                {"yaw": 40.0, "pitch": 0.0, "duration_s": 6.0},
            )
        return None
