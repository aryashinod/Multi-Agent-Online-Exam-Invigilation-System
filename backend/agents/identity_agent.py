"""
identity_agent.py — Identity Verification Agent
================================================

ALGORITHM JUSTIFICATION
-----------------------
Face Detection : OpenCV DNN (SSD with ResNet-10 backbone)
  • Why not Haar Cascades (Viola-Jones, 2001)?
    Viola-Jones achieves ~95% detection rate but has a high false-positive rate
    in non-frontal poses and variable lighting — common in home-exam settings.
  • Why SSD ResNet-10?
    Single-Shot MultiBox Detector runs in real-time (>40 fps on CPU),
    handles partial occlusion, and achieves ~97% AP on WIDER FACE benchmark.
    The cv2.dnn module loads the Caffe model without a GPU dependency, keeping
    the system deployable on student-grade hardware.

Face Recognition : dlib ResNet-128 (via face_recognition library)
  • The network maps a face crop to a 128-dimensional embedding vector using a
    deep metric-learning objective (triplet loss, Schroff et al. 2015).
  • Similarity is measured with Euclidean distance.  A threshold of 0.6 gives
    99.38% accuracy on the Labelled Faces in the Wild (LFW) benchmark (King 2017).
  • Why not Eigenfaces (PCA)?
    Eigenfaces struggle with illumination change and expression variance — both
    common in online exams.  Their top-1 accuracy on LFW is ~60%.
  • Why not Local Binary Patterns (LBP)?
    LBP is faster but peaks at ~87% LFW accuracy and is sensitive to pose.
  • Why not a full CNN (e.g. ArcFace)?
    ArcFace achieves 99.82% on LFW but requires a GPU for inference.  dlib's
    model gives near-equivalent accuracy on low-resolution webcam frames while
    running on CPU in ~50 ms.

Periodic Re-verification Strategy:
  Re-verify every IDENTITY_CHECK_INTERVAL seconds (default 30 s) rather than
  every frame, to balance accuracy with CPU load.  Each mismatch increments a
  suspicion counter; three mismatches in a session trigger a HIGH-severity event.
"""

import os
import time
import base64
import logging
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logging.warning("face_recognition not installed — identity agent in mock mode.")

from .base_agent import BaseAgent, AgentEvent
from ..config import (
    FACE_DISTANCE_THRESHOLD,
    FACE_CONFIDENCE_THRESHOLD,
    FACE_IDENTITY_THRESHOLD,
    IDENTITY_CHECK_INTERVAL,
    REGISTERED_FACES,
)

logger = logging.getLogger(__name__)


class IdentityAgent(BaseAgent):
    """
    Verifies that the person sitting the exam is the registered student.

    Workflow:
      1. Registration  : store_reference(student_id, frame) encodes and saves
                         the student's 128-D face embedding to disk.
      2. Verification  : process(payload) compares the current frame against
                         the stored embedding.  A distance > threshold is a
                         mismatch event.
    """

    def __init__(self):
        super().__init__("IdentityAgent")
        self._reference_encodings: dict = {}   # student_id -> np.ndarray (128-D)
        self._current_student: Optional[str] = None
        self._last_check_time: float = 0.0
        self._mismatch_count: int = 0          # consecutive mismatches (decays on match)
        self._total_mismatches: int = 0        # cumulative mismatch events this session
        self._no_ref_warned: bool = False      # warn once per session if no reference

        # ── SSD face detector (OpenCV DNN) ────────────────────────────────────
        # Paths relative to this file; deployed weights are in data/models/
        model_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")
        self._proto   = os.path.join(model_dir, "deploy.prototxt")
        self._weights = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        self._net = None

    def initialise(self):
        """Load the SSD face detector and any stored reference embeddings."""
        # Load SSD model if available
        if os.path.exists(self._proto) and os.path.exists(self._weights):
            self._net = cv2.dnn.readNetFromCaffe(self._proto, self._weights)
            logger.info("[IdentityAgent] SSD face detector loaded.")
        else:
            logger.warning("[IdentityAgent] SSD model not found — using Haar cascade fallback.")
            self._net = None
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        # Load any persisted reference embeddings
        os.makedirs(REGISTERED_FACES, exist_ok=True)
        for fname in os.listdir(REGISTERED_FACES):
            if fname.endswith(".npy"):
                sid = fname.replace(".npy", "")
                enc_path = os.path.join(REGISTERED_FACES, fname)
                self._reference_encodings[sid] = np.load(enc_path)
                logger.info("[IdentityAgent] Loaded embedding for student %s.", sid)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_current_student(self, student_id: str):
        """Tell the agent which student is currently sitting the exam."""
        self._current_student = student_id
        self._mismatch_count  = 0
        self._last_check_time = 0.0
        self._no_ref_warned   = False

    def store_reference(self, student_id: str, frame_b64: str) -> bool:
        """
        Encode the student's face from a registration frame and persist it.

        Parameters
        ----------
        student_id : str
            Unique identifier (e.g. enrolment number).
        frame_b64  : str
            Base64-encoded JPEG from the registration webcam snapshot.

        Returns True on success, False if no face was detected.
        """
        frame = self._decode_frame(frame_b64)
        if frame is None:
            return False

        if FACE_RECOGNITION_AVAILABLE:
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encs  = face_recognition.face_encodings(rgb)
            if not encs:
                logger.warning("[IdentityAgent] No face found during registration for %s.", student_id)
                return False
            encoding = encs[0]
        else:
            # Mock mode: store a random vector (for testing without dlib)
            encoding = np.random.rand(128).astype(np.float64)

        self._reference_encodings[student_id] = encoding
        save_path = os.path.join(REGISTERED_FACES, f"{student_id}.npy")
        np.save(save_path, encoding)
        logger.info("[IdentityAgent] Reference embedding saved for %s.", student_id)
        return True

    def process(self, payload: dict) -> Optional[AgentEvent]:
        """
        Analyse one webcam frame.

        payload = { "student_id": str, "frame": <base64 JPEG str> }

        Returns an AgentEvent on mismatch / no-face, else None.
        Respects IDENTITY_CHECK_INTERVAL — skips frames until the interval elapses.
        """
        now = time.time()
        if now - self._last_check_time < IDENTITY_CHECK_INTERVAL:
            return None
        self._last_check_time = now

        student_id = payload.get("student_id") or self._current_student
        if not student_id:
            return None

        frame = self._decode_frame(payload.get("frame", ""))
        if frame is None:
            return self._raise_event(
                "NO_FRAME", "MEDIUM",
                "Unable to decode webcam frame for identity check.",
            )

        # ── Step 1: Detect face ───────────────────────────────────────────────
        face_crop, confidence = self._detect_face(frame)
        if face_crop is None:
            return self._raise_event(
                "NO_FACE_DETECTED", "MEDIUM",
                f"No face detected during identity re-verification (t={now:.0f}).",
                {"confidence": 0.0},
            )

        # ── Step 2: Encode face ───────────────────────────────────────────────
        if FACE_RECOGNITION_AVAILABLE:
            rgb  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb)
            if not encs:
                # The face crop was too small or the dlib model couldn't landmark it.
                # This is a transient issue (head angle, blur) — skip this check
                # silently rather than raising an event that inflates the risk score.
                logger.debug("[IdentityAgent] Encoding failed for %s — skipping check.", student_id)
                return None
            current_encoding = encs[0]
        else:
            current_encoding = np.random.rand(128).astype(np.float64)

        # ── Step 3: No reference stored → warn once, then skip ───────────────────
        # Auto-enrolling during the exam is unsafe: an impersonator sitting from
        # the start would have their face stored as the reference and pass forever.
        # Instead, warn the admin once at session start that this student has no
        # face reference and identity verification is therefore unavailable.
        if student_id not in self._reference_encodings:
            if not self._no_ref_warned:
                self._no_ref_warned = True
                logger.warning(
                    "[IdentityAgent] No face reference for %s. "
                    "Student must register with a face photo for identity checks to work.",
                    student_id,
                )
                return self._raise_event(
                    "NO_FACE_REFERENCE", "HIGH",
                    f"No face reference registered for student {student_id}. "
                    f"Identity verification is DISABLED for this session. "
                    f"Admin: ensure the student registers with a face photo.",
                    {"student_id": student_id},
                )
            return None

        # ── Step 4: Compare against reference ────────────────────────────────
        if FACE_RECOGNITION_AVAILABLE:
            distance = face_recognition.face_distance(
                [self._reference_encodings[student_id]], current_encoding
            )[0]
        else:
            distance = float(np.random.beta(2, 5))

        match = distance < FACE_DISTANCE_THRESHOLD

        if not match:
            self._mismatch_count  += 1
            self._total_mismatches += 1

            if self._mismatch_count >= 3:
                # 3rd consecutive mismatch → definitive identity swap
                return self._raise_event(
                    "IDENTITY_MISMATCH_SUSPEND", "CRITICAL",
                    f"Identity verification failed {self._mismatch_count} consecutive times "
                    f"for student {student_id} (distance={distance:.3f}). "
                    f"A different person appears to be sitting the exam. "
                    f"Session suspended for mandatory review.",
                    {"distance": round(distance, 4),
                     "consecutive_mismatches": self._mismatch_count,
                     "total_mismatches": self._total_mismatches},
                )
            elif self._total_mismatches >= 2:
                severity = "HIGH"
                msg = (f"Identity mismatch #{self._total_mismatches} for {student_id} "
                       f"(distance={distance:.3f}). Another failure will suspend the exam.")
            else:
                severity = "MEDIUM"
                msg = (f"Face verification failed for {student_id} "
                       f"(distance={distance:.3f}). Please look directly at the camera.")

            return self._raise_event(
                "IDENTITY_MISMATCH", severity, msg,
                {"distance": round(distance, 4),
                 "consecutive_mismatches": self._mismatch_count,
                 "total_mismatches": self._total_mismatches},
            )

        # All clear — decay consecutive counter but keep total for audit trail
        self._mismatch_count = max(0, self._mismatch_count - 1)
        return None

    def get_risk(self) -> float:
        """
        Graduated risk based on cumulative mismatch count.

        Designed so that a single confirmed mismatch already pushes the
        Bayesian-fused risk into WARN territory (≥ 0.50), two mismatches
        reach FLAG territory (≥ 0.70), and three consecutive mismatches
        trigger the hard IDENTITY_MISMATCH_SUSPEND override.

          0 mismatches     → 0.0   (no evidence)
          1 mismatch       → 0.65  (WARN — different person possible)
          2 mismatches     → 0.85  (FLAG — strong indicator)
          3+ mismatches    → 1.0   (SUSPEND — certainty, hard override active)
        """
        if self.event_count("IDENTITY_MISMATCH_SUSPEND") > 0:
            return 1.0
        n = self._total_mismatches
        if n >= 3:
            return 1.0
        elif n == 2:
            return 0.85
        elif n == 1:
            return 0.65
        return 0.0

    def reset(self):
        self._current_student  = None
        self._mismatch_count   = 0
        self._total_mismatches = 0
        self._last_check_time  = 0.0
        self._no_ref_warned    = False
        self.events.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _decode_frame(self, b64_str: str) -> Optional[np.ndarray]:
        """Decode a base64 JPEG string to an OpenCV BGR frame."""
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_str)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            logger.debug("[IdentityAgent] Frame decode error: %s", exc)
            return None

    def _detect_face(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect the largest face in a frame.

        Returns (cropped_face_BGR, confidence) or (None, 0.0).

        Uses OpenCV SSD when available, falls back to Haar cascade.
        The SSD model returns a confidence score per detection; Haar does not,
        so we assign confidence=1.0 for Haar detections.
        """
        h, w = frame.shape[:2]

        if self._net is not None:
            # SSD forward pass — use FACE_IDENTITY_THRESHOLD (0.88) here so we
            # only attempt face encoding on high-confidence, high-quality detections.
            # The lower FACE_CONFIDENCE_THRESHOLD (0.75) is used in BehaviourAgent
            # for presence/absence detection only.
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0)
            )
            self._net.setInput(blob)
            detections = self._net.forward()

            best_conf, best_box = 0.0, None
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf >= FACE_IDENTITY_THRESHOLD and conf > best_conf:
                    best_conf = conf
                    best_box  = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            if best_box is not None:
                x1, y1, x2, y2 = best_box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                # Guard against zero-size crop (face detected entirely off-screen edge)
                if x2 <= x1 or y2 <= y1:
                    return None, 0.0
                return frame[y1:y2, x1:x2], float(best_conf)

        elif hasattr(self, "_haar"):
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            if len(faces):
                # Take the largest detection
                x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                return frame[y:y+fh, x:x+fw], 1.0

        return None, 0.0
