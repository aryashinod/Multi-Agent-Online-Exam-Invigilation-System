"""
behaviour_agent.py — Suspicious Behaviour Detection Agent
==========================================================

ALGORITHM JUSTIFICATION
-----------------------
Multiple-Face Detection : OpenCV SSD ResNet-10
  • SSD scans the entire frame in one forward pass (O(1) per frame).
  • Confidence threshold raised to 0.88 to suppress false positives from
    posters, photographs, mirror reflections, and small background faces.
  • Additional area filter: faces whose bounding-box area < 1.5% of the
    frame are discarded as background artefacts.
  • Temporal debouncing (3 consecutive frames): a single anomalous frame
    (lighting glitch, JPEG artefact) cannot trigger an alert on its own.
    Only sustained multi-face detections are reported.

Graduated Response Policy (warn → flag → suspend)
  • 1st confirmed multi-face event → MEDIUM warning shown to student.
    "There appears to be another person visible — please ensure you are
    alone. One more confirmation will flag your session for review."
  • 2nd confirmed event → HIGH flag for human reviewer.
    "Your session has been flagged. A reviewer has been notified."
  • 3rd+ confirmed event → CRITICAL → Orchestrator suspends session.
  This avoids punishing students for a single false positive (a roommate
  briefly walking behind them) while still deterring sustained collusion.

Head-Turn Detection : solvePnP (EPnP)
  • Large yaw > HEAD_TURN_YAW_LIMIT° sustained for > 4 s → alert.

Absence Detection : frame-level timeout (ABSENCE_THRESHOLD seconds).

Object Detection : MobileNet-SSD COCO (phone, book, laptop, remote).
"""

import time
import base64
import logging
from typing import Optional, List, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

from .base_agent import BaseAgent, AgentEvent
from ..config import (
    MAX_ALLOWED_FACES,
    ABSENCE_THRESHOLD,
    ABSENCE_SUSPEND_COUNT,
    HEAD_TURN_YAW_LIMIT,
    HEAD_TURN_SUSTAIN_S,
    HEAD_TURN_WARN_COUNT,
    HEAD_TURN_FLAG_COUNT,
    HEAD_TURN_SUSPEND_COUNT,
    PHONE_CONF_THRESHOLD,
    PHONE_CONFIRM_FRAMES,
    PHONE_SUSPEND_COUNT,
    PHONE_MAX_AREA_RATIO,
    PHONE_MIN_ASPECT_RATIO,
    PHONE_MAX_ASPECT_RATIO,
    FACE_CONFIDENCE_THRESHOLD,
    MULTI_FACE_CONFIDENCE,
    MULTI_FACE_MIN_AREA_RATIO,
    MULTI_FACE_CONFIRM_FRAMES,
    MULTI_FACE_WARN_COUNT,
    MULTI_FACE_SUSPEND_COUNT,
)
import os

logger = logging.getLogger(__name__)

# YOLOv3-tiny COCO class indices (0-indexed, matching coco.names)
# Only "cell phone" is flagged — laptop is the exam device itself, book
# detection has too many false positives (notebooks, textured backgrounds).
# Remote is excluded — too easily confused with pens and other desk items.
_YOLO_SUSPICIOUS = {67: "cell phone"}
# Legacy MobileNet-SSD COCO class IDs (kept for fallback)
_SSD_SUSPICIOUS  = {77: "cell phone"}


class BehaviourAgent(BaseAgent):
    """
    Monitors the video feed for behavioural cheating signals:
      1. Multiple people visible (with debounce + graduated response)
      2. Sustained head turn away from screen
      3. Student absent from frame for extended period
      4. Suspicious objects (phone, book) detected
    """

    def __init__(self):
        super().__init__("BehaviourAgent")
        self._face_net    = None
        self._object_net  = None
        self._object_type = None   # "yolo" | "ssd" | None
        self._coco_labels = []
        self._haar        = None
        self._face_mesh   = None

        self._last_face_time:    float = time.time()
        self._head_turn_since:   Optional[float] = None
        self._head_turn_count:   int = 0   # confirmed head-turn events this session

        # ── Multi-face debounce state ─────────────────────────────────────────
        self._multi_face_streak:     int = 0
        self._confirmed_event_count: int = 0

        # ── Phone detection state ─────────────────────────────────────────────
        self._phone_count:        int = 0    # confirmed phone events this session
        self._phone_streak:       int = 0    # consecutive frames currently showing phone
        self._phone_event_fired:  bool = False  # True once event fires for current streak;
                                               # prevents duplicate events per streak

        # ── Head turn state ───────────────────────────────────────────────────
        self._head_currently_turned: bool = False  # True while head is actively turned;
                                                   # prevents re-firing during same turn

        # ── Absence tracking ──────────────────────────────────────────────────
        self._absence_count:    int = 0    # confirmed absence events this session
        self._absence_alerted:  bool = False  # True while currently absent & warned

    def initialise(self):
        model_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "models"
        )
        proto   = os.path.join(model_dir, "deploy.prototxt")
        weights = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        if os.path.exists(proto) and os.path.exists(weights):
            self._face_net = cv2.dnn.readNetFromCaffe(proto, weights)
            logger.info("[BehaviourAgent] SSD face detector loaded.")
        else:
            logger.warning("[BehaviourAgent] SSD model missing — Haar cascade fallback.")
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

        # ── Object detection: YOLOv3-tiny (preferred) ────────────────────────
        yolo_cfg = os.path.join(model_dir, "yolov3-tiny.cfg")
        yolo_wts = os.path.join(model_dir, "yolov3-tiny.weights")
        if os.path.exists(yolo_cfg) and os.path.exists(yolo_wts):
            self._object_net  = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_wts)
            self._object_type = "yolo"
            logger.info("[BehaviourAgent] YOLOv3-tiny object detector loaded (phone/book detection active).")
        else:
            # Fallback: try legacy MobileNet-SSD COCO
            obj_pb  = os.path.join(model_dir, "ssd_mobilenet_v2_coco.pb")
            obj_txt = os.path.join(model_dir, "ssd_mobilenet_v2_coco.pbtxt")
            if os.path.exists(obj_pb) and os.path.exists(obj_txt):
                self._object_net  = cv2.dnn.readNetFromTensorflow(obj_pb, obj_txt)
                self._object_type = "ssd"
                logger.info("[BehaviourAgent] MobileNet-SSD object detector loaded (fallback).")
            else:
                self._object_type = None
                logger.warning("[BehaviourAgent] No object detection model found — phone detection disabled. "
                               "Restart the server to auto-download YOLOv3-tiny.")

        if MP_AVAILABLE:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=4,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            )

    # ── Main process ──────────────────────────────────────────────────────────

    def process(self, payload: dict) -> Optional[AgentEvent]:
        frame = self._decode(payload.get("frame", ""))
        if frame is None:
            return None

        h, w   = frame.shape[:2]
        events: List[AgentEvent] = []

        # ── 1. Face count ─────────────────────────────────────────────────────
        # Two thresholds serve different purposes:
        #   presence_faces  — lower confidence (FACE_CONFIDENCE_THRESHOLD=0.75)
        #                     used only to check if the student is in frame.
        #   strict_faces    — higher confidence (MULTI_FACE_CONFIDENCE=0.88) +
        #                     larger area (MULTI_FACE_MIN_AREA_RATIO=4%) + NMS.
        #                     Used to decide if a *second person* is in frame.
        #                     Posters, reflections and the same face double-detected
        #                     at slightly different crops are all suppressed here.
        presence_faces = self._detect_faces(frame, w, h,
                                            min_confidence=FACE_CONFIDENCE_THRESHOLD)
        strict_faces   = self._detect_faces(frame, w, h,
                                            min_confidence=MULTI_FACE_CONFIDENCE)
        n_faces        = len(presence_faces)   # for absence detection
        n_strict_faces = len(strict_faces)     # for multi-face detection

        if n_faces == 0:
            self._multi_face_streak = 0
            now     = time.time()
            absent_s = now - self._last_face_time

            if absent_s > ABSENCE_THRESHOLD and not self._absence_alerted:
                # First confirmed absence in this stretch
                self._absence_count  += 1
                self._absence_alerted = True

                if self._absence_count >= ABSENCE_SUSPEND_COUNT:
                    # Second absence → CRITICAL, orchestrator will suspend
                    events.append(self._raise_event(
                        "STUDENT_ABSENT_SUSPEND", "CRITICAL",
                        f"Student absent from frame for {absent_s:.0f}s "
                        f"(2nd confirmed absence). Exam suspended.",
                        {"absence_duration_s": round(absent_s, 1),
                         "absence_count": self._absence_count},
                    ))
                else:
                    # First absence → MEDIUM warning
                    events.append(self._raise_event(
                        "STUDENT_ABSENT", "MEDIUM",
                        f"Student absent from frame for {absent_s:.0f}s. "
                        f"Please return to your seat. "
                        f"A second absence will suspend your exam.",
                        {"absence_duration_s": round(absent_s, 1),
                         "absence_count": self._absence_count},
                    ))
        else:
            self._last_face_time  = time.time()
            self._absence_alerted = False   # reset when face returns

        if n_strict_faces > MAX_ALLOWED_FACES:
            self._multi_face_streak += 1

            if self._multi_face_streak >= MULTI_FACE_CONFIRM_FRAMES:
                # Streak confirmed — raise exactly one event per streak crossing
                if self._multi_face_streak == MULTI_FACE_CONFIRM_FRAMES:
                    self._confirmed_event_count += 1
                    evt = self._build_multi_face_event(n_strict_faces, strict_faces)
                    if evt:
                        events.append(evt)
        else:
            self._multi_face_streak = 0   # reset streak — back to single face

        # ── 2. Sustained head turn ────────────────────────────────────────────
        # One event per distinct turn: fire when the turn is first confirmed as
        # sustained (> HEAD_TURN_SUSTAIN_S), then suppress until the head returns
        # to normal (yaw back inside limit).  This prevents spamming events for
        # a single continuous turn held for several seconds.
        if n_faces >= 1 and self._face_mesh is not None:   # presence_faces
            yaw = self._get_head_yaw(frame, w, h)
            if yaw is not None and abs(yaw) > HEAD_TURN_YAW_LIMIT:
                now = time.time()
                if self._head_turn_since is None:
                    self._head_turn_since = now
                elif (now - self._head_turn_since > HEAD_TURN_SUSTAIN_S
                        and not self._head_currently_turned):
                    # First time this turn has been sustained long enough
                    duration = now - self._head_turn_since
                    self._head_turn_count += 1
                    self._head_currently_turned = True   # suppress until head returns
                    direction = "left" if yaw < 0 else "right"

                    if self._head_turn_count >= HEAD_TURN_SUSPEND_COUNT:
                        severity = "CRITICAL"
                        desc = (
                            f"Repeated head turns ({self._head_turn_count} times total). "
                            f"Turned {direction} ({yaw:.0f}°) for {duration:.1f}s. "
                            f"Risk score now critical — exam may be auto-suspended."
                        )
                    elif self._head_turn_count >= HEAD_TURN_FLAG_COUNT:
                        severity = "HIGH"
                        desc = (
                            f"Head turned {direction} again ({yaw:.0f}°, {duration:.1f}s) — "
                            f"turn #{self._head_turn_count}. Stop looking away. "
                            f"{HEAD_TURN_SUSPEND_COUNT - self._head_turn_count} more turn(s) "
                            f"will automatically suspend your exam."
                        )
                    else:
                        severity = "MEDIUM"
                        desc = (
                            f"Head turned {direction} ({yaw:.0f}°) for {duration:.1f}s. "
                            f"Please keep your eyes on the screen. "
                            f"Repeated turns will result in automatic suspension."
                        )

                    events.append(self._raise_event(
                        "SUSTAINED_HEAD_TURN", severity, desc,
                        {"yaw": round(yaw, 2), "direction": direction,
                         "duration_s": round(duration, 1),
                         "turn_count": self._head_turn_count},
                    ))
            else:
                # Head returned to normal — reset timer and the per-turn flag
                self._head_turn_since       = None
                self._head_currently_turned = False

        # ── 3. Object detection ───────────────────────────────────────────────
        if self._object_net is not None:
            obj_evt = self._detect_objects(frame, w, h)
            if obj_evt:
                events.append(obj_evt)

        if events:
            severity_order = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
            events.sort(key=lambda e: severity_order.get(e.severity, 0), reverse=True)
            return events[0]
        return None

    def _build_multi_face_event(self, n_faces: int, faces: list) -> Optional[AgentEvent]:
        """
        Build a graduated-severity event — fully automatic, no human review.

        count = 1 → MEDIUM  (automatic warning banner to student)
        count ≥ 2 → CRITICAL (automatic suspension via Orchestrator)
        """
        count = self._confirmed_event_count

        if count < MULTI_FACE_WARN_COUNT:
            return None   # not reached warn threshold yet

        if count < MULTI_FACE_SUSPEND_COUNT:
            severity = "MEDIUM"
            desc = (
                f"Another person detected in frame ({n_faces} faces, confirmation #{count}). "
                f"WARNING: You must be alone during the exam. "
                f"One more detection will automatically suspend your session."
            )
        else:
            severity = "CRITICAL"
            desc = (
                f"Multiple people confirmed {count} times ({n_faces} faces in frame). "
                f"Exam automatically suspended — exam integrity policy violated."
            )

        return self._raise_event(
            "MULTIPLE_FACES", severity, desc,
            {"face_count": n_faces, "confirmed_count": count,
             "face_boxes": [list(map(int, f)) for f in faces]},
        )

    # ── Risk scoring ──────────────────────────────────────────────────────────

    def get_risk(self) -> float:
        """
        Graduated risk based on confirmed multiple-face event count:
          1 confirmation (WARN)    → 0.55  (above WARN threshold 0.50)
          2 confirmations (FLAG)   → 0.75  (above FLAG threshold 0.70)
          3+ confirmations (SUSPEND) → 1.0 (above SUSPEND threshold 0.90)

        Other violations add to incremental score.
        """
        n = self._confirmed_event_count
        if n >= MULTI_FACE_SUSPEND_COUNT:
            multi_score = 1.0      # 2nd+ confirmation → auto-suspend
        elif n >= MULTI_FACE_WARN_COUNT:
            multi_score = 0.60     # 1st confirmation → warning
        else:
            multi_score = 0.0

        # Phone: graduated risk — 1st → FLAG, 2nd → near-suspend, 3rd+ → 1.0
        if self._phone_count >= PHONE_SUSPEND_COUNT:
            phone_score = 1.0
        elif self._phone_count == 2:
            phone_score = 0.88   # near SUSPEND threshold
        elif self._phone_count == 1:
            phone_score = 0.72   # above FLAG threshold (0.70)
        else:
            phone_score = 0.0

        # Absence: 1st → 0.55 (WARN), 2nd → 1.0 (suspend via hard override)
        if self._absence_count >= ABSENCE_SUSPEND_COUNT:
            absence_score = 1.0
        elif self._absence_count == 1:
            absence_score = 0.55
        else:
            absence_score = 0.0

        # Head turn: graduated risk matching severity thresholds
        n_turns = self._head_turn_count
        if n_turns >= HEAD_TURN_SUSPEND_COUNT:
            head_score = 0.90
        elif n_turns >= HEAD_TURN_FLAG_COUNT:
            head_score = 0.65
        elif n_turns >= HEAD_TURN_WARN_COUNT:
            head_score = 0.45
        else:
            head_score = 0.0

        other_score = max(phone_score, absence_score, head_score)
        return float(min(max(multi_score, other_score), 1.0))

    def reset(self):
        self._last_face_time          = time.time()
        self._head_turn_since         = None
        self._head_turn_count         = 0
        self._head_currently_turned   = False
        self._multi_face_streak       = 0
        self._confirmed_event_count   = 0
        self._phone_count             = 0
        self._phone_streak            = 0
        self._phone_event_fired       = False
        self._absence_count           = 0
        self._absence_alerted         = False
        self.events.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _decode(self, b64: str) -> Optional[np.ndarray]:
        try:
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            arr = np.frombuffer(base64.b64decode(b64), np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _detect_faces(
        self, frame: np.ndarray, w: int, h: int, min_confidence: float = None
    ) -> List[Tuple]:
        """
        Return list of (x1, y1, x2, y2, confidence) for every face passing
        both the confidence and minimum-area filters.

        Parameters
        ----------
        min_confidence : override the confidence gate.  Defaults to
            FACE_CONFIDENCE_THRESHOLD (presence detection).  Pass
            MULTI_FACE_CONFIDENCE for the stricter multi-face count.
        """
        if min_confidence is None:
            min_confidence = FACE_CONFIDENCE_THRESHOLD

        frame_area = w * h
        min_area   = MULTI_FACE_MIN_AREA_RATIO * frame_area
        candidates = []   # (box_tuple, conf)

        if self._face_net is not None:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123)
            )
            self._face_net.setInput(blob)
            dets = self._face_net.forward()
            for i in range(dets.shape[2]):
                conf = float(dets[0, 0, i, 2])
                if conf >= min_confidence:
                    box = tuple(
                        (dets[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int).tolist()
                    )
                    x1, y1, x2, y2 = box
                    area = max(0, x2 - x1) * max(0, y2 - y1)
                    if area >= min_area:
                        candidates.append((box, conf))

        elif self._haar is not None:
            min_px = int(min_area ** 0.5)
            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects  = self._haar.detectMultiScale(
                gray, 1.1, 5, minSize=(max(min_px, 40), max(min_px, 40))
            )
            for x, y, fw, fh in rects:
                if fw * fh >= min_area:
                    candidates.append(((x, y, x + fw, y + fh), 1.0))

        # NMS: if two boxes overlap heavily (IOU > 0.35) they are almost certainly
        # the same physical face detected at two slightly different crops (e.g. due
        # to JPEG compression or a slight head tilt).  Keep the higher-confidence box.
        kept = self._nms_boxes(candidates, iou_thresh=0.35)
        return [box for box, _ in kept]

    @staticmethod
    def _nms_boxes(
        boxes_confs: List[Tuple[Tuple, float]], iou_thresh: float = 0.35
    ) -> List[Tuple[Tuple, float]]:
        """
        Greedy NMS on (box, confidence) pairs.
        Processes detections from highest confidence to lowest; suppresses
        any box whose IOU with an already-kept box exceeds iou_thresh.
        """
        if not boxes_confs:
            return []
        boxes_confs = sorted(boxes_confs, key=lambda x: x[1], reverse=True)
        kept: List[Tuple[Tuple, float]] = []
        for box, conf in boxes_confs:
            x1, y1, x2, y2 = box
            suppress = False
            for kbox, _ in kept:
                kx1, ky1, kx2, ky2 = kbox
                ix1 = max(x1, kx1); iy1 = max(y1, ky1)
                ix2 = min(x2, kx2); iy2 = min(y2, ky2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                if inter == 0:
                    continue
                area_a = (x2 - x1) * (y2 - y1)
                area_b = (kx2 - kx1) * (ky2 - ky1)
                iou = inter / (area_a + area_b - inter + 1e-6)
                if iou > iou_thresh:
                    suppress = True
                    break
            if not suppress:
                kept.append((box, conf))
        return kept

    def _get_head_yaw(self, frame: np.ndarray, w: int, h: int) -> Optional[float]:
        try:
            import math
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                return None
            lm = res.multi_face_landmarks[0].landmark
            _LM_IDS = [1, 152, 263, 33, 287, 57]
            _MODEL_PTS = np.array([
                (0.0,0.0,0.0),(0.0,-330.0,-65.0),
                (-225.0,170.0,-135.0),(225.0,170.0,-135.0),
                (-150.0,-150.0,-125.0),(150.0,-150.0,-125.0),
            ], dtype=np.float64)
            img_pts = np.array([(lm[i].x*w, lm[i].y*h) for i in _LM_IDS], dtype=np.float64)
            focal = w
            cam   = np.array([[focal,0,w/2],[0,focal,h/2],[0,0,1]], dtype=np.float64)
            ok, rvec, _ = cv2.solvePnP(_MODEL_PTS, img_pts, cam, np.zeros((4,1)),
                                        flags=cv2.SOLVEPNP_EPNP)
            if not ok:
                return None
            rot, _ = cv2.Rodrigues(rvec)
            sy  = math.sqrt(rot[0,0]**2 + rot[1,0]**2)
            return math.degrees(math.atan2(-rot[2,0], sy))
        except Exception:
            return None

    def _detect_objects(self, frame: np.ndarray, w: int, h: int) -> Optional[AgentEvent]:
        """
        Detect suspicious objects (cell phone, book, laptop) in the frame.

        Uses YOLOv3-tiny when available (preferred):
          • Input blob: 416×416, normalised to [0,1]
          • 3 output layers; each row = [cx,cy,bw,bh, obj_conf, cls_conf×80]
          • Final confidence = obj_conf × cls_conf
          • NMS applied (IoU threshold 0.4) to remove duplicate boxes

        Falls back to MobileNet-SSD COCO if YOLOv3-tiny is unavailable.

        Algorithm note — why YOLO over a pure classifier?
          A classifier (e.g. MobileNet) tells us what the dominant object is.
          YOLO detects MULTIPLE objects simultaneously in a single forward pass,
          so it catches a phone held at the corner of the frame while the
          student's face is still centred — exactly the cheating scenario.
        """
        if self._object_net is None:
            return None

        best_label, best_conf = None, 0.0
        frame_area = w * h

        if self._object_type == "yolo":
            blob = cv2.dnn.blobFromImage(
                frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self._object_net.setInput(blob)
            layer_names = self._object_net.getLayerNames()
            output_layers = [
                layer_names[i - 1]
                for i in self._object_net.getUnconnectedOutLayers().flatten()
            ]
            outputs = self._object_net.forward(output_layers)

            for output in outputs:
                for det in output:
                    scores     = det[5:]
                    cls_id     = int(np.argmax(scores))
                    obj_conf   = float(det[4])
                    cls_conf   = float(scores[cls_id])
                    confidence = obj_conf * cls_conf

                    if confidence <= PHONE_CONF_THRESHOLD:
                        continue
                    if cls_id not in _YOLO_SUSPICIOUS:
                        continue

                    # ── Geometry validation ───────────────────────────────────
                    # YOLO outputs [cx, cy, bw, bh] normalised to [0,1].
                    cx, cy, bw, bh = det[0], det[1], det[2], det[3]
                    pw = bw * w
                    ph = bh * h
                    if pw <= 0 or ph <= 0:
                        continue

                    # Size: reject boxes covering more than PHONE_MAX_AREA_RATIO of
                    # the frame.  A real handheld phone is small; a monitor, laptop
                    # screen, or large book would be much larger.
                    box_area = pw * ph
                    if box_area > PHONE_MAX_AREA_RATIO * frame_area:
                        logger.debug(
                            "[BehaviourAgent] Phone-class detection rejected: "
                            "box area %.1f%% > max %.0f%%",
                            100 * box_area / frame_area,
                            100 * PHONE_MAX_AREA_RATIO,
                        )
                        continue

                    # Aspect ratio: phones are clearly rectangular.
                    # Rejects square objects (sticky notes, coasters) and
                    # thin lines (pens, cables, rulers).
                    aspect = max(pw, ph) / (min(pw, ph) + 1e-6)
                    if not (PHONE_MIN_ASPECT_RATIO <= aspect <= PHONE_MAX_ASPECT_RATIO):
                        logger.debug(
                            "[BehaviourAgent] Phone-class detection rejected: "
                            "aspect ratio %.2f outside [%.1f, %.1f]",
                            aspect, PHONE_MIN_ASPECT_RATIO, PHONE_MAX_ASPECT_RATIO,
                        )
                        continue

                    if confidence > best_conf:
                        best_conf  = confidence
                        best_label = _YOLO_SUSPICIOUS[cls_id]

        elif self._object_type == "ssd":
            blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
            self._object_net.setInput(blob)
            output = self._object_net.forward()
            for det in output[0, 0]:
                score = float(det[2])
                cls   = int(det[1])
                if score <= PHONE_CONF_THRESHOLD or cls not in _SSD_SUSPICIOUS:
                    continue
                # Geometry validation for SSD (box coords normalised to [0,1])
                x1n, y1n, x2n, y2n = det[3], det[4], det[5], det[6]
                bw = (x2n - x1n) * w; bh = (y2n - y1n) * h
                if bw <= 0 or bh <= 0:
                    continue
                if bw * bh > PHONE_MAX_AREA_RATIO * frame_area:
                    continue
                aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
                if not (PHONE_MIN_ASPECT_RATIO <= aspect <= PHONE_MAX_ASPECT_RATIO):
                    continue
                if score > best_conf:
                    best_conf, best_label = score, _SSD_SUSPICIOUS[cls]

        if best_label:
            # ── Temporal debounce ─────────────────────────────────────────────
            # Require PHONE_CONFIRM_FRAMES consecutive frames showing the phone
            # before raising any event.  A single bad frame (reflection, cable,
            # dark background) will never count.
            self._phone_streak += 1

            if self._phone_streak < PHONE_CONFIRM_FRAMES:
                return None   # not yet confirmed — accumulate more frames

            # Streak has reached the threshold.
            # Fire exactly ONE event per streak using _phone_event_fired.
            # Without this flag, frames beyond the threshold (streak=4,5,6…)
            # would also trigger new events if we used >=.
            if not self._phone_event_fired:
                self._phone_event_fired = True
                self._phone_count += 1

                if self._phone_count >= PHONE_SUSPEND_COUNT:
                    return self._raise_event(
                        "PHONE_DETECTED_SUSPEND", "CRITICAL",
                        f"Mobile phone confirmed {self._phone_count} times. "
                        f"Exam automatically suspended — unauthorised device policy violated.",
                        {"object": best_label, "confidence": round(best_conf, 3),
                         "phone_count": self._phone_count},
                    )
                else:
                    return self._raise_event(
                        "PHONE_DETECTED", "HIGH",
                        f"Mobile phone detected (confirmation #{self._phone_count}). "
                        f"Remove it immediately. "
                        f"{PHONE_SUSPEND_COUNT - self._phone_count} more confirmation(s) "
                        f"will automatically suspend your exam.",
                        {"object": best_label, "confidence": round(best_conf, 3),
                         "phone_count": self._phone_count},
                    )
        else:
            # No phone in this frame — reset streak AND the per-streak event flag
            # so the next streak can fire a new event.
            self._phone_streak      = 0
            self._phone_event_fired = False

        return None
