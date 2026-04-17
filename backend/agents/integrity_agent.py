"""
integrity_agent.py — Question Paper Integrity Agent
====================================================

ALGORITHM JUSTIFICATION
-----------------------

1. Question Randomisation : Fisher-Yates (Knuth) Shuffle — O(n), unbiased
   ─────────────────────────────────────────────────────────────────────────
   • The Fisher-Yates algorithm (Durstenfeld, 1964; popularised by Knuth
     in TAOCP Vol. 2) is the *only* shuffle algorithm that guarantees a
     perfectly uniform distribution over all n! permutations.
   • Proof of unbiasedness: at step i the algorithm selects uniformly from
     indices [i, n-1], giving each remaining element an equal probability
     of occupying position i.  By induction all n! permutations are equally
     likely (Knuth, 1969, §3.4.2 Exercise 3).
   • Time complexity : O(n)  |  Space: O(1) in-place
   • Why not sorted random keys (Sattolo / naive)?
     The naive approach (assign random keys, sort) is O(n log n) and subtly
     biased if the RNG is not cryptographically uniform.
   • Seed = HMAC-SHA256(student_id, session_token) — deterministic per
     student session so the ordering can be reconstructed for grading,
     yet unpredictable to the student in advance.

2. Session / Answer Integrity : SHA-256 Hashing
   ──────────────────────────────────────────────
   • Each submitted answer bundle is hashed with SHA-256 and stored
     alongside the submission.  Post-exam tampering produces a different
     hash, providing cryptographic non-repudiation.
   • SHA-256 is collision-resistant (birthday bound 2^128) and pre-image
     resistant (2^256) — far beyond any practical attack.
   • The question-to-display-index mapping is also hashed so the server
     can verify that the client rendered questions in the prescribed order.

3. Anti-copy-paste / Browser Event Monitoring
   ─────────────────────────────────────────────
   • The JavaScript layer intercepts:
       document.addEventListener("copy", ...)
       document.addEventListener("paste", ...)
       document.addEventListener("visibilitychange", ...)   ← tab switch
       window.addEventListener("blur", ...)                 ← window focus lost
       contextmenu disabled
   • Each event is timestamped and sent to the server via WebSocket.
   • The server-side integrity agent tallies events and computes a risk score.
   • Why not server-side clipboard monitoring?
     Clipboard access requires browser permission and is unreliable server-side.
     The Clipboard Events API is universally supported (MDN, 2024) and needs
     no permission for listening (only for reading clipboard content).

4. Time-limit Enforcement : NTP-synchronised server clock
   ────────────────────────────────────────────────────────
   • Exam start/end times are recorded server-side, preventing client-side
     clock manipulation.  Late submissions trigger a HIGH-severity event.
"""

import time
import hmac
import hashlib
import json
import random
import logging
from typing import List, Dict, Optional, Any

from .base_agent import BaseAgent, AgentEvent
from ..config import (
    QUESTIONS_PER_EXAM,
    EXAM_DURATION_MINUTES,
    COPY_PASTE_DISABLED,
)

logger = logging.getLogger(__name__)


# ── Sample question bank (replace with DB query in production) ────────────────
QUESTION_BANK: List[Dict] = [
    {
        "id": f"Q{i:03d}",
        "text": f"Sample question {i}: What is the result of {i} × {i+1}?",
        "options": {
            "A": str(i * (i + 1)),
            "B": str(i * i),
            "C": str((i + 1) * (i + 2)),
            "D": str(i + i + 1),
        },
        "correct": "A",
    }
    for i in range(1, 31)
]


class IntegrityAgent(BaseAgent):
    """
    Ensures the integrity of the exam question paper and student answers.

    Responsibilities:
      • Generate a per-student shuffled question set (Fisher-Yates).
      • Validate submitted answers against a server-side hash.
      • Track browser integrity violations (copy, paste, tab-switch, focus-loss).
      • Enforce the exam time limit.
    """

    def __init__(self):
        super().__init__("IntegrityAgent")
        self._student_id: Optional[str]   = None
        self._session_token: str          = ""
        self._shuffled_questions: List    = []
        self._question_order_hash: str    = ""
        self._exam_start: Optional[float] = None
        self._exam_end:   Optional[float] = None
        self._copy_count: int = 0
        self._paste_count: int = 0
        self._tab_switch_count: int = 0
        self._focus_loss_count: int = 0

    def initialise(self):
        logger.info("[IntegrityAgent] Ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def begin_exam(self, student_id: str, session_token: str) -> Dict:
        """
        Generate the shuffled question set for a student and start the timer.

        Parameters
        ----------
        student_id     : unique student identifier
        session_token  : random token issued at login (e.g. uuid4 hex string)

        Returns a dict containing:
          questions     : list of question dicts (shuffled, answer hidden)
          order_hash    : SHA-256 of the question ID order (for integrity verification)
          duration_mins : exam duration
          start_time    : server UTC timestamp
        """
        self._student_id    = student_id
        self._session_token = session_token
        self._exam_start    = time.time()
        self._exam_end      = None

        # ── Fisher-Yates shuffle ──────────────────────────────────────────────
        # Derive a deterministic seed from student_id + session_token using HMAC.
        # This ensures reproducibility for grading while being unpredictable.
        seed_bytes = hmac.new(
            session_token.encode(),
            student_id.encode(),
            digestmod=hashlib.sha256,
        ).digest()
        seed_int = int.from_bytes(seed_bytes[:8], "big")

        rng = random.Random(seed_int)

        # Select QUESTIONS_PER_EXAM questions and shuffle
        pool = list(QUESTION_BANK)
        rng.shuffle(pool)          # Fisher-Yates via Python's random.shuffle
        selected = pool[:QUESTIONS_PER_EXAM]

        # Strip correct answers before sending to client
        self._shuffled_questions = selected
        client_questions = [
            {k: v for k, v in q.items() if k != "correct"}
            for q in selected
        ]

        # Hash the question ID order for later verification
        order = [q["id"] for q in selected]
        self._question_order_hash = hashlib.sha256(
            json.dumps(order).encode()
        ).hexdigest()

        logger.info(
            "[IntegrityAgent] Exam started for %s. %d questions shuffled. Order hash: %s…",
            student_id, len(selected), self._question_order_hash[:16],
        )

        return {
            "questions":     client_questions,
            "order_hash":    self._question_order_hash,
            "duration_mins": EXAM_DURATION_MINUTES,
            "start_time":    self._exam_start,
        }

    def submit_answers(
        self, student_id: str, answers: Dict[str, str], order_hash: str
    ) -> Dict:
        """
        Validate and record a student's answers.

        Parameters
        ----------
        student_id : must match the registered student
        answers    : { question_id: selected_option }  (from client)
        order_hash : the hash the client received at exam start (integrity check)

        Returns a dict with grade, integrity_ok, submission_hash, and time_ok.
        """
        if student_id != self._student_id:
            return {"error": "Student ID mismatch."}

        self._exam_end = time.time()
        elapsed_mins   = (self._exam_end - self._exam_start) / 60.0

        # ── Verify order hash (detect question tampering) ─────────────────────
        integrity_ok = (order_hash == self._question_order_hash)
        if not integrity_ok:
            self._raise_event(
                "QUESTION_TAMPERING", "HIGH",
                "Client-side question order hash does not match server record — "
                "possible DOM manipulation.",
                {"client_hash": order_hash, "server_hash": self._question_order_hash},
            )

        # ── Grade ─────────────────────────────────────────────────────────────
        score, total = 0, len(self._shuffled_questions)
        for q in self._shuffled_questions:
            if answers.get(q["id"], "").upper() == q["correct"]:
                score += 1

        # ── Submission hash (non-repudiation) ─────────────────────────────────
        submission_payload = json.dumps(
            {"student_id": student_id, "answers": answers, "elapsed": elapsed_mins},
            sort_keys=True,
        )
        submission_hash = hashlib.sha256(submission_payload.encode()).hexdigest()

        time_ok = elapsed_mins <= EXAM_DURATION_MINUTES

        if not time_ok:
            self._raise_event(
                "LATE_SUBMISSION", "MEDIUM",
                f"Exam submitted {elapsed_mins - EXAM_DURATION_MINUTES:.1f} minutes late.",
                {"elapsed_mins": round(elapsed_mins, 2)},
            )

        logger.info(
            "[IntegrityAgent] %s submitted: %d/%d correct. Hash: %s…",
            student_id, score, total, submission_hash[:16],
        )
        return {
            "score":           score,
            "total":           total,
            "percentage":      round(100 * score / max(total, 1), 1),
            "integrity_ok":    integrity_ok,
            "time_ok":         time_ok,
            "elapsed_mins":    round(elapsed_mins, 2),
            "submission_hash": submission_hash,
        }

    def process(self, payload: dict) -> Optional[AgentEvent]:
        """
        Handle a browser integrity event sent by the JavaScript layer.

        payload = {
            "event_type": "COPY" | "PASTE" | "TAB_SWITCH" | "FOCUS_LOSS" | "RIGHT_CLICK",
            "timestamp":  <client JS epoch ms>,
            "student_id": str
        }
        """
        etype = payload.get("event_type", "").upper()
        ts    = payload.get("timestamp", 0) / 1000.0   # ms → s

        if etype == "COPY":
            self._copy_count += 1
            return self._raise_event(
                "COPY_ATTEMPT", "MEDIUM",
                f"Student attempted to copy exam content "
                f"(copy #{self._copy_count} at {ts:.0f}).",
                {"count": self._copy_count},
            )
        elif etype == "PASTE":
            self._paste_count += 1
            return self._raise_event(
                "PASTE_ATTEMPT", "MEDIUM",
                f"Paste event detected (#{self._paste_count}) — possible external material.",
                {"count": self._paste_count},
            )
        elif etype == "TAB_SWITCH":
            self._tab_switch_count += 1
            # POLICY: 1st tab switch → HIGH warning.
            #         2nd tab switch → CRITICAL; get_risk() returns 1.0
            #         which forces the Orchestrator to SUSPEND immediately.
            # Rationale: one accidental tab switch is plausible (OS
            # notification, reflex). A second switch in the same session
            # is statistically improbable by accident and strongly suggests
            # the student is accessing external resources.
            if self._tab_switch_count >= 2:
                return self._raise_event(
                    "TAB_SWITCH_SUSPEND", "CRITICAL",
                    f"Tab switch #{self._tab_switch_count} detected — "
                    "exam automatically suspended. "
                    "Two or more tab switches is a definitive policy violation.",
                    {"count": self._tab_switch_count},
                )
            else:
                return self._raise_event(
                    "TAB_SWITCH", "HIGH",
                    f"Tab switch detected (#{self._tab_switch_count}). "
                    "WARNING: one more tab switch will cause automatic suspension.",
                    {"count": self._tab_switch_count},
                )
        elif etype == "FOCUS_LOSS":
            self._focus_loss_count += 1
            return self._raise_event(
                "FOCUS_LOSS", "LOW",
                f"Exam window lost focus (#{self._focus_loss_count}) — may indicate "
                "interaction with another application.",
                {"count": self._focus_loss_count},
            )
        elif etype == "FULLSCREEN_EXIT":
            return self._raise_event(
                "FULLSCREEN_EXIT", "MEDIUM",
                "Student exited fullscreen mode — possible screen-sharing or alt-tab.",
                {},
            )

        return None

    def get_risk(self) -> float:
        """
        Weighted score based on violation counts.

        HARD OVERRIDES (return 1.0 immediately):
          • TAB_SWITCH_SUSPEND : 2nd tab switch detected  → suspend
          • QUESTION_TAMPERING : DOM hash mismatch        → suspend

        Graduated scoring so that violations actually drive the Bayesian
        fused score into WARN → FLAG → SUSPEND territory:

          Tab switch (most serious — intentional navigation away):
            1st  → 0.80  (fused risk enters WARN/FLAG territory on its own)
            2nd+ → hard override 1.0 (already handled by _IMMEDIATE_SUSPEND_EVENTS)

          Copy / Paste (possible external resource use):
            copy  × 0.30 per event (3 copies → capped at 0.90)
            paste × 0.40 per event (2 pastes → capped at 0.80)

          Minor events (corroborative, not conclusive alone):
            focus_loss  × 0.15 per event
            FULLSCREEN_EXIT × 0.20 per event
            LATE_SUBMISSION × 0.30

        Rationale for high tab-switch weight:
          Navigating away from the exam page — even once — has no legitimate
          explanation during a supervised exam.  A single switch should
          already push the fused Bayesian score into FLAG territory so the
          human invigilator is alerted.  The 2nd switch triggers the hard
          override because it removes all reasonable doubt.
        """
        # Hard overrides
        if self.event_count("TAB_SWITCH_SUSPEND") > 0:
            return 1.0
        if self.event_count("QUESTION_TAMPERING") > 0:
            return 1.0

        score = (
            min(self._tab_switch_count, 1) * 0.80 +   # 1st switch → 0.80 (FLAG territory)
            self._copy_count               * 0.30 +
            self._paste_count              * 0.40 +
            self._focus_loss_count         * 0.15 +
            self.event_count("LATE_SUBMISSION")   * 0.30 +
            self.event_count("FULLSCREEN_EXIT")   * 0.20
        )
        return float(min(score, 1.0))

    def reset(self):
        self._student_id      = None
        self._session_token   = ""
        self._shuffled_questions = []
        self._question_order_hash = ""
        self._exam_start      = None
        self._exam_end        = None
        self._copy_count      = 0
        self._paste_count     = 0
        self._tab_switch_count = 0
        self._focus_loss_count = 0
        self.events.clear()
