"""
orchestrator_agent.py — Orchestrator Agent (Multi-Agent Coordinator)
=====================================================================

ALGORITHM JUSTIFICATION
-----------------------

Risk Aggregation : Bayesian Log-Odds Fusion
  ──────────────────────────────────────────
  • Each agent independently produces a probability P(cheat | evidence_i) in
    [0, 1].  Assuming conditional independence among agents given the true
    cheating state (a standard naive-Bayes assumption), the joint posterior is:

        P(cheat | e1, e2, ..., en)
          ∝ P(cheat) × ∏ P(ei | cheat) / P(ei | honest)

  • In log-odds space (λ = log[P(cheat)/(1-P(cheat))]):

        λ_posterior = λ_prior + Σ_i  w_i × logit(P_i)

    where logit(p) = log(p / (1-p)) and w_i is the agent's reliability weight
    (defined in config.AGENT_WEIGHTS).

  • The final posterior probability is recovered via the sigmoid:

        P_final = sigmoid(λ_posterior) = 1 / (1 + exp(-λ_posterior))

  • Why Bayesian log-odds instead of a simple weighted average?
    1. Weighted average does not respect the [0,1] probability constraint — a
       weighted sum of probabilities can exceed 1 if scores are close to 1.
    2. Log-odds fusion correctly handles extreme evidence: two agents both
       reporting P=0.95 gives a much higher combined probability than the
       naive average of 0.95 — which is counter-intuitive but correct.
    3. The approach is equivalent to a Naive Bayes classifier, which is
       well-understood, interpretable, and fast (O(n) in number of agents).
    4. Prior P(cheat) = 0.05 reflects the base rate of academic misconduct
       in online exams (Lancaster & Cotarlan, 2021, found ~15.7% report
       cheating; we use a conservative 5% to reduce false positives).

  • Alternative considered: Dempster-Shafer Theory of Evidence
    Handles deeper uncertainty but is NP-hard for large frames of discernment
    and can produce counter-intuitive results when evidence sources conflict
    (Zadeh's paradox).  Not appropriate here.

  • Alternative considered: Fuzzy Logic aggregation
    Intuitive for rule-based systems but lacks the probabilistic grounding
    that allows risk thresholds to be calibrated against real misconduct rates.

Agent Communication : Asyncio Event Queue
  ──────────────────────────────────────────
  • Agents are run as coroutines scheduled on a shared asyncio event loop.
  • Each agent processes frames and posts AgentEvents to a central asyncio.Queue.
  • The orchestrator consumes the queue, updates risk scores, and emits
    SocketIO events to the admin dashboard.
  • Why asyncio over threading?
    Python's GIL prevents true CPU parallelism with threads, but CV processing
    (OpenCV, MediaPipe) releases the GIL.  asyncio gives cooperative multitasking
    with lower overhead than threading for I/O-bound SocketIO notifications.

Decision Policy : Threshold-based Hard Decisions + Hard Override Rules
  ────────────────────────────────────────────────────────────────────────
  Normal Bayesian path (fully automatic — no human review step):
    P < 0.50               → NONE    (no action)
    0.50 ≤ P < 0.75        → WARN    (automatic warning shown to student)
    P ≥ 0.75               → SUSPEND (automatic suspension, no human needed)

  Hard Override Rules (bypass Bayesian score, suspend immediately):
    Rule 1 — MULTIPLE_FACES_DETECTED:
      BehaviourAgent.get_risk() returns 1.0 permanently once multiple
      faces are seen.  With AGENT_WEIGHTS["behaviour"] = 0.30, logit(1.0)
      dominates the fused score, pushing P → SUSPEND in one step.
      Rationale: second person = unambiguous policy violation.

    Rule 2 — TAB_SWITCH_SUSPEND:
      IntegrityAgent raises TAB_SWITCH_SUSPEND on the 2nd tab switch and
      returns get_risk() = 1.0.  The orchestrator also checks for this
      event explicitly and suspends WITHOUT waiting for the Bayesian score
      to propagate — ensuring zero-latency suspension.
      Rationale: two tab switches within one exam session is beyond
      reasonable doubt an intentional act (Lancaster & Cotarlan, 2021).

  These thresholds are configurable in config.py (RISK_WARN/FLAG/SUSPEND_THRESHOLD).
"""

import math
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any

from .base_agent      import BaseAgent, AgentEvent
from .identity_agent  import IdentityAgent
from .gaze_agent      import GazeAgent
from .behaviour_agent import BehaviourAgent
from .integrity_agent import IntegrityAgent
from .audio_agent     import AudioAgent
from ..config import (
    AGENT_WEIGHTS,
    RISK_WARN_THRESHOLD,
    RISK_SUSPEND_THRESHOLD,
)

# Event types that trigger IMMEDIATE suspension regardless of Bayesian score.
#
# TAB_SWITCH_SUSPEND       — 2nd browser tab switch (IntegrityAgent)
# IDENTITY_MISMATCH_SUSPEND— 3 consecutive face mismatches (IdentityAgent)
#                            Indicates a different person is sitting the exam.
# STUDENT_ABSENT_SUSPEND   — 2nd confirmed absence > 30s (BehaviourAgent)
# PHONE_DETECTED_SUSPEND   — 2nd confirmed phone/device detection (BehaviourAgent)
#
# NOTE: MULTIPLE_FACES uses a graduated policy (warn→flag→suspend) handled
# via CRITICAL severity check below, NOT this set.
_IMMEDIATE_SUSPEND_EVENTS = {
    "TAB_SWITCH_SUSPEND",
    "IDENTITY_MISMATCH_SUSPEND",
    "STUDENT_ABSENT_SUSPEND",
    "PHONE_DETECTED_SUSPEND",
}

logger = logging.getLogger(__name__)


def _logit(p: float) -> float:
    """log-odds of probability p, clamped to avoid log(0)."""
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    """Inverse logit."""
    return 1.0 / (1.0 + math.exp(-x))


class OrchestratorAgent(BaseAgent):
    """
    Coordinates all four specialised agents and produces a unified risk score.

    Usage:
      orch = OrchestratorAgent()
      orch.start()

      # On each webcam frame from SocketIO:
      result = orch.process({"frame": b64, "student_id": sid})
      # result contains: risk_score, action, triggered_events

      # On browser integrity events:
      result = orch.process_integrity_event({"event_type": "TAB_SWITCH", ...})
    """

    # Base-rate prior P(cheating) = 0.05
    # logit(0.05) ≈ -2.944
    _PRIOR_LOGODDS: float = _logit(0.05)

    def __init__(self):
        super().__init__("OrchestratorAgent")
        self.identity_agent  = IdentityAgent()
        self.gaze_agent      = GazeAgent()
        self.behaviour_agent = BehaviourAgent()
        self.integrity_agent = IntegrityAgent()
        self.audio_agent     = AudioAgent()

        self._agents = {
            "identity":  self.identity_agent,
            "gaze":      self.gaze_agent,
            "behaviour": self.behaviour_agent,
            "integrity": self.integrity_agent,
            "audio":     self.audio_agent,
        }

        self._current_risk: float = 0.0
        self._current_action: str = "NONE"
        self._session_log: List[Dict] = []
        self._student_id: Optional[str] = None
        self._suspended: bool = False

        # asyncio queue for event passing (used by SocketIO integration)
        self._event_queue: Optional[asyncio.Queue] = None

    def initialise(self):
        """Start all sub-agents."""
        for name, agent in self._agents.items():
            try:
                agent.start()
                logger.info("[Orchestrator] %s started.", name)
            except Exception as exc:
                logger.error("[Orchestrator] Failed to start %s: %s", name, exc)

    def set_student(self, student_id: str, session_token: str = ""):
        """Configure all agents for the current student."""
        self._student_id = student_id
        self._suspended  = False
        self._current_risk = 0.0
        self._current_action = "NONE"
        self._session_log.clear()
        self.identity_agent.set_current_student(student_id)
        for agent in self._agents.values():
            agent.reset()
        logger.info("[Orchestrator] Session configured for student %s.", student_id)

    # ── Core process ──────────────────────────────────────────────────────────

    def process(self, payload: dict) -> Dict[str, Any]:
        """
        Process one webcam frame through all relevant agents.

        Steps:
          1. Run IdentityAgent, GazeAgent, BehaviourAgent (vision agents).
          2. Check for IMMEDIATE SUSPEND events (multiple faces).
          3. Compute Bayesian log-odds fused risk score.
          4. Determine action (NONE / WARN / FLAG / SUSPEND).
          5. Return structured result dict for SocketIO broadcast.

        payload = { "frame": <base64 JPEG>, "student_id": str }
        """
        if self._suspended:
            return self._make_result([], "SUSPENDED")

        triggered_events = []

        for name in ("identity", "gaze", "behaviour"):
            agent = self._agents[name]
            if not agent._active:
                continue
            try:
                # Capture ALL events raised in this call, not just the return value.
                # Some agents (e.g. BehaviourAgent) can raise multiple events in one
                # frame (e.g. phone detected AND head turn at the same time).
                # We snapshot the event-list length before and after to get every
                # new AgentEvent appended during this process() call.
                prev_count = len(agent.events)
                agent.process(payload)
                new_events = list(agent.events[prev_count:])
                triggered_events.extend(new_events)
                for evt in new_events:
                    self._session_log.append(evt.to_dict())
            except Exception as exc:
                logger.error("[Orchestrator] Error in %s.process(): %s", name, exc)

        # ── Hard override check: immediate suspension events ───────────────────
        # TAB_SWITCH_SUSPEND (2nd tab switch) → instant suspension.
        # MULTIPLE_FACES CRITICAL (3rd confirmation) → instant suspension.
        # All other MULTIPLE_FACES events (MEDIUM/HIGH) → warn/flag only.
        for evt in triggered_events:
            is_tab_suspend    = evt.event_type in _IMMEDIATE_SUSPEND_EVENTS
            is_multi_critical = (evt.event_type == "MULTIPLE_FACES"
                                 and evt.severity == "CRITICAL")
            if is_tab_suspend or is_multi_critical:
                self._suspended      = True
                self._current_risk   = 1.0
                self._current_action = "SUSPEND"
                logger.critical(
                    "[Orchestrator] IMMEDIATE SUSPEND for %s — hard override: %s (severity=%s).",
                    self._student_id, evt.event_type, evt.severity,
                )
                return self._make_result(triggered_events, "SUSPEND", 1.0)

        risk, action = self._fuse_and_decide()
        self._current_risk   = risk
        self._current_action = action

        if action == "SUSPEND":
            self._suspended = True
            logger.critical(
                "[Orchestrator] EXAM SUSPENDED for %s (Bayesian risk=%.3f).",
                self._student_id, risk,
            )

        return self._make_result(triggered_events, action, risk)

    def process_integrity_event(self, payload: dict) -> Dict[str, Any]:
        """
        Handle a browser integrity event (copy, paste, tab-switch, etc.).
        Routed separately from webcam frames for efficiency.

        Hard override: TAB_SWITCH_SUSPEND triggers immediate suspension
        without waiting for the Bayesian score to accumulate.
        """
        evt = None
        try:
            evt = self.integrity_agent.process(payload)
            if evt:
                self._session_log.append(evt.to_dict())
        except Exception as exc:
            logger.error("[Orchestrator] Error in integrity_agent.process(): %s", exc)

        # ── Hard override: 2nd tab switch = immediate suspension ──────────────
        if evt and evt.event_type == "TAB_SWITCH_SUSPEND":
            self._suspended      = True
            self._current_risk   = 1.0
            self._current_action = "SUSPEND"
            logger.critical(
                "[Orchestrator] IMMEDIATE SUSPEND for %s — 2nd tab switch detected.",
                self._student_id,
            )
            return {
                "risk_score": 1.0,
                "action":     "SUSPEND",
                "events":     [evt.to_dict()],
                "breakdown":  self.get_risk_breakdown(),
            }

        risk, action = self._fuse_and_decide()
        self._current_risk   = risk
        self._current_action = action
        return {
            "risk_score": round(risk, 4),
            "action":     action,
            "events":     [evt.to_dict()] if evt else [],
            "breakdown":  self.get_risk_breakdown(),
        }

    # ── Bayesian risk fusion ──────────────────────────────────────────────────

    def _fuse_and_decide(self):
        """
        Fuse agent risk scores using Bayesian log-odds accumulation.

        Derivation:
          λ_posterior = λ_prior + Σ_i  w_i × logit(P_adj_i)

        The prior λ_prior = logit(0.05) encodes our background belief that
        5% of students cheat.

        Key fix — neutral-agent problem:
          A raw agent risk of 0.0 gives logit(0) ≈ -13.8, which strongly
          pushes the fused score toward 0 even when other agents flag
          genuine violations.  Agents with no evidence should be NEUTRAL,
          not exculpatory.  We handle this in two steps:

          1. Skip any agent whose risk < 0.01 (no evidence → no contribution).
          2. Map the remaining risk scores into [0.5, 0.975] via:
               p_adj = 0.5 + min(p_i, 0.95) * 0.5
             This ensures a positive signal always pushes log-odds upward,
             and the mapping is monotone so relative ordering is preserved.

        Returns (risk_float, action_str).
        """
        log_odds     = self._PRIOR_LOGODDS
        any_evidence = False

        for name, agent in self._agents.items():
            p_i = agent.get_risk()
            if p_i < 0.01:
                continue                          # no evidence → neutral, skip
            any_evidence = True
            w_i   = AGENT_WEIGHTS.get(name, 0.5)
            p_adj = 0.5 + min(p_i, 0.95) * 0.5  # map [0.01,1] → [0.505, 0.975]
            log_odds += w_i * _logit(p_adj)

        # If every agent reported zero evidence, the Bayesian prior alone would
        # produce sigmoid(logit(0.05)) = 5 % — confusing and misleading to the
        # student.  The prior is a population-level belief, not a per-student
        # accusation; when we have NO evidence at all, report 0 % cleanly.
        if not any_evidence:
            return 0.0, "NONE"

        risk = _sigmoid(log_odds)

        if risk >= RISK_SUSPEND_THRESHOLD:
            action = "SUSPEND"
        elif risk >= RISK_WARN_THRESHOLD:
            action = "WARN"
        else:
            action = "NONE"

        return risk, action

    # ── Session management ────────────────────────────────────────────────────

    def begin_exam(self, student_id: str, session_token: str) -> Dict:
        """Delegate to IntegrityAgent and return shuffled question set."""
        self.set_student(student_id, session_token)
        return self.integrity_agent.begin_exam(student_id, session_token)

    def submit_exam(self, student_id: str, answers: Dict, order_hash: str) -> Dict:
        """Submit answers through IntegrityAgent and return grading result."""
        # Capture the event count before submission so we can detect any new
        # events raised inside submit_answers (QUESTION_TAMPERING, LATE_SUBMISSION).
        # These are raised via _raise_event (which appends to integrity_agent.events)
        # but the return value is discarded inside submit_answers, so without this
        # we'd lose them from the session log and admin broadcast.
        prev_count = len(self.integrity_agent.events)

        result = self.integrity_agent.submit_answers(student_id, answers, order_hash)

        # Append any newly raised integrity events to the orchestrator session log
        new_events = list(self.integrity_agent.events[prev_count:])
        for evt in new_events:
            self._session_log.append(evt.to_dict())

        risk, _ = self._fuse_and_decide()
        result["final_risk_score"]    = round(risk, 4)
        result["session_events"]      = self.get_full_log()
        result["submission_events"]   = [e.to_dict() for e in new_events]
        return result

    def register_student(self, student_id: str, frame_b64: str) -> bool:
        """Store a reference face encoding for identity verification."""
        return self.identity_agent.store_reference(student_id, frame_b64)

    # ── Dashboard queries ─────────────────────────────────────────────────────

    def get_full_log(self) -> List[Dict]:
        return list(self._session_log)

    def get_risk_breakdown(self) -> Dict:
        """Return per-agent risk scores alongside the fused score."""
        breakdown = {name: round(agent.get_risk(), 4)
                     for name, agent in self._agents.items()}
        breakdown["fused"] = round(self._current_risk, 4)
        breakdown["action"] = self._current_action
        return breakdown

    def get_risk(self) -> float:
        return self._current_risk

    def reset(self):
        for agent in self._agents.values():
            agent.reset()
        self._current_risk   = 0.0
        self._current_action = "NONE"
        self._session_log.clear()
        self._student_id = None
        self._suspended  = False
        self.events.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _make_result(
        self,
        events: List[AgentEvent],
        action: str,
        risk: Optional[float] = None,
    ) -> Dict:
        if risk is None:
            risk = self._current_risk
        return {
            "risk_score":  round(risk, 4),
            "action":      action,
            "suspended":   self._suspended,
            "events":      [e.to_dict() for e in events],
            "breakdown":   self.get_risk_breakdown(),
            "timestamp":   time.time(),
        }
