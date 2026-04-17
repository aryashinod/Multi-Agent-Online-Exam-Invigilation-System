"""
base_agent.py — Abstract base class for all invigilation agents.

Design Pattern: Template Method
  Each specialised agent inherits from BaseAgent and must implement:
    - initialise()  : load models, warm-up state
    - process()     : run one analysis cycle on a frame/event
    - get_risk()    : return a float in [0, 1] representing cheating probability
    - reset()       : clear per-student state between exam sessions

The orchestrator calls every agent through this uniform interface, enabling
the system to be extended with new agents (e.g. audio analysis) without
modifying the orchestration logic — satisfying the Open/Closed Principle.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentEvent:
    """
    Structured event raised by an agent when it detects an anomaly.

    Fields:
        agent_id    : which agent raised the event
        event_type  : short machine-readable label  (e.g. "GAZE_AWAY")
        severity    : LOW | MEDIUM | HIGH
        description : human-readable explanation
        timestamp   : Unix epoch
        metadata    : arbitrary extra data (bounding boxes, angles, etc.)
    """
    agent_id:    str
    event_type:  str
    severity:    str                     # "LOW" | "MEDIUM" | "HIGH"
    description: str
    timestamp:   float = field(default_factory=time.time)
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "agent_id":    self.agent_id,
            "event_type":  self.event_type,
            "severity":    self.severity,
            "description": self.description,
            "timestamp":   self.timestamp,
            "metadata":    self.metadata,
        }


class BaseAgent(ABC):
    """
    Abstract base class — all invigilation agents must extend this.

    The class maintains:
      - self.events  : chronological list of AgentEvents this session
      - self._active : guards against double-initialisation
    """

    def __init__(self, agent_id: str):
        self.agent_id  = agent_id
        self.events:   List[AgentEvent] = []
        self._active   = False
        self._risk     = 0.0        # cached risk score in [0, 1]
        logger.info("[%s] Agent created.", self.agent_id)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Initialise resources and mark agent as active."""
        if not self._active:
            self.initialise()
            self._active = True
            logger.info("[%s] Agent started.", self.agent_id)

    def stop(self):
        """Release resources."""
        if self._active:
            self._active = False
            logger.info("[%s] Agent stopped.", self.agent_id)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def initialise(self):
        """Load models, open streams, warm up state."""
        ...

    @abstractmethod
    def process(self, payload: Any) -> Optional[AgentEvent]:
        """
        Analyse one unit of input (a video frame, a browser event, etc.).

        Returns an AgentEvent if an anomaly was detected, else None.
        The returned event is automatically appended to self.events.
        """
        ...

    @abstractmethod
    def get_risk(self) -> float:
        """
        Return a probability in [0, 1] that the student is cheating,
        based on the agent's accumulated evidence this session.
        """
        ...

    @abstractmethod
    def reset(self):
        """Clear all per-session state (called between students / retakes)."""
        ...

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _raise_event(
        self,
        event_type:  str,
        severity:    str,
        description: str,
        metadata:    Optional[Dict] = None,
    ) -> AgentEvent:
        """
        Convenience method: create, record, and return an AgentEvent.
        """
        evt = AgentEvent(
            agent_id    = self.agent_id,
            event_type  = event_type,
            severity    = severity,
            description = description,
            metadata    = metadata or {},
        )
        self.events.append(evt)
        logger.warning("[%s] %s — %s (severity=%s)", self.agent_id,
                       event_type, description, severity)
        return evt

    def get_event_history(self) -> List[Dict]:
        """Return all events this session as serialisable dicts."""
        return [e.to_dict() for e in self.events]

    def event_count(self, event_type: Optional[str] = None) -> int:
        """Count events, optionally filtered by type."""
        if event_type is None:
            return len(self.events)
        return sum(1 for e in self.events if e.event_type == event_type)
