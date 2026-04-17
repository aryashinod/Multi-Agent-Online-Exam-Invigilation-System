"""
audio_agent.py — Audio Monitoring Agent
========================================

ALGORITHM JUSTIFICATION
------------------------

Voice Activity Detection (VAD) : WebRTC VAD (webrtcvad library)
  • Google's WebRTC project includes a GMM-based VAD that operates on
    10 ms / 20 ms / 30 ms frames of 8/16/32 kHz 16-bit mono PCM audio.
  • Aggressiveness modes 0–3 trade recall vs. false-positive rate.
    Mode 2 is used here: aggressive enough to catch whispering but
    conservative enough to ignore keyboard/fan noise.
  • Why not a neural VAD (e.g. Silero VAD, py-webrtcvad-wheels)?
    Silero achieves higher F1 (~0.97 vs ~0.92) but requires PyTorch
    and ~60 ms latency on CPU.  For real-time invigilation we need
    < 30 ms end-to-end; WebRTC VAD delivers ~5 ms per frame.
  • Why not energy threshold alone?
    Simple RMS thresholding produces many false positives from background
    noise.  WebRTC VAD uses spectral features (sub-band energy ratios)
    trained on diverse corpora, giving far fewer false alarms.

Multiple-Speaker Heuristic : Zero-Crossing Rate (ZCR) Variance
  • A single speaker produces a relatively stable ZCR over a short window.
  • Two overlapping speakers generate high-variance ZCR due to interference
    between different fundamental frequencies.
  • Threshold: if ZCR standard deviation over a 1-second window > 0.15,
    a multiple-speaker flag is raised (empirically tuned).
  • Reference: Rabiner & Schafer (1978) — foundational ZCR analysis for
    speech vs. non-speech discrimination.

Whisper Detection : Low RMS + High Spectral Centroid
  • Normal speech has RMS > 0.02 and spectral centroid 500–2000 Hz.
  • Whispered speech has RMS < 0.02 but spectral centroid > 2500 Hz
    (shifted toward sibilants and fricatives).
  • Detecting low-energy high-frequency bursts flags possible whispering.

Audio Intake : Web Audio API → PCM bytes via SocketIO
  • The frontend uses ScriptProcessorNode (or AudioWorkletNode) to capture
    4096-sample chunks at 16 kHz, convert to 16-bit PCM, base64-encode,
    and emit to the server as "audio_chunk" events.
  • 4096 samples @ 16 kHz = 256 ms per chunk; we accumulate 30 ms frames
    internally for the VAD.
"""

import base64
import logging
import math
import time
from collections import deque
from typing import Optional

import numpy as np

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False
    logging.warning("[AudioAgent] webrtcvad not installed — using energy-threshold fallback.")

from .base_agent import BaseAgent, AgentEvent
from ..config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_VAD_AGGRESSIVENESS,
    AUDIO_SPEECH_DURATION,
    AUDIO_SPEECH_GRACE_S,
    AUDIO_MIN_RMS,
    AUDIO_FALLBACK_RMS,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FRAME_DURATION_MS   = 30       # ms per VAD frame (10 | 20 | 30)
FRAME_SAMPLES       = int(AUDIO_SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 @ 16 kHz
WHISPER_RMS_MAX     = 0.015    # below this RMS → quiet enough for whisper
WHISPER_CENTROID_MIN= 2500     # Hz; above this + low RMS → possible whisper
ZCR_VAR_THRESHOLD   = 0.15     # std-dev of ZCR over 1 s → multiple speakers
BACKGROUND_WINDOW_S = 5        # seconds of rolling audio for background analysis

# Adaptive noise floor calibration
CALIBRATION_DURATION_S = 5.0   # seconds at session start used to measure ambient noise
SPEECH_SNR_RATIO       = 2.5   # dynamic gate = noise_floor × 2.5.
                                # Raised from 2.0: extra headroom to keep fan noise peaks
                                # below the gate even when the fan briefly spikes.
MAX_DYNAMIC_RMS        = 0.018  # hard cap raised from 0.012 → 0.018: allows the gate to
                                # sit higher for louder fans without blocking normal speech
                                # (conversational speech RMS is typically 0.025–0.15).


class AudioAgent(BaseAgent):
    """
    Monitors the student's microphone for suspicious audio patterns:
      1. Sustained speech when the student should be silent (background voice)
      2. Multiple speakers / overlapping voices
      3. Whispering (low-energy, high-frequency bursts)

    Receives raw PCM audio from the frontend via SocketIO "audio_chunk" events.
    """

    def __init__(self):
        super().__init__("AudioAgent")
        self._vad: Optional[object] = None

        # Rolling buffer: last BACKGROUND_WINDOW_S seconds of PCM samples
        self._pcm_buffer: deque = deque(maxlen=AUDIO_SAMPLE_RATE * BACKGROUND_WINDOW_S)
        # VAD frame buffer to accumulate exactly FRAME_SAMPLES samples
        self._frame_accum: list = []

        # Counters
        self._speech_frames:   int = 0
        self._silence_frames:  int = 0
        self._whisper_count:   int = 0
        self._multi_spk_count: int = 0

        # Continuous speech tracking (with silence grace period)
        self._speech_start:    Optional[float] = None   # when current speech episode began
        self._last_speech_ts:  Optional[float] = None   # when last speech frame was seen
        self._last_chunk_ts:   float = 0.0

        # ZCR history for multi-speaker detection
        self._zcr_history: deque = deque(maxlen=AUDIO_SAMPLE_RATE // 256)  # ~1 s

        # Adaptive noise floor calibration
        # During the first CALIBRATION_DURATION_S seconds the agent listens to
        # the ambient environment (fan, AC, room tone) and computes an average
        # RMS.  Afterwards, _dynamic_min_rms = noise_floor × SPEECH_SNR_RATIO
        # is used as the silence gate, so steady background noise is never
        # mistaken for speech regardless of how loud the fan is.
        self._calibration_rms:   list          = []
        self._calibration_start: Optional[float] = None
        self._calibrated:        bool           = False
        self._dynamic_min_rms:   float          = AUDIO_MIN_RMS  # updated post-calibration

    def initialise(self):
        if WEBRTCVAD_AVAILABLE:
            self._vad = webrtcvad.Vad(AUDIO_VAD_AGGRESSIVENESS)
            logger.info("[AudioAgent] WebRTC VAD initialised (aggressiveness=%d).",
                        AUDIO_VAD_AGGRESSIVENESS)
        else:
            logger.warning("[AudioAgent] Falling back to RMS energy VAD.")

    def process(self, payload: dict) -> Optional[AgentEvent]:
        """
        Process one audio chunk from the client.

        payload = {
            "student_id": str,
            "audio":      <base64-encoded 16-bit PCM, mono, 16 kHz>,
            "timestamp":  int (client JS ms epoch)
        }
        """
        audio_b64 = payload.get("audio", "")
        if not audio_b64:
            return None

        # Decode PCM bytes → float32 numpy array in [-1, 1]
        try:
            pcm_bytes = base64.b64decode(audio_b64)
            samples   = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as exc:
            logger.debug("[AudioAgent] Decode error: %s", exc)
            return None

        if len(samples) == 0:
            return None

        self._last_chunk_ts = time.time()
        chunk_rms = float(np.sqrt(np.mean(samples ** 2)))
        logger.debug(
            "[AudioAgent] chunk received: %d samples, RMS=%.5f, "
            "speech_start=%s, accum=%d",
            len(samples), chunk_rms,
            f"{time.time()-self._speech_start:.1f}s ago" if self._speech_start else "None",
            len(self._frame_accum),
        )
        self._pcm_buffer.extend(samples.tolist())

        # ── Adaptive noise floor calibration ──────────────────────────────────
        # For the first CALIBRATION_DURATION_S seconds we collect ambient RMS
        # values (fan, AC, room tone) and compute a noise floor.  After that,
        # only frames significantly louder than the floor are sent to the VAD.
        # This prevents steady background noise from ever triggering speech alerts
        # regardless of loudness.
        if not self._calibrated:
            if self._calibration_start is None:
                self._calibration_start = time.time()
                logger.info("[AudioAgent] Calibration started — measuring ambient noise for %.1fs.",
                            CALIBRATION_DURATION_S)
            self._calibration_rms.append(chunk_rms)
            if time.time() - self._calibration_start >= CALIBRATION_DURATION_S:
                if self._calibration_rms:
                    # Use the 75th percentile instead of the mean.
                    # The mean is pulled down by quiet moments and pulled up by
                    # brief loud events (cough, chair scrape) during calibration.
                    # The 75th percentile represents a typical "high ambient" RMS
                    # and is more robust to both outliers.
                    noise_floor = float(np.percentile(self._calibration_rms, 75))
                else:
                    noise_floor = AUDIO_MIN_RMS
                # Dynamic gate: noise_floor × SNR ratio, clamped to [AUDIO_MIN_RMS, MAX_DYNAMIC_RMS].
                raw_gate = noise_floor * SPEECH_SNR_RATIO
                self._dynamic_min_rms = max(AUDIO_MIN_RMS, min(raw_gate, MAX_DYNAMIC_RMS))
                self._calibrated = True
                logger.info(
                    "[AudioAgent] Calibration complete: noise_floor(p75)=%.5f, "
                    "raw_gate=%.5f, dynamic_threshold=%.5f.",
                    noise_floor, raw_gate, self._dynamic_min_rms,
                )
            else:
                # Still calibrating — don't process speech yet
                return None

        # ── ZCR for multi-speaker detection ───────────────────────────────────
        zcr = self._zero_crossing_rate(samples)
        self._zcr_history.append(zcr)

        # ── Run VAD frame-by-frame ─────────────────────────────────────────────
        # Grace-period speech tracking:
        #   _speech_start  : when the current speech episode began (set on first
        #                    speech frame; NOT reset on individual silence frames)
        #   _last_speech_ts: timestamp of the most recent speech frame
        #
        # The episode resets only when silence exceeds AUDIO_SPEECH_GRACE_S.
        # This prevents one VAD mis-classification (a single quiet frame in the
        # middle of real speech) from resetting the entire accumulated duration.
        self._frame_accum.extend(samples.tolist())
        evt = None
        now = time.time()

        while len(self._frame_accum) >= FRAME_SAMPLES:
            frame    = self._frame_accum[:FRAME_SAMPLES]
            self._frame_accum = self._frame_accum[FRAME_SAMPLES:]
            frame_np = np.array(frame, dtype=np.float32)

            is_speech = self._detect_speech(frame_np)

            if is_speech:
                self._speech_frames += 1
                now = time.time()
                self._last_speech_ts = now

                if self._speech_start is None:
                    self._speech_start = now   # start of new speech episode

                # ── Sustained speech check ────────────────────────────────────
                episode_duration = now - self._speech_start
                if episode_duration >= AUDIO_SPEECH_DURATION:
                    self._speech_start   = None   # reset for next episode
                    self._last_speech_ts = None
                    evt = self._raise_event(
                        "BACKGROUND_SPEECH", "HIGH",
                        f"Speech detected for {episode_duration:.1f}s — "
                        "possible dictation, answering from notes, or third-party assistance.",
                        {"duration_s": round(episode_duration, 1),
                         "speech_frames": self._speech_frames},
                    )
                    break

                # ── Whisper check ─────────────────────────────────────────────
                rms      = float(np.sqrt(np.mean(frame_np ** 2)))
                centroid = self._spectral_centroid(frame_np, AUDIO_SAMPLE_RATE)
                if rms < WHISPER_RMS_MAX and centroid > WHISPER_CENTROID_MIN:
                    self._whisper_count += 1
                    if self._whisper_count % 5 == 0:   # alert every 5 whisper frames
                        evt = self._raise_event(
                            "WHISPER_DETECTED", "MEDIUM",
                            f"Possible whispering detected "
                            f"(RMS={rms:.4f}, centroid={centroid:.0f} Hz).",
                            {"rms": round(rms, 4), "centroid_hz": round(centroid, 1)},
                        )
                        break
            else:
                # Silence frame — only reset the episode if silence exceeds the
                # grace period.  Brief gaps (one misfired VAD frame) are ignored.
                self._silence_frames += 1
                now = time.time()
                if (self._last_speech_ts is not None
                        and now - self._last_speech_ts > AUDIO_SPEECH_GRACE_S):
                    # Grace period expired — genuine silence, reset episode
                    self._speech_start   = None
                    self._last_speech_ts = None

        # ── Multi-speaker check (across full chunk) ───────────────────────────
        if len(self._zcr_history) >= 10:
            zcr_std = float(np.std(list(self._zcr_history)))
            if zcr_std > ZCR_VAR_THRESHOLD:
                self._multi_spk_count += 1
                if self._multi_spk_count % 3 == 1:   # alert on 1st, 4th, 7th…
                    evt = self._raise_event(
                        "MULTIPLE_SPEAKERS", "HIGH",
                        f"Multiple overlapping voices detected "
                        f"(ZCR σ={zcr_std:.3f} > threshold {ZCR_VAR_THRESHOLD}).",
                        {"zcr_std": round(zcr_std, 4)},
                    )

        return evt

    def get_risk(self) -> float:
        """
        Risk score based on accumulated audio violations.

        Weights:
          BACKGROUND_SPEECH   × 0.40  (most serious)
          MULTIPLE_SPEAKERS   × 0.35
          WHISPER_DETECTED    × 0.20
        """
        score = (
            self.event_count("BACKGROUND_SPEECH") * 0.40 +
            self.event_count("MULTIPLE_SPEAKERS")  * 0.35 +
            self.event_count("WHISPER_DETECTED")   * 0.20
        )
        return float(min(score, 1.0))

    def reset(self):
        self._pcm_buffer.clear()
        self._frame_accum.clear()
        self._zcr_history.clear()
        self._speech_frames   = 0
        self._silence_frames  = 0
        self._whisper_count   = 0
        self._multi_spk_count = 0
        self._speech_start    = None
        self._last_speech_ts  = None
        # Reset calibration so the next student gets a fresh noise floor measurement
        self._calibration_rms   = []
        self._calibration_start = None
        self._calibrated        = False
        self._dynamic_min_rms   = AUDIO_MIN_RMS
        self.events.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_speech(self, frame: np.ndarray) -> bool:
        """
        Detect speech in a single VAD frame using a three-tier strategy.

        Tier 1 — Silence gate (RMS < AUDIO_MIN_RMS):
          Unconditionally returns False for truly silent frames (digital
          silence, DC offset, 50/60 Hz hum).  Saves VAD cycles and avoids
          false positives from pure noise.

        Tier 2 — Clear speech energy (RMS ≥ 0.015):
          Unconditionally returns True.  Keyboard/mouse clicks are sharp
          transients that rarely sustain a full 30 ms VAD frame above 0.015
          RMS.  Conversational speech on a laptop mic with AGC typically
          sits at 0.03–0.20.  Bypassing the VAD here avoids false negatives
          where WebRTC VAD misclassifies clear speech as noise.

        Tier 3 — Marginal energy (AUDIO_MIN_RMS ≤ RMS < 0.015):
          Soft or whispered speech.  Try WebRTC VAD first (spectral
          features, most accurate); if unavailable or it returns False,
          fall back to AUDIO_FALLBACK_RMS (0.003) — transient noise
          rarely sustains above this in a quiet exam room.

        Rationale for the 0.015 boundary:
          Empirically, normal speech sits 0.03–0.20 RMS; whispers 0.005–
          0.03; keyboard clicks peak briefly but average < 0.01 over 30 ms.
          0.015 safely separates sustained voice from click transients.
        """
        rms = float(np.sqrt(np.mean(frame ** 2)))

        # Tier 1: adaptive silence gate.
        # Uses the calibrated noise floor × SNR ratio so that fan/AC noise
        # (which is measured during calibration) is always below this gate.
        if rms < self._dynamic_min_rms:
            logger.debug("[AudioAgent] Below noise gate: RMS=%.5f < %.5f → not speech",
                         rms, self._dynamic_min_rms)
            return False

        # Tier 2: WebRTC VAD (spectral analysis, not just energy).
        # This is the primary classifier — VAD mode 2 rejects fans/AC/hum
        # because they lack the harmonic formant structure of voiced speech.
        # The earlier RMS bypass is removed: fans can sustain high RMS just
        # like speech, so spectral inspection is essential to avoid false positives.
        if self._vad is not None:
            try:
                pcm_int16 = (frame * 32767).astype(np.int16).tobytes()
                result = self._vad.is_speech(pcm_int16, AUDIO_SAMPLE_RATE)
                logger.debug("[AudioAgent] VAD (mode %d): RMS=%.5f → %s",
                             AUDIO_VAD_AGGRESSIVENESS, rms, result)
                return result
            except Exception as exc:
                logger.warning("[AudioAgent] VAD error: %s — using RMS fallback", exc)

        # Tier 3 (fallback): webrtcvad not installed — use energy threshold only.
        # AUDIO_FALLBACK_RMS (0.003) is above typical fan/AC noise but below
        # normal speech.  Install webrtcvad for accurate detection.
        result = rms > AUDIO_FALLBACK_RMS
        logger.debug("[AudioAgent] RMS fallback: %.5f > %.5f → %s",
                     rms, AUDIO_FALLBACK_RMS, result)
        return result

    @staticmethod
    def _zero_crossing_rate(samples: np.ndarray) -> float:
        """
        ZCR = (1/N) × Σ |sign(x[n]) - sign(x[n-1])| / 2

        Higher ZCR indicates more high-frequency content or multiple speakers.
        """
        if len(samples) < 2:
            return 0.0
        signs  = np.sign(samples)
        crossings = np.sum(np.abs(np.diff(signs))) / 2
        return float(crossings / len(samples))

    @staticmethod
    def _spectral_centroid(samples: np.ndarray, sr: int) -> float:
        """
        Spectral centroid = Σ(f × |X[f]|) / Σ|X[f]|

        Gives the "centre of mass" of the spectrum in Hz.
        Whispered speech has a higher centroid than normal voiced speech.
        """
        if len(samples) == 0:
            return 0.0
        spectrum   = np.abs(np.fft.rfft(samples))
        freqs      = np.fft.rfftfreq(len(samples), d=1.0 / sr)
        total_power = np.sum(spectrum) + 1e-10
        centroid   = float(np.sum(freqs * spectrum) / total_power)
        return centroid
