/**
 * webcam_monitor.js
 * ─────────────────
 * Captures webcam frames at FRAME_RATE fps and transmits them to the
 * backend via SocketIO for real-time invigilation analysis.
 *
 * Algorithm note (Frame Capture):
 *   We use requestAnimationFrame + a timing gate rather than setInterval
 *   to avoid frame-queue build-up when the tab is backgrounded.
 *   Frames are JPEG-encoded at 60% quality — sufficient for face detection
 *   while keeping each payload < 20 KB (300×225 @ 60%).
 *
 * Exponential Backoff on errors:
 *   If frame transmission fails (network blip), we apply exponential backoff
 *   (wait = min(2^retries × 500ms, 30s)) to avoid hammering the server.
 */

const FRAME_RATE     = 2;     // frames per second to send for analysis
const JPEG_QUALITY   = 0.6;
const FRAME_INTERVAL = Math.floor(1000 / FRAME_RATE);

let _webcamStream    = null;
let _webcamVideo     = null;
let _webcamCanvas    = null;
let _webcamCtx       = null;
let _webcamSocket    = null;
let _webcamStudentId = null;
let _lastFrameTime   = 0;
let _retryCount      = 0;
let _frameLoopId     = null;
let _webcamActive    = false;

/**
 * Initialise webcam capture.
 * @param {string} studentId  - the current student's ID
 * @param {object} socket     - connected SocketIO instance
 */
async function initWebcam(studentId, socket) {
  _webcamStudentId = studentId;
  _webcamSocket    = socket;

  // Create off-screen canvas for frame capture
  _webcamCanvas = document.createElement("canvas");
  _webcamCtx    = _webcamCanvas.getContext("2d");

  // Use the thumbnail video element in the exam header
  _webcamVideo = document.getElementById("thumb-video");

  try {
    _webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    _webcamVideo.srcObject = _webcamStream;
    _webcamActive = true;

    const track    = _webcamStream.getVideoTracks()[0];
    const settings = track.getSettings();
    _webcamCanvas.width  = settings.width  || 320;
    _webcamCanvas.height = settings.height || 240;

    updateWebcamStatus("🟢 Camera active", "active");
    _frameLoop();

    // Start audio monitoring in parallel with video
    initAudioMonitor(studentId, socket);

  } catch (err) {
    updateWebcamStatus("🔴 Camera denied", "error");
    console.error("[webcam_monitor] Camera error:", err);
  }
}

/** Continuous frame-capture loop using requestAnimationFrame timing gate. */
function _frameLoop() {
  if (!_webcamActive) return;
  _frameLoopId = requestAnimationFrame(() => {
    const now = performance.now();
    if (now - _lastFrameTime >= FRAME_INTERVAL) {
      _lastFrameTime = now;
      _captureAndSend();
    }
    _frameLoop();
  });
}

/** Capture one frame and emit to server. */
function _captureAndSend() {
  if (!_webcamVideo || _webcamVideo.readyState < 2) return;

  try {
    _webcamCtx.drawImage(_webcamVideo, 0, 0,
      _webcamCanvas.width, _webcamCanvas.height);
    const b64Frame = _webcamCanvas.toDataURL("image/jpeg", JPEG_QUALITY);

    _webcamSocket.emit("frame", {
      student_id: _webcamStudentId,
      frame:      b64Frame,
      ts:         Date.now(),
    });
    _retryCount = 0;   // reset backoff on success

  } catch (err) {
    _retryCount++;
    const backoff = Math.min(Math.pow(2, _retryCount) * 500, 30000);
    console.warn(`[webcam_monitor] Frame send error (retry #${_retryCount}, backoff ${backoff}ms):`, err);
    if (_webcamActive) setTimeout(_frameLoop, backoff);
  }
}

/** Stop the webcam stream and cancel the frame loop. */
function stopWebcam() {
  _webcamActive = false;
  if (_frameLoopId) cancelAnimationFrame(_frameLoopId);
  if (_webcamStream) _webcamStream.getTracks().forEach(t => t.stop());
  updateWebcamStatus("⚫ Camera off", "dead");
  stopAudioMonitor();
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio Monitor
// ─────────────────────────────────────────────────────────────────────────────
//
// Algorithm notes:
//   • Web Audio API ScriptProcessorNode captures 4096-sample buffers at the
//     browser's native sample rate (usually 44.1 kHz or 48 kHz).
//   • We resample to 16 kHz (required by WebRTC VAD on the backend) using a
//     simple decimation approach via OfflineAudioContext.
//   • Each buffer is converted to 16-bit signed PCM (Int16Array), base64-
//     encoded, and emitted to the server as an "audio_chunk" event.
//   • Chunk interval: every 256 ms (~4 chunks/s) — enough for real-time VAD
//     without overwhelming the WebSocket.
//
// Why ScriptProcessorNode over AudioWorkletNode?
//   AudioWorklet provides better performance but requires a separate JS file
//   (worker) which complicates deployment.  ScriptProcessorNode is deprecated
//   but universally supported and sufficient for this use case (audio is
//   processed server-side, not in the browser).

const AUDIO_SAMPLE_RATE  = 16000;   // target rate for WebRTC VAD
const AUDIO_BUFFER_SIZE  = 4096;    // samples per ScriptProcessor callback
const AUDIO_SEND_INTERVAL= 256;     // ms between audio emissions to server

let _audioCtx        = null;
let _audioStream     = null;
let _scriptProcessor = null;
let _audioChunks     = [];          // accumulate Float32 samples
let _audioSendTimer  = null;

/**
 * Start microphone capture and begin sending audio chunks to the server.
 * Called automatically by initWebcam() after the webcam starts.
 */
async function initAudioMonitor(studentId, socket) {
  try {
    _audioStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate:   44100,
        channelCount: 1,
        echoCancellation: false,   // keep off — we want to hear background voices
        noiseSuppression: false,   // keep off — preserve whispers for backend VAD
        autoGainControl: true,     // MUST be on: normalises mic level so soft voices
                                   // reach the RMS gate on the backend.  Without AGC,
                                   // quiet microphones produce near-zero RMS and all
                                   // audio is silently discarded before VAD runs.
      },
      video: false,
    });

    // Reuse the AudioContext that was pre-created synchronously inside the
    // button-click handler (exam.html).  That context was resume()-d while the
    // user-gesture was still active, so it starts in "running" state.
    // Creating a NEW context here (after multiple awaits have consumed the
    // gesture) would leave it suspended permanently in Chrome.
    if (window._unlockedAudioCtx) {
      _audioCtx = window._unlockedAudioCtx;
      console.log("[audio_monitor] Reusing pre-unlocked AudioContext, state:", _audioCtx.state);
    } else {
      // Fallback — should not normally reach here
      _audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
      console.warn("[audio_monitor] Pre-unlocked AudioContext not available — creating fresh one.");
      await _audioCtx.resume();
    }

    // Re-resume if the browser auto-suspends (tab backgrounded, power save, etc.)
    _audioCtx.addEventListener("statechange", () => {
      if (_audioCtx && _audioCtx.state === "suspended") {
        _audioCtx.resume().catch(() => {});
        console.warn("[audio_monitor] AudioContext re-suspended — attempting resume.");
      }
    });

    const source    = _audioCtx.createMediaStreamSource(_audioStream);
    _scriptProcessor = _audioCtx.createScriptProcessor(AUDIO_BUFFER_SIZE, 1, 1);

    _scriptProcessor.onaudioprocess = (e) => {
      const samples = e.inputBuffer.getChannelData(0);   // Float32Array
      // Use a loop rather than spread (...samples) to avoid stack-size issues
      // on browsers that limit argument count for Function.apply / spread.
      for (let i = 0; i < samples.length; i++) _audioChunks.push(samples[i]);
    };

    source.connect(_scriptProcessor);
    _scriptProcessor.connect(_audioCtx.destination);

    // Send accumulated audio every AUDIO_SEND_INTERVAL ms
    _audioSendTimer = setInterval(() => {
      if (_audioChunks.length === 0) {
        // If this fires repeatedly with no data, onaudioprocess is not running —
        // AudioContext is likely still suspended.  Check the console for state logs.
        console.debug("[audio_monitor] No audio accumulated — ctx state:", _audioCtx?.state);
        return;
      }

      const raw = new Float32Array(_audioChunks.splice(0));
      _audioChunks = [];

      // Downsample from 44100 → 16000 Hz
      const downsampled = _downsample(raw, _audioCtx.sampleRate, AUDIO_SAMPLE_RATE);

      // Convert Float32 → Int16
      const int16 = new Int16Array(downsampled.length);
      for (let i = 0; i < downsampled.length; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, downsampled[i] * 32767));
      }

      // Base64 encode
      const bytes  = new Uint8Array(int16.buffer);
      let   binary = "";
      for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
      const b64 = btoa(binary);

      // Quick RMS check — if this is consistently near 0.000, the mic is silent
      // or AGC hasn't kicked in yet.  Normal speech is typically 0.02–0.20.
      let sumSq = 0;
      for (let i = 0; i < downsampled.length; i++) sumSq += downsampled[i] * downsampled[i];
      const rms = Math.sqrt(sumSq / downsampled.length);
      console.debug(`[audio_monitor] chunk: ${downsampled.length} samples, RMS=${rms.toFixed(4)}`);

      socket.emit("audio_chunk", {
        student_id: studentId,
        audio:      b64,
        timestamp:  Date.now(),
      });
    }, AUDIO_SEND_INTERVAL);

    if (typeof updateAudioStatus === "function") {
      updateAudioStatus("🎙 Mic: active", "active");
    }
    console.log("[audio_monitor] Microphone capture started.");

  } catch (err) {
    if (typeof updateAudioStatus === "function") {
      updateAudioStatus("🔴 Mic: denied", "error");
    }
    console.warn("[audio_monitor] Microphone access denied or unavailable:", err.message);
    // Non-fatal — exam continues without audio monitoring
  }
}

/** Linear decimation downsampler (Float32 → Float32 at lower rate). */
function _downsample(buffer, fromRate, toRate) {
  if (fromRate === toRate) return buffer;
  const ratio  = fromRate / toRate;
  const outLen = Math.floor(buffer.length / ratio);
  const out    = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    out[i] = buffer[Math.floor(i * ratio)];
  }
  return out;
}

/** Stop microphone capture. */
function stopAudioMonitor() {
  clearInterval(_audioSendTimer);
  if (_scriptProcessor) _scriptProcessor.disconnect();
  if (_audioStream)     _audioStream.getTracks().forEach(t => t.stop());
  if (_audioCtx)        _audioCtx.close();
  console.log("[audio_monitor] Stopped.");
}
