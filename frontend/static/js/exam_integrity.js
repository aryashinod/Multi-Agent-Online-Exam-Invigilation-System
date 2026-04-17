/**
 * exam_integrity.js
 * ─────────────────
 * Browser-side integrity enforcement and event monitoring.
 *
 * Monitors:
 *   1. Copy / Cut / Paste events       → COPY / PASTE
 *   2. Page Visibility API changes     → TAB_SWITCH
 *   3. Window focus/blur               → FOCUS_LOSS
 *   4. Fullscreen exit                 → FULLSCREEN_EXIT
 *   5. Right-click context menu        → (silently disabled)
 *   6. DevTools open detection         → DEV_TOOLS
 *
 * Each detected event is sent to the backend IntegrityAgent via SocketIO.
 * The server-side agent tallies violations and feeds them into the
 * Bayesian risk aggregation in the OrchestratorAgent.
 *
 * Algorithm note (Fullscreen enforcement):
 *   The Fullscreen API (MDN, 2024) is used to lock the exam to fullscreen.
 *   Exit events are detected via the "fullscreenchange" event listener.
 *   This does not prevent all screen-sharing but significantly increases
 *   friction for accessing other windows/tabs.
 *
 * Algorithm note (DevTools detection):
 *   Uses the window size heuristic: if outerWidth - innerWidth > 200 or
 *   outerHeight - innerHeight > 200, DevTools is likely docked.
 *   This is not foolproof but flags a meaningful subset of cases.
 */

let _intSocket    = null;
let _intStudentId = null;
let _intEnabled   = false;

/**
 * Initialise integrity monitoring.
 * Must be called AFTER the exam has started.
 */
function initIntegrityMonitor(studentId, socket) {
  _intStudentId = studentId;
  _intSocket    = socket;
  _intEnabled   = true;

  _bindClipboardEvents();
  _bindVisibilityEvents();
  _bindFullscreenEvents();
  _bindContextMenu();
  _startDevToolsPoller();

  console.log("[exam_integrity] Integrity monitor active.");
}

/** Emit a structured integrity event to the backend. */
function _emitIntegrityEvent(eventType) {
  if (!_intEnabled || !_intSocket) return;
  console.warn("[exam_integrity] Event:", eventType);
  _intSocket.emit("integrity_event", {
    student_id: _intStudentId,
    event_type: eventType,
    timestamp:  Date.now(),
  });
}

// ── 1. Clipboard events ───────────────────────────────────────────────────────
function _bindClipboardEvents() {
  ["copy", "cut"].forEach(evt => {
    document.addEventListener(evt, (e) => {
      e.preventDefault();
      _emitIntegrityEvent("COPY");
      _showWarningToast("Copying exam content is not permitted.");
    }, true);
  });

  document.addEventListener("paste", (e) => {
    e.preventDefault();
    _emitIntegrityEvent("PASTE");
    _showWarningToast("Pasting into the exam is not permitted.");
  }, true);
}

// ── 2. Page Visibility API ────────────────────────────────────────────────────
function _bindVisibilityEvents() {
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") {
      _emitIntegrityEvent("TAB_SWITCH");
    }
  });

  window.addEventListener("blur", () => {
    _emitIntegrityEvent("FOCUS_LOSS");
  });
}

// ── 3. Fullscreen ─────────────────────────────────────────────────────────────
//
// Why we don't auto-re-enter fullscreen:
//   Browsers require a user gesture (click/keypress) to call requestFullscreen().
//   A bare setTimeout() has no gesture attached, so the call is silently rejected
//   in Chrome/Firefox/Safari.  The only reliable recovery is to show a visible
//   overlay the student must click — that click IS the required gesture.
//
let _fsOverlay = null;  // reference to the live recovery overlay (or null)

function _bindFullscreenEvents() {
  const fsEvents = [
    "fullscreenchange", "webkitfullscreenchange",
    "mozfullscreenchange", "msfullscreenchange",
  ];
  fsEvents.forEach(evt => {
    document.addEventListener(evt, () => {
      const inFullscreen = !!(
        document.fullscreenElement      ||
        document.webkitFullscreenElement ||
        document.mozFullScreenElement   ||
        document.msFullscreenElement
      );
      if (!inFullscreen) {
        _emitIntegrityEvent("FULLSCREEN_EXIT");
        _showFullscreenRecovery();
      } else {
        // Student successfully re-entered — dismiss the recovery overlay.
        _hideFullscreenRecovery();
      }
    });
  });
}

/**
 * Show a blocking overlay that forces the student to click a button to
 * re-enter fullscreen.  The click provides the user-gesture browsers require
 * before honouring requestFullscreen().
 */
function _showFullscreenRecovery() {
  if (_fsOverlay) return;   // already visible

  _fsOverlay = document.createElement("div");
  _fsOverlay.id = "_fs-recovery-overlay";
  _fsOverlay.style.cssText = [
    "position:fixed", "inset:0", "z-index:2147483647",
    "background:rgba(0,0,0,0.88)",
    "display:flex", "flex-direction:column",
    "align-items:center", "justify-content:center",
    "font-family:system-ui,sans-serif",
  ].join(";");

  _fsOverlay.innerHTML = `
    <div style="
      background:#1a1a2e; border:2px solid #e53e3e; border-radius:16px;
      padding:40px 36px; max-width:420px; width:90%; text-align:center; color:#fff;
      box-shadow:0 8px 32px rgba(0,0,0,.6);
    ">
      <div style="font-size:52px; margin-bottom:12px;">⚠️</div>
      <h2 style="margin:0 0 10px; font-size:22px; color:#fc8181;">
        Fullscreen Required
      </h2>
      <p style="margin:0 0 8px; color:#e2e8f0; font-size:15px; line-height:1.5;">
        You exited fullscreen mode.<br>
        <strong style="color:#fc8181;">This exit has been recorded as a violation.</strong>
      </p>
      <p style="margin:0 0 28px; color:#a0aec0; font-size:13px;">
        The exam must run in fullscreen at all times.<br>
        Click the button below to continue.
      </p>
      <button
        id="_fs-reenter-btn"
        style="
          width:100%; padding:14px 0; background:#2b6cb0; color:#fff;
          border:none; border-radius:8px; font-size:16px; font-weight:700;
          cursor:pointer; transition:background .2s;
        "
        onmouseover="this.style.background='#2c5282'"
        onmouseout="this.style.background='#2b6cb0'"
      >
        🔲 Return to Fullscreen
      </button>
    </div>
  `;

  document.body.appendChild(_fsOverlay);

  // Wire up the button — this click IS the required browser gesture.
  document.getElementById("_fs-reenter-btn").addEventListener("click", () => {
    document.documentElement.requestFullscreen().catch(() => {
      // If the browser still refuses (e.g. sandboxed iframe), just dismiss
      // so the student isn't permanently blocked.
      _hideFullscreenRecovery();
    });
  });
}

function _hideFullscreenRecovery() {
  if (_fsOverlay) {
    _fsOverlay.remove();
    _fsOverlay = null;
  }
}

// ── 4. Context menu disable ───────────────────────────────────────────────────
function _bindContextMenu() {
  document.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    _showWarningToast("Right-click is disabled during the exam.");
  }, true);
}

// ── 5. DevTools detection ─────────────────────────────────────────────────────
let _devToolsOpen = false;
function _startDevToolsPoller() {
  setInterval(() => {
    const threshold = 200;
    const open = (
      window.outerWidth  - window.innerWidth  > threshold ||
      window.outerHeight - window.innerHeight > threshold
    );
    if (open && !_devToolsOpen) {
      _devToolsOpen = true;
      _emitIntegrityEvent("DEVTOOLS_OPEN");
      _showWarningToast("Developer tools detected — this has been flagged.");
    }
    if (!open && _devToolsOpen) {
      _devToolsOpen = false;
    }
  }, 3000);
}

// ── Stop monitoring ───────────────────────────────────────────────────────────
/**
 * Disable all integrity monitoring.
 * Called when the exam is submitted or the session is suspended so that
 * no further events are emitted to the backend.
 */
function stopIntegrityMonitor() {
  _intEnabled   = false;
  _intSocket    = null;
  _intStudentId = null;
  _hideFullscreenRecovery();   // dismiss overlay if exam ends while it's showing
  console.log("[exam_integrity] Integrity monitor stopped.");
}

// ── Warning toast UI ──────────────────────────────────────────────────────────
function _showWarningToast(msg) {
  let toast = document.getElementById("_integrity-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "_integrity-toast";
    toast.style.cssText = `
      position:fixed; bottom:20px; left:50%; transform:translateX(-50%);
      background:#e53e3e; color:#fff; padding:10px 22px; border-radius:8px;
      font-weight:600; font-size:14px; z-index:9999; box-shadow:0 4px 12px rgba(0,0,0,.3);
      transition: opacity .3s;
    `;
    document.body.appendChild(toast);
  }
  toast.textContent = "⚠ " + msg;
  toast.style.opacity = "1";
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { toast.style.opacity = "0"; }, 4000);
}
