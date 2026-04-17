/**
 * admin_dashboard.js
 * ──────────────────
 * Real-time admin dashboard powered by SocketIO.
 *
 * Features:
 *   • Live risk cards for every student — auto-created on first event
 *   • Live webcam thumbnail updated every frame (relayed from server)
 *   • Colour-coded risk bar (green → amber → red)
 *   • Per-agent score breakdown (Identity, Gaze, Behaviour, Integrity, Audio)
 *   • Chronological live event feed with severity colouring
 *   • Summary counters (Active / Warned / Flagged / Suspended)
 */

const socket = io({
  transports:           ["websocket"],
  reconnection:         true,
  reconnectionAttempts: 10,
  reconnectionDelay:    1000,
});

socket.on("connect", () => {
  socket.emit("join_admin");
  console.log("[admin] Connected to SocketIO.");
  document.getElementById("live-badge").classList.add("active");
});

socket.on("disconnect", () => {
  document.getElementById("live-badge").classList.remove("active");
});

// ── Live update handler ───────────────────────────────────────────────────────
socket.on("admin_update", (data) => {
  const sid = data.student_id;
  if (!sid) return;

  updateStudentCard(sid, data);
  appendFeedEvent(sid, data);
  updateSummaryCounters();
});

// ── Build / refresh a student card ────────────────────────────────────────────
function updateStudentCard(sid, data) {
  const breakdown  = data.breakdown || {};
  const action     = (breakdown.action || data.action || "NONE").toUpperCase();
  const risk       = breakdown.fused  || data.risk_score || 0;
  const suspended  = data.suspended  || (action === "SUSPEND");
  const submitted  = data.submitted  || (action === "SUBMITTED");
  const thumb      = data.thumb      || "";   // base64 JPEG relayed from backend

  let card = document.getElementById("card-" + sid);
  const isNew = !card;

  if (isNew) {
    card = document.createElement("div");
    card.className = "student-card";
    card.id = "card-" + sid;
    document.getElementById("student-grid")
      .querySelector(".empty-state")?.remove();
    document.getElementById("student-grid").appendChild(card);
  }

  card.dataset.risk      = risk;
  card.dataset.suspended = suspended ? "1" : "0";
  card.dataset.submitted = submitted ? "1" : "0";

  // Track the highest-severity event ever seen for this student.
  // This drives the summary counters independently of the Bayesian risk score,
  // because low-weighted agents (audio) can raise legitimate events that never
  // push the fused risk above the 50% WARN threshold on their own.
  const _SEV = { CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1, NONE: 0 };
  let prevSev  = card.dataset.highestSeverity || "NONE";
  let prevSevN = _SEV[prevSev] || 0;
  (data.events || []).forEach(evt => {
    const n = _SEV[evt.severity] || 0;
    if (n > prevSevN) { prevSev = evt.severity; prevSevN = n; }
  });
  card.dataset.highestSeverity = prevSev;

  // Risk bar colour class — thresholds match backend config.py exactly:
  // RISK_WARN_THRESHOLD=0.50, RISK_SUSPEND_THRESHOLD=0.75
  const riskClass = risk >= 0.75 ? "high" : risk >= 0.5 ? "mid" : "low";

  // Card state class
  const stateClass = suspended  ? " card-suspended"
                   : submitted  ? " card-submitted"
                   : "";

  card.className = "student-card" + stateClass;

  card.innerHTML = `
    <div class="card-header">
      <span class="card-sid">${sid}</span>
      <span class="card-action action-${action.toLowerCase()}">${action}</span>
      ${suspended ? '<span class="card-suspended-badge">SUSPENDED</span>' : ""}
      ${submitted && !suspended ? '<span class="card-submitted-badge">SUBMITTED</span>' : ""}
    </div>

    <!-- Live webcam thumbnail -->
    <div class="card-thumb-wrap">
      ${thumb
        ? `<img class="card-thumb" id="thumb-${sid}" src="${thumb}" alt="Live feed" />`
        : `<div class="card-thumb-placeholder" id="thumb-${sid}">No feed</div>`
      }
    </div>

    <!-- Risk bar -->
    <div class="risk-bar-wrap" title="Fused risk: ${(risk*100).toFixed(1)}%">
      <div class="risk-bar risk-bar-${riskClass}" style="width:${Math.round(risk*100)}%"></div>
    </div>
    <div class="card-risk-val">Risk: <strong>${(risk*100).toFixed(1)}%</strong></div>

    <!-- Per-agent scores -->
    <div class="agent-grid">
      <div class="agent-score" title="Identity Agent">
        🪪 <span>${Math.round((breakdown.identity ||0)*100)}%</span>
      </div>
      <div class="agent-score" title="Gaze Agent">
        👁 <span>${Math.round((breakdown.gaze      ||0)*100)}%</span>
      </div>
      <div class="agent-score" title="Behaviour Agent">
        🔍 <span>${Math.round((breakdown.behaviour ||0)*100)}%</span>
      </div>
      <div class="agent-score" title="Integrity Agent">
        🔒 <span>${Math.round((breakdown.integrity ||0)*100)}%</span>
      </div>
      <div class="agent-score" title="Audio Agent">
        🎙 <span>${Math.round((breakdown.audio     ||0)*100)}%</span>
      </div>
    </div>
  `;

  // Update thumbnail efficiently — replace only the src if card already existed
  // (avoid full innerHTML re-render flicker on fast updates)
  if (!isNew && thumb) {
    const img = document.getElementById("thumb-" + sid);
    if (img && img.tagName === "IMG") {
      img.src = thumb;
    } else if (img) {
      img.outerHTML = `<img class="card-thumb" id="thumb-${sid}" src="${thumb}" alt="Live feed" />`;
    }
  }
}

// ── Append to live event feed ─────────────────────────────────────────────────
function appendFeedEvent(sid, data) {
  const events = data.events || [];
  if (events.length === 0) return;

  const feed = document.getElementById("event-feed");
  feed.querySelector(".feed-placeholder")?.remove();

  events.forEach(evt => {
    const row = document.createElement("div");
    row.className = "feed-row severity-" + (evt.severity || "LOW").toLowerCase();
    const ts = new Date((evt.timestamp || Date.now() / 1000) * 1000)
                 .toLocaleTimeString();
    row.innerHTML = `
      <span class="feed-ts">${ts}</span>
      <span class="feed-sid">${sid}</span>
      <span class="feed-type">${evt.event_type || ""}</span>
      <span class="feed-desc">${evt.description || ""}</span>
    `;
    feed.insertBefore(row, feed.firstChild);
  });

  // Trim to last 150 events
  while (feed.children.length > 150) feed.removeChild(feed.lastChild);
}

// ── Summary counters ──────────────────────────────────────────────────────────
function updateSummaryCounters() {
  const cards = document.querySelectorAll(".student-card");

  // Counters are event-driven, not purely risk-threshold-driven.
  //
  // Why: The Bayesian prior (logit 0.05 = –2.94) is strong enough that low-
  // weighted agents (audio weight 0.6) can never push fused risk above the 0.50
  // WARN threshold on their own, even at maximum risk score.  This caused events
  // to appear in the feed while counters stayed at 0 — confusing and misleading.
  //
  // New logic:
  //   SUSPENDED  : student was auto-suspended (hard override or Bayesian ≥ 0.75)
  //   SUBMITTED  : student submitted the exam without being suspended
  //   FLAGGED    : highest event severity ≥ HIGH  OR  Bayesian risk ≥ 0.75
  //                (approaching suspension; serious violation seen)
  //   WARNED     : highest event severity ≥ MEDIUM OR  Bayesian risk ≥ 0.50
  //                (at least one notable event, or risk in warn band)
  //
  // A student progresses through the bands monotonically — once FLAGGED they
  // stay FLAGGED even if the next frame's event is LOW.

  let active = 0, warned = 0, flagged = 0, suspended = 0, submitted = 0;

  cards.forEach(card => {
    const risk  = parseFloat(card.dataset.risk           || 0);
    const susp  = card.dataset.suspended                 === "1";
    const subm  = card.dataset.submitted                 === "1";
    const sev   = card.dataset.highestSeverity           || "NONE";

    if (susp) {
      suspended++;
    } else if (subm) {
      submitted++;
    } else {
      // Student is still in the exam — count as active
      active++;
      if (sev === "CRITICAL" || risk >= 0.75) {
        flagged++;    // only CRITICAL events or risk ≥ 75% → flagged
      } else if (sev === "HIGH" || sev === "MEDIUM" || risk >= 0.5) {
        warned++;     // HIGH and MEDIUM both shown as warnings to student
      }
    }
  });

  document.getElementById("count-active").textContent    = active;
  document.getElementById("count-warned").textContent    = warned;
  document.getElementById("count-flagged").textContent   = flagged;
  document.getElementById("count-suspended").textContent = suspended;

  const submEl = document.getElementById("count-submitted");
  if (submEl) submEl.textContent = submitted;
}

// ── Manual refresh from REST API ──────────────────────────────────────────────
async function refreshAll() {
  try {
    const resp = await fetch("/api/admin/all_sessions");
    if (!resp.ok) return;
    const data = await resp.json();
    Object.entries(data).forEach(([sid, info]) => {
      updateStudentCard(sid, {
        student_id: sid,
        breakdown:  info.breakdown,
        suspended:  info.suspended,
        action:     info.breakdown?.action,
      });
      if (info.events?.length) appendFeedEvent(sid, { events: info.events });
    });
    updateSummaryCounters();
  } catch (e) {
    console.warn("[admin] refreshAll failed:", e);
  }
}

// Initial load on page open
refreshAll();
