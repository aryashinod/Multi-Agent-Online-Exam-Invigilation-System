"""
app.py — Flask + SocketIO Application Entry Point
==================================================

Routes:
  GET  /                      → registration / login page
  GET  /exam                  → exam interface (requires session)
  GET  /admin                 → admin dashboard (requires admin session)
  POST /api/register          → store student face + create account
  POST /api/login             → authenticate student, issue session
  POST /api/begin_exam        → start exam, return shuffled questions
  POST /api/submit_exam       → submit answers, return grade
  GET  /api/status/<sid>      → current risk breakdown for a student

SocketIO events (client → server):
  "frame"           : { student_id, frame (base64 JPEG) }
  "integrity_event" : { student_id, event_type, timestamp }

SocketIO events (server → client):
  "alert"           : { action, risk_score, events, breakdown }
  "suspended"       : { reason }
  "risk_update"     : { risk_score, breakdown }
"""

import os
import time
import uuid
import json
import logging
import threading
from functools import wraps
from typing import Dict

from flask import (
    Flask, request, session, jsonify,
    render_template, redirect, url_for, abort,
)
from flask_socketio import SocketIO, emit, join_room, leave_room

from .config import SECRET_KEY, DEBUG, HOST, PORT
from .agents import OrchestratorAgent

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Flask app setup ───────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "..", "frontend", "static"),
)
app.secret_key = SECRET_KEY

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
    ping_timeout=60,        # seconds before client is considered disconnected
    ping_interval=25,       # heartbeat interval (default 25s)
)

# ── In-memory session store (use Redis / DB in production) ───────────────────
# student_id → OrchestratorAgent instance
_orchestrators: Dict[str, OrchestratorAgent] = {}
_orchestrators_lock = threading.Lock()

# User store — loaded from users.json on startup, saved on every change
import json as _json

_USERS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "users.json")

def _load_users() -> dict:
    try:
        with open(_USERS_FILE, "r") as f:
            return _json.load(f)
    except (FileNotFoundError, _json.JSONDecodeError):
        return {}

def _save_users(users: dict):
    os.makedirs(os.path.dirname(_USERS_FILE), exist_ok=True)
    with open(_USERS_FILE, "w") as f:
        _json.dump(users, f, indent=2)

_users: Dict[str, dict] = _load_users()   # student_id → { password_hash, name, is_admin }

# Active exam sessions
_exam_sessions: Dict[str, str] = {}   # student_id → session_token


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_orchestrator(student_id: str) -> OrchestratorAgent:
    """Return (or create) the OrchestratorAgent for a student."""
    with _orchestrators_lock:
        if student_id not in _orchestrators:
            orch = OrchestratorAgent()
            orch.start()
            _orchestrators[student_id] = orch
        return _orchestrators[student_id]


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "student_id" not in session:
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("is_admin"):
            abort(403)
        return f(*args, **kwargs)
    return decorated


# ── Page routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/exam")
@login_required
def exam_page():
    return render_template("exam.html",
                           student_id=session["student_id"],
                           student_name=session.get("name", "Student"))


@app.route("/admin")
@login_required
@admin_required
def admin_page():
    students = [
        {
            "student_id": sid,
            "risk":       orch.get_risk_breakdown(),
            "suspended":  orch._suspended,
        }
        for sid, orch in _orchestrators.items()
    ]
    return render_template("admin.html", students=students)


# ── REST API routes ───────────────────────────────────────────────────────────

@app.route("/api/register", methods=["POST"])
def api_register():
    """
    Register a new student.
    Body: { student_id, name, password, face_frame (base64 JPEG) }
    """
    data = request.get_json() or {}
    sid   = data.get("student_id", "").strip()
    name  = data.get("name", "").strip()
    pwd   = data.get("password", "")
    frame = data.get("face_frame", "")

    if not sid or not name or not pwd:
        return jsonify({"success": False, "error": "Missing required fields."}), 400

    if sid in _users:
        return jsonify({"success": False, "error": "Student ID already registered."}), 409

    import hashlib
    pwd_hash = hashlib.sha256(pwd.encode()).hexdigest()
    _users[sid] = {"name": name, "password_hash": pwd_hash, "is_admin": False}
    _save_users(_users)

    # Store face reference for identity verification
    orch = get_orchestrator(sid)
    face_ok = orch.register_student(sid, frame) if frame else False

    logger.info("Registered student %s (face_ok=%s).", sid, face_ok)
    return jsonify({"success": True, "face_registered": face_ok})


@app.route("/api/login", methods=["POST"])
def api_login():
    """
    Authenticate a student.
    Body: { student_id, password }
    """
    data = request.get_json() or {}
    sid  = data.get("student_id", "").strip()
    pwd  = data.get("password", "")

    import hashlib
    pwd_hash = hashlib.sha256(pwd.encode()).hexdigest()

    user = _users.get(sid)
    if not user or user["password_hash"] != pwd_hash:
        return jsonify({"success": False, "error": "Invalid credentials."}), 401

    session["student_id"] = sid
    session["name"]       = user["name"]
    session["is_admin"]   = user.get("is_admin", False)

    logger.info("Student %s logged in.", sid)
    return jsonify({"success": True, "name": user["name"], "is_admin": user["is_admin"]})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/api/begin_exam", methods=["POST"])
@login_required
def api_begin_exam():
    """
    Start an exam session.
    Body: { student_id }  (must match session)
    """
    data = request.get_json() or {}
    sid  = data.get("student_id", "").strip()

    if sid != session.get("student_id"):
        return jsonify({"error": "Unauthorised."}), 403

    token = uuid.uuid4().hex
    _exam_sessions[sid] = token

    orch   = get_orchestrator(sid)
    result = orch.begin_exam(sid, token)
    result["session_token"] = token

    logger.info("Exam started for %s (token=%s…).", sid, token[:8])
    return jsonify(result)


@app.route("/api/submit_exam", methods=["POST"])
@login_required
def api_submit_exam():
    """
    Submit exam answers.
    Body: { student_id, answers: {qid: option}, order_hash }
    """
    data       = request.get_json() or {}
    sid        = data.get("student_id", "").strip()
    answers    = data.get("answers", {})
    order_hash = data.get("order_hash", "")

    if sid != session.get("student_id"):
        return jsonify({"error": "Unauthorised."}), 403

    orch   = get_orchestrator(sid)
    result = orch.submit_exam(sid, answers, order_hash)

    logger.info("Exam submitted by %s — score %s/%s.", sid,
                result.get("score"), result.get("total"))

    # Broadcast a submission event to the admin dashboard so the invigilator
    # can see exactly when each student finished and what they scored.
    submission_event = {
        "event_type":  "EXAM_SUBMITTED",
        "severity":    "LOW",
        "description": (
            f"Exam submitted — "
            f"{result.get('score')}/{result.get('total')} "
            f"({result.get('percentage')}%) | "
            f"Risk: {round((result.get('final_risk_score', 0) * 100), 1)}% | "
            f"Integrity: {'OK' if result.get('integrity_ok') else 'TAMPERED'}"
        ),
        "timestamp":   time.time(),
    }
    socketio.emit("admin_update", {
        "student_id":  sid,
        "risk_score":  result.get("final_risk_score", 0),
        "action":      "SUBMITTED",
        "submitted":   True,
        "events":      [submission_event],
        "breakdown":   orch.get_risk_breakdown(),
    }, room="admin_room")

    return jsonify(result)


@app.route("/api/status/<student_id>")
@login_required
def api_status(student_id):
    """Return the current risk breakdown for a student (admin view)."""
    if student_id != session.get("student_id") and not session.get("is_admin"):
        abort(403)
    orch = _orchestrators.get(student_id)
    if not orch:
        return jsonify({"error": "No active session."}), 404
    return jsonify(orch.get_risk_breakdown())


@app.route("/api/admin/all_sessions")
@login_required
@admin_required
def api_all_sessions():
    """Return risk breakdown for ALL active student sessions."""
    result = {}
    for sid, orch in _orchestrators.items():
        result[sid] = {
            "breakdown": orch.get_risk_breakdown(),
            "suspended": orch._suspended,
            "events":    orch.get_full_log()[-5:],  # last 5 events
        }
    return jsonify(result)


@app.route("/api/admin/register_admin", methods=["POST"])
def api_register_admin():
    """Quick endpoint to create an admin account (for demo purposes)."""
    import hashlib
    data = request.get_json() or {}
    sid  = data.get("student_id", "admin")
    pwd  = data.get("password", "admin123")
    _users[sid] = {
        "name":          "Administrator",
        "password_hash": hashlib.sha256(pwd.encode()).hexdigest(),
        "is_admin":      True,
    }
    _save_users(_users)
    return jsonify({"success": True})


# ── WebSocket events ──────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    sid = session.get("student_id", "unknown")
    logger.debug("SocketIO connect: %s (sid=%s)", sid, request.sid)
    if sid != "unknown":
        join_room(sid)


@socketio.on("disconnect")
def on_disconnect():
    sid = session.get("student_id", "unknown")
    if sid != "unknown":
        leave_room(sid)
    logger.debug("SocketIO disconnect: %s", sid)


@socketio.on("frame")
def on_frame(data):
    """
    Receive a webcam frame from the student client.

    data = { "student_id": str, "frame": <base64 JPEG data-URL> }

    Processes the frame through IdentityAgent, GazeAgent, and BehaviourAgent
    via the OrchestratorAgent, then emits an "alert" event back to the student
    and an "admin_update" event to the admin room.
    """
    sid = data.get("student_id") or session.get("student_id")
    if not sid:
        return

    try:
        orch   = get_orchestrator(sid)
        result = orch.process({"frame": data.get("frame", ""), "student_id": sid})

        # Notify the student
        emit("alert", result, room=sid)

        # Relay thumbnail to admin dashboard (include frame so admin sees live feed)
        emit("admin_update", {
            "student_id": sid,
            "thumb":      data.get("frame", ""),   # base64 JPEG thumbnail
            **result,
        }, room="admin_room")

        if result.get("action") == "SUSPEND":
            events      = result.get("events", [])
            event_types = {e.get("event_type", "") for e in events}
            if "IDENTITY_MISMATCH_SUSPEND" in event_types:
                reason = ("Exam suspended: identity verification failed repeatedly. "
                          "A different person appears to be sitting your exam.")
            elif "STUDENT_ABSENT_SUSPEND" in event_types:
                reason = ("Exam suspended: you were absent from the camera frame twice. "
                          "Please contact your invigilator.")
            elif "PHONE_DETECTED_SUSPEND" in event_types:
                reason = ("Exam suspended: a mobile phone or unauthorised device was "
                          "detected multiple times.")
            else:
                reason = "Exam suspended due to repeated integrity violations."
            emit("suspended", {"reason": reason}, room=sid)

    except Exception as exc:
        logger.error("[app] on_frame error for %s: %s", sid, exc, exc_info=True)


@socketio.on("integrity_event")
def on_integrity_event(data):
    """
    Receive a browser integrity violation from the student client.

    data = { "student_id": str, "event_type": str, "timestamp": int (ms) }
    """
    sid = data.get("student_id") or session.get("student_id")
    if not sid:
        return

    orch   = get_orchestrator(sid)
    result = orch.process_integrity_event(data)

    emit("alert", result, room=sid)
    emit("admin_update", {"student_id": sid, **result}, room="admin_room")

    # Hard suspend path: critical integrity violation → immediate suspension
    if result.get("action") == "SUSPEND":
        events      = result.get("events", [])
        event_types = {e.get("event_type", "") for e in events}
        if "TAB_SWITCH_SUSPEND" in event_types:
            reason = ("Exam suspended: you switched browser tabs more than once. "
                      "Accessing external resources during an exam is a policy violation.")
        else:
            reason = "Exam suspended due to a critical integrity violation."
        emit("suspended", {"reason": reason}, room=sid)


@socketio.on("audio_chunk")
def on_audio_chunk(data):
    """
    Receive a raw audio chunk from the student's microphone.

    data = {
        "student_id": str,
        "audio":      <base64-encoded 16-bit PCM, mono 16 kHz>,
        "timestamp":  int (client JS ms epoch)
    }

    Routed to AudioAgent via OrchestratorAgent.
    """
    sid = data.get("student_id") or session.get("student_id")
    if not sid:
        return

    orch = get_orchestrator(sid)
    try:
        evt = orch.audio_agent.process(data)
        if evt:
            orch._session_log.append(evt.to_dict())
            # Re-fuse risk with updated audio score
            risk, action = orch._fuse_and_decide()
            orch._current_risk   = risk
            orch._current_action = action
            result = {
                "risk_score": round(risk, 4),
                "action":     action,
                "events":     [evt.to_dict()],
                "breakdown":  orch.get_risk_breakdown(),
            }
            emit("alert",        result,                          room=sid)
            emit("admin_update", {"student_id": sid, **result},  room="admin_room")

            if action == "SUSPEND" and not orch._suspended:
                orch._suspended = True
                emit("suspended", {
                    "reason": "Exam suspended due to audio anomaly: " + evt.description
                }, room=sid)
    except Exception as exc:
        logger.error("[app] audio_chunk error for %s: %s", sid, exc)


@socketio.on("join_admin")
def on_join_admin():
    """Allow admin clients to join the admin broadcast room."""
    if session.get("is_admin"):
        join_room("admin_room")
        emit("admin_joined", {"status": "ok"})


# ── Run ───────────────────────────────────────────────────────────────────────

def create_app():
    return app


if __name__ == "__main__":
    socketio.run(app, host=HOST, port=PORT, debug=DEBUG)
