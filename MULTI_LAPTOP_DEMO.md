# ExamGuard — Multi-Laptop Demo Guide

## Why HTTPS is required

Chrome and Edge **block camera and microphone access on plain HTTP** when the
page is not `localhost`. Since student laptops connect via a LAN IP address
(`http://192.168.x.x:5000`), the browser will silently deny camera permission.

**You must use HTTPS.** Pick one of the three options below.

---

## Option A — Self-signed certificate (recommended for coursework demo)

### Setup (one time)

Install the cert library:
```bash
pip install cryptography --break-system-packages
```

### Start the server with HTTPS

```bash
python run.py --https
```

The first run auto-generates `data/ssl/cert.pem` and `data/ssl/key.pem`.
Output will look like:

```
  HTTPS MODE — self-signed certificate
  Student URL  →  https://192.168.1.42:5000
  Admin URL    →  https://192.168.1.42:5000/admin
```

### Student one-time step (first time only)

When each student laptop opens the URL in Chrome they will see:

> **"Your connection is not private"**  NET::ERR_CERT_AUTHORITY_INVALID

Tell them:
1. Click **Advanced**
2. Click **Proceed to 192.168.1.42 (unsafe)**

Done — camera and microphone will work normally after this.

---

## Option B — ngrok public HTTPS tunnel (zero config, best for demo)

This gives every student a proper `https://` URL with no browser warnings at all.

```bash
# Terminal 1 — start server normally (HTTP is fine for ngrok)
python run.py

# Terminal 2 — create the tunnel
ngrok http 5000
```

ngrok prints:
```
Forwarding   https://abc123.ngrok-free.app  →  http://localhost:5000
```

Share `https://abc123.ngrok-free.app` with everyone — students open it on their
laptops, no security warnings, camera works immediately.

Install ngrok: https://ngrok.com/download (free tier, no signup needed for basic use)

---

## Option C — Chrome flag (for demo environment only, not production)

If you can control the Chrome installation on each student laptop, launch Chrome
with a flag that allows camera on a specific insecure origin:

**Windows:**
```
"C:\Program Files\Google\Chrome\Application\chrome.exe" --unsafely-treat-insecure-origin-as-secure="http://192.168.1.42:5000" --user-data-dir=C:\ChromeTemp
```

**macOS:**
```bash
open -a "Google Chrome" --args --unsafely-treat-insecure-origin-as-secure=http://192.168.1.42:5000 --user-data-dir=/tmp/chrome_tmp
```

Then students can use plain `http://192.168.1.42:5000` with camera working.

---

## Complete demo flow (using Option A or B)

### Step 1 — Start the server

```bash
# Option A (self-signed cert, LAN only)
python run.py --https

# Option B (ngrok, public HTTPS)
python run.py         # terminal 1
ngrok http 5000       # terminal 2
```

### Step 2 — Pre-register student accounts

```bash
# In a new terminal (while server is running)
python demo_setup.py --students 4
```

This prints a credential table with one row per student laptop:

```
  Student ID     Password      URL
  student01      pass01        https://192.168.1.42:5000
  student02      pass02        https://192.168.1.42:5000
  student03      pass03        https://192.168.1.42:5000
  student04      pass04        https://192.168.1.42:5000

  Admin URL  →  https://192.168.1.42:5000/admin   (admin / admin123)
```

### Step 3 — Open admin dashboard

On your (server) laptop:
- Open `https://192.168.1.42:5000/admin`
- Login: `admin` / `admin123`
- Dashboard is empty until students join

### Step 4 — Each student laptop

1. Open Chrome → `https://192.168.1.42:5000`
2. Accept the self-signed cert warning (Option A only, once)
3. Login with their credentials from the table
4. Click **Begin Exam**
5. Click **Allow** for camera and microphone

Their card appears on the admin dashboard within seconds.

---

## What each laptop does independently

All invigilation happens **inside the student's own browser**:

| Detection | Where it runs | How it works |
|---|---|---|
| Tab switching | Student's browser (JS) | `document.visibilitychange` event |
| Fullscreen exit | Student's browser (JS) | `document.fullscreenchange` event |
| Window focus loss | Student's browser (JS) | `window.blur` event |
| Copy / paste | Student's browser (JS) | `copy`, `paste` events blocked |
| Right-click | Student's browser (JS) | `contextmenu` event blocked |
| Webcam frames | Student's browser (JS) | `getUserMedia` → WebSocket to server |
| Microphone audio | Student's browser (JS) | `getUserMedia` → WebSocket to server |

The server receives events and frames via WebSocket, runs the AI agents, and
sends risk scores back. **Each student's session is fully independent.**

---

## Architecture (multi-laptop)

```
                        Wi-Fi / LAN
                       ┌────────────────────────────────────────┐
 Laptop 1 (student01)  │  Chrome tab  ──WebSocket──┐           │
 Laptop 2 (student02)  │  Chrome tab  ──WebSocket──┤           │
 Laptop 3 (student03)  │  Chrome tab  ──WebSocket──┼──► Flask  │  ← server laptop
 Laptop 4 (student04)  │  Chrome tab  ──WebSocket──┘  SocketIO │
                        │                              (port 5000, HTTPS)
 Admin laptop          │  /admin tab  ◄─WebSocket────────────  │
 (live dashboard)       └────────────────────────────────────────┘
```

Each student's WebSocket connection carries:
- **`frame`** events (base64 JPEG, 2 fps) → processed by 5 AI agents
- **`integrity_event`** events (tab switch, fullscreen, etc.) → processed by IntegrityAgent
- **`audio_chunk`** events (16 kHz PCM, 4 fps) → processed by AudioAgent

Server replies:
- **`alert`** → back to that student (triggers warning modal / suspension overlay)
- **`admin_update`** → to admin dashboard (updates risk card + live feed)

---

## Firewall (Windows only)

If student laptops cannot reach the server, run this once on the server laptop as Administrator:

```
netsh advfirewall firewall add rule name="ExamGuard" dir=in action=allow protocol=TCP localport=5000
```

---

## Quick-reference commands

```bash
# Start HTTPS server (LAN demo)
python run.py --https

# Start server + ngrok tunnel (public demo)
python run.py &
ngrok http 5000

# Register demo accounts (run while server is up)
python demo_setup.py --students 4

# Show ngrok instructions only
python run.py --ngrok
```
