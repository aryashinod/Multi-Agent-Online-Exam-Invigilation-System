"""
run.py — Application launcher for the Online Exam Invigilation System
======================================================================

Usage:
  python run.py                 # HTTP on localhost (dev)
  python run.py --https         # HTTPS with auto-generated self-signed cert (LAN demo)
  python run.py --https --port 443   # standard HTTPS port (needs admin/sudo)

Why HTTPS?
  Chrome (and most modern browsers) block camera and microphone access on
  non-localhost HTTP pages (Mixed Content / Secure Context requirement).
  When students join from other laptops on the LAN you MUST use HTTPS so
  their browsers will grant camera permission.

  The --https flag auto-generates a self-signed certificate using Python's
  built-in 'cryptography' library (or falls back to 'pyOpenSSL' / 'openssl'
  CLI).  Students will see a "Your connection is not private" warning — they
  click Advanced → Proceed to <IP> (unsafe) once, and that's it.

Environment variables (see backend/config.py):
  SECRET_KEY  — Flask session secret (default: built-in dev key)
  DEBUG       — "true" | "false"  (default: true)
  PORT        — integer            (default: 5000)
"""

import argparse
import os
import sys
import logging
import socket

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ExamGuard server")
parser.add_argument("--https",  action="store_true",
                    help="Serve over HTTPS with a self-signed certificate (required for LAN multi-laptop demo)")
parser.add_argument("--port",   type=int, default=None,
                    help="Override PORT from config (default 5000, or 443 for HTTPS)")
parser.add_argument("--ngrok",  action="store_true",
                    help="Print ngrok command to create a public HTTPS tunnel instead")
args = parser.parse_args()

# ── Add project root to path ──────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from backend.config import HOST, PORT as CFG_PORT, DEBUG, REGISTERED_FACES, EXAM_LOGS_DIR

PORT = args.port or CFG_PORT

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_local_ip():
    """Return best-guess LAN IP of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def ensure_dirs():
    """Create data directories if they don't exist."""
    dirs = [REGISTERED_FACES, EXAM_LOGS_DIR, os.path.join(ROOT, "data", "models")]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Data directories ready.")


def download_ssd_model():
    """Download OpenCV SSD face detection model weights if not present."""
    model_dir    = os.path.join(ROOT, "data", "models")
    proto_path   = os.path.join(model_dir, "deploy.prototxt")
    weights_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    if os.path.exists(proto_path) and os.path.exists(weights_path):
        logger.info("SSD face model already present — skipping download.")
        return

    try:
        import urllib.request
        BASE  = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/"
        BASE2 = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
        logger.info("Downloading SSD deploy.prototxt…")
        urllib.request.urlretrieve(BASE + "deploy.prototxt", proto_path)
        logger.info("Downloading SSD caffemodel weights (~10 MB)…")
        urllib.request.urlretrieve(
            BASE2 + "res10_300x300_ssd_iter_140000.caffemodel", weights_path)
        logger.info("SSD face model downloaded successfully.")
    except Exception as exc:
        logger.warning("Could not download SSD model (%s). Falling back to Haar.", exc)


def download_yolo_model():
    """
    Download YOLOv3-tiny for object detection (phone, book, laptop).

    Files (~36 MB total):
      yolov3-tiny.cfg      — network architecture  (<5 KB)
      yolov3-tiny.weights  — pre-trained COCO weights (~35 MB)
      coco.names           — 80-class label list    (<2 KB)

    YOLOv3-tiny is chosen over MobileNet-SSD COCO because:
      • Official weights available at a single stable URL (pjreddie.com)
      • 80 COCO classes include cell phone (67), book (73), laptop (63)
      • Runs at ~15 fps on CPU with OpenCV DNN backend (sufficient for 2 fps
        exam monitoring)
      • Weight file is only 35 MB vs. 200 MB for full YOLOv3
    """
    import urllib.request
    model_dir  = os.path.join(ROOT, "data", "models")
    cfg_path   = os.path.join(model_dir, "yolov3-tiny.cfg")
    wts_path   = os.path.join(model_dir, "yolov3-tiny.weights")
    names_path = os.path.join(model_dir, "coco.names")

    if os.path.exists(cfg_path) and os.path.exists(wts_path):
        logger.info("YOLOv3-tiny already present — skipping download.")
        return

    # pjreddie.com returns 403 to Python's default urllib agent but serves
    # fine to browser User-Agents.  We spoof a Chrome header so the server
    # treats this as a normal browser download.
    BROWSER_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }

    MIRRORS = {
        "cfg": [
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny.cfg",
            "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
        ],
        "wts": [
            # pjreddie.com — works with browser User-Agent
            "https://pjreddie.com/media/files/yolov3-tiny.weights",
            # AlexeyAB GitHub (v4 pre-release also ships tiny weights)
            "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov3-tiny.weights",
        ],
        "names": [
            "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
            "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
        ],
    }

    def _download(urls, dest, label):
        for url in urls:
            try:
                logger.info("Downloading %s …", label)
                req = urllib.request.Request(url, headers=BROWSER_HEADERS)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    with open(dest, "wb") as f:
                        while True:
                            chunk = resp.read(65536)
                            if not chunk:
                                break
                            f.write(chunk)
                logger.info("%s downloaded OK.", label)
                return True
            except Exception as exc:
                logger.warning("  Mirror failed (%s): %s", url, exc)
        return False

    try:
        ok  = _download(MIRRORS["cfg"],   cfg_path,   "YOLOv3-tiny config")
        ok &= _download(MIRRORS["wts"],   wts_path,   "YOLOv3-tiny weights (~35 MB)")
        ok &= _download(MIRRORS["names"], names_path, "COCO labels")
        if ok:
            logger.info("YOLOv3-tiny downloaded — phone/book detection now active.")
        else:
            raise RuntimeError("All mirrors failed.")
    except Exception as exc:
        logger.warning("Could not auto-download YOLOv3-tiny (%s). "
                       "Phone detection disabled.\n"
                       "  To enable it manually, run these commands:\n"
                       "    curl -A \"Mozilla\" -o data/models/yolov3-tiny.weights "
                       "https://pjreddie.com/media/files/yolov3-tiny.weights\n"
                       "    curl -o data/models/yolov3-tiny.cfg "
                       "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny.cfg\n"
                       "    curl -o data/models/coco.names "
                       "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names\n"
                       "  Then restart the server.", exc)
        for p in [cfg_path, wts_path, names_path]:
            try:
                if os.path.exists(p): os.remove(p)
            except Exception:
                pass


def generate_self_signed_cert():
    """
    Generate a self-signed TLS certificate valid for 1 year.

    Returns (cert_path, key_path) or raises RuntimeError if no suitable
    library is available.

    Tries in order:
      1. cryptography (pip install cryptography)
      2. pyOpenSSL   (pip install pyopenssl)
      3. openssl CLI (system install)
    """
    cert_dir  = os.path.join(ROOT, "data", "ssl")
    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "cert.pem")
    key_path  = os.path.join(cert_dir, "key.pem")

    if os.path.exists(cert_path) and os.path.exists(key_path):
        logger.info("Existing self-signed cert found — reusing.")
        return cert_path, key_path

    lan_ip = get_local_ip()

    # ── Method 1: cryptography library ───────────────────────────────────────
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend
        import datetime, ipaddress

        key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend())

        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "ExamGuard"),
        ])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                    x509.IPAddress(ipaddress.IPv4Address(lan_ip)),
                ]),
                critical=False,
            )
            .sign(key, hashes.SHA256(), default_backend())
        )

        with open(key_path, "wb") as f:
            f.write(key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            ))
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        logger.info("Self-signed cert generated via 'cryptography' library.")
        return cert_path, key_path

    except ImportError:
        pass

    # ── Method 2: pyOpenSSL ───────────────────────────────────────────────────
    try:
        from OpenSSL import crypto

        k = crypto.PKey()
        k.generate_key(crypto.TYPE_RSA, 2048)
        cert = crypto.X509()
        cert.get_subject().CN = "ExamGuard"
        cert.set_serial_number(1000)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(k)
        cert.sign(k, "sha256")

        with open(cert_path, "wb") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        with open(key_path, "wb") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, k))

        logger.info("Self-signed cert generated via 'pyOpenSSL'.")
        return cert_path, key_path

    except ImportError:
        pass

    # ── Method 3: openssl CLI ─────────────────────────────────────────────────
    import subprocess
    try:
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_path, "-out", cert_path,
            "-days", "365", "-nodes", "-subj", "/CN=ExamGuard",
        ], check=True, capture_output=True)
        logger.info("Self-signed cert generated via openssl CLI.")
        return cert_path, key_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    raise RuntimeError(
        "Cannot generate SSL cert. Install one of:\n"
        "  pip install cryptography\n"
        "  pip install pyopenssl\n"
        "Or install OpenSSL system package."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ExamGuard — Online Exam Invigilation System")
    print("  Multi-Agent AI Invigilation Platform")
    print("=" * 60)

    ensure_dirs()
    download_ssd_model()
    download_yolo_model()

    from backend.app import socketio, app

    lan_ip = get_local_ip()
    scheme = "https" if args.https else "http"

    # ── ngrok shortcut ────────────────────────────────────────────────────────
    if args.ngrok:
        print("""
  NGROK MODE — public HTTPS tunnel (no cert warning for students)
  ─────────────────────────────────────────────────────────────────
  1. Install ngrok:  https://ngrok.com/download
  2. Run the server normally in one terminal:
       python run.py
  3. In a second terminal run:
       ngrok http 5000
  4. ngrok will print a URL like:
       Forwarding  https://abc123.ngrok-free.app -> http://localhost:5000
  5. Share that https://... URL with all student laptops.
     No browser warnings, camera works immediately.
  ─────────────────────────────────────────────────────────────────
""")
        sys.exit(0)

    # ── SSL cert ──────────────────────────────────────────────────────────────
    ssl_context = None
    if args.https:
        try:
            cert, key = generate_self_signed_cert()
            ssl_context = (cert, key)
            print(f"""
  HTTPS MODE — self-signed certificate
  ─────────────────────────────────────────────────────────────────
  Students will see a browser security warning once.
  Tell them: click "Advanced" → "Proceed to {lan_ip} (unsafe)"
  This is normal for self-signed certs and safe on a local network.
  ─────────────────────────────────────────────────────────────────""")
        except RuntimeError as e:
            print(f"\n  WARNING: {e}\n  Falling back to HTTP (camera may be blocked).\n")
            ssl_context = None
            scheme = "http"

    url = f"{scheme}://{lan_ip}:{PORT}"
    print(f"""
  Student URL  →  {url}
  Admin URL    →  {url}/admin
  Localhost    →  {scheme}://localhost:{PORT}
  Login:  admin / admin123   (run demo_setup.py to add students)

  Press Ctrl+C to stop.
""")

    socketio.run(
        app,
        host=HOST,
        port=PORT,
        debug=DEBUG,
        use_reloader=False,
        ssl_context=ssl_context,
    )
