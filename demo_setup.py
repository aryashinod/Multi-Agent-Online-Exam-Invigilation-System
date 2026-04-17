"""
demo_setup.py
=============
Pre-registers student and admin accounts for a multi-laptop demo.

Run this BEFORE or AFTER starting the server — it writes directly to
data/users.json so there is no network or SSL dependency.

Usage:
    python demo_setup.py              # creates 4 students
    python demo_setup.py --students 6 # creates 6 students
"""

import argparse
import hashlib
import json
import os
import socket

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ExamGuard demo setup")
parser.add_argument("--students", type=int, default=4,
                    help="Number of student accounts to create (default 4)")
args = parser.parse_args()
N = args.students

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(ROOT, "data", "users.json")
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)

# ── Load existing users ───────────────────────────────────────────────────────
try:
    with open(USERS_FILE) as f:
        users = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    users = {}

# ── Helper ────────────────────────────────────────────────────────────────────
def sha256(s):
    return hashlib.sha256(s.encode()).hexdigest()

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# ── Banner ────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  ExamGuard — Multi-Laptop Demo Setup")
print("=" * 60)

# ── Create admin ──────────────────────────────────────────────────────────────
if "admin" not in users:
    users["admin"] = {
        "name":          "Administrator",
        "password_hash": sha256("admin123"),
        "is_admin":      True,
    }
    print("  Admin account created  →  admin / admin123")
else:
    print("  Admin account already exists  →  admin / admin123")

# ── Create students ───────────────────────────────────────────────────────────
accounts = []
print(f"  Registering {N} student accounts …")
for i in range(1, N + 1):
    sid = f"student{i:02d}"
    pwd = f"pass{i:02d}"
    if sid not in users:
        users[sid] = {
            "name":          f"Student {i:02d}",
            "password_hash": sha256(pwd),
            "is_admin":      False,
        }
        status = "created"
    else:
        status = "already exists"
    print(f"    {sid}  /  {pwd}  →  {status}")
    accounts.append((sid, pwd))

# ── Save ──────────────────────────────────────────────────────────────────────
with open(USERS_FILE, "w") as f:
    json.dump(users, f, indent=2)
print(f"\n  Saved to: {USERS_FILE}")

# ── Print connection info ──────────────────────────────────────────────────────
lan_ip  = get_local_ip()
https_url = f"https://{lan_ip}:5000"

print()
print("=" * 60)
print("  CREDENTIALS — give one row to each student laptop")
print("=" * 60)
print(f"  {'Student ID':<14} {'Password':<12}  URL")
print(f"  {'-'*14} {'-'*12}  {'-'*36}")
for sid, pwd in accounts:
    print(f"  {sid:<14} {pwd:<12}  {https_url}")
print()
print(f"  Admin →  {https_url}/admin   (admin / admin123)")
print()
print("=" * 60)
print("  STEPS")
print("=" * 60)
print(f"""
  SERVER LAPTOP (you):
    1. python run.py --https      ← keep this running
    2. Open {https_url}/admin
    3. Click Advanced → Proceed (self-signed cert warning, once only)
    4. Login: admin / admin123

  EACH STUDENT LAPTOP:
    1. Open Chrome → {https_url}
    2. Click Advanced → Proceed (cert warning, once only)
    3. Login with Student ID and Password from the table above
    4. Click Begin Exam
    5. Allow Camera and Microphone when browser asks

  All laptops must be on the SAME Wi-Fi network.
""")
print("=" * 60)
print("  Done. Start the server with:  python run.py --https")
print("=" * 60)
print()
