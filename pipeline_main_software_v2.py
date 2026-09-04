"""
Alibaba Cloud AI Hackathon Pakistan 2026
AI for Pakistan's Future
Ifrah Gohar
===============================================
Handles BOTH analog and digital gauges simultaneously from a single camera.

Features:
  - Analog + Digital gauge detection and reading
  - MobileNetV2 gauge type classifier
  - HSV needle reading with calibration
  - 7-segment LED decoder
  - CAE anomaly detection for both gauges
  - Combined 3-level alert system
  - CCTV-style video recording
  - Analog + Digital frequency stats panels
  - /analytics page with diagnostics + prognostics
  - CSV data logging (every 30 seconds)
  - Separate Alert CSV logging
  - Shift Indicator (Morning/Evening/Night)
  - Shift-wise Report (Analog + Digital separate)
  - Standalone viewer when pipeline stopped

Dashboard  : http://localhost:5000
Analytics  : http://localhost:5000/analytics
Recordings : http://localhost:5000/recordings
Report     : http://localhost:5000/report
"""

import cv2
import numpy as np
import json
import os
import csv
import time
import threading
import subprocess
import shutil
from datetime import datetime
from collections import deque, Counter
import tensorflow as tf
from ultralytics import YOLO
from flask import Flask, render_template_string, Response, jsonify, send_from_directory, abort, request
from flask_socketio import SocketIO, emit
from segment_decoder import decode_all_digits, insert_decimal

# ================================================================
#  SETTINGS
# ================================================================

MODE       = "video"
VIDEO_PATH = "demo.mp4"
WEBCAM_ID  = 0

VOLTAGE_THRESHOLD = 25.0
PRE_WARNING_RATIO = 0.80
CRITICAL_RATIO    = 1.20

GAUGE_ID       = "GAUGE-001"
GAUGE_LOCATION = "Control Room"

# ================================================================
#  PAKISTAN INDUSTRY PRESETS
# ================================================================

INDUSTRY_PRESETS = {
    "Custom":         {"threshold": 25.0,  "pre_warn_ratio": 0.80, "critical_ratio": 1.20, "unit": "V"},
    "WAPDA":          {"threshold": 220.0, "pre_warn_ratio": 0.85, "critical_ratio": 1.10, "unit": "V"},
    "SNGPL":          {"threshold": 50.0,  "pre_warn_ratio": 0.80, "critical_ratio": 1.12, "unit": "PSI"},
    "OGDCL":          {"threshold": 100.0, "pre_warn_ratio": 0.80, "critical_ratio": 1.15, "unit": "PSI"},
    "Pakistan_Steel": {"threshold": 150.0, "pre_warn_ratio": 0.75, "critical_ratio": 1.20, "unit": "PSI"},
}
current_industry = "Custom"

FLASK_PORT = 5000
FLASK_HOST = "0.0.0.0"

SESSION_ID     = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR     = os.path.join(BASE_DIR, "recordings")
LOG_DIR        = os.path.join(BASE_DIR, "pipeline_logs")
RECORD_FPS     = 15
RECORD_ENABLED = True
RECORD_EVERY_N = 2
os.makedirs(RECORD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
MAINTENANCE_LOG = os.path.join(LOG_DIR, "maintenance_log.csv")

# ================================================================
#  SHIFT HELPER
# ================================================================

def get_shift():
    h = datetime.now().hour
    if 6 <= h < 14:  return "Morning"
    if 14 <= h < 22: return "Evening"
    return "Night"

# ================================================================
#  PATHS
# ================================================================

ANALOG_YOLO_MODEL  = os.path.join(BASE_DIR, "best_v8_centroid.pt")
CLASSIFIER_MODEL   = os.path.join(BASE_DIR, "gauge_classifier.keras")
CALIBRATION_FILE   = os.path.join(BASE_DIR, "sd670_calibration_FINAL.json")
MODEL_CONFIG_FILE  = os.path.join(BASE_DIR, "model_config.json")
ANALOG_CAE_MODEL   = os.path.join(BASE_DIR, "cae_best.tflite")
ANALOG_THRESHOLD_F = os.path.join(BASE_DIR, "auto_threshold.json")

DIGITAL_YOLO_MODEL = os.path.join(BASE_DIR, "best.pt")
DIGITAL_CAE_MODEL  = os.path.join(BASE_DIR, "cae_gauge.tflite")
DIGITAL_CAE_CONFIG = os.path.join(BASE_DIR, "cae_config.json")

# ================================================================
#  CONSTANTS
# ================================================================

CLS_GAUGE    = 0
CLS_CENTROID = 1
CLS_DIAL     = 2
CLS_NEEDLE   = 3
ANALOG_CONF  = 0.15
IMG_SIZE     = 128
CLASSIFIER_INPUT_SIZE = 160
CLASSIFIER_SMOOTH     = 10

DIGITAL_CONF  = 0.10
MIN_W         = 5    # lowered so narrow '1' digits pass through
MIN_H         = 30
DISPLAY_COLOR = "blue"
CAE_INTERVAL  = 10

GREEN  = (0,   200,   0)
YELLOW = (0,   200, 255)
ORANGE = (0,   140, 255)
RED    = (0,     0, 255)
WHITE  = (255, 255, 255)
CYAN   = (255, 200,   0)

RED_HSV_1_STD  = (np.array([0,   80, 60]),  np.array([10,  255, 255]))
RED_HSV_2_STD  = (np.array([170, 80, 60]),  np.array([180, 255, 255]))
RED_HSV_1_DARK = (np.array([0,   40, 20]),  np.array([15,  255, 255]))
RED_HSV_2_DARK = (np.array([165, 40, 20]),  np.array([180, 255, 255]))
DARK_THRESH    = 50

GAUGE_INFO = {
    "current_gauge":     {"unit": "A",   "label": "Current"},
    "pressure_gauge":    {"unit": "PSI", "label": "Pressure"},
    "temperature_gauge": {"unit": "C",   "label": "Temperature"},
    "voltage_gauge":     {"unit": "V",   "label": "Voltage"},
    "unknown":           {"unit": "?",   "label": "Unknown"},
}

ERRATIC_JUMP_THRESHOLD = 5.0
STUCK_REPEAT_COUNT     = 5

# ================================================================
#  GLOBAL STATE
# ================================================================

state = {
    "gauge_id":            GAUGE_ID,
    "gauge_location":      GAUGE_LOCATION,
    "analog_reading":      0.0,
    "analog_angle":        0.0,
    "analog_gauge_type":   "unknown",
    "analog_gauge_unit":   "V",
    "analog_gauge_label":  "Voltage",
    "analog_gauge_conf":   0.0,
    "analog_anomaly":      "NORMAL",
    "analog_mse":          0.0,
    "analog_detected":     False,
    "digital_reading":     0.0,
    "digital_reading_str": "N/A",
    "digital_anomaly":     "NORMAL",
    "digital_mse":         0.0,
    "digital_detected":    False,
    "digits_found":        0,
    "alert_level":         0,
    "alert_message":       "ALL SYSTEMS NORMAL",
    "timestamp":           "",
    "frame_count":         0,
    "log_count":           0,
    "pipeline_running":    True,
    "analog_stats":        {"total": 0, "bins": [], "levels": {"normal":0,"prewarn":0,"level2":0,"level3":0}},
    "digital_stats":       {"total": 0, "bins": [], "levels": {"normal":0,"prewarn":0,"level2":0,"level3":0}},
}

alert_history = deque(maxlen=20)
frame_lock    = threading.Lock()
current_frame = None
current_jpeg  = None

# ================================================================
#  READING STATS
# ================================================================

class ReadingStats:
    def __init__(self, bin_width=5, v_max=50):
        self.bin_width    = bin_width
        self.v_max        = v_max
        self.bin_counts   = Counter()
        self.level_counts = Counter()
        self.total        = 0

    def _bin_label(self, v):
        if v >= self.v_max:
            return f"{self.v_max}+"
        lo = int(v // self.bin_width) * self.bin_width
        return f"{lo}-{lo + self.bin_width}"

    def add(self, voltage, alert_level):
        self.level_counts[int(alert_level)] += 1
        self.total += 1
        if voltage is not None:
            self.bin_counts[self._bin_label(float(voltage))] += 1

    def summary(self):
        def low_edge(lbl):
            return int(lbl.replace("+","").split("-")[0])
        bins = sorted(self.bin_counts.items(), key=lambda kv: low_edge(kv[0]))
        return {
            "total": self.total,
            "bins":  [{"range": k+"V", "count": c} for k, c in bins],
            "levels": {
                "normal":  self.level_counts.get(0, 0),
                "prewarn": self.level_counts.get(1, 0),
                "level2":  self.level_counts.get(2, 0),
                "level3":  self.level_counts.get(3, 0),
            },
        }

analog_stats  = ReadingStats()
digital_stats = ReadingStats()

# ================================================================
#  VIDEO RECORDER
# ================================================================

class VideoRecorder:
    def __init__(self, out_dir, fps, session_id):
        self.out_dir      = out_dir
        self.fps          = fps
        self.current_path = os.path.join(out_dir, f"session_{session_id}.mp4")
        self.proc         = None
        self.lock         = threading.Lock()

    def _open(self, w, h):
        ffmpeg = next((p for p in [
            r"C:\Users\awan\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe",
            shutil.which("ffmpeg") or "",
        ] if p and os.path.isfile(p)), None)
        if not ffmpeg:
            print("[Recorder] ffmpeg not found")
            return
        cmd = [ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
               "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(self.fps),
               "-i", "pipe:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart+frag_keyframe+empty_moov",
               "-preset", "ultrafast", self.current_path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[Recorder] {os.path.basename(self.current_path)}")

    def write(self, frame):
        if not RECORD_ENABLED or frame is None:
            return
        h, w = frame.shape[:2]
        with self.lock:
            if self.proc is None:
                self._open(w, h)
            if self.proc and self.proc.poll() is None:
                try:
                    self.proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    pass

    def finalize(self):
        with self.lock:
            if self.proc:
                try:
                    self.proc.stdin.close()
                    self.proc.wait(timeout=10)
                except:
                    self.proc.kill()
                self.proc = None
        print(f"[Recorder] Saved: {os.path.basename(self.current_path)}")

recorder = VideoRecorder(RECORD_DIR, RECORD_FPS, SESSION_ID)

def recover_orphan_recordings():
    ff = shutil.which("ffmpeg")
    if ff is None:
        return
    for f in os.listdir(RECORD_DIR):
        if f.startswith("_raw_") and f.endswith(".mp4"):
            sid = f.replace("_raw_","").rsplit(".",1)[0]
            raw = os.path.join(RECORD_DIR, f)
            out = os.path.join(RECORD_DIR, f"session_{sid}.mp4")
            if os.path.exists(out):
                try: os.remove(raw)
                except: pass
                continue
            try:
                subprocess.run([ff,"-y","-i",raw,"-c:v","libx264",
                                "-pix_fmt","yuv420p","-movflags","+faststart",out],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.remove(raw)
            except Exception as e:
                print(f"[Recorder] Could not recover {f}: {e}")

# ================================================================
#  CSV LOGGERS
# ================================================================

class Logger:
    def __init__(self):
        self.path  = os.path.join(LOG_DIR, f"session_{SESSION_ID}.csv")
        self.count = 0
        with open(self.path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'timestamp','analog_reading','analog_gauge_type',
                'analog_anomaly','analog_mse',
                'digital_reading','digital_anomaly','digital_mse',
                'alert_level','alert_message','shift'])
        print(f"[Logger] {self.path}")

    def log(self, ar, agt, aa, am, dr, da, dm, cl, cmsg):
        self.count += 1
        clean_msg = cmsg.replace('\u2014', '-').replace('\u2013', '-')
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round(ar,2), agt,
                "DEFECTIVE" if aa else "NORMAL", round(am,6),
                dr, "DEFECTIVE" if da else "NORMAL", round(dm,6),
                cl, clean_msg, get_shift()])


class AlertLogger:
    """Separate CSV for alerts only"""
    def __init__(self):
        self.path = os.path.join(LOG_DIR, f"alerts_{SESSION_ID}.csv")
        with open(self.path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'timestamp','alert_level','alert_type',
                'analog_reading','digital_reading','message','shift'])
        print(f"[AlertLogger] {self.path}")

    def log(self, cl, analog_v, digital_str, cmsg):
        if cl == 0:
            return
        clean_msg = cmsg.replace('\u2014', '-').replace('\u2013', '-')
        alert_type = "L1-PreWarn" if cl==1 else ("L2-Warning" if cl==2 else ("L3-Critical" if cl==3 else "L4-Anomaly"))
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                cl, alert_type,
                round(analog_v,2), digital_str,
                clean_msg, get_shift()])

# ================================================================
#  FLASK APP
# ================================================================

app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A Smart Vision Based Industrial Gauge Monitoring System</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@400;600;700;900&display=swap');
  :root{--bg:#0a0e1a;--panel:#111827;--border:#1e293b;--green:#00e676;--yellow:#ffd600;--orange:#ff6d00;--red:#ff1744;--cyan:#00e5ff;--teal:#00bfa5;--text:#e2e8f0;--muted:#64748b;}
  *{margin:0;padding:0;box-sizing:border-box;}
  body{background:var(--bg);color:var(--text);font-family:'Barlow',sans-serif;min-height:100vh;}
  header{background:linear-gradient(90deg,#0f172a,#1e293b);border-bottom:1px solid var(--border);padding:14px 28px;display:flex;align-items:center;justify-content:space-between;}
  header h1{font-size:1.1rem;font-weight:700;letter-spacing:.05em;color:var(--teal);text-transform:uppercase;}
  header .sub{font-size:.75rem;color:var(--muted);margin-top:2px;}
  .nav-link{color:var(--cyan);text-decoration:none;font-size:.8rem;border:1px solid var(--cyan);padding:6px 14px;border-radius:6px;margin-left:14px;}
  #clock{font-family:'Share Tech Mono',monospace;font-size:1.1rem;color:var(--cyan);}
  .shift-strip{display:flex;gap:20px;align-items:center;padding:8px 28px;background:#0d1424;border-bottom:1px solid var(--border);}
  .shift-badge{padding:3px 10px;border-radius:4px;font-size:.68rem;font-weight:700;}
  .shift-Morning{background:#1a2a0d;color:#a3e635;border:1px solid #a3e635;}
  .shift-Evening{background:#2a1500;color:var(--orange);border:1px solid var(--orange);}
  .shift-Night{background:#0d1424;color:var(--cyan);border:1px solid var(--cyan);}
  .pipeline-status{font-size:.72rem;font-family:'Share Tech Mono',monospace;padding:3px 10px;border-radius:4px;}
  .status-running{background:#003020;color:var(--green);border:1px solid var(--green);}
  .status-stopped{background:#2a0010;color:var(--red);border:1px solid var(--red);}
  #alert-banner{padding:14px 28px;font-size:1.1rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;text-align:center;background:#1a2a1a;color:var(--green);border-bottom:2px solid var(--green);transition:all .4s;}
  #alert-banner.level1{background:#2a2200;color:var(--yellow);border-color:var(--yellow);}
  #alert-banner.level2{background:#2a1500;color:var(--orange);border-color:var(--orange);}
  #alert-banner.level3{background:#2a0010;color:var(--red);border-color:var(--red);animation:flash .8s infinite alternate;}
  #alert-banner.level4{background:#0a1a2a;color:#00e5ff;border-color:#00e5ff;animation:flash2 1s infinite alternate;}
  @keyframes flash{from{opacity:1}to{opacity:.5}}
  @keyframes flash2{from{opacity:1}to{opacity:.6}}
  .grid{display:grid;grid-template-columns:1fr 1fr 1.4fr;gap:16px;padding:16px 28px;}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:20px;}
  .card-title{font-size:.7rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:14px;border-bottom:1px solid var(--border);padding-bottom:8px;}
  .gauge-type-badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.72rem;font-weight:700;text-transform:uppercase;margin-bottom:8px;background:#0d2a3a;color:var(--teal);border:1px solid var(--teal);}
  .gauge-conf{font-size:.7rem;color:var(--muted);margin-left:6px;}
  .reading-big{font-family:'Share Tech Mono',monospace;font-size:3.5rem;font-weight:900;line-height:1;color:var(--green);transition:color .3s;}
  .reading-unit{font-size:1.3rem;color:var(--muted);margin-left:4px;}
  .threshold-row{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;align-items:center;}
  .thr-badge{font-size:.68rem;padding:3px 8px;border-radius:4px;font-family:'Share Tech Mono',monospace;}
  .thr-badge.norm{background:#003020;color:#00e676;border:1px solid #00e676;}
  .thr-badge.warn{background:#2a2200;color:var(--yellow);border:1px solid var(--yellow);}
  .thr-badge.alert{background:#2a1500;color:var(--orange);border:1px solid var(--orange);}
  .thr-badge.crit{background:#2a0010;color:var(--red);border:1px solid var(--red);}
  .thr-badge.anom-always{background:#0a1a2a;color:#00e5ff;border:1px solid #00e5ff;}
  .thr-badge.anom{background:#0a1a2a;color:#00e5ff;border:1px solid #00e5ff;}
  .anom-status-label{font-size:.62rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:4px;margin-top:2px;}
  .volt-bar-wrap{margin-top:10px;height:8px;background:var(--border);border-radius:4px;overflow:hidden;}
  .volt-bar{height:100%;border-radius:4px;transition:width .3s,background .3s;background:var(--green);}
  .anomaly-status{font-size:1.6rem;font-weight:900;transition:color .3s;}
  .anomaly-mse{font-family:'Share Tech Mono',monospace;font-size:.75rem;color:var(--muted);margin-top:6px;}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;animation:pulse 1.5s infinite;}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
  #camera-feed{width:100%;height:300px;object-fit:contain;border-radius:6px;border:1px solid var(--border);background:#000;cursor:zoom-in;}
  .rec-dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--red);margin-right:6px;animation:pulse 1.2s infinite;}
  .chart-card{grid-column:1/3;}
  canvas{max-height:160px;}
  .alerts-card{overflow-y:auto;max-height:240px;}
  table{width:100%;border-collapse:collapse;font-size:.72rem;}
  th{text-align:left;padding:5px 7px;color:var(--muted);border-bottom:1px solid var(--border);position:sticky;top:0;background:var(--panel);}
  td{padding:4px 7px;border-bottom:1px solid #1a2030;}
  .badge{padding:2px 6px;border-radius:3px;font-size:.62rem;font-weight:700;font-family:'Share Tech Mono',monospace;}
  .badge.l0{background:#003020;color:var(--green);}
  .badge.l1{background:#2a2200;color:var(--yellow);}
  .badge.l2{background:#2a1500;color:var(--orange);}
  .badge.l3{background:#2a0010;color:var(--red);}
  .badge.l4{background:#0a1a2a;color:#00e5ff;border:1px solid #00e5ff;}
  .stats-row{display:flex;gap:16px;margin-top:10px;flex-wrap:wrap;}
  .stat{font-size:.72rem;}
  .stat-label{color:var(--muted);font-size:.62rem;text-transform:uppercase;}
  .stat-value{font-family:'Share Tech Mono',monospace;color:var(--cyan);}
  .divider{border:none;border-top:1px solid var(--border);margin:12px 0;}
  .freq-row{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:.72rem;}
  .freq-label{width:60px;font-family:'Share Tech Mono',monospace;color:var(--muted);}
  .freq-bar-wrap{flex:1;height:12px;background:var(--border);border-radius:3px;overflow:hidden;}
  .freq-bar{height:100%;border-radius:3px;}
  .freq-count{width:40px;text-align:right;font-family:'Share Tech Mono',monospace;color:var(--text);}
  .level-chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;}
  .level-chip{font-family:'Share Tech Mono',monospace;font-size:.68rem;padding:3px 7px;border-radius:4px;}
  footer{text-align:center;padding:10px;font-size:.7rem;color:var(--muted);border-top:1px solid var(--border);}
</style>
</head>
<body>
<header>
  <div>
    <h1>A Smart Vision Based Monitoring System For Industrial Gauges</h1>
    <div class="sub">Alibaba Cloud AI Hackathon Pakistan 2026 &nbsp;|&nbsp; AI for Pakistan's Future &nbsp;|&nbsp; Ifrah Gohar</div>
    <div class="sub" style="margin-top:3px;color:var(--cyan);font-family:'Share Tech Mono',monospace;font-size:.7rem;">
      &#9632; {{ gauge_id }} &nbsp;|&nbsp; {{ gauge_location }}
    </div>
  </div>
  <div style="display:flex;align-items:center;">
    <div id="clock">--:--:--</div>
    <a class="nav-link" href="/analytics" target="_blank">Analytics</a>
    <a class="nav-link" href="/recordings" target="_blank">Recordings</a>
    <a class="nav-link" href="/report" target="_blank">Report</a>
    <a class="nav-link" href="/maintenance" target="_blank" style="border-color:#ff6d00;color:#ff6d00;">&#128295; Maintenance</a>
  </div>
</header>

<div class="shift-strip">
  <span style="font-size:.72rem;font-family:'Share Tech Mono',monospace;color:var(--muted);">Shift:</span>
  <span id="info-shift" class="shift-badge shift-Night">Night</span>
  <span id="pipeline-status" class="pipeline-status status-running">PIPELINE RUNNING</span>
</div>

<div id="alert-banner">ALL SYSTEMS NORMAL</div>

<div class="grid">
  <!-- ANALOG GAUGE -->
  <div class="card">
    <div class="card-title">Analog Gauge</div>
    <div>
      <span id="analog-badge" class="gauge-type-badge">Detecting...</span>
      <span class="gauge-conf" id="analog-conf"></span>
    </div>
    <div>
      <span class="reading-big" id="analog-reading">--.-</span>
      <span class="reading-unit" id="analog-unit">V</span>
    </div>
    <div class="threshold-row">
      <span class="thr-badge norm">&#9679; NORMAL</span>
      <span class="thr-badge warn">Pre-warn: {{ pre_warn }}V</span>
      <span class="thr-badge alert">Alert: {{ threshold }}V</span>
      <span class="thr-badge crit">Critical: {{ critical }}V</span>
      <span class="thr-badge anom-always">&#9888; ANOMALY ALERT</span>
      <span class="thr-badge anom" id="analog-anom-badge">&#9888; ANOMALY</span>
    </div>
    <div class="volt-bar-wrap"><div class="volt-bar" id="analog-bar" style="width:0%"></div></div>
    <hr class="divider">
    <div class="anom-status-label">ANOMALY STATUS</div>
    <div class="anomaly-status" id="analog-anomaly"><span class="dot" style="background:var(--green)"></span>NORMAL</div>
    <div class="anomaly-mse" id="analog-mse">MSE: 0.00000</div>
    <div class="stats-row">
      <div class="stat"><div class="stat-label">Angle</div><div class="stat-value" id="stat-angle">--</div></div>
      <div class="stat"><div class="stat-label">Frames</div><div class="stat-value" id="stat-frames">0</div></div>
      <div class="stat"><div class="stat-label">Logs</div><div class="stat-value" id="stat-logs">0</div></div>
    </div>
  </div>

  <!-- DIGITAL GAUGE -->
  <div class="card">
    <div class="card-title">Digital Gauge</div>
    <div class="gauge-type-badge" style="background:#1a2a0d;color:#a3e635;border-color:#a3e635;">7-Segment LED</div>
    <div>
      <span class="reading-big" id="digital-reading">--.-</span>
      <span class="reading-unit">V</span>
    </div>
    <div style="font-size:.72rem;color:var(--muted);margin-top:6px;">
      Digits: <span id="stat-digits" style="color:var(--cyan)">0</span>
    </div>
    <div class="threshold-row">
      <span class="thr-badge norm">&#9679; NORMAL</span>
      <span class="thr-badge warn">Pre-warn: {{ pre_warn }}V</span>
      <span class="thr-badge alert">Alert: {{ threshold }}V</span>
      <span class="thr-badge crit">Critical: {{ critical }}V</span>
      <span class="thr-badge anom-always">&#9888; ANOMALY ALERT</span>
      <span class="thr-badge anom" id="digital-anom-badge">&#9888; ANOMALY</span>
    </div>
    <div class="volt-bar-wrap"><div class="volt-bar" id="digital-bar" style="width:0%"></div></div>
    <hr class="divider">
    <div class="anom-status-label">ANOMALY STATUS</div>
    <div class="anomaly-status" id="digital-anomaly"><span class="dot" style="background:var(--green)"></span>NORMAL</div>
    <div class="anomaly-mse" id="digital-mse">MSE: 0.00000</div>
    <hr class="divider">
    <div style="font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;">Pakistan Industry</div>
    <select id="industry-select" onchange="changeIndustry()" style="background:#0d1424;border:1px solid var(--border);color:var(--cyan);padding:6px 10px;border-radius:6px;font-family:'Share Tech Mono',monospace;font-size:.78rem;width:100%;">
      <option value="Custom">Custom (25V)</option>
      <option value="WAPDA">WAPDA Power (220V)</option>
      <option value="SNGPL">SNGPL Gas (50PSI)</option>
      <option value="OGDCL">OGDCL Oil (100PSI)</option>
      <option value="Pakistan_Steel">Pakistan Steel (150PSI)</option>
    </select>
    <div style="font-size:.65rem;color:var(--muted);margin-top:4px;">Selected: <span id="current-industry" style="color:var(--cyan);">Custom</span></div>
  </div>

  <!-- CAMERA FEED -->
  <div class="card">
    <div class="card-title"><span class="rec-dot"></span>Live Camera Feed - click to maximize</div>
    <img id="camera-feed" src="" alt="Camera Feed"/>
  </div>

  <!-- CHART -->
  <div class="card chart-card">
    <div class="card-title">Reading History (last 60)</div>
    <canvas id="chart"></canvas>
  </div>

  <!-- ALERTS -->
  <div class="card alerts-card">
    <div class="card-title">Recent Alerts</div>
    <table>
      <thead><tr><th>Time</th><th>Level</th><th>Type</th><th>Value</th><th>Message</th></tr></thead>
      <tbody id="alerts-tbody">
        <tr><td colspan="5" style="color:var(--muted);text-align:center;">No alerts yet</td></tr>
      </tbody>
    </table>
  </div>

  <!-- ANALOG FREQ -->
  <div class="card" style="grid-column:1/2;">
    <div class="card-title">Analog Reading Frequency</div>
    <div id="analog-freq-bins"><div style="color:var(--muted);font-size:.8rem;">Collecting data...</div></div>
    <div class="level-chips" id="analog-level-chips"></div>
    <div style="margin-top:6px;font-size:.7rem;color:var(--muted);">Total: <span id="analog-stat-total" style="color:var(--cyan);">0</span></div>
  </div>

  <!-- DIGITAL FREQ -->
  <div class="card" style="grid-column:2/3;">
    <div class="card-title">Digital Reading Frequency</div>
    <div id="digital-freq-bins"><div style="color:var(--muted);font-size:.8rem;">Collecting data...</div></div>
    <div class="level-chips" id="digital-level-chips"></div>
    <div style="margin-top:6px;font-size:.7rem;color:var(--muted);">Total: <span id="digital-stat-total" style="color:var(--cyan);">0</span></div>
  </div>
</div>

<footer>Smart Vision Based Monitoring System For Industrial Gauges &nbsp;|&nbsp; Alibaba Cloud AI Hackathon Pakistan 2026 &nbsp;|&nbsp; AI for Pakistan's Future</footer>

<script>
const socket=io();
const camImg=document.getElementById('camera-feed');
function refreshSnapshot(){
  const n=new Image();
  n.onload=()=>{camImg.src=n.src;setTimeout(refreshSnapshot,100);};
  n.onerror=()=>setTimeout(refreshSnapshot,500);
  n.src='/snapshot?t='+Date.now();
}
refreshSnapshot();
camImg.addEventListener('click',()=>{if(camImg.requestFullscreen)camImg.requestFullscreen();});

setInterval(()=>{
  document.getElementById('clock').textContent=new Date().toLocaleTimeString('en-GB');
  const h=new Date().getHours();
  const shift=h>=6&&h<14?'Morning':h>=14&&h<22?'Evening':'Night';
  const el=document.getElementById('info-shift');
  el.textContent=shift;
  el.className='shift-badge shift-'+shift;
},1000);

const ctx=document.getElementById('chart').getContext('2d');
const chart=new Chart(ctx,{
  type:'line',
  data:{labels:[],datasets:[
    {label:'Analog',data:[],borderColor:'#00e5ff',backgroundColor:'rgba(0,229,255,0.05)',borderWidth:2,pointRadius:0,tension:0.3,fill:true},
    {label:'Digital',data:[],borderColor:'#a3e635',backgroundColor:'rgba(163,230,53,0.05)',borderWidth:2,pointRadius:0,tension:0.3,fill:true}
  ]},
  options:{responsive:true,maintainAspectRatio:true,animation:false,
    plugins:{legend:{labels:{color:'#64748b',font:{family:'Share Tech Mono'}}}},
    scales:{x:{display:false},y:{min:0,max:50,grid:{color:'rgba(255,255,255,0.05)'},ticks:{color:'#64748b',font:{family:'Share Tech Mono'}}}}}
});

const COLORS={0:{banner:'',c:'#00e676'},1:{banner:'level1',c:'#ffd600'},2:{banner:'level2',c:'#ff6d00'},3:{banner:'level3',c:'#ff1744'},4:{banner:'level4',c:'#00e5ff'}};

function changeIndustry(){
  const val=document.getElementById('industry-select').value;
  fetch('/api/set_industry',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({industry:val})})
  .then(r=>r.json()).then(d=>{
    document.getElementById('current-industry').textContent=d.name;
  });
}

function renderFreq(stats,binsId,chipsId,totalId,barColor){
  if(!stats)return;
  document.getElementById(totalId).textContent=stats.total||0;
  const bins=stats.bins||[];
  const maxC=bins.reduce((m,b)=>Math.max(m,b.count),1);
  const wrap=document.getElementById(binsId);
  if(!bins.length){wrap.innerHTML='<div style="color:var(--muted);font-size:.8rem;">Collecting...</div>';}
  else{wrap.innerHTML=bins.map(b=>{const pct=Math.round(b.count/maxC*100);return'<div class="freq-row"><div class="freq-label">'+b.range+'</div><div class="freq-bar-wrap"><div class="freq-bar" style="width:'+pct+'%;background:'+barColor+'"></div></div><div class="freq-count">'+b.count+'</div></div>';}).join('');}
  const lv=stats.levels||{normal:0,prewarn:0,level2:0,level3:0};
  document.getElementById(chipsId).innerHTML='<span class="level-chip badge l0">NORMAL x'+lv.normal+'</span><span class="level-chip badge l1">PRE-WARN x'+lv.prewarn+'</span><span class="level-chip badge l2">L2 x'+lv.level2+'</span><span class="level-chip badge l3">L3 x'+lv.level3+'</span>';
}

socket.on('state_update',(d)=>{
  const c=COLORS[d.alert_level]||COLORS[0];
  const banner=document.getElementById('alert-banner');
  banner.textContent=d.alert_message;
  banner.className=c.banner;

  const ps=document.getElementById('pipeline-status');
  if(d.pipeline_running){ps.textContent='PIPELINE RUNNING';ps.className='pipeline-status status-running';}
  else{ps.textContent='PIPELINE STOPPED';ps.className='pipeline-status status-stopped';}

  document.getElementById('analog-badge').textContent=d.analog_gauge_label||'Voltage';
  document.getElementById('analog-conf').textContent=d.analog_gauge_conf?(Math.round(d.analog_gauge_conf*100)+'%'):'';
  document.getElementById('analog-reading').textContent=d.analog_detected?d.analog_reading.toFixed(2):'--.-';
  document.getElementById('analog-reading').style.color=d.analog_detected?c.c:'#64748b';
  document.getElementById('analog-unit').textContent=d.analog_gauge_unit||'V';
  const ap=Math.min((d.analog_reading||0)/50*100,100);
  document.getElementById('analog-bar').style.width=ap+'%';
  document.getElementById('analog-bar').style.background=c.c;
  document.getElementById('stat-angle').textContent=(d.analog_angle||0).toFixed(1)+'deg';
  document.getElementById('stat-frames').textContent=d.frame_count||0;
  document.getElementById('stat-logs').textContent=d.log_count||0;
  const aAnom=d.analog_anomaly==='DEFECTIVE';
  document.getElementById('analog-anomaly').innerHTML='<span class="dot" style="background:'+(aAnom?'#ff1744':'#00e676')+'"></span>'+(aAnom?'DEFECTIVE':'NORMAL');
  document.getElementById('analog-anomaly').style.color=aAnom?'#ff1744':'#00e676';
  document.getElementById('analog-mse').textContent='MSE: '+(d.analog_mse||0).toFixed(5);
  // Show anomaly badge only when DEFECTIVE
  document.getElementById('analog-anom-badge').style.display=aAnom?'inline-block':'none';

  document.getElementById('digital-reading').textContent=d.digital_detected?(d.digital_reading_str||'--.-'):'--.-';
  document.getElementById('digital-reading').style.color=d.digital_detected?'#a3e635':'#64748b';
  const dp=Math.min((d.digital_reading||0)/50*100,100);
  document.getElementById('digital-bar').style.width=dp+'%';
  document.getElementById('digital-bar').style.background=d.digital_detected?'#a3e635':'#64748b';
  document.getElementById('stat-digits').textContent=d.digits_found||0;
  const dAnom=d.digital_anomaly==='DEFECTIVE';
  document.getElementById('digital-anomaly').innerHTML='<span class="dot" style="background:'+(dAnom?'#ff1744':'#00e676')+'"></span>'+(dAnom?'DEFECTIVE':'NORMAL');
  document.getElementById('digital-anomaly').style.color=dAnom?'#ff1744':'#00e676';
  document.getElementById('digital-mse').textContent='MSE: '+(d.digital_mse||0).toFixed(5);
  // Show anomaly badge only when DEFECTIVE
  document.getElementById('digital-anom-badge').style.display=dAnom?'inline-block':'none';

  chart.data.labels.push(new Date().toLocaleTimeString('en-GB'));
  chart.data.datasets[0].data.push(d.analog_reading||0);
  chart.data.datasets[1].data.push(d.digital_detected?(d.digital_reading||0):null);
  if(chart.data.labels.length>60){chart.data.labels.shift();chart.data.datasets.forEach(ds=>ds.data.shift());}
  chart.update();

  renderFreq(d.analog_stats,'analog-freq-bins','analog-level-chips','analog-stat-total','#00e5ff');
  renderFreq(d.digital_stats,'digital-freq-bins','digital-level-chips','digital-stat-total','#a3e635');
});

socket.on('new_alert',(d)=>{
  const tbody=document.getElementById('alerts-tbody');
  const ph=tbody.querySelector('td[colspan]');
  if(ph)ph.parentElement.remove();
  const cls=['l0','l1','l2','l3','l4'][d.level]||'l0';
  const row=document.createElement('tr');
  row.innerHTML='<td style="font-family:Share Tech Mono;font-size:.68rem">'+d.time+'</td><td><span class="badge '+cls+'">L'+d.level+'</span></td><td style="color:var(--teal);font-size:.68rem">'+d.gauge_type+'</td><td style="font-family:Share Tech Mono">'+d.value+'</td><td style="color:#94a3b8;font-size:.68rem">'+d.message+'</td>';
  tbody.insertBefore(row,tbody.firstChild);
  while(tbody.rows.length>20)tbody.deleteRow(tbody.rows.length-1);
});
</script>
</body>
</html>
"""

ANALYTICS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Analytics</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@400;600;700;900&display=swap');
  :root{--bg:#0a0e1a;--panel:#111827;--border:#1e293b;--green:#00e676;--yellow:#ffd600;--orange:#ff6d00;--red:#ff1744;--cyan:#00e5ff;--teal:#00bfa5;--text:#e2e8f0;--muted:#64748b;}
  *{margin:0;padding:0;box-sizing:border-box;}
  body{background:var(--bg);color:var(--text);font-family:'Barlow',sans-serif;}
  header{background:linear-gradient(90deg,#0f172a,#1e293b);border-bottom:1px solid var(--border);padding:14px 28px;display:flex;align-items:center;justify-content:space-between;}
  header h1{font-size:1.1rem;font-weight:700;color:var(--teal);text-transform:uppercase;}
  .nav-link{color:var(--cyan);text-decoration:none;font-size:.8rem;border:1px solid var(--cyan);padding:6px 14px;border-radius:6px;margin-left:10px;}
  .wrap{padding:20px 28px;}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:16px;}
  .card-title{font-size:.7rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:14px;border-bottom:1px solid var(--border);padding-bottom:8px;}
  .session-list{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px;}
  .session-cb{display:flex;align-items:center;gap:6px;background:#0d1424;border:1px solid var(--border);border-radius:6px;padding:7px 12px;cursor:pointer;font-family:'Share Tech Mono',monospace;font-size:.75rem;}
  .session-cb:hover{border-color:var(--cyan);}
  .session-cb input{accent-color:var(--cyan);}
  .btn{background:var(--teal);color:#000;border:none;padding:9px 22px;border-radius:6px;font-weight:700;cursor:pointer;font-size:.85rem;margin-right:8px;}
  .btn.sec{background:transparent;color:var(--cyan);border:1px solid var(--cyan);}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
  .grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;}
  canvas{max-height:220px;}
  .kpi{text-align:center;padding:16px;}
  .kpi-val{font-family:'Share Tech Mono',monospace;font-size:2rem;font-weight:900;color:var(--cyan);}
  .kpi-label{font-size:.72rem;color:var(--muted);margin-top:4px;text-transform:uppercase;}
  .kpi-val.warn{color:var(--yellow);}.kpi-val.danger{color:var(--red);}.kpi-val.ok{color:var(--green);}
  table{width:100%;border-collapse:collapse;font-size:.75rem;}
  th{text-align:left;padding:6px 8px;color:var(--muted);border-bottom:1px solid var(--border);}
  td{padding:5px 8px;border-bottom:1px solid #1a2030;}
  .badge{padding:2px 7px;border-radius:3px;font-size:.65rem;font-weight:700;}
  .badge.erratic{background:#2a1500;color:var(--orange);}
  .badge.stuck{background:#2a2200;color:var(--yellow);}
  .badge.range{background:#2a0010;color:var(--red);}
  .eta-box{background:#0d1424;border:1px solid var(--border);border-radius:8px;padding:14px;margin-top:10px;}
  .eta-label{font-size:.7rem;color:var(--muted);text-transform:uppercase;margin-bottom:4px;}
  .eta-val{font-family:'Share Tech Mono',monospace;font-size:1.1rem;font-weight:700;}
  .health-bar-wrap{height:16px;background:var(--border);border-radius:8px;overflow:hidden;margin-top:10px;}
  .health-bar{height:100%;border-radius:8px;transition:width .5s;}
  #status-msg{font-size:.85rem;color:var(--muted);margin-top:10px;}
  .stopped-banner{background:#2a0010;color:var(--red);border:1px solid var(--red);padding:10px 20px;border-radius:8px;font-family:'Share Tech Mono',monospace;font-size:.85rem;margin-bottom:16px;display:none;}
</style>
</head>
<body>
<header>
  <h1>Historical Data Analytics</h1>
  <div>
    <a class="nav-link" href="/">Dashboard</a>
    <a class="nav-link" href="/recordings">Recordings</a>
    <a class="nav-link" href="/report">Report</a>
  </div>
</header>
<div class="wrap">
  <div id="stopped-banner" class="stopped-banner">Pipeline stopped — showing saved session data</div>
  <div class="card">
    <div class="card-title">Select Sessions to Analyze</div>
    <div class="session-list" id="session-list"><div style="color:var(--muted)">Loading sessions...</div></div>
    <button class="btn" onclick="analyze()">Analyze Selected</button>
    <button class="btn sec" onclick="selectAll()">Select All</button>
    <button class="btn sec" onclick="clearAll()">Clear</button>
    <div id="status-msg"></div>
  </div>
  <div class="card grid3" id="kpi-card" style="display:none;">
    <div class="kpi"><div class="kpi-val" id="kpi-total">0</div><div class="kpi-label">Total Readings</div></div>
    <div class="kpi"><div class="kpi-val" id="kpi-avg">0.0V</div><div class="kpi-label">Avg Analog Voltage</div></div>
    <div class="kpi"><div class="kpi-val" id="kpi-max">0.0V</div><div class="kpi-label">Peak Voltage</div></div>
    <div class="kpi"><div class="kpi-val" id="kpi-erratic">0</div><div class="kpi-label">Erratic Readings</div></div>
    <div class="kpi"><div class="kpi-val" id="kpi-stuck">0</div><div class="kpi-label">Stuck Readings</div></div>
    <div class="kpi"><div class="kpi-val" id="kpi-alerts">0</div><div class="kpi-label">Total Alerts</div></div>
  </div>
  <div class="card" id="trend-card" style="display:none;">
    <div class="card-title">Voltage Trend — Analog (with anomaly markers)</div>
    <canvas id="trendChart"></canvas>
  </div>
  <div class="grid2" id="diag-prg-card" style="display:none;">
    <div class="card">
      <div class="card-title">Diagnostics — Reading Anomaly Detection</div>
      <table>
        <thead><tr><th>Timestamp</th><th>Voltage</th><th>Type</th><th>Detail</th></tr></thead>
        <tbody id="diag-tbody"></tbody>
      </table>
    </div>
    <div class="card">
      <div class="card-title">Prognostics — Trend Prediction</div>
      <div class="eta-box"><div class="eta-label">Trend Direction</div><div class="eta-val" id="prg-trend">--</div></div>
      <div class="eta-box"><div class="eta-label">ETA to Pre-Warning ({{ pre_warn }}V)</div><div class="eta-val" id="prg-eta-warn">--</div></div>
      <div class="eta-box"><div class="eta-label">ETA to Alert Threshold ({{ threshold }}V)</div><div class="eta-val" id="prg-eta-alert">--</div></div>
      <div class="eta-box"><div class="eta-label">ETA to Critical ({{ critical }}V)</div><div class="eta-val" id="prg-eta-crit">--</div></div>
      <div style="margin-top:12px;">
        <div style="font-size:.72rem;color:var(--muted);margin-bottom:4px;">System Health Score</div>
        <div class="health-bar-wrap"><div class="health-bar" id="health-bar" style="width:100%;background:var(--green);"></div></div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:.85rem;margin-top:4px;" id="health-score">100%</div>
      </div>
    </div>
  </div>
  <div class="card" id="alert-dist-card" style="display:none;">
    <div class="card-title">Alert Level Distribution</div>
    <div style="max-width:400px;margin:0 auto;"><canvas id="alertPieChart"></canvas></div>
  </div>
</div>
<script>
let trendChartObj=null,pieChartObj=null;
fetch('/api/pipeline_status').then(r=>r.json()).then(d=>{
  if(!d.running) document.getElementById('stopped-banner').style.display='block';
});
fetch('/api/log_sessions').then(r=>r.json()).then(sessions=>{
  const el=document.getElementById('session-list');
  if(!sessions.length){el.innerHTML='<div style="color:var(--muted)">No session logs found.</div>';return;}
  el.innerHTML=sessions.map(s=>'<label class="session-cb"><input type="checkbox" value="'+s.file+'"> '+s.label+'</label>').join('');
});
function selectAll(){document.querySelectorAll('#session-list input').forEach(c=>c.checked=true);}
function clearAll(){document.querySelectorAll('#session-list input').forEach(c=>c.checked=false);}
function analyze(){
  const selected=[...document.querySelectorAll('#session-list input:checked')].map(c=>c.value);
  if(!selected.length){document.getElementById('status-msg').textContent='Select at least one session.';return;}
  document.getElementById('status-msg').textContent='Analyzing...';
  fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({files:selected})})
  .then(r=>r.json()).then(data=>{document.getElementById('status-msg').textContent='Analysis complete - '+data.total+' readings.';renderAll(data);});
}
function renderAll(d){
  ['kpi-card','trend-card','diag-prg-card','alert-dist-card'].forEach(id=>{document.getElementById(id).style.display='';});
  document.getElementById('kpi-total').textContent=d.total;
  document.getElementById('kpi-avg').textContent=d.avg_voltage.toFixed(1)+'V';
  const maxEl=document.getElementById('kpi-max');
  maxEl.textContent=d.max_voltage.toFixed(1)+'V';
  maxEl.className='kpi-val'+(d.max_voltage>=30?' danger':d.max_voltage>=25?' warn':' ok');
  document.getElementById('kpi-erratic').textContent=d.erratic_count;
  document.getElementById('kpi-stuck').textContent=d.stuck_count;
  document.getElementById('kpi-alerts').textContent=d.alert_counts.l1+d.alert_counts.l2+d.alert_counts.l3;
  if(trendChartObj)trendChartObj.destroy();
  const ctx=document.getElementById('trendChart').getContext('2d');
  const ptColors=d.timestamps.map((_,i)=>{const t=d.point_types[i];if(t==='erratic')return'#ff6d00';if(t==='stuck')return'#ffd600';if(t==='outofrange')return'#ff1744';return'#00e5ff';});
  const ptSizes=d.point_types.map(t=>t==='normal'?0:5);
  const datasets=[{label:'Analog Voltage',data:d.analog_readings,borderColor:'#00e5ff',backgroundColor:'rgba(0,229,255,0.05)',borderWidth:1.5,pointRadius:ptSizes,pointBackgroundColor:ptColors,tension:0.2,fill:true}];
  if(d.regression_line&&d.regression_line.length>0){datasets.push({label:'Trend (Linear Regression)',data:d.regression_line,borderColor:'#ff6d00',borderWidth:2,borderDash:[6,4],pointRadius:0,tension:0,fill:false});}
  trendChartObj=new Chart(ctx,{type:'line',data:{labels:d.timestamps,datasets:datasets},options:{responsive:true,animation:false,plugins:{legend:{labels:{color:'#64748b'}}},scales:{x:{ticks:{color:'#64748b',maxTicksLimit:10}},y:{min:0,max:55,ticks:{color:'#64748b'}}}}});
  const tbody=document.getElementById('diag-tbody');
  tbody.innerHTML=d.anomalies.length?d.anomalies.map(a=>'<tr><td style="font-family:monospace;font-size:.68rem">'+a.time+'</td><td style="font-family:monospace">'+a.voltage.toFixed(2)+'V</td><td><span class="badge '+a.type+'">'+a.type.toUpperCase()+'</span></td><td style="color:#94a3b8;font-size:.7rem">'+a.detail+'</td></tr>').join(''):'<tr><td colspan="4" style="color:var(--muted)">No anomalies</td></tr>';
  const prg=d.prognostics;
  const trendEl=document.getElementById('prg-trend');
  trendEl.textContent=prg.trend_direction;
  trendEl.style.color=prg.trend_direction==='RISING'?'#ff6d00':prg.trend_direction==='FALLING'?'#00e676':'#64748b';
  document.getElementById('prg-eta-warn').textContent=prg.eta_prewarn||'N/A';
  document.getElementById('prg-eta-alert').textContent=prg.eta_alert||'N/A';
  document.getElementById('prg-eta-crit').textContent=prg.eta_critical||'N/A';
  const hs=Math.max(0,Math.min(100,prg.health_score));
  document.getElementById('health-score').textContent=hs.toFixed(0)+'%';
  const hbar=document.getElementById('health-bar');
  hbar.style.width=hs+'%';
  hbar.style.background=hs>70?'#00e676':hs>40?'#ffd600':'#ff1744';
  if(pieChartObj)pieChartObj.destroy();
  const pctx=document.getElementById('alertPieChart').getContext('2d');
  pieChartObj=new Chart(pctx,{type:'doughnut',data:{labels:['Normal','Pre-Warning','Level 2','Level 3'],datasets:[{data:[d.alert_counts.l0,d.alert_counts.l1,d.alert_counts.l2,d.alert_counts.l3],backgroundColor:['#00e676','#ffd600','#ff6d00','#ff1744'],borderColor:'#111827',borderWidth:2}]},options:{responsive:true,plugins:{legend:{labels:{color:'#64748b'}}}}});
}
</script>
</body>
</html>
"""

RECORDINGS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Recordings</title>
<style>
  :root{--bg:#0a0e1a;--panel:#111827;--border:#1e293b;--cyan:#00e5ff;--teal:#00bfa5;--text:#e2e8f0;--muted:#64748b;--red:#ff1744;}
  *{margin:0;padding:0;box-sizing:border-box;}
  body{background:var(--bg);color:var(--text);font-family:'Barlow',sans-serif;}
  header{background:linear-gradient(90deg,#0f172a,#1e293b);border-bottom:1px solid var(--border);padding:14px 28px;display:flex;justify-content:space-between;align-items:center;}
  header h1{font-size:1.1rem;font-weight:700;color:var(--teal);text-transform:uppercase;}
  .nav-link{color:var(--cyan);text-decoration:none;font-size:.8rem;border:1px solid var(--cyan);padding:6px 14px;border-radius:6px;margin-left:8px;}
  .wrap{display:grid;grid-template-columns:300px 1fr;gap:16px;padding:16px 28px;}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:20px;}
  .card-title{font-size:.7rem;font-weight:600;text-transform:uppercase;color:var(--muted);margin-bottom:14px;border-bottom:1px solid var(--border);padding-bottom:8px;}
  .clip{display:block;width:100%;text-align:left;background:#0d1424;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:10px 12px;margin-bottom:8px;cursor:pointer;font-family:monospace;font-size:.78rem;}
  .clip:hover,.clip.active{border-color:var(--cyan);color:var(--cyan);}
  video{width:100%;border-radius:8px;background:#000;}
  .stopped-banner{background:#2a0010;color:var(--red);border:1px solid var(--red);padding:8px 16px;border-radius:6px;font-family:monospace;font-size:.78rem;margin-bottom:12px;display:none;}
</style>
</head>
<body>
<header>
  <h1>Recordings</h1>
  <div>
    <a class="nav-link" href="/">Dashboard</a>
    <a class="nav-link" href="/analytics">Analytics</a>
    <a class="nav-link" href="/report">Report</a>
  </div>
</header>
<div style="padding:0 28px;margin-top:12px;">
  <div id="stopped-banner" class="stopped-banner">Pipeline stopped - viewing saved recordings</div>
</div>
<div class="wrap">
  <div class="card"><div class="card-title">Clips</div><div id="clip-list"><div style="color:var(--muted)">Loading...</div></div></div>
  <div class="card"><div class="card-title" id="now-playing">Select a clip</div><video id="player" controls></video></div>
</div>
<script>
fetch('/api/pipeline_status').then(r=>r.json()).then(d=>{
  if(!d.running) document.getElementById('stopped-banner').style.display='block';
});
const player=document.getElementById('player'),listEl=document.getElementById('clip-list'),nowEl=document.getElementById('now-playing');
function fmt(n){const m=n.match(/session_(\\d{4})(\\d{2})(\\d{2})_(\\d{2})(\\d{2})(\\d{2})/);return m?m[1]+'-'+m[2]+'-'+m[3]+' '+m[4]+':'+m[5]+':'+m[6]:n;}
function load(){fetch('/api/recordings').then(r=>r.json()).then(files=>{
  if(!files.length){listEl.innerHTML='<div style="color:var(--muted)">No recordings yet.</div>';return;}
  listEl.innerHTML=files.map(f=>'<button class="clip" data-f="'+f+'">'+fmt(f)+'</button>').join('');
  document.querySelectorAll('.clip').forEach(btn=>{btn.onclick=()=>{
    document.querySelectorAll('.clip').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    const f=btn.getAttribute('data-f');player.src='/recordings_file/'+f;player.play();nowEl.textContent='Playing: '+fmt(f);
  };});});}
load();setInterval(load,15000);
</script>
</body>
</html>
"""

REPORT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Shift Report</title>
<style>
  :root{--bg:#0a0e1a;--panel:#111827;--border:#1e293b;--cyan:#00e5ff;--teal:#00bfa5;--text:#e2e8f0;--muted:#64748b;--green:#00e676;--orange:#ff6d00;--red:#ff1744;--yellow:#ffd600;}
  *{margin:0;padding:0;box-sizing:border-box;}
  body{background:var(--bg);color:var(--text);font-family:'Barlow',sans-serif;}
  header{background:linear-gradient(90deg,#0f172a,#1e293b);border-bottom:1px solid var(--border);padding:14px 28px;display:flex;justify-content:space-between;align-items:center;}
  header h1{font-size:1.1rem;font-weight:700;color:var(--teal);text-transform:uppercase;}
  .nav-link{color:var(--cyan);text-decoration:none;font-size:.8rem;border:1px solid var(--cyan);padding:6px 14px;border-radius:6px;margin-left:8px;}
  .wrap{padding:20px 28px;}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:20px;margin-bottom:16px;}
  .card-title{font-size:.7rem;font-weight:600;text-transform:uppercase;color:var(--muted);margin-bottom:14px;border-bottom:1px solid var(--border);padding-bottom:8px;}
  .shift-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;}
  .shift-card{background:#0d1424;border-radius:8px;padding:16px;}
  .shift-morning{border:2px solid #a3e635;}
  .shift-evening{border:2px solid var(--orange);}
  .shift-night{border:2px solid var(--cyan);}
  .shift-title{font-size:.9rem;font-weight:700;margin-bottom:12px;}
  .st-m{color:#a3e635;}.st-e{color:var(--orange);}.st-n{color:var(--cyan);}
  .gauge-section{margin-top:10px;border-top:1px solid var(--border);padding-top:10px;}
  .gauge-section-title{font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;}
  .analog-title{color:var(--cyan);}
  .digital-title{color:#a3e635;}
  .kpi{margin:6px 0;}
  .kpi-label{font-size:.62rem;color:var(--muted);text-transform:uppercase;}
  .kpi-val{font-family:'Share Tech Mono',monospace;font-size:1rem;font-weight:700;color:var(--cyan);}
  .kpi-val.dig{color:#a3e635;}
</style>
</head>
<body>
<header>
  <h1>Shift-wise Report</h1>
  <div>
    <a class="nav-link" href="/">Dashboard</a>
    <a class="nav-link" href="/analytics">Analytics</a>
    <a class="nav-link" href="/recordings">Recordings</a>
  </div>
</header>
<div class="wrap">
  <div class="card">
    <div class="card-title">Current Session - Shift Summary</div>
    <div class="shift-grid" id="shift-grid"><div style="color:var(--muted);">Loading...</div></div>
  </div>
</div>
<script>
fetch('/api/shift_report').then(r=>r.json()).then(data=>{
  const shifts=[['Morning','shift-morning','st-m'],['Evening','shift-evening','st-e'],['Night','shift-night','st-n']];
  document.getElementById('shift-grid').innerHTML=shifts.map(([s,sc,tc])=>{
    const d=data[s]||{analog:{total:0,alerts:0,avg_voltage:0,peak_voltage:0},digital:{total:0,alerts:0,avg_voltage:0,peak_voltage:0}};
    const a=d.analog||{total:0,alerts:0,avg_voltage:0,peak_voltage:0};
    const dg=d.digital||{total:0,alerts:0,avg_voltage:0,peak_voltage:0};
    return '<div class="shift-card '+sc+'">'+
    '<div class="shift-title '+tc+'">'+s+' Shift</div>'+
    '<div class="gauge-section">'+
    '<div class="gauge-section-title analog-title">Analog Gauge</div>'+
    '<div class="kpi"><div class="kpi-label">Total Readings</div><div class="kpi-val">'+a.total+'</div></div>'+
    '<div class="kpi"><div class="kpi-label">Total Alerts</div><div class="kpi-val">'+a.alerts+'</div></div>'+
    '<div class="kpi"><div class="kpi-label">Avg Voltage</div><div class="kpi-val">'+(a.avg_voltage||0).toFixed(1)+'V</div></div>'+
    '<div class="kpi"><div class="kpi-label">Peak Voltage</div><div class="kpi-val">'+(a.peak_voltage||0).toFixed(1)+'V</div></div>'+
    '</div>'+
    '<div class="gauge-section">'+
    '<div class="gauge-section-title digital-title">Digital Gauge</div>'+
    '<div class="kpi"><div class="kpi-label">Total Readings</div><div class="kpi-val dig">'+dg.total+'</div></div>'+
    '<div class="kpi"><div class="kpi-label">Total Alerts</div><div class="kpi-val dig">'+dg.alerts+'</div></div>'+
    '<div class="kpi"><div class="kpi-label">Avg Voltage</div><div class="kpi-val dig">'+(dg.avg_voltage||0).toFixed(1)+'V</div></div>'+
    '<div class="kpi"><div class="kpi-label">Peak Voltage</div><div class="kpi-val dig">'+(dg.peak_voltage||0).toFixed(1)+'V</div></div>'+
    '</div>'+
    '</div>';
  }).join('');
});
</script>
</body>
</html>
"""

MAINTENANCE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Maintenance Log</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@400;600;700;900&display=swap');
  :root{--bg:#0a0e1a;--panel:#111827;--border:#1e293b;--green:#00e676;--yellow:#ffd600;--orange:#ff6d00;--red:#ff1744;--cyan:#00e5ff;--teal:#00bfa5;--text:#e2e8f0;--muted:#64748b;}
  *{margin:0;padding:0;box-sizing:border-box;}
  body{background:var(--bg);color:var(--text);font-family:'Barlow',sans-serif;}
  header{background:linear-gradient(90deg,#0f172a,#1e293b);border-bottom:1px solid var(--border);padding:14px 28px;display:flex;justify-content:space-between;align-items:center;}
  header h1{font-size:1.1rem;font-weight:700;color:var(--teal);text-transform:uppercase;}
  .nav-link{color:var(--cyan);text-decoration:none;font-size:.8rem;border:1px solid var(--cyan);padding:6px 14px;border-radius:6px;margin-left:8px;}
  .wrap{padding:24px 28px;max-width:1100px;margin:0 auto;}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:24px;margin-bottom:20px;}
  .card-title{font-size:.7rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:18px;border-bottom:1px solid var(--border);padding-bottom:10px;}
  .form-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;}
  .form-group{display:flex;flex-direction:column;gap:6px;}
  .form-group.full{grid-column:1/3;}
  label{font-size:.7rem;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);}
  input,select,textarea{background:#0d1424;border:1px solid var(--border);color:var(--text);padding:10px 12px;border-radius:6px;font-family:'Barlow',sans-serif;font-size:.88rem;width:100%;outline:none;transition:border .2s;}
  input:focus,select:focus,textarea:focus{border-color:var(--teal);}
  textarea{resize:vertical;min-height:80px;}
  select option{background:#0d1424;}
  .btn-submit{background:var(--teal);color:#000;border:none;padding:12px 32px;border-radius:6px;font-weight:700;font-size:.95rem;cursor:pointer;margin-top:8px;letter-spacing:.04em;}
  .btn-submit:hover{background:#00d4a0;}
  .success-msg{background:#003020;border:1px solid var(--green);color:var(--green);padding:10px 16px;border-radius:6px;font-size:.85rem;margin-top:12px;display:none;}
  table{width:100%;border-collapse:collapse;font-size:.75rem;}
  th{text-align:left;padding:8px 10px;color:var(--muted);border-bottom:1px solid var(--border);font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;}
  td{padding:7px 10px;border-bottom:1px solid #1a2030;vertical-align:top;}
  .badge-l1{background:#2a2200;color:var(--yellow);padding:2px 7px;border-radius:3px;font-size:.65rem;font-weight:700;}
  .badge-l2{background:#2a1500;color:var(--orange);padding:2px 7px;border-radius:3px;font-size:.65rem;font-weight:700;}
  .badge-l3{background:#2a0010;color:var(--red);padding:2px 7px;border-radius:3px;font-size:.65rem;font-weight:700;}
  .badge-l4{background:#0a1a2a;color:var(--cyan);padding:2px 7px;border-radius:3px;font-size:.65rem;font-weight:700;border:1px solid var(--cyan);}
  .empty-msg{text-align:center;color:var(--muted);padding:28px;font-size:.85rem;}
  .info-strip{background:#0d1424;border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin-bottom:18px;font-size:.8rem;color:var(--muted);line-height:1.6;}
  .info-strip span{color:var(--cyan);}
</style>
</head>
<body>
<header>
  <h1>&#128295; Maintenance Log</h1>
  <div>
    <a class="nav-link" href="/">Dashboard</a>
    <a class="nav-link" href="/analytics">Analytics</a>
    <a class="nav-link" href="/report">Report</a>
    <a class="nav-link" href="/recordings">Recordings</a>
  </div>
</header>
<div class="wrap">

  <!-- INFO STRIP: latest alert context -->
  <div class="info-strip" id="alert-context">
    Loading current system state...
  </div>

  <!-- LOG FORM -->
  <div class="card">
    <div class="card-title">&#128221; Log Maintenance Action</div>
    <form id="maint-form">
      <div class="form-grid">
        <div class="form-group">
          <label>Gauge ID</label>
          <input type="text" id="f-gauge-id" value="GAUGE-001" required>
        </div>
        <div class="form-group">
          <label>Location</label>
          <input type="text" id="f-location" value="Control Room" required>
        </div>
        <div class="form-group">
          <label>Alert Level Responded To</label>
          <select id="f-alert-level" required>
            <option value="">-- Select --</option>
            <option value="L1 - Pre-Warning">L1 - Pre-Warning</option>
            <option value="L2 - Warning">L2 - Warning</option>
            <option value="L3 - Critical">L3 - Critical</option>
            <option value="L4 - Anomaly">L4 - Anomaly (Physical Defect)</option>
          </select>
        </div>
        <div class="form-group">
          <label>Technician Name</label>
          <input type="text" id="f-tech-name" placeholder="e.g. Engr. Ahmed Khan" required>
        </div>
        <div class="form-group full">
          <label>Action Taken</label>
          <input type="text" id="f-action" placeholder="e.g. Relay tripped and reset, Engineer called, Valve shut off" required>
        </div>
        <div class="form-group full">
          <label>Notes (optional)</label>
          <textarea id="f-notes" placeholder="Additional observations, follow-up required, parts replaced, etc."></textarea>
        </div>
      </div>
      <button type="submit" class="btn-submit">&#10003; Submit Log Entry</button>
      <div class="success-msg" id="success-msg">&#10003; Maintenance entry logged successfully and saved to CSV audit trail.</div>
    </form>
  </div>

  <!-- HISTORY TABLE -->
  <div class="card">
    <div class="card-title">&#128196; Maintenance History — Audit Trail</div>
    <div id="history-wrap"><div class="empty-msg">Loading...</div></div>
  </div>

</div>
<script>
// Pre-fill from current alert state
fetch('/api/state').then(r=>r.json()).then(d=>{
  const strip=document.getElementById('alert-context');
  const lvl=d.alert_level||0;
  const lvlText=['NORMAL','L1 PRE-WARNING','L2 WARNING','L3 CRITICAL','L4 ANOMALY'][lvl]||'NORMAL';
  const color=['#00e676','#ffd600','#ff6d00','#ff1744','#00e5ff'][lvl]||'#00e676';
  strip.innerHTML=`Current System State: <span style="color:${color};font-weight:700;">${lvlText}</span> &nbsp;|&nbsp; Analog: <span>${(d.analog_reading||0).toFixed(1)}V</span> &nbsp;|&nbsp; Digital: <span>${d.digital_reading_str||'N/A'}</span> &nbsp;|&nbsp; Shift: <span>${d.timestamp||''}</span>`;
  // Auto-select alert level in form
  if(lvl>=1){
    const opts=['','L1 - Pre-Warning','L2 - Warning','L3 - Critical','L4 - Anomaly'];
    document.getElementById('f-alert-level').value=opts[lvl]||'';
  }
  document.getElementById('f-gauge-id').value=d.gauge_id||'GAUGE-001';
  document.getElementById('f-location').value=d.gauge_location||'Control Room';
});

function loadHistory(){
  fetch('/api/maintenance').then(r=>r.json()).then(entries=>{
    const wrap=document.getElementById('history-wrap');
    if(!entries.length){wrap.innerHTML='<div class="empty-msg">No maintenance entries yet. Entries appear here after submission.</div>';return;}
    const badgeClass=(lvl)=>{
      if(lvl.includes('L1'))return'badge-l1';
      if(lvl.includes('L2'))return'badge-l2';
      if(lvl.includes('L3'))return'badge-l3';
      if(lvl.includes('L4'))return'badge-l4';
      return'badge-l1';
    };
    wrap.innerHTML='<table><thead><tr><th>Timestamp</th><th>Gauge ID</th><th>Location</th><th>Alert Level</th><th>Technician</th><th>Action Taken</th><th>Notes</th></tr></thead><tbody>'+
    entries.slice().reverse().map(e=>`<tr>
      <td style="font-family:'Share Tech Mono',monospace;font-size:.68rem;white-space:nowrap">${e.timestamp}</td>
      <td style="font-family:'Share Tech Mono',monospace;color:var(--cyan)">${e.gauge_id}</td>
      <td style="color:var(--muted)">${e.location}</td>
      <td><span class="${badgeClass(e.alert_level)}">${e.alert_level}</span></td>
      <td style="color:var(--teal)">${e.technician}</td>
      <td>${e.action}</td>
      <td style="color:var(--muted);font-size:.7rem">${e.notes||'-'}</td>
    </tr>`).join('')+'</tbody></table>';
  });
}
loadHistory();

document.getElementById('maint-form').addEventListener('submit',function(e){
  e.preventDefault();
  const payload={
    gauge_id: document.getElementById('f-gauge-id').value.trim(),
    location: document.getElementById('f-location').value.trim(),
    alert_level: document.getElementById('f-alert-level').value,
    technician: document.getElementById('f-tech-name').value.trim(),
    action: document.getElementById('f-action').value.trim(),
    notes: document.getElementById('f-notes').value.trim(),
  };
  fetch('/api/maintenance',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})
  .then(r=>r.json()).then(d=>{
    if(d.status==='ok'){
      const msg=document.getElementById('success-msg');
      msg.style.display='block';
      setTimeout(()=>msg.style.display='none',4000);
      document.getElementById('f-tech-name').value='';
      document.getElementById('f-action').value='';
      document.getElementById('f-notes').value='';
      loadHistory();
    }
  });
});
</script>
</body>
</html>
"""

# ================================================================
#  FLASK ROUTES
# ================================================================

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML,
        threshold=VOLTAGE_THRESHOLD,
        pre_warn=round(VOLTAGE_THRESHOLD*PRE_WARNING_RATIO,1),
        critical=round(VOLTAGE_THRESHOLD*CRITICAL_RATIO,1),
        gauge_id=GAUGE_ID,
        gauge_location=GAUGE_LOCATION)

@app.route('/analytics')
def analytics_page():
    return render_template_string(ANALYTICS_HTML,
        threshold=VOLTAGE_THRESHOLD,
        pre_warn=round(VOLTAGE_THRESHOLD*PRE_WARNING_RATIO,1),
        critical=round(VOLTAGE_THRESHOLD*CRITICAL_RATIO,1))

@app.route('/recordings')
def recordings_page():
    return render_template_string(RECORDINGS_HTML)

@app.route('/report')
def report_page():
    return render_template_string(REPORT_HTML)

@app.route('/maintenance')
def maintenance_page():
    return render_template_string(MAINTENANCE_HTML)

@app.route('/api/maintenance', methods=['GET','POST'])
def api_maintenance():
    if request.method == 'POST':
        data = request.get_json()
        # Init CSV if new
        write_header = not os.path.exists(MAINTENANCE_LOG)
        with open(MAINTENANCE_LOG, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['timestamp','gauge_id','location','alert_level',
                            'technician','action','notes','shift'])
            w.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data.get('gauge_id',''),
                data.get('location',''),
                data.get('alert_level',''),
                data.get('technician',''),
                data.get('action',''),
                data.get('notes',''),
                get_shift()
            ])
        print(f"[Maintenance] Logged: {data.get('technician','')} — {data.get('action','')}")
        return jsonify({"status": "ok"})
    else:
        # GET — return all entries
        entries = []
        if os.path.exists(MAINTENANCE_LOG):
            with open(MAINTENANCE_LOG, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entries.append(row)
        return jsonify(entries)

@app.route('/snapshot')
def snapshot():
    with frame_lock:
        jpg = current_jpeg
    if jpg is None:
        return Response(status=503)
    return Response(jpg, mimetype='image/jpeg', headers={'Cache-Control':'no-store'})

@app.route('/api/state')
def api_state():
    return jsonify(state)

@app.route('/api/pipeline_status')
def api_pipeline_status():
    return jsonify({"running": state.get("pipeline_running", False)})

@app.route('/api/set_industry', methods=['POST'])
def api_set_industry():
    global VOLTAGE_THRESHOLD, PRE_WARNING_RATIO, CRITICAL_RATIO, current_industry
    data     = request.get_json()
    industry = data.get('industry', 'Custom')
    if industry in INDUSTRY_PRESETS:
        preset            = INDUSTRY_PRESETS[industry]
        VOLTAGE_THRESHOLD = preset['threshold']
        PRE_WARNING_RATIO = preset['pre_warn_ratio']
        CRITICAL_RATIO    = preset['critical_ratio']
        current_industry  = industry
        print(f"[Industry] Switched to {industry} — Threshold: {VOLTAGE_THRESHOLD}")
        return jsonify({
            "name":     industry,
            "threshold": VOLTAGE_THRESHOLD,
            "pre_warn": round(VOLTAGE_THRESHOLD * PRE_WARNING_RATIO, 1),
            "critical": round(VOLTAGE_THRESHOLD * CRITICAL_RATIO, 1),
        })
    return jsonify({"error": "Unknown industry"}), 400

@app.route('/api/recordings')
def api_recordings():
    files = sorted([f for f in os.listdir(RECORD_DIR)
                    if f.startswith("session_") and f.endswith(".mp4")], reverse=True)
    return jsonify(files)

@app.route('/recordings_file/<path:filename>')
def recordings_file(filename):
    if not (filename.startswith("session_") and filename.endswith(".mp4")):
        abort(404)
    return send_from_directory(RECORD_DIR, filename, mimetype="video/mp4")

@app.route('/api/shift_report')
def api_shift_report():
    report = {
        "Morning": {"analog":{"total":0,"alerts":0,"voltages":[]}, "digital":{"total":0,"alerts":0,"voltages":[]}},
        "Evening": {"analog":{"total":0,"alerts":0,"voltages":[]}, "digital":{"total":0,"alerts":0,"voltages":[]}},
        "Night":   {"analog":{"total":0,"alerts":0,"voltages":[]}, "digital":{"total":0,"alerts":0,"voltages":[]}},
    }
    log_file = os.path.join(LOG_DIR, f"session_{SESSION_ID}.csv")
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts    = datetime.strptime(row['timestamp'][:19], "%Y-%m-%d %H:%M:%S")
                    h     = ts.hour
                    shift = "Morning" if 6<=h<14 else ("Evening" if 14<=h<22 else "Night")
                    av    = float(row.get('analog_reading', 0) or 0)
                    dr    = row.get('digital_reading', 'N/A')
                    al    = int(row.get('alert_level', 0) or 0)
                    report[shift]['analog']['total'] += 1
                    report[shift]['analog']['voltages'].append(av)
                    if al > 0: report[shift]['analog']['alerts'] += 1
                    try:
                        dv = float(dr)
                        if dv > 0:
                            report[shift]['digital']['total'] += 1
                            report[shift]['digital']['voltages'].append(dv)
                            if al > 0: report[shift]['digital']['alerts'] += 1
                    except: pass
                except: continue
    result = {}
    for shift, d in report.items():
        avs = d['analog']['voltages']
        dvs = d['digital']['voltages']
        result[shift] = {
            "analog": {
                "total":        d['analog']['total'],
                "alerts":       d['analog']['alerts'],
                "avg_voltage":  sum(avs)/len(avs) if avs else 0,
                "peak_voltage": max(avs) if avs else 0,
            },
            "digital": {
                "total":        d['digital']['total'],
                "alerts":       d['digital']['alerts'],
                "avg_voltage":  sum(dvs)/len(dvs) if dvs else 0,
                "peak_voltage": max(dvs) if dvs else 0,
            }
        }
    return jsonify(result)

@app.route('/api/log_sessions')
def api_log_sessions():
    files = sorted([f for f in os.listdir(LOG_DIR)
                    if f.startswith("session_") and f.endswith(".csv")], reverse=True)
    result = []
    for f in files:
        m = __import__('re').match(r'session_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.csv', f)
        label = f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}:{m.group(6)}" if m else f
        result.append({"file": f, "label": label})
    return jsonify(result)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json(); files = data.get('files', []); rows = []
    for fname in files:
        fpath = os.path.join(LOG_DIR, fname)
        if not os.path.exists(fpath): continue
        with open(fpath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({'timestamp': row.get('timestamp',''),
                                 'analog': float(row.get('analog_reading', 0) or 0),
                                 'alert_level': int(row.get('alert_level', 0) or 0)})
                except: continue
    if not rows: return jsonify({"error": "No data", "total": 0})
    rows.sort(key=lambda r: r['timestamp'])
    analog_vals  = [r['analog'] for r in rows]
    timestamps   = [r['timestamp'] for r in rows]
    alert_levels = [r['alert_level'] for r in rows]
    point_types  = ['normal'] * len(rows); anomalies = []
    prev_val = None; stuck_count_local = 0; stuck_val = None
    erratic_count = stuck_count = out_count = 0
    for i, v in enumerate(analog_vals):
        atype = 'normal'; detail = ''
        if v < 0 or v > 50: atype = 'outofrange'; detail = f"Value {v:.2f}V outside 0-50V"; out_count += 1
        elif prev_val is not None and abs(v-prev_val) > ERRATIC_JUMP_THRESHOLD: atype = 'erratic'; detail = f"Jump of {abs(v-prev_val):.2f}V from previous"; erratic_count += 1
        if v == stuck_val:
            stuck_count_local += 1
            if stuck_count_local >= STUCK_REPEAT_COUNT and atype == 'normal': atype = 'stuck'; detail = f"Same value {v:.2f}V repeated {stuck_count_local} times"; stuck_count += 1
        else: stuck_val = v; stuck_count_local = 1
        if atype != 'normal': point_types[i] = atype; anomalies.append({'time': timestamps[i], 'voltage': v, 'type': atype, 'detail': detail})
        prev_val = v
    l0=sum(1 for a in alert_levels if a==0); l1=sum(1 for a in alert_levels if a==1)
    l2=sum(1 for a in alert_levels if a==2); l3=sum(1 for a in alert_levels if a==3)
    n=len(analog_vals); xs=list(range(n)); mean_x=sum(xs)/n; mean_y=sum(analog_vals)/n
    num=sum((xs[i]-mean_x)*(analog_vals[i]-mean_y) for i in range(n))
    den=sum((xs[i]-mean_x)**2 for i in range(n))
    slope=num/den if den!=0 else 0; intercept=mean_y-slope*mean_x
    regression_line=[slope*x+intercept for x in xs]
    def eta(target):
        if slope<=0: return None
        steps=(target-(slope*(n-1)+intercept))/slope
        return "Already exceeded" if steps<=0 else f"~{steps:.0f} readings away"
    trend_dir="RISING" if slope>0.01 else ("FALLING" if slope<-0.01 else "STABLE")
    pw=VOLTAGE_THRESHOLD*PRE_WARNING_RATIO; cr=VOLTAGE_THRESHOLD*CRITICAL_RATIO
    health=100.0-min(40,(l2+l3*2)*2)-min(20,erratic_count*2)-min(20,out_count*5); health=max(0,health)
    return jsonify({"total":n,"avg_voltage":mean_y,"max_voltage":max(analog_vals),"erratic_count":erratic_count,"stuck_count":stuck_count,"timestamps":timestamps,"analog_readings":analog_vals,"point_types":point_types,"regression_line":regression_line,"anomalies":anomalies[:50],"alert_counts":{"l0":l0,"l1":l1,"l2":l2,"l3":l3},"prognostics":{"trend_direction":trend_dir,"slope":round(slope,4),"eta_prewarn":eta(pw),"eta_alert":eta(VOLTAGE_THRESHOLD),"eta_critical":eta(cr),"health_score":round(health,1)}})

# ================================================================
#  LOAD MODELS
# ================================================================

def load_all_models():
    print("="*55)
    print("  SMART VISION BASED MONITORING SYSTEM")
    print("  Alibaba Cloud AI Hackathon Pakistan 2026")
    print("="*55)
    print("\n[1/5] Loading Analog YOLO...")
    analog_yolo = YOLO(ANALOG_YOLO_MODEL)
    print("[2/5] Loading Digital YOLO...")
    digital_yolo = YOLO(DIGITAL_YOLO_MODEL)
    print("[3/5] Loading MobileNetV2 Classifier...")
    classifier = tf.keras.models.load_model(CLASSIFIER_MODEL)
    with open(MODEL_CONFIG_FILE) as f: model_cfg = json.load(f)
    class_names = model_cfg['class_names']
    print(f"      Classes: {class_names}")
    print("[4/5] Loading Analog CAE...")
    analog_cae = tf.lite.Interpreter(model_path=ANALOG_CAE_MODEL)
    analog_cae.allocate_tensors()
    with open(ANALOG_THRESHOLD_F) as f: thr_data = json.load(f)
    analog_thr = thr_data['threshold'] * 1.5
    print("[5/5] Loading Digital CAE...")
    digital_cae = tf.lite.Interpreter(model_path=DIGITAL_CAE_MODEL)
    digital_cae.allocate_tensors()
    with open(DIGITAL_CAE_CONFIG) as f: dig_cfg = json.load(f)
    digital_thr = dig_cfg.get("anomaly_threshold", 0.025)
    print("[CAL] Loading calibration...")
    with open(CALIBRATION_FILE) as f: cal = json.load(f)
    return (analog_yolo, digital_yolo, classifier, class_names, cal, analog_cae, analog_thr, digital_cae, digital_thr)

# ================================================================
#  ANALOG PIPELINE
# ================================================================

classifier_votes = deque(maxlen=CLASSIFIER_SMOOTH)

def classify_gauge(classifier, class_names, image, dial_bbox):
    try:
        H,W=image.shape[:2]; x1,y1,x2,y2=dial_bbox; pad=20
        crop=image[max(0,y1-pad):min(H,y2+pad),max(0,x1-pad):min(W,x2+pad)]
        if crop.size==0: return "unknown",0.0
        img=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(CLASSIFIER_INPUT_SIZE,CLASSIFIER_INPUT_SIZE))
        img=(img.astype(np.float32)/127.5)-1.0
        preds=classifier.predict(np.expand_dims(img,0),verbose=0)[0]
        idx=int(np.argmax(preds))
        return class_names[idx],float(preds[idx])
    except: return "unknown",0.0

def get_stable_gauge_type(gauge_type,conf):
    classifier_votes.append((gauge_type,conf))
    types=[v[0] for v in classifier_votes]
    most_common=Counter(types).most_common(1)[0][0]
    avg_conf=float(np.mean([v[1] for v in classifier_votes if v[0]==most_common]))
    return most_common,avg_conf

def detect_needle_axis(image,search_bbox,pivot_xy,padding=15):
    x1,y1,x2,y2=search_bbox; H,W=image.shape[:2]
    crop=image[max(0,y1-padding):min(H,y2+padding),max(0,x1-padding):min(W,x2+padding)]
    if crop.size==0: return None
    x1p,y1p=max(0,x1-padding),max(0,y1-padding)
    bright=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).mean()
    r1,r2=(RED_HSV_1_DARK,RED_HSV_2_DARK) if bright<DARK_THRESH else (RED_HSV_1_STD,RED_HSV_2_STD)
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    mask=cv2.bitwise_or(cv2.inRange(hsv,r1[0],r1[1]),cv2.inRange(hsv,r2[0],r2[1]))
    k=np.ones((3,3),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k,iterations=1)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k,iterations=2)
    ys,xs=np.where(mask>0)
    if len(xs)<20: return None
    pts=np.column_stack([(xs+x1p).astype(np.float32),(ys+y1p).astype(np.float32)])
    px,py=pivot_xy
    dists=np.sqrt((pts[:,0]-px)**2+(pts[:,1]-py)**2)
    weights=np.clip(dists/(dists.max()+1e-6),0.1,1.0)
    centroid=np.average(pts,axis=0,weights=weights)
    _,eigvecs=np.linalg.eigh(np.cov((pts-centroid).T))
    direction=eigvecs[:,-1]
    if np.dot(np.array([px,py])-centroid,direction)>0: direction=-direction
    return direction,centroid,len(pts),"dark" if bright<DARK_THRESH else "standard"

def read_analog_gauge(analog_yolo,image,cal):
    results=analog_yolo(image,verbose=False,conf=ANALOG_CONF)[0]
    detections={}
    if results.boxes is None: return None
    for box in results.boxes:
        cls=int(box.cls.item()); conf=float(box.conf.item())
        x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
        if cls not in detections or conf>detections[cls]['conf']:
            detections[cls]={'bbox':(int(x1),int(y1),int(x2),int(y2)),'conf':conf}
    if CLS_DIAL not in detections: return None
    dx1,dy1,dx2,dy2=detections[CLS_DIAL]['bbox']
    pivot_x=dx1+(dx2-dx1)*cal['pivot_x_pct']
    pivot_y=dy1+(dy2-dy1)*cal['pivot_y_pct']
    sbbox=detections[CLS_NEEDLE]['bbox'] if CLS_NEEDLE in detections else detections[CLS_DIAL]['bbox']
    axis=detect_needle_axis(image,sbbox,(pivot_x,pivot_y))
    if axis is None: return None
    direction,centroid,num_pixels,lighting=axis
    tip=centroid+direction*100
    angle=float(np.degrees(np.arctan2(-(tip[1]-pivot_y),tip[0]-pivot_x)))
    a0,a50=cal['angle_at_0v'],cal['angle_at_50v']; span=a50-a0
    v=0.0 if abs(span)<1e-6 else cal['v_min']+(angle-a0)/span*(cal['v_max']-cal['v_min'])
    v=float(max(cal['v_min'],min(cal['v_max'],v)))
    return {'voltage':round(v,2),'angle':angle,'pivot':(int(pivot_x),int(pivot_y)),
            'centroid':centroid,'direction':direction,'lighting':lighting,
            'detections':detections,'dial_bbox':detections[CLS_DIAL]['bbox']}

# ================================================================
#  DIGITAL PIPELINE
# ================================================================

def read_digital_gauge(digital_yolo,frame):
    clean=frame.copy()
    results=digital_yolo(frame,conf=DIGITAL_CONF,verbose=False)[0]
    all_dets=[]
    for box in results.boxes:
        cls_id=int(box.cls[0]); cls_name=results.names[cls_id]
        conf=float(box.conf[0]); x1,y1,x2,y2=map(int,box.xyxy[0].tolist())
        cx=(x1+x2)//2; w=x2-x1; h=y2-y1
        if cls_name=="digit" and (w<MIN_W or h<MIN_H): continue
        roi=clean[y1:y2,x1:x2].copy()
        all_dets.append({"class":cls_name,"conf":conf,"bbox":(x1,y1,x2,y2),"roi":roi,"cx":cx})
    gauge_detected=any(d["class"]=="gauge" for d in all_dets)
    if not gauge_detected:
        return "N/A",0.0,0,False
    digits_raw=sorted([d for d in all_dets if d["class"]=="digit"],key=lambda x:x["cx"])
    digits_dets=[]
    for d in digits_raw:
        overlap=False
        for kept in digits_dets:
            # Use bbox overlap instead of center-distance so narrow '1' is not eaten
            d_x1, d_x2 = d["bbox"][0], d["bbox"][2]
            k_x1, k_x2 = kept["bbox"][0], kept["bbox"][2]
            actual_overlap = max(0, min(d_x2, k_x2) - max(d_x1, k_x1))
            d_w = max(d_x2 - d_x1, 1)
            if actual_overlap > 0.4 * d_w:  # >40% pixel overlap = true duplicate
                if d["conf"]>kept["conf"]: digits_dets.remove(kept)
                else: overlap=True
                break
        if not overlap: digits_dets.append(d)
    digits_dets=sorted(digits_dets,key=lambda x:x["cx"])
    decimals=[d for d in all_dets if d["class"]=="decimal"]
    decimal_x=max(decimals,key=lambda x:x["conf"])["cx"] if decimals else None
    reading_str="N/A"; reading_val=0.0
    if digits_dets:
        digits_str=decode_all_digits(digits_dets,display_color=DISPLAY_COLOR)
        reading_str=insert_decimal(digits_str,decimal_x,digits_dets)
        try: reading_val=float(reading_str.replace("?","0"))
        except: reading_val=0.0
    return reading_str,reading_val,len(digits_dets),True

# ================================================================
#  CAE
# ================================================================

def run_cae(interpreter,frame,threshold):
    inp_d=interpreter.get_input_details(); out_d=interpreter.get_output_details()
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(IMG_SIZE,IMG_SIZE)).astype(np.float32)/255.0
    interpreter.set_tensor(inp_d[0]['index'],np.expand_dims(img,0))
    interpreter.invoke()
    recon=interpreter.get_tensor(out_d[0]['index'])[0]
    mse=float(np.mean((img-recon)**2))
    return mse>threshold,mse

# ================================================================
#  ALERT LOGIC
# ================================================================

def get_alert_level(analog_v,digital_v,analog_anom,digital_anom):
    """
    Alert Levels:
      L0 - Normal
      L1 - Pre-Warning (voltage approaching limit)
      L2 - Warning (voltage exceeded threshold)
      L3 - Critical (voltage exceeded critical OR threshold + anomaly)
      L4 - Anomaly Alert (physical gauge defect, voltage still normal)
    """
    pw=VOLTAGE_THRESHOLD*PRE_WARNING_RATIO; cr=VOLTAGE_THRESHOLD*CRITICAL_RATIO
    max_v=max(analog_v,digital_v); any_anom=analog_anom or digital_anom
    if max_v>=cr: return 3,f"LEVEL 3 CRITICAL - {max_v:.2f}V Exceeded {cr:.0f}V"
    if max_v>=VOLTAGE_THRESHOLD and any_anom: return 3,"LEVEL 3 CRITICAL - Threshold + Physical Anomaly"
    if max_v>=VOLTAGE_THRESHOLD: return 2,f"LEVEL 2 WARNING - {max_v:.2f}V Exceeded {VOLTAGE_THRESHOLD:.0f}V"
    if max_v>=pw and any_anom: return 2,"LEVEL 2 WARNING - Approaching Limit + Anomaly Detected"
    if any_anom: return 4,"ANOMALY ALERT - Physical Gauge Defect Detected"
    if max_v>=pw: return 1,f"PRE-WARNING - {max_v:.2f}V Approaching Limit"
    return 0,f"ALL SYSTEMS NORMAL  A:{analog_v:.1f}V  D:{digital_v:.1f}V"

# ================================================================
#  MAIN PIPELINE THREAD
# ================================================================

def pipeline_thread(analog_yolo,digital_yolo,classifier,class_names,cal,
                    analog_cae,analog_thr,digital_cae,digital_thr):
    global current_frame,current_jpeg,state

    src=WEBCAM_ID if MODE=="webcam" else VIDEO_PATH
    cap=cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {src}")
        state["pipeline_running"]=False
        return

    logger       = Logger()
    alert_logger = AlertLogger()
    frame_idx    = 0
    last_log_time = time.time()-30.0
    last_alert_level=-1
    current_gauge="voltage_gauge"; current_conf=0.0

    analog_v=0.0; analog_angle=0.0; analog_anom=False; analog_mse=0.0; analog_det=False
    digital_v=0.0; digital_str="N/A"; digital_anom=False; digital_mse=0.0
    digital_det=False; digits_found=0

    state["pipeline_running"]=True
    print(f"\n[Pipeline] Started - {MODE.upper()}")
    print(f"[Flask] http://localhost:{FLASK_PORT}\n")

    while True:
        ret,frame=cap.read()
        if not ret:
            if MODE=="video":
                cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                continue
            break

        frame_idx+=1

        # ANALOG
        vr=read_analog_gauge(analog_yolo,frame,cal)
        if vr:
            analog_det=True; analog_v=vr['voltage']; analog_angle=vr['angle']
            if CLS_DIAL in vr['detections'] and frame_idx%5==0:
                raw_type,raw_conf=classify_gauge(classifier,class_names,frame,vr['dial_bbox'])
                current_gauge,current_conf=get_stable_gauge_type(raw_type,raw_conf)
            if frame_idx%CAE_INTERVAL==0:
                analog_anom,analog_mse=run_cae(analog_cae,frame,analog_thr)
        else:
            analog_det=False

        # DIGITAL
        digital_str,digital_v,digits_found,digital_det=read_digital_gauge(digital_yolo,frame)
        if frame_idx%CAE_INTERVAL==0:
            digital_anom,digital_mse=run_cae(digital_cae,frame,digital_thr)

        # ALERT
        cl,cmsg=get_alert_level(analog_v,digital_v,analog_anom,digital_anom)

        # STATS
        if analog_det: analog_stats.add(analog_v,cl)
        if digital_det and digital_v>0: digital_stats.add(digital_v,cl)

        # AUTO LOG every 30 seconds
        if (time.time()-last_log_time)>=30.0:
            last_log_time=time.time()
            logger.log(analog_v,current_gauge,analog_anom,analog_mse,
                       digital_str,digital_anom,digital_mse,cl,cmsg)

        # ALERT LOG
        if cl!=last_alert_level and cl>=1:
            alert_logger.log(cl,analog_v,digital_str,cmsg)

        # DRAW
        H,W=frame.shape[:2]
        colour=GREEN if cl==0 else (YELLOW if cl==1 else (ORANGE if cl==2 else RED))
        cv2.rectangle(frame,(0,0),(W,58),(20,20,30),-1)
        cv2.putText(frame,cmsg,(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,WHITE,2)
        cv2.putText(frame,f"A:{analog_v:.2f}V  D:{digital_str}V  Shift:{get_shift()}",
                    (10,48),cv2.FONT_HERSHEY_SIMPLEX,0.55,CYAN,1)
        if vr and analog_det:
            det=vr['detections']
            if CLS_DIAL in det: cv2.rectangle(frame,det[CLS_DIAL]['bbox'][:2],det[CLS_DIAL]['bbox'][2:],colour,2)
            le=vr['centroid']+vr['direction']*200
            cv2.line(frame,tuple(vr['centroid'].astype(int)),tuple(le.astype(int)),CYAN,2)
            cv2.drawMarker(frame,vr['pivot'],RED,cv2.MARKER_CROSS,20,2)
        cv2.putText(frame,datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    (10,H-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,WHITE,1)

        # STATE
        gauge_info=GAUGE_INFO.get(current_gauge,GAUGE_INFO["unknown"])
        ts=datetime.now().strftime("%H:%M:%S")
        state.update({
            "analog_reading":      analog_v,
            "analog_angle":        analog_angle,
            "analog_gauge_type":   current_gauge,
            "analog_gauge_unit":   gauge_info["unit"],
            "analog_gauge_label":  gauge_info["label"],
            "analog_gauge_conf":   round(current_conf,3),
            "analog_anomaly":      "DEFECTIVE" if analog_anom else "NORMAL",
            "analog_mse":          round(analog_mse,5),
            "analog_detected":     analog_det,
            "digital_reading":     digital_v,
            "digital_reading_str": digital_str,
            "digital_anomaly":     "DEFECTIVE" if digital_anom else "NORMAL",
            "digital_mse":         round(digital_mse,5),
            "digital_detected":    digital_det,
            "digits_found":        digits_found,
            "alert_level":         cl,
            "alert_message":       cmsg,
            "timestamp":           ts,
            "frame_count":         frame_idx,
            "log_count":           logger.count,
            "pipeline_running":    True,
            "analog_stats":        analog_stats.summary(),
            "digital_stats":       digital_stats.summary(),
        })
        socketio.emit('state_update',state)

        if cl!=last_alert_level and cl>=1:
            gauge_type_str="Both" if (analog_det and digital_det) else ("Analog" if analog_det else "Digital")
            entry={"time":ts,"level":cl,"gauge_type":gauge_type_str,
                   "value":f"A:{analog_v:.1f} D:{digital_str}","message":cmsg}
            alert_history.appendleft(entry)
            socketio.emit('new_alert',entry)
            if cl==4:
                print(f"[ANOMALY] Physical gauge defect detected!")
            else:
                print(f"[Alert] L{cl} - {cmsg}")
        last_alert_level=cl

        with frame_lock:
            current_frame=frame
            ok,buf=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,70])
            if ok: current_jpeg=buf.tobytes()

        if frame_idx%RECORD_EVERY_N==0:
            recorder.write(frame)

        if frame_idx%60==0:
            print(f"  Frame {frame_idx:5d} | A:{analog_v:.2f}V | D:{digital_str}V | L{cl} | {get_shift()}")

    cap.release()
    recorder.finalize()
    state["pipeline_running"]=False
    socketio.emit('state_update',state)
    print("[Pipeline] Done.")

# ================================================================
#  ENTRY POINT
# ================================================================

if __name__=="__main__":
    recover_orphan_recordings()
    print("\n"+"="*55)
    print("  SMART VISION BASED MONITORING SYSTEM")
    print("  FOR INDUSTRIAL GAUGES")
    print("  Alibaba Cloud AI Hackathon Pakistan 2026")
    print("  AI for Pakistan's Future")
    print("  Ifrah Gohar | IST Islamabad")
    print("="*55)
    print(f"\nVideo     : {VIDEO_PATH}")
    print(f"Thresholds: Pre={VOLTAGE_THRESHOLD*PRE_WARNING_RATIO:.0f}V  Alert={VOLTAGE_THRESHOLD:.0f}V  Critical={VOLTAGE_THRESHOLD*CRITICAL_RATIO:.0f}V")
    print(f"Auto Log  : Every 30 seconds (first log immediate)")
    print(f"Alert Log : Separate alerts CSV\n")

    (analog_yolo,digital_yolo,classifier,class_names,cal,
     analog_cae,analog_thr,digital_cae,digital_thr)=load_all_models()

    t=threading.Thread(
        target=pipeline_thread,
        args=(analog_yolo,digital_yolo,classifier,class_names,cal,
              analog_cae,analog_thr,digital_cae,digital_thr),
        daemon=True)
    t.start()

    print(f"\n[Flask] http://localhost:{FLASK_PORT}")
    print(f"[Flask] Analytics:  http://localhost:{FLASK_PORT}/analytics")
    print(f"[Flask] Recordings: http://localhost:{FLASK_PORT}/recordings")
    print(f"[Flask] Report:     http://localhost:{FLASK_PORT}/report\n")

    try:
        socketio.run(app,host=FLASK_HOST,port=FLASK_PORT,
                     debug=False,allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping...")
    finally:
        recorder.finalize()
        state["pipeline_running"]=False
        print("[Shutdown] Done.")