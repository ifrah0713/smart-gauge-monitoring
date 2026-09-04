# A Smart Vision Based Monitoring System for Industrial Gauges

**Alibaba Cloud AI Hackathon Pakistan 2026 — AI for Pakistan's Future**
**Team:** Ifrah Gohar
**Project ID:** P00212

---

## Background & Problem Statement

During formal meetings between EMC Solutions and several Chambers of Commerce across Pakistan, industrial representatives raised a concern common across factories, power facilities, and processing plants:

> *"There are control rooms that monitor the performance of various machines. Some machines have analog displays and some have digital displays. It is not possible to tap their signals electronically. Many of these processes run on 24-hour cycles and require a person to be present at all times."*

Many industrial plants in the world — and especially SMEs in developing countries like Pakistan — still use analog gauges and stand-alone digital displays without electronic output. It is not possible to directly integrate these instruments with modern monitoring systems, making them heavily reliant on manual observation by human operators. This method is prone to fatigue, inattention, and monitoring gaps that can lead to delayed detection of abnormal operating conditions.

**Core problems this system solves:**

- Legacy gauges have no electronic output — impossible to connect to modern systems without physical modification
- Human operators get tired, distracted, and shift changes create dangerous monitoring gaps
- Built-in machine alarms use fixed manufacturer-defined thresholds — operators cannot set custom warning levels for their specific process without physically modifying the machine
- Commercial vision-based alternatives (Honeywell, Siemens, Aveva) cost USD 50,000–500,000 per installation — completely inaccessible to Pakistani SMEs
- No affordable, locally relevant implementation exists

Previous research has studied analog gauge reading, digital display reading, anomaly detection, and edge AI systems — but in isolation. An integrated system combining all of these capabilities on affordable edge hardware has not been explored.

---

## Project Objective

This project addresses these challenges by developing an **Edge-AI vision-based monitoring system** for industrial gauges. The objective is to automatically read both analog and digital gauges in real time using a camera, compare every reading with user-defined safety limits, and generate instant warnings whenever abnormal conditions are detected — with zero physical connection to or modification of the monitored equipment.

---

## Solution Overview

The system uses a camera as its only point of contact with the monitored environment. Positioned in front of existing gauges, it continuously captures images and processes them on a Raspberry Pi using computer vision and AI to:

- Identify gauge type and extract readings in real time
- Compare every reading against user-defined operating limits
- Generate immediate alerts when thresholds are exceeded
- Detect physical gauge faults (damaged display, stuck needle, condensation)
- Log all readings and events automatically
- Provide a live web dashboard for remote monitoring

The architecture combines supervised and unsupervised learning. Supervised models handle gauge detection, type classification, and reading extraction. An unsupervised Convolutional Autoencoder handles anomaly detection — trained on normal gauge appearance, it flags abnormal visual states without needing labelled defect examples.

---

## System Architecture

```
Camera Feed (Pi Camera Module V3 / Webcam)
    |
    v
YOLOv8n Object Detection
    |
    |-- ANALOG GAUGE PATH
    |     |-- MobileNetV2 Classifier (voltage / pressure / temperature / current)
    |     |-- HSV Masking + PCA Needle Axis --> Angle --> Reading
    |     +-- Convolutional Autoencoder (TFLite) --> Anomaly Detection
    |
    +-- DIGITAL GAUGE PATH
          |-- YOLOv8n Digit + Decimal Detection
          |-- Custom 7-Segment Decoder (HSV threshold + segment truth table)
          +-- Convolutional Autoencoder (TFLite) --> Anomaly Detection
    |
    v
4-Level Alert Engine (L0 Normal / L1 Pre-Warn / L2 Warning / L3 Critical / L4 Anomaly)
    |
    |-- Flask + Socket.IO Real-Time Web Dashboard
    |-- CSV Data Logging (every 30 seconds)
    |-- Alert CSV (every new alert event)
    +-- Maintenance Log CSV (technician audit trail)
```

---

## Features

### Core System (Original FYP Build)
| Feature | Description |
|---|---|
| Dual Gauge Reading | Analog + Digital simultaneously from one camera |
| YOLOv8n Detection | Detects dial, needle, centroid, digit, decimal, gauge bounding boxes |
| MobileNetV2 Classifier | Identifies gauge type: voltage, pressure, temperature, current |
| HSV + PCA Needle Reading | Extracts needle angle, maps to calibrated reading value |
| 7-Segment Digit Decoder | Custom HSV-based decoder with segment truth table |
| CAE Anomaly Detection | Convolutional Autoencoder detects physical gauge defects |
| Multi-Level Alert System | Threshold-based alert levels with real-time dashboard update |
| Flask Web Dashboard | Live camera feed, readings, charts, alert history |
| CSV Data Logging | Readings logged every 30 seconds with timestamp and shift |

### New Features Added for Hackathon Build
| Feature | Description |
|---|---|
| **Hackathon Branding** | Updated header — Alibaba Cloud AI Hackathon Pakistan 2026 branding |
| **Pakistan Industry Presets** | Dropdown: WAPDA (220V), SNGPL (50PSI), OGDCL (100PSI), Pakistan Steel (150PSI), Custom — auto-sets thresholds and units |
| **Gauge ID + Location Tags** | Every gauge tagged with unique ID and location (e.g. GAUGE-001, Boiler Room Block-A) — shown in dashboard header |
| **Shift Indicator** | Automatic Morning / Evening / Night shift detection based on system time |
| **Shift-wise Report Page** | `/report` — separate analog and digital stats per shift with avg and peak readings |
| **Maintenance Log Page** | `/maintenance` — technician form: gauge ID, alert level, action taken, notes → saved to CSV audit trail |
| **Separate Alert CSV** | Every alert event logged to dedicated alerts CSV in addition to main data log |
| **L4 Anomaly Alert Level** | Dedicated alert level for physical gauge defect detection separate from voltage alerts |
| **Anomaly Status Label** | Dashboard shows ANOMALY STATUS label with live NORMAL / DEFECTIVE indicator per gauge |
| **Normal + Anomaly Badges** | Threshold row shows: NORMAL / Pre-warn / Alert / Critical / ANOMALY ALERT badges |
| **Pipeline Status Indicator** | PIPELINE RUNNING / PIPELINE STOPPED badge — analytics and recordings remain accessible after pipeline stops |
| **Remote Access via ngrok** | One-click BAT file launches pipeline + ngrok tunnel — generates public URL for remote dashboard access |
| **launch_full_system.bat** | Double-click launcher — starts pipeline and ngrok automatically |
| **Digital Stats on Real Detection Only** | Frequency statistics only counted when gauge is actively detected — no phantom readings |
| **Improved 7-Segment Decoder** | Enhanced narrow-digit detection, overlap filter using pixel overlap instead of centre distance — correctly reads digits including '1' and '3' |

---

## Dashboard Pages

| URL | Page |
|---|---|
| `http://localhost:5000` | Live Dashboard |
| `http://localhost:5000/analytics` | Historical Analytics + Prognostics |
| `http://localhost:5000/recordings` | CCTV Session Recordings |
| `http://localhost:5000/report` | Shift-wise Report |
| `http://localhost:5000/maintenance` | Maintenance Log — Technician Audit Trail |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Hardware** | Raspberry Pi 5, Pi Camera Module V3 |
| **Object Detection** | YOLOv8n (Ultralytics) |
| **Gauge Classification** | MobileNetV2 (TensorFlow / Keras) |
| **Needle Reading** | HSV colour masking + PCA axis detection (OpenCV + NumPy) |
| **Digit Decoding** | Custom 7-segment decoder — HSV thresholding + segment truth table |
| **Anomaly Detection** | Convolutional Autoencoder — TFLite (quantized for edge inference) |
| **Web Dashboard** | Flask + Flask-SocketIO + Chart.js |
| **Remote Access** | ngrok |
| **Data Logging** | CSV — readings, alerts, maintenance |
| **Video Recording** | FFmpeg via subprocess |

---

## Model Files & Assets

> Model files (.pt, .keras, .tflite) and calibration files are not included in this repository due to GitHub file size limits. All files are available on Google Drive:

### [Download All Model Files + Demo Video (Google Drive)](https://drive.google.com/drive/folders/1CMpYtJo0TAJtl5kbrvT0IC235iCu62g6?usp=sharing)

Place all downloaded files in the project root directory (same folder as pipeline_main_software_v2.py).

| File | Description |
|---|---|
| `best_v8_centroid.pt` | YOLOv8n — Analog gauge detector (dial, needle, centroid) |
| `best.pt` | YOLOv8n — Digital gauge detector (digit, decimal, gauge) |
| `gauge_classifier.keras` | MobileNetV2 — Gauge type classifier |
| `cae_best.keras` | CAE — Analog anomaly detection (training weights) |
| `cae_best.tflite` | CAE TFLite — Analog anomaly detection (edge inference) |
| `cae_gauge.tflite` | CAE TFLite — Digital anomaly detection (edge inference) |
| `sd670_calibration_FINAL.json` | Analog needle calibration (angle-to-value mapping) |
| `auto_threshold.json` | CAE anomaly threshold (computed from training data) |
| `cae_config.json` | Digital CAE configuration |
| `model_config.json` | Gauge classifier class names |
| `demo.mp4` | Demo video — both gauges being read simultaneously |

---

## Installation & Setup

### Step 1 — Clone the Repository

```bash
git clone https://github.com/ifrah0713/smart-gauge-monitoring.git
cd smart-gauge-monitoring
```

### Step 2 — Download Model Files

Download all files from the Google Drive link above and place them in the project root folder.

### Step 3 — Install Python Packages

Requires Python 3.10+

```bash
pip install ultralytics tensorflow flask flask-socketio opencv-python numpy
```

Full command:
```bash
pip install ultralytics==8.0.196 tensorflow flask flask-socketio opencv-python numpy
```

### Step 4 — Configure the Pipeline

Open pipeline_main_software_v2.py and edit the top section:

```python
# Input source
MODE           = "video"         # "video" for demo file, "webcam" for live camera
VIDEO_PATH     = "demo.mp4"      # path to demo video (downloaded from Drive)

# Gauge identification
GAUGE_ID       = "GAUGE-001"          # unique identifier
GAUGE_LOCATION = "Control Room"       # installation location

# Alert thresholds
VOLTAGE_THRESHOLD = 25.0    # main alert threshold
PRE_WARNING_RATIO = 0.80    # pre-warning at 80% of threshold
CRITICAL_RATIO    = 1.20    # critical at 120% of threshold
```

### Step 5 — Run

```bash
python pipeline_main_software_v2.py
```

Open browser: http://localhost:5000

---

## Remote Access via ngrok

To share the live dashboard over the internet:

### Install ngrok

```bash
winget install ngrok.ngrok
```

Or download from: https://ngrok.com/download

### Get Free Account and Token

1. Sign up at https://ngrok.com/signup
2. Go to https://dashboard.ngrok.com/get-started/your-authtoken
3. Copy your authtoken

### Add Token

```bash
ngrok config add-authtoken YOUR_TOKEN_HERE
```

### Start Remote Access

Option A — One click (starts pipeline + ngrok together):
```
Double-click launch_full_system.bat
```

Option B — Manual (if pipeline already running):
```bash
ngrok http 5000
```

Output:
```
Forwarding: https://xxxx-xxxx.ngrok-free.app --> http://localhost:5000
```

Share this URL with anyone — they can view the live dashboard from anywhere.

---

## Alert Level System

| Level | Trigger | Dashboard |
|---|---|---|
| L0 Normal | Below pre-warning threshold | Green |
| L1 Pre-Warning | Above 80% of threshold | Yellow |
| L2 Warning | Exceeded alert threshold | Orange |
| L3 Critical | Exceeded critical limit | Red (flashing) |
| L4 Anomaly | Physical gauge defect detected by CAE | Cyan |

---

## Pakistan Industry Presets

Select from dashboard dropdown — thresholds and units auto-update:

| Industry | Threshold | Unit |
|---|---|---|
| Custom | 25.0 | V |
| WAPDA Power | 220.0 | V |
| SNGPL Gas | 50.0 | PSI |
| OGDCL Oil & Gas | 100.0 | PSI |
| Pakistan Steel | 150.0 | PSI |

---

## Maintenance Log Workflow

Real industrial audit trail:
1. L3 Critical alert fires on dashboard
2. Supervisor opens http://localhost:5000/maintenance
3. Form auto-fills current alert level and gauge ID
4. Technician enters name, action taken (e.g. "Relay tripped, engineer called"), notes
5. Submit — saved to pipeline_logs/maintenance_log.csv with timestamp and shift
6. Full history visible in table below form

---

## Performance

| Metric | Value |
|---|---|
| Throughput | ~17.4 FPS (Raspberry Pi 5) |
| Detection | YOLOv8n nano — optimized for edge |
| Anomaly Inference | TFLite quantized CAE |
| Data Log Interval | Every 30 seconds |
| Video Recording | 15 FPS H.264 auto-saved per session |

---

## Note on This Submission

This submission demonstrates a **software prototype** running on a laptop using a pre-recorded demo video (`demo.mp4`). The complete system has been previously deployed on Raspberry Pi 5 hardware with a Pi Camera Module V3 . The software prototype demonstrates all pipeline features, dashboard pages, alert levels, and logging functionality identically to the hardware deployment.

The hardware deployment architecture:
```
Raspberry Pi 5
    |
    |-- Standalone WiFi hotspot "GaugeMonitor" (no internet required)
    |     +-- Any device connects --> http://192.168.4.1:5000
    |
    +-- ngrok tunnel (when internet available)
          +-- Remote access from anywhere --> https://xxxx.ngrok-free.app
```

Edge-First, Cloud-Enhanced: fully operational with zero internet. Cloud connectivity adds remote visibility when available.

---

## Repository Structure

```
smart-gauge-monitoring/
|
|-- pipeline_main_software_v2.py   # Main pipeline — full system
|-- segment_decoder.py             # 7-segment digit decoder
|-- launch_full_system.bat         # One-click: pipeline + ngrok
|-- start_remote_access.bat        # ngrok only (pipeline already running)
|-- .gitignore
+-- README.md
```

---

*Alibaba Cloud AI Hackathon Pakistan 2026 | AI for Pakistan's Future | P00212*
*Ifrah Gohar*
