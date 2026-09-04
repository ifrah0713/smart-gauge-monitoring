"""
segment_decoder.py
==================
Module 1b — 7-Segment Digit Decoder
Works for ANY display color (blue, red, green, white, yellow).
"""

import cv2
import numpy as np
import os

DEBUG_DIR = "debug_output"

# Truth table: (top, top-right, bot-right, bottom, bot-left, top-left, middle)
SEGMENT_MAP = {
    (1,1,1,1,1,1,0): '0',
    (0,1,1,0,0,0,0): '7',
    (1,1,0,1,1,0,1): '2',
    (1,1,1,1,0,0,1): '3',
    (0,1,1,0,0,1,1): '4',
    (1,0,1,1,0,1,1): '5',
    (1,0,1,1,1,1,1): '6',
    (1,1,1,0,0,0,0): '7',
    (1,1,1,1,1,1,1): '8',
    (1,1,1,1,0,1,1): '9',
    (0,1,0,1,0,0,1): '1',
    (0,1,0,0,0,0,1): '1',
    (0,1,0,1,0,0,0): '1',
    (0,0,1,0,0,0,0): '1',
    (1,1,1,1,1,0,1): '2',
    (1,0,0,1,1,1,1): '6',
    (1,1,0,1,1,1,1): '6',
    (1,0,1,0,0,0,0): '7',
    (1,1,0,0,0,0,0): '7',
    (1,1,1,0,0,1,0): '7',
    (0,1,1,1,0,0,1): '3',
    (1,0,1,1,0,0,1): '5',
    (0,1,1,0,1,1,1): '4',
    (1,1,1,1,1,0,0): '0',
    (1,0,1,1,1,0,1): '6',
    (0,1,1,0,0,1,0): '4',
    (1,0,1,1,0,1,0): '5',
    (0,1,1,1,0,1,1): '9',
    (1,1,0,1,0,0,1): '2',
    (1,0,1,0,1,1,1): '6',
}

# Patterns that look like other digits but are actually '1' when digit is narrow
NARROW_ONE_PATTERNS = {
    # Only patterns that CANNOT be '3' (3 needs top+mid+bot+top-right+bot-right)
    (1,1,1,0,0,0,0),  # reads as 7 normally — safe, 3 needs middle
    (0,1,1,0,0,0,0),  # reads as 7 normally — safe
    (1,0,1,0,0,0,0),  # reads as 7 normally — safe
    (1,1,0,0,0,0,0),  # reads as 7 normally — safe
    (0,0,1,1,0,0,0),  # only right+bottom lit
    (0,1,0,0,0,0,0),  # only top-right lit
    (0,0,1,0,0,0,0),  # only bot-right lit
    (1,1,0,1,0,0,0),  # safe — no middle
    (0,1,1,1,0,0,0),  # safe — no middle
    # NOTE: (1,1,1,1,0,0,1) REMOVED — that is '3', not '1'
}

NARROW_WIDTH_THRESHOLD = 35  # pixels — tightened from 80 so '3' is not caught

SEGMENT_REGIONS = {
    'top':       (0.00, 0.15, 0.15, 0.85),
    'top-right': (0.10, 0.50, 0.70, 1.00),
    'bot-right': (0.50, 0.90, 0.70, 1.00),
    'bottom':    (0.85, 1.00, 0.15, 0.85),
    'bot-left':  (0.50, 0.90, 0.00, 0.30),
    'top-left':  (0.10, 0.50, 0.00, 0.30),
    'middle':    (0.42, 0.58, 0.15, 0.85),
}

SEG_ORDER = ['top','top-right','bot-right','bottom',
             'bot-left','top-left','middle']

COLOR_PROFILES = {
    "blue":   {"invert": False},
    "red":    {"invert": False},
    "green":  {"invert": False},
    "white":  {"invert": True},
    "yellow": {"invert": False},
    "auto":   {"invert": False},
}

MIN_DIGIT_WIDTH  = 5   # lowered from 15 so narrow '1' digits are not skipped
MIN_DIGIT_HEIGHT = 30


def preprocess_digit(digit_crop_bgr, display_color="auto"):
    gray = cv2.cvtColor(digit_crop_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    profile = COLOR_PROFILES.get(display_color, COLOR_PROFILES["auto"])
    if profile["invert"]:
        thresh = cv2.bitwise_not(thresh)
    if display_color == "auto":
        white_pixels = np.sum(thresh > 127)
        if white_pixels > thresh.size * 0.55:
            thresh = cv2.bitwise_not(thresh)
    return thresh


def decode_single_digit(digit_crop_bgr, index=0,
                         display_color="auto", debug=True):
    os.makedirs(DEBUG_DIR, exist_ok=True)

    h_orig, w_orig = digit_crop_bgr.shape[:2]
    if w_orig < MIN_DIGIT_WIDTH or h_orig < MIN_DIGIT_HEIGHT:
        print(f"  [SEG] Digit {index+1}: SKIP (too small: {w_orig}x{h_orig})")
        return None

    # ── VERY NARROW DIGIT — force '1' immediately ────────────────────────────
    # A '1' on a 7-segment display is extremely thin.
    # Only force if BOTH: width < 15px AND aspect ratio > 3.0 (very tall & thin)
    # This prevents '3' (wider digit) from being misread as '1'
    aspect_ratio = h_orig / max(w_orig, 1)
    if w_orig < 15 and aspect_ratio > 3.0:
        print(f"  [SEG] Digit {index+1}: FORCE '1' (very narrow w={w_orig}px, aspect={aspect_ratio:.1f})")
        return '1'

    thresh = preprocess_digit(digit_crop_bgr, display_color)

    if debug:
        cv2.imwrite(f"{DEBUG_DIR}/seg_digit_{index}_thresh.jpg", thresh)

    img    = cv2.resize(thresh, (60, 100))
    kernel = np.ones((3, 3), np.uint8)
    img    = cv2.erode(img, kernel, iterations=2)
    h, w   = img.shape

    def get_segments(threshold):
        segs = []
        for name in SEG_ORDER:
            y1p, y2p, x1p, x2p = SEGMENT_REGIONS[name]
            y1 = int(y1p * h); y2 = int(y2p * h)
            x1 = int(x1p * w); x2 = int(x2p * w)
            region = img[y1:y2, x1:x2]
            if region.size == 0:
                segs.append(0)
                continue
            white_ratio = np.sum(region > 127) / region.size
            segs.append(1 if white_ratio > threshold else 0)
        return segs

    digit     = '?'
    seg_tuple = None

    for thresh_val in [0.10, 0.08, 0.15, 0.05, 0.20]:
        segments  = get_segments(thresh_val)
        seg_t     = tuple(segments)

        # ── NARROW DIGIT OVERRIDE — runs BEFORE map lookup ───────────────────
        # If the digit is physically narrow AND matches a known narrow-1 pattern
        # it must be a '1' regardless of what the map says
        if w_orig < NARROW_WIDTH_THRESHOLD and seg_t in NARROW_ONE_PATTERNS:
            digit     = '1'
            seg_tuple = seg_t
            print(f"  [SEG] Narrow-1 override: {seg_t} → '1' (w={w_orig}px)")
            break

        # ── MAP LOOKUP ───────────────────────────────────────────────────────
        candidate = SEGMENT_MAP.get(seg_t, '?')
        if candidate != '?':
            digit     = candidate
            seg_tuple = seg_t
            break

        # ── Special case: only right segments lit = '1' ──────────────────────
        if segments[1] == 1 and segments[2] == 1 and \
           sum([segments[0], segments[3], segments[4],
                segments[5], segments[6]]) == 0:
            digit     = '1'
            seg_tuple = seg_t
            break

        # ── Aspect ratio fallback: very tall & narrow = '1' ─────────────────
        aspect = h_orig / max(w_orig, 1)
        if aspect > 2.5 and candidate == '?':
            digit     = '1'
            seg_tuple = seg_t
            print(f"  [SEG] Aspect-ratio '1' fallback: aspect={aspect:.1f} (w={w_orig} h={h_orig})")
            break

    if seg_tuple is None:
        seg_tuple = tuple(get_segments(0.10))

    print(f"  [SEG] Digit {index+1}: '{digit}' | segments={seg_tuple} | w={w_orig}px")
    return digit


def decode_all_digits(digit_detections, display_color="auto"):
    result = ""
    for i, det in enumerate(digit_detections):
        digit = decode_single_digit(
            det["roi"], index=i, display_color=display_color
        )
        if digit is not None:
            result += digit
    print(f"[SEG] All digits: '{result}'")
    return result


def insert_decimal(digits_str, decimal_x, digit_detections):
    if decimal_x is None or len(digit_detections) == 0:
        print("[DECIMAL] No decimal position → no dot inserted")
        return digits_str

    digits_left = sum(1 for d in digit_detections if d["cx"] < decimal_x)
    print(f"[DECIMAL] {digits_left} digit(s) left of decimal point")

    if digits_left <= 0 or digits_left >= len(digits_str):
        print("[DECIMAL] Position out of range → no dot inserted")
        return digits_str

    result = digits_str[:digits_left] + "." + digits_str[digits_left:]
    print(f"[DECIMAL] '{digits_str}' → '{result}'")
    return result