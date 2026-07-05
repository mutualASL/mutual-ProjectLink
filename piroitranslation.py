# ASL Live Translation — Raspberry Pi version
# Uses Keras model + libcamera GStreamer pipeline

import os
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ.setdefault("DISPLAY", ":0")

# ── Splash — single root window, logo + progress bar while imports load ───────
import tkinter as tk
_root = tk.Tk()
_root.attributes('-fullscreen', True)
_root.configure(bg='white')
_root.update()

# Logo — static image
try:
    from PIL import Image as _Img, ImageTk as _ITk
    _logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'link_logo.png')
    _size = 300
    _pil = _Img.open(_logo_path).convert('RGBA')
    # Scale to fit, maintain aspect ratio
    _pw, _ph = _pil.size
    _sc = _size / max(_pw, _ph)
    _pil = _pil.resize((int(_pw * _sc), int(_ph * _sc)), _Img.LANCZOS)
    # Center on white background
    _bg2 = _Img.new('RGBA', (_size, _size), (255, 255, 255, 255))
    _ox = (_size - _pil.size[0]) // 2
    _oy = (_size - _pil.size[1]) // 2
    _bg2.paste(_pil, (_ox, _oy), _pil)
    _photo = _ITk.PhotoImage(_bg2.convert('RGB'))
    _lbl = tk.Label(_root, image=_photo, bg='white')
    _lbl.image = _photo
    _lbl.pack(pady=(0, 0), expand=True)
except Exception:
    tk.Label(_root, text="ASL", bg='white', fg='#CCCCCC',
             font=('Helvetica', 72, 'bold')).pack(expand=True)

# Progress bar + status label
_BAR_W, _BAR_H, _BAR_R = 320, 10, 5
_prog_canvas = tk.Canvas(_root, width=_BAR_W, height=_BAR_H,
                          bg='white', highlightthickness=0)
_prog_canvas.pack(pady=(16, 6))

def _rrect(cv, x1, y1, x2, y2, r, **kw):
    pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
           x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
           x1,y2, x1,y2-r, x1,y1+r, x1,y1]
    cv.create_polygon(pts, smooth=True, **kw)

_rrect(_prog_canvas, 0, 0, _BAR_W, _BAR_H, _BAR_R, fill='#E8E8E8', outline='')
_prog_fill = _prog_canvas.create_polygon([0,0,0,0], fill='#1A1A1A', outline='')

_status_lbl = tk.Label(_root, text="Starting up\u2026", bg='white', fg='#AAAAAA',
                        font=('Helvetica', 13))
_status_lbl.pack(pady=(0, 60))
_root.update()

def _set_progress(pct, msg):
    w = max(0, int(_BAR_W * pct / 100))
    r = min(_BAR_H // 2, _BAR_R)
    pts = ([0, 0, w-r, 0, w, 0, w, r,
            w, _BAR_H-r, w, _BAR_H, w-r, _BAR_H, r, _BAR_H,
            0, _BAR_H, 0, _BAR_H-r, 0, r, 0, 0]
           if w > _BAR_R * 2 else
           [0, 0, w, 0, w, _BAR_H, 0, _BAR_H])
    _prog_canvas.coords(_prog_fill, pts)
    _status_lbl.config(text=msg)
    _root.update()

_set_progress(8, "Waking up the camera\u2026")

# ── Staged imports ─────────────────────────────────────────────────────────────
import time
import math
import random
from collections import deque
import threading
_set_progress(18, "Loading utilities\u2026")

import cv2
import numpy as np
_set_progress(32, "Preparing vision pipeline\u2026")

import mediapipe as mp
_set_progress(52, "Tracking hands\u2026")

import tensorflow as tf
_set_progress(74, "Loading the brain\u2026")

import requests
import pygame
from tkinter import font as tkfont
from PIL import Image, ImageTk, ImageDraw
_set_progress(86, "Almost there, hang tight\u2026")


# ------------------- Configuration -------------------
IMG_SIZE                = 200
HISTORY_LENGTH          = 10
FRAMES_THRESHOLD        = 3
CONFIDENCE_THRESHOLD    = 0.73
TOLERANCE_THRESHOLD     = 0.58
STABLE_TAIL             = 4
WORD_TIMEOUT            = 0.35
SENTENCE_TIMEOUT        = 5.0
CLEAR_TIMEOUT           = 9.0
DOUBLE_LETTER_TIME      = 0.65
DOUBLE_LETTER_STABILITY = 0.94
HF_TIMEOUT              = 30
CAPSULE_LINGER          = 2.0
LLM_PAUSE_TIMEOUT       = 4.0   # generous — user must clearly stop signing

# Inference thread: cap at this many fps to prevent sustained 100% CPU
INFER_FPS_CAP           = 15    # 15fps — accuracy/speed already good, saves heat
INFER_INTERVAL          = 1.0 / INFER_FPS_CAP

# LLM trigger: hand must be absent this many consecutive frames before firing
# At 18fps, 12 frames ≈ 660ms of genuine absence — filters lag dropouts
HAND_ABSENT_THRESH      = 12

# ── Performance / heat management ─────────────────────────────────────────────
MOTION_EPS              = 0.0035  # mean landmark delta below this = hand holding still
RECHECK_EVERY           = 4       # while still, force a real CNN pass at least every N+1 frames
IDLE_AFTER              = 2.0     # secs of no hand before slowing the capture loop
IDLE_EXTRA_SLEEP        = 0.10    # extra sleep per frame while idle (~6-7 fps effective)
WORD_GAP_SPACE          = 2.0     # FLOOR for a word-boundary pause; actual threshold adapts to signing speed
LLM_WARM_AFTER          = 240     # pre-warm the relay if it has been idle this long (s)

GOODBYE_PHRASES = [
    "Until next time\u2026",
    "Signing off \U0001f44b",
    "Good signing!",
    "See you soon \U0001f919",
    "Take care out there",
    "Catch you later",
    "Goodbye, friend \U0001f44b",
    "Peace \u270c\ufe0f",
    "That's a wrap!",
    "Stay expressive \U0001f91f",
]

PANEL_W       = 340
PANEL_PAD     = 20
PANEL_R       = 32

SMALL_SCREEN  = False
WIN_W         = 480  if SMALL_SCREEN else 1200
WIN_H         = 320  if SMALL_SCREEN else 700

# ------------------- Audio feedback -------------------
pygame.mixer.init()

def _make_tone(freq, duration, volume=0.4, fade_out=True):
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration), False)
    wave = np.sin(2 * np.pi * freq * t)
    if fade_out:
        env = np.exp(-t * 6.0 / duration)  # exponential decay
        wave *= env
    wave = (wave * volume * 32767).astype(np.int16)
    snd = pygame.mixer.Sound(wave.tobytes())
    return snd

def _make_chime():
    """Two-note ascending chime — plays when LLM translation arrives."""
    sr = 44100
    # Note 1: C5 (523 Hz), 0.12s
    d1 = 0.12
    t1 = np.linspace(0, d1, int(sr * d1), False)
    w1 = np.sin(2 * np.pi * 523 * t1) * np.exp(-t1 * 5 / d1)
    # Note 2: E5 (659 Hz), 0.18s — slightly louder, longer ring
    d2 = 0.18
    t2 = np.linspace(0, d2, int(sr * d2), False)
    w2 = np.sin(2 * np.pi * 659 * t2) * np.exp(-t2 * 4 / d2)
    # Gap between notes
    gap = np.zeros(int(sr * 0.04))
    wave = np.concatenate([w1, gap, w2])
    wave = (wave * 0.45 * 32767).astype(np.int16)
    return pygame.mixer.Sound(wave.tobytes())

# Soft tick for letter detection
letter_sound = _make_tone(880, 0.06, volume=0.25)
# Pleasant chime for translation complete
chime_sound  = _make_chime()

# ------------------- Text-to-speech -------------------
def _speak(text):
    """Speak translated text aloud through the USB speaker."""
    import subprocess, threading
    def _run():
        clean = text.replace('"', '').replace("'", '').replace(';', '').strip()
        if not clean:
            return
        print(f"[TTS] Speaking: {clean}")
        try:
            from gtts import gTTS
            tmp = "/tmp/asl_tts.mp3"
            gTTS(text=clean, lang="en", slow=False).save(tmp)
            print("[TTS] gTTS saved, playing with mpg123...")
            r = subprocess.run(["mpg123", "-q", "-a", "plughw:2,0", tmp], timeout=15,
                              capture_output=True, text=True)
            if r.returncode == 0:
                print("[TTS] mpg123 done")
                return
            print(f"[TTS] mpg123 failed: {r.stderr}")
        except Exception as e:
            print(f"[TTS] gTTS/mpg123 error: {e}")
        try:
            print("[TTS] Trying espeak...")
            r = subprocess.run(
                ["espeak", "-s", "150", "--stdout", clean],
                capture_output=True, timeout=10)
            subprocess.run(
                ["aplay", "-D", "plughw:2,0"],
                input=r.stdout, timeout=10)
            print("[TTS] espeak done")
        except Exception as e:
            print(f"[TTS] espeak error: {e}")
    threading.Thread(target=_run, daemon=True).start()

# ------------------- Mediapipe / Model setup -------------------
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# model_complexity=0: lite model, ~2x faster on Pi with no meaningful accuracy loss
hands      = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                             model_complexity=0,
                             min_detection_confidence=0.7,
                             min_tracking_confidence=0.6)

try:
    model = tf.keras.models.load_model('v3piroiasl_landmark_model.keras')
    model.summary()
    with open('class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print("Model and class names loaded. Classes:", class_names)
except Exception as e:
    print(f"Error loading model: {e}")
    raise
_set_progress(100, "Ready!")

def tflite_predict(processed_image):
    # Direct call skips model.predict()'s per-call graph/dataset overhead —
    # noticeably faster and cooler for single-frame inference on the Pi.
    return np.asarray(model(processed_image, training=False))

# ------------------- HF Relay -------------------
HF_RELAY_URL = "https://hfrelay-production.up.railway.app/translate"

APP_VERSION = "1.0"   # bump this when you push an update via the dashboard


# Session-persistent history of finalized English outputs.
# Persists until the process exits (device powers off) - never cleared by timeouts.
llm_history = deque(maxlen=6)

# Retry config for the relay call. The HF free tier unloads models when idle,
# so the first request after a pause can time out or 503 while the model warms
# up (typically 10-30s). We retry a couple of times before giving up.
HF_RETRIES     = 2      # extra attempts after the first (3 tries total)
HF_RETRY_WAIT  = 2.0    # seconds between attempts

# Tracks the last successful contact with the relay - used to decide when a
# pre-warm request is worthwhile.
_last_llm_contact = [0.0]

def query_hf_llm(letters: str, history=None) -> str:
    if not letters:
        return ""
    payload = {
        "text": letters,
        "mode": "asl",
        "history": list(history) if history else [],
    }
    last_err = None
    for attempt in range(HF_RETRIES + 1):
        try:
            r = requests.post(HF_RELAY_URL,
                              headers={"Content-Type": "application/json"},
                              json=payload,
                              timeout=HF_TIMEOUT)
            r.raise_for_status()
            result = r.json().get("result", "")
            if result:
                _last_llm_contact[0] = time.time()
                return result
            # Empty result (model warming up / transient) - fall through to retry
            last_err = "empty result"
        except Exception as e:
            last_err = e
            print(f"HF relay error (attempt {attempt+1}/{HF_RETRIES+1}): {e}")
        if attempt < HF_RETRIES:
            time.sleep(HF_RETRY_WAIT)   # brief wait, then retry (cold-start warmup)
    print(f"HF relay gave up after {HF_RETRIES+1} attempts: {last_err}")
    return ""

def _warm_llm_async():
    """Fire-and-forget tiny request so the HF model gets loaded BEFORE the real
    query arrives. Called when an utterance starts after a long idle period —
    by the time the user finishes spelling, the model is warm and the real
    translation returns fast instead of hitting a 10-30s cold start."""
    def _w():
        try:
            requests.post(HF_RELAY_URL,
                          headers={"Content-Type": "application/json"},
                          json={"text": "hi", "mode": "asl", "history": []},
                          timeout=25)
            _last_llm_contact[0] = time.time()
            print("LLM pre-warm done")
        except Exception:
            pass
    threading.Thread(target=_w, daemon=True).start()

# ------------------- Hand landmark processing -------------------
def get_hand_landmarks(image, already_rgb=False):
    rgb     = image if already_rgb else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    return results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None

def check_wrist_slope(hand_landmarks, image_shape):
    if not hand_landmarks:
        return None
    h, w   = image_shape[:2]
    wrist  = (int(hand_landmarks.landmark[0].x * w),  int(hand_landmarks.landmark[0].y * h))
    pinky  = (int(hand_landmarks.landmark[17].x * w), int(hand_landmarks.landmark[17].y * h))
    dx, dy = pinky[0] - wrist[0], pinky[1] - wrist[1]
    slope  = float('inf') if dx == 0 else dy / dx
    return slope, wrist, pinky

def calculate_slope_and_adjust(hand_landmarks, image_shape, target_size=(200, 200)):
    if not hand_landmarks:
        return None
    h, w   = image_shape[:2]
    th, tw = target_size
    lm     = [(int(l.x * w), int(l.y * h)) for l in hand_landmarks.landmark]

    wrist                       = lm[0]
    i_mcp, m_mcp, r_mcp, p_mcp = lm[5], lm[9], lm[13], lm[17]
    xr_min = min(i_mcp[0], m_mcp[0], r_mcp[0], p_mcp[0])
    xr_max = max(i_mcp[0], m_mcp[0], r_mcp[0], p_mcp[0])

    slope_info  = check_wrist_slope(hand_landmarks, image_shape)
    slope       = slope_info[0] if slope_info else None
    is_vertical = xr_min <= wrist[0] <= xr_max
    is_horiz    = (wrist[0] < xr_min or wrist[0] > xr_max) or \
                  (slope is not None and -0.5 <= slope <= 0.5)

    if is_vertical and not is_horiz:
        fw      = (tw // 2, th - 50)
        mc      = ((i_mcp[0]+m_mcp[0]+r_mcp[0]+p_mcp[0])//4,
                   (i_mcp[1]+m_mcp[1]+r_mcp[1]+p_mcp[1])//4)
        dx, dy  = mc[0]-wrist[0], mc[1]-wrist[1]
        ang     = (-math.pi/2 - math.atan2(dy, dx)) if (dx or dy) else 0
        ca, sa  = math.cos(ang), math.sin(ang)
        xlm     = [(int((x-wrist[0])*ca-(y-wrist[1])*sa+fw[0]),
                    int((x-wrist[0])*sa+(y-wrist[1])*ca+fw[1])) for x, y in lm]
    elif is_horiz and slope is not None and -0.5 <= slope <= 0.5:
        dx, dy  = p_mcp[0]-wrist[0], p_mcp[1]-wrist[1]
        ang     = (0 - math.atan2(dy, dx)) if (dx or dy) else 0
        ca, sa  = math.cos(ang), math.sin(ang)
        xlm     = [(int((x-wrist[0])*ca-(y-wrist[1])*sa+wrist[0]),
                    int((x-wrist[0])*sa+(y-wrist[1])*ca+wrist[1])) for x, y in lm]
    elif is_horiz and slope is not None and abs(slope) > 0.5:
        fw      = (tw // 2, th - 50)
        mc      = ((i_mcp[0]+m_mcp[0]+r_mcp[0]+p_mcp[0])//4,
                   (i_mcp[1]+m_mcp[1]+r_mcp[1]+p_mcp[1])//4)
        dx, dy  = mc[0]-wrist[0], mc[1]-wrist[1]
        ang     = (-math.pi/2 - math.atan2(dy, dx)) if (dx or dy) else 0
        ca, sa  = math.cos(ang), math.sin(ang)
        xlm     = [(int((x-wrist[0])*ca-(y-wrist[1])*sa+fw[0]),
                    int((x-wrist[0])*sa+(y-wrist[1])*ca+fw[1])) for x, y in lm]
    else:
        xlm = lm

    xs, ys   = [p[0] for p in xlm], [p[1] for p in xlm]
    xmn, xmx = min(xs), max(xs)
    ymn, ymx = min(ys), max(ys)
    pad      = 20
    sc       = min((tw-2*pad) / (xmx-xmn if xmx>xmn else 1),
                   (th-2*pad) / (ymx-ymn if ymx>ymn else 1))
    ox       = (tw - (xmx-xmn)*sc) // 2
    oy       = (th - (ymx-ymn)*sc) // 2
    flm      = [(int((x-xmn)*sc+ox), int((y-ymn)*sc+oy)) for x, y in xlm]

    img = np.ones((th, tw, 3), dtype=np.uint8) * 255
    for s, e in mp_hands.HAND_CONNECTIONS:
        if s < len(flm) and e < len(flm):
            cv2.line(img, flm[s], flm[e], (0, 0, 0), 1)
    for x, y in flm:
        cv2.circle(img, (x, y), 3, (0, 0, 0), 1)
    if len(flm) >= 18:
        cv2.line(img, flm[0], flm[17], (255, 0, 0), 2)
    return img

def preprocess_image(lm_img):
    if lm_img is None:
        return None
    gray = cv2.cvtColor(lm_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return (resized.astype('float32') / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)

def get_hand_crop_with_landmarks(frame_bgr, hand_landmarks, padding=50):
    if hand_landmarks is None or frame_bgr is None:
        return None
    h, w = frame_bgr.shape[:2]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x1 = max(0, int(min(xs)) - padding)
    x2 = min(w, int(max(xs)) + padding)
    y1 = max(0, int(min(ys)) - padding)
    y2 = min(h, int(max(ys)) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    frame_copy = frame_bgr.copy()
    mp_drawing.draw_landmarks(
        frame_copy, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 200, 80),  thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(255, 80,  0),  thickness=2)
    )
    return frame_copy[y1:y2, x1:x2]

# ------------------- Sequence processing -------------------
prediction_history  = deque(maxlen=HISTORY_LENGTH)
current_letters     = []
all_raw_letters     = []
letter_times        = []   # timestamp per entry in all_raw_letters (word-gap segmentation)
is_recording        = False
last_confident_time = None
last_letter_time    = None
current_letter      = ""
sentence_text       = ""
last_translation_time = 0.0   # time a real LLM translation was last shown (protects it from inactivity-clear)

def process_letter(prediction_history, current_time):
    global current_letters, all_raw_letters, is_recording
    global last_confident_time, last_letter_time, current_letter

    if not prediction_history or len(prediction_history) < FRAMES_THRESHOLD:
        return

    history_list = list(prediction_history)
    n            = len(history_list)
    weights      = {}
    for i, letter in enumerate(history_list):
        w = 1.0 + (i / n)
        weights[letter] = weights.get(letter, 0) + w

    most_common      = max(weights, key=weights.get)
    total_weight     = sum(weights.values())
    confidence_ratio = weights[most_common] / total_weight

    unique_preds = len(set(history_list))
    is_chaotic   = (history_list.count(most_common) / n < 0.45
                    or unique_preds / HISTORY_LENGTH > 0.5)

    if confidence_ratio >= TOLERANCE_THRESHOLD and most_common in class_names:
        tail = history_list[-STABLE_TAIL:]
        if len(tail) < STABLE_TAIL or any(l != most_common for l in tail):
            last_confident_time = current_time
            return

        if not is_recording:
            is_recording    = True
            current_letters = []

        last_letter = current_letters[-1] if current_letters else None

        trailing = 0
        for ch in reversed(current_letters):
            if ch == most_common:
                trailing += 1
            else:
                break

        if trailing >= 2:
            pass

        elif last_letter != most_common:
            current_letters.append(most_common)
            all_raw_letters.append(most_common)
            letter_times.append(current_time)
            current_letter   = most_common
            last_letter_time = current_time
            prediction_history.clear()
            try: letter_sound.play()
            except: pass

        else:
            if last_letter_time and (current_time - last_letter_time >= DOUBLE_LETTER_TIME):
                stability = weights[most_common] / total_weight
                if not is_chaotic and stability >= DOUBLE_LETTER_STABILITY:
                    current_letters.append(most_common)
                    all_raw_letters.append(most_common)
                    letter_times.append(current_time)
                    current_letter   = most_common
                    last_letter_time = current_time
                    prediction_history.clear()
                    try: letter_sound.play()
                    except: pass

        last_confident_time = current_time

# ------------------- Capsule spring physics -------------------
def _spring(pos, vel, target, k, d):
    f   = (target - pos) * k
    vel = vel * d + f
    return pos + vel, vel


class Pill:
    W = 40
    H = 40

    def __init__(self, letter, tx):
        self.letter  = letter
        self.tx      = float(tx)
        self.x       = float(tx)
        self.y       = -float(self.H) * 1.8
        self.ty      = 0.0
        self.vx      = 0.0
        self.vy      = 0.0
        self.scale   = 0.0
        self.vs      = 0.0
        self.alpha   = 1.0
        self.exiting = False
        self.done    = False

    def tick(self):
        self.x,     self.vx = _spring(self.x,     self.vx, self.tx,  0.42, 0.60)
        self.y,     self.vy = _spring(self.y,     self.vy, self.ty,  0.42, 0.60)
        ts = 0.0 if self.exiting else 1.0
        self.scale, self.vs = _spring(self.scale, self.vs, ts,       0.36, 0.56)
        self.scale = max(0.0, min(self.scale, 1.6))
        if self.exiting:
            self.alpha = max(0.0, self.alpha - 0.10)
            if self.alpha <= 0.02:
                self.done = True

    def exit(self):
        self.exiting = True
        self.ty      = float(self.H) * 1.5

    @property
    def settled(self):
        return (abs(self.ty - self.y)                            < 0.6 and
                abs(self.tx - self.x)                            < 0.6 and
                abs((0.0 if self.exiting else 1.0) - self.scale) < 0.02)


class CapsuleLetterDisplay(tk.Canvas):
    PILL_W  = 34
    PILL_H  = 34
    GAP     = 4
    PAD_X   = 10
    CAP_H   = 66

    BG       = "#F2F2F2"
    OUTLINE  = "#CCCCCC"
    C_NORM   = "#E0E0E0"
    C_LATEST = "#1A1A1A"
    T_NORM   = "#333333"
    T_LATEST = "#FFFFFF"

    def __init__(self, parent, max_visible=14, **kw):
        self._letters     = []
        self._pills       = []
        self._exiting     = []
        self._running     = False
        self._max_visible = max_visible
        self._pill_family = "Helvetica"
        self._pill_italic = False
        w = self._W(max_visible)
        super().__init__(parent, width=w, height=self.CAP_H,
                         bg=parent["bg"], highlightthickness=0, **kw)
        self._redraw()

    def set_font(self, family, italic=False):
        self._pill_family = family
        self._pill_italic = italic

    def set_max_visible(self, n):
        if n == self._max_visible:
            return
        self._max_visible = max(1, n)
        self.config(width=self._W(self._max_visible))
        self.set_letters(self._letters)

    def set_letters(self, letters: list):
        if letters == self._letters:
            return   # nothing changed — skip all pill update logic
        new     = list(letters)
        visible = new[-self._max_visible:]

        if not new:
            for p in self._pills:
                p.exit()
            self._exiting.extend(self._pills)
            self._pills = []
            self._letters = new
            self._kick()
            return

        prev_start = max(0, len(self._letters) - self._max_visible)
        new_start  = max(0, len(new)           - self._max_visible)
        n_to_drop  = max(
            len(self._pills) - len(visible),
            new_start - prev_start,
        )
        n_to_drop = max(0, min(n_to_drop, len(self._pills)))

        if n_to_drop > 0:
            for p in self._pills[:n_to_drop]:
                p.exit()
            self._exiting.extend(self._pills[:n_to_drop])
            self._pills = self._pills[n_to_drop:]

        for i, p in enumerate(self._pills):
            p.tx = float(self._X(i))

        while len(self._pills) < len(visible):
            idx = len(self._pills)
            self._pills.append(Pill(visible[idx], self._X(idx)))

        self._letters = new
        self._kick()

    def _W(self, n):
        return 2*self.PAD_X + n*self.PILL_W + max(n-1, 0)*self.GAP

    def _X(self, i):
        return self.PAD_X + i * (self.PILL_W + self.GAP)

    def _kick(self):
        if not self._running:
            self._running = True
            self._tick()

    def _tick(self):
        for p in self._pills:
            p.tick()
        for p in self._exiting:
            p.tick()
        self._exiting = [p for p in self._exiting if not p.done]
        self._redraw()
        moving = (any(not p.settled or p.exiting for p in self._pills)
                  or bool(self._exiting))
        if moving:
            self.after(16, self._tick)   # ~62fps during movement — brief bursts, not continuous
        else:
            self._running = False

    def _redraw(self):
        self.delete("all")
        w  = self._W(self._max_visible)
        h  = self.CAP_H
        cy = h // 2
        n  = len(self._pills)
        self.config(width=w)
        self._rrect(0, 4, w, h-4, 26, fill=self.BG, outline=self.OUTLINE, width=1)
        for i, pill in enumerate(self._pills):
            self._dpill(pill, cy, is_latest=(i == n-1))
        for pill in self._exiting:
            self._dpill(pill, cy, is_latest=False)

    def _dpill(self, pill, cy, is_latest):
        s = max(0.0, pill.scale)
        a = pill.alpha
        if s < 0.03 or a < 0.03:
            return
        pw  = max(4, int(self.PILL_W * s))
        ph  = max(4, int(self.PILL_H * s))
        pr  = ph // 2
        cx  = int(pill.x + self.PILL_W / 2)
        cy2 = int(cy + pill.y)
        fill = self.C_LATEST if is_latest else self.C_NORM
        tcol = self.T_LATEST if is_latest else self.T_NORM
        fc = self._blend(self._rgb(fill), self._rgb(self.BG), a)
        tc = self._blend(self._rgb(tcol), self._rgb(self.BG), a)
        self._rrect(cx-pw//2, cy2-ph//2, cx+pw//2, cy2+ph//2,
                    pr, fill=self._hex(fc), outline="")
        if s > 0.45:
            slant = "italic" if self._pill_italic else "roman"
            self.create_text(cx, cy2, text=pill.letter.upper(),
                             fill=self._hex(tc),
                             font=(self._pill_family, max(9, int(15*s)), "bold", slant))
        if is_latest and s > 0.7:
            dot_y   = cy2 + ph//2 + 6
            dot_col = self._blend(self._rgb('#1A1A1A'), self._rgb(self.BG), a * 0.7)
            lbl_col = self._blend(self._rgb('#888888'), self._rgb(self.BG), a * 0.8)
            self.create_oval(cx-2, dot_y-2, cx+2, dot_y+2,
                             fill=self._hex(dot_col), outline="")
            self.create_text(cx, dot_y + 10, text="now",
                             fill=self._hex(lbl_col),
                             font=(self._pill_family, 8))

    def _rrect(self, x1, y1, x2, y2, r, **kw):
        r   = max(1, r)
        pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
               x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
               x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        return self.create_polygon(pts, smooth=True, **kw)

    @staticmethod
    def _rgb(h):
        h = h.lstrip("#")
        return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

    @staticmethod
    def _blend(fg, bg, a):
        return tuple(int(fg[i]*a + bg[i]*(1-a)) for i in range(3))

    @staticmethod
    def _hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)


# ------------------- Logo Animation Player -------------------
LOGO_NORMAL_FILE = "link_default.mov"
LOGO_RECOG_FILE  = "link_query.mov"
LOGO_SIZE        = 180
LOGO_PAD         = 14

class LogoPlayer:
    def __init__(self, root, size=LOGO_SIZE, pad=LOGO_PAD, on_tap=None):
        self.root       = root
        self.size       = size
        self.pad        = pad
        self._mode      = 'normal'
        self._frame_idx = 0
        self._job       = None
        self._recog_pending = False
        self._on_tap    = on_tap
        self.canvas = tk.Canvas(root, width=size, height=size,
                                highlightthickness=0, bg='white',
                                cursor='hand2' if on_tap else 'arrow')
        self.canvas.place(x=pad, y=pad)
        self._img_item = self.canvas.create_image(0, 0, anchor='nw')
        if on_tap:
            self.canvas.bind('<ButtonPress-1>', lambda e: on_tap())
        self._img_item = self.canvas.create_image(0, 0, anchor='nw')
        self._fps_normal, pil_normal = self._load_video(LOGO_NORMAL_FILE)
        self._fps_recog,  pil_recog  = self._load_video(LOGO_RECOG_FILE)
        self._pil_normal = pil_normal
        self._pil_recog  = pil_recog
        self._frames_normal = [ImageTk.PhotoImage(f) for f in pil_normal]
        self._frames_recog  = [ImageTk.PhotoImage(f) for f in pil_recog]
        self._scaled_cache  = {}
        if not self._frames_normal:
            print(f"Warning: could not load {LOGO_NORMAL_FILE} — logo disabled")
            self.canvas.place_forget()
            return
        self._start_normal()

    def trigger_recog(self):
        if not self._frames_recog:
            return
        self._recog_pending = True
        if self._mode == 'normal':
            self._start_recog()

    def set_size(self, new_size):
        new_size = max(40, int(new_size))
        if abs(new_size - self.size) < 2:
            return
        self.size = new_size
        self.canvas.config(width=new_size, height=new_size)
        self._scaled_cache = {}

    def _get_frame(self, pil_frames, idx):
        key = (id(pil_frames), idx, self.size)
        if key not in self._scaled_cache:
            # Cap cache at 200 entries to prevent unbounded memory growth
            if len(self._scaled_cache) > 200:
                keys = list(self._scaled_cache.keys())
                for k in keys[:100]:
                    del self._scaled_cache[k]
            pil = pil_frames[idx % len(pil_frames)]
            if pil.size[0] != self.size:
                pil = pil.resize((self.size, self.size), Image.LANCZOS)
            self._scaled_cache[key] = ImageTk.PhotoImage(pil)
        return self._scaled_cache[key]

    def destroy(self):
        if self._job:
            self.root.after_cancel(self._job)
        self.canvas.destroy()

    def _start_normal(self):
        if self._job:
            self.root.after_cancel(self._job)
            self._job = None
        self._mode = 'normal'; self._frame_idx = 0; self._tick()

    def _start_recog(self):
        if self._job:
            self.root.after_cancel(self._job)
            self._job = None
        self._mode = 'recog'; self._frame_idx = 0; self._recog_pending = False; self._tick()

    def _tick(self):
        pil_frames = self._pil_normal if self._mode == 'normal' else self._pil_recog
        fps        = self._fps_normal if self._mode == 'normal' else self._fps_recog
        if not pil_frames:
            return
        if self._mode == 'normal':
            delay = max(66, int(1000 / fps))    # normal loop capped at 15fps
        else:
            delay = max(20, int(1000 / fps / 2))  # query plays at 2x speed
        idx   = self._frame_idx % len(pil_frames)
        photo = self._get_frame(pil_frames, idx)
        self.canvas.itemconfig(self._img_item, image=photo)
        self._frame_idx += 1
        if self._mode == 'normal':
            self._frame_idx = self._frame_idx % len(pil_frames)
            self._job = self.root.after(delay, self._tick)
        else:
            if self._frame_idx >= len(pil_frames):
                self._start_normal()
            else:
                self._job = self.root.after(delay, self._tick)

    def _load_video(self, filename):
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, filename)
        if not os.path.exists(path):
            print(f"Logo: file not found — {path}"); return 30, []
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Logo: cannot open — {path}"); return 30, []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 240: fps = 30
        fps = float(fps)
        size = self.size
        mask = Image.new('L', (size, size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, size-1, size-1), fill=255)
        pil_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            fh, fw = frame.shape[:2]
            side = min(fh, fw)
            y0 = (fh-side)//2; x0 = (fw-side)//2
            frame = frame[y0:y0+side, x0:x0+side]
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame).convert('RGBA')
            pil.putalpha(mask)
            bg = Image.new('RGBA', (size, size), (255,255,255,255))
            bg.paste(pil, (0,0), pil)
            pil_frames.append(bg.convert('RGB'))
        cap.release()
        print(f"Logo: loaded {len(pil_frames)} frames from {filename} @ {fps:.1f} fps")
        return fps, pil_frames


# ------------------- Hand View Panel -------------------
PANEL_H_BOTTOM  = 0.52
SENTENCE_BAR_H  = 180

def _panel_h(win_h):
    return min(int(win_h * PANEL_H_BOTTOM), win_h - SENTENCE_BAR_H - 8)

class HandViewPanel:
    def __init__(self, root):
        self.root        = root
        self._visible    = False
        self._anim       = None
        self._photo_crop = None
        self._photo_lm   = None
        self._win_w      = WIN_W
        self._win_h      = WIN_H

        self._canvas = tk.Canvas(root, bg='white', highlightthickness=0)
        self._canvas.place(x=0, y=WIN_H)
        self._draw_bg(WIN_W, _panel_h(WIN_H))

        PAD = 12
        self._frame = tk.Frame(self._canvas, bg='#F5F5F5')
        self._frame_win = self._canvas.create_window(
            WIN_W//2, _panel_h(WIN_H)//2,
            window=self._frame,
            width=WIN_W - PAD*2,
            height=_panel_h(WIN_H) - PAD*2)
        self._frame.pack_propagate(False)

        self._left  = tk.Frame(self._frame, bg='#F5F5F5')
        self._right = tk.Frame(self._frame, bg='#F5F5F5')
        self._left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8,4), pady=8)
        self._right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4,8), pady=8)

        tk.Label(self._left, text="camera", bg='#F5F5F5', fg='#AAAAAA',
                 font=tkfont.Font(family="Helvetica", size=9)).pack()
        self.crop_label = tk.Label(self._left, bg='#E0E0E0')
        self.crop_label.pack(fill=tk.BOTH, expand=True)

        tk.Label(self._right, text="landmarks", bg='#F5F5F5', fg='#AAAAAA',
                 font=tkfont.Font(family="Helvetica", size=9)).pack()
        self.lm_label = tk.Label(self._right, bg='#EBEBEB')
        self.lm_label.pack(fill=tk.BOTH, expand=True)

    def on_resize(self, win_w, win_h):
        self._win_w  = win_w
        self._win_h  = win_h
        panel_h = _panel_h(win_h)
        PAD = 12
        self._canvas.config(width=win_w, height=panel_h)
        self._draw_bg(win_w, panel_h)
        self._canvas.coords(self._frame_win, win_w//2, panel_h//2)
        self._canvas.itemconfig(self._frame_win,
                                width=win_w - PAD*2,
                                height=panel_h - PAD*2)
        if self._visible:
            self._canvas.place(x=0, y=win_h - panel_h)
        else:
            self._canvas.place(x=0, y=win_h)

    def _draw_bg(self, w, h):
        self._canvas.delete("bg")
        r = PANEL_R
        for i, shade in enumerate(['#E0E0E0', '#D8D8D8', '#CECECE']):
            self._rrect(i+2, i+2, w-2+i, h-2+i, r, fill=shade, outline='', tag='bg')
        self._rrect(0, 0, w-6, h, r, fill='#F6F6F6', outline='#E2E2E2', width=1, tag='bg')

    def _rrect(self, x1, y1, x2, y2, r, tag='', **kw):
        r   = max(1, r)
        pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
               x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
               x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        return self._canvas.create_polygon(pts, smooth=True, tags=tag, **kw)

    def show(self):
        if self._visible: return
        self._visible = True
        panel_h = _panel_h(self._win_h)
        self._animate(start=self._win_h, end=self._win_h - panel_h)

    def hide(self):
        if not self._visible: return
        self._visible = False
        self._animate(start=self._win_h - _panel_h(self._win_h), end=self._win_h)

    def update(self, crop_bgr, landmark_bgr):
        if not self._visible: return
        panel_h = _panel_h(self._win_h)
        half_w  = (self._win_w - 32) // 2
        img_h   = panel_h - 40
        if crop_bgr is not None and crop_bgr.size > 0:
            try:
                rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil    = Image.fromarray(rgb)
                ch, cw = crop_bgr.shape[:2]
                asp    = ch / max(cw, 1)
                nw     = min(half_w, int(img_h / asp))
                nh     = int(nw * asp)
                pil    = pil.resize((nw, nh), Image.BILINEAR)
                self._photo_crop = ImageTk.PhotoImage(pil)
                self.crop_label.config(image=self._photo_crop)
            except Exception: pass
        if landmark_bgr is not None:
            try:
                rgb  = cv2.cvtColor(landmark_bgr, cv2.COLOR_BGR2RGB)
                sz   = min(half_w, img_h)
                pil  = Image.fromarray(rgb).resize((sz, sz), Image.BILINEAR)
                self._photo_lm = ImageTk.PhotoImage(pil)
                self.lm_label.config(image=self._photo_lm)
            except Exception: pass

    @property
    def visible(self): return self._visible

    def _animate(self, start, end, steps=8):   # 8 steps x 12ms = 96ms — snappy slide
        if self._anim: self.root.after_cancel(self._anim)
        def _tick(i):
            t = i / steps
            t = 1 - (1 - t) ** 3
            y = int(start + (end - start) * t)
            self._canvas.place(x=0, y=y)
            if i < steps:
                self._anim = self.root.after(12, lambda: _tick(i+1))
            else:
                self._anim = None
        _tick(0)


# ------------------- Circular Hand Toggle Button -------------------
class CircularHandButton(tk.Canvas):
    SIZE = 54
    PAD  = 16

    def __init__(self, root, on_toggle):
        super().__init__(root, width=self.SIZE, height=self.SIZE,
                         bg='white', highlightthickness=0, cursor='hand2')
        self._on_toggle = on_toggle
        self._active    = False
        self._hovered   = False
        self._pressed   = False
        self._reposition(WIN_W)
        self._draw()
        self.bind('<ButtonPress-1>',   self._press)
        self.bind('<ButtonRelease-1>', self._release)
        self.bind('<Enter>',  lambda e: self._set_hover(True))
        self.bind('<Leave>',  lambda e: (self._set_hover(False),
                                          setattr(self, '_pressed', False),
                                          self._draw()))

    def on_resize(self, win_w, win_h):
        self._reposition(win_w)

    def _reposition(self, win_w):
        self.place(x=win_w - self.SIZE - self.PAD, y=self.PAD)

    def _draw(self):
        self.delete("all")
        s = self.SIZE
        if self._pressed:
            bg, ol, fg = '#CCCCCC', '#AAAAAA', '#222222'
        elif self._active:
            bg, ol, fg = '#1A1A1A', '#1A1A1A', '#FFFFFF'
        elif self._hovered:
            bg, ol, fg = '#F0F0F0', '#CCCCCC', '#222222'
        else:
            bg, ol, fg = '#FFFFFF', '#DDDDDD', '#333333'
        self.create_oval(2, 2, s-2, s-2, fill=bg, outline=ol, width=1.5)
        self.create_text(s//2, s//2, text="\u270b",
                         font=("Noto Color Emoji", 20), fill=fg)

    def _press(self, _):
        self._pressed = True; self._draw()

    def _release(self, _):
        self._pressed = False
        self._active  = not self._active
        self._draw()
        self._on_toggle(self._active)

    def _set_hover(self, val):
        self._hovered = val; self._draw()


# ------------------- Slide To Shutdown -------------------
class SlideToShutdown(tk.Canvas):
    """
    Slide-to-shutdown bar.
    Grey pill track whose ends are flush with the circle diameter.
    White circle with black outline as thumb.
    Drag all the way right to confirm shutdown; spring back if released early.
    """
    THUMB_R = 40   # circle radius — slightly larger, proportional scale
    PAD     = 5    # padding between circle and track edge

    def __init__(self, parent, width, on_confirm, on_cancel, **kw):
        # Track height exactly = circle diameter; pill radius = THUMB_R
        H = (self.THUMB_R + self.PAD) * 2
        super().__init__(parent, width=width, height=H,
                         bg='white', highlightthickness=0,
                         cursor='hand2', **kw)
        self._W          = width
        self._H          = H
        self._on_confirm = on_confirm
        self._on_cancel  = on_cancel
        self._tx         = float(self.THUMB_R + self.PAD)
        self._vel        = 0.0
        self._dragging   = False
        self._confirmed  = False
        self._spring_job = None

        self.bind('<ButtonPress-1>',   self._press)
        self.bind('<B1-Motion>',       self._motion)
        self.bind('<ButtonRelease-1>', self._release)
        self._draw()

    def _pill(self, x1, y1, x2, y2, r, **kw):
        r = max(1, min(r, (y2-y1)//2, max(1,(x2-x1)//2)))
        pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
               x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
               x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        self.create_polygon(pts, smooth=True, **kw)

    def _draw(self):
        self.delete('all')
        W, H, R = self._W, self._H, self.THUMB_R
        P  = self.PAD
        tx = self._tx
        cy = H // 2
        lo = R + P
        hi = W - R - P
        progress = max(0.0, min(1.0, (tx - lo) / max(1, hi - lo)))

        # Grey pill track — true semicircular ends (r = track height / 2)
        track_r = (H - 2*P) // 2
        self._pill(0, P, W, H-P, track_r, fill='#D8D8D8', outline='')

        # Fill left of thumb: grey → black as thumb slides right
        if progress > 0.02:
            shade = int(0xD8 + (0x1A - 0xD8) * progress)   # #D8D8D8 → #1A1A1A
            col   = f'#{shade:02x}{shade:02x}{shade:02x}'
            self._pill(0, P, int(tx), H-P, track_r, fill=col, outline='')

        # Label — fades as thumb moves right
        if progress < 0.55:
            alpha = max(0.0, 1.0 - progress * 2.0)
            grey  = int(0x44 + (0xBB - 0x44) * (1 - alpha))
            col   = f'#{grey:02x}{grey:02x}{grey:02x}'
            self.create_text(W//2 + R//2, cy,
                             text="slide to power off  \u2192",
                             fill=col, font=('Helvetica', 15, 'bold'))

        # Thumb: black outline ring
        self.create_oval(tx-R, cy-R, tx+R, cy+R,
                         fill='#1A1A1A', outline='')
        # Thumb: white inner circle (3px inset)
        self.create_oval(tx-R+3, cy-R+3, tx+R-3, cy+R-3,
                         fill='white', outline='')
        # Power symbol drawn manually — reliable on Pi regardless of font
        _ri  = max(5, int(R * 0.40))          # arc radius
        _lw  = max(2, int(R * 0.13))          # stroke thickness
        _col = '#1A1A1A'
        # Arc: 300° circle with 60° gap centred at top (start=120°, extent=300°)
        self.create_arc(int(tx)-_ri, cy-_ri, int(tx)+_ri, cy+_ri,
                        start=120, extent=300,
                        style='arc', outline=_col, width=_lw)
        # Vertical line from centre upward through the gap
        self.create_line(int(tx), cy, int(tx), cy-_ri-_lw,
                         fill=_col, width=_lw, capstyle='round')

    # ── interaction ───────────────────────────────────────────────────────────
    def _clamp_x(self, x):
        lo = self.THUMB_R + self.PAD
        hi = self._W - self.THUMB_R - self.PAD
        return max(lo, min(hi, x))

    def _press(self, e):
        if self._confirmed: return
        if self._spring_job:
            self.after_cancel(self._spring_job)
            self._spring_job = None
        self._dragging = True
        self._tx = self._clamp_x(e.x)
        self._draw()

    def _motion(self, e):
        if not self._dragging or self._confirmed: return
        self._tx = self._clamp_x(e.x)
        self._draw()

    def _release(self, e):
        if not self._dragging or self._confirmed: return
        self._dragging = False
        hi       = self._W - self.THUMB_R - self.PAD
        progress = (self._tx - self.THUMB_R - self.PAD) / max(1, hi - self.THUMB_R - self.PAD)
        if progress >= 0.88:
            self._confirmed = True
            self._draw()
            self.after(300, self._on_confirm)
        else:
            self._spring_back()

    def _spring_back(self):
        target = float(self.THUMB_R + self.PAD)
        self._vel = 0.0
        def _tick():
            force      = (target - self._tx) * 0.35
            self._vel  = self._vel * 0.60 + force
            self._tx  += self._vel
            self._draw()
            if abs(self._tx - target) > 1.0 or abs(self._vel) > 0.5:
                self._spring_job = self.after(14, _tick)
            else:
                self._tx = target
                self._vel = 0.0
                self._draw()
                self._spring_job = None
        _tick()


# ------------------- UI -------------------
class ASLTranslationUI:
    _GLOW    = ["#00EE55","#00DD44","#00CC3A","#00AA2D",
                "#008820","#006616","#00440F","#000000"]
    _GLOW_MS = 18   # faster flash — 18ms x 8 steps = 144ms total
    PILL_W   = 40
    PILL_GAP = 6
    PILL_PAD = 14

    def __init__(self, root):
        self.root        = root
        self.root.title("ASL Live Translation")
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.resizable(True, True)
        self.root.configure(bg='white')
        self.root.attributes('-fullscreen', True)

        self._panel_open  = False
        self._win_w       = WIN_W
        self._win_h       = WIN_H
        self._shift_job   = None
        self._cur_shift   = 0
        self._drawer_open = False

        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
            from asl_settings import Store as _Store
            _s = _Store()
            self._font_family = _s.get("font_family")
            self._font_size   = _s.get("font_size")
            self._font_bold   = bool(_s.get("font_bold"))
            self._font_italic = bool(_s.get("font_italic"))
        except Exception:
            self._font_family = "Helvetica"
            self._font_size   = 48
            self._font_bold   = True
            self._font_italic = False

        self.top_frame = tk.Frame(root, bg='white')
        self.top_frame.pack(fill=tk.BOTH, expand=True)

        self.word_label = tk.Label(
            self.top_frame, text="", bg='white', fg='black',
            font=tkfont.Font(family="Helvetica", size=48, weight="bold"),
            wraplength=750
        )
        self.word_label.pack(pady=(50, 4))

        self._lf = tk.Frame(self.top_frame, bg='white')
        self._lf.pack(pady=(0, 2))

        self.big_letter = tk.Label(
            self._lf, text="", bg='white', fg='black',
            font=tkfont.Font(family="Helvetica", size=80, weight="bold"),
            width=2, anchor='center', bd=0, highlightthickness=0
        )
        self.big_letter.pack()

        self.confirmed_label = tk.Label(
            self._lf, text="", bg='white', fg='#999999',
            font=tkfont.Font(family="Helvetica", size=13),
            anchor='center', bd=0, highlightthickness=0
        )
        self.confirmed_label.pack()

        self._cf = tk.Frame(self.top_frame, bg='white', bd=0, highlightthickness=0)
        self._cf.pack(pady=(8, 2))
        self.capsule = CapsuleLetterDisplay(self._cf, max_visible=14)
        self.capsule.pack()

        self._oof_label = tk.Label(
            self.top_frame, text="\u26a0 MOVE HAND INTO VIEW",
            bg='white', fg='#CC0000',
            font=tkfont.Font(family="Helvetica", size=18, weight="bold"),
            anchor='center'
        )
        self._oof_state = None   # dirty flag — only pack/unpack on change

        self.bottom_frame = tk.Frame(root, bg='#808080', height=180)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True)
        self.bottom_frame.pack_propagate(False)

        # Rounded top corners on the grey bar
        self.sentence_label = tk.Label(
            self.bottom_frame, text="", bg='#808080', fg='white',
            font=tkfont.Font(family="Helvetica", size=20),
            justify=tk.CENTER, anchor='center'
        )
        self.sentence_label.pack(pady=8, padx=16, fill=tk.BOTH, expand=True)
        self._font_family = 'Helvetica'
        self._font_italic = False

        self._prev_big      = ""
        self._prev_conf     = ""
        self._prev_word     = ""
        self._prev_sentence = ""
        self._glow_job  = None
        self._bullseye_jobs = []
        self._panel_tick    = 0   # hand panel update throttle

        self._strip_frame = tk.Frame(self.top_frame, bg='#909090', padx=16, pady=10)
        self._strip_label = tk.Label(
            self._strip_frame, text="", bg='#909090', fg='white',
            font=tkfont.Font(family="Helvetica", size=20),
            wraplength=self._win_w - 40, justify=tk.LEFT
        )
        self._strip_label.pack(anchor='w')

        self.logo       = LogoPlayer(root, on_tap=self._show_shutdown_overlay)
        self.hand_panel = HandViewPanel(root)
        self.hand_btn   = CircularHandButton(root, self.toggle_panel)

        # ── Updates button (top-centre) ───────────────────────────────────────
        # ── Updates button (top-centre, hidden until updates found) ────────────
        self._upd_btn_cv = tk.Canvas(root, width=130, height=54,
                                     bg='white', highlightthickness=0,
                                     cursor='hand2')
        self._upd_btn_cv.bind('<ButtonPress-1>', lambda e: self._show_updates())
        self._upd_btn_cv.place_forget()   # hidden until check finds updates

        try:
            import piupdater as _pu
            _pu.start_check(APP_VERSION,
                            on_done=lambda: root.after(0, self._refresh_upd_btn))
        except Exception as _ue:
            print(f"piupdater unavailable: {_ue}")

        try:
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
            from asl_settings import SettingsDrawer
            self._settings = SettingsDrawer(
                root,
                on_font_change=self._apply_font,
                on_volume_change=self._apply_volume,
            )
        except Exception as e:
            print(f"Settings drawer unavailable: {e}")
            self._settings = None

        self._settings_btn = tk.Canvas(root, width=54, height=54,
                                       bg='white', highlightthickness=0,
                                       cursor='hand2')
        self._settings_btn.place(x=WIN_W - 54 - 16 - 54 - 8, y=16)
        self._s_pressed  = False
        self._s_active   = False
        self._draw_settings_btn(False)
        self._settings_btn.bind('<ButtonPress-1>',   self._settings_press)
        self._settings_btn.bind('<ButtonRelease-1>', self._settings_release)
        self._settings_btn.bind('<Leave>', lambda e: (
            setattr(self, '_s_pressed', False), self._draw_settings_btn(False)))

        root.bind('<Configure>', self._on_configure)

        root.after(100, lambda: self._apply_font(
            self._font_family, self._font_size,
            self._font_bold, self._font_italic))

        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ── Dynamic sentence sizing ──────────────────────────────────────────────
    def _wrap_text(self, text, font, max_w):
        """Simulate tkinter word-wrap; return list of line word-lists."""
        words = text.split()
        lines, line, line_w = [], [], 0
        sp = font.measure(' ')
        for word in words:
            w = font.measure(word)
            needed = (sp + w) if line else w
            if line and line_w + needed > max_w:
                lines.append(line)
                line, line_w = [word], w
            else:
                line.append(word)
                line_w += needed
        if line:
            lines.append(line)
        return lines or [['']]

    def _fit_sentence(self, text):
        """Scale sentence_label to the largest font where all text fits inside
        the grey bar with a small margin on every edge."""
        if not text:
            self.sentence_label.config(text='')
            self._prev_sentence = ''
            return

        bw = self.bottom_frame.winfo_width()
        bh = self.bottom_frame.winfo_height()
        if bw < 10: bw = getattr(self, '_win_w', 800)
        if bh < 10: bh = 180

        avail_w = bw - 32   # 16 px each side
        avail_h = bh - 20   # 10 px top + bottom

        fam   = getattr(self, '_font_family', 'Helvetica')
        slant = 'italic' if getattr(self, '_font_italic', False) else 'roman'

        best_size = 10
        for size in range(52, 9, -1):
            font = tkfont.Font(family=fam, size=size, slant=slant)
            n_lines = len(self._wrap_text(text, font, avail_w))
            # linespace includes leading; add a 6% safety margin
            if n_lines * font.metrics('linespace') <= avail_h * 0.94:
                best_size = size
                break

        font = tkfont.Font(family=fam, size=best_size, slant=slant)
        self.sentence_label.config(
            text=text,
            font=font,
            wraplength=avail_w,
        )
        self._prev_sentence = text

    def _on_configure(self, event):
        if event.widget != self.root:
            return
        w, h = event.width, event.height
        if w == self._win_w and h == self._win_h:
            return
        self._win_w = w
        self._win_h = h
        self._relayout(w, h, self._cur_shift)
        cur = self.sentence_label.cget('text')
        if cur:
            self.root.after(10, lambda: self._fit_sentence(cur))

    def _relayout(self, w, h, shift):
        avail = max(200, w - shift)
        wl = int(avail * 0.62)
        self.word_label.config(wraplength=wl)
        max_pills = max(1, (avail - 2*self.PILL_PAD + self.PILL_GAP) //
                           (self.PILL_W + self.PILL_GAP))
        max_pills = min(max_pills, 20)
        self.capsule.set_max_visible(max_pills)
        self.hand_btn.on_resize(w, h)
        self.hand_panel.on_resize(w, h)
        if self._settings:
            self._settings.on_resize(w, h)
        self._settings_btn.place(x=w - 54 - 16 - 54 - 8, y=16)

    def _animate_shift(self, target, steps=10):   # 10 steps x 12ms = 120ms
        if self._shift_job:
            self.root.after_cancel(self._shift_job)
        start      = self._cur_shift
        fam        = self._font_family
        bold       = self._font_bold
        max_target = max(1, target if target > 0 else self._win_h // 2)
        weight     = "bold"   if bold              else "normal"
        slant      = "italic" if self._font_italic else "roman"
        _fc        = {}   # font cache — reuse Font objects across ticks

        def _f(sz, use_weight=True):
            key = (sz, use_weight)
            if key not in _fc:
                w = weight if use_weight else 'normal'
                _fc[key] = tkfont.Font(family=fam, size=sz, weight=w, slant=slant)
            return _fc[key]

        def _tick(i):
            t   = i / steps
            t   = 1 - (1 - t) ** 3
            pad = int(start + (target - start) * t)
            self._cur_shift = pad
            frac    = pad / max_target
            top_pad = max(8,  int(50 * (1.0 - frac * 0.7)))
            lf_pad  = max(1,  int(2  * (1.0 - frac * 0.5)))
            word_sz = max(22, int(self._font_size * (1.0 - frac * 0.45)))
            big_sz  = max(36, int(min(80, self._font_size*2) * (1.0 - frac * 0.50)))
            conf_sz = max(9,  int(13 * (1.0 - frac * 0.40)))
            self.word_label.pack_configure(pady=(top_pad, 4))
            self.word_label.config(font=_f(word_sz))
            self._lf.pack_configure(pady=(lf_pad, lf_pad))
            self.big_letter.config(font=_f(big_sz))
            self.confirmed_label.config(font=_f(conf_sz, use_weight=False))
            self._cf.pack_configure(pady=(4, 2))
            self._relayout(self._win_w, self._win_h, 0)
            if i < steps:
                self._shift_job = self.root.after(12, lambda: _tick(i+1))
            else:
                self._shift_job = None
        _tick(0)

    def toggle_panel(self, open_panel: bool):
        self._panel_open  = open_panel
        self._drawer_open = False
        if open_panel:
            self.bottom_frame.pack_forget()
            self._strip_label.config(text=self.sentence_label.cget("text"))
            self._strip_frame.pack(fill=tk.X, padx=12, pady=(4, 4))
            self.hand_panel.show()
            self._animate_shift(_panel_h(self._win_h))
        else:
            self._strip_frame.pack_forget()
            self.bottom_frame.pack(fill=tk.BOTH, expand=True)
            self.hand_panel.hide()
            self._animate_shift(0)

    def update_hand_panel(self, crop_bgr, landmark_bgr):
        if not self._panel_open:
            return
        # Throttle to ~4fps — PIL resize + ImageTk.PhotoImage every 80ms is expensive
        self._panel_tick += 1
        if self._panel_tick % 3 != 0:
            return
        self.hand_panel.update(crop_bgr, landmark_bgr)

    # ── Bullseye radiating effect ─────────────────────────────────────────────
    def _trigger_bullseye(self):
        for j in self._bullseye_jobs:
            try: self.root.after_cancel(j)
            except: pass
        self._bullseye_jobs = []

        MAX_R    = 170
        BLUR     = 14      # reduced for performance
        SIZE     = (MAX_R + BLUR + 20) * 2
        RINGS    = 3       # reduced for performance
        STEPS    = 22
        INTERVAL = 15
        GREEN    = (0, 220, 55)
        BG       = (255, 255, 255)

        try:
            self.root.update_idletasks()
            lx = self._lf.winfo_x() + self.big_letter.winfo_x() \
                 + self.big_letter.winfo_width()  // 2
            ly = self._lf.winfo_y() + self.big_letter.winfo_y() \
                 + self.big_letter.winfo_height() // 2
        except Exception:
            lx = self._win_w // 2
            ly = self._win_h // 3

        if not hasattr(self, '_be_cv') or not self._be_cv.winfo_exists():
            self._be_cv = tk.Canvas(self.top_frame,
                                    highlightthickness=0, bg='white')
        self._be_cv.config(width=SIZE, height=SIZE)
        self._be_cv.place(x=lx - SIZE//2, y=ly - SIZE//2)
        self._be_cv.delete('all')
        tk.Misc.lower(self._be_cv)

        cx = cy = SIZE // 2

        def _spring(t):
            base  = 1.0 - (1.0 - t) ** 3
            bump  = 0.10 * math.sin(t * math.pi) * math.exp(-t * 4)
            return min(1.15, base + bump)

        def _col(intensity):
            a  = max(0.0, min(1.0, intensity))
            rc = int(GREEN[0] * a + BG[0] * (1 - a))
            gc = int(GREEN[1] * a + BG[1] * (1 - a))
            bc = int(GREEN[2] * a + BG[2] * (1 - a))
            return f'#{rc:02x}{gc:02x}{bc:02x}'

        def _draw_ring(canvas, cx, cy, radius, peak_intensity, blur):
            for k in range(-blur, blur + 1, 2):
                r = radius + k
                if r < 1: continue
                gw = math.exp(-0.5 * (k / (blur * 0.42)) ** 2)
                canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                   outline=_col(peak_intensity * gw), width=3)

        def _step(ring, i):
            try:
                if not self._be_cv.winfo_exists(): return
            except: return
            t         = i / STEPS
            radius    = int(_spring(t) * MAX_R)
            rise      = 1.0 - math.exp(-t * 18)
            fall      = math.exp(-2.8 * t)
            intensity = max(0.0, min(1.0, rise * fall * 1.4))
            tag = f'ring{ring}'
            self._be_cv.delete(tag)
            if radius > 0 and intensity > 0.015:
                items_before = set(self._be_cv.find_all())
                _draw_ring(self._be_cv, cx, cy, radius, intensity, BLUR)
                for item in self._be_cv.find_all():
                    if item not in items_before:
                        self._be_cv.itemconfig(item, tags=tag)
            if i < STEPS:
                j = self.root.after(INTERVAL, lambda r=ring, ii=i+1: _step(r, ii))
                self._bullseye_jobs.append(j)
            else:
                try: self._be_cv.delete(tag)
                except: pass

        for ring in range(RINGS):
            j = self.root.after(ring * 5 * INTERVAL, lambda r=ring: _step(r, 0))
            self._bullseye_jobs.append(j)

        total = (RINGS * 5 + STEPS) * INTERVAL + 250
        def _cleanup():
            try: self._be_cv.delete('all')
            except: pass
        j = self.root.after(total, _cleanup)
        self._bullseye_jobs.append(j)

    # ── Letter bounce ─────────────────────────────────────────────────────────
    def _trigger_letter_bounce(self):
        if hasattr(self, '_bounce_job') and self._bounce_job:
            try: self.root.after_cancel(self._bounce_job)
            except: pass
        self._bounce_job = None

        fam    = self._font_family
        bold   = self._font_bold
        slant  = "italic" if self._font_italic else "roman"
        weight = "bold" if bold else "normal"
        base   = min(80, self._font_size * 2)
        GREEN  = (0, 220, 55)

        # Font cache — avoids creating a new Font object every 18ms tick
        _font_cache = {}
        def _get_font(sz):
            if sz not in _font_cache:
                _font_cache[sz] = tkfont.Font(family=fam, size=sz,
                                              weight=weight, slant=slant)
            return _font_cache[sz]

        pos = [1.45]; vel = [0.0]
        gpos = [1.0];  gvel = [0.0]

        def _tick():
            f       = (1.0 - pos[0]) * 0.38
            vel[0]  = vel[0] * 0.62 + f
            pos[0] += vel[0]
            gf      = (0.0 - gpos[0]) * 0.28
            gvel[0] = gvel[0] * 0.58 + gf
            gpos[0] = max(0.0, gpos[0] + gvel[0])
            sz  = max(20, int(base * max(0.5, pos[0])))
            a   = max(0.0, min(1.0, gpos[0]))
            rc  = int(GREEN[0] * a)
            gc  = int(GREEN[1] * a)
            bc  = int(GREEN[2] * a)
            col = f'#{rc:02x}{gc:02x}{bc:02x}' if a > 0.04 else 'black'
            try:
                self.big_letter.config(font=_get_font(sz), fg=col)
            except Exception:
                return
            settled = (abs(pos[0] - 1.0) < 0.004
                       and abs(vel[0])    < 0.004
                       and gpos[0]        < 0.01)
            if not settled:
                self._bounce_job = self.root.after(14, _tick)   # 14ms — snappier spring
            else:
                self.big_letter.config(font=_get_font(base), fg='black')
                self._bounce_job = None

        self._bounce_job = self.root.after(0, _tick)

    # ── Green glow ────────────────────────────────────────────────────────────
    def _do_glow(self, step=0):
        if step < len(self._GLOW):
            self.big_letter.config(fg=self._GLOW[step])
            self._glow_job = self.root.after(self._GLOW_MS, lambda: self._do_glow(step + 1))
        else:
            self._glow_job = None

    def _trigger_glow(self):
        if self._glow_job:
            self.root.after_cancel(self._glow_job)
        self._do_glow(0)

    def set_out_of_frame(self, is_oof: bool):
        if is_oof == self._oof_state:
            return   # no change — skip pack/unpack overhead
        self._oof_state = is_oof
        if is_oof:
            self._oof_label.pack(after=self._cf, pady=(2, 0))
        else:
            self._oof_label.pack_forget()

    def update_display(self, last_word, last_letter, full_sentence,
                       raw_letters_list, new_letter_confirmed=False,
                       out_of_frame=False):
        # Dirty flags — avoids redundant Tcl calls at 12.5 Hz
        if last_word != self._prev_word:
            self._prev_word = last_word
            self.word_label.config(text=last_word)
        if full_sentence != self._prev_sentence:
            self._fit_sentence(full_sentence)
            if self._panel_open:
                self._strip_label.config(text=full_sentence)
        if last_letter != self._prev_big:
            self._prev_big = last_letter
            self.big_letter.config(text=last_letter.upper() if last_letter else "")
        if new_letter_confirmed:
            self._trigger_glow()
            self._trigger_bullseye()
            self._trigger_letter_bounce()
        conf = f"confirmed: {last_letter.upper()}" if last_letter else ""
        if conf != self._prev_conf:
            self._prev_conf = conf
            self.confirmed_label.config(text=conf)
        self.capsule.set_letters(raw_letters_list)
        self.set_out_of_frame(out_of_frame)

    def _draw_settings_btn(self, pressed):
        self._settings_btn.delete("all")
        s = 54
        if pressed or self._s_active:
            bg, fg = '#1A1A1A', 'white'
        else:
            bg, fg = '#FFFFFF', '#808080'
        self._settings_btn.create_oval(2, 2, s-2, s-2,
                                       fill=bg, outline='#DDDDDD', width=1.5)
        self._settings_btn.create_text(s//2, s//2, text="\u2699",
                                       font=("Noto Color Emoji", 20), fill=fg)

    def _settings_press(self, _):
        self._s_pressed = True
        self._draw_settings_btn(True)

    def _settings_release(self, _):
        self._s_pressed = False
        self._s_active  = not self._s_active
        self._draw_settings_btn(False)
        if self._settings:
            self._settings.toggle()
            if self._settings.visible:
                self._animate_shift(self._settings.H)
            else:
                self._animate_shift(0)

    def _apply_font(self, family, size, bold, italic):
        self._font_family = family
        self._font_size   = size
        self._font_bold   = bold
        self._font_italic = italic
        weight = "bold"   if bold   else "normal"
        slant  = "italic" if italic else "roman"
        try:
            self.word_label.config(
                font=tkfont.Font(family=family, size=size, weight=weight, slant=slant))
            self.big_letter.config(
                font=tkfont.Font(family=family, size=min(80, size*2), weight=weight, slant=slant))
            self.confirmed_label.config(
                font=tkfont.Font(family=family, size=max(9, size-5), slant=slant))
            self._font_family = family
            self._font_italic = (slant == 'italic')
            self._fit_sentence(self.sentence_label.cget('text'))
            self._strip_label.config(
                font=tkfont.Font(family=family, size=max(14, size), slant=slant))
            self.capsule.set_font(family, italic)
        except Exception as e:
            print(f"Font apply error: {e}")

    def _apply_volume(self, pct):
        try:
            from asl_settings import set_volume
            set_volume(pct)
        except Exception:
            pass

    # ── Shutdown overlay ──────────────────────────────────────────────────────
    def _show_shutdown_overlay(self):
        if hasattr(self, '_shutdown_overlay') and self._shutdown_overlay:
            try:
                if self._shutdown_overlay.winfo_exists():
                    return
            except: pass

        ow, oh = self._win_w, self._win_h

        # Full-screen event blocker sits beneath the overlay.
        # When the overlay closes, this briefly stays alive to absorb any
        # residual touch-release events that would otherwise hit the buttons below.
        blocker = tk.Canvas(self.root, bg='white', highlightthickness=0)
        blocker.bind('<ButtonPress-1>',   lambda e: 'break')
        blocker.bind('<ButtonRelease-1>', lambda e: 'break')
        blocker.place(x=0, y=0, width=ow, height=oh)

        ov = tk.Frame(self.root, bg='white')
        ov.place(x=0, y=0, width=ow, height=oh)
        ov.bind('<ButtonPress-1>',   lambda e: 'break')
        ov.bind('<ButtonRelease-1>', lambda e: 'break')
        tk.Misc.lift(ov)
        ov.update()
        self._shutdown_overlay = ov

        logo_anim_job = [None]

        def _cancel():
            if logo_anim_job[0]:
                try: ov.after_cancel(logo_anim_job[0])
                except: pass
            def _on_faded():
                ov.place_forget()
                def _destroy():
                    self._shutdown_overlay = None
                    try: blocker.destroy()
                    except: pass
                    try: ov.destroy()
                    except: pass
                ov.after(300, _destroy)
            self._fade_close(_on_faded)

        # ── Dark top bar — drawn as a single Canvas so corners are correct ─────
        # Child widgets placed on a Canvas always render ABOVE canvas items,
        # so no lower()/lift() tricks needed.
        BAR_H = 68
        R     = 20
        bar_cv = tk.Canvas(ov, height=BAR_H, bg='white', highlightthickness=0)
        bar_cv.pack(fill=tk.X)

        def _draw_sd_bar(e=None):
            bar_cv.delete('bg')
            w = bar_cv.winfo_width()
            if w < 10: return
            dc = '#1D1D1F'
            # Full rectangle above the curve zone
            bar_cv.create_rectangle(0, 0, w, BAR_H-R,   fill=dc, outline='', tags='bg')
            # Bottom middle strip between the two arcs
            bar_cv.create_rectangle(R, BAR_H-R, w-R, BAR_H, fill=dc, outline='', tags='bg')
            # Bottom-left arc: center=(R, BAR_H-R), CCW west→south = SW quadrant
            bar_cv.create_arc(0, BAR_H-2*R, 2*R, BAR_H,
                              start=180, extent=90,
                              fill=dc, outline=dc, style='pieslice', tags='bg')
            # Bottom-right arc: center=(w-R, BAR_H-R), CCW south→east = SE quadrant
            bar_cv.create_arc(w-2*R, BAR_H-2*R, w, BAR_H,
                              start=270, extent=90,
                              fill=dc, outline=dc, style='pieslice', tags='bg')
            bar_cv.tag_lower('bg')   # keep drawn shapes behind placed children

        bar_cv.bind('<Configure>', lambda e: _draw_sd_bar())
        bar_cv.after(50, _draw_sd_bar)

        # Title and Back placed on bar_cv — placed children always above canvas items
        tk.Label(bar_cv, text='Power Off', bg='#1D1D1F', fg='white',
                 font=tkfont.Font(family='Helvetica', size=18, weight='bold')
                 ).place(x=20, rely=0.5, anchor='w')

        bw, bh = 110, 44
        back = tk.Canvas(bar_cv, width=bw, height=bh,
                         bg='#1D1D1F', highlightthickness=0, cursor='hand2')
        br = bh // 2
        back.create_oval(0, 0, bh, bh, fill='white', outline='')
        back.create_rectangle(br, 0, bw-br, bh, fill='white', outline='')
        back.create_oval(bw-bh, 0, bw, bh, fill='white', outline='')
        back.create_text(bw//2, bh//2, text='Back', fill='#1D1D1F',
                         font=tkfont.Font(family='Helvetica', size=15, weight='bold'))
        back.place(relx=1.0, rely=0.5, anchor='e', x=-16)
        back.bind('<ButtonPress-1>', lambda e: _cancel())

        # ── Body ─────────────────────────────────────────────────────────────
        body = tk.Frame(ov, bg='white')
        body.pack(fill=tk.BOTH, expand=True)

        # ── Animated logo (unchanged from original) ───────────────────────────
        LOGO_SZ = 200
        logo_cv = tk.Canvas(body, width=LOGO_SZ, height=LOGO_SZ,
                             bg='white', highlightthickness=0)
        logo_cv.place(x=ow//2 - LOGO_SZ//2, y=40)
        img_item = logo_cv.create_image(0, 0, anchor='nw')

        _cache = {}
        def _get(idx):
            if idx not in _cache:
                pil = self.logo._pil_normal[idx % len(self.logo._pil_normal)]
                if pil.size[0] != LOGO_SZ:
                    pil = pil.resize((LOGO_SZ, LOGO_SZ), Image.LANCZOS)
                # Composite on grey to match body background
                mask = Image.new('L', (LOGO_SZ, LOGO_SZ), 0)
                ImageDraw.Draw(mask).ellipse((0, 0, LOGO_SZ-1, LOGO_SZ-1), fill=255)
                rgba = pil.convert('RGBA'); rgba.putalpha(mask)
                bg_  = Image.new('RGBA', (LOGO_SZ, LOGO_SZ), (255, 255, 255, 255))
                bg_.paste(rgba, mask=rgba.split()[3])
                _cache[idx] = ImageTk.PhotoImage(bg_.convert('RGB'))
            return _cache[idx]

        frame_state = [0]

        def _logo_tick():
            if not ov.winfo_exists(): return
            logo_cv.itemconfig(img_item, image=_get(frame_state[0]))
            frame_state[0] = (frame_state[0] + 1) % len(self.logo._pil_normal)
            logo_anim_job[0] = ov.after(50, _logo_tick)

        # Spring bounce on appear (original unchanged)
        sc = [1.4]; sv = [0.0]
        def _bounce():
            f    = (1.0 - sc[0]) * 0.30
            sv[0]= sv[0] * 0.65 + f
            sc[0]+= sv[0]
            sz   = max(80, int(LOGO_SZ * sc[0]))
            logo_cv.config(width=sz, height=sz)
            logo_cv.place(x=ow//2 - sz//2, y=max(10, 40 - (sz-LOGO_SZ)//2))
            if abs(sc[0]-1.0) > 0.005 or abs(sv[0]) > 0.005:
                ov.after(14, _bounce)
        _bounce()
        _logo_tick()

        # ── Goodbye phrase (original, grey background) ────────────────────────
        phrase = random.choice(GOODBYE_PHRASES)
        tk.Label(body, text=phrase, bg='white', fg='#1A1A1A',
                 font=tkfont.Font(family='Helvetica', size=28, weight='bold')
                 ).place(relx=0.5, y=265, anchor='center')
        tk.Label(body, text="Power off the device?", bg='white', fg='#555555',
                 font=tkfont.Font(family='Helvetica', size=20, weight='bold')
                 ).place(relx=0.5, y=316, anchor='center')

        # ── Slide to shutdown (original, unchanged) ───────────────────────────
        BAR_W = min(480, ow - 80)

        def _do_shutdown():
            if logo_anim_job[0]:
                try: ov.after_cancel(logo_anim_job[0])
                except: pass
            tk.Label(body, text='Powering off…', bg='white', fg='#888888',
                     font=tkfont.Font(family='Helvetica', size=18)
                     ).place(relx=0.5, rely=0.75, anchor='center')
            try: ov.update()
            except: pass
            import subprocess
            self.root.after(900, lambda: (
                subprocess.Popen(["sudo", "shutdown", "-h", "now"]),
                self.on_closing()
            ))

        sbar = SlideToShutdown(body, width=BAR_W,
                               on_confirm=_do_shutdown, on_cancel=_cancel)
        sbar.place(relx=0.5, y=oh - 220, anchor='center')

    def _refresh_upd_btn(self):
        try:
            import piupdater as _pu
            if _pu.total() > 0:
                self._draw_upd_btn()
                self._upd_btn_cv.place(relx=0.5, y=16, anchor='n')
            else:
                self._upd_btn_cv.place_forget()
        except Exception:
            pass

    def _draw_upd_btn(self):
        c = self._upd_btn_cv
        c.delete('all')
        W, H = 130, 54
        R = H // 2
        c.create_oval(0, 0, H, H, fill='#1D1D1F', outline='')
        c.create_rectangle(R, 0, W-R, H, fill='#1D1D1F', outline='')
        c.create_oval(W-H, 0, W, H, fill='#1D1D1F', outline='')
        c.create_text(W//2, H//2, text='Updates', fill='white',
                      font=tkfont.Font(family='Helvetica', size=14, weight='bold'))

    def _show_updates(self):
        import piupdater as _pu
        ow, oh = self._win_w, self._win_h

        blocker = tk.Canvas(self.root, bg='white', highlightthickness=0)
        blocker.bind('<ButtonPress-1>',   lambda e: 'break')
        blocker.bind('<ButtonRelease-1>', lambda e: 'break')
        blocker.place(x=0, y=0, width=ow, height=oh)

        ov = tk.Frame(self.root, bg='white')
        ov.place(x=0, y=0, width=ow, height=oh)
        ov.bind('<ButtonPress-1>',   lambda e: 'break')
        ov.bind('<ButtonRelease-1>', lambda e: 'break')
        tk.Misc.lift(ov)
        ov.update()

        def _close():
            def _on_faded():
                ov.place_forget()
                def _destroy():
                    try: blocker.destroy()
                    except: pass
                    try: ov.destroy()
                    except: pass
                    self._refresh_upd_btn()
                ov.after(300, _destroy)
            self._fade_close(_on_faded)

        # ── Dark top bar as a single Canvas — child widgets always above canvas items ──
        BAR_H = 68
        R     = 20
        bar_cv = tk.Canvas(ov, height=BAR_H, bg='white', highlightthickness=0)
        bar_cv.pack(fill=tk.X)

        def _draw_upd_bar(e=None):
            bar_cv.delete('bg')
            w = bar_cv.winfo_width()
            if w < 10: return
            dc = '#1D1D1F'
            bar_cv.create_rectangle(0, 0, w, BAR_H-R,        fill=dc, outline='', tags='bg')
            bar_cv.create_rectangle(R, BAR_H-R, w-R, BAR_H,  fill=dc, outline='', tags='bg')
            # Bottom-left: CCW west→south = SW quadrant, center=(R, BAR_H-R)
            bar_cv.create_arc(0, BAR_H-2*R, 2*R, BAR_H,
                              start=180, extent=90,
                              fill=dc, outline=dc, style='pieslice', tags='bg')
            # Bottom-right: CCW south→east = SE quadrant, center=(w-R, BAR_H-R)
            bar_cv.create_arc(w-2*R, BAR_H-2*R, w, BAR_H,
                              start=270, extent=90,
                              fill=dc, outline=dc, style='pieslice', tags='bg')
            bar_cv.tag_lower('bg')

        bar_cv.bind('<Configure>', lambda e: _draw_upd_bar())
        bar_cv.after(50, _draw_upd_bar)

        tk.Label(bar_cv, text='Updates', bg='#1D1D1F', fg='white',
                 font=tkfont.Font(family='Helvetica', size=18, weight='bold')
                 ).place(x=20, rely=0.5, anchor='w')

        bw, bh = 110, 44
        back = tk.Canvas(bar_cv, width=bw, height=bh,
                         bg='#1D1D1F', highlightthickness=0, cursor='hand2')
        br = bh // 2
        back.create_oval(0, 0, bh, bh, fill='white', outline='')
        back.create_rectangle(br, 0, bw-br, bh, fill='white', outline='')
        back.create_oval(bw-bh, 0, bw, bh, fill='white', outline='')
        back.create_text(bw//2, bh//2, text='Back', fill='#1D1D1F',
                         font=tkfont.Font(family='Helvetica', size=15, weight='bold'))
        back.place(relx=1.0, rely=0.5, anchor='e', x=-16)
        back.bind('<ButtonPress-1>', lambda e: _close())
        ov.bind('<Escape>', lambda e: _close())

        # Body
        body = tk.Frame(ov, bg='white')
        body.pack(fill=tk.BOTH, expand=True, padx=28, pady=20)



        def _card(parent):
            outer = tk.Frame(parent, bg='white',
                             highlightthickness=1,
                             highlightbackground='#E5E5EA')
            outer.pack(fill=tk.X, pady=(0, 16))
            inner = tk.Frame(outer, bg='white')
            inner.pack(fill=tk.X, padx=20, pady=18)
            return inner

        def _pill(parent, label, bg, fg, cmd):
            PW, PH = ow - 160, 52
            c = tk.Canvas(parent, width=PW, height=PH,
                          bg='white', highlightthickness=0, cursor='hand2')
            PR = PH // 2
            c.create_oval(0, 0, PH, PH, fill=bg, outline='')
            c.create_rectangle(PR, 0, PW-PR, PH, fill=bg, outline='')
            c.create_oval(PW-PH, 0, PW, PH, fill=bg, outline='')
            c.create_text(PW//2, PH//2, text=label, fill=fg,
                          font=tkfont.Font(family='Helvetica', size=15,
                                           weight='bold'))
            c.bind('<ButtonPress-1>', lambda e: cmd())
            return c

        if not _pu.checked[0]:
            tk.Label(body, text='Checking for updates…',
                     bg='white', fg='#6E6E73',
                     font=tkfont.Font(family='Helvetica', size=16)
                     ).pack(pady=60)

        elif _pu.total() == 0:
            c = _card(body)
            tk.Label(c, text='✓  Everything is up to date',
                     bg='white', fg='#34C759',
                     font=tkfont.Font(family='Helvetica', size=16, weight='bold')
                     ).pack(anchor='w', pady=(0, 4))
            tk.Label(c, text=f'App v{APP_VERSION}  ·  No system updates pending',
                     bg='white', fg='#6E6E73',
                     font=tkfont.Font(family='Helvetica', size=13)
                     ).pack(anchor='w')

        else:
            # Shared reboot screen — used by both system and app updates
            def _do_reboot():
                rbt = tk.Frame(self.root, bg='white')
                rbt.place(x=0, y=0, relwidth=1.0, relheight=1.0)
                tk.Misc.lift(rbt)
                tk.Label(rbt,
                         text='Rebooting…',
                         bg='white', fg='#1D1D1F',
                         font=tkfont.Font(family='Helvetica', size=30, weight='bold')
                         ).place(relx=0.5, rely=0.42, anchor='center')
                tk.Label(rbt,
                         text='This will take about 30 seconds.',
                         bg='white', fg='#6E6E73',
                         font=tkfont.Font(family='Helvetica', size=16)
                         ).place(relx=0.5, rely=0.56, anchor='center')
                self.root.update()
                self.root.after(800,
                    lambda: __import__('subprocess').Popen(
                        ['sudo', 'shutdown', '-r', 'now']))

            # System updates card
            if _pu.system_count[0] > 0:
                c = _card(body)
                n = _pu.system_count[0]
                tk.Label(c, text='System Updates', bg='white', fg='#1D1D1F',
                         font=tkfont.Font(family='Helvetica', size=16,
                                          weight='bold')
                         ).pack(anchor='w', pady=(0, 4))
                tk.Label(c,
                         text=f'{n} package{"s" if n!=1 else ""} available',
                         bg='white', fg='#6E6E73',
                         font=tkfont.Font(family='Helvetica', size=13)
                         ).pack(anchor='w', pady=(0, 14))
                sys_lbl = tk.Label(c, text='', bg='white', fg='#6E6E73',
                                   font=tkfont.Font(family='Helvetica', size=12))
                sys_lbl.pack(anchor='w', pady=(0, 8))
                sys_btn = _pill(c, 'Install System Updates',
                                '#1D1D1F', 'white',
                                lambda: None)
                sys_btn.pack()

                def _do_system():
                    sys_btn.config(state=tk.DISABLED)
                    _pu.install_system(
                        on_status=lambda m, col: self.root.after(
                            0, lambda: sys_lbl.config(text=m, fg=col or '#6E6E73')),
                        on_restart=lambda: self.root.after(0, _do_reboot))

                sys_btn.bind('<ButtonPress-1>', lambda e: _do_system())

            # App update card
            if _pu.app_available[0]:
                c = _card(body)
                tk.Label(c, text='App Update', bg='white', fg='#1D1D1F',
                         font=tkfont.Font(family='Helvetica', size=16,
                                          weight='bold')
                         ).pack(anchor='w', pady=(0, 4))
                tk.Label(c,
                         text=(f'Version {_pu.app_version[0]} available'
                               f'  (current: {APP_VERSION})'),
                         bg='white', fg='#6E6E73',
                         font=tkfont.Font(family='Helvetica', size=13)
                         ).pack(anchor='w', pady=(0, 14))
                app_lbl = tk.Label(c, text='', bg='white', fg='#6E6E73',
                                   font=tkfont.Font(family='Helvetica', size=12))
                app_lbl.pack(anchor='w', pady=(0, 8))
                app_btn = _pill(c, 'Download & Install',
                                '#0A84FF', 'white', lambda: None)
                app_btn.pack()

                def _do_app():
                    app_btn.config(state=tk.DISABLED)
                    _pu.install_app(
                        on_status=lambda m, col: self.root.after(
                            0, lambda: app_lbl.config(text=m, fg=col or '#6E6E73')),
                        on_restart=lambda: self.root.after(0, _do_reboot))

                app_btn.bind('<ButtonPress-1>', lambda e: _do_app())

    def _fade_close(self, on_done):
        """Fade root window out, call on_done, fade back in."""
        alphas_out = [0.85, 0.65, 0.40, 0.15, 0.0]
        alphas_in  = [0.2,  0.5,  0.8,  1.0]
        def _step_out(i=0):
            if i >= len(alphas_out):
                on_done()
                def _step_in(j=0):
                    if j >= len(alphas_in):
                        return
                    try: self.root.attributes('-alpha', alphas_in[j])
                    except: pass
                    self.root.after(15, lambda: _step_in(j+1))
                _step_in()
                return
            try: self.root.attributes('-alpha', alphas_out[i])
            except: pass
            self.root.after(20, lambda: _step_out(i+1))
        try:
            _step_out()
        except Exception:
            on_done()   # fallback: instant

    def trigger_recog(self):
        self.logo.trigger_recog()

    def show_llm_error(self):
        """Briefly flash a subtle 'connection lost' note in the sentence bar so a
        failed query isn't silent. Auto-clears; does not touch letters/history."""
        try:
            prev = self.sentence_label.cget('text')
            self.sentence_label.config(text="\u26a0 connection lost \u2014 will retry",
                                       fg='#E8A020')
            def _restore():
                try:
                    # Only restore if we didn't get a real translation meanwhile
                    if self.sentence_label.cget('text').startswith("\u26a0"):
                        self.sentence_label.config(fg='white')
                        self._fit_sentence(prev)
                except Exception:
                    pass
            self.root.after(2200, _restore)
        except Exception:
            pass

    def on_closing(self):
        self.running = False
        self.logo.destroy()
        self.root.destroy()



def _welcome_back(root, user_name):
    """
    Play welcome.mov at 2x speed with user name centred below.
    Fades out when the video ends, then proceeds to the main app.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    sw = root.winfo_screenwidth()  or WIN_W
    sh = root.winfo_screenheight() or WIN_H

    # ── Load and resize welcome.mov frames ────────────────────────────────────
    VID_MAX_W = sw - 80
    VID_MAX_H = int(sh * 0.62)
    vfps      = 30.0
    vid_tk    = []
    disp_w    = VID_MAX_W
    disp_h    = VID_MAX_H

    try:
        vc = cv2.VideoCapture(os.path.join(base_dir, 'welcome.mov'))
        fps = vc.get(cv2.CAP_PROP_FPS)
        if fps and 0 < fps < 240:
            vfps = float(fps)
        raw = []
        while True:
            ret, frm = vc.read()
            if not ret: break
            raw.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        vc.release()
        if raw:
            fh, fw = raw[0].shape[:2]
            sc     = min(VID_MAX_W / fw, VID_MAX_H / fh, 1.0)
            disp_w = max(1, int(fw * sc))
            disp_h = max(1, int(fh * sc))
            for frm in raw:
                pil = Image.fromarray(frm).resize((disp_w, disp_h), Image.LANCZOS)
                vid_tk.append(ImageTk.PhotoImage(pil))
    except Exception as ex:
        print(f"welcome_back video: {ex}")

    # ── Layout: video + name stacked, centred on screen ───────────────────────
    wrap = tk.Frame(root, bg='white')
    wrap.place(relx=0.5, rely=0.5, anchor='center')

    vid_lbl = tk.Label(wrap, bg='white')
    vid_lbl.pack()

    tk.Label(wrap, text=user_name, bg='white', fg='#1D1D1F',
             font=tkfont.Font(family='Helvetica', size=28, weight='bold')
             ).pack(pady=(10, 0))

    root.update()

    # ── Play at 2x speed ──────────────────────────────────────────────────────
    vdelay_2x = max(8, int(1000 / (vfps * 2))) / 1000.0   # seconds per frame
    for photo in vid_tk:
        vid_lbl.config(image=photo)
        vid_lbl.image = photo
        root.update()
        time.sleep(vdelay_2x)

    # ── Fade out over ~1.5 s ──────────────────────────────────────────────────
    STEPS = 45
    FADE  = 1.5
    for i in range(STEPS + 1):
        try: root.attributes('-alpha', 1.0 - i / STEPS)
        except: pass
        root.update()
        time.sleep(FADE / STEPS)

    try: root.attributes('-alpha', 1.0)
    except: pass

    for w in root.winfo_children():
        try: w.destroy()
        except: pass
    root.configure(bg='white')
    root.update()

# ── First-boot registration + welcome screen ──────────────────────────────────
def _run_first_boot(root) -> tuple:
    """
    First-boot experience. Returns (name, email).
    Phase 1   : Logo animation + step dot 1 + name field + OSK
    Transition: 'Hi [name]!' spring in → slide p1 left / p2 right
    Phase 2   : Logo + step dot 2 + email field + OSK (with '.') + preloads welcome.mov
    Phase 3   : white cut → welcome.mov → welcome_img.png → Get started
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for w in root.winfo_children():
        w.destroy()
    root.configure(bg='white')
    root.update()

    sw = root.winfo_screenwidth()  or WIN_W
    sh = root.winfo_screenheight() or WIN_H

    result = {'name': '', 'email': ''}
    phase  = [1]   # 1=name  2=email  3=welcome  0=done

    # ── OSK palette matches asl_settings.py exactly ───────────────────────────
    _BG = '#FFFFFF'; _MID = '#E8E8E8'; _PANEL = '#F5F5F5'
    _DARK = '#1A1A1A'; _TEXT = '#000000'; _OUTLINE = '#DDDDDD'

    OSK_ROWS_LETTERS = [list("qwertyuiop"), list("asdfghjkl"), list("zxcvbnm")]
    OSK_ROWS_SYMBOLS = [list("1234567890"), list("!@#$%^&*()"), list("-_=+[]{}|")]

    _osk_var    = [None]   # StringVar the OSK is currently typing into
    _osk_shift  = [False]
    _osk_syms   = [False]
    _letter_btns = []
    _symbol_btns = []
    _cur_form    = [None]  # form frame currently shifted up by OSK

    osk_frame = tk.Frame(root, bg=_BG)

    def _key(ch):
        v = _osk_var[0]
        if v is None: return
        if _osk_shift[0] and not _osk_syms[0]:
            ch = ch.upper(); _osk_shift[0] = False; _ref_shift()
        v.set(v.get() + ch)

    def _bs():
        v = _osk_var[0]
        if v and v.get(): v.set(v.get()[:-1])

    def _ref_shift():
        on = _osk_shift[0] and not _osk_syms[0]
        shift_btn.config(bg=_DARK if on else _MID, fg=_BG if on else _TEXT)
        for b, ch in _letter_btns: b.config(text=ch.upper() if on else ch)

    def _tog_shift():
        _osk_shift[0] = not _osk_shift[0]; _ref_shift()

    def _tog_sym():
        _osk_syms[0] = not _osk_syms[0]
        if _osk_syms[0]:
            lkb.pack_forget(); skb.pack(fill=tk.X)
            sym_btn.config(bg=_DARK, fg=_BG)
            shift_btn.config(state=tk.DISABLED, bg=_MID, fg='#AAAAAA')
        else:
            _osk_shift[0] = False; skb.pack_forget(); lkb.pack(fill=tk.X)
            sym_btn.config(bg=_MID, fg=_TEXT); shift_btn.config(state=tk.NORMAL); _ref_shift()

    def _rows(parent, rows, store):
        for row in rows:
            rf = tk.Frame(parent, bg=_BG); rf.pack(fill=tk.X, pady=2)
            for col, ch in enumerate(row):
                b = tk.Button(rf, text=ch, command=lambda c=ch: _key(c),
                              bg=_PANEL, fg=_TEXT,
                              font=tkfont.Font(family='Helvetica', size=14, weight='bold'),
                              relief=tk.FLAT, bd=0, pady=9,
                              highlightthickness=1, highlightbackground=_OUTLINE,
                              cursor='hand2', activebackground=_MID)
                b.grid(row=0, column=col, sticky='ew', padx=2); store.append((b, ch))
            for c in range(len(row)): rf.columnconfigure(c, weight=1)

    PAD = 12
    kbo = tk.Frame(osk_frame, bg=_BG); kbo.pack(fill=tk.X, padx=PAD, pady=(10,2))
    lkb = tk.Frame(kbo, bg=_BG); skb = tk.Frame(kbo, bg=_BG)
    lkb.pack(fill=tk.X)
    _rows(lkb, OSK_ROWS_LETTERS, _letter_btns)
    _rows(skb, OSK_ROWS_SYMBOLS, _symbol_btns)

    bot = tk.Frame(osk_frame, bg=_BG); bot.pack(fill=tk.X, padx=PAD, pady=(4,10))
    shift_btn = tk.Button(bot, text='\u21e7 Shift', command=_tog_shift,
                          bg=_MID, fg=_TEXT, relief=tk.FLAT, bd=0,
                          font=tkfont.Font(family='Helvetica', size=13, weight='bold'),
                          padx=12, pady=10, cursor='hand2')
    shift_btn.pack(side=tk.LEFT, padx=(0,4))

    sym_btn = tk.Button(bot, text='!@# Sym', command=_tog_sym,
                        bg=_MID, fg=_TEXT, relief=tk.FLAT, bd=0,
                        font=tkfont.Font(family='Helvetica', size=13, weight='bold'),
                        padx=12, pady=10, cursor='hand2')
    sym_btn.pack(side=tk.LEFT, padx=(0,4))

    # Period key — always visible, essential for email
    tk.Button(bot, text='.', command=lambda: _key('.'),
              bg=_PANEL, fg=_TEXT, relief=tk.FLAT, bd=0,
              font=tkfont.Font(family='Helvetica', size=16, weight='bold'),
              padx=13, pady=10, cursor='hand2',
              highlightthickness=1, highlightbackground=_OUTLINE
              ).pack(side=tk.LEFT, padx=(0,4))

    tk.Button(bot, text='Space', command=lambda: _key(' '),
              bg=_PANEL, fg=_TEXT, relief=tk.FLAT, bd=0,
              font=tkfont.Font(family='Helvetica', size=13),
              padx=0, pady=10, cursor='hand2',
              highlightthickness=1, highlightbackground=_OUTLINE
              ).pack(side=tk.LEFT, padx=(0,4), expand=True, fill=tk.X)

    tk.Button(bot, text='\u232b', command=_bs,
              bg=_MID, fg=_TEXT, relief=tk.FLAT, bd=0,
              font=tkfont.Font(family='Helvetica', size=15, weight='bold'),
              padx=12, pady=10, cursor='hand2'
              ).pack(side=tk.LEFT, padx=(0,4))

    osk_act = tk.Button(bot, text='Next \u2192', bg=_DARK, fg=_BG,
                        relief=tk.FLAT, bd=0, cursor='hand2',
                        font=tkfont.Font(family='Helvetica', size=13, weight='bold'),
                        padx=14, pady=10)
    osk_act.pack(side=tk.LEFT)

    osk_frame.update_idletasks()
    OSK_H = osk_frame.winfo_reqheight()

    def _show_osk(var, label, cmd, entry_w):
        _osk_var[0] = var; _osk_shift[0] = _osk_syms[0] = False
        skb.pack_forget(); lkb.pack(fill=tk.X)
        sym_btn.config(bg=_MID, fg=_TEXT); shift_btn.config(state=tk.NORMAL); _ref_shift()
        osk_act.config(text=label, command=cmd)
        osk_frame.place(x=0, y=sh-OSK_H, width=sw); tk.Misc.lift(osk_frame)
        root.update_idletasks()
        try:
            fb = entry_w.winfo_rooty() - root.winfo_rooty() + entry_w.winfo_height()
        except Exception:
            fb = sh
        shift = max(0, fb - (sh-OSK_H) + 24)
        cf = _cur_form[0]
        if cf:
            try: cf.place_configure(y=-shift)
            except: pass
        entry_w.focus_set()

    def _hide_osk():
        osk_frame.place_forget()
        cf = _cur_form[0]
        if cf:
            try: cf.place_configure(y=0)
            except: pass

    # ── Logo frames ───────────────────────────────────────────────────────────
    LOGO_SZ = 84; _logo_tk = []; _logo_fps = 30.0; _logo_job = [None]
    try:
        _lc = cv2.VideoCapture(os.path.join(base_dir, LOGO_NORMAL_FILE))
        _lf = _lc.get(cv2.CAP_PROP_FPS)
        if _lf and 0 < _lf < 240: _logo_fps = float(_lf)
        _lm = Image.new('L', (LOGO_SZ, LOGO_SZ), 0)
        ImageDraw.Draw(_lm).ellipse((0, 0, LOGO_SZ-1, LOGO_SZ-1), fill=255)
        while True:
            ret, frm = _lc.read()
            if not ret: break
            fh, fw = frm.shape[:2]; s = min(fh, fw)
            frm = frm[(fh-s)//2:(fh-s)//2+s, (fw-s)//2:(fw-s)//2+s]
            frm = cv2.resize(frm, (LOGO_SZ, LOGO_SZ), interpolation=cv2.INTER_AREA)
            pil = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)).convert('RGBA')
            pil.putalpha(_lm)
            _bg2 = Image.new('RGBA', (LOGO_SZ, LOGO_SZ), (255,255,255,255))
            _bg2.paste(pil, (0,0), pil); _logo_tk.append(ImageTk.PhotoImage(_bg2.convert('RGB')))
        _lc.release()
    except Exception as ex:
        print(f"First boot logo: {ex}")

    # Logo canvas lives on root — stays centered during slide transitions
    LOGO_Y = 30
    lcv = tk.Canvas(root, width=LOGO_SZ, height=LOGO_SZ, bg='white', highlightthickness=0)
    lcv.place(relx=0.5, y=LOGO_Y, anchor='n')
    _li = lcv.create_image(0, 0, anchor='nw'); _lidx = [0]
    _ld = max(40, int(1000/max(_logo_fps, 1)))
    def _logo_tick():
        if phase[0] == 0 or not lcv.winfo_exists(): return
        if _logo_tk: lcv.itemconfig(_li, image=_logo_tk[_lidx[0] % len(_logo_tk)]); _lidx[0] += 1
        _logo_job[0] = root.after(_ld, _logo_tick)
    _logo_tick()

    # Step indicator dots
    DOT_Y = LOGO_Y + LOGO_SZ + 12
    sdc = tk.Canvas(root, width=64, height=16, bg='white', highlightthickness=0)
    sdc.place(relx=0.5, y=DOT_Y, anchor='n')
    def _dots(n):
        sdc.delete('all')
        sdc.create_oval(0,  3, 14, 14, fill='#1D1D1F' if n>=1 else '#D1D1D6', outline='')
        sdc.create_rectangle(18, 7, 46, 10, fill='#D1D1D6', outline='')
        sdc.create_oval(50, 3, 64, 14, fill='#1D1D1F' if n>=2 else '#D1D1D6', outline='')
    _dots(1)

    FW = min(400, sw-80); FORM_TOP = DOT_Y + 22

    def _mkfield(parent, lbl, var):
        tk.Label(parent, text=lbl, bg='white', fg='#1D1D1F',
                 font=tkfont.Font(family='Helvetica', size=18, weight='bold'),
                 anchor='w').pack(fill=tk.X, pady=(0,4))
        bdr = tk.Frame(parent, bg='#D1D1D6'); bdr.pack(fill=tk.X, pady=(0,16))
        e = tk.Entry(bdr, textvariable=var,
                     font=tkfont.Font(family='Helvetica', size=20),
                     bd=0, highlightthickness=0, bg='white', fg='#1D1D1F',
                     insertbackground='#1D1D1F')
        e.pack(fill=tk.X, padx=1, pady=1, ipady=9)
        return e, bdr

    def _pill(parent, lbl, cmd):
        BH = 52; c = tk.Canvas(parent, width=FW, height=BH,
                               bg='white', highlightthickness=0, cursor='hand2')
        c.pack(); R = BH//2
        c.create_oval(0,0,BH,BH,fill='#1D1D1F',outline='')
        c.create_rectangle(R,0,FW-R,BH,fill='#1D1D1F',outline='')
        c.create_oval(FW-BH,0,FW,BH,fill='#1D1D1F',outline='')
        c.create_text(FW//2,BH//2,text=lbl,fill='white',
                      font=tkfont.Font(family='Helvetica',size=18,weight='bold'))
        c.bind('<ButtonPress-1>', lambda e: cmd())

    # ── welcome.mov background preload ────────────────────────────────────────
    _pre = {'raw': None, 'fps': 30.0}
    def _preload_welcome():
        try:
            vc = cv2.VideoCapture(os.path.join(base_dir, 'welcome.mov'))
            fps = vc.get(cv2.CAP_PROP_FPS)
            if fps and 0 < fps < 240: _pre['fps'] = float(fps)
            raw = []
            while True:
                ret, f = vc.read()
                if not ret: break
                raw.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            vc.release(); _pre['raw'] = raw
        except Exception as ex:
            print(f"welcome.mov preload: {ex}")

    # ── Phase 1: Name ─────────────────────────────────────────────────────────
    p1 = tk.Frame(root, bg='white')
    p1.place(x=0, y=0, width=sw, height=sh)
    _cur_form[0] = p1
    tk.Misc.lift(lcv); tk.Misc.lift(sdc)

    nf = tk.Frame(p1, bg='white')
    nf.place(relx=0.5, y=FORM_TOP, anchor='n', width=FW)
    name_var = tk.StringVar()
    name_e, name_bdr = _mkfield(nf, "What\u2019s your first name?", name_var)
    name_err = tk.Label(nf, text='', bg='white', fg='#FF3B30',
                        font=tkfont.Font(family='Helvetica', size=13))
    name_err.pack(pady=(0,12))

    def _sub_name():
        nm = name_var.get().strip()
        if not nm: name_err.config(text='Please enter your name.'); return
        result['name'] = nm; _hide_osk(); _transition(nm)

    _pill(nf, 'Next  \u2192', _sub_name)
    name_e.focus_set()
    for w in (name_e, name_bdr):
        w.bind('<ButtonPress-1>', lambda e: _show_osk(name_var,'Next \u2192',_sub_name,name_e))
    name_e.bind('<Return>', lambda e: _sub_name())

    # ── Transition: spring "Hi [name]!" → slide to email ─────────────────────
    def _transition(nm):
        for w in list(nf.winfo_children()):
            try: w.destroy()
            except: pass
        hi = tk.Label(p1, text=f"Hi, {nm}! \U0001f44b", bg='white', fg='#1D1D1F',
                      font=tkfont.Font(family='Helvetica', size=38, weight='bold'))
        hi.place(relx=0.5, rely=0.5, anchor='center')
        sc=[1.7]; sv=[0.0]
        def _spring():
            f=(1.0-sc[0])*0.28; sv[0]=sv[0]*0.65+f; sc[0]+=sv[0]
            sz=max(10,int(38*sc[0]))
            try: hi.config(font=tkfont.Font(family='Helvetica',size=sz,weight='bold'))
            except: return
            if abs(sc[0]-1.0)>0.003 or abs(sv[0])>0.003: root.after(14,_spring)
            else:
                hi.config(font=tkfont.Font(family='Helvetica',size=38,weight='bold'))
                threading.Thread(target=_preload_welcome, daemon=True).start()
                root.after(680, lambda: _slide_to_email(p1))
        _spring()

    def _slide_to_email(old_p):
        _dots(2)
        p2 = tk.Frame(root, bg='white')
        p2.place(x=sw, y=0, width=sw, height=sh)
        tk.Misc.lift(lcv); tk.Misc.lift(sdc)

        ef = tk.Frame(p2, bg='white')
        ef.place(relx=0.5, y=FORM_TOP, anchor='n', width=FW)
        email_var = tk.StringVar()
        email_e, email_bdr = _mkfield(ef, 'Contact email', email_var)
        email_err = tk.Label(ef, text='', bg='white', fg='#FF3B30',
                             font=tkfont.Font(family='Helvetica', size=13))
        email_err.pack(pady=(0,12))

        def _sub_email():
            result['email'] = email_var.get().strip(); _hide_osk(); phase[0]=3; _phase3()

        _pill(ef, 'Get started', _sub_email)

        def _tap_email(e=None):
            _cur_form[0] = p2
            _show_osk(email_var,'Done \u2713',_sub_email,email_e)
        for w in (email_e, email_bdr): w.bind('<ButtonPress-1>', _tap_email)
        email_e.bind('<Return>', lambda e: _sub_email())

        # Slide old_p → left, p2 → center
        STEPS=14
        def _slide(i):
            t=i/STEPS; t=1-(1-t)**3
            try: old_p.place_configure(x=int(-sw*t)); p2.place_configure(x=int(sw*(1-t)))
            except: pass
            if i<STEPS: root.after(10,lambda:_slide(i+1))
            else:
                try: old_p.destroy()
                except: pass
                p2.place_configure(x=0); _cur_form[0]=p2
                tk.Misc.lift(lcv); tk.Misc.lift(sdc)
                root.after(80, _tap_email)   # auto-open OSK once slide completes
        _slide(0)

    # ── Phase 3: welcome video → image → Get started ──────────────────────────
    def _phase3():
        if _logo_job[0]:
            try: root.after_cancel(_logo_job[0])
            except: pass
        for w in root.winfo_children():
            try: w.destroy()
            except: pass
        root.configure(bg='white'); root.update()

        # Brief wait if preload still running (max 600ms — usually done by now)
        t0=time.time()
        while _pre['raw'] is None and time.time()-t0<0.6:
            root.update(); time.sleep(0.04)

        VID_MAX_W=sw-60; VID_MAX_H=int(sh*0.64)
        vdelay=max(16,int(1000/max(_pre['fps'],1)))
        vid_tk=[]; disp_w,disp_h=VID_MAX_W,VID_MAX_H
        raw=_pre.get('raw') or []
        if raw:
            fh,fw=raw[0].shape[:2]; sc=min(VID_MAX_W/fw,VID_MAX_H/fh,1.0)
            disp_w=max(1,int(fw*sc)); disp_h=max(1,int(fh*sc))
            for frm in raw:
                pil=Image.fromarray(frm).resize((disp_w,disp_h),Image.LANCZOS)
                vid_tk.append(ImageTk.PhotoImage(pil))

        sph=[None]
        try:
            sp=Image.open(os.path.join(base_dir,'welcome_img.png')).convert('RGB')
            sp=sp.resize((disp_w,disp_h),Image.LANCZOS); sph[0]=ImageTk.PhotoImage(sp)
        except Exception as ex: print(f"welcome_img.png: {ex}")

        VID_Y=max(16,(sh-disp_h-80)//2); BTN_Y=VID_Y+disp_h+24
        vl=tk.Label(root,bg='white'); vl.place(relx=0.5,y=VID_Y+disp_h//2,anchor='center')
        GSW,GSH=min(320,sw-80),56; GR=GSH//2
        gs=tk.Canvas(root,width=GSW,height=GSH,bg='white',highlightthickness=0,cursor='hand2')
        gs.create_oval(0,0,GSH,GSH,fill='#1D1D1F',outline='')
        gs.create_rectangle(GR,0,GSW-GR,GSH,fill='#1D1D1F',outline='')
        gs.create_oval(GSW-GSH,0,GSW,GSH,fill='#1D1D1F',outline='')
        gs.create_text(GSW//2,GSH//2,text='Get started',fill='white',
                       font=tkfont.Font(family='Helvetica',size=20,weight='bold'))
        def _done(*_): phase[0]=0
        gs.bind('<ButtonPress-1>',_done)
        _vi=[0]
        def _play():
            if not root.winfo_exists(): return
            i=_vi[0]
            if i<len(vid_tk):
                vl.config(image=vid_tk[i]); vl.image=vid_tk[i]; _vi[0]=i+1
                root.after(vdelay,_play)
            else:
                if sph[0]: vl.config(image=sph[0]); vl.image=sph[0]
                gs.place(relx=0.5,y=BTN_Y+GSH//2,anchor='center')
        if vid_tk: _play()
        else:
            if sph[0]: vl.config(image=sph[0]); vl.image=sph[0]
            gs.place(relx=0.5,y=BTN_Y+GSH//2,anchor='center')

    # ── Event loop ────────────────────────────────────────────────────────────
    while phase[0] != 0:
        root.update()
        time.sleep(0.016)
    for w in root.winfo_children():
        try: w.destroy()
        except: pass
    root.configure(bg='white')
    return result['name'], result['email']



def main():
    global cap, current_letter, sentence_text, all_raw_letters, current_letters
    global is_recording, last_confident_time, last_letter_time, prediction_history
    cap = None

    # ── First-run registration ─────────────────────────────────────────────────
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from asl_settings import Store as _Store
        _store = _Store()
        _user_name = _store.get("user_name") if _store else None
    except Exception:
        _store = None; _user_name = None

    # ── Device serial — generated once, stored in settings, sent to server ─────
    _serial = _store.get("device_serial") if _store else None
    if not _serial:
        import uuid as _uuid
        _serial = "MU-" + _uuid.uuid4().hex[:8].upper()
        try:
            if _store: _store.set("device_serial", _serial)
        except Exception:
            pass

    _was_registered = bool(_user_name)   # True if already in settings before this run

    if not _user_name:
        _user_name, _user_email = _run_first_boot(_root)
        try:
            if _store:
                _store.set("user_name", _user_name)
                _store.set("user_email", _user_email)
        except Exception:
            pass

    root = _root
    for w in root.winfo_children():
        w.destroy()
    root.configure(bg='white')
    root.update()

    # ── Welcome back overlay (returning users only) ───────────────────────────
    if _was_registered and _user_name:
        _welcome_back(root, _user_name)

    # ── Fleet monitor — non-blocking, app works fine if server is unreachable ──
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        import asl_monitor
        asl_monitor.start(_user_name or "Unknown", APP_VERSION)
    except Exception as _me:
        print(f"Monitor unavailable (no server yet?): {_me}")

    ui = ASLTranslationUI(root)

    root.deiconify()
    root.attributes('-topmost', False)

    gst = ("libcamerasrc ! "
           "video/x-raw,width=640,height=480,framerate=30/1 ! "
           "videoconvert ! video/x-raw,format=BGR ! "
           "appsink drop=true max-buffers=1 sync=false")
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("GStreamer failed, trying simpler pipeline")
        gst2 = ("libcamerasrc ! videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=1 sync=false")
        cap  = cv2.VideoCapture(gst2, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("GStreamer failed, falling back to index 0")
        cap = cv2.VideoCapture(0)
    print("Live ASL Translation — running")

    infer_lock   = threading.Lock()
    infer_result = {
        "hand_present"  : False,
        "out_of_frame"  : False,
        "landmark_image": None,
        "hand_crop"     : None,
        "pred_letter"   : None,
        "confidence"    : 0.0,
        "current_time"  : 0.0,
    }
    infer_running    = threading.Event()
    infer_running.set()
    _last_infer_time = [0.0]

    def inference_loop():
        # Motion-gating state: when the hand is holding still, the CNN would
        # return the same letter frame after frame — so reuse the last result
        # and skip the inference. A real pass is still forced every
        # RECHECK_EVERY+1 frames as a safety against a stale reuse.
        prev_coords = [None]
        reuse       = {"letter": None, "conf": 0.0, "count": 0}
        idle_since  = [0.0]

        while infer_running.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Rate cap — sleep remaining time to hit INFER_FPS_CAP
            now     = time.time()
            elapsed = now - _last_infer_time[0]
            if elapsed < INFER_INTERVAL:
                time.sleep(INFER_INTERVAL - elapsed)
            _last_infer_time[0] = time.time()

            frame          = cv2.flip(frame, 1)
            hand_landmarks = get_hand_landmarks(frame)
            lm_img         = calculate_slope_and_adjust(hand_landmarks, frame.shape) \
                             if hand_landmarks else None
            hand_present   = hand_landmarks is not None
            current_time   = time.time()

            oof = False
            if hand_landmarks:
                for lm in hand_landmarks.landmark:
                    if lm.x < 0.02 or lm.x > 0.98 or lm.y < 0.02 or lm.y > 0.98:
                        oof = True
                        break

            hand_crop = None
            if hand_landmarks and ui._panel_open:
                hand_crop = get_hand_crop_with_landmarks(frame, hand_landmarks)

            pred_letter = None
            confidence  = 0.0
            if lm_img is not None:
                # ── Motion gate ──────────────────────────────────────────────
                coords = np.array([(l.x, l.y) for l in hand_landmarks.landmark],
                                  dtype=np.float32)
                still = False
                if prev_coords[0] is not None and prev_coords[0].shape == coords.shape:
                    motion = float(np.mean(np.abs(coords - prev_coords[0])))
                    still  = motion < MOTION_EPS
                prev_coords[0] = coords

                if (still and reuse["letter"] is not None
                        and reuse["count"] < RECHECK_EVERY):
                    # Hand is holding a letter — reuse last prediction, skip CNN
                    pred_letter = reuse["letter"]
                    confidence  = reuse["conf"]
                    reuse["count"] += 1
                else:
                    processed = preprocess_image(lm_img)
                    if processed is not None:
                        pred       = tflite_predict(processed)
                        pred_idx   = int(np.argmax(pred))
                        confidence = float(np.max(pred))
                        if confidence > CONFIDENCE_THRESHOLD:
                            pred_letter = class_names[pred_idx] \
                                          if pred_idx < len(class_names) else None
                    reuse["letter"] = pred_letter
                    reuse["conf"]   = confidence
                    reuse["count"]  = 0
            else:
                prev_coords[0]  = None
                reuse["letter"] = None
                reuse["conf"]   = 0.0
                reuse["count"]  = 0

            with infer_lock:
                infer_result["hand_present"]   = hand_present
                infer_result["out_of_frame"]   = oof
                infer_result["landmark_image"] = lm_img
                infer_result["hand_crop"]      = hand_crop
                infer_result["pred_letter"]    = pred_letter
                infer_result["confidence"]     = confidence
                infer_result["current_time"]   = current_time

            # ── Idle throttle: nothing in frame for a while → poll slower ─────
            if not hand_present:
                if idle_since[0] == 0.0:
                    idle_since[0] = current_time
                elif current_time - idle_since[0] > IDLE_AFTER:
                    time.sleep(IDLE_EXTRA_SLEEP)   # ~6-7 fps while idle, cool Pi
            else:
                idle_since[0] = 0.0

    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()

    last_word          = ""
    hand_was_present   = False
    hand_absent_frames = 0     # consecutive frames with no hand
    oof_count          = 0
    last_letter_added  = 0.0
    llm_queried_for    = ""

    def _fire_llm():
        nonlocal last_word, llm_queried_for
        combined = ''.join(all_raw_letters)
        if not combined or combined == llm_queried_for:
            return
        llm_queried_for = combined

        # ── Word-gap segmentation (ADAPTIVE): fingerspelling pace varies, so a
        # fixed threshold marks mid-word gaps as boundaries for slow signers.
        # Instead, measure this utterance's own rhythm: a boundary is a pause
        # clearly longer than the signer's typical letter-to-letter gap
        # (2.2x median), never below the WORD_GAP_SPACE floor. The relay also
        # treats these spaces as soft hints only, so a wrong space cannot
        # corrupt the output - this just makes the hints actually meaningful.
        gaps = [letter_times[i] - letter_times[i-1]
                for i in range(1, min(len(all_raw_letters), len(letter_times)))]
        if gaps:
            med = sorted(gaps)[len(gaps) // 2]
            gap_thresh = max(WORD_GAP_SPACE, 2.2 * med)
        else:
            gap_thresh = WORD_GAP_SPACE
        seg = []
        for i, ch in enumerate(all_raw_letters):
            if (i > 0 and i < len(letter_times)
                    and (letter_times[i] - letter_times[i-1]) > gap_thresh):
                seg.append(' ')
            seg.append(ch)
        snap = ''.join(seg)

        print(f"\u2192 querying LLM: {snap}")
        ui.trigger_recog()
        def qt(s=snap):
            global sentence_text, last_translation_time
            nonlocal last_word, llm_queried_for
            result = query_hf_llm(s, llm_history)
            if result:
                sentence_text = result
                last_word     = result.split()[-1] if result.split() else ""
                last_translation_time = time.time()   # protect it from inactivity-clear
                if (len(result.strip()) > 1
                        and (not llm_history or llm_history[-1] != result)):
                    llm_history.append(result)        # skip exact-duplicate context
                print("LLM \u2192", sentence_text)
                try: chime_sound.play()
                except: pass
                _speak(sentence_text)
            else:
                # Query failed (network drop / cold model / timeout after retries).
                # Reset the dedupe guard so this same input can be retried on the
                # next trigger, and flash a brief indicator so it isn't silent.
                llm_queried_for = ""
                print("LLM query failed - will retry on next trigger")
                def _show_fail():
                    try: ui.show_llm_error()
                    except Exception: pass
                root.after(0, _show_fail)
        threading.Thread(target=qt, daemon=True).start()
        def _clear():
            global all_raw_letters, current_letters
            all_raw_letters.clear(); current_letters.clear()
            letter_times.clear()
        root.after(int(CAPSULE_LINGER * 1000), _clear)
        global is_recording, last_letter_time
        is_recording     = False
        last_letter_time = None

    def update_frame():
        global current_letter, sentence_text, all_raw_letters, current_letters, last_translation_time
        global is_recording, last_confident_time, last_letter_time, prediction_history
        nonlocal last_word, hand_was_present, hand_absent_frames, oof_count, last_letter_added, llm_queried_for

        if not ui.running:
            infer_running.clear()
            cap.release()
            pygame.mixer.quit()
            return

        with infer_lock:
            hand_present   = infer_result["hand_present"]
            out_of_frame   = infer_result["out_of_frame"]
            lm_img         = infer_result["landmark_image"]
            hand_crop      = infer_result["hand_crop"]
            pred_letter    = infer_result["pred_letter"]
            confidence     = infer_result["confidence"]
            current_time   = infer_result["current_time"]

        if out_of_frame and hand_present:
            oof_count += 1
        else:
            oof_count = 0
        show_oof = oof_count >= 8

        if ui._panel_open and hand_crop is not None:
            ui.update_hand_panel(hand_crop, lm_img)

        if current_time > 0:
            if pred_letter:
                prediction_history.append(pred_letter)
                last_confident_time = current_time
            elif not hand_present:
                prediction_history.clear()

        new_confirmed = False
        prev_letter   = current_letter
        prev_len      = len(all_raw_letters)
        if current_time > 0:
            process_letter(prediction_history, current_time)
        new_confirmed = (current_letter != prev_letter)
        if len(all_raw_letters) > prev_len:
            last_letter_added = current_time
            llm_queried_for   = ""
            # First letter of a fresh utterance after a long LLM-idle period:
            # pre-warm the relay in the background so the model is loaded by
            # the time the user finishes spelling (kills cold-start latency).
            if prev_len == 0 and (time.time() - _last_llm_contact[0]) > LLM_WARM_AFTER:
                _warm_llm_async()

        # ── LLM trigger 1: hand genuinely and consistently absent ─────────────
        if hand_present:
            hand_absent_frames = 0
        else:
            if hand_was_present or hand_absent_frames > 0:
                hand_absent_frames += 1
            if hand_absent_frames == HAND_ABSENT_THRESH and all_raw_letters:
                print(f"Hand absent {hand_absent_frames} consecutive frames \u2192 LLM")
                _fire_llm()

        # ── LLM trigger 2: long deliberate pause with stable prediction ───────
        if (hand_present and not show_oof
                and all_raw_letters and last_letter_added > 0):
            pause_elapsed = current_time - last_letter_added
            hist_list  = list(prediction_history)
            if hist_list:
                top       = max(set(hist_list), key=hist_list.count)
                top_ratio = hist_list.count(top) / len(hist_list)
                is_stable = top_ratio >= 0.80
            else:
                is_stable = True
            if pause_elapsed > LLM_PAUSE_TIMEOUT and is_stable:
                print(f"Deliberate pause {pause_elapsed:.1f}s \u2192 LLM")
                _fire_llm()
                last_letter_added = 0.0

        hand_was_present = hand_present

        # Inactivity clear - but never wipe a translation that just arrived.
        # A slow (retried/cold-start) query can land after CLEAR_TIMEOUT would
        # otherwise fire, so give any fresh result at least SENTENCE_TIMEOUT on screen.
        recent_translation = (last_translation_time > 0 and
                              (current_time - last_translation_time) < SENTENCE_TIMEOUT)
        if (last_confident_time and (current_time - last_confident_time > CLEAR_TIMEOUT)
                and not recent_translation):
            all_raw_letters.clear(); current_letters.clear()
            letter_times.clear()
            sentence_text = ""; current_letter = ""; last_word = ""
            is_recording = False; last_confident_time = None
            last_translation_time = 0.0
            prediction_history.clear()
            print("State cleared (inactivity)")

        ui.update_display(
            last_word, current_letter, sentence_text,
            list(all_raw_letters),
            new_letter_confirmed=new_confirmed,
            out_of_frame=show_oof,
        )

        root.after(80, update_frame)

    update_frame()
    root.mainloop()

    infer_running.clear()
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
