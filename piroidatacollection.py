#!/usr/bin/env python3
"""
ASL Data Collection Server — runs on Raspberry Pi
Control from your Mac browser: http://asl.local:5000

Data saved to ~/asl/roitraining_data/<LETTER>/*.jpg
Identical landmark format to roidatacollection.py
"""

import os, math, time, threading, json
os.environ.setdefault("DISPLAY", ":0")

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, request, render_template_string

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR  = os.path.expanduser("~/asl/roitraining_data")
ASL_SIGNS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
GOAL      = 200

KEYFRAME_MAP = {
    'J': ['1. Start (Back-of-Hand I)', '2. Mid-Scoop (Turning)', '3. End (Back-of-Hand I)'],
    'Z': ['1. Start (Top-Left)', '2. Corner 1 (Top-Right)',
          '3. Corner 2 (Bottom-Left)', '4. End (Bottom-Right)'],
}

os.makedirs(BASE_DIR, exist_ok=True)

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                             min_detection_confidence=0.7)

# ── Shared state ──────────────────────────────────────────────────────────────
lock  = threading.Lock()
state = {
    "letter"        : "A",
    "keyframe_index": 0,
    "recording"     : False,
    "session_count" : 0,
    "save_counter"  : 0,   # set from disk on letter change, incremented atomically
    "hand_present"  : False,
}

# MJPEG frame buffers — bytes assignment is atomic under the GIL
frame_buf = {"camera": None, "landmark": None}


# ── Camera ────────────────────────────────────────────────────────────────────
def open_camera():
    # sync=false is critical — without it appsink waits on clock and adds 300ms+ latency
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
        print("GStreamer failed, trying index 0")
        cap = cv2.VideoCapture(0)
    return cap


# ── Landmark processing — verbatim from roidatacollection.py ──────────────────
def check_wrist_slope(hand_landmarks, image_shape):
    if not hand_landmarks:
        return None
    h, w      = image_shape[:2]
    wrist     = (int(hand_landmarks.landmark[0].x  * w),
                 int(hand_landmarks.landmark[0].y  * h))
    pinky_mcp = (int(hand_landmarks.landmark[17].x * w),
                 int(hand_landmarks.landmark[17].y * h))
    dx = pinky_mcp[0] - wrist[0]
    dy = pinky_mcp[1] - wrist[1]
    slope = float('inf') if dx == 0 else dy / dx
    return slope, wrist, pinky_mcp


def calculate_slope_and_adjust(hand_landmarks, image_shape, target_size=(200, 200)):
    if not hand_landmarks:
        return None
    h, w               = image_shape[:2]
    target_h, target_w = target_size
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

    wrist      = landmarks[0]
    index_mcp  = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp   = landmarks[13]
    pinky_mcp  = landmarks[17]

    x_range_min = min(index_mcp[0], middle_mcp[0], ring_mcp[0], pinky_mcp[0])
    x_range_max = max(index_mcp[0], middle_mcp[0], ring_mcp[0], pinky_mcp[0])
    x_range     = x_range_max - x_range_min
    x_alignment_threshold = w * 0.1

    is_vertical = x_range_min <= wrist[0] <= x_range_max
    slope_info  = check_wrist_slope(hand_landmarks, image_shape)
    slope = slope_info[0] if slope_info else None
    is_horizontal = (wrist[0] < x_range_min or wrist[0] > x_range_max) or \
                    (slope is not None and -0.5 <= slope <= 0.5)

    if is_vertical and not is_horizontal:
        fixed_wrist   = (target_w // 2, target_h - 50)
        mcp_center_x  = (index_mcp[0]+middle_mcp[0]+ring_mcp[0]+pinky_mcp[0]) // 4
        mcp_center_y  = (index_mcp[1]+middle_mcp[1]+ring_mcp[1]+pinky_mcp[1]) // 4
        mcp_center    = (mcp_center_x, mcp_center_y)
        dx = mcp_center[0] - wrist[0]
        dy = mcp_center[1] - wrist[1]
        rotation_angle = (-math.pi / 2 - math.atan2(dy, dx)) if (dx or dy) else 0
        cos_theta, sin_theta = math.cos(rotation_angle), math.sin(rotation_angle)
        rotated_landmarks = []
        for x, y in landmarks:
            rel_x = x - wrist[0]; rel_y = y - wrist[1]
            rotated_landmarks.append((
                int(rel_x * cos_theta - rel_y * sin_theta + fixed_wrist[0]),
                int(rel_x * sin_theta + rel_y * cos_theta + fixed_wrist[1])))
        transformed_landmarks = rotated_landmarks

    elif is_horizontal:
        if slope is not None and -0.5 <= slope <= 0.5:
            dx = pinky_mcp[0] - wrist[0]
            dy = pinky_mcp[1] - wrist[1]
            rotation_angle = (0 - math.atan2(dy, dx)) if (dx or dy) else 0
            cos_theta, sin_theta = math.cos(rotation_angle), math.sin(rotation_angle)
            rotated_landmarks = []
            for x, y in landmarks:
                rel_x = x - wrist[0]; rel_y = y - wrist[1]
                rotated_landmarks.append((
                    int(rel_x * cos_theta - rel_y * sin_theta + wrist[0]),
                    int(rel_x * sin_theta + rel_y * cos_theta + wrist[1])))
            transformed_landmarks = rotated_landmarks

        elif slope is not None and abs(slope) > 0.5:
            # Steep tilt either direction (slope > 0.5 or slope < -0.5) —
            # normalize to vertical using MCP-centre direction.
            fixed_wrist  = (target_w // 2, target_h - 50)
            mcp_center_x = (index_mcp[0]+middle_mcp[0]+ring_mcp[0]+pinky_mcp[0]) // 4
            mcp_center_y = (index_mcp[1]+middle_mcp[1]+ring_mcp[1]+pinky_mcp[1]) // 4
            mcp_center   = (mcp_center_x, mcp_center_y)
            dx = mcp_center[0] - wrist[0]
            dy = mcp_center[1] - wrist[1]
            rotation_angle = (-math.pi / 2 - math.atan2(dy, dx)) if (dx or dy) else 0
            cos_theta, sin_theta = math.cos(rotation_angle), math.sin(rotation_angle)
            rotated_landmarks = []
            for x, y in landmarks:
                rel_x = x - wrist[0]; rel_y = y - wrist[1]
                rotated_landmarks.append((
                    int(rel_x * cos_theta - rel_y * sin_theta + fixed_wrist[0]),
                    int(rel_x * sin_theta + rel_y * cos_theta + fixed_wrist[1])))
            transformed_landmarks = rotated_landmarks
        else:
            transformed_landmarks = landmarks
    else:
        transformed_landmarks = landmarks

    x_coords = [pt[0] for pt in transformed_landmarks]
    y_coords = [pt[1] for pt in transformed_landmarks]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    padding = 20
    scale   = min(
        (target_w - 2*padding) / (x_max - x_min) if x_max > x_min else 1,
        (target_h - 2*padding) / (y_max - y_min) if y_max > y_min else 1)
    offset_x = (target_w - (x_max - x_min) * scale) // 2
    offset_y = (target_h - (y_max - y_min) * scale) // 2
    final_landmarks = [
        (int((x - x_min) * scale + offset_x),
         int((y - y_min) * scale + offset_y))
        for x, y in transformed_landmarks]

    landmark_image = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    for connection in mp_hands.HAND_CONNECTIONS:
        s_idx, e_idx = connection
        if s_idx < len(final_landmarks) and e_idx < len(final_landmarks):
            cv2.line(landmark_image,
                     final_landmarks[s_idx], final_landmarks[e_idx], (0, 0, 0), 1)
    for point in final_landmarks:
        cv2.circle(landmark_image, point, 3, (0, 0, 0), 1)
    if len(final_landmarks) >= 18:
        cv2.line(landmark_image, final_landmarks[0], final_landmarks[17], (255, 0, 0), 2)
    return landmark_image


# ── File helpers ──────────────────────────────────────────────────────────────
def disk_count(letter):
    d = os.path.join(BASE_DIR, letter)
    if not os.path.exists(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith('.jpg')])


def next_filepath(letter):
    """Return unique path using atomic counter. Never reads disk mid-session."""
    sign_dir = os.path.join(BASE_DIR, letter)
    os.makedirs(sign_dir, exist_ok=True)
    with lock:
        n = state["save_counter"]
        state["save_counter"]  += 1
        state["session_count"] += 1
    return os.path.join(sign_dir, f"{n:04d}.jpg")


# ── Camera loop ───────────────────────────────────────────────────────────────
def camera_loop():
    cap = open_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame        = cv2.flip(frame, 1)
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results      = hands.process(rgb)
        hlm          = results.multi_hand_landmarks[0] \
                       if results.multi_hand_landmarks else None
        lm_img       = calculate_slope_and_adjust(hlm, frame.shape) if hlm else None
        hand_present = hlm is not None

        with lock:
            letter        = state["letter"]
            recording     = state["recording"]
            kf_idx        = state["keyframe_index"]
            session_count = state["session_count"]

        # ── Save ─────────────────────────────────────────────────────────────
        if recording and lm_img is not None:
            path = next_filepath(letter)
            cv2.imwrite(path, lm_img)

        # ── Annotate camera preview ───────────────────────────────────────────
        preview = frame.copy()
        if hlm:
            mp_drawing.draw_landmarks(
                preview, hlm, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 200, 80),  thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 80,  0), thickness=2))
            slope_info = check_wrist_slope(hlm, frame.shape)
            if slope_info:
                _, wrist_pt, pinky_pt = slope_info
                cv2.line(preview, wrist_pt, pinky_pt, (255, 0, 0), 2)

        rec_color = (0, 0, 255) if recording else (0, 200, 80)
        rec_txt   = "● REC  " if recording else ""
        cv2.putText(preview, f"{rec_txt}Letter: {letter}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, rec_color, 2)

        if letter in KEYFRAME_MAP:
            poses = KEYFRAME_MAP[letter]
            hint  = f"Pose {kf_idx+1}/{len(poses)}: {poses[kf_idx]}"
            cv2.putText(preview, hint,
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        cv2.putText(preview, f"Session: {session_count}  |  Disk: {disk_count(letter)}",
                    (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        if not hlm:
            cv2.putText(preview, "NO HAND DETECTED",
                        (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        # ── Encode JPEG buffers ───────────────────────────────────────────────
        ok, cam_jpg = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            frame_buf["camera"] = cam_jpg.tobytes()

        if lm_img is not None:
            disp = cv2.resize(lm_img, (300, 300), interpolation=cv2.INTER_NEAREST)
            ok, lm_jpg = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                frame_buf["landmark"] = lm_jpg.tobytes()
        else:
            blank = np.ones((300, 300, 3), dtype=np.uint8) * 230
            cv2.putText(blank, "No hand", (65, 158),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 160, 160), 2)
            ok, lm_jpg = cv2.imencode('.jpg', blank)
            if ok:
                frame_buf["landmark"] = lm_jpg.tobytes()

        with lock:
            state["hand_present"] = hand_present


# ── MJPEG generator ───────────────────────────────────────────────────────────
def mjpeg_gen(key):
    while True:
        jpg = frame_buf.get(key)
        if jpg:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
        time.sleep(0.033)


# ── HTML UI ───────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASL Data Collection</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,Helvetica,sans-serif;background:#111;color:#eee;padding:14px}
h1{font-size:1.2rem;margin-bottom:12px}
.grid{display:grid;grid-template-columns:1fr 270px;gap:12px}
.panel{background:#1c1c1c;border-radius:12px;padding:12px;margin-bottom:10px}
.streams{display:flex;gap:8px;flex-wrap:wrap}
.streams img{border-radius:8px;background:#000;flex:1;min-width:130px;max-width:100%;height:auto}
.letters{display:grid;grid-template-columns:repeat(7,1fr);gap:4px;margin-top:8px}
.lbtn{padding:8px 0;border:none;border-radius:7px;background:#262626;color:#999;
      font-size:.9rem;font-weight:700;cursor:pointer}
.lbtn:hover{background:#333}
.lbtn.active{background:#009929;color:#fff}
.ctrl{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
.btn{flex:1;padding:10px;border:none;border-radius:9px;font-size:.9rem;
     font-weight:700;cursor:pointer}
.btn:active{opacity:.7}
.btn-rec{background:#b01c1c;color:#fff}
.btn-rec.on{background:#ff2222;animation:pulse 1s infinite}
.btn-nav{background:#262626;color:#ccc}
.btn-del{background:#3a1010;color:#ccc;flex:.5}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}
.kf-panel{margin-top:8px;background:#182218;border-radius:8px;
          padding:8px 10px;display:none}
.kf-panel.show{display:block}
.kf-label{font-size:.75rem;color:#7c7;margin-bottom:4px}
.kf-btn{background:#1e3a1e;border:none;border-radius:6px;color:#9e9;
        padding:5px 7px;margin:2px;font-size:.72rem;cursor:pointer}
.kf-btn.active{background:#009929;color:#fff}
.stat{font-size:.8rem;color:#888;line-height:1.9}
.stat b{color:#ddd}
.stat .big{color:#00cc44;font-size:1.5rem;font-weight:700}
.prog{margin-top:6px;background:#222;border-radius:4px;height:6px;overflow:hidden}
.prog-bar{height:100%;background:#009929;border-radius:4px;transition:width .4s}
.ctable{width:100%;border-collapse:collapse;font-size:.73rem}
.ctable td,.ctable th{padding:3px 5px;border-bottom:1px solid #222;text-align:center}
.ctable th{color:#555;font-weight:normal}
.zero{color:#333}.low{color:#b87800}.ok{color:#009929}
#status{margin-top:6px;font-size:.75rem;color:#666;min-height:15px}
</style>
</head>
<body>
<h1>🤚 ASL Data Collection — Pi Camera</h1>
<div class="grid">
  <div>
    <div class="panel">
      <div class="streams">
        <img src="/stream/camera"   alt="camera">
        <img src="/stream/landmark" alt="landmark">
      </div>
      <div id="status">Connecting…</div>
    </div>
    <div class="panel">
      <div class="letters" id="lgrid"></div>
      <div class="kf-panel" id="kfpanel">
        <div class="kf-label">Keyframe poses — tap to annotate:</div>
        <div id="kfbtns"></div>
      </div>
      <div class="ctrl">
        <button class="btn btn-nav" onclick="prevLetter()">◀ Prev</button>
        <button class="btn btn-rec" id="recbtn" onclick="toggleRec()">⏺ Record</button>
        <button class="btn btn-nav" onclick="nextLetter()">Next ▶</button>
      </div>
      <div class="ctrl">
        <button class="btn btn-del" onclick="deleteLast()">🗑 Delete last</button>
      </div>
    </div>
  </div>
  <div>
    <div class="panel">
      <div class="stat">
        Letter: <b id="curletter">A</b><br>
        On disk: <span class="big" id="disktotal">0</span><br>
        This session: <b id="sesscnt">0</b><br>
        Hand: <b id="handpres">–</b>
      </div>
      <div class="prog"><div class="prog-bar" id="progbar" style="width:0%"></div></div>
      <small style="color:#444">Goal: %(GOAL)s per letter</small>
    </div>
    <div class="panel">
      <table class="ctable">
        <tr><th>Ltr</th><th>Count</th></tr>
        <tbody id="ctable"></tbody>
      </table>
    </div>
  </div>
</div>
<script>
const LETTERS   = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const KEYFRAMES = %(KEYFRAMES_JSON)s;
const GOAL      = %(GOAL)s;
let curLetter = 'A', recording = false;

// Build letter grid
const grid = document.getElementById('lgrid');
LETTERS.forEach(l => {
  const b = document.createElement('button');
  b.className = 'lbtn' + (l==='A' ? ' active' : '');
  b.id = 'lb-'+l; b.textContent = l;
  b.onclick = () => setLetter(l);
  grid.appendChild(b);
});

function setLetter(l) {
  fetch('/set_letter', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({letter:l})});
  document.querySelectorAll('.lbtn').forEach(b=>b.classList.remove('active'));
  document.getElementById('lb-'+l).classList.add('active');
  curLetter = l;
  renderKF(l, 0);
}

function renderKF(letter, kfIdx) {
  const panel = document.getElementById('kfpanel');
  const poses = KEYFRAMES[letter];
  if (!poses) { panel.classList.remove('show'); return; }
  panel.classList.add('show');
  const btns = document.getElementById('kfbtns');
  btns.innerHTML = '';
  poses.forEach((p, i) => {
    const b = document.createElement('button');
    b.className = 'kf-btn' + (i===kfIdx ? ' active' : '');
    b.textContent = p; b.id = 'kf-'+i;
    b.onclick = () => {
      fetch('/set_keyframe', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({index:i})});
    };
    btns.appendChild(b);
  });
}

function prevLetter() { setLetter(LETTERS[(LETTERS.indexOf(curLetter)-1+26)%26]); }
function nextLetter() { setLetter(LETTERS[(LETTERS.indexOf(curLetter)+1)%26]); }

function toggleRec() {
  fetch('/toggle_record', {method:'POST'});
}

function deleteLast() {
  if (!confirm('Delete last image for '+curLetter+'?')) return;
  fetch('/delete_last',{method:'POST'}).then(r=>r.json()).then(d=>{
    document.getElementById('status').textContent = d.msg;
    setTimeout(()=>document.getElementById('status').textContent='', 3500);
  });
}

function poll() {
  fetch('/state').then(r=>r.json()).then(s => {
    if (s.letter !== curLetter) {
      curLetter = s.letter;
      document.querySelectorAll('.lbtn').forEach(b=>b.classList.remove('active'));
      document.getElementById('lb-'+s.letter).classList.add('active');
      renderKF(s.letter, s.keyframe_index);
    }
    // Sync KF button highlights
    document.querySelectorAll('.kf-btn').forEach((b,i)=>
      b.classList.toggle('active', i===s.keyframe_index));

    recording = s.recording;
    const rb = document.getElementById('recbtn');
    rb.textContent = recording ? '⏹ Stop' : '⏺ Record';
    rb.className = 'btn btn-rec'+(recording?' on':'');

    document.getElementById('curletter').textContent = s.letter;
    document.getElementById('disktotal').textContent = s.disk_count;
    document.getElementById('sesscnt').textContent   = s.session_count;
    document.getElementById('handpres').textContent  = s.hand_present ? '✅' : '❌';
    document.getElementById('progbar').style.width   =
      Math.min(100, s.disk_count/GOAL*100)+'%%';

    const tb = document.getElementById('ctable');
    tb.innerHTML = '';
    Object.entries(s.all_counts).forEach(([l,n])=>{
      const c = n===0?'zero':n<50?'low':'ok';
      tb.innerHTML += `<tr><td>${l}</td><td class="${c}">${n}</td></tr>`;
    });
  }).catch(()=>{});
}
setInterval(poll, 500);
poll();

document.addEventListener('keydown', e => {
  if (e.target.tagName==='INPUT') return;
  if (e.key==='r'||e.key==='R') toggleRec();
  if (e.key==='ArrowRight'||e.key==='n') nextLetter();
  if (e.key==='ArrowLeft' ||e.key==='p') prevLetter();
});
</script>
</body>
</html>"""

HTML = HTML \
    .replace("%(GOAL)s",           str(GOAL)) \
    .replace("%(KEYFRAMES_JSON)s", json.dumps(KEYFRAME_MAP))


# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route('/')
def index():
    return HTML

@app.route('/stream/camera')
def stream_camera():
    return Response(mjpeg_gen('camera'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream/landmark')
def stream_landmark():
    return Response(mjpeg_gen('landmark'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    with lock:
        letter    = state["letter"]
        recording = state["recording"]
        sc        = state["session_count"]
        hp        = state["hand_present"]
        kf        = state["keyframe_index"]
    dc     = disk_count(letter)
    counts = {l: disk_count(l) for l in ASL_SIGNS}
    return jsonify({
        "letter"        : letter,
        "keyframe_index": kf,
        "recording"     : recording,
        "session_count" : sc,
        "hand_present"  : hp,
        "disk_count"    : dc,
        "all_counts"    : counts,
    })

@app.route('/set_letter', methods=['POST'])
def set_letter():
    l = (request.json or {}).get('letter', 'A').upper()
    if l not in ASL_SIGNS:
        return jsonify({"ok": False})
    dc = disk_count(l)
    with lock:
        state["letter"]         = l
        state["keyframe_index"] = 0
        state["recording"]      = False
        state["session_count"]  = 0
        state["save_counter"]   = dc   # start filenames from current disk total
    print(f"→ Letter: {l}  (disk: {dc})")
    return jsonify({"ok": True})

@app.route('/set_keyframe', methods=['POST'])
def set_keyframe():
    idx    = int((request.json or {}).get('index', 0))
    with lock:
        letter = state["letter"]
        poses  = KEYFRAME_MAP.get(letter, [])
        if 0 <= idx < len(poses):
            state["keyframe_index"] = idx
            print(f"  Keyframe → {poses[idx]}")
    return jsonify({"ok": True})

@app.route('/toggle_record', methods=['POST'])
def toggle_record():
    with lock:
        state["recording"] = not state["recording"]
        if state["recording"]:
            state["session_count"] = 0
            letter = state["letter"]
            # Sync counter from disk so filenames never collide with existing files
            state["save_counter"] = disk_count(letter)
            print(f"▶ Recording  {letter}  (files from #{state['save_counter']})")
        else:
            print(f"⏹ Stopped.  Session: {state['session_count']}")
    return jsonify({"ok": True})

@app.route('/delete_last', methods=['POST'])
def delete_last():
    with lock:
        letter = state["letter"]
    sign_dir = os.path.join(BASE_DIR, letter)
    if not os.path.exists(sign_dir):
        return jsonify({"msg": f"No data folder for {letter}"})
    files = sorted(f for f in os.listdir(sign_dir) if f.endswith('.jpg'))
    if not files:
        return jsonify({"msg": f"No images for {letter}"})
    os.remove(os.path.join(sign_dir, files[-1]))
    remaining = disk_count(letter)
    # Sync counter so next save doesn't reuse deleted number
    with lock:
        state["save_counter"] = remaining
    return jsonify({"msg": f"Deleted {files[-1]}. {letter} now has {remaining} images."})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Starting camera thread…")
    threading.Thread(target=camera_loop, daemon=True).start()
    time.sleep(2.0)   # camera + mediapipe warmup
    print()
    print("=" * 50)
    print("  Open on your Mac:")
    print("  http://asl.local:5000")
    print("=" * 50)
    print()
    app.run(host='0.0.0.0', port=5000, threaded=True)
