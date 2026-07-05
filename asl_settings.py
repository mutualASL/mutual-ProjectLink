#!/usr/bin/env python3
"""
ASL Device Settings
  Standalone:  python3 asl_settings.py
  Embedded:    from asl_settings import SettingsDrawer
               drawer = SettingsDrawer(root, on_font_change=..., on_volume_change=...)
"""

import argparse, json, os, subprocess, threading
import tkinter as tk
from tkinter import font as tkfont, messagebox

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#FFFFFF"
TEXT    = "#000000"
SUB     = "#888888"
PANEL   = "#F5F5F5"
OUTLINE = "#DDDDDD"
DARK    = "#1A1A1A"
MID     = "#E8E8E8"
BAR     = "#1A1A1A"
GREEN   = "#00CC3A"
ACCENT  = "#00AA2D"

IS_PI = os.path.exists("/proc/device-tree/model")

# ── Persistence ───────────────────────────────────────────────────────────────
SETTINGS_FILE = os.path.expanduser("~/.asl_settings.json")
DEFAULTS = {
    "volume":      75,
    "tts_speed":   150,
    "tts_pitch":   50,
    "font_family": "Helvetica",
    "font_size":   18,
    "font_bold":   False,
    "font_italic": False,
}

class Store:
    def __init__(self, on_change=None):
        self._d = dict(DEFAULTS); self._on_change = on_change
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE) as f: self._d.update(json.load(f))
        except Exception: pass

    def get(self, k):   return self._d.get(k, DEFAULTS.get(k))
    def set(self, k, v):
        self._d[k] = v
        try:
            with open(SETTINGS_FILE, "w") as f: json.dump(self._d, f, indent=2)
        except Exception: pass
        if self._on_change:
            try: self._on_change(k, v)
            except Exception: pass

# ── System helpers ────────────────────────────────────────────────────────────
def set_volume(pct):
    try:
        if IS_PI: subprocess.run(["amixer","sset","Master",f"{pct}%"],
                                  capture_output=True, timeout=3)
        else:     subprocess.run(["osascript","-e",
                                  f"set volume output volume {pct}"],
                                  capture_output=True, timeout=3)
    except Exception: pass

def speak(text, speed, pitch):
    try:
        if IS_PI:
            for exe in ("espeak-ng","espeak"):
                if subprocess.run([exe,"--version"],capture_output=True,timeout=2).returncode==0:
                    subprocess.run([exe,"-s",str(speed),"-p",str(pitch),text],
                                   capture_output=True,timeout=15); return
        else: subprocess.run(["say","-r",str(speed),text],capture_output=True,timeout=15)
    except Exception: pass

# ── WiFi helpers ──────────────────────────────────────────────────────────────
def _run(cmd, timeout=20, sudo=False):
    if sudo: cmd = ["sudo"] + list(cmd)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired: return "", "timed out", 1
    except FileNotFoundError:         return "", f"not found: {cmd[0]}", 1

def _auth_error(err):
    return any(k in err.lower() for k in ("authorization","permission","denied","policy"))

def _run_wifi(cmd, timeout=30):
    """
    Run nmcli with privilege escalation.
    Skips pkexec — it needs a graphical polkit agent not present in our fullscreen app.
    Strategy: direct first (works if polkit rule installed), then sudo (NOPASSWD in sudoers).
    """
    out, err, rc = _run(cmd, timeout)
    if rc == 0:
        return out, err, rc
    # Escalate via sudo — deploy script adds: asl ALL=(ALL) NOPASSWD: /usr/bin/nmcli
    out2, err2, rc2 = _run(cmd, timeout, sudo=True)
    return (out2 or out), (err2 or err), rc2

def _nmcli_available():
    _, _, rc = _run(["nmcli","--version"])
    return rc == 0

def _split_terse(line):
    parts, cur, i = [], [], 0
    while i < len(line):
        if line[i]=="\\" and i+1<len(line) and line[i+1]==":":
            cur.append(":"); i += 2
        elif line[i]==":":
            parts.append("".join(cur)); cur=[]; i += 1
        else:
            cur.append(line[i]); i += 1
    parts.append("".join(cur)); return parts

def _get_networks():
    out, _, rc = _run(["nmcli","--terse","--fields",
                       "IN-USE,SSID,SIGNAL,SECURITY",
                       "device","wifi","list","--rescan","yes"], timeout=25)
    if rc != 0: return []
    nets, seen = [], set()
    for line in out.strip().splitlines():
        p = _split_terse(line)
        if len(p) < 4: continue
        ssid = p[1].replace("\\:",":")
        if not ssid or ssid in seen: continue
        seen.add(ssid)
        nets.append({"in_use":p[0].strip()=="*","ssid":ssid,
                     "signal":p[2].strip(),"security":p[3].strip()})
    nets.sort(key=lambda n:(not n["in_use"],-int(n["signal"] or 0)))
    return nets

def _active_ssid():
    out, _, rc = _run(["nmcli","--terse","--fields","NAME,TYPE,STATE",
                       "connection","show","--active"])
    if rc != 0: return None
    for line in out.strip().splitlines():
        p = line.split(":")
        if len(p)>=3 and "wireless" in p[1] and "activated" in p[2]:
            return p[0]
    return None

# ── Animation helpers ─────────────────────────────────────────────────────────
def _rgb(widget, color):
    try:
        r, g, b = widget.winfo_rgb(color)
        return r>>8, g>>8, b>>8
    except Exception: return 128,128,128

def anim_color(widget, to_color, key="bg", steps=10, ms=14):
    try: r1,g1,b1 = _rgb(widget, widget.cget(key))
    except Exception: return
    try: r2,g2,b2 = _rgb(widget, to_color)
    except Exception: return
    def _step(i):
        try:
            if not widget.winfo_exists(): return
            t = (i/steps)**2*(3-2*i/steps)
            widget.config(**{key:"#{:02x}{:02x}{:02x}".format(
                int(r1+(r2-r1)*t),int(g1+(g2-g1)*t),int(b1+(b2-b1)*t))})
            if i < steps: widget.after(ms, lambda: _step(i+1))
        except tk.TclError: pass
    _step(0)

def anim_int(widget, start, end, cb, steps=10, ms=14):
    def _step(i):
        try:
            if not widget.winfo_exists(): return
            t = (i/steps)**2*(3-2*i/steps)
            cb(int(start+(end-start)*t))
            if i < steps: widget.after(ms, lambda: _step(i+1))
        except tk.TclError: pass
    _step(0)

# ── Scrollable frame helper ───────────────────────────────────────────────────
def make_scroll_frame(parent):
    """Returns (outer_frame, inner_frame). Pack outer, put widgets in inner."""
    outer = tk.Frame(parent, bg=BG)
    cv    = tk.Canvas(outer, bg=BG, highlightthickness=0)
    sb    = tk.Scrollbar(outer, orient=tk.VERTICAL, command=cv.yview,
                         bg=DARK, troughcolor=MID, activebackground="#444",
                         relief=tk.FLAT, width=14, bd=0)
    inner = tk.Frame(cv, bg=BG)
    win   = cv.create_window(0, 0, anchor="nw", window=inner)

    def _resize(e):
        cv.itemconfig(win, width=cv.winfo_width())
        cv.configure(scrollregion=cv.bbox("all"))
    inner.bind("<Configure>", _resize)
    cv.bind("<Configure>",    lambda e: cv.itemconfig(win, width=e.width))

    cv.configure(yscrollcommand=sb.set)
    cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    def _scroll(e):
        cv.yview_scroll(int(-1*(e.delta/120)), "units")
    cv.bind_all("<MouseWheel>", _scroll)
    return outer, inner

# ── Rounded card ─────────────────────────────────────────────────────────────
def card(parent, **kw):
    """A rounded-looking card frame with border."""
    f = tk.Frame(parent, bg=PANEL,
                 highlightthickness=1, highlightbackground=OUTLINE,
                 **kw)
    return f

# ── Pill toggle button ────────────────────────────────────────────────────────
class AnimPill(tk.Radiobutton):
    def __init__(self, parent, text, var, value, cmd, size=12, **kw):
        self._var=var; self._value=value; self._cmd=cmd
        super().__init__(parent, text=text, variable=var, value=value,
                         command=self._clicked,
                         bg=MID, fg=TEXT, selectcolor=DARK,
                         activebackground=MID, activeforeground=TEXT,
                         font=tkfont.Font(family="Helvetica",size=size,weight="bold"),
                         indicatoron=False, relief=tk.FLAT, bd=0,
                         padx=16, pady=12, cursor="hand2", **kw)
        self._tid = var.trace_add("write", self._sync)
        self.bind("<Destroy>", lambda _: self._rm())

    def _rm(self):
        try: self._var.trace_remove("write", self._tid)
        except Exception: pass

    def _clicked(self):
        self._sync()
        if self._cmd: self._cmd()

    def _sync(self, *_):
        try:
            active = str(self._var.get()) == str(self._value)
            anim_color(self, DARK if active else MID, "bg")
            anim_color(self, BG   if active else TEXT, "fg")
        except tk.TclError: pass

class AnimCheck(tk.Checkbutton):
    def __init__(self, parent, text, var, cmd, size=12, **kw):
        self._var=var; self._cmd=cmd
        super().__init__(parent, text=text, variable=var,
                         command=self._clicked,
                         bg=MID, fg=TEXT, selectcolor=DARK,
                         activebackground=MID, activeforeground=TEXT,
                         font=tkfont.Font(family="Helvetica",size=size,weight="bold"),
                         indicatoron=False, relief=tk.FLAT, bd=0,
                         padx=16, pady=12, cursor="hand2", **kw)
        self._tid = var.trace_add("write", self._sync)
        self.bind("<Destroy>", lambda _: self._rm())

    def _rm(self):
        try: self._var.trace_remove("write", self._tid)
        except Exception: pass

    def _clicked(self):
        self._sync()
        if self._cmd: self._cmd()

    def _sync(self, *_):
        try:
            on = bool(self._var.get())
            anim_color(self, DARK if on else MID, "bg")
            anim_color(self, BG   if on else TEXT, "fg")
        except tk.TclError: pass

# ── Widget helpers ────────────────────────────────────────────────────────────
def action_btn(parent, text, cmd, bg=DARK, fg=BG, size=13, **kw):
    b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                  activebackground="#333" if bg==DARK else OUTLINE,
                  activeforeground=fg,
                  font=tkfont.Font(family="Helvetica",size=size,weight="bold"),
                  relief=tk.FLAT, bd=0, padx=20, pady=13, cursor="hand2", **kw)
    b.set_text    = lambda t: b.config(text=t)
    b.config_fill = lambda c, p=None: b.config(bg=c, activebackground=p or "#333")
    return b

def page_header(parent, title):
    hdr = tk.Frame(parent, bg=BG)
    hdr.pack(fill=tk.X, padx=24, pady=(20, 0))
    tk.Label(hdr, text=title, bg=BG, fg=TEXT,
             font=tkfont.Font(family="Helvetica",size=22,weight="bold")
             ).pack(side=tk.LEFT)
    tk.Frame(parent, bg=OUTLINE, height=1).pack(fill=tk.X, padx=24, pady=(12,8))
    return hdr

def sec_label(parent, text):
    tk.Label(parent, text=text.upper(), bg=parent["bg"], fg=SUB,
             font=tkfont.Font(family="Helvetica",size=10,weight="bold")
             ).pack(anchor="w", pady=(14,4))

# ── Custom touch-friendly slider ─────────────────────────────────────────────
class CustomSlider(tk.Canvas):
    """
    Canvas-based slider with a large black circular thumb and thin grey track.
    Much easier to grab on a touchscreen than the default tk.Scale.
    """
    TRACK_H = 5
    THUMB_R = 18
    PAD     = 22   # horizontal padding so thumb never clips edges

    def __init__(self, parent, from_=0, to=100, value=50, command=None, **kw):
        self._from = from_
        self._to   = to
        self._val  = float(value)
        self._cmd  = command
        H = self.THUMB_R * 2 + 8
        super().__init__(parent, height=H, bg=PANEL,
                         highlightthickness=0, cursor="hand2", **kw)
        self.bind("<Configure>",       lambda e: self._redraw())
        self.bind("<ButtonPress-1>",   self._press)
        self.bind("<B1-Motion>",       self._motion)
        self.bind("<ButtonRelease-1>", lambda e: None)
        self.after(10, self._redraw)

    def _x(self, val):
        w = self.winfo_width() or 300
        r = (val - self._from) / max(1, self._to - self._from)
        return self.PAD + r * (w - 2 * self.PAD)

    def _val_from_x(self, x):
        w = self.winfo_width() or 300
        r = (x - self.PAD) / max(1, w - 2 * self.PAD)
        return self._from + max(0.0, min(1.0, r)) * (self._to - self._from)

    def _rrect(self, x1, y1, x2, y2, r, **kw):
        if x2 < x1 + 2: x2 = x1 + 2
        r = min(r, (x2 - x1) // 2, (y2 - y1) // 2)
        r = max(1, r)
        pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
               x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
               x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        self.create_polygon(pts, smooth=True, **kw)

    def _redraw(self):
        self.delete("all")
        w  = self.winfo_width() or 300
        h  = int(str(self["height"]))
        cy = h // 2
        R  = self.THUMB_R
        tx = self._x(self._val)

        # Full track (grey)
        self._rrect(self.PAD, cy - self.TRACK_H//2,
                    w - self.PAD, cy + self.TRACK_H//2,
                    self.TRACK_H//2, fill=MID, outline="")

        # Filled track left of thumb (black)
        self._rrect(self.PAD, cy - self.TRACK_H//2,
                    tx, cy + self.TRACK_H//2,
                    self.TRACK_H//2, fill=DARK, outline="")

        # Shadow ring for depth
        self.create_oval(tx-R-2, cy-R-2, tx+R+2, cy+R+2,
                         fill="#CCCCCC", outline="")

        # Thumb — large black circle
        self.create_oval(tx-R, cy-R, tx+R, cy+R,
                         fill=DARK, outline="")

    def _press(self, e):
        self._update(e.x)

    def _motion(self, e):
        self._update(e.x)

    def _update(self, x):
        self._val = self._val_from_x(x)
        self._redraw()
        if self._cmd:
            self._cmd(int(round(self._val)))

    def get(self):
        return int(round(self._val))

    def set(self, val):
        self._val = max(float(self._from), min(float(self._to), float(val)))
        self._redraw()


# ── Slider card ───────────────────────────────────────────────────────────────
class SliderCard(tk.Frame):
    def __init__(self, parent, label, key, lo, hi, unit, store, on_change=None):
        super().__init__(parent, bg=BG)
        self.store=store; self._key=key; self._unit=unit
        self._on_change=on_change; self._prev=store.get(key)

        inner = card(self)
        inner.pack(fill=tk.X, pady=4, padx=0)

        hdr = tk.Frame(inner, bg=PANEL)
        hdr.pack(fill=tk.X, padx=16, pady=(14,4))
        tk.Label(hdr, text=label, bg=PANEL, fg=TEXT,
                 font=tkfont.Font(family="Helvetica",size=13,weight="bold")
                 ).pack(side=tk.LEFT)
        self._lbl = tk.Label(hdr, text=f"{self._prev}{unit}",
                             bg=PANEL, fg=DARK,
                             font=tkfont.Font(family="Helvetica",size=14,weight="bold"),
                             width=7, anchor="e")
        self._lbl.pack(side=tk.RIGHT)

        self._var = tk.IntVar(value=self._prev)
        self._slider = CustomSlider(
            inner, from_=lo, to=hi, value=self._prev,
            command=self._moved)
        self._slider.pack(fill=tk.X, padx=16, pady=(4, 16))

    def _moved(self, v):
        iv = int(v)
        self._lbl.config(text=f"{iv}{self._unit}")
        self._prev = iv
        self.store.set(self._key, iv)
        if self._on_change: self._on_change(iv)

# ── Inline OSK ────────────────────────────────────────────────────────────────
class InlineOSK(tk.Frame):
    """
    Full QWERTY keyboard embedded inside WiFiPage.
    Keys stretch to fill the full frame width using grid.
    Symbols toggle swaps the entire keyboard layout cleanly.
    """
    ROWS_LETTERS = [
        list("qwertyuiop"),
        list("asdfghjkl"),
        list("zxcvbnm"),
    ]
    ROWS_SYMBOLS = [
        list("1234567890"),
        list("!@#$%^&*()"),
        list("-_=+[]{}\\|"),
    ]

    def __init__(self, parent):
        super().__init__(parent, bg=BG)
        self._on_done = self._on_cancel = None
        self._shift   = False
        self._symbols = False
        self._show_pw = False
        self._var     = tk.StringVar()
        self._build()

    def _build(self):
        PAD = 16
        # Prompt
        self._prompt = tk.Label(self, text="", bg=BG, fg=TEXT,
                                font=tkfont.Font(family="Helvetica",size=13,weight="bold"))
        self._prompt.pack(pady=(14,6), padx=PAD, anchor="w")

        # Entry row
        ef = tk.Frame(self, bg=BG)
        ef.pack(fill=tk.X, padx=PAD, pady=(0,10))
        self._entry = tk.Entry(ef, textvariable=self._var, show="●",
                               bg=PANEL, fg=TEXT, insertbackground=DARK,
                               font=tkfont.Font(family="Helvetica",size=16),
                               relief=tk.FLAT, bd=0,
                               highlightthickness=2,
                               highlightbackground=OUTLINE,
                               highlightcolor=DARK)
        self._entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10, padx=(0,10))
        self._eye = tk.Button(ef, text="👁", command=self._toggle_show,
                              bg=MID, fg=TEXT, relief=tk.FLAT, bd=0,
                              font=("Helvetica",14), padx=10, pady=8, cursor="hand2")
        self._eye.pack(side=tk.LEFT)

        # Keyboard container — swap between letters and symbols
        self._kb_outer = tk.Frame(self, bg=BG)
        self._kb_outer.pack(fill=tk.X, padx=PAD, pady=(0,4))

        self._letter_kb = tk.Frame(self._kb_outer, bg=BG)
        self._symbol_kb = tk.Frame(self._kb_outer, bg=BG)
        self._letter_kb.pack(fill=tk.X)   # letters visible by default

        self._letter_btns = []   # (btn, char)
        self._symbol_btns = []

        self._build_rows(self._letter_kb, self.ROWS_LETTERS, self._letter_btns)
        self._build_rows(self._symbol_kb, self.ROWS_SYMBOLS, self._symbol_btns)

        # Bottom control row
        bot = tk.Frame(self, bg=BG)
        bot.pack(fill=tk.X, padx=PAD, pady=(6, 14))

        self._shift_btn = tk.Button(bot, text="⇧ Shift",
                                    command=self._toggle_shift,
                                    bg=MID, fg=TEXT, relief=tk.FLAT, bd=0,
                                    font=tkfont.Font(family="Helvetica",size=13,weight="bold"),
                                    padx=14, pady=11, cursor="hand2")
        self._shift_btn.pack(side=tk.LEFT, padx=(0,6))

        self._sym_btn = tk.Button(bot, text="!@# Symbols",
                                  command=self._toggle_symbols,
                                  bg=MID, fg=TEXT, relief=tk.FLAT, bd=0,
                                  font=tkfont.Font(family="Helvetica",size=13,weight="bold"),
                                  padx=14, pady=11, cursor="hand2")
        self._sym_btn.pack(side=tk.LEFT, padx=(0,6))

        tk.Button(bot, text="Space", command=lambda: self._key(" "),
                  bg=PANEL, fg=TEXT, relief=tk.FLAT, bd=0,
                  font=tkfont.Font(family="Helvetica",size=13),
                  padx=0, pady=11, cursor="hand2",
                  highlightthickness=1, highlightbackground=OUTLINE
                  ).pack(side=tk.LEFT, padx=(0,6), expand=True, fill=tk.X)

        tk.Button(bot, text=".", command=lambda: self._key("."),
                  bg=PANEL, fg=TEXT, relief=tk.FLAT, bd=0,
                  font=tkfont.Font(family="Helvetica",size=16,weight="bold"),
                  padx=13, pady=11, cursor="hand2",
                  highlightthickness=1, highlightbackground=OUTLINE
                  ).pack(side=tk.LEFT, padx=(0,6))

        tk.Button(bot, text="⌫", command=self._backspace,
                  bg=MID, fg=TEXT, relief=tk.FLAT, bd=0,
                  font=tkfont.Font(family="Helvetica",size=15,weight="bold"),
                  padx=14, pady=11, cursor="hand2"
                  ).pack(side=tk.LEFT, padx=(0,6))

        tk.Button(bot, text="↵ Enter", command=self._done,
                  bg=MID, fg=TEXT, relief=tk.FLAT, bd=0,
                  font=tkfont.Font(family="Helvetica",size=13,weight="bold"),
                  padx=14, pady=11, cursor="hand2"
                  ).pack(side=tk.LEFT, padx=(0,6))

        tk.Button(bot, text="✕ Cancel", command=self._cancel,
                  bg=MID, fg=TEXT, relief=tk.FLAT, bd=0,
                  font=tkfont.Font(family="Helvetica",size=13),
                  padx=14, pady=11, cursor="hand2"
                  ).pack(side=tk.LEFT, padx=(0,6))

        tk.Button(bot, text="✓ Connect", command=self._done,
                  bg=DARK, fg=BG, relief=tk.FLAT, bd=0,
                  font=tkfont.Font(family="Helvetica",size=13,weight="bold"),
                  padx=14, pady=11, cursor="hand2"
                  ).pack(side=tk.LEFT)

    def _build_rows(self, parent, rows, store):
        KEY_H = 11   # pady inside key
        for row in rows:
            rf = tk.Frame(parent, bg=BG)
            rf.pack(fill=tk.X, pady=3)
            n = len(row)
            for col, ch in enumerate(row):
                b = tk.Button(rf, text=ch,
                              command=lambda c=ch: self._key(c),
                              bg=PANEL, fg=TEXT,
                              font=tkfont.Font(family="Helvetica",size=14,weight="bold"),
                              relief=tk.FLAT, bd=0,
                              pady=KEY_H,
                              highlightthickness=1, highlightbackground=OUTLINE,
                              cursor="hand2", activebackground=MID)
                b.grid(row=0, column=col, sticky="ew", padx=2)
                store.append((b, ch))
            for col in range(n):
                rf.columnconfigure(col, weight=1)

    # ── Public ────────────────────────────────────────────────────────────────
    def show(self, prompt, on_done, on_cancel):
        self._prompt.config(text=prompt)
        self._on_done=on_done; self._on_cancel=on_cancel
        self._var.set(""); self._shift=False; self._symbols=False; self._show_pw=False
        self._entry.config(show="●")
        self._show_letters()
        self._refresh_shift()
        self.pack(fill=tk.BOTH, expand=True)
        self._entry.focus_set()
        self._entry.bind("<Return>", lambda e: self._done())

    def hide(self):
        self.pack_forget(); self._var.set("")

    # ── Internal ──────────────────────────────────────────────────────────────
    def _key(self, ch):
        if self._shift and not self._symbols:
            ch = ch.upper()
            self._shift = False
            self._refresh_shift()
        self._var.set(self._var.get() + ch)

    def _backspace(self):
        v = self._var.get()
        if v: self._var.set(v[:-1])

    def _toggle_shift(self):
        self._shift = not self._shift
        self._refresh_shift()

    def _toggle_symbols(self):
        self._symbols = not self._symbols
        if self._symbols:
            self._show_symbols()
        else:
            self._shift = False
            self._show_letters()
            self._refresh_shift()

    def _show_letters(self):
        self._symbol_kb.pack_forget()
        self._letter_kb.pack(fill=tk.X)
        self._sym_btn.config(bg=MID, fg=TEXT)
        self._shift_btn.config(state=tk.NORMAL, bg=MID, fg=TEXT)

    def _show_symbols(self):
        self._letter_kb.pack_forget()
        self._symbol_kb.pack(fill=tk.X)
        self._sym_btn.config(bg=DARK, fg=BG)
        self._shift_btn.config(state=tk.DISABLED, bg=MID, fg="#AAAAAA")

    def _toggle_show(self):
        self._show_pw = not self._show_pw
        self._entry.config(show="" if self._show_pw else "●")

    def _refresh_shift(self):
        on = self._shift and not self._symbols
        self._shift_btn.config(bg=DARK if on else MID, fg=BG if on else TEXT)
        for btn, ch in self._letter_btns:
            btn.config(text=ch.upper() if on else ch)

    def _done(self):
        pw=self._var.get(); cb=self._on_done; self.hide()
        if cb: cb(pw)

    def _cancel(self):
        cb=self._on_cancel; self.hide()
        if cb: cb()

# ── Pages ─────────────────────────────────────────────────────────────────────
class WiFiPage(tk.Frame):
    def __init__(self, parent, store):
        super().__init__(parent, bg=BG)
        self.store=store; self._nets=[]; self._scanning=False
        self._ok=_nmcli_available(); self._pending_ssid=None
        self._build()

    def _build(self):
        page_header(self, "WiFi")

        self._list_view = tk.Frame(self, bg=BG)
        self._list_view.pack(fill=tk.BOTH, expand=True)

        # Status label
        self._status = tk.Label(self._list_view, text="", bg=BG, fg=SUB,
                                font=tkfont.Font(family="Helvetica",size=11))
        self._status.pack(padx=24, pady=(0,6), anchor="w")

        # Network list card
        lc = card(self._list_view)
        lc.pack(fill=tk.BOTH, expand=True, padx=24, pady=(0,8))

        sb = tk.Scrollbar(lc, orient=tk.VERTICAL, bg=DARK,
                          troughcolor=MID, activebackground="#444",
                          relief=tk.FLAT, width=14, bd=0)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=6, padx=(0,6))

        self._lb = tk.Listbox(lc, yscrollcommand=sb.set,
                              bg=PANEL, fg=TEXT,
                              font=tkfont.Font(family="Helvetica",size=13),
                              selectbackground=DARK, selectforeground=BG,
                              activestyle="none", relief=tk.FLAT, bd=0,
                              highlightthickness=0)
        self._lb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        sb.config(command=self._lb.yview)

        # Buttons
        row = tk.Frame(self._list_view, bg=BG)
        row.pack(fill=tk.X, padx=24, pady=(4,18))
        self._sbtn = action_btn(row, "Scan", self.scan, size=12)
        self._sbtn.pack(side=tk.LEFT, padx=(0,10))
        action_btn(row,"Connect",self._connect,bg=MID,fg=TEXT,size=12).pack(side=tk.LEFT,padx=(0,10))
        action_btn(row,"Forget", self._forget, bg=MID,fg=TEXT,size=12).pack(side=tk.LEFT)

        if not self._ok:
            self._lb.insert(tk.END,"  nmcli not available (Pi only)")

        # Inline OSK (hidden until needed)
        self._osk = InlineOSK(self)

    def on_show(self): self.scan()

    def scan(self):
        if self._scanning or not self._ok: return
        self._scanning=True
        self._sbtn.config(text="Scanning…",state=tk.DISABLED)
        self._lb.delete(0,tk.END); self._lb.insert(tk.END,"   Scanning…")
        threading.Thread(target=self._bg_scan, daemon=True).start()

    def _bg_scan(self):
        nets = _get_networks()
        self.after(0, self._done_scan, nets)

    def _done_scan(self, nets):
        self._nets=nets; self._lb.delete(0,tk.END)
        if not nets:
            self._lb.insert(tk.END,"   No networks found.")
        else:
            for n in nets:
                pct  = int(n["signal"] or 0)
                bar  = "█"*round(pct/20)+"░"*(5-round(pct/20))
                lock = "🔒 " if n["security"] and n["security"]!="--" else "   "
                star = "●  " if n["in_use"] else "   "
                self._lb.insert(tk.END, f"  {star}{lock}{n['ssid']:<24} {bar}")
                if n["in_use"]: self._lb.itemconfig(tk.END, fg=ACCENT)
        self._scanning=False
        self._sbtn.config(text="Scan",state=tk.NORMAL)
        connected = _active_ssid()
        self._status.config(
            text=f"Connected: {connected}" if connected else "Not connected",
            fg=ACCENT if connected else SUB)

    def _connect(self):
        sel = self._lb.curselection()
        if not sel or sel[0]>=len(self._nets):
            messagebox.showwarning("Select","Tap a network first.",parent=self); return
        n = self._nets[sel[0]]
        if not n["security"] or n["security"]=="--":
            self._lb.delete(0,tk.END)
            self._lb.insert(tk.END,f"   Connecting to {n['ssid']}…")
            threading.Thread(target=self._bg_connect,args=(n["ssid"],None),daemon=True).start()
            return
        self._pending_ssid = n["ssid"]
        self._list_view.pack_forget()
        self._osk.show(prompt=f'Password for "{n["ssid"]}"',
                       on_done=self._on_pw_done, on_cancel=self._on_pw_cancel)

    def _on_pw_done(self, pw):
        self._list_view.pack(fill=tk.BOTH, expand=True)
        self._lb.delete(0,tk.END)
        self._lb.insert(tk.END,f"   Connecting to {self._pending_ssid}…")
        threading.Thread(target=self._bg_connect,
                         args=(self._pending_ssid,pw),daemon=True).start()

    def _on_pw_cancel(self):
        self._list_view.pack(fill=tk.BOTH, expand=True)
        self._pending_ssid=None

    def _bg_connect(self, ssid, pw):
        cmd = (["nmcli","device","wifi","connect",ssid,"password",pw]
               if pw else ["nmcli","device","wifi","connect",ssid])
        out, err, rc = _run_wifi(cmd)
        if rc==0 and "successfully" in out.lower():
            self.after(0,lambda: messagebox.showinfo("Connected",
                f'Connected to "{ssid}".', parent=self))
        else:
            msg = err.strip() or out.strip() or "Unknown error"
            self.after(0,lambda: messagebox.showerror("Failed", msg, parent=self))
        self.after(0, self.scan)

    def _forget(self):
        sel = self._lb.curselection()
        if not sel or sel[0]>=len(self._nets):
            messagebox.showwarning("Select","Tap a network first.",parent=self); return
        ssid = self._nets[sel[0]]["ssid"]
        if messagebox.askyesno("Forget",f'Forget "{ssid}"?',parent=self):
            threading.Thread(
                target=lambda: (_run_wifi(["nmcli","connection","delete",ssid]),
                                self.after(600,self.scan)),
                daemon=True).start()


class AudioPage(tk.Frame):
    def __init__(self, parent, store):
        super().__init__(parent, bg=BG)
        self.store=store; self._speaking=False; self._build()

    def _build(self):
        page_header(self, "Audio")
        scroll_outer, inner = make_scroll_frame(self)
        scroll_outer.pack(fill=tk.BOTH, expand=True, padx=24)

        SliderCard(inner,"Volume",      "volume",   0,100,"%",   self.store,set_volume).pack(fill=tk.X,pady=6)
        SliderCard(inner,"Speech Speed","tts_speed",80,300," wpm",self.store).pack(fill=tk.X,pady=6)
        SliderCard(inner,"Speech Pitch","tts_pitch",0, 99, "",   self.store).pack(fill=tk.X,pady=6)

        sec_label(inner,"Voice Test")
        c = card(inner)
        c.pack(fill=tk.X,pady=6)
        self._entry = tk.Entry(c, bg=PANEL, fg=TEXT, insertbackground=DARK,
                               font=tkfont.Font(family="Helvetica",size=13),
                               relief=tk.FLAT, bd=0,
                               highlightthickness=1, highlightbackground=OUTLINE,
                               highlightcolor=DARK)
        self._entry.insert(0,"Hello, ASL translation is working.")
        self._entry.pack(fill=tk.X, padx=16, pady=(16,10))
        tk.Frame(c, bg=OUTLINE, height=1).pack(fill=tk.X, padx=16)
        self._tbtn = action_btn(c,"▶  Test Voice",self._test,size=12)
        self._tbtn.pack(pady=12)

    def _test(self):
        if self._speaking: return
        text = self._entry.get().strip() or "Testing audio."
        speed=self.store.get("tts_speed"); pitch=self.store.get("tts_pitch")
        self._speaking=True; self._tbtn.set_text("Speaking…")
        def _work():
            speak(text,speed,pitch)
            self.after(0,self._done)
        threading.Thread(target=_work,daemon=True).start()

    def _done(self):
        self._speaking=False; self._tbtn.set_text("▶  Test Voice")


FONT_FAMILIES = ["Helvetica","Arial","DejaVu Sans","Liberation Sans","Courier New"]
FONT_SIZES    = [12,14,16,18,20,24,28,32]

class FontPage(tk.Frame):
    def __init__(self, parent, store, on_font_change=None):
        super().__init__(parent, bg=BG)
        self.store=store; self._on_font_change=on_font_change
        self._fams    = self._avail()
        self._fam_var = tk.StringVar(value=store.get("font_family"))
        self._sz_var  = tk.IntVar(value=store.get("font_size"))
        self._b_var   = tk.BooleanVar(value=bool(store.get("font_bold")))
        self._i_var   = tk.BooleanVar(value=bool(store.get("font_italic")))
        self._prev_sz = store.get("font_size")
        self._build()

    @staticmethod
    def _avail():
        try:
            s = set(tkfont.families())
            a = [f for f in FONT_FAMILIES if f in s]
            return a or ["Helvetica"]
        except Exception: return ["Helvetica"]

    def _build(self):
        page_header(self,"Font")
        scroll_outer, inner = make_scroll_frame(self)
        scroll_outer.pack(fill=tk.BOTH, expand=True, padx=24)

        sec_label(inner,"Family")
        fc = card(inner)
        fc.pack(fill=tk.X, pady=6)
        fr = tk.Frame(fc, bg=PANEL)
        fr.pack(fill=tk.X, padx=12, pady=10)
        for i,fam in enumerate(self._fams):
            p = AnimPill(fr,fam,self._fam_var,fam,self._update,size=11)
            p.grid(row=i//2,column=i%2,padx=5,pady=4,sticky="ew")
            fr.columnconfigure(i%2,weight=1)

        sec_label(inner,"Size")
        sc = card(inner)
        sc.pack(fill=tk.X, pady=6)
        sr = tk.Frame(sc, bg=PANEL)
        sr.pack(fill=tk.X, padx=12, pady=10)
        for i,sz in enumerate(FONT_SIZES):
            p = AnimPill(sr,str(sz),self._sz_var,sz,self._update,size=11)
            p.grid(row=i//4,column=i%4,padx=4,pady=4)
        for c in range(4): sr.columnconfigure(c,weight=1)

        sec_label(inner,"Style")
        stc = card(inner)
        stc.pack(fill=tk.X, pady=6)
        st = tk.Frame(stc, bg=PANEL)
        st.pack(fill=tk.X, padx=12, pady=10)
        AnimCheck(st,"Bold",  self._b_var,self._update,size=12).pack(side=tk.LEFT,padx=(0,10))
        AnimCheck(st,"Italic",self._i_var,self._update,size=12).pack(side=tk.LEFT)

        sec_label(inner,"Preview")
        pc = card(inner)
        pc.pack(fill=tk.X, pady=6)
        self._prev_lbl = tk.Label(pc,
            text="Hello\nA  B  C\nThe quick brown fox",
            bg=PANEL, fg=TEXT, justify=tk.CENTER,
            font=(self.store.get("font_family"),self.store.get("font_size")))
        self._prev_lbl.pack(expand=True,fill=tk.BOTH,padx=16,pady=24)

        self._abtn = action_btn(inner,"Apply & Save",self._apply,size=13)
        self._abtn.pack(pady=16, fill=tk.X)

    def _update(self):
        fam=self._fam_var.get(); sz=self._sz_var.get()
        w="bold" if self._b_var.get() else "normal"
        s="italic" if self._i_var.get() else "roman"
        if sz!=self._prev_sz:
            start=self._prev_sz; self._prev_sz=sz
            def _set(v):
                try: self._prev_lbl.config(font=(fam,v,w,s))
                except tk.TclError: pass
            anim_int(self._prev_lbl,start,sz,_set)
        else:
            try: self._prev_lbl.config(font=(fam,sz,w,s))
            except tk.TclError: self._prev_lbl.config(font=(fam,sz))

    def _apply(self):
        fam=self._fam_var.get(); sz=self._sz_var.get()
        bold=self._b_var.get(); italic=self._i_var.get()
        self.store.set("font_family",fam); self.store.set("font_size",sz)
        self.store.set("font_bold",bold);  self.store.set("font_italic",italic)
        if self._on_font_change: self._on_font_change(fam,sz,bold,italic)
        self._abtn.config_fill(GREEN); self._abtn.set_text("Saved ✓")
        self.after(1600,lambda:(self._abtn.config_fill(DARK),
                                self._abtn.set_text("Apply & Save")))


# ── SettingsDrawer ────────────────────────────────────────────────────────────
class SettingsDrawer:
    H_FRAC = 0.72

    def __init__(self, root, on_font_change=None, on_volume_change=None):
        self.root=root
        self._on_font_change=on_font_change
        self._on_vol_change=on_volume_change
        self._visible=False; self._anim_job=None
        self._win_w=root.winfo_width() or 1200
        self._win_h=root.winfo_height() or 700

        def _store_change(k,v):
            if k=="volume" and self._on_vol_change: self._on_vol_change(v)
        self.store = Store(on_change=_store_change)

        h = self._drawer_h()
        self._canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
        self._canvas.place(x=0,y=self._win_h,width=self._win_w,height=h)
        self._draw_bg()
        self._canvas.bind('<Configure>',lambda e:self._draw_bg())

        self._frame = tk.Frame(self._canvas, bg='#F5F5F7')
        self._frame.place(x=0,y=0,relwidth=1,relheight=1)

        # Nav bar
        nav = tk.Frame(self._frame, bg=BAR, height=56)
        nav.pack(side=tk.TOP, fill=tk.X); nav.pack_propagate(False)
        tk.Label(nav,text="Settings",bg=BAR,fg="white",
                 font=tkfont.Font(family="Helvetica",size=15,weight="bold"),
                 padx=20).pack(side=tk.LEFT,fill=tk.Y)
        tk.Frame(nav,bg="#444444",width=1).pack(side=tk.LEFT,fill=tk.Y,pady=10)

        self._pages={}; self._tabs={}
        container = tk.Frame(self._frame, bg='#F5F5F7')
        container.pack(fill=tk.BOTH, expand=True)

        PAGES = [
            ("WiFi",  WiFiPage),
            ("Audio", AudioPage),
            ("Font",  lambda p,s: FontPage(p,s,on_font_change=self._on_font_change)),
        ]
        for label, PageClass in PAGES:
            t = tk.Button(nav, text=label,
                          command=lambda l=label: self._show(l),
                          bg=BAR, fg="white",
                          activebackground=DARK, activeforeground="white",
                          font=tkfont.Font(family="Helvetica",size=13),
                          relief=tk.FLAT, bd=0, padx=22, cursor="hand2")
            t.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self._tabs[label]=t
            p = PageClass(container, self.store)
            p.place(relx=0,rely=0,relwidth=1,relheight=1)
            self._pages[label]=p

        self._show("WiFi")

    def _drawer_h(self): return int(self._win_h*self.H_FRAC)

    def _draw_bg(self):
        try:
            self._canvas.delete("bg")
            w = self._canvas.winfo_width()  or self._win_w
            h = self._canvas.winfo_height() or self._drawer_h()
            r = 28
            pts = [r,0, w-r,0, w,0, w,r, w,h, 0,h, 0,r, 0,0]
            self._canvas.create_polygon(pts,smooth=True,
                                        fill=BG,outline=OUTLINE,
                                        width=1.5,tags="bg")
            self._canvas.create_line(r,1,w-r,1,fill=OUTLINE,width=1,tags="bg")
            self._canvas.tag_lower("bg")
        except tk.TclError: pass

    def _show(self, key):
        for k,t in self._tabs.items():
            active = k==key
            t.config(bg=DARK if active else BAR,
                     font=tkfont.Font(family="Helvetica",size=13,
                                      weight="bold" if active else "normal"))
        self._pages[key].tkraise()
        if hasattr(self._pages[key],"on_show"):
            self._pages[key].on_show()

    def on_resize(self, win_w, win_h):
        self._win_w=win_w; self._win_h=win_h
        h=self._drawer_h()
        y = win_h-h if self._visible else win_h
        self._canvas.place(x=0,y=y,width=win_w,height=h)

    def open(self):
        if self._visible: return
        self._visible=True
        self._animate(self._win_h, self._win_h-self._drawer_h())

    def close(self):
        if not self._visible: return
        self._visible=False
        self._animate(self._win_h-self._drawer_h(), self._win_h)

    def toggle(self):
        if self._visible: self.close()
        else: self.open()

    @property
    def visible(self): return self._visible

    @property
    def H(self): return self._drawer_h()

    def _animate(self, start, end, steps=14):
        if self._anim_job: self.root.after_cancel(self._anim_job)
        w=self._win_w; h=self._drawer_h()
        def _tick(i):
            t=i/steps; t=1-(1-t)**3
            y=int(start+(end-start)*t)
            self._canvas.place(x=0,y=y,width=w,height=h)
            if i<steps: self._anim_job=self.root.after(14,lambda:_tick(i+1))
            else: self._anim_job=None
        _tick(0)


# ── Standalone ────────────────────────────────────────────────────────────────
class ASLSettings(tk.Tk):
    def __init__(self, fullscreen=False, geometry="860x600"):
        super().__init__()
        self.title("ASL Settings")
        self.configure(bg=BG)
        self.resizable(True,True)
        if fullscreen: self.attributes("-fullscreen",True)
        else: self.geometry(geometry)

        store=Store()
        PAGES=[("WiFi",WiFiPage),("Audio",AudioPage),
               ("Font",lambda p,s:FontPage(p,s))]
        pages={}; tabs={}

        nav=tk.Frame(self,bg=BAR,height=60)
        nav.pack(side=tk.TOP,fill=tk.X); nav.pack_propagate(False)
        tk.Label(nav,text="⚙  Settings",bg=BAR,fg="white",
                 font=tkfont.Font(family="Helvetica",size=16,weight="bold"),
                 padx=24).pack(side=tk.LEFT,fill=tk.Y)
        tk.Frame(nav,bg="#444",width=1).pack(side=tk.LEFT,fill=tk.Y,pady=12)

        def show(key):
            for k,t in tabs.items():
                active=k==key
                t.config(bg=DARK if active else BAR,
                         font=tkfont.Font(family="Helvetica",size=14,
                                          weight="bold" if active else "normal"))
            pages[key].tkraise()
            if hasattr(pages[key],"on_show"): pages[key].on_show()

        container=tk.Frame(self,bg=BG)
        container.pack(fill=tk.BOTH,expand=True)
        for label,PageClass in PAGES:
            t=tk.Button(nav,text=label,command=lambda l=label:show(l),
                        bg=BAR,fg="white",
                        activebackground=DARK,activeforeground="white",
                        font=tkfont.Font(family="Helvetica",size=14),
                        relief=tk.FLAT,bd=0,padx=26,cursor="hand2")
            t.pack(side=tk.LEFT,fill=tk.BOTH,expand=True)
            tabs[label]=t
            p=PageClass(container,store)
            p.place(relx=0,rely=0,relwidth=1,relheight=1)
            pages[label]=p
        show("WiFi")
        self.protocol("WM_DELETE_WINDOW",self.destroy)


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--fullscreen","-f",action="store_true")
    ap.add_argument("--geometry","-g",default="860x600")
    args=ap.parse_args()
    ASLSettings(fullscreen=args.fullscreen,geometry=args.geometry).mainloop()
