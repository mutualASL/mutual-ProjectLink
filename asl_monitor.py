"""
asl_monitor.py — Pi-side fleet monitor
Imported by piroitranslation.py.

Set MONITOR_URL to wherever asl_server.py is running.
Set MONITOR_KEY to match ASL_API_KEY on the server.
"""

import os, sys, ast, socket, threading, time, requests

# ── Configure these to match your server ─────────────────────────────────────
MONITOR_URL = "https://mu-fleet-production.up.railway.app"
MONITOR_KEY = "fleet2026"
HEARTBEAT_EVERY  = 60    # seconds between heartbeats
REQUEST_TIMEOUT  = 12    # seconds before giving up on a server call
# ─────────────────────────────────────────────────────────────────────────────

_HEADERS = {"X-ASL-Key": MONITOR_KEY, "Content-Type": "application/json"}

# ── Pi identity ───────────────────────────────────────────────────────────────
def get_pi_id() -> str:
    """
    Returns this device's stable serial.
    Priority:
      1. ~/.asl_settings.json  device_serial  (set by piroitranslation.py on first run)
      2. /proc/cpuinfo hardware serial         (Pi-specific fallback)
      3. UUID saved to ~/.asl_pi_id            (last-resort fallback)
    The same value is returned on every call, so the server never creates duplicates.
    """
    # 1. Read from shared settings file — primary source
    import json as _json
    settings_path = os.path.expanduser("~/.asl_settings.json")
    try:
        with open(settings_path) as f:
            data = _json.load(f)
        serial = data.get("device_serial", "")
        if serial:
            return serial
    except Exception:
        pass

    # 2. Pi hardware serial from /proc/cpuinfo
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "Serial" in line:
                    serial = line.split(":")[1].strip()
                    if serial and serial != "0000000000000000":
                        return serial
    except Exception:
        pass

    # 3. Persistent UUID fallback
    id_file = os.path.expanduser("~/.asl_pi_id")
    if os.path.exists(id_file):
        with open(id_file) as f:
            return f.read().strip()
    import uuid
    pid = "MU-" + uuid.uuid4().hex[:8].upper()
    with open(id_file, "w") as f:
        f.write(pid)
    return pid


def get_hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


# ── Public API ────────────────────────────────────────────────────────────────
_device_number = None   # filled in after registration response

def register(user_name: str, version: str):
    """
    Register this device with the monitoring server.
    Returns the assigned device_number, or None on failure.
    Called once on startup after user has entered their name.
    """
    global _device_number
    try:
        r = requests.post(
            f"{MONITOR_URL}/register",
            headers=_HEADERS,
            json={
                "pi_id":     get_pi_id(),
                "user_name": user_name,
                "hostname":  get_hostname(),
                "version":   version,
            },
            timeout=REQUEST_TIMEOUT,
        )
        if r.ok:
            data = r.json()
            _device_number = data.get("device_number")
            # Check if the server immediately has a newer version
            if data.get("update_available"):
                current = data.get("current_version", "")
                if current and current != version:
                    print(f"[monitor] Update available on server ({version} → {current}). "
                          "Will apply on next heartbeat.")
            print(f"[monitor] Registered as device #{_device_number}")
            return _device_number
    except Exception as e:
        print(f"[monitor] Registration failed (server unreachable?): {e}")
    return None


def start(user_name: str, version: str):
    """
    Register then begin background heartbeat loop.
    Call this once from main() after the user name is known.
    Non-blocking — heartbeat runs in a daemon thread.
    """
    register(user_name, version)
    t = threading.Thread(
        target=_heartbeat_loop, args=(version,), daemon=True, name="asl-monitor"
    )
    t.start()


# ── Heartbeat loop ────────────────────────────────────────────────────────────
def _heartbeat_loop(version: str):
    while True:
        time.sleep(HEARTBEAT_EVERY)
        try:
            r = requests.post(
                f"{MONITOR_URL}/heartbeat",
                headers=_HEADERS,
                json={"pi_id": get_pi_id(), "version": version},
                timeout=REQUEST_TIMEOUT,
            )
            if not r.ok:
                continue
            data = r.json()
            server_version = data.get("current_version", version)
            if data.get("update_available") and server_version != version:
                print(f"[monitor] New version available: {version} → {server_version}")
                _apply_update(version, server_version)
        except Exception as e:
            print(f"[monitor] Heartbeat failed: {e}")


# ── Update mechanism ──────────────────────────────────────────────────────────
def _apply_update(current_version: str, new_version: str):
    """
    Download new piroitranslation.py from server, validate syntax,
    replace the current script, then restart the process.
    If anything fails, the current version keeps running untouched.
    """
    print(f"[monitor] Downloading update {new_version}…")
    try:
        r = requests.get(
            f"{MONITOR_URL}/update",
            headers=_HEADERS,
            timeout=60,
        )
        if not r.ok:
            print(f"[monitor] Update download failed: HTTP {r.status_code}")
            return

        new_content = r.text

        # Validate syntax before touching anything
        try:
            ast.parse(new_content)
        except SyntaxError as e:
            print(f"[monitor] Update rejected — syntax error: {e}")
            return

        # Write to a temp file first
        script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "piroitranslation.py")
        )
        tmp_path = script_path + ".update_tmp"

        with open(tmp_path, "w") as f:
            f.write(new_content)

        # Atomic replace (on Linux os.replace is atomic)
        os.replace(tmp_path, script_path)
        print(f"[monitor] Update applied ({new_version}). Restarting…")

        # Restart: re-exec this Python process with the new script
        time.sleep(1)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    except Exception as e:
        print(f"[monitor] Update failed: {e}")
        # Clean up temp if it exists
        try:
            os.remove(script_path + ".update_tmp")
        except Exception:
            pass
