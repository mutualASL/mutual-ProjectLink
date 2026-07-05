#!/usr/bin/env python3
"""
piupdater.py — Background checks and install helpers only.
No tkinter classes here. UI is built inline in piroitranslation.py.
"""
import os, sys, subprocess, threading, time, requests

# ── Server config from asl_monitor ───────────────────────────────────────────
try:
    import asl_monitor as _am
    MONITOR_URL = _am.MONITOR_URL
    HEADERS     = dict(_am._HEADERS)
except Exception:
    MONITOR_URL = ""
    HEADERS     = {}

# ── State (module-level so both files share it) ───────────────────────────────
system_count  = [0]
app_available = [False]
app_version   = ['']
checked       = [False]
checking      = [False]


def total():
    return system_count[0] + (1 if app_available[0] else 0)


# ── Background check ──────────────────────────────────────────────────────────
def start_check(current_version, on_done=None):
    if checking[0]:
        return
    checking[0] = True

    def _run():
        # System (uses cached apt data — no sudo, fast)
        try:
            r = subprocess.run('apt list --upgradable 2>/dev/null',
                               shell=True, capture_output=True,
                               text=True, timeout=20)
            system_count[0] = sum(1 for l in r.stdout.splitlines() if '/' in l)
        except Exception:
            system_count[0] = 0

        # App version from fleet server
        if MONITOR_URL:
            try:
                r = requests.get(f"{MONITOR_URL}/version",
                                 headers=HEADERS, timeout=10)
                data = r.json()
                sv = data.get('current_version', current_version)
                app_available[0] = (data.get('update_available', False)
                                    and sv != current_version)
                app_version[0] = sv
            except Exception:
                pass

        checked[0]  = True
        checking[0] = False
        if on_done:
            try: on_done()
            except: pass

    threading.Thread(target=_run, daemon=True, name='piupdater').start()


# ── Install helpers (called from overlay threads) ─────────────────────────────
def install_system(on_status, on_restart=None):
    """Run apt upgrade. Calls on_status(msg, colour) during progress.
    Calls on_restart() when complete (if provided)."""
    def _run():
        on_status('Fetching package lists…', None)
        subprocess.run('sudo apt-get update -q -q',
                       shell=True, capture_output=True, timeout=120)
        on_status('Installing… may take a few minutes', None)
        r = subprocess.run('sudo apt-get upgrade -y -q',
                           shell=True, capture_output=True,
                           text=True, timeout=600)
        if r.returncode == 0:
            system_count[0] = 0   # hide button immediately
            on_status('✓ Done — rebooting…', '#34C759')
            time.sleep(1.5)
            if on_restart:
                on_restart()
        else:
            on_status('Failed — check logs', '#FF3B30')
    threading.Thread(target=_run, daemon=True).start()


def install_app(on_status, on_restart):
    """
    Download all staged files from the fleet server.
    For each file:
      - if a local file with that name already exists → delete it first
      - download the new version → save with the exact same filename
    Then reboot.
    """
    import ast
    def _run():
        base_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            # Get manifest of staged files from server
            on_status('Checking staged files…', None)
            r = requests.get(f"{MONITOR_URL}/version", headers=HEADERS, timeout=10)
            if not r.ok:
                on_status(f'Server unreachable (HTTP {r.status_code})', '#FF3B30')
                return
            data  = r.json()
            files = data.get('files', [])
            if not files:
                on_status('No files staged on server', '#FF9F0A')
                return

            total = len(files)
            for i, fname in enumerate(files):
                on_status(f'Downloading {fname}  ({i+1}/{total})…', None)
                r2 = requests.get(f"{MONITOR_URL}/update/{fname}",
                                  headers=HEADERS, timeout=60)
                if not r2.ok:
                    on_status(f'Failed: {fname} (HTTP {r2.status_code})', '#FF3B30')
                    return
                content = r2.text

                # Validate Python files before touching anything on disk
                if fname.endswith('.py'):
                    on_status(f'Validating {fname}…', None)
                    try:
                        ast.parse(content)
                    except SyntaxError as e:
                        on_status(f'Bad file — {fname}: {e}', '#FF3B30')
                        return

                # Stage to .tmp, delete old file, rename to final name
                local = os.path.join(base_dir, fname)
                tmp   = local + '.tmp'
                with open(tmp, 'w') as fh:
                    fh.write(content)
                if os.path.exists(local):
                    os.remove(local)       # delete old file first
                os.rename(tmp, local)      # new file has exact same name

            # Reset in-memory flag so the update button disappears immediately
            app_available[0] = False

            # Tell server to clear staged files so the next check shows clean
            try:
                requests.post(f"{MONITOR_URL}/admin/clear_staged",
                              headers=HEADERS, timeout=8)
            except Exception:
                pass  # non-fatal — button will still be hidden locally

            on_status(
                f'✓ {total} file{"s" if total != 1 else ""} installed — rebooting…',
                '#34C759')
            time.sleep(1.5)
            on_restart()

        except Exception as e:
            on_status(f'Error: {e}', '#FF3B30')

    threading.Thread(target=_run, daemon=True).start()
