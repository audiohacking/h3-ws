#!/usr/bin/env python3
"""
h3_desktop.py — macOS app entry (AceForge pattern).

Modes (same frozen binary):
  (default)     Supervisor: first-run model path setup, spawn/restart server, pywebview
  --server      Run ``server.main()`` (child process)
  --run-h3-av   Re-enter as the PyAV ffmpeg/ffprobe shim

Existing git-clone users: first launch auto-detects ``models/MiniMax-H3`` trees and
lets you point at them so weights are never re-downloaded.
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

# Mark executed early (AceForge guard against double-entry in frozen apps).
if hasattr(sys.modules.get(__name__, None), "_h3_desktop_executed"):
    sys.exit(1)
sys.modules[__name__]._h3_desktop_executed = True  # type: ignore[attr-defined]

try:
    import fcntl

    _FCNTL = True
except ImportError:
    _FCNTL = False

from h3_paths import (
    apply_desktop_config_env,
    configured_server_host,
    configured_server_port,
    default_logs_dir,
    default_model_dir,
    discover_existing_model_dirs,
    ensure_writable_tree,
    install_frozen_h3_av_wrapper,
    is_frozen,
    load_desktop_config,
    looks_like_minimax_h3,
    save_desktop_config,
    writable_root,
)

# Health probes always use loopback — connecting to 0.0.0.0 is unreliable.
HEALTH_HOST = "127.0.0.1"
RESTART_DELAY_S = 1.5
HEALTH_TIMEOUT_S = 90.0
REBIND_REQUEST = "rebind.request"

_LOCK_FD: int | None = None
_LOCK_PATH: Path | None = None
_server_proc: subprocess.Popen[bytes] | None = None
_server_owned = False  # only kill servers we spawned
_stop = threading.Event()
_shutting_down = False
_log_file = None
_webview_module = None


def _server_port() -> int:
    return configured_server_port()


def _bind_host() -> str:
    return configured_server_host()


def _ui_url() -> str:
    """URL for the embedded webview (always loopback)."""
    return f"http://{HEALTH_HOST}:{_server_port()}/"


def _rebind_path() -> Path:
    return writable_root() / REBIND_REQUEST


def request_server_rebind() -> None:
    """Ask the desktop supervisor to stop/start the child with a fresh bind."""
    path = _rebind_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{time.time():.3f}\n", encoding="utf-8")


def _rebind_requested() -> bool:
    return _rebind_path().is_file()


def _clear_rebind_request() -> None:
    try:
        _rebind_path().unlink(missing_ok=True)
    except OSError:
        pass


def _log(msg: str) -> None:
    line = f"[H3-WS] {msg}"
    print(line, flush=True)
    if _log_file is not None:
        try:
            _log_file.write(line + "\n")
            _log_file.flush()
        except OSError:
            pass


def _setup_logging() -> None:
    global _log_file
    log_dir = default_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "desktop.log"
    try:
        _log_file = open(path, "a", encoding="utf-8")
    except OSError:
        _log_file = None


def _server_pid_path() -> Path:
    return writable_root() / "server.pid"


def _write_server_pid(pid: int) -> None:
    try:
        _server_pid_path().write_text(str(pid) + "\n", encoding="utf-8")
    except OSError:
        pass


def _clear_server_pid() -> None:
    try:
        _server_pid_path().unlink(missing_ok=True)
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _pid_args(pid: int) -> str:
    try:
        return subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "args="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _is_h3_ws_argv(args: str) -> bool:
    """True for our frozen app or ``h3_desktop`` / ``server.py`` entrypoints."""
    if not args:
        return False
    # Short-lived helpers re-enter the same binary; treat as ours so leftovers die.
    markers = ("H3-WS", "h3_desktop.py", "h3_desktop", "/server.py", " server.py")
    if not any(m in args for m in markers):
        return False
    # Ignore unrelated tools that merely mention the name in a path/arg.
    if "H3-WS" in args or "h3_desktop" in args:
        return True
    return "server.py" in args and ("h3" in args.lower() or "8765" in args)


def _is_h3_ws_server_argv(args: str) -> bool:
    return _is_h3_ws_argv(args) and "--server" in args


def _pid_looks_like_our_server(pid: int) -> bool:
    """True when ``pid`` is a live H3-WS ``--server`` (or orphaned) process."""
    if not _pid_alive(pid):
        return False
    return _is_h3_ws_server_argv(_pid_args(pid))


def _kill_pid(pid: int, *, label: str) -> None:
    """SIGTERM then SIGKILL a process (and its group when it is a session leader)."""
    if not _pid_alive(pid):
        return
    _log(f"Stopping {label} pid={pid}")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, OSError):
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, OSError):
                return
        time.sleep(0.35 if sig == signal.SIGTERM else 0.1)
        if not _pid_alive(pid):
            return


def _iter_process_table() -> list[tuple[int, str]]:
    """Return ``(pid, args)`` rows from ``ps`` (best-effort)."""
    try:
        out = subprocess.check_output(
            ["ps", "-ax", "-o", "pid=", "-o", "args="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    rows: list[tuple[int, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        rows.append((pid, parts[1]))
    return rows


def _pids_listening_on_port(port: int) -> list[int]:
    """PIDs with a TCP LISTEN socket on ``port`` (macOS ``lsof``)."""
    try:
        out = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    pids: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pids.append(int(line))
        except ValueError:
            continue
    return pids


def claim_exclusive_runtime() -> None:
    """Primary UI only: tear down every other H3-WS app/server before we bind :8765.

    Must run **after** ``acquire_instance_lock`` succeeds. A secondary launch must
    never call this — that was killing live generations when ``open`` raced.
    """
    keep = {os.getpid(), os.getppid()}
    killed: set[int] = set()

    for pid, args in _iter_process_table():
        if pid in keep or pid in killed:
            continue
        if not _is_h3_ws_argv(args):
            continue
        kind = "server" if _is_h3_ws_server_argv(args) else "app"
        _kill_pid(pid, label=f"leftover H3-WS {kind}")
        killed.add(pid)

    # PID file from a Force-Quit parent (may already be gone above).
    path = _server_pid_path()
    if path.is_file():
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = -1
        if pid not in keep and pid not in killed and _pid_looks_like_our_server(pid):
            _kill_pid(pid, label="pid-file server")
            killed.add(pid)
        _clear_server_pid()

    # Anything still listening on our port that looks like us (or blocks bind).
    for pid in _pids_listening_on_port(_server_port()):
        if pid in keep or pid in killed:
            continue
        args = _pid_args(pid)
        if _is_h3_ws_argv(args) or not args:
            _kill_pid(pid, label=f":{_server_port()} listener")
            killed.add(pid)

    deadline = time.time() + 8.0
    while time.time() < deadline and _port_open(HEALTH_HOST, _server_port()):
        time.sleep(0.2)
    if _port_open(HEALTH_HOST, _server_port()):
        _log(f"WARNING: {HEALTH_HOST}:{_server_port()} still busy after cleanup")
    elif killed:
        _log(f"Exclusive runtime claimed — cleared {len(killed)} leftover process(es)")


def release_instance_lock() -> None:
    """Drop the flock. Keep the lock file so path identity stays stable (no unlink race)."""
    global _LOCK_FD
    if _LOCK_FD is None:
        return
    try:
        if _FCNTL:
            fcntl.flock(_LOCK_FD, fcntl.LOCK_UN)
        os.close(_LOCK_FD)
    except OSError:
        pass
    _LOCK_FD = None


def shutdown_all(*, reason: str = "") -> None:
    """Idempotent teardown: stop supervisor, kill owned server, release lock."""
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True
    _stop.set()
    _log(f"Shutting down{f' ({reason})' if reason else ''}")
    stop_server()
    release_instance_lock()


def _close_webview_windows() -> None:
    mod = _webview_module
    if mod is None:
        return
    try:
        for win in list(getattr(mod, "windows", []) or []):
            try:
                win.destroy()
            except Exception:
                pass
    except Exception:
        pass


def _install_signal_handlers() -> None:
    def _handler(signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        shutdown_all(reason=name)
        _close_webview_windows()
        # Force exit if the Cocoa loop ignores destroy (Finder Quit / SIGTERM).
        threading.Timer(0.75, lambda: os._exit(0)).start()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


def _dispatch_special_modes() -> None:
    """Handle ``--run-h3-av`` / ``--server`` / ``--run-download`` (never returns)."""
    if "--run-h3-av" in sys.argv:
        idx = sys.argv.index("--run-h3-av")
        sys.argv = [sys.argv[0], *sys.argv[idx + 1 :]]
        from h3_av import main as av_main

        raise SystemExit(av_main())

    if "--run-download" in sys.argv:
        idx = sys.argv.index("--run-download")
        sys.argv = [sys.argv[0], *sys.argv[idx + 1 :]]
        import runpy
        from h3_paths import resource_root

        script = resource_root() / "scripts" / "download_model.py"
        if not script.is_file():
            raise SystemExit(f"download_model.py missing at {script}")
        runpy.run_path(str(script), run_name="__main__")
        raise SystemExit(0)

    if "--server" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--server"]
        os.environ.setdefault("DEBUG", "false")
        from server import main as server_main

        server_main()
        raise SystemExit(0)


def acquire_instance_lock() -> bool:
    global _LOCK_FD, _LOCK_PATH
    if not _FCNTL:
        return True
    lock_dir = writable_root()
    lock_dir.mkdir(parents=True, exist_ok=True)
    _LOCK_PATH = lock_dir / "h3-ws.lock"
    try:
        _LOCK_FD = os.open(str(_LOCK_PATH), os.O_CREAT | os.O_RDWR)
        fcntl.flock(_LOCK_FD, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(_LOCK_FD, 0)
        os.write(_LOCK_FD, str(os.getpid()).encode())
        os.fsync(_LOCK_FD)
        atexit.register(release_instance_lock)
        return True
    except BlockingIOError:
        _log("Another H3-WS instance is already running.")
        return False
    except OSError as exc:
        _log(f"Could not acquire lock: {exc}")
        return True


def _activate_running_instance() -> None:
    """Bring the primary UI forward; secondary launches must not kill anything."""
    _log("Activating the running H3-WS instance")
    try:
        subprocess.run(
            ["osascript", "-e", 'tell application "H3-WS" to activate'],
            check=False,
            capture_output=True,
        )
    except Exception:
        pass


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def wait_for_server(timeout: float = HEALTH_TIMEOUT_S) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline and not _stop.is_set():
        if _port_open(HEALTH_HOST, _server_port()):
            return True
        time.sleep(0.4)
    return False


def _server_argv() -> list[str]:
    if is_frozen():
        base = [sys.executable, "--server"]
    else:
        base = [sys.executable, str(Path(__file__).resolve()), "--server"]
    return [
        *base,
        "--host",
        _bind_host(),
        "--port",
        str(_server_port()),
        "--model-dir",
        str(default_model_dir()),
    ]


def spawn_server() -> subprocess.Popen[bytes]:
    global _server_proc, _server_owned
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("DEBUG", "false")
    env["H3_WS_DESKTOP"] = "1"
    apply_desktop_config_env()
    if is_frozen():
        install_frozen_h3_av_wrapper()
    log_dir = default_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    server_log = open(log_dir / "server.log", "a", encoding="utf-8")
    bind = _bind_host()
    _log(f"Starting server → bind {bind}:{_server_port()} (UI {_ui_url()})")
    # New session so we can killpg on quit; PID file covers Force Quit orphans.
    _server_proc = subprocess.Popen(
        _server_argv(),
        stdout=server_log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    _server_owned = True
    _write_server_pid(_server_proc.pid)
    return _server_proc


def stop_server() -> None:
    """Stop the server subprocess we own (never a foreign process on :8765)."""
    global _server_proc, _server_owned
    if not _server_owned:
        _server_proc = None
        return
    proc = _server_proc
    _server_proc = None
    _server_owned = False
    if proc is None:
        # Fall back to PID file (parent may have lost the Popen handle).
        path = _server_pid_path()
        if path.is_file():
            try:
                pid = int(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                _clear_server_pid()
                return
            if _pid_looks_like_our_server(pid):
                _log(f"Stopping server via pid file pid={pid}")
                try:
                    os.killpg(pid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except (ProcessLookupError, OSError):
                        pass
                time.sleep(0.5)
                if _pid_looks_like_our_server(pid):
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except (ProcessLookupError, OSError):
                            pass
            _clear_server_pid()
        return

    if proc.poll() is None:
        _log(f"Stopping server pid={proc.pid}")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            try:
                proc.terminate()
            except OSError:
                pass
        try:
            proc.wait(timeout=6)
        except subprocess.TimeoutExpired:
            _log("Server did not exit — SIGKILL")
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                try:
                    proc.kill()
                except OSError:
                    pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
    _clear_server_pid()


def supervisor_loop() -> None:
    """Keep the HTTP server alive while the desktop window is open."""
    consecutive_fail = 0
    while not _stop.is_set() and not _shutting_down:
        if _rebind_requested():
            _log("Network bind change requested — restarting server")
            _clear_rebind_request()
            stop_server()
            time.sleep(RESTART_DELAY_S)
            if _stop.is_set() or _shutting_down:
                break
            try:
                spawn_server()
                wait_for_server(timeout=60)
            except OSError as exc:
                _log(f"Rebind restart failed: {exc}")
            continue
        if _port_open(HEALTH_HOST, _server_port()):
            consecutive_fail = 0
            time.sleep(2.0)
            continue
        if not _server_owned:
            # Foreign server went away — do not hijack; wait for exit.
            time.sleep(2.0)
            continue
        proc = _server_proc
        if proc is not None and proc.poll() is None:
            time.sleep(0.5)
            continue
        if _stop.is_set() or _shutting_down:
            break
        consecutive_fail += 1
        _log(f"Server not reachable — restarting (attempt {consecutive_fail})")
        stop_server()
        time.sleep(RESTART_DELAY_S)
        if _stop.is_set() or _shutting_down:
            break
        try:
            spawn_server()
            wait_for_server(timeout=60)
        except OSError as exc:
            _log(f"Restart failed: {exc}")
            time.sleep(3.0)


def _center_on_screen(win, width: int, height: int) -> None:
    """Place a Tk window on the primary display (avoids off-screen multi-monitor dialogs)."""
    win.update_idletasks()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    x = max(0, (sw - width) // 2)
    y = max(0, (sh - height) // 3)
    win.geometry(f"{width}x{height}+{x}+{y}")


def _tk_pick_folder(title: str, initial: str | None = None) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    # Force dialog onto the primary screen's working area.
    root.update_idletasks()
    path = filedialog.askdirectory(
        parent=root,
        title=title,
        initialdir=initial or str(Path.home()),
        mustexist=True,
    )
    root.destroy()
    return path or None


def _attach_repo_sidecars(payload: dict, repo: Path) -> None:
    """When the user picks a checkout, reuse its models/loras, outputs, uploads, taeh3."""
    repo = Path(repo)
    payload["repo_root"] = str(repo.resolve())
    if (repo / "web_outputs").is_dir():
        payload["output_dir"] = str((repo / "web_outputs").resolve())
    if (repo / "web_uploads").is_dir():
        payload["upload_dir"] = str((repo / "web_uploads").resolve())
    loras = repo / "models" / "loras"
    if loras.is_dir():
        payload["lora_dir"] = str(loras.resolve())
    taeh3 = repo / "models" / "vae_approx" / "taeh3.safetensors"
    if taeh3.is_file():
        payload["taeh3_path"] = str(taeh3.resolve())


def _resolve_model_choice(choice: Path) -> dict:
    """Map a user/discovered path to config keys (model_dir, optional repo outputs)."""
    payload: dict = {"setup_done": True}
    model = choice
    if choice.name != "MiniMax-H3" and looks_like_minimax_h3(choice / "models" / "MiniMax-H3"):
        model = choice / "models" / "MiniMax-H3"
        _attach_repo_sidecars(payload, choice)
    elif looks_like_minimax_h3(choice):
        model = choice
        if choice.parent.name == "models":
            _attach_repo_sidecars(payload, choice.parent.parent)
    payload["model_dir"] = str(model.resolve())
    return payload


def run_first_run_setup() -> None:
    """Always confirm model location on first launch (centered primary-screen window)."""
    ensure_writable_tree()
    apply_desktop_config_env()
    cfg = load_desktop_config()

    if cfg.get("setup_done"):
        apply_desktop_config_env()
        _log(f"Setup already done — model_dir={default_model_dir()}")
        return

    if os.environ.get("H3_WS_SKIP_SETUP", "").strip().lower() in {"1", "true", "yes"}:
        found = discover_existing_model_dirs()
        if found:
            save_desktop_config(_resolve_model_choice(found[0]))
        else:
            save_desktop_config({"setup_done": True})
        apply_desktop_config_env()
        return

    found = discover_existing_model_dirs()
    current = default_model_dir()
    _log(f"First-run setup — {len(found)} candidate model tree(s)")

    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        if found:
            save_desktop_config(_resolve_model_choice(found[0]))
        else:
            save_desktop_config({"setup_done": True})
        apply_desktop_config_env()
        return

    result: dict[str, Path | None] = {"path": found[0] if found else None}

    root = tk.Tk()
    root.title("H3-WS — Choose model folder")
    root.resizable(False, False)
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    _center_on_screen(root, 560, 340 if found else 240)

    frame = ttk.Frame(root, padding=20)
    frame.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text=(
            "Where are your MiniMax-H3 model weights?\n\n"
            "If you already downloaded them in a git clone, pick that folder\n"
            "so nothing is re-downloaded (~134 GB)."
        ),
        justify="left",
    ).pack(anchor="w", pady=(0, 12))

    selected = tk.StringVar(value=str(found[0]) if found else "")

    if found:
        list_frame = ttk.LabelFrame(frame, text="Detected on this Mac", padding=8)
        list_frame.pack(fill="both", expand=True, pady=(0, 12))
        for path in found:
            ttk.Radiobutton(
                list_frame,
                text=str(path),
                value=str(path),
                variable=selected,
            ).pack(anchor="w", pady=2)

    btn_row = ttk.Frame(frame)
    btn_row.pack(fill="x")

    def on_choose_other() -> None:
        initial = str(found[0].parent) if found else str(Path.home() / "Documents" / "git")
        picked = _tk_pick_folder(
            "Select MiniMax-H3 folder (or the h3-ws clone root)",
            initial=initial,
        )
        if picked:
            result["path"] = Path(picked)
            root.destroy()

    def on_continue() -> None:
        raw = selected.get().strip()
        result["path"] = Path(raw) if raw else None
        root.destroy()

    def on_skip() -> None:
        result["path"] = None
        root.destroy()

    ttk.Button(btn_row, text="Choose other folder…", command=on_choose_other).pack(side="left")
    if found:
        ttk.Button(btn_row, text="Use selected", command=on_continue).pack(side="right", padx=(8, 0))
    ttk.Button(
        btn_row,
        text="Skip — download later",
        command=on_skip,
    ).pack(side="right")

    root.lift()
    root.focus_force()
    root.mainloop()

    if result["path"] is not None:
        payload = _resolve_model_choice(result["path"])
        save_desktop_config(payload)
        _log(f"Configured model_dir → {payload['model_dir']}")
    else:
        save_desktop_config({"setup_done": True, "model_dir": str(current)})
        _log(f"Skipped — default {current} (download via Models panel)")
    apply_desktop_config_env()


def open_webview() -> None:
    """Always host the Continuity UI in an embedded window (AceForge pattern)."""
    global _webview_module
    try:
        import webview

        _webview_module = webview
    except ImportError as exc:
        _log(f"pywebview missing: {exc}")
        _show_fatal(
            "H3-WS could not open its built-in window (pywebview missing).\n"
            f"Open {_ui_url()} in Safari as a fallback."
        )
        import webbrowser

        webbrowser.open(_ui_url())
        try:
            while not _stop.is_set() and not _shutting_down:
                time.sleep(1.0)
        except KeyboardInterrupt:
            shutdown_all(reason="KeyboardInterrupt")
        return

    def on_closed() -> None:
        shutdown_all(reason="window closed")
        # Leave the Cocoa run loop; force-exit if it hangs.
        threading.Timer(0.5, lambda: os._exit(0)).start()

    ui = _ui_url()
    _log(f"Opening embedded UI → {ui}")
    window = webview.create_window(
        title="H3-WS",
        url=ui,
        width=1440,
        height=960,
        min_size=(1024, 720),
        confirm_close=False,
        focus=True,
        # Default is False — blocks selecting/copying red error banners for reports.
        text_select=True,
    )
    try:
        window.events.closed += on_closed
    except Exception:
        atexit.register(lambda: shutdown_all(reason="atexit"))
    try:
        webview.start(debug=False)
    finally:
        shutdown_all(reason="webview exited")


def _show_fatal(message: str) -> None:
    """Best-effort on-screen error when the embedded UI cannot start."""
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        _center_on_screen(root, 100, 100)
        messagebox.showerror("H3-WS", message, parent=root)
        root.destroy()
    except Exception:
        _log(message)


def main() -> None:
    _dispatch_special_modes()

    _setup_logging()
    ensure_writable_tree()
    _log("H3-WS desktop starting")
    atexit.register(lambda: shutdown_all(reason="atexit"))
    _install_signal_handlers()

    # Lock first — never reap/kill until we know we are the primary UI.
    if not acquire_instance_lock():
        _activate_running_instance()
        sys.exit(0)

    # We own the lock: clear every leftover H3-WS app/server, then start fresh.
    claim_exclusive_runtime()

    try:
        run_first_run_setup()
    except Exception:
        _log("Setup error:\n" + traceback.format_exc())

    apply_desktop_config_env()
    if is_frozen():
        install_frozen_h3_av_wrapper()

    _log(f"model_dir={default_model_dir()}")

    spawn_server()
    if not wait_for_server():
        _log("ERROR: server failed to become ready — see Logs/H3-WS/server.log")
        shutdown_all(reason="server start failed")
        _show_fatal(
            "H3-WS server failed to start.\n"
            "Check ~/Library/Logs/H3-WS/server.log"
        )
        sys.exit(1)

    threading.Thread(target=supervisor_loop, name="h3-supervisor", daemon=True).start()
    try:
        open_webview()
    finally:
        shutdown_all(reason="main finally")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        _log("FATAL:\n" + err)
        try:
            shutdown_all(reason="fatal")
        except Exception:
            pass
        try:
            log_dir = default_logs_dir()
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / "error.log").write_text(err, encoding="utf-8")
        except OSError:
            pass
        sys.exit(1)
