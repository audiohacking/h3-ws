"""In-memory backend console buffer for the Web UI troubleshooting panel.

Captures ``console_h3`` lines, Python logging, and optional tails of on-disk
``server.log`` / ``desktop.log`` so remote testers can copy diagnostics without
opening Terminal.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

_MAX_LINES = 2500
_lock = threading.Lock()
_lines: deque[str] = deque(maxlen=_MAX_LINES)
_handler_installed = False


def append_console(message: str, *args: object) -> None:
    """Append one line to the ring buffer (always on — independent of DEBUG)."""
    text = message % args if args else str(message)
    text = " ".join(str(text).split())
    if not text:
        return
    ts = time.strftime("%H:%M:%S")
    entry = f"[{ts}] {text[:800]}"
    with _lock:
        if _lines and _lines[-1][11:] == entry[11:]:
            return
        _lines.append(entry)


def clear_console() -> None:
    with _lock:
        _lines.clear()


def get_console_lines(limit: int = 800) -> list[str]:
    n = max(1, min(int(limit), _MAX_LINES))
    with _lock:
        items = list(_lines)
    return items[-n:]


def _tail_file(path: Path, max_lines: int = 200) -> list[str]:
    if not path.is_file():
        return []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
    return rows[-max_lines:]


def console_snapshot(limit: int = 800) -> dict[str, Any]:
    """Ring buffer plus on-disk log tails for the Console UI."""
    from h3_paths import default_logs_dir

    limit = max(50, min(int(limit), _MAX_LINES))
    ring = get_console_lines(limit)
    log_dir = default_logs_dir()
    paths = {
        "server": str(log_dir / "server.log"),
        "desktop": str(log_dir / "desktop.log"),
        "error": str(log_dir / "error.log"),
    }
    file_lines: list[str] = []
    for key in ("server", "desktop", "error"):
        path = Path(paths[key])
        for row in _tail_file(path, max_lines=120):
            file_lines.append(f"[{key}] {row}")
    # Prefer live ring; fall back to files when the process just started cold.
    lines = ring if ring else file_lines[-limit:]
    if ring and file_lines:
        # Append a short recent file tail so desktop-supervisor lines show up too.
        extra = [ln for ln in file_lines[-80:] if ln not in ring]
        if extra:
            lines = (ring + extra)[-limit:]
    return {
        "ok": True,
        "lines": lines,
        "count": len(lines),
        "log_dir": str(log_dir),
        "log_paths": paths,
    }


class _ConsoleLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        name = record.name or "log"
        if name.startswith("uvicorn.access"):
            return
        append_console("%s: %s", name, msg)


def install_console_logging() -> None:
    """Attach a root logging handler once (safe to call repeatedly)."""
    global _handler_installed
    if _handler_installed:
        return
    handler = _ConsoleLogHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)
    _handler_installed = True
    append_console("console capture ready")
