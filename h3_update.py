"""Check GitHub Releases for a newer H3-WS build."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from h3_paths import resource_root

log = logging.getLogger("h3-update")

# Product Continuity repo (releases + source of truth for the macOS DMG).
GITHUB_REPO = os.environ.get("H3_WS_GITHUB_REPO", "audiohacking/h3-ws").strip() or "audiohacking/h3-ws"
RELEASES_URL = f"https://github.com/{GITHUB_REPO}/releases"
API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


def installed_version() -> str:
    """Read the shipped VERSION file (Release builds write the tag here)."""
    env = os.environ.get("H3_WS_VERSION", "").strip()
    if env:
        return env.lstrip("vV")
    for candidate in (
        resource_root() / "VERSION",
        Path(__file__).resolve().parent / "VERSION",
    ):
        try:
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8").strip()
                if text:
                    return text.lstrip("vV")
        except OSError:
            continue
    return "0.0.0"


def parse_version(raw: str) -> tuple[int, ...]:
    text = (raw or "").strip().lstrip("vV")
    # Allow "0.0.1-macos" → 0.0.1
    text = re.split(r"[-+_]", text, maxsplit=1)[0]
    parts: list[int] = []
    for chunk in text.split("."):
        m = re.match(r"^(\d+)", chunk)
        parts.append(int(m.group(1)) if m else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:4])


def is_newer(remote: str, local: str) -> bool:
    return parse_version(remote) > parse_version(local)


def check_for_update(*, timeout_s: float = 4.0) -> dict[str, Any]:
    """Return update status against GitHub Releases (best-effort, never raises)."""
    local = installed_version()
    out: dict[str, Any] = {
        "ok": True,
        "repo": GITHUB_REPO,
        "releases_url": RELEASES_URL,
        "installed": local,
        "latest": None,
        "update_available": False,
        "html_url": RELEASES_URL,
        "name": None,
        "error": None,
    }
    req = urllib.request.Request(
        API_LATEST,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"h3-ws/{local}",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        out["ok"] = False
        out["error"] = f"GitHub HTTP {exc.code}"
        return out
    except Exception as exc:  # noqa: BLE001 — network is best-effort
        out["ok"] = False
        out["error"] = str(exc)
        return out

    tag = str(raw.get("tag_name") or "").strip()
    if not tag:
        out["ok"] = False
        out["error"] = "no tag_name on latest release"
        return out
    latest = tag.lstrip("vV")
    out["latest"] = latest
    out["name"] = str(raw.get("name") or tag)
    out["html_url"] = str(raw.get("html_url") or RELEASES_URL)
    out["update_available"] = is_newer(latest, local)
    if out["update_available"]:
        log.info("Update available: installed=%s latest=%s", local, latest)
    return out
