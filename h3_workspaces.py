"""Declarative workspace plugins (tabs) for h3-ws Web UI + MCP."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("h3")

_REQUIRED = ("id", "label", "panel")
_PANELS = frozenset({"composer", "recipe", "enhance"})


def shipped_workspaces_dir() -> Path:
    return Path(__file__).resolve().parent / "workspaces"


def user_workspaces_dir() -> Path | None:
    """Optional overrides: ``~/Library/Application Support/H3-WS/workspaces``."""
    try:
        from h3_paths import writable_root

        return writable_root() / "workspaces"
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("workspace manifest %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        return None
    return data


def _normalize(raw: dict[str, Any], *, source: str) -> dict[str, Any] | None:
    wid = str(raw.get("id") or "").strip()
    label = str(raw.get("label") or "").strip()
    panel = str(raw.get("panel") or "").strip()
    if not wid or not label or panel not in _PANELS:
        log.warning("workspace %s: missing id/label or bad panel=%r", source, panel)
        return None
    isolation = raw.get("isolation") if isinstance(raw.get("isolation"), dict) else {}
    engine = raw.get("engine") if isinstance(raw.get("engine"), dict) else {}
    fields = raw.get("fields") if isinstance(raw.get("fields"), list) else []
    submit = raw.get("submit") if isinstance(raw.get("submit"), dict) else {}
    enabled = raw.get("enabled", True)
    return {
        "id": wid,
        "label": label,
        "hint": str(raw.get("hint") or "").strip(),
        "order": int(raw.get("order") or 100),
        "enabled": bool(enabled),
        "panel": panel,
        "isolation": {
            "lora_categories": isolation.get("lora_categories"),
            "forbid": list(isolation.get("forbid") or []),
            "hide_from_full_lora": bool(isolation.get("hide_from_full_lora")),
        },
        "engine": dict(engine),
        "fields": [f for f in fields if isinstance(f, dict)],
        "submit": dict(submit),
        "docs": raw.get("docs"),
        "source": source,
    }


def _scan_dir(root: Path | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if root is None or not root.is_dir():
        return out
    for path in sorted(root.glob("*.json")):
        raw = _load_json(path)
        if not raw:
            continue
        norm = _normalize(raw, source=str(path))
        if norm and norm.get("enabled", True):
            out[norm["id"]] = norm
    return out


def list_workspaces() -> list[dict[str, Any]]:
    """Shipped manifests, then user overrides (same id replaces)."""
    merged = _scan_dir(shipped_workspaces_dir())
    merged.update(_scan_dir(user_workspaces_dir()))
    items = list(merged.values())
    items.sort(key=lambda w: (int(w.get("order") or 100), str(w.get("id") or "")))
    return items


def workspace_by_id(workspace_id: str) -> dict[str, Any] | None:
    want = str(workspace_id or "").strip()
    if not want:
        return None
    for item in list_workspaces():
        if item.get("id") == want:
            return item
    return None
