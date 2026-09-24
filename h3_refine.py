"""Optional OpenAI-compatible prompt Refine (remote only — no local VL weights)."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

log = logging.getLogger("h3-refine")

REFINE_SYSTEM = """\
You are the prompt pre-processing stage for MiniMax-H3, a video-and-audio \
generation model. Expand the user's short request into a detailed Context-IR \
description H3 reads well.

Write plain prose covering, in order when relevant:
- Scene: place, time of day, weather, atmosphere
- Action: what subjects do, beat by beat
- Camera: framing, move, lens feel
- Look: medium, palette, lighting, film/grade references
- Audio: diegetic sound, dialogue (if any), music mood

Keep every concrete detail the user named. Expand style names into visual \
signature (line, grain, proportions, palette, motion). Use present tense. \
Do not greet, refuse, or explain — return only the refined prompt text.
"""


def normalize_base_url(url: str) -> str:
    text = (url or "").strip().rstrip("/")
    if not text:
        return ""
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(
            "Refine base URL must start with http:// or https:// "
            "(e.g. http://127.0.0.1:1234/v1)"
        )
    for tail in ("/chat/completions", "/completions", "/models"):
        if text.endswith(tail):
            text = text[: -len(tail)]
    return text


def refine_settings_public(raw: dict[str, Any] | None) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    return {
        "enabled": bool(data.get("enabled")),
        "base_url": str(data.get("base_url") or ""),
        "model": str(data.get("model") or ""),
        "key_set": bool(str(data.get("api_key") or "").strip()),
    }


def merge_refine_settings(
    existing: dict[str, Any] | None, patch: dict[str, Any]
) -> dict[str, Any]:
    cur = dict(existing) if isinstance(existing, dict) else {}
    if "enabled" in patch:
        cur["enabled"] = bool(patch["enabled"])
    if "base_url" in patch:
        cur["base_url"] = normalize_base_url(str(patch.get("base_url") or ""))
    if "model" in patch:
        cur["model"] = str(patch.get("model") or "").strip()
    if "api_key" in patch:
        key = str(patch.get("api_key") or "")
        # Empty string clears; omit/None from clients that send key_set only
        # is handled by not including api_key in the patch.
        cur["api_key"] = key.strip()
    return cur


def refine_prompt(
    prompt: str,
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    mode: str = "t2va",
    refs: list[str] | None = None,
    timeout_s: float = 120.0,
) -> dict[str, str]:
    base = normalize_base_url(base_url)
    if not base:
        raise ValueError("Refine base URL is empty")
    model_name = (model or "").strip() or "default"
    text = (prompt or "").strip()
    if not text:
        raise ValueError("prompt is empty")

    user_parts = [f"<request>\n{text}\n</request>"]
    if mode:
        user_parts.append(f"Mode: {mode}")
    if refs:
        user_parts.append("References (in order):\n" + "\n".join(f"- {r}" for r in refs))
    user_parts.append("Return only the refined prompt.")

    payload = {
        "model": model_name,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": REFINE_SYSTEM},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ],
    }
    url = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Refine HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Refine unreachable: {exc.reason}") from exc

    refined = _extract_content(raw)
    if not refined:
        raise RuntimeError("Refine returned an empty reply")
    return {"refined": refined, "model": model_name}


def _extract_content(raw: dict[str, Any]) -> str:
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(str(block.get("text") or ""))
                    elif isinstance(block, str):
                        parts.append(block)
                return "\n".join(p for p in parts if p).strip()
        text = choices[0].get("text") if isinstance(choices[0], dict) else None
        if isinstance(text, str):
            return text.strip()
    return ""


# Well-known local OpenAI-compatible servers (LM Studio, Ollama, llama.cpp, …).
_DISCOVER_CANDIDATES: tuple[tuple[str, str, str], ...] = (
    ("LM Studio", "127.0.0.1", "1234"),
    ("Ollama", "127.0.0.1", "11434"),
    ("llama.cpp / Open WebUI", "127.0.0.1", "8080"),
    ("vLLM / OpenAI-compatible", "127.0.0.1", "8000"),
    ("Jan", "127.0.0.1", "1337"),
    ("LocalAI", "127.0.0.1", "8080"),
    ("TabbyAPI", "127.0.0.1", "5000"),
    ("text-generation-webui", "127.0.0.1", "5000"),
    ("KoboldCpp", "127.0.0.1", "5001"),
)


def _port_open(host: str, port: int, timeout_s: float = 0.25) -> bool:
    import socket

    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            return True
    except OSError:
        return False


def _probe_models(base_url: str, timeout_s: float = 1.5) -> list[str]:
    base = normalize_base_url(base_url)
    if not base:
        return []
    url = f"{base}/models"
    req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    models: list[str] = []
    data = raw.get("data") if isinstance(raw, dict) else None
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("id"):
                models.append(str(item["id"]))
            elif isinstance(item, str):
                models.append(item)
    return models[:24]


def _agent_cli_hints() -> list[dict[str, Any]]:
    """Non-HTTP agents installed on PATH (Codex CLI, ollama, …)."""
    import shutil

    hints: list[dict[str, Any]] = []
    checks = (
        (
            "codex",
            "OpenAI Codex CLI — start a local OpenAI-compatible proxy or paste its base URL.",
        ),
        (
            "ollama",
            "Ollama is installed — run `ollama serve` (default http://127.0.0.1:11434/v1).",
        ),
        (
            "lmstudio",
            "LM Studio CLI present — enable the local server (default http://127.0.0.1:1234/v1).",
        ),
    )
    for binary, note in checks:
        path = shutil.which(binary)
        if path:
            hints.append({"kind": "cli", "name": binary, "path": path, "note": note})
    return hints


def discover_refine_endpoints(*, timeout_s: float = 0.3) -> dict[str, Any]:
    """Scan localhost for OpenAI-compatible Refine backends."""
    found: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for name, host, port in _DISCOVER_CANDIDATES:
        if not _port_open(host, int(port), timeout_s=timeout_s):
            continue
        for suffix in ("/v1", ""):
            base = f"http://{host}:{port}{suffix}"
            try:
                normalized = normalize_base_url(base)
            except ValueError:
                continue
            if normalized in seen_urls:
                continue
            models = _probe_models(normalized, timeout_s=max(timeout_s * 4, 1.0))
            # Prefer /v1 when the port is open, even if /models needs auth.
            if models or suffix == "/v1":
                seen_urls.add(normalized)
                found.append(
                    {
                        "name": name,
                        "base_url": normalized,
                        "models": models,
                        "reachable": True,
                    }
                )
                break
    return {
        "ok": True,
        "endpoints": found,
        "cli_hints": _agent_cli_hints(),
        "note": (
            "Pick a discovered OpenAI-compatible server, or point Refine at any "
            "local Agent/Codex proxy that speaks /v1/chat/completions."
        ),
    }
