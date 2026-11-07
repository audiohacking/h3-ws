#!/usr/bin/env python3
"""Thin stdio MCP client for a running h3-ws server (H3WS_URL).

Talks HTTP to server.py — does not embed or spawn ``./h3``.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8765"


def _base() -> str:
    return (os.environ.get("H3WS_URL") or DEFAULT_URL).rstrip("/")


def _timeout() -> float:
    try:
        return float(os.environ.get("H3WS_MCP_TIMEOUT_S") or "1800")
    except ValueError:
        return 1800.0


def _http(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> Any:
    url = f"{_base()}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout or 60) as resp:
            raw = resp.read()
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"{method} {path} → {exc.code}: {detail[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach h3-ws at {_base()} ({exc.reason}). "
            "Start server.py / H3-WS.app first."
        ) from exc


def _upload_file(path: Path, kind: str) -> dict[str, Any]:
    boundary = f"----h3wsmcp{int(time.time() * 1000)}"
    file_bytes = path.read_bytes()
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts: list[bytes] = []
    for name, value in (("kind", kind),):
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode()
        + file_bytes
        + b"\r\n"
        + f"--{boundary}--\r\n".encode()
    )
    data = b"".join(parts)
    req = urllib.request.Request(
        f"{_base()}/api/upload",
        data=data,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _tool_result(payload: Any) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    return {"content": [{"type": "text", "text": text}]}


def _tool_error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "isError": True}


TOOLS: list[dict[str, Any]] = [
    {
        "name": "h3_health",
        "description": "Check that h3-ws is up and report engine / models readiness.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "h3_list_workspaces",
        "description": "List workspace plugins (Create / Enhance / future recipe tabs).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "h3_list_library",
        "description": "List library assets. kind: image | video | audio | renders.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "default": "image"},
                "q": {"type": "string", "default": ""},
            },
        },
    },
    {
        "name": "h3_upload",
        "description": "Upload a local file into the h3-ws library.",
        "inputSchema": {
            "type": "object",
            "required": ["path", "kind"],
            "properties": {
                "path": {"type": "string"},
                "kind": {"type": "string", "enum": ["image", "video", "audio"]},
            },
        },
    },
    {
        "name": "h3_improve_prompt",
        "description": "Rewrite a prompt via the server Refine endpoint (must be enabled in Features).",
        "inputSchema": {
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {"type": "string"},
                "mode": {"type": "string", "default": "t2va"},
                "refs": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "name": "h3_generate",
        "description": (
            "Queue a generation. Modes: t2va, first_frame, last_frame, fl2va, ref2va. "
            "Pass refs for Ref2VA; image_path / end_image_path for FL2VA anchors. "
            "Do not mix refs with first/last frames."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {"type": "string"},
                "mode": {"type": "string", "default": "t2va"},
                "refs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string"},
                            "path": {"type": "string"},
                            "name": {"type": "string"},
                        },
                        "required": ["kind", "path"],
                    },
                },
                "image_path": {"type": "string"},
                "end_image_path": {"type": "string"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "num_frames": {"type": "integer"},
                "quality": {"type": "string"},
                "num_steps": {"type": "integer"},
                "layers": {"type": "integer"},
                "reuse": {"type": "integer"},
                "seed": {"type": "integer"},
                "loras": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "spec": {"type": "string"},
                            "scale": {"type": "number"},
                        },
                    },
                },
            },
        },
    },
    {
        "name": "h3_enhance",
        "description": "Run Upframe and/or ReDetail on a finished library clip id.",
        "inputSchema": {
            "type": "object",
            "required": ["clip_id"],
            "properties": {
                "clip_id": {"type": "string"},
                "upframe": {"type": "boolean", "default": True},
                "upframe_multiplier": {"type": "integer", "default": 2},
                "redetail": {"type": "boolean", "default": False},
                "redetail_denoise": {"type": "number", "default": 0.45},
                "redetail_seed": {"type": "integer", "default": 42},
            },
        },
    },
    {
        "name": "h3_wait",
        "description": "Block until a run finishes (or timeout). Returns clip status / video URL.",
        "inputSchema": {
            "type": "object",
            "required": ["run_id"],
            "properties": {
                "run_id": {"type": "string"},
                "max_s": {"type": "number"},
            },
        },
    },
    {
        "name": "h3_cancel",
        "description": "Cancel a queued or running generate.",
        "inputSchema": {
            "type": "object",
            "required": ["run_id"],
            "properties": {"run_id": {"type": "string"}},
        },
    },
    {
        "name": "h3_status",
        "description": "Snapshot of recent clips and optional run_id status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "limit": {"type": "integer", "default": 12},
            },
        },
    },
]


def call_tool(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    args = arguments or {}
    try:
        if name == "h3_health":
            health = _http("GET", "/api/health")
            cfg = _http("GET", "/api/config")
            return _tool_result(
                {
                    "health": health,
                    "engine_ok": cfg.get("engine_ok"),
                    "models_ready": cfg.get("models_ready"),
                    "ref2va_ready": cfg.get("ref2va_ready"),
                    "server_url": cfg.get("server_url"),
                    "web_url": cfg.get("web_url"),
                    "workspaces": [w.get("id") for w in (cfg.get("workspaces") or [])],
                }
            )
        if name == "h3_list_workspaces":
            cfg = _http("GET", "/api/config")
            return _tool_result(cfg.get("workspaces") or [])
        if name == "h3_list_library":
            kind = str(args.get("kind") or "image")
            q = str(args.get("q") or "")
            qs = urllib.parse.urlencode({"kind": kind, "q": q})
            return _tool_result(_http("GET", f"/api/library?{qs}"))
        if name == "h3_upload":
            path = Path(str(args.get("path") or "")).expanduser()
            kind = str(args.get("kind") or "")
            if not path.is_file():
                return _tool_error(f"file not found: {path}")
            if kind not in ("image", "video", "audio"):
                return _tool_error("kind must be image|video|audio")
            return _tool_result(_upload_file(path, kind))
        if name == "h3_improve_prompt":
            body = {
                "prompt": str(args.get("prompt") or ""),
                "mode": str(args.get("mode") or "t2va"),
                "refs": args.get("refs") or [],
            }
            return _tool_result(_http("POST", "/api/refine", body=body))
        if name == "h3_generate":
            body: dict[str, Any] = {
                "prompt": str(args.get("prompt") or "").strip(),
                "mode": str(args.get("mode") or "t2va").strip().lower(),
            }
            if not body["prompt"]:
                return _tool_error("prompt is required")
            for key in (
                "refs",
                "image_path",
                "end_image_path",
                "width",
                "height",
                "num_frames",
                "quality",
                "num_steps",
                "layers",
                "reuse",
                "seed",
                "loras",
            ):
                if key in args and args[key] is not None:
                    body[key] = args[key]
            return _tool_result(_http("POST", "/api/generate", body=body, timeout=120))
        if name == "h3_enhance":
            clip_id = str(args.get("clip_id") or "").strip()
            if not clip_id:
                return _tool_error("clip_id is required")
            out: dict[str, Any] = {"clip_id": clip_id, "steps": []}
            working = clip_id
            if args.get("redetail"):
                body = {
                    "denoise": float(args.get("redetail_denoise") or 0.45),
                    "seed": int(args.get("redetail_seed") or 42),
                }
                rd = _http("POST", f"/api/clips/{working}/redetail", body=body, timeout=120)
                out["steps"].append({"redetail": rd})
                working = str(rd.get("clip_id") or rd.get("id") or working)
            if args.get("upframe", True):
                body = {"multiplier": int(args.get("upframe_multiplier") or 2)}
                uf = _http("POST", f"/api/clips/{working}/upframe", body=body, timeout=120)
                out["steps"].append({"upframe": uf})
                working = str(uf.get("clip_id") or uf.get("id") or working)
            out["result_clip_id"] = working
            return _tool_result(out)
        if name == "h3_wait":
            run_id = str(args.get("run_id") or "").strip()
            if not run_id:
                return _tool_error("run_id is required")
            max_s = float(args.get("max_s") or _timeout())
            deadline = time.time() + max_s
            last: Any = None
            while time.time() < deadline:
                clips = _http("GET", "/api/clips")
                items = clips if isinstance(clips, list) else clips.get("clips") or []
                related = [
                    c
                    for c in items
                    if str(c.get("run_id") or "") == run_id
                    or run_id in (c.get("run_id"), c.get("id"))
                ]
                if related:
                    last = related
                    statuses = {str(c.get("status") or "") for c in related}
                    if statuses and statuses <= {"done", "failed", "cancelled", "error"}:
                        return _tool_result({"run_id": run_id, "clips": related})
                time.sleep(2.0)
            return _tool_result(
                {"run_id": run_id, "timed_out": True, "clips": last or []}
            )
        if name == "h3_cancel":
            run_id = str(args.get("run_id") or "").strip()
            return _tool_result(_http("POST", f"/api/runs/{run_id}/cancel", body={}))
        if name == "h3_status":
            clips = _http("GET", "/api/clips")
            items = clips if isinstance(clips, list) else clips.get("clips") or []
            limit = max(1, min(50, int(args.get("limit") or 12)))
            payload: dict[str, Any] = {"clips": items[:limit]}
            run_id = str(args.get("run_id") or "").strip()
            if run_id:
                payload["run_clips"] = [
                    c for c in items if str(c.get("run_id") or "") == run_id
                ]
            return _tool_result(payload)
        return _tool_error(f"unknown tool: {name}")
    except Exception as exc:  # noqa: BLE001 — surface to MCP client
        return _tool_error(str(exc))


def main(argv: list[str] | None = None) -> int:
    import asyncio

    parser = argparse.ArgumentParser(description="h3-ws MCP (stdio)")
    parser.add_argument(
        "--url",
        default=None,
        help="Override H3WS_URL (default http://127.0.0.1:8765)",
    )
    args = parser.parse_args(argv)
    if args.url:
        os.environ["H3WS_URL"] = args.url.rstrip("/")

    try:
        import mcp.types as types
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
    except ImportError:
        print("mcp package required: pip install 'mcp>=1.0'", file=sys.stderr)
        return 1

    async def on_list_tools(_ctx, _params):  # type: ignore[no-untyped-def]
        tools = [
            types.Tool(
                name=str(t["name"]),
                description=str(t.get("description") or ""),
                inputSchema=t.get("inputSchema") or {"type": "object", "properties": {}},
            )
            for t in TOOLS
        ]
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(_ctx, params):  # type: ignore[no-untyped-def]
        name = str(params.name)
        arguments = dict(params.arguments or {})
        # HTTP helpers are sync — run off the event loop.
        result = await asyncio.to_thread(call_tool, name, arguments)
        content = [
            types.TextContent(type="text", text=str(c.get("text") or ""))
            for c in (result.get("content") or [])
            if isinstance(c, dict)
        ]
        return types.CallToolResult(
            content=content or [types.TextContent(type="text", text="")],
            isError=bool(result.get("isError")),
        )

    server = Server(
        "h3-ws",
        instructions=(
            "Thin MCP for local MiniMax-H3 via a running h3-ws server. "
            "See mcp/AGENTS.md for modes and prompting."
        ),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )

    async def run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
