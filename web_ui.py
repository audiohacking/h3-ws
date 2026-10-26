"""HTTP API, library, and job orchestration for h3-ws."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
import zipfile
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from h3_bootstrap import ensure_python_requirements

ensure_python_requirements()
from starlette.requests import Request  # noqa: E402

from h3_backend import (
    GENERATION_MODES,
    H3_DEFAULT_LAYERS,
    H3_DEFAULT_REUSE,
    H3_DEFAULT_STEPS,
    QUALITY_PRESET_LIST,
    GenerateRequest,
    GenerationCancelledError,
    H3Engine,
    LoraRef,
    fl2va_dir,
    model_layout_ok,
    parse_refs_payload,
    ram_gb,
    recommend_ssd_streaming,
    ref2va_dir,
    validate_refs,
)
from h3_media import (
    DURATION_PRESETS,
    FPS,
    RESOLUTION_PRESETS,
    adapt_audio_for_h3,
    audio_peaks,
    concat_mp4s,
    ensure_video_poster,
    extract_last_frame,
    frame_video,
    media_available,
    probe_duration_seconds,
    probe_media,
    require_ui_canvas,
    resize_still_to_canvas,
    sanitize_filename,
    seconds_to_frames,
    snap_frames,
    trim_media,
    validate_canvas,
    video_poster_path,
)
from h3_paths import (
    REPO_ROOT,
    configure_scratch_root,
    default_output_dir,
    default_upload_dir,
    is_frozen,
    mk_scratch_dir,
    resource_root,
)
from h3_preview import (
    LatentPreviewWatcher,
    cleanup_preview_artifacts,
    taeh3_available,
    taeh3_decode_ready,
)
from h3_refine import (
    discover_refine_endpoints,
    merge_refine_settings,
    refine_prompt,
    refine_settings_public,
)
from h3_lora import (
    ensure_lora,
    lora_catalog,
    normalize_lora_spec,
    read_custom_loras,
    write_custom_loras,
    _label_for_spec,
)

log = logging.getLogger("h3-web")

INDEX_FILE = "index.json"
SETTINGS_FILE = "settings.json"
USER_DATA_FILE = "user_data.json"
CLIP_MULTIPLIER_MAX = 10
DEFAULT_OUTPUT_DIR = default_output_dir()
DEFAULT_UPLOAD_DIR = default_upload_dir()
PROGRESS_KEEPALIVE_INTERVAL_S = 1.0

_RUN_BODIES: dict[str, dict[str, Any]] = {}


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ClipRecord:
    id: str
    prompt: str
    label: str
    video_url: str
    filename: str
    chain_id: str
    clip_index: int
    mode: str
    status: str
    created_at: str
    elapsed_s: Optional[float] = None
    bytes: Optional[int] = None
    error: Optional[str] = None
    num_frames: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    num_steps: Optional[int] = None
    layers: Optional[int] = None
    reuse: Optional[int] = None
    duration_seconds: Optional[float] = None
    clip_count: Optional[int] = None
    autocontinue: Optional[bool] = None
    autoconcat: Optional[bool] = None
    quality: Optional[str] = None
    loras: Optional[list[dict[str, Any]]] = None
    project_id: Optional[str] = None
    # Full composer snapshot so Library can re-run the generation (prompt with
    # @handles, refs + trim/crop, mode/routing, sampler, LoRAs, anchors…).
    generation: Optional[dict[str, Any]] = None
    # Wall time per h3.c phase (+ --profile Metal summaries), kept for done,
    # failed and cancelled clips so slow stages can be inspected later.
    timings: Optional[dict[str, Any]] = None
    # Owning run — library queue cards use this to follow / cancel.
    run_id: Optional[str] = None


@dataclass
class ProjectRecord:
    id: str
    name: str
    created_at: str
    updated_at: str = ""


@dataclass
class RunRecord:
    id: str
    status: str
    prompts: list[str]
    chain_id: str
    clip_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    error: Optional[str] = None
    autocontinue: bool = False
    autoconcat: bool = False
    merged_url: Optional[str] = None
    merged_clip_id: Optional[str] = None
    # Optional Long take refine (default off) → new standalone library clip.
    longtake: Optional[dict[str, Any]] = None
    # Independent post-passes (default off) — each yields its own library clip.
    redetail: Optional[dict[str, Any]] = None
    upframe: Optional[dict[str, Any]] = None
    # Continuity Faces — new standalone clip; source kept.
    faces: Optional[dict[str, Any]] = None


class AppState:
    def __init__(
        self,
        output_dir: Path,
        upload_dir: Path,
        engine: H3Engine,
        *,
        embedded: bool = True,
        http_url: str = "",
        server_url: str = "",
        runtime_defaults: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.upload_dir = Path(upload_dir)
        self.engine = engine
        from h3_selftest import SelfTestRunner

        # Debug panel self-test; generation waits while it holds the GPU.
        self.selftest = SelfTestRunner(engine)
        self.embedded = embedded
        self.http_url = http_url
        self.server_url = server_url
        self.runtime_defaults = runtime_defaults or {}
        configure_scratch_root(self.output_dir / ".scratch")
        self.runs: dict[str, RunRecord] = {}
        self.clips: dict[str, ClipRecord] = {}
        self.event_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._pending: asyncio.Queue[str] = asyncio.Queue()
        self._submit_lock = asyncio.Lock()
        self._worker_started = False
        self._worker_task: asyncio.Task[None] | None = None
        self._cancelled_runs: set[str] = set()
        self._active_run_id: str | None = None
        # Live status for library passes (Upframe / ReDetail / …) — polled by the UI.
        self._pass_progress: dict[str, Any] | None = None
        self._sigint_count = 0
        self._sigint_last_ts = 0.0
        self._uvicorn_server: Any = None
        self.user_data_path = self.output_dir / USER_DATA_FILE
        self.presets: list[dict[str, Any]] = []
        self.cast_members: list[dict[str, Any]] = []
        self.projects: dict[str, ProjectRecord] = {}
        self.active_project_id: str | None = None
        self._load_user_data()

    def set_pass_progress(
        self,
        phase: str,
        extra: dict[str, Any] | None = None,
        *,
        kind: str | None = None,
        clip_id: str | None = None,
    ) -> None:
        """Publish library-pass progress for the UI poller."""
        extra = dict(extra or {})
        msg = str(extra.get("message") or phase)
        payload: dict[str, Any] = {
            "idle": False,
            "phase": phase,
            "message": msg,
            "updated_at": time.time(),
        }
        if kind:
            payload["kind"] = kind
        if clip_id:
            payload["clip_id"] = clip_id
        idx = extra.get("index", extra.get("step"))
        total = extra.get("total")
        if idx is not None:
            payload["step"] = idx
        if total is not None:
            payload["total"] = total
        if payload.get("step") is not None and payload.get("total"):
            try:
                payload["pct"] = round(
                    100.0 * float(payload["step"]) / max(1.0, float(payload["total"])),
                    1,
                )
            except (TypeError, ValueError):
                pass
        mp = extra.get("model_progress")
        if isinstance(mp, dict):
            payload["model_progress"] = mp
            if mp.get("step") is not None:
                payload["step"] = mp["step"]
            if mp.get("total") is not None:
                payload["total"] = mp["total"]
            if payload.get("step") is not None and payload.get("total"):
                try:
                    payload["pct"] = round(
                        100.0
                        * float(payload["step"])
                        / max(1.0, float(payload["total"])),
                        1,
                    )
                except (TypeError, ValueError):
                    pass
            # Prefer a composed message when nested denoise is active.
            stage = str(mp.get("stage") or "").strip()
            bits = [msg]
            if stage and stage.lower() not in msg.lower():
                bits.append(stage)
            if mp.get("step") is not None and mp.get("total") is not None:
                bits.append(f"{mp['step']}/{mp['total']}")
            eta = mp.get("eta_s")
            if eta is not None:
                eta_i = max(0, int(round(float(eta))))
                bits.append(f"{eta_i // 60:02d}:{eta_i % 60:02d} remaining")
            payload["message"] = " · ".join(bits)
        self._pass_progress = payload

    def clear_pass_progress(self) -> None:
        self._pass_progress = None

    # ── User data (presets + cast + projects) persistence ────────────────────
    # Stored separately from the generation index so presets/cast/projects
    # survive server restarts and are not wiped by session clear.

    def _load_user_data(self) -> None:
        try:
            if not self.user_data_path.is_file():
                self.ensure_projects()
                return
            data = json.loads(self.user_data_path.read_text(encoding="utf-8"))
            self.presets = data.get("presets", []) or []
            self.cast_members = data.get("cast", []) or []
            self.projects = {}
            for raw in data.get("projects", []) or []:
                if not isinstance(raw, dict) or not raw.get("id"):
                    continue
                pid = str(raw["id"])
                self.projects[pid] = ProjectRecord(
                    id=pid,
                    name=str(raw.get("name") or "Untitled").strip() or "Untitled",
                    created_at=str(raw.get("created_at") or datetime.now().isoformat()),
                    updated_at=str(raw.get("updated_at") or ""),
                )
            active = data.get("active_project_id")
            self.active_project_id = str(active) if active else None
            self.ensure_projects()
        except (json.JSONDecodeError, OSError, TypeError, KeyError):
            log.warning("Could not read user data at %s", self.user_data_path)
            self.ensure_projects()

    def _save_user_data(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "presets": self.presets,
                "cast": self.cast_members,
                "projects": [asdict(p) for p in self.projects.values()],
                "active_project_id": self.active_project_id,
            }
            self.user_data_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("Could not save user data: %s", exc)

    def ensure_projects(self) -> ProjectRecord:
        """Guarantee at least one project and a valid active_project_id."""
        if not self.projects:
            now = datetime.now().isoformat()
            pid = str(uuid.uuid4())
            project = ProjectRecord(
                id=pid, name="Project 1", created_at=now, updated_at=now
            )
            self.projects[pid] = project
            self.active_project_id = pid
            self._save_user_data()
            return project
        if not self.active_project_id or self.active_project_id not in self.projects:
            self.active_project_id = next(iter(self.projects))
            self._save_user_data()
        return self.projects[self.active_project_id]

    def list_projects(self) -> list[dict[str, Any]]:
        self.ensure_projects()
        out: list[dict[str, Any]] = []
        for project in sorted(
            self.projects.values(), key=lambda p: p.created_at or p.id
        ):
            out.append(
                {
                    **asdict(project),
                    "clip_count": len(self.clips_for_project(project.id)),
                    "active": project.id == self.active_project_id,
                }
            )
        return out

    def create_project(self, name: str | None = None) -> ProjectRecord:
        self.ensure_projects()
        now = datetime.now().isoformat()
        n = len(self.projects) + 1
        label = (name or "").strip() or f"Project {n}"
        project = ProjectRecord(
            id=str(uuid.uuid4()), name=label, created_at=now, updated_at=now
        )
        self.projects[project.id] = project
        self.active_project_id = project.id
        self._save_user_data()
        return project

    def rename_project(self, project_id: str, name: str) -> ProjectRecord | None:
        project = self.projects.get(project_id)
        if not project:
            return None
        label = name.strip()
        if not label:
            return None
        project.name = label
        project.updated_at = datetime.now().isoformat()
        self._save_user_data()
        return project

    def set_active_project(self, project_id: str) -> ProjectRecord | None:
        project = self.projects.get(project_id)
        if not project:
            return None
        self.active_project_id = project_id
        project.updated_at = datetime.now().isoformat()
        self._save_user_data()
        return project

    def delete_project(self, project_id: str, *, delete_files: bool = True) -> dict[str, Any]:
        self.ensure_projects()
        if project_id not in self.projects:
            return {"ok": False, "error": "not_found"}
        if len(self.projects) <= 1:
            return {"ok": False, "error": "last_project"}
        default_id = next(iter(self.projects))
        removed = 0
        for clip_id, clip in list(self.clips.items()):
            belongs = clip.project_id == project_id or (
                not clip.project_id and project_id == default_id
            )
            if not belongs:
                continue
            if delete_files:
                if self.delete_clip_record(clip_id):
                    removed += 1
            else:
                del self.clips[clip_id]
                removed += 1
        del self.projects[project_id]
        if self.active_project_id == project_id:
            self.active_project_id = next(iter(self.projects))
        self.save_index()
        self._save_user_data()
        return {
            "ok": True,
            "deleted": project_id,
            "deleted_clips": removed,
            "active_project_id": self.active_project_id,
        }

    def clips_for_project(self, project_id: str | None = None) -> list[ClipRecord]:
        self.ensure_projects()
        pid = project_id or self.active_project_id
        if not pid:
            return list(self.clips.values())
        default_id = next(iter(self.projects))
        out: list[ClipRecord] = []
        for clip in self.clips.values():
            if clip.project_id == pid:
                out.append(clip)
            elif not clip.project_id and pid == default_id:
                out.append(clip)
        return out

    def touch_project(self, project_id: str | None) -> None:
        if not project_id:
            return
        project = self.projects.get(project_id)
        if project:
            project.updated_at = datetime.now().isoformat()
            self._save_user_data()

    def is_generation_active(self) -> bool:
        return self._active_run_id is not None

    def is_pipeline_idle(self) -> bool:
        return self._active_run_id is None and self._pending.qsize() == 0

    async def enqueue_generation_run(self, run_id: str) -> bool:
        async with self._submit_lock:
            idle = self.is_pipeline_idle()
            run = self.runs.get(run_id)
            if run is not None:
                run.status = RunStatus.RUNNING.value if idle else RunStatus.QUEUED.value
            await self._pending.put(run_id)
            return idle

    def request_shutdown(self) -> None:
        uv = self._uvicorn_server
        if uv is not None:
            uv.should_exit = True

    def on_console_interrupt(self) -> None:
        if not self.is_generation_active():
            log.info("Shutting down…")
            self.request_shutdown()
            return
        now = time.monotonic()
        if now - self._sigint_last_ts > 2.0:
            self._sigint_count = 0
        self._sigint_last_ts = now
        self._sigint_count += 1
        self.engine.request_cancel()
        if self._active_run_id:
            self._cancelled_runs.add(self._active_run_id)
        if self._sigint_count == 1:
            log.warning(
                "Interrupt received — cancelling generation "
                "(press Ctrl+C again within 2s to force quit)"
            )
        else:
            log.warning("Force quit")
            self.engine.shutdown(wait=False)
            os._exit(130)

    def is_run_cancelled(self, run_id: str) -> bool:
        return run_id in self._cancelled_runs

    def request_cancel_run(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        if not run:
            return False
        if run.status in (
            RunStatus.DONE.value,
            RunStatus.FAILED.value,
            RunStatus.CANCELLED.value,
        ):
            return False
        self._cancelled_runs.add(run_id)
        if self._active_run_id == run_id:
            self.engine.request_cancel()
        return True

    def ensure_worker(self) -> None:
        task = self._worker_task
        if self._worker_started and task is not None and not task.done():
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.load_index()
        self._worker_task = asyncio.create_task(_worker_loop(self))
        self._worker_started = True

    def load_index(self) -> None:
        path = self.output_dir / INDEX_FILE
        if not path.exists():
            self.ensure_projects()
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for c in data.get("clips", []):
                self.clips[c["id"]] = ClipRecord(
                    **{k: v for k, v in c.items() if k in ClipRecord.__dataclass_fields__}
                )
            for r in data.get("runs", []):
                self.runs[r["id"]] = RunRecord(
                    **{k: v for k, v in r.items() if k in RunRecord.__dataclass_fields__}
                )
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        # Nothing is queued when the server starts, so a run or clip still
        # marked queued/running was interrupted (app closed or crashed).
        interrupted = "interrupted: H3-WS closed during generation"
        busy = {RunStatus.QUEUED.value, RunStatus.RUNNING.value}
        stale = False
        for record in [*self.runs.values(), *self.clips.values()]:
            if record.status in busy:
                record.status = RunStatus.FAILED.value
                record.error = record.error or interrupted
                stale = True
        # Assign legacy clips (no project_id) to the oldest project.
        self.ensure_projects()
        default_id = next(iter(self.projects))
        migrated = False
        for clip in self.clips.values():
            if not clip.project_id:
                clip.project_id = default_id
                migrated = True
        if migrated or stale:
            self.save_index()

    def save_index(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / INDEX_FILE
        data = {
            "clips": [asdict(c) for c in self.clips.values()],
            "runs": [asdict(r) for r in self.runs.values()],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def clip_url(self, filename: str) -> str:
        return f"/api/videos/{filename}"

    def clip_thumb_url(self, filename: str) -> str:
        return f"/api/videos/{filename}/thumb"

    def delete_clip_record(self, clip_id: str) -> bool:
        """Remove clip from the index and wipe its media so disk space returns.

        Safe while other jobs run — only this clip's files are touched. Active
        GPU work for a *different* run is unaffected.
        """
        clip = self.clips.get(clip_id)
        if not clip:
            return False
        filename = str(clip.filename or "").strip()
        if filename:
            path = self.output_dir / Path(filename).name
            if path.is_file():
                try:
                    path.unlink()
                except OSError as exc:
                    log.warning("Could not delete clip file %s: %s", path, exc)
            poster = video_poster_path(path)
            if poster.is_file():
                try:
                    poster.unlink()
                except OSError:
                    pass
            # Drop empty thumbs dir noise; ignore if other posters remain.
            thumbs = poster.parent
            if thumbs.is_dir() and thumbs.name == ".thumbs":
                try:
                    next(thumbs.iterdir())
                except StopIteration:
                    try:
                        thumbs.rmdir()
                    except OSError:
                        pass
        # Live / Faces preview artifacts keyed by clip id.
        preview_root = self.output_dir / ".preview"
        if preview_root.is_dir():
            for entry in list(preview_root.iterdir()):
                if clip_id not in entry.name:
                    continue
                try:
                    if entry.is_file():
                        entry.unlink()
                    elif entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                except OSError as exc:
                    log.debug("preview wipe %s: %s", entry, exc)
        del self.clips[clip_id]
        for run in list(self.runs.values()):
            if clip_id in run.clip_ids:
                run.clip_ids = [cid for cid in run.clip_ids if cid != clip_id]
            if not run.clip_ids and run.status in (
                RunStatus.QUEUED.value,
                RunStatus.RUNNING.value,
            ):
                # Clip was the only member — request cancel so the worker no-ops.
                try:
                    self.request_cancel_run(run.id)
                except Exception as exc:  # noqa: BLE001
                    log.debug("cancel after delete %s: %s", run.id, exc)
        return True

    def delete_chain(self, chain_id: str) -> int:
        removed = 0
        for clip_id, clip in list(self.clips.items()):
            if clip.chain_id == chain_id:
                if self.delete_clip_record(clip_id):
                    removed += 1
        for run_id, run in list(self.runs.items()):
            if run.chain_id == chain_id:
                del self.runs[run_id]
        self.save_index()
        return removed

    def clear_session(self) -> dict[str, int]:
        """Clear clips in the active project only (other projects stay intact)."""
        self.ensure_projects()
        active = self.active_project_id
        deleted_files = 0
        deleted_clips = 0
        for clip in list(self.clips_for_project(active)):
            if self.delete_clip_record(clip.id):
                deleted_clips += 1
                deleted_files += 1
        # Drop runs that no longer have any remaining clips in this project.
        for run_id, run in list(self.runs.items()):
            if any(cid in self.clips for cid in run.clip_ids):
                continue
            del self.runs[run_id]
        self.save_index()
        self.touch_project(active)
        return {"deleted_clips": deleted_clips, "deleted_files": deleted_files}

    async def emit(self, run_id: str, event: dict[str, Any]) -> None:
        q = self.event_queues.get(run_id)
        if q:
            await q.put(event)


def _clip_for_api(state: AppState, clip: ClipRecord) -> dict[str, Any]:
    data = asdict(clip)
    filename = str(data.get("filename") or "").strip()
    if filename:
        file_path = state.output_dir / filename
        if file_path.is_file():
            data["path"] = str(file_path)
            if not data.get("video_url"):
                data["video_url"] = state.clip_url(filename)
            # JPEG poster URL — WebKit leaves <video preload=metadata> black on
            # non-faststart MP4s (moov at end), which is what h3-av writes.
            data["thumb_url"] = state.clip_thumb_url(filename)
        else:
            data["video_url"] = ""
            data["thumb_url"] = ""
    return data


def read_web_settings(output_dir: Path) -> dict[str, Any]:
    path = output_dir / SETTINGS_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def write_web_settings(output_dir: Path, data: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SETTINGS_FILE).write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def _purge_preview_stem(stem: Path) -> None:
    """Remove served looping preview files for one run/clip stem."""
    parent = stem.parent
    if not parent.is_dir():
        return
    for sib in parent.glob(stem.name + ".*"):
        cleanup_preview_artifacts(sib)
    cleanup_preview_artifacts(stem)


def _promote_video_over_baseline(baseline: Path, refined: Path) -> Path:
    """Replace ``baseline`` with ``refined`` (test helper / rare reclaim).

    Product library passes must **not** use this on a user's original — they
    spawn a new file + ClipRecord via ``_spawn_processed_clip`` instead.
    """
    baseline = Path(baseline)
    refined = Path(refined)
    if not refined.is_file():
        raise FileNotFoundError(f"refined video missing: {refined}")
    if refined.resolve() == baseline.resolve():
        return baseline
    tmp = baseline.with_suffix(".promoting.mp4")
    if tmp.exists():
        tmp.unlink()
    refined.replace(tmp)
    if baseline.is_file():
        baseline.unlink()
    tmp.replace(baseline)
    poster = video_poster_path(baseline)
    if poster.is_file():
        try:
            poster.unlink()
        except OSError:
            pass
    return baseline


def _unique_pass_filename(output_dir: Path, source_filename: str, tag: str) -> str:
    """``stem_tag.mp4``, or ``stem_tag_<hex>.mp4`` if that name already exists."""
    stem = Path(source_filename).stem
    candidate = f"{stem}_{tag}.mp4"
    if not (output_dir / candidate).is_file():
        return candidate
    return f"{stem}_{tag}_{uuid.uuid4().hex[:8]}.mp4"


_PASS_LABELS = {
    "upframe": "UPFRAME",
    "redetail": "REDETAIL",
    "longtake": "LONGTAKE",
    "faces": "FACES",
}


def _pass_label_from_patch(generation_patch: dict[str, Any] | None) -> str:
    if not generation_patch:
        return "CURRENT"
    for key, label in _PASS_LABELS.items():
        if key not in generation_patch:
            continue
        val = generation_patch[key]
        if val is True or (isinstance(val, dict) and val.get("enabled", True)):
            return label
    return "CURRENT"


def _spawn_processed_clip(
    state: "AppState",
    source: "ClipRecord",
    out_path: Path,
    *,
    generation_patch: dict[str, Any] | None = None,
    label: str | None = None,
    width: int | None = None,
    height: int | None = None,
    num_frames: int | None = None,
    seed: int | None = None,
    run_id: str | None = None,
) -> "ClipRecord":
    """Register a new standalone library clip. Source clip/file stay untouched.

    Own ``chain_id`` so the UI never treats this as a parent/child of the source.
    ``label`` tags the pass (UPFRAME / REDETAIL / …) so cards are not anonymous copies.
    """
    new_id = str(uuid.uuid4())
    filename = out_path.name
    gen = dict(source.generation or {}) if isinstance(source.generation, dict) else {}
    if generation_patch:
        gen.update(generation_patch)
    gen["source_clip_id"] = source.id
    tag = (label or _pass_label_from_patch(generation_patch)).strip() or "CURRENT"
    gen["pass"] = tag.lower()
    bust = int(time.time() * 1000)
    size: int | None = None
    if out_path.is_file():
        st = out_path.stat()
        size = st.st_size
        bust = int(st.st_mtime_ns // 1_000_000)
    record = ClipRecord(
        id=new_id,
        prompt=source.prompt,
        label=tag,
        video_url=f"{state.clip_url(filename)}?v={bust}",
        filename=filename,
        chain_id=str(uuid.uuid4()),
        clip_index=0,
        mode=source.mode,
        status=RunStatus.DONE.value,
        created_at=datetime.now().isoformat(),
        bytes=size,
        project_id=source.project_id,
        width=int(width) if width is not None else source.width,
        height=int(height) if height is not None else source.height,
        generation=gen,
        run_id=run_id if run_id is not None else source.run_id,
        num_frames=int(num_frames) if num_frames is not None else source.num_frames,
        seed=int(seed) if seed is not None else source.seed,
        num_steps=source.num_steps,
        layers=source.layers,
        reuse=source.reuse,
        quality=source.quality,
        loras=list(source.loras) if source.loras else None,
    )
    state.clips[new_id] = record
    return record


def _frames_dir(output_dir: Path) -> Path:
    return output_dir / "frames"


def _read_frame_library(output_dir: Path) -> list[dict[str, Any]]:
    raw = read_web_settings(output_dir).get("frame_library")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        fid = str(item.get("id") or "").strip() or f"frame_{uuid.uuid4().hex[:8]}"
        filename = str(item.get("filename") or Path(path).name)
        entry: dict[str, Any] = {
            "id": fid,
            "label": str(item.get("label") or "Frame"),
            "path": path,
            "filename": filename,
            "created_at": str(item.get("created_at") or datetime.now().isoformat()),
        }
        for key in ("width", "height", "source_clip_id", "time_s"):
            if item.get(key) is not None:
                entry[key] = item[key]
        out.append(entry)
    return out


def _write_frame_library(output_dir: Path, entries: list[dict[str, Any]]) -> None:
    data = read_web_settings(output_dir)
    data["frame_library"] = entries
    write_web_settings(output_dir, data)


def _frame_for_api(entry: dict[str, Any]) -> dict[str, Any]:
    filename = str(entry.get("filename") or Path(str(entry.get("path") or "")).name)
    out = dict(entry)
    out["image_url"] = f"/api/frames/files/{filename}"
    return out


def resolve_web_dist() -> Path:
    return resource_root() / "web" / "dist"


def _download_model_cmd(*extra: str) -> list[str]:
    """Build argv for scripts/download_model.py (frozen apps re-enter via --run-download)."""
    if is_frozen():
        return [sys.executable, "--run-download", *extra]
    script = resource_root() / "scripts" / "download_model.py"
    return [sys.executable, str(script), *extra]


def web_dist_stale() -> bool:
    dist = resolve_web_dist()
    if not dist.is_dir():
        return True
    assets = dist / "assets"
    js_files = list(assets.glob("index-*.js")) if assets.is_dir() else []
    if not js_files:
        return True
    newest_js = max(js_files, key=lambda path: path.stat().st_mtime)
    src_root = resource_root() / "web" / "src"
    if not src_root.is_dir():
        return False
    try:
        newest_src = max(
            path.stat().st_mtime for path in src_root.rglob("*") if path.is_file()
        )
    except ValueError:
        return False
    return newest_src > newest_js.stat().st_mtime


def ensure_web_dist_built(*, auto_build: bool = True) -> bool:
    dist = resolve_web_dist()
    if dist.is_dir() and not web_dist_stale():
        return True
    # Frozen apps ship a prebuilt UI — never try npm inside the bundle.
    if is_frozen():
        return dist.is_dir()
    if not auto_build:
        return dist.is_dir() and not web_dist_stale()
    web_dir = resource_root() / "web"
    if not (web_dir / "package.json").is_file():
        return False
    npm = shutil.which("npm")
    if not npm:
        log.warning("web/dist missing and npm not found — run: cd web && npm run build")
        return False
    if not (web_dir / "node_modules").is_dir():
        log.info("Installing Web UI deps…")
        try:
            subprocess.run(
                [npm, "install", "--no-fund", "--no-audit"],
                cwd=str(web_dir),
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            log.warning("Web UI npm install failed: %s", exc)
            return False
    log.info("Building Web UI…")
    try:
        subprocess.run([npm, "run", "build"], cwd=str(web_dir), check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        log.warning("Web UI build failed: %s", exc)
        return False
    return dist.is_dir() and not web_dist_stale()


def local_hostname() -> str:
    try:
        name = socket.gethostname().strip().split(".")[0]
        if name:
            return name
    except OSError:
        pass
    return "localhost"


def public_host(bind_host: str) -> str:
    host = (bind_host or "").strip()
    if not host or host in ("0.0.0.0", "::", "[::]"):
        return local_hostname()
    return host


def network_settings_public(*, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Serialize LAN-listen settings for the Features / Settings UI."""
    from h3_paths import (
        configured_server_host,
        configured_server_port,
        lan_ipv4_addresses,
        load_desktop_config,
    )

    data = cfg if cfg is not None else load_desktop_config()
    bind = configured_server_host(cfg=data)
    port = configured_server_port(cfg=data)
    listen_lan = bind in ("0.0.0.0", "::", "[::]") or data.get("listen_lan") is True
    lan_ips = lan_ipv4_addresses() if listen_lan else []
    return {
        "listen_lan": bool(listen_lan),
        "bind_host": bind,
        "port": port,
        "lan_urls": [f"http://{ip}:{port}/" for ip in lan_ips],
        "desktop_supervised": os.environ.get("H3_WS_DESKTOP", "").strip() in ("1", "true", "yes"),
    }


def request_network_rebind() -> bool:
    """Ask the desktop supervisor to rebind, if we were spawned by it."""
    if os.environ.get("H3_WS_DESKTOP", "").strip() not in ("1", "true", "yes"):
        return False
    # Do not import h3_desktop here — its module guard can exit the process.
    from h3_paths import writable_root

    path = writable_root() / "rebind.request"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{time.time():.3f}\n", encoding="utf-8")
        return True
    except OSError:
        return False


def build_server_urls(bind_host: str, port: int) -> tuple[str, str]:
    host = public_host(bind_host)
    return f"ws://{host}:{port}/ws", f"http://{host}:{port}/"


def urls_from_request(request: Any) -> tuple[str, str]:
    try:
        host = request.headers.get("host") or ""
        scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
    except Exception:
        return "", ""
    if not host:
        return "", ""
    ws_scheme = "wss" if scheme == "https" else "ws"
    return f"{ws_scheme}://{host}/ws", f"{scheme}://{host}/"


def _upload_extension(kind: str, filename: str | None) -> str:
    ext = Path(filename or "").suffix.lower()
    allowed: dict[str, set[str]] = {
        "image": {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"},
        "audio": {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm"},
        "video": {".mp4", ".mov", ".webm", ".mkv", ".avi"},
    }
    if ext and ext in allowed.get(kind, set()):
        return ext
    defaults = {"image": ".jpg", "audio": ".mp3", "video": ".mp4"}
    return defaults.get(kind, ".bin")


async def _save_upload_file(
    request: Request, upload_dir: Path, *, kind: str = "image"
) -> dict[str, Any]:
    form = await request.form()
    upload_file = form.get("file")
    if upload_file is None:
        raise ValueError("file is required")
    read = getattr(upload_file, "read", None)
    if read is None:
        raise ValueError("file is required")
    filename = getattr(upload_file, "filename", None) or "upload.bin"
    ext = _upload_extension(kind, filename)
    dest = upload_dir / f"{uuid.uuid4()}{ext}"
    content = await read()
    dest.write_bytes(content)
    payload: dict[str, Any] = {"path": str(dest), "filename": filename, "kind": kind}
    if kind in ("audio", "video"):
        # Normalize soundtracks with PyAV so h3-av can always decode stereo PCM.
        # Silent videos stay silent — UI/backend will attach them as silent_video.
        info = probe_media(dest)
        payload["has_audio"] = bool(info.get("has_audio"))
        if info.get("has_audio"):
            tmp_suffix = ".wav" if kind == "audio" else ".mp4"
            tmp = dest.with_name(f".{dest.stem}.adapt{tmp_suffix}")
            try:
                adapted = adapt_audio_for_h3(dest, tmp)
                if adapted.resolve() != dest.resolve():
                    if adapted.suffix.lower() != dest.suffix.lower():
                        dest.unlink(missing_ok=True)
                        dest = dest.with_suffix(adapted.suffix)
                        adapted.replace(dest)
                        payload["path"] = str(dest)
                    else:
                        adapted.replace(dest)
            except Exception:
                # Keep the original bytes; generate-time materialize retries.
                tmp.unlink(missing_ok=True)
        duration = probe_duration_seconds(dest)
        if duration is not None:
            payload["duration_s"] = duration
    _library_index_record(
        upload_dir,
        {
            "path": str(dest),
            "name": filename,
            "kind": kind,
            "created_at": datetime.now().isoformat(),
            **({"duration_s": payload["duration_s"]} if "duration_s" in payload else {}),
            **({"has_audio": payload["has_audio"]} if "has_audio" in payload else {}),
        },
    )
    return payload


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
_LIBRARY_INDEX = "library.json"


def _library_index_path(upload_dir: Path) -> Path:
    return Path(upload_dir) / _LIBRARY_INDEX


def _library_index_load(upload_dir: Path) -> dict[str, dict[str, Any]]:
    path = _library_index_path(upload_dir)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for raw in items:
        if not isinstance(raw, dict) or not raw.get("path"):
            continue
        out[str(raw["path"])] = raw
    return out


def _library_index_record(upload_dir: Path, entry: dict[str, Any]) -> None:
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    by_path = _library_index_load(upload_dir)
    by_path[str(entry["path"])] = entry
    payload = {"items": sorted(by_path.values(), key=lambda e: str(e.get("created_at") or ""), reverse=True)}
    _library_index_path(upload_dir).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _kind_for_suffix(suffix: str) -> str | None:
    low = suffix.lower()
    if low in _IMAGE_EXTS:
        return "image"
    if low in _VIDEO_EXTS:
        return "video"
    if low in _AUDIO_EXTS:
        return "audio"
    return None


def list_library_assets(
    state: AppState,
    *,
    tab: str = "image",
    query: str = "",
) -> list[dict[str, Any]]:
    """Continuity picker listing — recycle prior uploads, frames, or renders."""
    tab = (tab or "image").strip().lower()
    q = (query or "").strip().lower()
    indexed = _library_index_load(state.upload_dir)
    rows: list[dict[str, Any]] = []

    def push(
        *,
        path: Path,
        kind: str,
        name: str,
        created_at: str = "",
        duration_s: float | None = None,
        source: str = "upload",
        thumb_url: str | None = None,
        video_url: str | None = None,
        has_audio: bool | None = None,
    ) -> None:
        if q and q not in name.lower() and q not in path.name.lower():
            return
        media = f"/api/media?path={path}"
        row: dict[str, Any] = {
            "id": str(path),
            "path": str(path),
            "name": name,
            "kind": kind,
            "source": source,
            "created_at": created_at,
            "duration_s": duration_s,
            "thumb_url": thumb_url or (media if kind == "image" else None),
            "video_url": video_url or (media if kind == "video" else None),
            "media_url": media,
        }
        if kind == "video":
            if has_audio is None:
                has_audio = bool(probe_media(path).get("has_audio"))
            row["has_audio"] = bool(has_audio)
        rows.append(row)

    if tab in ("image", "video", "audio"):
        want = tab
        for path, meta in indexed.items():
            p = Path(path)
            if not p.is_file():
                continue
            kind = str(meta.get("kind") or _kind_for_suffix(p.suffix) or "")
            if kind != want:
                continue
            push(
                path=p,
                kind=kind,
                name=str(meta.get("name") or p.name),
                created_at=str(meta.get("created_at") or ""),
                duration_s=float(meta["duration_s"]) if meta.get("duration_s") is not None else None,
                source="upload",
                has_audio=bool(meta["has_audio"]) if meta.get("has_audio") is not None else None,
            )
        # Also surface unindexed files still sitting in the upload dir.
        known = {Path(p).resolve() for p in indexed}
        if state.upload_dir.is_dir():
            for p in sorted(state.upload_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if not p.is_file() or p.name == _LIBRARY_INDEX or p.resolve() in known:
                    continue
                kind = _kind_for_suffix(p.suffix)
                if kind != want:
                    continue
                push(path=p, kind=kind, name=p.name, source="upload")
        if tab == "image":
            frames_root = _frames_dir(state.output_dir)
            if frames_root.is_dir():
                for p in sorted(frames_root.glob("frame_*.png"), key=lambda x: x.stat().st_mtime, reverse=True):
                    push(
                        path=p,
                        kind="image",
                        name=p.name,
                        source="frame",
                        thumb_url=f"/api/frames/files/{p.name}",
                    )

    elif tab == "renders":
        for clip in sorted(
            state.clips.values(),
            key=lambda c: str(c.created_at or ""),
            reverse=True,
        ):
            if clip.status != RunStatus.DONE.value or not clip.filename:
                continue
            path = state.output_dir / clip.filename
            if not path.is_file():
                continue
            name = _clip_display_name(clip)
            if q and q not in name.lower() and q not in clip.filename.lower():
                continue
            rows.append(
                {
                    "id": clip.id,
                    "path": str(path),
                    "name": name,
                    "kind": "video",
                    "source": "render",
                    "created_at": clip.created_at or "",
                    "duration_s": clip.duration_seconds,
                    "thumb_url": state.clip_thumb_url(clip.filename),
                    "video_url": clip.video_url or state.clip_url(clip.filename),
                    "media_url": clip.video_url or state.clip_url(clip.filename),
                    "clip_id": clip.id,
                    # Renders from H3 always carry a stereo soundtrack.
                    "has_audio": True,
                }
            )

    return rows


def _clip_display_name(clip: ClipRecord) -> str:
    prompt = (clip.prompt or "").strip()
    if prompt:
        short = prompt.replace("\n", " ")
        return short if len(short) <= 48 else short[:45] + "…"
    return clip.filename or clip.id


def _clip_settings_from_body(body: dict[str, Any]) -> dict[str, Any]:
    width = int(body.get("width") or 512)
    height = int(body.get("height") or 512)
    try:
        width, height = validate_canvas(width, height)
    except ValueError:
        pass
    num_frames = body.get("num_frames")
    if num_frames is None and body.get("duration_seconds") is not None:
        num_frames = seconds_to_frames(float(body["duration_seconds"]))
    num_frames = snap_frames(int(num_frames or 22))
    seed = body.get("seed")
    try:
        seed_i = int(seed) if seed is not None and str(seed).strip() != "" else None
    except (TypeError, ValueError):
        seed_i = None
    return {
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "seed": seed_i,
        "num_steps": int(body.get("num_steps") or body.get("steps") or H3_DEFAULT_STEPS),
        "layers": int(body["layers"]) if body.get("layers") is not None else H3_DEFAULT_LAYERS,
        "reuse": int(body["reuse"]) if body.get("reuse") is not None else H3_DEFAULT_REUSE,
        "duration_seconds": float(body.get("duration_seconds") or num_frames / FPS),
        "clip_count": int(body.get("clip_count") or 1),
        "autocontinue": bool(body.get("autocontinue")),
        "autoconcat": bool(body.get("autoconcat")),
        "quality": str(body.get("quality") or "fast"),
        "render_width": int(body["render_width"]) if body.get("render_width") else None,
        "render_height": int(body["render_height"]) if body.get("render_height") else None,
        "loras": body.get("loras") or body.get("lora_specs") or [],
    }


def _loras_from_body(state: AppState, body: dict[str, Any]) -> list[LoraRef]:
    from h3_lora import lora_catalog, parse_lora_specs, resolve_lora_path

    specs = parse_lora_specs(
        body.get("loras") or body.get("lora_specs"),
        lora_catalog(state.output_dir),
    )
    return [
        LoraRef(spec=spec, path=resolve_lora_path(spec), scale=scale)
        for spec, scale in specs
    ]


def _resolve_existing_media(state: AppState, raw: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("reference path is required")
    p = Path(text)
    if p.is_file():
        return p.resolve()
    name = Path(text).name
    for cand in (
        state.output_dir / name,
        state.upload_dir / name,
        _frames_dir(state.output_dir) / name,
    ):
        if cand.is_file():
            return cand.resolve()
    raise ValueError(f"reference file not found: {text}")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_refs(state: AppState, refs: list[Any]) -> list[Any]:
    for item in refs:
        item.path = _resolve_existing_media(state, str(item.path))
        if item.audio_path is not None:
            item.audio_path = _resolve_existing_media(state, str(item.audio_path))
    return refs


def _materialize_ref_edits(state: AppState, refs: list[Any]) -> list[Any]:
    """Apply Continuity trim/crop to copies under upload_dir/edited/ before generate."""
    if not refs:
        return refs
    from h3_crop import apply_still, parse as parse_crop

    edit_dir = state.upload_dir / "edited"
    edit_dir.mkdir(parents=True, exist_ok=True)
    for item in refs:
        kind = (item.kind or "").strip().lower()
        # Temporal segment first so framing sees the used stretch.
        if item.has_trim() and kind in ("audio", "silent_video", "video", "video_audio"):
            start = float(item.trim_start or 0.0)
            end = float(item.trim_end) if item.trim_end is not None else None
            suffix = item.path.suffix or (".wav" if kind == "audio" else ".mp4")
            dest = edit_dir / f"{uuid.uuid4().hex[:10]}_trim{suffix}"
            item.path = trim_media(item.path, dest, start_s=start, end_s=end)
            if kind == "video_audio" and item.audio_path is not None:
                a_suffix = item.audio_path.suffix or ".wav"
                a_dest = edit_dir / f"{uuid.uuid4().hex[:10]}_trim_a{a_suffix}"
                item.audio_path = trim_media(
                    item.audio_path, a_dest, start_s=start, end_s=end
                )
        if item.has_crop() and kind == "image":
            crop = parse_crop(item.crop)
            if crop is not None:
                dest = edit_dir / f"{uuid.uuid4().hex[:10]}_crop.png"
                apply_still(item.path, dest, crop)
                item.path = dest
        elif item.has_crop() and kind in ("silent_video", "video", "video_audio"):
            # Spatial framing on a clip: bake Continuity turn/mirror/window per frame.
            crop = parse_crop(item.crop)
            if crop is not None:
                dest = edit_dir / f"{uuid.uuid4().hex[:10]}_frame.mp4"
                item.path = frame_video(item.path, dest, crop)
        # H3 audio ingest: stereo soundtrack or silent_video — never feed a
        # video-only file through --ref-video / h3_ffmpeg_read_audio_f32.
        kind = (item.kind or "").strip().lower()
        if kind == "video":
            info = probe_media(item.path)
            if not info.get("has_audio"):
                item.kind = "silent_video"
            else:
                dest = edit_dir / f"{uuid.uuid4().hex[:10]}_audio.mp4"
                item.path = adapt_audio_for_h3(item.path, dest)
        elif kind == "video_audio" and item.audio_path is not None:
            a_dest = edit_dir / f"{uuid.uuid4().hex[:10]}_audio.wav"
            item.audio_path = adapt_audio_for_h3(item.audio_path, a_dest)
        elif kind == "audio":
            a_dest = edit_dir / f"{uuid.uuid4().hex[:10]}_audio.wav"
            item.path = adapt_audio_for_h3(item.path, a_dest)
    return refs


def _request_from_body(
    body: dict[str, Any],
    prompt: str,
    output: Path,
    *,
    state: AppState | None = None,
    first_frame: Path | None = None,
    last_frame: Path | None = None,
) -> GenerateRequest:
    settings = _clip_settings_from_body(body)
    mode = str(body.get("mode") or "t2va").strip().lower()
    first = first_frame
    last = last_frame
    if first is None and body.get("image_path"):
        first = Path(str(body["image_path"]))
    if last is None and body.get("end_image_path"):
        last = Path(str(body["end_image_path"]))
    if mode == "last_frame" and last is None and first is not None:
        last, first = first, None
    if mode == "first_frame" and first is None:
        raise ValueError("first_frame mode requires an image")
    if mode == "last_frame" and last is None:
        raise ValueError("last_frame mode requires an image")
    if mode == "fl2va" and (first is None or last is None):
        raise ValueError("fl2va mode requires first and last images")
    refs = parse_refs_payload(body.get("refs"))
    if refs:
        mode = "ref2va"
        first = None
        last = None
    elif mode == "ref2va":
        raise ValueError("ref2va requires at least one image, video, or audio reference")
    rw = settings.get("render_width")
    rh = settings.get("render_height")
    loras = _loras_from_body(state, body) if state is not None else []
    return GenerateRequest(
        prompt=prompt,
        output_path=output,
        width=int(settings["width"] or 512),
        height=int(settings["height"] or 512),
        num_frames=int(settings["num_frames"] or 22),
        quality=str(settings.get("quality") or "fast"),
        steps=int(settings["num_steps"] or H3_DEFAULT_STEPS),
        layers=int(settings["layers"]) if settings.get("layers") is not None else None,
        reuse=int(settings["reuse"]) if settings.get("reuse") is not None else None,
        core_reuse=int(body["core_reuse"]) if body.get("core_reuse") is not None else None,
        token_reduction=bool(body["token_reduction"])
        if body.get("token_reduction") is not None
        else None,
        render_width=int(rw) if rw else None,
        render_height=int(rh) if rh else None,
        seed=settings.get("seed"),
        ssd_streaming=bool(body.get("ssd_streaming")) and not loras,
        first_frame=first,
        last_frame=last,
        refs=refs,
        loras=loras,
        mode=mode,
    )


async def _fail_run(state: AppState, run_id: str, message: str) -> None:
    run = state.runs.get(run_id)
    if not run:
        return
    run.status = RunStatus.FAILED.value
    run.error = message
    for cid in run.clip_ids:
        clip = state.clips.get(cid)
        if clip and clip.status != RunStatus.DONE.value:
            clip.status = RunStatus.FAILED.value
            clip.error = message
    state.save_index()
    await state.emit(run_id, {"type": "error", "error": message, "run_id": run_id})


async def _abort_run_cancelled(state: AppState, run_id: str) -> None:
    run = state.runs.get(run_id)
    if not run:
        return
    run.status = RunStatus.CANCELLED.value
    run.error = "cancelled"
    for cid in run.clip_ids:
        clip = state.clips.get(cid)
        if clip and clip.status not in (RunStatus.DONE.value,):
            clip.status = RunStatus.CANCELLED.value
            clip.error = "cancelled"
    state.save_index()
    await state.emit(
        run_id,
        {"type": "run_cancelled", "run_id": run_id, "message": "Generation cancelled"},
    )


async def _execute_run(state: AppState, run_id: str) -> None:
    run = state.runs.get(run_id)
    if not run:
        return
    if state.is_run_cancelled(run_id):
        await _abort_run_cancelled(state, run_id)
        return
    state._active_run_id = run_id
    run.status = RunStatus.RUNNING.value
    body = dict(_RUN_BODIES.get(run_id, {}))
    await state.emit(
        run_id,
        {
            "type": "run_started",
            "run_id": run_id,
            "clip_count": len(run.prompts),
            "autoconcat": run.autoconcat,
            "autocontinue": run.autocontinue,
        },
    )
    done_paths: list[Path] = []
    continue_from = body.get("continue_from")
    prev_frame: Path | None = None
    if continue_from:
        parent = state.clips.get(str(continue_from))
        if parent and parent.filename:
            parent_path = state.output_dir / parent.filename
            if parent_path.is_file() and media_available():
                tmp = mk_scratch_dir("h3_cont_")
                prev_frame = extract_last_frame(parent_path, tmp / "last.png")

    try:
        for i, (clip_id, prompt) in enumerate(zip(run.clip_ids, run.prompts)):
            if state.is_run_cancelled(run_id):
                await _abort_run_cancelled(state, run_id)
                return
            clip = state.clips[clip_id]
            clip.status = RunStatus.RUNNING.value
            dest = state.output_dir / clip.filename
            await state.emit(
                run_id,
                {
                    "type": "clip_started",
                    "clip_id": clip_id,
                    "index": i,
                    "total": len(run.prompts),
                    "prompt": prompt,
                },
            )
            first = prev_frame
            last = Path(str(body["end_image_path"])) if body.get("end_image_path") else None
            if i == 0 and first is None and body.get("image_path"):
                first = Path(str(body["image_path"]))
            loop = asyncio.get_running_loop()

            def _progress(mp: dict[str, Any], *, _rid=run_id) -> None:
                asyncio.run_coroutine_threadsafe(
                    state.emit(
                        _rid,
                        {
                            "type": "progress",
                            "phase": mp.get("stage") or "generating",
                            "elapsed_s": mp.get("elapsed_s"),
                            "model_progress": mp,
                        },
                    ),
                    loop,
                )

            t0 = time.time()
            req = _request_from_body(
                body,
                prompt,
                dest,
                state=state,
                first_frame=first,
                last_frame=last if i == 0 else None,
            )
            if req.loras:
                req.ssd_streaming = False
            if req.refs:
                req.first_frame = None
                req.last_frame = None
                req.mode = "ref2va"
            elif run.autocontinue and i > 0 and first is not None:
                req.mode = "first_frame"

            preview_dir: Path | None = None
            preview_stem = state.output_dir / ".preview" / f"{run_id}_{clip_id}"
            watcher: LatentPreviewWatcher | None = None
            if taeh3_decode_ready():
                # Engine-owned warm dir — same path every job so interactive
                # ``--preview-latent`` stays valid without respawning h3.
                preview_dir = state.engine.ensure_preview_latent_dir()
                req.preview_latent_dir = preview_dir

                def _on_preview(
                    info: dict[str, Any],
                    *,
                    _rid=run_id,
                    _stem=preview_stem.name,
                ) -> None:
                    served = Path(str(info.get("path") or ""))
                    ext = served.suffix if served.suffix else ".mp4"
                    url = f"/api/preview/{_stem}{ext}?t={int(time.time() * 1000)}"
                    asyncio.run_coroutine_threadsafe(
                        state.emit(
                            _rid,
                            {
                                "type": "preview",
                                "url": url,
                                "mime": info.get("mime") or "image/webp",
                                "step": info.get("step"),
                                "total": info.get("total"),
                                "fps": info.get("fps"),
                            },
                        ),
                        loop,
                    )

                watcher = LatentPreviewWatcher(
                    preview_dir, preview_stem, on_preview=_on_preview
                )
                watcher.start()

            try:
                await asyncio.to_thread(state.engine.generate, req, on_progress=_progress)
            except GenerationCancelledError:
                clip.timings = getattr(state.engine, "last_timings", None)
                if watcher is not None:
                    watcher.stop(cleanup=False)
                state.engine.clear_preview_latent_dumps()
                _purge_preview_stem(preview_stem)
                await _abort_run_cancelled(state, run_id)
                return
            except Exception:
                clip.timings = getattr(state.engine, "last_timings", None)
                # Full failure — stop watcher, then drop all preview scratch.
                if watcher is not None:
                    watcher.stop(cleanup=False)
                state.engine.clear_preview_latent_dumps()
                _purge_preview_stem(preview_stem)
                raise
            else:
                # Success — stop watcher and clear latent dumps only; keep the
                # looping served preview until clip_done hands off to the MP4.
                if watcher is not None:
                    watcher.stop(cleanup=False)
                state.engine.clear_preview_latent_dumps()
            clip.timings = getattr(state.engine, "last_timings", None)
            elapsed = round(time.time() - t0, 2)
            size = dest.stat().st_size if dest.is_file() else 0
            clip.elapsed_s = elapsed
            clip.bytes = size
            clip.video_url = state.clip_url(clip.filename)
            clip.label = "CURRENT" if i == len(run.prompts) - 1 else f"CLIP {i + 1}"
            if i == 0:
                clip.label = "ORIGINAL" if len(run.prompts) > 1 else "CURRENT"

            # Generate-time auto-passes: ReDetail only (resolution mode).
            # Faces / Upframe / Long take are library actions — spawning them here
            # littered the library with near-duplicate CURRENT/FACES/… peers.
            longtake_cfg = None
            longtake_clip_id: str | None = None
            if longtake_cfg and longtake_cfg.get("enabled") and dest.is_file():
                from h3_longtake import (
                    DEFAULT_INTERPOLATION,
                    DEFAULT_KEYFRAMES,
                    DEFAULT_LONGTAKE_DENOISE,
                    LongTakeError,
                    LongTakeSettings,
                    clamp_longtake_denoise,
                    run_longtake_refine,
                )

                lt_watcher: LatentPreviewWatcher | None = None
                lt_stem = state.output_dir / ".preview" / f"{run_id}_{clip.id}_longtake"
                lt_name = _unique_pass_filename(state.output_dir, clip.filename, "longtake")
                lt_path = state.output_dir / lt_name
                try:
                    await state.emit(
                        run_id,
                        {
                            "type": "progress",
                            "phase": "longtake_refine",
                            "message": "Long take refine…",
                            "clip_id": clip.id,
                        },
                    )
                    lt_settings = LongTakeSettings(
                        denoise=clamp_longtake_denoise(
                            float(
                                longtake_cfg.get("denoise")
                                if longtake_cfg.get("denoise") is not None
                                else DEFAULT_LONGTAKE_DENOISE
                            )
                        ),
                        seed=int(
                            longtake_cfg.get("seed")
                            if longtake_cfg.get("seed") is not None
                            else (clip.seed or 42)
                        ),
                        keyframes=str(
                            longtake_cfg.get("keyframes") or DEFAULT_KEYFRAMES
                        ),
                        interpolation=str(
                            longtake_cfg.get("interpolation") or DEFAULT_INTERPOLATION
                        ),
                    )
                    lt_refs = parse_refs_payload(body.get("refs"), validate=False)
                    lt_loras = _loras_from_body(state, body)
                    lt_quality = str(body.get("quality") or clip.quality or "fast")
                    lt_steps = (
                        int(body["num_steps"])
                        if body.get("num_steps") is not None
                        else clip.num_steps
                    )
                    lt_layers = (
                        int(body["layers"]) if body.get("layers") is not None else clip.layers
                    )
                    lt_reuse = (
                        int(body["reuse"]) if body.get("reuse") is not None else clip.reuse
                    )

                    lt_preview_dir: Path | None = None
                    if taeh3_decode_ready():
                        lt_preview_dir = state.engine.ensure_preview_latent_dir()

                        def _on_lt_preview(
                            info: dict[str, Any],
                            *,
                            _rid=run_id,
                            _stem=lt_stem.name,
                        ) -> None:
                            served = Path(str(info.get("path") or ""))
                            ext = served.suffix if served.suffix else ".mp4"
                            url = f"/api/preview/{_stem}{ext}?t={int(time.time() * 1000)}"
                            asyncio.run_coroutine_threadsafe(
                                state.emit(
                                    _rid,
                                    {
                                        "type": "preview",
                                        "url": url,
                                        "mime": info.get("mime") or "image/webp",
                                        "step": info.get("step"),
                                        "total": info.get("total"),
                                        "fps": info.get("fps"),
                                    },
                                ),
                                loop,
                            )

                        lt_watcher = LatentPreviewWatcher(
                            lt_preview_dir, lt_stem, on_preview=_on_lt_preview
                        )
                        lt_watcher.start()

                    def _lt_prog(phase: str, extra: dict[str, Any]) -> None:
                        msg = str(extra.get("message") or phase)
                        mp = extra.get("model_progress")
                        payload: dict[str, Any] = {
                            "type": "progress",
                            "phase": phase,
                            "message": msg,
                            "clip_id": clip.id,
                        }
                        if isinstance(mp, dict):
                            payload["model_progress"] = mp
                            if mp.get("elapsed_s") is not None:
                                payload["elapsed_s"] = mp.get("elapsed_s")
                            stage = str(mp.get("stage") or "").strip()
                            step, total = mp.get("step"), mp.get("total")
                            bits = [msg]
                            if step is not None and total is not None:
                                label = stage.capitalize() if stage else "Step"
                                bits.append(f"{label} {step}/{total}")
                            eta = mp.get("eta_s")
                            if eta is not None:
                                eta_i = max(0, int(round(float(eta))))
                                bits.append(f"{eta_i // 60:02d}:{eta_i % 60:02d} remaining")
                            elif mp.get("avg_step_s") is not None:
                                bits.append(f"{mp['avg_step_s']}s/it")
                            payload["message"] = " · ".join(bits)
                        log.info("longtake %s clip=%s %s", phase, clip.id, payload["message"])
                        asyncio.run_coroutine_threadsafe(
                            state.emit(run_id, payload),
                            loop,
                        )

                    if lt_path.exists():
                        lt_path.unlink()
                    await asyncio.to_thread(
                        run_longtake_refine,
                        dest,
                        lt_path,
                        prompt=clip.prompt,
                        engine=state.engine,
                        settings=lt_settings,
                        on_progress=_lt_prog,
                        refs=lt_refs,
                        loras=lt_loras,
                        quality=lt_quality,
                        steps=lt_steps,
                        layers=lt_layers,
                        reuse=lt_reuse,
                        preview_latent_dir=lt_preview_dir,
                    )
                    if lt_path.is_file():
                        new_lt = _spawn_processed_clip(
                            state,
                            clip,
                            lt_path,
                            generation_patch={
                                "longtake": {
                                    "enabled": True,
                                    "denoise": lt_settings.denoise,
                                    "keyframes": lt_settings.keyframes,
                                    "seed": lt_settings.seed,
                                },
                            },
                            seed=lt_settings.seed,
                            run_id=run_id,
                        )
                        longtake_clip_id = new_lt.id
                        await state.emit(
                            run_id,
                            {
                                "type": "progress",
                                "phase": "longtake_done",
                                "message": "Long take clip ready",
                                "clip_id": new_lt.id,
                                "video_url": new_lt.video_url,
                            },
                        )
                except LongTakeError as exc:
                    log.warning("Long take skipped for %s: %s", clip.id, exc)
                    await state.emit(
                        run_id,
                        {
                            "type": "progress",
                            "phase": "longtake_skip",
                            "message": f"Long take skipped: {exc}",
                            "clip_id": clip.id,
                        },
                    )
                except Exception as exc:
                    log.exception("Long take failed for %s", clip.id)
                    await state.emit(
                        run_id,
                        {
                            "type": "progress",
                            "phase": "longtake_error",
                            "message": f"Long take failed: {exc}",
                            "clip_id": clip.id,
                        },
                    )
                finally:
                    if lt_watcher is not None:
                        lt_watcher.stop(cleanup=False)
                    state.engine.clear_preview_latent_dumps()
                    _purge_preview_stem(lt_stem)

            redetail_cfg = run.redetail if isinstance(run.redetail, dict) else None
            redetail_clip_id: str | None = None
            if redetail_cfg and redetail_cfg.get("enabled") and dest.is_file():
                from h3_redetail import (
                    DEFAULT_REDETAIL_DENOISE,
                    RedetailError,
                    RedetailSettings,
                    clamp_redetail_denoise,
                    run_redetail,
                )

                rd_name = _unique_pass_filename(state.output_dir, clip.filename, "redetail")
                rd_path = state.output_dir / rd_name
                try:
                    await state.emit(
                        run_id,
                        {
                            "type": "progress",
                            "phase": "redetail_upscale",
                            "message": "ReDetail · ×2 refine…",
                            "clip_id": clip.id,
                        },
                    )
                    settings_rd = RedetailSettings(
                        denoise=clamp_redetail_denoise(
                            float(
                                redetail_cfg.get("denoise")
                                if redetail_cfg.get("denoise") is not None
                                else DEFAULT_REDETAIL_DENOISE
                            )
                        ),
                        seed=int(
                            redetail_cfg.get("seed")
                            if redetail_cfg.get("seed") is not None
                            else (clip.seed or 42)
                        ),
                    )

                    def _rd_progress(phase: str, extra: dict[str, Any]) -> None:
                        msg = str(extra.get("message") or phase)
                        mp = extra.get("model_progress")
                        payload: dict[str, Any] = {
                            "type": "progress",
                            "phase": phase,
                            "message": msg,
                            "clip_id": clip.id,
                        }
                        if isinstance(mp, dict):
                            payload["model_progress"] = mp
                            if mp.get("elapsed_s") is not None:
                                payload["elapsed_s"] = mp.get("elapsed_s")
                            stage = str(mp.get("stage") or "").strip()
                            step, total = mp.get("step"), mp.get("total")
                            bits = [msg]
                            if stage and stage.lower() not in msg.lower():
                                bits.append(stage)
                            if step is not None and total is not None:
                                bits.append(f"{step}/{total}")
                            eta = mp.get("eta_s")
                            if eta is not None:
                                eta_i = max(0, int(round(float(eta))))
                                bits.append(
                                    f"{eta_i // 60:02d}:{eta_i % 60:02d} remaining"
                                )
                            elif mp.get("avg_step_s") is not None:
                                bits.append(f"{mp['avg_step_s']}s/it")
                            payload["message"] = " · ".join(bits)
                        log.info(
                            "redetail %s clip=%s %s", phase, clip.id, payload["message"]
                        )
                        state.set_pass_progress(
                            phase, extra, kind="redetail", clip_id=clip.id
                        )
                        asyncio.run_coroutine_threadsafe(
                            state.emit(run_id, payload),
                            loop,
                        )

                    if rd_path.exists():
                        rd_path.unlink()
                    state.set_pass_progress(
                        "redetail_upscale",
                        {"message": "ReDetail · ×2 refine…"},
                        kind="redetail",
                        clip_id=clip.id,
                    )
                    try:
                        await asyncio.to_thread(
                            run_redetail,
                            dest,
                            rd_path,
                            prompt=clip.prompt or "detailed cinematic shot",
                            engine=state.engine,
                            settings=settings_rd,
                            on_progress=_rd_progress,
                            refs=req.refs,
                            loras=req.loras,
                            quality=req.quality,
                            steps=req.steps,
                            layers=req.layers,
                            reuse=1,
                        )
                    finally:
                        state.clear_pass_progress()
                    if rd_path.is_file():
                        from h3_media import redetail_output_size

                        try:
                            tw, th = redetail_output_size(
                                int(clip.width or 512), int(clip.height or 512)
                            )
                        except Exception:
                            tw, th = clip.width, clip.height
                        new_rd = _spawn_processed_clip(
                            state,
                            clip,
                            rd_path,
                            generation_patch={
                                "redetail": {
                                    "enabled": True,
                                    "denoise": settings_rd.denoise,
                                    "seed": settings_rd.seed,
                                },
                            },
                            width=tw,
                            height=th,
                            run_id=run_id,
                        )
                        # Intermediate sample pass — not the product the user ordered.
                        clip.label = "BASE"
                        gen = dict(clip.generation or {}) if isinstance(clip.generation, dict) else {}
                        gen["superseded_by"] = new_rd.id
                        gen["pass"] = "base"
                        clip.generation = gen
                        redetail_clip_id = new_rd.id
                        await state.emit(
                            run_id,
                            {
                                "type": "progress",
                                "phase": "redetail_done",
                                "message": "ReDetail clip ready",
                                "clip_id": new_rd.id,
                                "video_url": new_rd.video_url,
                            },
                        )
                except RedetailError as exc:
                    log.warning("ReDetail skipped for %s: %s", clip.id, exc)
                    await state.emit(
                        run_id,
                        {
                            "type": "progress",
                            "phase": "redetail_skip",
                            "message": f"ReDetail skipped: {exc}",
                            "clip_id": clip.id,
                        },
                    )
                except Exception as exc:
                    log.exception("ReDetail failed for %s", clip.id)
                    await state.emit(
                        run_id,
                        {
                            "type": "progress",
                            "phase": "redetail_error",
                            "message": f"ReDetail failed: {exc}",
                            "clip_id": clip.id,
                        },
                    )

            upframe_cfg = None  # library-only
            upframe_clip_id: str | None = None
            if upframe_cfg and upframe_cfg.get("enabled") and dest.is_file():
                from h3_upframe import (
                    DEFAULT_MULTIPLIER,
                    UpframeError,
                    UpframeSettings,
                    run_upframe,
                )

                uf_name = _unique_pass_filename(state.output_dir, clip.filename, "upframe")
                uf_path = state.output_dir / uf_name
                try:
                    await state.emit(
                        run_id,
                        {
                            "type": "progress",
                            "phase": "upframe_run",
                            "message": "Upframe · temporal interpolate…",
                            "clip_id": clip.id,
                        },
                    )
                    uf_mult = int(upframe_cfg.get("multiplier") or DEFAULT_MULTIPLIER)
                    if uf_path.exists():
                        uf_path.unlink()
                    await asyncio.to_thread(
                        run_upframe,
                        dest,
                        uf_path,
                        settings=UpframeSettings(multiplier=uf_mult),
                        model_dir=state.engine.model_dir,
                        require_weights=True,
                    )
                    if uf_path.is_file():
                        new_uf = _spawn_processed_clip(
                            state,
                            clip,
                            uf_path,
                            generation_patch={
                                "upframe": {"enabled": True, "multiplier": uf_mult},
                            },
                            run_id=run_id,
                        )
                        upframe_clip_id = new_uf.id
                        await state.emit(
                            run_id,
                            {
                                "type": "progress",
                                "phase": "upframe_done",
                                "message": "Upframe clip ready",
                                "clip_id": new_uf.id,
                                "video_url": new_uf.video_url,
                            },
                        )
                except UpframeError as exc:
                    log.warning("Upframe skipped for %s: %s", clip.id, exc)
                    await state.emit(
                        run_id,
                        {
                            "type": "progress",
                            "phase": "upframe_skip",
                            "message": f"Upframe skipped: {exc}",
                            "clip_id": clip.id,
                        },
                    )
                except Exception as exc:
                    log.exception("Upframe failed for %s", clip.id)
                    await state.emit(
                        run_id,
                        {
                            "type": "progress",
                            "phase": "upframe_error",
                            "message": f"Upframe failed: {exc}",
                            "clip_id": clip.id,
                        },
                    )

            faces_cfg = None  # library-only
            faces_clip_id: str | None = None
            if faces_cfg and faces_cfg.get("enabled") and dest.is_file():
                from h3_facepass import FacesSettings, run_faces_pass
                from h3_faces import (
                    DEFAULT_FACE_CANVAS,
                    DEFAULT_FACE_DENOISE,
                    FaceError,
                    clamp_face_denoise,
                    face_canvas_for_video,
                )

                faces_watcher: LatentPreviewWatcher | None = None
                faces_stem = state.output_dir / ".preview" / f"{run_id}_{clip.id}_faces"
                faces_name = _unique_pass_filename(state.output_dir, clip.filename, "faces")
                faces_path = state.output_dir / faces_name
                try:
                    await state.emit(
                        run_id,
                        {
                            "type": "progress",
                            "phase": "faces_detect",
                            "message": "Faces pass…",
                            "clip_id": clip.id,
                        },
                    )
                    from h3_sam import SamDownloadError, download_sam3, sam3_ready

                    if not sam3_ready(state.engine.model_dir):
                        await state.emit(
                            run_id,
                            {
                                "type": "progress",
                                "phase": "faces_detect",
                                "message": "Downloading SAM 3.1 for Faces…",
                                "clip_id": clip.id,
                            },
                        )
                        try:
                            await asyncio.to_thread(download_sam3, state.engine.model_dir)
                        except SamDownloadError as exc:
                            raise FaceError(str(exc)) from exc
                    # Faces canvas tracks the shot short edge (not a free dial).
                    face_canvas = face_canvas_for_video(
                        clip.width,
                        clip.height,
                        fallback=(
                            int(faces_cfg["canvas"])
                            if faces_cfg.get("canvas") is not None
                            else DEFAULT_FACE_CANVAS
                        ),
                    )
                    fsettings = FacesSettings(
                        canvas=face_canvas,
                        denoise=clamp_face_denoise(
                            float(
                                faces_cfg.get("denoise")
                                if faces_cfg.get("denoise") is not None
                                else DEFAULT_FACE_DENOISE
                            )
                        ),
                        seed=int(
                            faces_cfg.get("seed")
                            if faces_cfg.get("seed") is not None
                            else (clip.seed or 42)
                        ),
                        abstain=bool(faces_cfg.get("abstain", False)),
                    )
                    face_refs = parse_refs_payload(body.get("refs"), validate=False)
                    face_loras = _loras_from_body(state, body)
                    face_quality = str(body.get("quality") or clip.quality or "close")
                    face_steps = (
                        int(body["num_steps"])
                        if body.get("num_steps") is not None
                        else clip.num_steps
                    )
                    face_layers = (
                        int(body["layers"]) if body.get("layers") is not None else clip.layers
                    )
                    face_reuse = (
                        int(body["reuse"]) if body.get("reuse") is not None else clip.reuse
                    )

                    faces_preview_dir: Path | None = None
                    if taeh3_decode_ready():
                        faces_preview_dir = state.engine.ensure_preview_latent_dir()

                        def _on_faces_preview(
                            info: dict[str, Any],
                            *,
                            _rid=run_id,
                            _stem=faces_stem.name,
                        ) -> None:
                            served = Path(str(info.get("path") or ""))
                            ext = served.suffix if served.suffix else ".mp4"
                            url = f"/api/preview/{_stem}{ext}?t={int(time.time() * 1000)}"
                            asyncio.run_coroutine_threadsafe(
                                state.emit(
                                    _rid,
                                    {
                                        "type": "preview",
                                        "url": url,
                                        "mime": info.get("mime") or "image/webp",
                                        "step": info.get("step"),
                                        "total": info.get("total"),
                                    },
                                ),
                                loop,
                            )

                        faces_watcher = LatentPreviewWatcher(
                            faces_preview_dir, faces_stem, on_preview=_on_faces_preview
                        )
                        faces_watcher.start()

                    def _faces_prog(phase: str, extra: dict[str, Any]) -> None:
                        # Same composition as longtake: window label + nested H3
                        # denoise so logs and the SSE progress line stay in lockstep
                        # (never leave the UI stuck on a bare "video vae decode").
                        msg = str(extra.get("message") or phase)
                        mp = extra.get("model_progress")
                        payload: dict[str, Any] = {
                            "type": "progress",
                            "phase": phase,
                            "message": msg,
                            "clip_id": clip.id,
                        }
                        for k in (
                            "index",
                            "total",
                            "start",
                            "end",
                            "strength",
                            "windows",
                            "canvas",
                            "mode",
                            "refs",
                        ):
                            if k in extra:
                                payload[k] = extra[k]
                        if isinstance(mp, dict):
                            payload["model_progress"] = mp
                            if mp.get("elapsed_s") is not None:
                                payload["elapsed_s"] = mp.get("elapsed_s")
                            stage = str(mp.get("stage") or "").strip()
                            step, total = mp.get("step"), mp.get("total")
                            bits = [msg]
                            if stage and stage.lower() not in msg.lower():
                                bits.append(stage)
                            if step is not None and total is not None:
                                bits.append(f"{step}/{total}")
                            eta = mp.get("eta_s")
                            if eta is not None:
                                eta_i = max(0, int(round(float(eta))))
                                bits.append(
                                    f"{eta_i // 60:02d}:{eta_i % 60:02d} remaining"
                                )
                            elif mp.get("avg_step_s") is not None:
                                bits.append(f"{mp['avg_step_s']}s/it")
                            payload["message"] = " · ".join(bits)
                        log.info("faces %s clip=%s %s", phase, clip.id, payload["message"])
                        asyncio.run_coroutine_threadsafe(
                            state.emit(run_id, payload), loop
                        )

                    if faces_path.exists():
                        faces_path.unlink()
                    await asyncio.to_thread(
                        run_faces_pass,
                        dest,
                        faces_path,
                        prompt=clip.prompt or "a person speaking, detailed face",
                        engine=state.engine,
                        settings=fsettings,
                        model_dir=state.engine.model_dir,
                        on_progress=_faces_prog,
                        refs=face_refs,
                        loras=face_loras,
                        quality=face_quality,
                        steps=face_steps,
                        layers=face_layers,
                        reuse=face_reuse,
                        preview_latent_dir=faces_preview_dir,
                    )
                    if faces_path.is_file():
                        new_faces = _spawn_processed_clip(
                            state,
                            clip,
                            faces_path,
                            generation_patch={
                                "faces": {
                                    "enabled": True,
                                    "canvas": fsettings.canvas,
                                    "denoise": fsettings.denoise,
                                    "seed": fsettings.seed,
                                },
                            },
                            seed=fsettings.seed,
                            run_id=run_id,
                        )
                        faces_clip_id = new_faces.id
                        await state.emit(
                            run_id,
                            {
                                "type": "progress",
                                "phase": "faces_done",
                                "message": "Faces clip ready",
                                "clip_id": new_faces.id,
                                "video_url": new_faces.video_url,
                            },
                        )
                except FaceError as exc:
                    log.warning("Faces post-pass skipped for %s: %s", clip.id, exc)
                    await state.emit(
                        run_id,
                        {
                            "type": "progress",
                            "phase": "faces_skip",
                            "message": f"Faces skipped: {exc}",
                            "clip_id": clip.id,
                        },
                    )
                except Exception as exc:
                    log.exception("Faces post-pass failed for %s", clip.id)
                    await state.emit(
                        run_id,
                        {
                            "type": "progress",
                            "phase": "faces_error",
                            "message": f"Faces failed: {exc}",
                            "clip_id": clip.id,
                        },
                    )
                finally:
                    if faces_watcher is not None:
                        faces_watcher.stop(cleanup=False)
                    _purge_preview_stem(faces_stem)

            clip.status = RunStatus.DONE.value
            clip.elapsed_s = round(time.time() - t0, 2)
            clip.bytes = dest.stat().st_size if dest.is_file() else size
            clip.video_url = state.clip_url(clip.filename)
            thumb_url = ""
            if dest.is_file() and media_available():
                poster = await asyncio.to_thread(ensure_video_poster, dest)
                if poster is not None:
                    thumb_url = state.clip_thumb_url(clip.filename)
            state.save_index()
            await state.emit(
                run_id,
                {
                    "type": "clip_done",
                    "clip_id": clip.id,
                    "video_url": clip.video_url,
                    "thumb_url": thumb_url,
                    "bytes": clip.bytes,
                    "filename": clip.filename,
                    "chain_id": clip.chain_id,
                    "longtake": bool(longtake_clip_id),
                    "longtake_clip_id": longtake_clip_id,
                    "redetail": bool(redetail_clip_id),
                    "redetail_clip_id": redetail_clip_id,
                    "upframe": bool(upframe_clip_id),
                    "upframe_clip_id": upframe_clip_id,
                    "faces": bool(faces_clip_id),
                    "faces_clip_id": faces_clip_id,
                },
            )
            _purge_preview_stem(preview_stem)
            done_paths.append(dest)
            if run.autocontinue and i < len(run.prompts) - 1 and media_available():
                tmp = mk_scratch_dir("h3_chain_")
                prev_frame = extract_last_frame(dest, tmp / "last.png")

        if run.autoconcat and len(done_paths) > 1 and media_available():
            merged_name = f"web_{sanitize_filename(run.prompts[0])}_merged.mp4"
            merged_path = state.output_dir / merged_name
            concat_mp4s(done_paths, merged_path)
            mid = str(uuid.uuid4())
            source_project = None
            for cid in run.clip_ids:
                src = state.clips.get(cid)
                if src and src.project_id:
                    source_project = src.project_id
                    break
            mclip = ClipRecord(
                id=mid,
                prompt=run.prompts[0] + f" (×{len(done_paths)} merged)",
                label="MERGED",
                video_url=state.clip_url(merged_name),
                filename=merged_name,
                chain_id=run.chain_id,
                clip_index=len(run.clip_ids),
                mode=str(body.get("mode") or "t2va"),
                status=RunStatus.DONE.value,
                created_at=datetime.now().isoformat(),
                bytes=merged_path.stat().st_size if merged_path.is_file() else None,
                project_id=source_project or state.active_project_id,
                generation=dict(body["generation"]) if isinstance(body.get("generation"), dict) else None,
                run_id=run_id,
                **{
                    k: v
                    for k, v in _clip_settings_from_body(body).items()
                    if k in ClipRecord.__dataclass_fields__
                    and k not in ("project_id", "generation", "run_id")
                },
            )
            state.clips[mid] = mclip
            run.merged_clip_id = mid
            run.merged_url = mclip.video_url
            await state.emit(
                run_id,
                {
                    "type": "merged",
                    "video_url": mclip.video_url,
                    "clip_id": mid,
                    "filename": merged_name,
                    "chain_id": run.chain_id,
                },
            )

        run.status = RunStatus.DONE.value
        state.save_index()
        await state.emit(
            run_id, {"type": "run_complete", "run_id": run_id, "chain_id": run.chain_id}
        )
    except Exception as exc:
        log.exception("run %s failed", run_id)
        await _fail_run(state, run_id, str(exc))
    finally:
        state._active_run_id = None
        _RUN_BODIES.pop(run_id, None)


async def _worker_loop(state: AppState) -> None:
    while True:
        run_id = await state._pending.get()
        # A running self-test owns the GPU; timings would be meaningless and
        # the deep test stops the warm session.
        while state.selftest.running:
            await asyncio.sleep(1.0)
        try:
            await _execute_run(state, run_id)
        except Exception:
            log.exception("worker crashed on run %s", run_id)
        finally:
            state._pending.task_done()


def _ensure_web_deps() -> None:
    ensure_python_requirements()


def create_app(
    state: AppState,
    mount_static: bool = True,
    ws_handler: Callable[..., Any] | None = None,
) -> Any:
    _ensure_web_deps()
    from fastapi import FastAPI, HTTPException, WebSocket
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from h3_console import install_console_logging

        install_console_logging()
        state.ensure_worker()
        loop = asyncio.get_running_loop()

        def _on_interrupt() -> None:
            state.on_console_interrupt()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _on_interrupt)
            except (NotImplementedError, RuntimeError):
                pass

        # Preview weights are tiny (~22 MB) and required for live denoise previews.
        # Also adopt any on-disk MLX SAM pack so Faces upgrades need no click.
        def _boot_light_models() -> None:
            try:
                from h3_ssl import ensure_ssl_certs

                ensure_ssl_certs()
            except Exception:
                log.exception("background SSL bootstrap failed")
            try:
                from h3_preview import ensure_taeh3

                ensure_taeh3(model_dir=state.engine.model_dir)
            except Exception:
                log.exception("background taeh3 ensure failed")
            try:
                from h3_sam import adopt_local_sam3

                adopt_local_sam3(state.engine.model_dir)
            except Exception:
                log.exception("background SAM adopt failed")

        loop.run_in_executor(None, _boot_light_models)
        yield
        state.engine.shutdown(wait=True)

    app = FastAPI(title="h3-ws", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if ws_handler is not None:

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            from server import WsProtocolAdapter

            await ws_handler(WsProtocolAdapter(websocket, websocket.client))

    def _defaults() -> dict[str, Any]:
        if state.runtime_defaults:
            return dict(state.runtime_defaults)
        return {
            "num_frames": 22,
            "width": 512,
            "height": 512,
            "num_steps": H3_DEFAULT_STEPS,
            "layers": H3_DEFAULT_LAYERS,
            "reuse": H3_DEFAULT_REUSE,
            "fps": FPS,
            "quality": "fast",
        }

    def _longtake_public() -> dict[str, Any]:
        from h3_longtake import (
            DEFAULT_INTERPOLATION,
            DEFAULT_KEYFRAMES,
            DEFAULT_LONGTAKE_DENOISE,
            MAX_LONGTAKE_DENOISE,
            MIN_LONGTAKE_DENOISE,
        )

        return {
            "available": True,
            "enabled_default": False,
            "denoise": DEFAULT_LONGTAKE_DENOISE,
            "denoise_min": MIN_LONGTAKE_DENOISE,
            "denoise_max": MAX_LONGTAKE_DENOISE,
            "keyframes": DEFAULT_KEYFRAMES,
            "interpolation": DEFAULT_INTERPOLATION,
            "note": (
                "Opt-in long-take refine: init-video re-denoise + temporal blend. "
                "Adds a new library clip; the original is kept."
            ),
        }

    def _redetail_public() -> dict[str, Any]:
        from h3_redetail import (
            DEFAULT_REDETAIL_DENOISE,
            MAX_REDETAIL_DENOISE,
            MIN_REDETAIL_DENOISE,
            redetail_available,
        )

        return {
            "available": redetail_available(),
            "enabled_default": False,
            "denoise": DEFAULT_REDETAIL_DENOISE,
            "denoise_min": MIN_REDETAIL_DENOISE,
            "denoise_max": MAX_REDETAIL_DENOISE,
            "note": (
                "Opt-in ReDetail: ×2 canvas + H3 init-video refine as a new library clip. "
                "Original kept. Independent of Upframe. Heavy — default off."
            ),
        }

    def _upframe_public() -> dict[str, Any]:
        from h3_upframe import DEFAULT_MULTIPLIER, film_net_path, upframe_available

        weights = film_net_path(state.engine.model_dir) is not None
        return {
            "available": upframe_available(state.engine.model_dir),
            "enabled_default": False,
            "multiplier": DEFAULT_MULTIPLIER,
            "weights": weights,
            "note": (
                "Opt-in Upframe: real FILM (torch/MPS) temporal interpolate as a new "
                "library clip. Original kept. Needs film_net_fp16 in Models. "
                "Independent of ReDetail."
            ),
        }

    def _faces_public(app_state: AppState) -> dict[str, Any]:
        from h3_faces import (
            ABSTAIN_FACE_PX,
            DEFAULT_FACE_CANVAS,
            DEFAULT_FACE_DENOISE,
            MAX_FACE_CANVAS,
            MAX_FACE_DENOISE,
            MIN_FACE_CANVAS,
            MIN_FACE_DENOISE,
        )
        from h3_sam import _active_backend_name, sam3_ready

        ready = sam3_ready(app_state.engine.model_dir)
        return {
            # Faces is always selectable; SAM is fetched when the pass runs.
            "available": True,
            "sam3_ready": ready,
            "backend": _active_backend_name(app_state.engine.model_dir),
            "enabled_default": False,
            "canvas": DEFAULT_FACE_CANVAS,
            "canvas_min": MIN_FACE_CANVAS,
            "canvas_max": MAX_FACE_CANVAS,
            "canvas_follows_video": True,
            "denoise": DEFAULT_FACE_DENOISE,
            "denoise_min": MIN_FACE_DENOISE,
            "denoise_max": MAX_FACE_DENOISE,
            "abstain_px": ABSTAIN_FACE_PX,
            "note": (
                "Opt-in Faces (Continuity MiniMaxH3FacePass): face refine as a new "
                "library clip; the original is kept. "
                "Face crop canvas matches the shot short edge (384–768). "
                "SAM 3.1 downloads on first Faces run (~3.3 GB MLX). "
                "Independent of ReDetail / Upframe."
            ),
        }

    @app.get("/api/health")
    async def api_health(request: Request):
        ws_url, http_url = urls_from_request(request)
        if ws_url:
            state.server_url = ws_url
        if http_url:
            state.http_url = http_url
        info = state.engine.info()
        return {
            "ok": True,
            "engine_ok": bool(info.get("ok")),
            "server_url": state.server_url,
            "web_url": state.http_url,
            "engine": info,
        }

    @app.get("/api/console")
    async def api_console(limit: int = 800):
        """Ring-buffer + log-file tail for the in-app Console panel."""
        from h3_console import console_snapshot

        return console_snapshot(limit=limit)

    @app.post("/api/console/clear")
    async def api_console_clear():
        from h3_console import clear_console

        clear_console()
        return {"ok": True}

    @app.get("/api/console/stream")
    async def api_console_stream(request: Request, after: int = 0):
        """Near-realtime console: pushes new lines as they are appended."""
        from sse_starlette.sse import EventSourceResponse

        from h3_console import get_console_since

        # sse_starlette logs every chunk at DEBUG; the console would echo
        # itself into server.log.
        logging.getLogger("sse_starlette").setLevel(logging.INFO)

        async def events():
            cursor = max(0, after)
            while not await request.is_disconnected():
                chunk = get_console_since(cursor)
                if chunk["lines"] or chunk["dropped"]:
                    cursor = chunk["seq"]
                    yield {"event": "lines", "data": json.dumps(chunk)}
                await asyncio.sleep(0.25)

        return EventSourceResponse(events(), ping=15)

    def _generation_active() -> bool:
        # Worker state, not index.json statuses: a crash can leave runs
        # marked "running" forever.
        return not state.is_pipeline_idle()

    @app.get("/api/debug/selftest")
    async def api_selftest_status():
        return state.selftest.status()

    @app.post("/api/debug/selftest")
    async def api_selftest_start(body: dict[str, Any]):
        mode = str(body.get("mode") or "quick")
        if _generation_active():
            raise HTTPException(409, "Wait for the current generation to finish")
        try:
            return state.selftest.start(mode)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/debug/selftest/cancel")
    async def api_selftest_cancel():
        state.selftest.cancel()
        return state.selftest.status()

    saved_reports: set[str] = set()

    @app.post("/api/debug/report/save")
    async def api_debug_report_save(body: dict[str, Any]):
        """Write the report zip to ~/Downloads (the desktop webview cannot
        download files) and return its path for "Show in Finder"."""
        from h3_selftest import build_report_zip

        kind = str(body.get("kind") or "snapshot")
        status = state.selftest.status()
        payload = await asyncio.to_thread(
            build_report_zip,
            engine=state.engine,
            clips=[asdict(clip) for clip in state.clips.values()],
            selftest=status if status.get("state") != "idle" else None,
            kind=kind,
        )
        folder = Path.home() / "Downloads"
        if not folder.is_dir():
            folder = state.output_dir / "reports"
            folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"h3-ws-{kind}-{time.strftime('%Y%m%d-%H%M%S')}.zip"
        path.write_bytes(payload)
        saved_reports.add(str(path))
        log.info("debug report saved: %s (%d KiB)", path, len(payload) // 1024)
        return {"ok": True, "path": str(path), "name": path.name,
                "bytes": len(payload)}

    @app.post("/api/debug/reveal")
    async def api_debug_reveal(body: dict[str, Any]):
        path = str(body.get("path") or "")
        # Only reveal reports this server wrote; never an arbitrary path.
        if path not in saved_reports or not Path(path).is_file():
            raise HTTPException(404, "Unknown report")
        subprocess.run(["open", "-R", path], check=False)
        return {"ok": True}

    @app.get("/api/debug/report")
    async def api_debug_report(kind: str = "snapshot"):
        """Zip for bug reports: device facts, self-test, timings, console, logs."""
        from fastapi import Response

        from h3_selftest import build_report_zip

        status = state.selftest.status()
        payload = await asyncio.to_thread(
            build_report_zip,
            engine=state.engine,
            clips=[asdict(clip) for clip in state.clips.values()],
            selftest=status if status.get("state") != "idle" else None,
            kind=kind,
        )
        name = f"h3-ws-{kind}-{time.strftime('%Y%m%d-%H%M%S')}.zip"
        return Response(
            payload,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @app.get("/api/pass-progress")
    async def pass_progress():
        """Latest library-pass status (Upframe / ReDetail / Faces / Long take)."""
        cur = state._pass_progress
        if not cur:
            return {"idle": True}
        return cur

    @app.get("/api/config")
    async def api_config(request: Request):
        ws_url, http_url = urls_from_request(request)
        if ws_url:
            state.server_url = ws_url
        if http_url:
            state.http_url = http_url
        info = state.engine.info()
        gb = ram_gb()
        ssd = recommend_ssd_streaming(gb)
        note = f"Native h3.c Metal. Model dir: {state.engine.model_dir}."
        if not info.get("ok"):
            note += f" Engine: {info.get('error') or 'not ready'}."
        fl_ok, fl_note = model_layout_ok(state.engine.model_dir)
        r2_ok, _ = model_layout_ok(state.engine.model_dir, need_ref2va=True)
        return {
            "server_connected": True,
            "embedded": state.embedded,
            "server_url": state.server_url,
            "web_url": state.http_url,
            "engine_ok": bool(info.get("ok")),
            "engine_error": None if info.get("ok") else info.get("error"),
            "h3_bin": str(state.engine.h3_bin),
            "model_dir": str(state.engine.model_dir),
            "models_ready": bool(fl_ok),
            "ref2va_ready": bool(r2_ok),
            "models_note": None if fl_ok else (fl_note or "Download FL2VA from Models to generate."),
            "ram_gb": gb,
            "recommend_ssd_streaming": ssd,
            "metal4": bool(info.get("metal4")),
            "chip_class": info.get("chip_class") or "other",
            "chip_brand": info.get("chip_brand") or "",
            "quality_presets": QUALITY_PRESET_LIST,
            "lora_presets": lora_catalog(state.output_dir),
            "resolution_presets": RESOLUTION_PRESETS,
            "duration_presets": DURATION_PRESETS,
            "generation_modes": GENERATION_MODES,
            "ref_kinds": [
                {"id": "image", "label": "Image", "flag": "--ref-image"},
                {"id": "silent_video", "label": "Silent video", "flag": "--ref-silent-video"},
                {"id": "video", "label": "Video (keep audio)", "flag": "--ref-video"},
                {"id": "video_audio", "label": "Video + replacement audio", "flag": "--ref-video-audio"},
                {"id": "audio", "label": "Audio (with image or video)", "flag": "--ref-audio"},
            ],
            "clip_multiplier_max": CLIP_MULTIPLIER_MAX,
            "defaults": _defaults(),
            "model_note": note,
            "pyav_available": media_available(),
            "taeh3_available": taeh3_decode_ready(),
            "taeh3_weights": taeh3_available(),
            "refine": refine_settings_public(
                read_web_settings(state.output_dir).get("refine")
            ),
            "network": network_settings_public(),
            "longtake": _longtake_public(),
            "redetail": _redetail_public(),
            "upframe": _upframe_public(),
            "faces": _faces_public(state),
        }

    # ── Models management (status + user-confirmed download) ─────────────────

    def _component_status(model_dir: Path) -> list[dict[str, Any]]:
        """Return present/missing status for FL2VA, Ref2VA, TAEH3, and TaoMate."""
        from h3_lora import BUILTIN_LORAS, lora_cached_path
        from h3_preview import preferred_taeh3_path, taeh3_available

        def _dir_gib(path: Path) -> float:
            total = 0
            for p in path.rglob("*"):
                if p.is_file():
                    try:
                        total += p.stat().st_size
                    except OSError:
                        pass
            return total / (1024 ** 3)

        def _file_gib(path: Path) -> float:
            try:
                return path.stat().st_size / (1024 ** 3) if path.is_file() else 0.0
            except OSError:
                return 0.0

        fl = fl2va_dir(model_dir)
        r2 = ref2va_dir(model_dir)
        fl_ok, _ = model_layout_ok(model_dir)
        r2_ok, _ = model_layout_ok(model_dir, need_ref2va=True)
        tae = preferred_taeh3_path(model_dir)
        tae_ok = taeh3_available(tae)
        if not tae_ok:
            # Fall back to any discovered copy (writable_root / HF-adjacent).
            from h3_preview import default_taeh3_path

            discovered = default_taeh3_path()
            if taeh3_available(discovered):
                tae = discovered
                tae_ok = True
        tao = next((p for p in BUILTIN_LORAS if p.get("id") == "taomate_h3_3step"), None)
        tao_spec = str(tao["spec"]) if tao else ""
        tao_path = lora_cached_path(tao_spec) if tao_spec else None
        tao_ok = tao_path is not None and tao_path.is_file()

        from h3_redetail import latent_upscaler_path
        from h3_sam import sam3_status
        from h3_upframe import film_net_path

        film = film_net_path(model_dir)
        film_ok = film is not None and film.is_file()
        lat = latent_upscaler_path(model_dir)
        lat_ok = lat is not None and lat.is_file()

        components = [
            {
                "id": "fl2va",
                "label": "FL2VA (core)",
                "present": fl_ok,
                "path": str(fl),
                "size_gib": round(_dir_gib(fl), 1),
                "note": "Required for text-to-video and first/last-frame modes.",
                "essential": True,
            },
            {
                "id": "ref2va",
                "label": "Ref2VA (references)",
                "present": r2_ok,
                "path": str(r2),
                "size_gib": round(_dir_gib(r2), 1),
                "note": "Required for reference modes (image / video / audio).",
                "essential": True,
            },
            {
                "id": "taeh3",
                "label": "TAEH3 live preview",
                "present": tae_ok,
                "path": str(tae),
                "size_gib": round(_file_gib(tae), 3),
                "note": "Tiny live-preview decoder (~22 MB). Auto-fetched on launch.",
                "essential": True,
            },
            {
                "id": "taomate",
                "label": "TaoMate turbo",
                "present": tao_ok,
                "path": str(tao_path) if tao_path else tao_spec,
                "size_gib": round(_file_gib(tao_path), 2) if tao_path else 2.4,
                "note": "3-step turbo LoRA (~2.4 GB). Optional — enables Turbo quality.",
                "essential": False,
            },
            {
                "id": "film_net",
                "label": "Upframe (FILM)",
                "present": film_ok,
                "path": str(film) if film else "models/frame_interpolation/film_net_fp16.safetensors",
                "size_gib": round(_file_gib(film), 3) if film else 0.066,
                "note": "FILM frame interpolation (~69 MB). Enables Upframe.",
                "essential": False,
            },
            {
                "id": "latent_upscale",
                "label": "ReDetail latent ×2",
                "present": lat_ok,
                "path": str(lat) if lat else "models/latent_upscale/",
                "size_gib": round(_file_gib(lat), 2) if lat else 0.0,
                "note": "Optional learned H3 latent upscaler. ReDetail v1 works without it (pixel bridge).",
                "essential": False,
            },
            sam3_status(model_dir),
        ]
        return components

    @app.get("/api/models")
    async def api_models_status():
        from h3_paths import discover_existing_model_dirs, load_desktop_config
        from h3_lora import lora_cache_dir, scan_disk_loras

        model_dir = state.engine.model_dir
        cfg = load_desktop_config()
        disk = scan_disk_loras()
        return {
            "ok": True,
            "model_dir": str(model_dir),
            "repo_root": str(cfg.get("repo_root") or ""),
            "lora_dir": str(lora_cache_dir()),
            "lora_count": len(disk),
            "components": _component_status(model_dir),
            "candidates": [str(p) for p in discover_existing_model_dirs()],
        }

    def _normalize_model_dir_choice(raw: str | Path) -> Path:
        """Accept MiniMax-H3 itself or an h3-ws clone root containing models/MiniMax-H3."""
        from h3_paths import looks_like_minimax_h3

        choice = Path(raw).expanduser()
        try:
            choice = choice.resolve()
        except OSError:
            choice = choice.absolute()
        if choice.name != "MiniMax-H3" and looks_like_minimax_h3(choice / "models" / "MiniMax-H3"):
            return (choice / "models" / "MiniMax-H3").resolve()
        return choice

    def _apply_model_dir(new_dir: Path) -> dict[str, Any]:
        from h3_paths import (
            apply_desktop_config_env,
            discover_existing_model_dirs,
            looks_like_minimax_h3,
            save_desktop_config,
        )
        from h3_lora import lora_cache_dir, lora_catalog, scan_disk_loras

        path = _normalize_model_dir_choice(new_dir)
        path.mkdir(parents=True, exist_ok=True)
        # Drop warm FL2VA session so the next gen reloads from the new tree.
        stop = getattr(state.engine, "_stop_session", None)
        if callable(stop):
            stop()
        state.engine.model_dir = path
        payload: dict[str, Any] = {
            "setup_done": True,
            "model_dir": str(path),
        }
        if path.parent.name == "models":
            repo = path.parent.parent
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
        save_desktop_config(payload)
        apply_desktop_config_env()
        disk = scan_disk_loras()
        return {
            "ok": True,
            "model_dir": str(path),
            "lora_dir": str(lora_cache_dir()),
            "lora_count": len(disk),
            "lora_presets": lora_catalog(state.output_dir),
            "looks_valid": looks_like_minimax_h3(path),
            "components": _component_status(path),
            "candidates": [str(p) for p in discover_existing_model_dirs()],
        }

    @app.post("/api/models/path")
    async def api_models_set_path(body: dict[str, Any]):
        """Point the engine at an existing MiniMax-H3 folder (or clone root)."""
        raw = str(body.get("path") or "").strip()
        if not raw:
            raise HTTPException(400, "path is required")
        try:
            return await asyncio.to_thread(_apply_model_dir, Path(raw))
        except OSError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/models/browse")
    async def api_models_browse():
        """Open a native folder picker (server-side) and apply the chosen path."""

        def _pick() -> str | None:
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
            root.update_idletasks()
            from h3_paths import default_model_dir

            initial = str(default_model_dir().parent)
            path = filedialog.askdirectory(
                parent=root,
                title="Select MiniMax-H3 folder (or h3-ws clone root)",
                initialdir=initial,
                mustexist=True,
            )
            root.destroy()
            return path or None

        picked = await asyncio.to_thread(_pick)
        if not picked:
            return {"ok": False, "cancelled": True, "model_dir": str(state.engine.model_dir)}
        return await asyncio.to_thread(_apply_model_dir, Path(picked))

    # ── Download state for SSE progress ───────────────────────────────────────
    # Owned server-side so the task survives client disconnects. Stores the
    # asyncio.Task handle so a reconnect re-attaches instead of double-starting.
    _download_state: dict[str, Any] = {
        "active": False,
        "component": None,
        "error": None,
        "task": None,
        "terminal_pending": False,
    }

    # Approximate expected sizes in bytes (matching scripts/download_model.py).
    EXPECTED_BYTES = {
        "fl2va": 134.1 * 1024**3,   # ~134 GB
        "ref2va": 61.7 * 1024**3,   # ~62 GB (transformer only)
        "taeh3": 22 * 1024**2,      # ~22 MB
        "taomate": 2.4 * 1024**3,   # ~2.4 GB
        "dmad": 1.4 * 1024**3,      # API-compat only (not listed in Models UI)
        "film_net": 69 * 1024**2,    # ~69 MB FILM
        "latent_upscale": 500 * 1024**2,  # ~0.5 GB estimate; optional
        "sam3": 3.3 * 1024**3,      # ~3.3 GB mlx-community/sam3.1-bf16
    }
    _DOWNLOAD_COMPONENT_HELP = (
        "fl2va, ref2va, taeh3, taomate, film_net, latent_upscale, sam3"
    )

    _COMPONENT_DIR = {"fl2va": "FL2VA", "ref2va": "Ref2VA"}
    _LIGHT_COMPONENTS = frozenset(
        {"taeh3", "taomate", "film_net", "latent_upscale", "sam3"}
    )
    _ALL_DOWNLOAD_COMPONENTS = frozenset(
        {
            "fl2va",
            "ref2va",
            "taeh3",
            "taomate",
            "dmad",
            "film_net",
            "latent_upscale",
            "sam3",
        }
    )

    def _sum_hf_local_progress(root: Path) -> tuple[int, int]:
        """Sum complete + .incomplete bytes under a huggingface_hub local_dir cache."""
        if not root.is_dir():
            return 0, 0
        incomplete = 0
        complete = 0
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if p.suffix == ".incomplete" or p.name.endswith(".download"):
                incomplete += size
            elif ".cache" not in p.parts:
                complete += size
        return complete, incomplete

    def _component_progress(model_dir: Path, component: str) -> tuple[int, int]:
        """Sum bytes of .incomplete partials + already-relocated final files.

        ``snapshot_download --local-dir=models/MiniMax-H3`` writes partials into
        the component's PRIVATE cache dir:
            models/MiniMax-H3/.cache/huggingface/download/{FL2VA|Ref2VA}/...incomplete
        The global hub cache (~/.cache/huggingface/hub/blobs) is a DIFFERENT, stale
        repo and must not drive the progress bar.
        """
        if component == "taeh3":
            from h3_preview import preferred_taeh3_path

            path = preferred_taeh3_path(model_dir)
            tmp = path.with_suffix(".download")
            complete = path.stat().st_size if path.is_file() else 0
            incomplete = tmp.stat().st_size if tmp.is_file() else 0
            return complete, incomplete
        if component == "film_net":
            from h3_upframe import film_net_dest, film_net_path

            path = film_net_path(model_dir) or film_net_dest(model_dir)
            complete = path.stat().st_size if path.is_file() else 0
            return complete, 0
        if component == "latent_upscale":
            from h3_redetail import latent_upscale_dest, latent_upscaler_path

            path = latent_upscaler_path(model_dir) or latent_upscale_dest(model_dir)
            complete = path.stat().st_size if path.is_file() else 0
            return complete, 0
        if component == "sam3":
            from h3_sam import find_comfy_sam3_multiplex, sam3_ready, sam3_root

            root = sam3_root(model_dir)
            if sam3_ready(model_dir):
                total = 0
                for p in root.rglob("*"):
                    if p.is_file() and ".cache" not in p.parts:
                        try:
                            total += p.stat().st_size
                        except OSError:
                            pass
                if total <= 0:
                    comfy = find_comfy_sam3_multiplex()
                    if comfy is not None:
                        try:
                            total = comfy.stat().st_size
                        except OSError:
                            total = 0
                return total, 0
            return _sum_hf_local_progress(root)
        if component in ("taomate", "dmad"):
            from h3_lora import BUILTIN_LORAS, lora_cached_path, lora_download_dir

            lid = "taomate_h3_3step" if component == "taomate" else "dmad_h3_4step"
            recipe = next((p for p in BUILTIN_LORAS if p.get("id") == lid), None)
            if not recipe:
                return 0, 0
            hit = lora_cached_path(str(recipe["spec"]))
            if hit is not None and hit.is_file():
                return hit.stat().st_size, 0
            # Partials land under models/loras/<repo>/.cache while hf_hub_download runs.
            dest = lora_download_dir(str(recipe["spec"]))
            return _sum_hf_local_progress(dest) if dest is not None else (0, 0)
        comp_dir = _COMPONENT_DIR.get(component)
        if not comp_dir:
            return 0, 0
        root = model_dir / ".cache" / "huggingface" / "download" / comp_dir
        if not root.is_dir():
            return 0, 0
        incomplete = 0
        complete = 0
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if p.suffix == ".incomplete":
                incomplete += size
            else:
                complete += size
        return complete, incomplete

    async def _run_download(component: str) -> None:
        """Run download in the background; update _download_state on finish."""
        model_dir = state.engine.model_dir
        try:
            from h3_ssl import ensure_ssl_certs

            ensure_ssl_certs()
            expected = EXPECTED_BYTES.get(component, 0)
            if expected > 0:
                try:
                    free = shutil.disk_usage(str(model_dir)).free
                except OSError:
                    free = shutil.disk_usage(str(Path.home())).free
                # Leave ~2 GB headroom beyond the component estimate.
                need = int(expected * 1.05) + 2 * 1024**3
                if free < need:
                    raise RuntimeError(
                        f"Not enough free disk for {component}: "
                        f"need ~{need / 1024**3:.0f} GB, have {free / 1024**3:.0f} GB free."
                    )
            if component == "taeh3":
                from h3_preview import download_taeh3

                await asyncio.to_thread(download_taeh3, model_dir=model_dir)
                _download_state["error"] = None
            elif component == "taomate":
                from h3_lora import BUILTIN_LORAS, ensure_lora

                tao = next(
                    (p for p in BUILTIN_LORAS if p.get("id") == "taomate_h3_3step"),
                    None,
                )
                if not tao:
                    raise RuntimeError("TaoMate builtin recipe missing")
                await asyncio.to_thread(ensure_lora, str(tao["spec"]))
                _download_state["error"] = None
            elif component == "dmad":
                from h3_lora import BUILTIN_LORAS, ensure_lora

                dmad = next(
                    (p for p in BUILTIN_LORAS if p.get("id") == "dmad_h3_4step"),
                    None,
                )
                if not dmad:
                    raise RuntimeError("DMAD builtin recipe missing")
                await asyncio.to_thread(ensure_lora, str(dmad["spec"]))
                _download_state["error"] = None
            elif component == "film_net":
                from h3_upframe import download_film_net

                await asyncio.to_thread(download_film_net, model_dir=model_dir)
                _download_state["error"] = None
            elif component == "latent_upscale":
                from h3_redetail import download_latent_upscaler

                await asyncio.to_thread(download_latent_upscaler, model_dir=model_dir)
                _download_state["error"] = None
            elif component == "sam3":
                from h3_sam import SamDownloadError, download_sam3

                try:
                    await asyncio.to_thread(download_sam3, model_dir)
                    _download_state["error"] = None
                except SamDownloadError as exc:
                    _download_state["error"] = str(exc)
                    log.warning("SAM 3.1 download blocked: %s", exc)
            else:
                cmd = _download_model_cmd("--local-dir", str(model_dir))
                if component == "ref2va":
                    cmd.append("--with-ref2va")
                # Always pull the tiny preview decoder alongside FL2VA.
                if component == "fl2va":
                    cmd.append("--with-taeh3")
                # Inherit certifi CA env so the download_model subprocess verifies TLS.
                env = os.environ.copy()
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                )
                output, _ = await proc.communicate()
                text = output.decode("utf-8", "replace")
                if proc.returncode != 0:
                    _download_state["error"] = text.strip()[-2000:] or "download failed"
                else:
                    _download_state["error"] = None
                    # Belt-and-suspenders if the script skipped taeh3.
                    if component == "fl2va":
                        try:
                            from h3_preview import ensure_taeh3

                            await asyncio.to_thread(ensure_taeh3, model_dir=model_dir)
                        except Exception:
                            pass
        except Exception as exc:
            _download_state["error"] = str(exc)
        finally:
            _download_state["active"] = False
            _download_state["task"] = None
            # Leave component/error so a reconnecting SSE client can emit the
            # terminal event without spawning a duplicate download.
            _download_state["terminal_pending"] = True

    @app.get("/api/models/download/status")
    async def api_models_download_status():
        """Return whether a download is in flight and, if so, its live progress."""
        st = _download_state
        task = st.get("task")
        active = bool(st.get("active")) and task is not None and not task.done()
        out: dict[str, Any] = {
            "active": active,
            "component": st.get("component"),
            "error": st.get("error"),
        }
        if active and st.get("component"):
            comp = st["component"]
            complete, incomplete = await asyncio.to_thread(
                _component_progress, state.engine.model_dir, comp
            )
            current = complete + incomplete
            expected = EXPECTED_BYTES.get(comp, 0)
            pct = min(99, int(100 * current / expected)) if expected > 0 else 0
            out["progress"] = {
                "percent": pct,
                "downloaded_gb": round(current / 1024**3, 2),
                "expected_gb": round(expected / 1024**3, 1),
            }
        return out

    @app.get("/api/models/download/stream")
    async def api_models_download_stream(component: str):
        """SSE endpoint for download progress.

        The download task is owned server-side and survives client disconnects:
        a reconnect re-attaches to the same in-flight task. Only ONE terminal
        event fires when the task finishes; state is then cleared so a later
        manual re-download works.
        """
        from sse_starlette.sse import EventSourceResponse
        import time

        component = component.strip().lower()
        if component not in _ALL_DOWNLOAD_COMPONENTS:
            raise HTTPException(
                400, f"component must be one of: {_DOWNLOAD_COMPONENT_HELP}"
            )

        st = _download_state
        task = st.get("task")
        running = task is not None and not task.done()

        if running and st.get("component") != component:
            raise HTTPException(409, f"another component ({st['component']}) is downloading")

        # Finished download whose terminal SSE event was never delivered (client
        # dropped mid-stream). Re-emit success without starting a duplicate fetch.
        # Failed attempts always restart so Retry works.
        pending_terminal = (
            not running
            and st.get("component") == component
            and not st.get("active")
            and st.get("terminal_pending")
            and not st.get("error")
        )

        # Spawn the download only if none is running; otherwise re-attach.
        if not running and not pending_terminal:
            st["active"] = True
            st["component"] = component
            st["error"] = None
            st["terminal_pending"] = False
            st["task"] = asyncio.create_task(_run_download(component))

        expected = EXPECTED_BYTES.get(component, 0)
        last_total = 0
        last_time = time.time()

        async def event_generator():
            nonlocal last_total, last_time
            try:
                while True:
                    cur_task = st.get("task")
                    if cur_task is None or cur_task.done():
                        break
                    complete, incomplete = await asyncio.to_thread(
                        _component_progress, state.engine.model_dir, component
                    )
                    current = complete + incomplete
                    now = time.time()
                    speed = 0.0
                    if now - last_time >= 0.5:
                        speed = (current - last_total) / (now - last_time)
                        last_total = current
                        last_time = now
                    pct = min(99, int(100 * current / expected)) if expected > 0 else 0
                    if speed > 1024**2:
                        speed_str = f"{speed / 1024**2:.1f} MB/s"
                    elif speed > 1024:
                        speed_str = f"{speed / 1024:.0f} KB/s"
                    else:
                        speed_str = f"{speed:.0f} B/s" if speed > 0 else "starting..."
                    expected_gb = round(expected / 1024**3, 4) if expected else 0
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            "percent": pct,
                            "downloaded_gb": round(current / 1024**3, 4),
                            "expected_gb": expected_gb,
                            "speed": speed_str,
                            "active": True,
                        }),
                    }
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                # Client disconnected — the download task continues server-side.
                raise

            # Terminal event — the task has finished.
            err = st.get("error")
            if err:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": err}),
                }
            else:
                yield {
                    "event": "complete",
                    "data": json.dumps({
                        "ok": True,
                        "components": _component_status(state.engine.model_dir),
                    }),
                }
            # Clear latch so a later manual re-download can start cleanly.
            if st.get("component") == component:
                st["component"] = None
                st["error"] = None
                st["terminal_pending"] = False

        return EventSourceResponse(event_generator())

    @app.post("/api/models/download")
    async def api_models_download(body: dict[str, Any]):
        """User-confirmed download of a missing model component (legacy blocking).

        Prefer /api/models/download/stream for progress tracking.
        huggingface_hub natively resumes partial downloads.
        """
        component = str(body.get("component") or "").strip().lower()
        if component not in _ALL_DOWNLOAD_COMPONENTS:
            raise HTTPException(
                400, f"component must be one of: {_DOWNLOAD_COMPONENT_HELP}"
            )

        async def _run() -> dict[str, Any]:
            try:
                await _run_download(component)
                if _download_state.get("error"):
                    return {"ok": False, "error": _download_state["error"]}
                return {
                    "ok": True,
                    "components": _component_status(state.engine.model_dir),
                }
            except Exception as exc:  # pragma: no cover - os-level failures
                return {"ok": False, "error": str(exc)}

        result = await _run()
        if not result["ok"]:
            raise HTTPException(500, result.get("error", "download failed"))
        return result

    @app.post("/api/loras/ensure")
    async def api_lora_ensure(body: dict[str, Any]):
        spec = normalize_lora_spec(str(body.get("spec") or body.get("url") or ""))
        if not spec:
            raise HTTPException(400, "spec or url is required")
        try:
            return await asyncio.to_thread(ensure_lora, spec)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/loras/custom")
    async def api_lora_custom(body: dict[str, Any]):
        spec = normalize_lora_spec(str(body.get("spec") or body.get("url") or ""))
        if not spec:
            raise HTTPException(400, "spec or url is required")
        try:
            scale = float(body.get("scale", 1.0))
        except (TypeError, ValueError):
            raise HTTPException(400, "scale must be a number")
        label = str(body.get("label") or "").strip() or _label_for_spec(spec)
        entries = read_custom_loras(state.output_dir)
        existing = next((e for e in entries if e.get("spec") == spec), None)
        if existing is not None:
            lid = str(existing["id"])
            existing["label"] = label
            existing["scale"] = scale
        else:
            lid = f"custom_{uuid.uuid4().hex[:8]}"
            entries.append(
                {"id": lid, "label": label, "spec": spec, "scale": scale, "custom": True}
            )
        write_custom_loras(state.output_dir, entries)
        try:
            await asyncio.to_thread(ensure_lora, spec)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        catalog = lora_catalog(state.output_dir)
        preset = next((p for p in catalog if p.get("id") == lid), None)
        return {
            "ok": True,
            "id": lid,
            "reused": existing is not None,
            "preset": preset,
            "lora_presets": catalog,
        }

    @app.delete("/api/loras/custom/{lora_id}")
    async def api_lora_delete(lora_id: str):
        entries = [e for e in read_custom_loras(state.output_dir) if e["id"] != lora_id]
        write_custom_loras(state.output_dir, entries)
        return {"ok": True, "lora_presets": lora_catalog(state.output_dir)}

    # ── Presets (persistent, survive restarts) ───────────────────────────────

    @app.get("/api/presets")
    async def api_presets_list():
        state.presets.sort(key=lambda p: p.get("updatedAt", ""), reverse=True)
        return {"presets": state.presets}

    @app.post("/api/presets")
    async def api_presets_save(body: dict[str, Any]):
        name = str(body.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "name is required")
        preset = dict(body)
        now = datetime.now().isoformat()
        if not preset.get("id"):
            preset["id"] = f"preset_{uuid.uuid4().hex[:8]}"
        preset["name"] = name
        preset["createdAt"] = preset.get("createdAt") or now
        preset["updatedAt"] = now
        existing = next((p for p in state.presets if p.get("id") == preset["id"]), None)
        if existing is not None:
            state.presets.remove(existing)
        state.presets.append(preset)
        state._save_user_data()
        return {"ok": True, "preset": preset}

    @app.delete("/api/presets/{preset_id}")
    async def api_presets_delete(preset_id: str):
        before = len(state.presets)
        state.presets = [p for p in state.presets if p.get("id") != preset_id]
        if len(state.presets) == before:
            raise HTTPException(404, "Preset not found")
        state._save_user_data()
        return {"ok": True, "deleted": preset_id}

    # ── Cast members (persistent, survive restarts) ──────────────────────────

    @app.get("/api/cast")
    async def api_cast_list():
        state.cast_members.sort(key=lambda c: c.get("name", "").lower())
        return {"cast": state.cast_members}

    @app.post("/api/cast")
    async def api_cast_create(body: dict[str, Any]):
        name = str(body.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "name is required")
        now = datetime.now().isoformat()
        member = {
            "id": f"cast_{uuid.uuid4().hex[:8]}",
            "name": name,
            "description": str(body.get("description") or "").strip() or None,
            "media": body.get("media") or [],
            "createdAt": now,
            "updatedAt": now,
        }
        state.cast_members.append(member)
        state._save_user_data()
        return {"ok": True, "cast": member}

    @app.put("/api/cast/{cast_id}")
    async def api_cast_update(cast_id: str, body: dict[str, Any]):
        member = next((c for c in state.cast_members if c.get("id") == cast_id), None)
        if member is None:
            raise HTTPException(404, "Cast member not found")
        name = str(body.get("name") or "").strip()
        if name:
            member["name"] = name
        if "description" in body:
            member["description"] = str(body.get("description") or "").strip() or None
        if "media" in body:
            member["media"] = body.get("media") or []
        member["updatedAt"] = datetime.now().isoformat()
        state._save_user_data()
        return {"ok": True, "cast": member}

    @app.delete("/api/cast/{cast_id}")
    async def api_cast_delete(cast_id: str):
        before = len(state.cast_members)
        state.cast_members = [c for c in state.cast_members if c.get("id") != cast_id]
        if len(state.cast_members) == before:
            raise HTTPException(404, "Cast member not found")
        state._save_user_data()
        return {"ok": True, "deleted": cast_id}

    @app.get("/api/clips")
    async def list_clips(
        chain_id: Optional[str] = None,
        project_id: Optional[str] = None,
        all_projects: bool = False,
    ):
        state.ensure_projects()
        if all_projects:
            clips = list(state.clips.values())
        elif chain_id:
            clips = [c for c in state.clips.values() if c.chain_id == chain_id]
        else:
            clips = state.clips_for_project(project_id)
        clips.sort(key=lambda c: c.created_at)
        return {
            "clips": [_clip_for_api(state, c) for c in clips],
            "project_id": project_id or state.active_project_id,
        }

    @app.get("/api/projects")
    async def list_projects():
        state.ensure_worker()
        return {
            "projects": state.list_projects(),
            "active_project_id": state.active_project_id,
        }

    @app.post("/api/projects")
    async def create_project(body: dict[str, Any] = None):  # type: ignore[assignment]
        payload = body or {}
        project = state.create_project(str(payload.get("name") or "") or None)
        return {
            "ok": True,
            "project": asdict(project),
            "projects": state.list_projects(),
            "active_project_id": state.active_project_id,
        }

    @app.patch("/api/projects/{project_id}")
    async def rename_project(project_id: str, body: dict[str, Any]):
        name = str(body.get("name") or "").strip()
        if not name:
            raise HTTPException(400, "name is required")
        project = state.rename_project(project_id, name)
        if not project:
            raise HTTPException(404, "Project not found")
        return {
            "ok": True,
            "project": asdict(project),
            "projects": state.list_projects(),
            "active_project_id": state.active_project_id,
        }

    @app.post("/api/projects/{project_id}/activate")
    async def activate_project(project_id: str):
        project = state.set_active_project(project_id)
        if not project:
            raise HTTPException(404, "Project not found")
        return {
            "ok": True,
            "project": asdict(project),
            "projects": state.list_projects(),
            "active_project_id": state.active_project_id,
        }

    @app.delete("/api/projects/{project_id}")
    async def delete_project(project_id: str):
        result = state.delete_project(project_id, delete_files=True)
        if not result.get("ok"):
            err = result.get("error")
            if err == "not_found":
                raise HTTPException(404, "Project not found")
            if err == "last_project":
                raise HTTPException(400, "Cannot delete the last project")
            raise HTTPException(400, str(err or "delete failed"))
        return {**result, "projects": state.list_projects()}

    @app.post("/api/session/clear")
    async def clear_session():
        return {"ok": True, **state.clear_session()}

    @app.delete("/api/clips/{clip_id}")
    async def delete_clip(clip_id: str):
        if not state.delete_clip_record(clip_id):
            raise HTTPException(404, "Clip not found")
        state.save_index()
        return {"ok": True, "deleted": clip_id}

    @app.get("/api/clips/{clip_id}/export")
    async def export_clip_recipe(clip_id: str):
        """Download a zip with generation settings + referenced media (+ video)."""
        from fastapi.responses import Response

        from h3_recipe import build_recipe_zip

        clip = state.clips.get(clip_id)
        if not clip:
            raise HTTPException(404, "Clip not found")
        payload = await asyncio.to_thread(
            build_recipe_zip,
            clip=clip,
            generation=clip.generation if isinstance(clip.generation, dict) else None,
            output_dir=state.output_dir,
            upload_dir=state.upload_dir,
            frames_dir=_frames_dir(state.output_dir),
            include_video=True,
        )
        stem = Path(clip.filename or clip_id).stem or clip_id
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:48]
        name = f"h3-recipe-{safe or clip_id[:8]}.zip"
        return Response(
            payload,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    @app.post("/api/clips/import")
    async def import_clip_recipe(request: Request):
        """Import a recipe zip into the active project library."""
        from h3_recipe import import_recipe_zip

        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(400, "file is required")
        read = getattr(upload, "read", None)
        if read is None:
            raise HTTPException(400, "file is required")
        raw = await read()
        if not raw:
            raise HTTPException(400, "empty archive")

        fields = set(ClipRecord.__dataclass_fields__.keys())

        def make_clip(**kwargs: Any) -> ClipRecord:
            filtered = {k: v for k, v in kwargs.items() if k in fields}
            clip = ClipRecord(**filtered)
            if clip.filename:
                clip.video_url = state.clip_url(clip.filename)
                clip.bytes = (
                    (state.output_dir / clip.filename).stat().st_size
                    if (state.output_dir / clip.filename).is_file()
                    else clip.bytes
                )
            state.clips[clip.id] = clip
            return clip

        try:
            result = await asyncio.to_thread(
                import_recipe_zip,
                raw,
                output_dir=state.output_dir,
                upload_dir=state.upload_dir,
                project_id=state.active_project_id,
                make_clip=make_clip,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except zipfile.BadZipFile as exc:
            raise HTTPException(400, "not a valid zip archive") from exc

        clip = result["clip"]
        state.save_index()
        state.touch_project(state.active_project_id)
        # Best-effort poster so the library thumb isn't black.
        if clip.filename:
            video_path = state.output_dir / clip.filename
            if video_path.is_file():
                try:
                    await asyncio.to_thread(ensure_video_poster, video_path)
                except Exception:  # noqa: BLE001
                    pass
        return {
            "ok": True,
            "clip": _clip_for_api(state, clip),
            "generation": result.get("generation"),
            "had_video": bool(result.get("had_video")),
        }

    @app.delete("/api/chains/{chain_id}")
    async def delete_chain(chain_id: str):
        count = state.delete_chain(chain_id)
        if count == 0:
            raise HTTPException(404, "Chain not found")
        return {"ok": True, "deleted": count, "chain_id": chain_id}
    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str):
        if run_id not in state.runs:
            raise HTTPException(404, "Run not found")
        run = state.runs[run_id]
        if not state.request_cancel_run(run_id):
            raise HTTPException(409, f"Cannot cancel run in state {run.status}")
        # Not yet on the GPU — mark cancelled now so library queue cards drop.
        # The worker still no-ops via is_run_cancelled when the id is dequeued.
        if state._active_run_id != run_id:
            await _abort_run_cancelled(state, run_id)
            return {"ok": True, "status": "cancelled"}
        return {"ok": True, "status": "cancelling"}

    @app.post("/api/generate")
    async def generate(body: dict[str, Any]):
        state.ensure_worker()
        prompt = str(body.get("prompt") or "").strip()
        prompts = body.get("prompts") or []
        if prompt:
            prompts = [prompt] + [p for p in prompts if str(p).strip()]
        prompts = [str(p).strip() for p in prompts if p and str(p).strip()]
        if not prompts:
            raise HTTPException(400, "prompt is required")

        ui_mode = (body.get("mode") or "t2va").strip().lower()
        try:
            refs = parse_refs_payload(body.get("refs"), validate=False)
            refs = _resolve_refs(state, refs)
            refs = _materialize_ref_edits(state, refs)
            validate_refs(refs)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if refs:
            ui_mode = "ref2va"
            body = dict(body)
            body["mode"] = "ref2va"
            width = int(body.get("width") or 512)
            height = int(body.get("height") or 512)
            match_dir = state.upload_dir / "match"
            for item in refs:
                if item.kind != "image" or (item.ref_size or "max") != "match":
                    continue
                dest = match_dir / f"{uuid.uuid4().hex[:8]}.png"
                try:
                    resize_still_to_canvas(item.path, dest, width, height)
                    item.path = dest
                except Exception:
                    log.exception("match-resize failed for %s", item.path)
            body["refs"] = [
                {
                    "kind": item.kind,
                    "path": str(item.path),
                    "audio_path": str(item.audio_path) if item.audio_path else "",
                    "name": item.name,
                    "ref_size": item.ref_size,
                }
                for item in refs
            ]
            if body.get("image_path") or body.get("end_image_path"):
                raise HTTPException(
                    400,
                    "Ref2VA references cannot be mixed with first/last-frame anchors",
                )
        elif ui_mode == "ref2va":
            raise HTTPException(
                400,
                "ref2va requires at least one reference (image, silent video, video, or audio)",
            )

        clip_count = max(1, min(CLIP_MULTIPLIER_MAX, int(body.get("clip_count") or 1)))
        if ui_mode == "ref2va":
            clip_count = 1
        continue_from = None if ui_mode == "ref2va" else body.get("continue_from")
        if clip_count > 1:
            continue_from = None
            chain_id = str(uuid.uuid4())
            prompts = [prompts[0]] * clip_count if len(prompts) == 1 else prompts
        else:
            chain_id = str(body.get("chain_id") or uuid.uuid4())

        if ui_mode == "first_frame" and not body.get("image_path") and not continue_from:
            raise HTTPException(400, "first_frame mode requires an image")
        if ui_mode == "last_frame" and not body.get("end_image_path") and not body.get("image_path"):
            raise HTTPException(400, "last_frame mode requires an image")
        if ui_mode == "fl2va" and (
            not body.get("image_path") or not body.get("end_image_path")
        ):
            raise HTTPException(400, "fl2va requires first and last images")

        try:
            settings = _clip_settings_from_body(body)
            require_ui_canvas(int(settings["width"]), int(settings["height"]))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        autocontinue = bool(body.get("autocontinue")) or clip_count > 1 or bool(continue_from)
        autoconcat = bool(body.get("autoconcat")) or clip_count > 1
        body = dict(body)
        body["autocontinue"] = autocontinue
        body["autoconcat"] = autoconcat
        if continue_from:
            body["continue_from"] = continue_from

        existing = [c for c in state.clips.values() if c.chain_id == chain_id]
        base_index = len(existing)
        run_id = str(uuid.uuid4())
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        state.ensure_projects()
        project_id = str(body.get("project_id") or state.active_project_id or "")
        if project_id and project_id not in state.projects:
            project_id = state.active_project_id or next(iter(state.projects))
        # Client-authored restore blob (pre-materialize refs). Opaque to the engine.
        generation_snap = body.get("generation")
        if not isinstance(generation_snap, dict):
            generation_snap = None

        clip_ids: list[str] = []
        for i, p in enumerate(prompts):
            clip_id = str(uuid.uuid4())
            slug = sanitize_filename(p) or "clip"
            filename = f"web_{slug}_{ts}_{i}.mp4"
            clip = ClipRecord(
                id=clip_id,
                prompt=p,
                label="ORIGINAL" if base_index == 0 and i == 0 and not continue_from else "CURRENT",
                video_url="",
                filename=filename,
                chain_id=chain_id,
                clip_index=base_index + i,
                mode=ui_mode,
                status=RunStatus.QUEUED.value,
                created_at=datetime.now().isoformat(),
                project_id=project_id,
                generation=dict(generation_snap) if generation_snap else None,
                run_id=run_id,
                **{
                    k: v
                    for k, v in settings.items()
                    if k in ClipRecord.__dataclass_fields__
                    and k not in ("project_id", "generation", "run_id")
                },
            )
            state.clips[clip_id] = clip
            clip_ids.append(clip_id)
        state.touch_project(project_id)

        longtake_body = body.get("longtake") if isinstance(body.get("longtake"), dict) else None
        redetail_body = body.get("redetail") if isinstance(body.get("redetail"), dict) else None
        upframe_body = body.get("upframe") if isinstance(body.get("upframe"), dict) else None
        faces_body = body.get("faces") if isinstance(body.get("faces"), dict) else None
        run = RunRecord(
            id=run_id,
            status=RunStatus.QUEUED.value,
            prompts=prompts,
            chain_id=chain_id,
            clip_ids=clip_ids,
            created_at=datetime.now().isoformat(),
            autocontinue=autocontinue,
            autoconcat=autoconcat,
            longtake=dict(longtake_body) if longtake_body else None,
            redetail=dict(redetail_body) if redetail_body else None,
            upframe=dict(upframe_body) if upframe_body else None,
            faces=dict(faces_body) if faces_body else None,
        )
        state.runs[run_id] = run
        _RUN_BODIES[run_id] = body
        state.save_index()
        state.event_queues[run_id] = asyncio.Queue()
        started = await state.enqueue_generation_run(run_id)
        state.save_index()
        log.info(
            "Web UI: %s run %s  clips=%d  mode=%s  %sx%s  frames=%s  quality=%s  steps=%s  layers=%s  reuse=%s",
            "starting" if started else "queued",
            run_id,
            len(clip_ids),
            ui_mode,
            settings.get("width"),
            settings.get("height"),
            settings.get("num_frames"),
            settings.get("quality"),
            settings.get("num_steps"),
            settings.get("layers"),
            settings.get("reuse"),
        )
        return {
            "run_id": run_id,
            "chain_id": chain_id,
            "clip_ids": clip_ids,
            "status": state.runs[run_id].status,
            "started_immediately": started,
        }

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str):
        if run_id not in state.runs:
            raise HTTPException(404, "Run not found")
        if run_id not in state.event_queues:
            state.event_queues[run_id] = asyncio.Queue()

        async def stream() -> AsyncIterator[str]:
            q = state.event_queues[run_id]
            run = state.runs[run_id]
            if run.status in (RunStatus.DONE.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value):
                if run.status == RunStatus.FAILED.value:
                    yield f"data: {json.dumps({'type': 'error', 'error': run.error or 'Generation failed', 'run_id': run_id})}\n\n"
                    return
                if run.status == RunStatus.CANCELLED.value:
                    yield f"data: {json.dumps({'type': 'run_cancelled', 'run_id': run_id, 'message': 'Generation cancelled'})}\n\n"
                    return
                if run.status == RunStatus.DONE.value and run.merged_clip_id:
                    merged = state.clips.get(run.merged_clip_id)
                    if merged and merged.video_url:
                        yield f"data: {json.dumps({'type': 'merged', 'video_url': merged.video_url, 'clip_id': merged.id, 'filename': merged.filename, 'chain_id': merged.chain_id})}\n\n"
                yield f"data: {json.dumps({'type': 'run_complete', 'run_id': run_id, 'chain_id': run.chain_id})}\n\n"
                return
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=120.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("run_complete", "run_done", "error", "run_cancelled"):
                        break
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/upload")
    async def upload(request: Request, kind: str = "image"):
        kind = (kind or "image").strip().lower()
        if kind not in ("image", "audio", "video"):
            raise HTTPException(400, f"unsupported upload kind: {kind}")
        try:
            return await _save_upload_file(request, state.upload_dir, kind=kind)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/library")
    async def library_list(kind: str = "image", q: str = ""):
        """Continuity-style media picker listing (uploads / frames / renders)."""
        tab = (kind or "image").strip().lower()
        if tab not in ("image", "video", "audio", "renders"):
            raise HTTPException(400, f"unsupported library tab: {tab}")
        items = await asyncio.to_thread(list_library_assets, state, tab=tab, query=q)
        return {"kind": tab, "items": items, "count": len(items)}

    @app.get("/api/media")
    async def serve_media(path: str = ""):
        """Serve an upload/output/frame file for the Continuity segment editors."""
        try:
            resolved = _resolve_existing_media(state, path)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        allowed = (
            state.upload_dir.resolve(),
            state.output_dir.resolve(),
            _frames_dir(state.output_dir).resolve(),
        )
        if not any(_is_under(resolved, root) for root in allowed):
            raise HTTPException(403, "media path not allowed")
        return FileResponse(resolved)

    @app.get("/api/media/peaks")
    async def media_peaks(path: str = ""):
        """Waveform peaks for the Continuity trim editor."""
        try:
            resolved = _resolve_existing_media(state, path)
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc
        try:
            return await asyncio.to_thread(audio_peaks, resolved)
        except Exception as exc:
            log.exception("peaks failed for %s", resolved)
            raise HTTPException(400, f"cannot read peaks: {exc}") from exc

    @app.post("/api/clips/{clip_id}/upscale")
    async def upscale_clip(clip_id: str, body: dict[str, Any] | None = None):
        raise HTTPException(
            410,
            "Pixel Lanczos upscale was removed. Use ReDetail "
            "(POST /api/clips/{id}/redetail) for ×2 H3 refine, or a larger "
            "one-pass resolution preset.",
        )

    @app.get("/api/clips/{clip_id}/faces/inspect")
    async def faces_inspect(clip_id: str):
        """Detect faces and return box overlays (no H3 repair)."""
        clip = state.clips.get(clip_id)
        if not clip or clip.status != "done":
            raise HTTPException(404, "done clip not found")
        path = state.output_dir / clip.filename
        if not path.is_file():
            raise HTTPException(404, "clip file missing")
        from h3_facepass import inspect_faces
        from h3_faces import FaceError

        try:
            result = await asyncio.to_thread(
                inspect_faces, path, model_dir=state.engine.model_dir
            )
        except FaceError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            log.exception("faces inspect failed")
            raise HTTPException(500, f"faces inspect failed: {exc}") from exc
        return {
            "ok": True,
            "clip_id": clip_id,
            "frames": result.frames,
            "width": result.width,
            "height": result.height,
            "backend": result.backend,
            "abstain": result.abstain,
            "overlays": result.overlays,
            "heights": result.heights,
            "found_count": sum(1 for ok in result.found if ok),
        }

    @app.get("/api/clips/{clip_id}/faces/overlay.png")
    async def faces_overlay_png(clip_id: str, frame: int = 0):
        """PNG of one frame with face boxes drawn (Inspect faces preview)."""
        from fastapi.responses import Response

        clip = state.clips.get(clip_id)
        if not clip or clip.status != "done":
            raise HTTPException(404, "done clip not found")
        path = state.output_dir / clip.filename
        if not path.is_file():
            raise HTTPException(404, "clip file missing")
        from h3_facepass import decode_clip_rgb, overlay_png
        from h3_sam import detect_faces_in_frame

        frames, _, _ = await asyncio.to_thread(decode_clip_rgb, path)
        if frame < 0 or frame >= len(frames):
            raise HTTPException(400, f"frame must be 0..{len(frames) - 1}")
        boxes = await asyncio.to_thread(
            detect_faces_in_frame, frames[frame], model_dir=state.engine.model_dir
        )
        png = await asyncio.to_thread(overlay_png, frames[frame], boxes, label=None)
        return Response(content=png, media_type="image/png")

    @app.post("/api/clips/{clip_id}/faces")
    async def faces_pass(clip_id: str, body: dict[str, Any] | None = None):
        """Run Faces on a done clip → new standalone library clip (source kept)."""
        body = body or {}
        clip = state.clips.get(clip_id)
        if not clip or clip.status != "done":
            raise HTTPException(404, "done clip not found")
        path = state.output_dir / clip.filename
        if not path.is_file():
            raise HTTPException(404, "clip file missing")
        if state._active_run_id is not None:
            raise HTTPException(409, "engine is busy")

        from h3_facepass import FacesSettings, run_faces_pass
        from h3_faces import (
            DEFAULT_FACE_CANVAS,
            DEFAULT_FACE_DENOISE,
            FaceError,
            clamp_face_denoise,
            face_canvas_for_video,
        )
        from h3_paths import console_h3

        canvas = face_canvas_for_video(
            clip.width,
            clip.height,
            fallback=(
                int(body["canvas"])
                if body.get("canvas") is not None
                else DEFAULT_FACE_CANVAS
            ),
        )
        denoise = clamp_face_denoise(
            float(
                body.get("denoise")
                if body.get("denoise") is not None
                else DEFAULT_FACE_DENOISE
            )
        )
        seed = int(body.get("seed") if body.get("seed") is not None else (clip.seed or 42))
        abstain = bool(body.get("abstain", False))
        confirm = bool(body.get("confirm_boxes", False))
        confirmed = body.get("boxes")
        confirmed_boxes = None
        if confirmed is not None:
            try:
                confirmed_boxes = [
                    (float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"]))
                    for b in confirmed
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(400, "boxes must be {x,y,w,h} list") from exc

        if confirm and confirmed_boxes is None:
            from h3_facepass import inspect_faces

            preview = await asyncio.to_thread(
                inspect_faces, path, model_dir=state.engine.model_dir, canvas=canvas
            )
            return {
                "ok": True,
                "needs_confirm": True,
                "overlays": preview.overlays,
                "abstain": preview.abstain,
                "backend": preview.backend,
                "frames": preview.frames,
            }

        out_name = _unique_pass_filename(state.output_dir, clip.filename, "faces")
        out_path = state.output_dir / out_name
        settings = FacesSettings(
            canvas=canvas, denoise=denoise, seed=seed, abstain=abstain
        )
        prompt = str(body.get("prompt") or clip.prompt or "a person speaking, detailed face")
        gen = clip.generation if isinstance(clip.generation, dict) else {}
        refs_raw = body.get("refs") if body.get("refs") is not None else gen.get("refs")
        try:
            face_refs = parse_refs_payload(refs_raw, validate=False)
            face_refs = _resolve_refs(state, face_refs)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        face_loras = _loras_from_body(
            state,
            {
                "loras": body.get("loras")
                if body.get("loras") is not None
                else (clip.loras or gen.get("loras") or []),
            },
        )
        face_quality = str(
            body.get("quality") or gen.get("quality") or clip.quality or "close"
        )
        face_steps = (
            int(body["num_steps"]) if body.get("num_steps") is not None else clip.num_steps
        )
        face_layers = (
            int(body["layers"]) if body.get("layers") is not None else clip.layers
        )
        face_reuse = int(body["reuse"]) if body.get("reuse") is not None else clip.reuse

        events: list[dict[str, Any]] = []

        def on_progress(phase: str, extra: dict[str, Any]) -> None:
            events.append({"phase": phase, **extra})
            console_h3("faces %s %s", phase, extra.get("message") or "")
            state.set_pass_progress(phase, extra, kind="faces", clip_id=clip_id)

        async with state._submit_lock:
            if state._active_run_id is not None:
                raise HTTPException(409, "engine is busy")
            state._active_run_id = f"faces:{clip_id}"
            state.set_pass_progress(
                "faces_detect",
                {"message": "Faces…"},
                kind="faces",
                clip_id=clip_id,
            )
            try:
                from h3_sam import SamDownloadError, download_sam3, sam3_ready

                if not sam3_ready(state.engine.model_dir):
                    console_h3("faces downloading SAM 3.1…")
                    state.set_pass_progress(
                        "faces_detect",
                        {"message": "Downloading SAM 3.1…"},
                        kind="faces",
                        clip_id=clip_id,
                    )
                    try:
                        await asyncio.to_thread(download_sam3, state.engine.model_dir)
                    except SamDownloadError as exc:
                        raise HTTPException(400, str(exc)) from exc
                if out_path.exists():
                    out_path.unlink()
                await asyncio.to_thread(
                    run_faces_pass,
                    path,
                    out_path,
                    prompt=prompt,
                    engine=state.engine,
                    settings=settings,
                    model_dir=state.engine.model_dir,
                    on_progress=on_progress,
                    confirmed_boxes=confirmed_boxes,
                    refs=face_refs,
                    loras=face_loras,
                    quality=face_quality,
                    steps=face_steps,
                    layers=face_layers,
                    reuse=face_reuse,
                )
                if not out_path.is_file():
                    raise FaceError("Faces output missing")
            except FaceError as exc:
                raise HTTPException(400, str(exc)) from exc
            except Exception as exc:
                log.exception("faces pass failed")
                raise HTTPException(500, f"faces pass failed: {exc}") from exc
            finally:
                state._active_run_id = None
                state.clear_pass_progress()

        new_clip = _spawn_processed_clip(
            state,
            clip,
            out_path,
            generation_patch={
                "faces": {
                    "enabled": True,
                    "canvas": canvas,
                    "denoise": denoise,
                    "seed": seed,
                },
            },
            seed=seed,
        )
        if out_path.is_file() and media_available():
            try:
                await asyncio.to_thread(ensure_video_poster, out_path)
            except Exception:  # noqa: BLE001
                pass
        state.save_index()
        return {
            "ok": True,
            "clip": _clip_for_api(state, new_clip),
            "events": events[-20:],
        }

    @app.post("/api/clips/{clip_id}/redetail")
    async def redetail_pass(clip_id: str, body: dict[str, Any] | None = None):
        """Run ReDetail on a done clip → new standalone library clip (source kept)."""
        body = body or {}
        clip = state.clips.get(clip_id)
        if not clip or clip.status != "done":
            raise HTTPException(404, "done clip not found")
        path = state.output_dir / clip.filename
        if not path.is_file():
            raise HTTPException(404, "clip file missing")
        if state._active_run_id is not None:
            raise HTTPException(409, "engine is busy")

        from h3_redetail import (
            DEFAULT_REDETAIL_DENOISE,
            RedetailError,
            RedetailSettings,
            clamp_redetail_denoise,
            run_redetail,
        )
        from h3_paths import console_h3

        denoise = clamp_redetail_denoise(
            float(
                body.get("denoise")
                if body.get("denoise") is not None
                else DEFAULT_REDETAIL_DENOISE
            )
        )
        seed = int(body.get("seed") if body.get("seed") is not None else (clip.seed or 42))
        out_name = _unique_pass_filename(state.output_dir, clip.filename, "redetail")
        out_path = state.output_dir / out_name
        settings = RedetailSettings(denoise=denoise, seed=seed)
        prompt = str(body.get("prompt") or clip.prompt or "detailed cinematic shot")
        gen = clip.generation if isinstance(clip.generation, dict) else {}
        refs_raw = body.get("refs") if body.get("refs") is not None else gen.get("refs")
        try:
            rd_refs = parse_refs_payload(refs_raw, validate=False)
            rd_refs = _resolve_refs(state, rd_refs)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        rd_loras = _loras_from_body(
            state,
            {
                "loras": body.get("loras")
                if body.get("loras") is not None
                else (clip.loras or gen.get("loras") or []),
            },
        )

        def on_rd_progress(phase: str, extra: dict[str, Any]) -> None:
            console_h3("redetail %s %s", phase, extra.get("message") or "")
            state.set_pass_progress(phase, extra, kind="redetail", clip_id=clip_id)

        async with state._submit_lock:
            if state._active_run_id is not None:
                raise HTTPException(409, "engine is busy")
            state._active_run_id = f"redetail:{clip_id}"
            state.set_pass_progress(
                "redetail_upscale",
                {"message": "ReDetail · ×2 refine…"},
                kind="redetail",
                clip_id=clip_id,
            )
            try:
                if out_path.exists():
                    out_path.unlink()
                await asyncio.to_thread(
                    run_redetail,
                    path,
                    out_path,
                    prompt=prompt,
                    engine=state.engine,
                    settings=settings,
                    on_progress=on_rd_progress,
                    refs=rd_refs,
                    loras=rd_loras,
                    quality=str(
                        body.get("quality") or gen.get("quality") or clip.quality or "fast"
                    ),
                    steps=int(body["num_steps"])
                    if body.get("num_steps") is not None
                    else clip.num_steps,
                    layers=int(body["layers"]) if body.get("layers") is not None else clip.layers,
                    reuse=1,
                )
                if not out_path.is_file():
                    raise RedetailError("ReDetail output missing")
            except RedetailError as exc:
                raise HTTPException(400, str(exc)) from exc
            finally:
                state._active_run_id = None
                state.clear_pass_progress()

        from h3_media import redetail_output_size

        try:
            tw, th = redetail_output_size(int(clip.width or 512), int(clip.height or 512))
        except Exception:
            tw, th = clip.width, clip.height
        new_clip = _spawn_processed_clip(
            state,
            clip,
            out_path,
            generation_patch={
                "redetail": {"enabled": True, "denoise": denoise, "seed": seed},
            },
            width=tw,
            height=th,
        )
        if out_path.is_file() and media_available():
            try:
                await asyncio.to_thread(ensure_video_poster, out_path)
            except Exception:  # noqa: BLE001
                pass
        state.save_index()
        return {"ok": True, "clip": _clip_for_api(state, new_clip)}

    @app.post("/api/clips/{clip_id}/upframe")
    async def upframe_pass(clip_id: str, body: dict[str, Any] | None = None):
        """Run Upframe (FILM) on a done clip → new standalone library clip (source kept)."""
        body = body or {}
        clip = state.clips.get(clip_id)
        if not clip or clip.status != "done":
            raise HTTPException(404, "done clip not found")
        path = state.output_dir / clip.filename
        if not path.is_file():
            raise HTTPException(404, "clip file missing")
        if state._active_run_id is not None:
            raise HTTPException(409, "engine is busy")

        from h3_upframe import DEFAULT_MULTIPLIER, UpframeError, UpframeSettings, run_upframe

        out_name = _unique_pass_filename(state.output_dir, clip.filename, "upframe")
        out_path = state.output_dir / out_name
        settings = UpframeSettings(
            multiplier=int(body.get("multiplier") or DEFAULT_MULTIPLIER)
        )

        def on_uf_progress(phase: str, extra: dict[str, Any]) -> None:
            from h3_paths import console_h3

            console_h3("upframe %s %s", phase, extra.get("message") or "")
            state.set_pass_progress(phase, extra, kind="upframe", clip_id=clip_id)

        async with state._submit_lock:
            if state._active_run_id is not None:
                raise HTTPException(409, "engine is busy")
            state._active_run_id = f"upframe:{clip_id}"
            state.set_pass_progress(
                "upframe_run",
                {"message": f"Upframe · ×{settings.multiplier}…"},
                kind="upframe",
                clip_id=clip_id,
            )
            try:
                if out_path.exists():
                    out_path.unlink()
                await asyncio.to_thread(
                    run_upframe,
                    path,
                    out_path,
                    settings=settings,
                    model_dir=state.engine.model_dir,
                    require_weights=True,
                    on_progress=on_uf_progress,
                )
                if not out_path.is_file():
                    raise UpframeError("Upframe output missing")
            except UpframeError as exc:
                raise HTTPException(400, str(exc)) from exc
            finally:
                state._active_run_id = None
                state.clear_pass_progress()

        new_clip = _spawn_processed_clip(
            state,
            clip,
            out_path,
            generation_patch={
                "upframe": {"enabled": True, "multiplier": settings.multiplier},
            },
        )
        if out_path.is_file() and media_available():
            try:
                await asyncio.to_thread(ensure_video_poster, out_path)
            except Exception:  # noqa: BLE001
                pass
        state.save_index()
        return {"ok": True, "clip": _clip_for_api(state, new_clip)}

    @app.post("/api/clips/{clip_id}/longtake")
    async def longtake_pass(clip_id: str, body: dict[str, Any] | None = None):
        """Run Long take on a done clip → new standalone library clip (source kept)."""
        body = body or {}
        clip = state.clips.get(clip_id)
        if not clip or clip.status != "done":
            raise HTTPException(404, "done clip not found")
        path = state.output_dir / clip.filename
        if not path.is_file():
            raise HTTPException(404, "clip file missing")
        if state._active_run_id is not None:
            raise HTTPException(409, "engine is busy")

        from h3_longtake import (
            DEFAULT_INTERPOLATION,
            DEFAULT_KEYFRAMES,
            DEFAULT_LONGTAKE_DENOISE,
            LongTakeError,
            LongTakeSettings,
            clamp_longtake_denoise,
            run_longtake_refine,
        )
        from h3_paths import console_h3

        denoise = clamp_longtake_denoise(
            float(
                body.get("denoise")
                if body.get("denoise") is not None
                else DEFAULT_LONGTAKE_DENOISE
            )
        )
        seed = int(body.get("seed") if body.get("seed") is not None else (clip.seed or 42))
        keyframes = str(body.get("keyframes") or DEFAULT_KEYFRAMES)
        interpolation = str(body.get("interpolation") or DEFAULT_INTERPOLATION)
        out_name = _unique_pass_filename(state.output_dir, clip.filename, "longtake")
        out_path = state.output_dir / out_name
        settings = LongTakeSettings(
            denoise=denoise,
            seed=seed,
            keyframes=keyframes,
            interpolation=interpolation,
        )
        prompt = str(body.get("prompt") or clip.prompt or "detailed cinematic shot")

        gen = clip.generation if isinstance(clip.generation, dict) else {}
        refs_raw = body.get("refs") if body.get("refs") is not None else gen.get("refs")
        try:
            lt_refs = parse_refs_payload(refs_raw, validate=False)
            lt_refs = _resolve_refs(state, lt_refs)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        lora_body = {
            "loras": body.get("loras")
            if body.get("loras") is not None
            else (clip.loras or gen.get("loras") or []),
        }
        lt_loras = _loras_from_body(state, lora_body)
        lt_quality = str(
            body.get("quality") or gen.get("quality") or clip.quality or "fast"
        )
        lt_steps = (
            int(body["num_steps"])
            if body.get("num_steps") is not None
            else clip.num_steps
        )
        lt_layers = (
            int(body["layers"]) if body.get("layers") is not None else clip.layers
        )
        lt_reuse = int(body["reuse"]) if body.get("reuse") is not None else clip.reuse

        events: list[dict[str, Any]] = []

        def on_progress(phase: str, extra: dict[str, Any]) -> None:
            events.append({"phase": phase, **extra})
            console_h3("longtake %s %s", phase, extra.get("message") or "")
            state.set_pass_progress(phase, extra, kind="longtake", clip_id=clip_id)

        async with state._submit_lock:
            if state._active_run_id is not None:
                raise HTTPException(409, "engine is busy")
            state._active_run_id = f"longtake:{clip_id}"
            state.set_pass_progress(
                "longtake_refine",
                {"message": "Long take refine…"},
                kind="longtake",
                clip_id=clip_id,
            )
            try:
                if out_path.exists():
                    out_path.unlink()
                await asyncio.to_thread(
                    run_longtake_refine,
                    path,
                    out_path,
                    prompt=prompt,
                    engine=state.engine,
                    settings=settings,
                    on_progress=on_progress,
                    refs=lt_refs,
                    loras=lt_loras,
                    quality=lt_quality,
                    steps=lt_steps,
                    layers=lt_layers,
                    reuse=lt_reuse,
                )
                if not out_path.is_file():
                    raise LongTakeError("Long take output missing")
            except LongTakeError as exc:
                raise HTTPException(400, str(exc)) from exc
            except Exception as exc:
                log.exception("longtake pass failed")
                raise HTTPException(500, f"longtake pass failed: {exc}") from exc
            finally:
                state._active_run_id = None
                state.clear_pass_progress()

        new_clip = _spawn_processed_clip(
            state,
            clip,
            out_path,
            generation_patch={
                "longtake": {
                    "enabled": True,
                    "denoise": denoise,
                    "keyframes": keyframes,
                    "seed": seed,
                },
            },
            seed=seed,
        )
        if out_path.is_file() and media_available():
            try:
                await asyncio.to_thread(ensure_video_poster, out_path)
            except Exception:  # noqa: BLE001
                pass
        state.save_index()
        return {
            "ok": True,
            "clip": _clip_for_api(state, new_clip),
            "events": events[-20:],
        }

    @app.get("/api/frames")
    async def list_frames():
        entries = _read_frame_library(state.output_dir)
        entries.sort(key=lambda e: e.get("created_at") or "", reverse=True)
        return {"frames": [_frame_for_api(e) for e in entries]}

    @app.post("/api/frames")
    async def save_frame(request: Request):
        form = await request.form()
        upload_file = form.get("file")
        if upload_file is None:
            raise HTTPException(400, "file is required")
        read = getattr(upload_file, "read", None)
        if read is None:
            raise HTTPException(400, "file is required")
        content = await read()
        if not content:
            raise HTTPException(400, "empty frame file")
        frames_root = _frames_dir(state.output_dir)
        frames_root.mkdir(parents=True, exist_ok=True)
        fid = f"frame_{uuid.uuid4().hex[:8]}"
        filename = f"{fid}.png"
        dest = frames_root / filename
        dest.write_bytes(content)
        time_raw = form.get("time_s")
        time_s: float | None = None
        if time_raw is not None and str(time_raw).strip():
            try:
                time_s = float(str(time_raw))
            except (TypeError, ValueError):
                raise HTTPException(400, "time_s must be a number") from None
        label = str(form.get("label") or "").strip() or (
            f"Frame @ {time_s:.1f}s" if time_s is not None else "Saved frame"
        )
        entry: dict[str, Any] = {
            "id": fid,
            "label": label,
            "path": str(dest.resolve()),
            "filename": filename,
            "created_at": datetime.now().isoformat(),
        }
        source_clip_id = str(form.get("source_clip_id") or "").strip() or None
        if source_clip_id:
            entry["source_clip_id"] = source_clip_id
        if time_s is not None:
            entry["time_s"] = round(time_s, 3)
        entries = _read_frame_library(state.output_dir)
        entries.append(entry)
        _write_frame_library(state.output_dir, entries)
        return {"ok": True, "frame": _frame_for_api(entry)}

    @app.delete("/api/frames/{frame_id}")
    async def delete_frame(frame_id: str):
        fid = (frame_id or "").strip()
        entries = _read_frame_library(state.output_dir)
        kept: list[dict[str, Any]] = []
        removed = None
        for entry in entries:
            if entry.get("id") == fid:
                removed = entry
            else:
                kept.append(entry)
        if removed is None:
            raise HTTPException(404, "Frame not found")
        path = Path(str(removed.get("path") or ""))
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass
        _write_frame_library(state.output_dir, kept)
        return {"ok": True, "deleted": fid, "frames": [_frame_for_api(e) for e in kept]}

    @app.get("/api/frames/files/{filename}")
    async def frame_file(filename: str):
        path = _frames_dir(state.output_dir) / Path(filename).name
        if not path.is_file():
            raise HTTPException(404, "Frame file not found")
        return FileResponse(path)

    @app.get("/api/preview/{filename}")
    async def preview_file(filename: str):
        """Serve the latest TAEH3 looping preview (mp4 or animated webp)."""
        name = Path(filename).name
        path = state.output_dir / ".preview" / name
        if not path.is_file():
            raise HTTPException(404, "Preview not found")
        suffix = path.suffix.lower()
        media = {
            ".mp4": "video/mp4",
            ".webp": "image/webp",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }.get(suffix, "application/octet-stream")
        return FileResponse(path, media_type=media)

    @app.get("/api/settings/refine")
    async def get_refine_settings():
        return refine_settings_public(read_web_settings(state.output_dir).get("refine"))

    @app.post("/api/settings/refine/discover")
    async def api_refine_discover():
        """Probe localhost for OpenAI-compatible Refine backends (LM Studio, Ollama, …)."""
        return await asyncio.to_thread(discover_refine_endpoints)

    @app.get("/api/update")
    async def api_update_check():
        from h3_update import check_for_update

        return await asyncio.to_thread(check_for_update)

    @app.get("/api/settings/network")
    async def get_network_settings():
        return network_settings_public()

    @app.put("/api/settings/network")
    async def put_network_settings(body: dict[str, Any]):
        """Persist LAN listen preference; desktop supervisor rebinds when supervised."""
        from h3_paths import (
            apply_desktop_config_env,
            configured_server_host,
            load_desktop_config,
            save_desktop_config,
        )

        if "listen_lan" not in body:
            raise HTTPException(400, "listen_lan is required")
        listen_lan = bool(body.get("listen_lan"))
        before = configured_server_host()
        payload: dict[str, Any] = {
            "listen_lan": listen_lan,
            "server_host": "0.0.0.0" if listen_lan else "127.0.0.1",
        }
        save_desktop_config(payload)
        apply_desktop_config_env()
        after = configured_server_host(cfg=load_desktop_config())
        rebinding = False
        if before != after:
            rebinding = request_network_rebind()
        out = network_settings_public()
        out["rebinding"] = rebinding
        out["restart_required"] = before != after and not rebinding
        return out

    @app.put("/api/settings/refine")
    async def put_refine_settings(body: dict[str, Any]):
        data = read_web_settings(state.output_dir)
        try:
            merged = merge_refine_settings(data.get("refine"), body)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        data["refine"] = merged
        write_web_settings(state.output_dir, data)
        return refine_settings_public(merged)

    @app.post("/api/refine")
    async def api_refine(body: dict[str, Any]):
        settings = read_web_settings(state.output_dir).get("refine") or {}
        if not settings.get("enabled"):
            raise HTTPException(400, "Refine is disabled — enable it in Features")
        base_url = str(settings.get("base_url") or "")
        if not base_url.strip():
            raise HTTPException(400, "Refine base URL is empty")
        prompt = str(body.get("prompt") or "")
        mode = str(body.get("mode") or "t2va")
        refs_raw = body.get("refs")
        ref_labels: list[str] = []
        if isinstance(refs_raw, list):
            for item in refs_raw:
                if isinstance(item, str):
                    ref_labels.append(item)
                elif isinstance(item, dict):
                    kind = item.get("kind") or item.get("type") or "ref"
                    name = item.get("name") or item.get("path") or ""
                    ref_labels.append(f"{kind}: {name}".strip(": "))
        try:
            result = await asyncio.to_thread(
                refine_prompt,
                prompt,
                base_url=base_url,
                model=str(settings.get("model") or ""),
                api_key=str(settings.get("api_key") or ""),
                mode=mode,
                refs=ref_labels or None,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(502, str(exc)) from exc
        return result

    @app.get("/api/videos/{filename}")
    async def video_file(filename: str):
        path = state.output_dir / Path(filename).name
        if not path.is_file():
            raise HTTPException(404, "Video not found")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/videos/{filename}/thumb")
    async def video_thumb(filename: str):
        """JPEG poster for library / timeline — avoids black WebKit video thumbs."""
        path = state.output_dir / Path(filename).name
        if not path.is_file():
            raise HTTPException(404, "Video not found")
        poster = await asyncio.to_thread(ensure_video_poster, path)
        if poster is None or not poster.is_file():
            raise HTTPException(404, "Poster unavailable")
        return FileResponse(poster, media_type="image/jpeg")

    if mount_static:
        dist = resolve_web_dist()
        if dist.is_dir():
            app.mount("/", StaticFiles(directory=str(dist), html=True), name="ui")

    return app


def build_combined_application(ws_handler: Callable[..., Any], state: AppState) -> Any:
    return create_app(state, mount_static=True, ws_handler=ws_handler)


async def run_uvicorn(app: Any, host: str, port: int, state: AppState | None = None) -> None:
    _ensure_web_deps()
    import uvicorn

    from h3_paths import debug_console

    # Keep our console logger; uvicorn's default log_config would replace it.
    log_level = "debug" if debug_console() else "info"
    config = uvicorn.Config(
        app, host=host, port=port, log_level=log_level, log_config=None
    )
    server = uvicorn.Server(config)
    if state is not None:
        state._uvicorn_server = server
    await server.serve()
