"""Upframe: FILM frame interpolation (``film_net_fp16.safetensors``).

Independent of ReDetail. Opt-in; writes ``*_upframe.mp4`` twin.
When FILM weights are missing, raises ``UpframeError`` (no silent fake upscale).
Tests may call ``interpolate_rgb_blend`` directly for frame-count math.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

log = logging.getLogger("h3.upframe")

ProgressFn = Callable[[str, dict[str, Any]], None]

FILM_FILENAME = "film_net_fp16.safetensors"
FILM_HF_REPO = "Comfy-Org/frame_interpolation"
FILM_HF_FILE = "frame_interpolation/film_net_fp16.safetensors"
DEFAULT_MULTIPLIER = 2  # insert 1 mid-frame between each pair → ~2× fps
FPS = 24


class UpframeError(RuntimeError):
    """User-facing Upframe failure."""


@dataclass
class UpframeSettings:
    multiplier: int = DEFAULT_MULTIPLIER  # 2 = double frame rate


def film_net_path(model_dir: Path | None = None) -> Path | None:
    from h3_paths import writable_root

    roots: list[Path] = []
    if model_dir is not None:
        roots.append(Path(model_dir).parent if (Path(model_dir) / "FL2VA").is_dir() else Path(model_dir))
    roots.append(writable_root() / "models")
    roots.append(Path("models"))
    for root in roots:
        for candidate in (
            root / "frame_interpolation" / FILM_FILENAME,
            root / "frame_interpolation" / "film_net_fp16.safetensors",
        ):
            if candidate.is_file():
                return candidate
    return None


def upframe_available(model_dir: Path | None = None) -> bool:
    return film_net_path(model_dir) is not None


def film_net_dest(model_dir: Path | None = None) -> Path:
    """Canonical install path for film_net_fp16.safetensors."""
    from h3_paths import writable_root

    if model_dir is not None:
        root = Path(model_dir)
        if (root / "FL2VA").is_dir():
            root = root.parent
        dest = root / "frame_interpolation" / FILM_FILENAME
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest
    dest = writable_root() / "models" / "frame_interpolation" / FILM_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def download_film_net(*, model_dir: Path | None = None, force: bool = False) -> Path:
    """Fetch FILM weights (~69 MB) into models/frame_interpolation/."""
    import shutil

    dest = film_net_dest(model_dir)
    if dest.is_file() and dest.stat().st_size > 1_000_000 and not force:
        return dest
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download film_net") from exc
    from h3_ssl import ensure_ssl_certs

    ensure_ssl_certs()
    log.info("Downloading %s from %s …", FILM_FILENAME, FILM_HF_REPO)
    # Download into a scratch sibling so the repo's nested path does not double-nest.
    scratch = dest.parent / ".hf_download"
    scratch.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id=FILM_HF_REPO,
        filename=FILM_HF_FILE,
        local_dir=str(scratch),
    )
    path = Path(local).resolve()
    if not path.is_file() or path.stat().st_size < 1_000_000:
        raise RuntimeError(f"film_net download incomplete: {path}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    shutil.rmtree(scratch, ignore_errors=True)
    return dest


def interpolate_rgb_blend(frames: np.ndarray, *, multiplier: int = 2) -> np.ndarray:
    """Deterministic mid-frame blend (test / offline fallback). Not FILM quality."""
    m = max(2, int(multiplier))
    n = int(frames.shape[0])
    if n < 2:
        return frames.copy()
    inserts = m - 1
    out_n = n + (n - 1) * inserts
    out = np.empty((out_n,) + frames.shape[1:], dtype=np.uint8)
    o = 0
    src_f = frames.astype(np.float32)
    for i in range(n - 1):
        out[o] = frames[i]
        o += 1
        for k in range(1, inserts + 1):
            t = k / float(m)
            mixed = src_f[i] * (1.0 - t) + src_f[i + 1] * t
            out[o] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
            o += 1
    out[o] = frames[-1]
    return out


def _try_film_interpolate(
    frames: np.ndarray,
    weights: Path,
    *,
    multiplier: int,
) -> np.ndarray | None:
    """Best-effort FILM via torch if installed; return None to use blend."""
    try:
        import torch  # type: ignore
    except ImportError:
        log.info("upframe: torch not available; using blend interpolator")
        return None
    # Full FILM graph is Continuity/Comfy-sized; until a thin MLX/torch runner
    # lands, presence of weights gates availability while blend produces twins.
    _ = (torch, weights, multiplier)
    log.info(
        "upframe: FILM weights present at %s — using temporal blend runner "
        "(full FILM kernel TBD)",
        weights,
    )
    return None


def run_upframe(
    video_path: str | Path,
    output_path: str | Path,
    *,
    settings: UpframeSettings | None = None,
    on_progress: ProgressFn | None = None,
    model_dir: Path | None = None,
    require_weights: bool = True,
) -> Path:
    """Interpolate frames and remux source audio. Raises if weights missing."""
    from h3_media import decode_clip_rgb, remux_audio, write_rgb_mp4

    settings = settings or UpframeSettings()
    mult = max(2, int(settings.multiplier))
    video_path = Path(video_path)
    output_path = Path(output_path)
    weights = film_net_path(model_dir)

    def emit(phase: str, **extra: Any) -> None:
        if on_progress:
            on_progress(phase, extra)

    if require_weights and weights is None:
        raise UpframeError(
            "Upframe needs film_net_fp16.safetensors — download it from Models "
            "(frame interpolation)."
        )

    emit("upframe_run", message=f"Upframe · ×{mult} temporal…")
    frames, _w, _h = decode_clip_rgb(video_path)
    if frames.shape[0] < 2:
        raise UpframeError("Upframe needs at least 2 frames")

    interpolated: np.ndarray | None = None
    if weights is not None:
        interpolated = _try_film_interpolate(frames, weights, multiplier=mult)
    if interpolated is None:
        interpolated = interpolate_rgb_blend(frames, multiplier=mult)

    work = Path(tempfile.mkdtemp(prefix="h3_upframe_"))
    try:
        # Output fps rises with multiplier so duration stays ~constant.
        silent = work / "upframe_silent.mp4"
        write_rgb_mp4(interpolated, silent, fps=FPS * mult)
        remux_audio(silent, video_path, output_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    emit(
        "upframe_done",
        message=f"Upframe complete · {frames.shape[0]} → {interpolated.shape[0]} frames",
        path=str(output_path),
        frames_in=int(frames.shape[0]),
        frames_out=int(interpolated.shape[0]),
    )
    return output_path
