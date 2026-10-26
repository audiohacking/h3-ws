"""Upframe: FILM frame interpolation (``film_net_fp16.safetensors``).

Independent of ReDetail. Opt-in; caller writes a new output file (source kept).
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


_FILM_MODEL_CACHE: dict[str, Any] = {}


def _torch_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_film_net(weights: Path, device, dtype):
    """Load / cache FILMNet from ``film_net_fp16.safetensors``."""
    import torch
    from safetensors.torch import load_file

    from h3_film_net import FILMNet

    key = f"{weights.resolve()}::{device}::{dtype}"
    hit = _FILM_MODEL_CACHE.get(key)
    if hit is not None:
        return hit

    sd = load_file(str(weights))
    if "extract.extract_sublevels.convs.0.0.conv.weight" not in sd:
        raise UpframeError(
            f"Unrecognized FILM weights at {weights} "
            "(expected Comfy-Org film_net_fp16.safetensors)"
        )
    model = FILMNet(device=device, dtype=dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise UpframeError(
            f"FILM weight load incomplete ({len(missing)} missing keys)"
        )
    if unexpected:
        log.warning("upframe: unexpected FILM keys ignored: %s", unexpected[:8])
    model.eval()
    model.to(device=device, dtype=dtype)
    _FILM_MODEL_CACHE[key] = model
    log.info("upframe: loaded FILM from %s on %s/%s", weights, device, dtype)
    return model


def _pad_to_align(t, align: int = 64):
    """Reflect-pad NCHW tensor so H,W are multiples of ``align`` (FILM pyramid)."""
    import torch.nn.functional as F

    _n, _c, h, w = t.shape
    ph = (align - h % align) % align
    pw = (align - w % align) % align
    if ph == 0 and pw == 0:
        return t, h, w
    # pad = (left, right, top, bottom)
    return F.pad(t, (0, pw, 0, ph), mode="reflect"), h, w


def film_interpolate(
    frames: np.ndarray,
    weights: Path,
    *,
    multiplier: int = 2,
    on_progress: ProgressFn | None = None,
) -> np.ndarray:
    """Run real FILM interpolation. ``frames`` is uint8 [T,H,W,3]."""
    import torch

    mult = max(2, int(multiplier))
    n = int(frames.shape[0])
    if n < 2:
        raise UpframeError("Upframe needs at least 2 frames")

    device = _torch_device()
    dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32
    model = _load_film_net(Path(weights), device, dtype)

    # uint8 HWC → float NCHW in [0,1]
    src = torch.from_numpy(frames).to(dtype=torch.float32).div_(255.0)
    src = src.movedim(-1, 1)  # T,C,H,W

    inserts = mult - 1
    t_values = [t / float(mult) for t in range(1, mult)]
    out_n = n + (n - 1) * inserts
    h0, w0 = int(src.shape[2]), int(src.shape[3])
    out = np.empty((out_n, h0, w0, 3), dtype=np.uint8)

    def emit(phase: str, **extra: Any) -> None:
        if on_progress:
            on_progress(phase, extra)

    emit(
        "upframe_run",
        message=f"Upframe · FILM ×{mult} on {device.type}…",
        pairs=n - 1,
    )

    out_idx = 0
    feat_cache: dict[str, Any] = {}
    prev_padded = None

    with torch.inference_mode():
        for i in range(n - 1):
            if prev_padded is None:
                img0 = src[i : i + 1].to(device=device, dtype=dtype)
                img0, _, _ = _pad_to_align(img0)
            else:
                img0 = prev_padded
            img1 = src[i + 1 : i + 1 + 1].to(device=device, dtype=dtype)
            img1, _, _ = _pad_to_align(img1)
            prev_padded = img1

            feat_cache["img0"] = (
                feat_cache.pop("next")
                if "next" in feat_cache
                else model.extract_features(img0)
            )
            feat_cache["img1"] = model.extract_features(img1)
            feat_cache["next"] = feat_cache["img1"]

            mids = model.forward_multi_timestep(
                img0, img1, t_values, cache=feat_cache
            )
            # Crop pad, move to CPU uint8
            mids = mids[:, :, :h0, :w0].float().clamp_(0.0, 1.0).cpu()
            # Write source frame i
            out[out_idx] = frames[i]
            out_idx += 1
            for k in range(inserts):
                frame = (
                    mids[k]
                    .movedim(0, -1)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .numpy()
                )
                out[out_idx] = frame
                out_idx += 1

            if on_progress and ((i + 1) % 4 == 0 or i + 1 == n - 1):
                emit(
                    "upframe_run",
                    message=f"Upframe · FILM pair {i + 1}/{n - 1}",
                    index=i + 1,
                    total=n - 1,
                )

    out[out_idx] = frames[-1]
    return out


def run_upframe(
    video_path: str | Path,
    output_path: str | Path,
    *,
    settings: UpframeSettings | None = None,
    on_progress: ProgressFn | None = None,
    model_dir: Path | None = None,
    require_weights: bool = True,
    allow_blend_fallback: bool = False,
) -> Path:
    """Interpolate frames with FILM and remux source audio.

    Product path (``require_weights=True``) refuses silent blend — that looked
    like a no-op copy. Tests may pass ``allow_blend_fallback=True``.
    """
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

    if weights is None:
        if not allow_blend_fallback:
            raise UpframeError("FILM weights missing")
        log.warning("upframe: no weights — blend fallback (tests only)")
        interpolated = interpolate_rgb_blend(frames, multiplier=mult)
        backend = "blend"
    else:
        try:
            interpolated = film_interpolate(
                frames, weights, multiplier=mult, on_progress=on_progress
            )
            backend = "film"
        except Exception as exc:
            if not allow_blend_fallback:
                raise UpframeError(f"FILM interpolation failed: {exc}") from exc
            log.warning("upframe: FILM failed (%s); blend fallback", exc)
            interpolated = interpolate_rgb_blend(frames, multiplier=mult)
            backend = "blend"

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
        message=(
            f"Upframe complete ({backend}) · "
            f"{frames.shape[0]} → {interpolated.shape[0]} frames"
        ),
        path=str(output_path),
        frames_in=int(frames.shape[0]),
        frames_out=int(interpolated.shape[0]),
        backend=backend,
    )
    return output_path
