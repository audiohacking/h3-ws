"""ReDetail: sample → spatial ×2 → H3 init-video refine (Continuity labels, H3 path).

v1 uses a high-quality pixel bridge into ``--init-video`` at 2× canvas (learned
latent ×2 weights are optional under ``models/latent_upscale/`` for a later
swap). Opt-in only; writes a ``*_redetail.mp4`` twin beside the baseline.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

log = logging.getLogger("h3.redetail")

ProgressFn = Callable[[str, dict[str, Any]], None]

DEFAULT_REDETAIL_DENOISE = 0.45
MIN_REDETAIL_DENOISE = 0.15
MAX_REDETAIL_DENOISE = 0.85
FPS = 24


class RedetailError(RuntimeError):
    """User-facing ReDetail failure."""


@dataclass
class RedetailSettings:
    denoise: float = DEFAULT_REDETAIL_DENOISE
    seed: int = 42


def clamp_redetail_denoise(value: float) -> float:
    return max(MIN_REDETAIL_DENOISE, min(MAX_REDETAIL_DENOISE, float(value)))


def redetail_available() -> bool:
    """ReDetail needs a working h3 engine path; weights for latent ×2 are optional."""
    return True


LATENT_UPSCALE_HF_REPO = "LBH-123-AI/Minimax_h3_latent_Upscaler"
LATENT_UPSCALE_NAMES = (
    "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
    "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
    "h3_clean_latent_upscaler_film_epoch200.safetensors",
)


def latent_upscaler_path(model_dir: Path | None = None) -> Path | None:
    """Return first found learned H3 latent upscaler checkpoint, if any."""
    from h3_paths import writable_root

    roots = []
    if model_dir is not None:
        roots.append(Path(model_dir))
        # When model_dir is …/MiniMax-H3, sibling models/ also holds adapters.
        roots.append(Path(model_dir).parent)
    roots.append(writable_root() / "models")
    roots.append(Path("models"))
    for root in roots:
        base = root / "latent_upscale"
        if not base.is_dir():
            continue
        for name in LATENT_UPSCALE_NAMES:
            p = base / name
            if p.is_file():
                return p
        for p in sorted(base.glob("*.safetensors")):
            return p
    return None


def latent_upscale_dest(model_dir: Path | None = None) -> Path:
    from h3_paths import writable_root

    if model_dir is not None:
        root = Path(model_dir)
        if (root / "FL2VA").is_dir():
            root = root.parent
        dest_dir = root / "latent_upscale"
    else:
        dest_dir = writable_root() / "models" / "latent_upscale"
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / LATENT_UPSCALE_NAMES[0]


def download_latent_upscaler(*, model_dir: Path | None = None, force: bool = False) -> Path:
    """Best-effort fetch of MiniMax H3 latent ×2 weights (optional for v1 bridge)."""
    hit = latent_upscaler_path(model_dir)
    if hit is not None and hit.is_file() and not force:
        return hit
    dest = latent_upscale_dest(model_dir)
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download latent upscaler") from exc
    from h3_ssl import ensure_ssl_certs

    ensure_ssl_certs()
    files = list(list_repo_files(LATENT_UPSCALE_HF_REPO))
    pick = next(
        (f for f in files if f.endswith(".safetensors") and "latent" in f.lower()),
        next((f for f in files if f.endswith(".safetensors")), None),
    )
    if not pick:
        raise RuntimeError(f"no safetensors in {LATENT_UPSCALE_HF_REPO}")
    log.info("Downloading latent upscaler %s …", pick)
    local = hf_hub_download(
        repo_id=LATENT_UPSCALE_HF_REPO,
        filename=pick,
        local_dir=str(dest.parent),
    )
    path = Path(local).resolve()
    if path != dest and path.is_file():
        import shutil

        shutil.copy2(path, dest)
    if not dest.is_file():
        return path
    return dest


def _upscale_rgb(frames: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Lanczos spatial upscale of an RGB uint8 stack (bridge until latent ×2 lands)."""
    from PIL import Image

    n = int(frames.shape[0])
    out = np.empty((n, out_h, out_w, 3), dtype=np.uint8)
    for i in range(n):
        im = Image.fromarray(frames[i], mode="RGB")
        out[i] = np.asarray(
            im.resize((out_w, out_h), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    return out


def run_redetail(
    video_path: str | Path,
    output_path: str | Path,
    *,
    prompt: str,
    engine: Any,
    settings: RedetailSettings | None = None,
    on_progress: ProgressFn | None = None,
    refs: Sequence[Any] | None = None,
    loras: Sequence[Any] | None = None,
    quality: str = "fast",
    steps: int | None = None,
    layers: int | None = None,
    reuse: int | None = None,
    preview_latent_dir: Path | None = None,
) -> Path:
    """Full ReDetail: ×2 canvas + init-video refine. Keeps source audio."""
    from h3_backend import GenerateRequest
    from h3_media import decode_clip_rgb, redetail_output_size, remux_audio, write_rgb_mp4

    settings = settings or RedetailSettings()
    denoise = clamp_redetail_denoise(settings.denoise)
    seed = int(settings.seed)
    video_path = Path(video_path)
    output_path = Path(output_path)
    refs_list = list(refs or [])
    loras_list = list(loras or [])
    mode = "ref2va" if refs_list else "t2va"

    def emit(phase: str, **extra: Any) -> None:
        if on_progress:
            on_progress(phase, extra)

    emit("redetail_upscale", message="ReDetail · scaling to ×2 canvas…")
    frames, width, height = decode_clip_rgb(video_path)
    n = int(frames.shape[0])
    out_w, out_h = redetail_output_size(width, height)
    up = _upscale_rgb(frames, out_w, out_h)

    work = Path(tempfile.mkdtemp(prefix="h3_redetail_"))
    try:
        init_mp4 = work / "init_x2.mp4"
        refined_mp4 = work / "refined_x2.mp4"
        write_rgb_mp4(up, init_mp4, fps=FPS)

        refine_steps = int(steps) if steps is not None else 8
        emit(
            "redetail_refine",
            message=f"ReDetail · refine {out_w}×{out_h} · denoise {denoise:.2f}",
        )

        req = GenerateRequest(
            prompt=prompt or "detailed cinematic shot, sharp subject, fine detail",
            output_path=refined_mp4,
            width=out_w,
            height=out_h,
            num_frames=n,
            quality=str(quality or "fast"),
            steps=refine_steps,
            layers=layers,
            reuse=reuse if reuse is not None else 1,
            seed=seed,
            mode=mode,
            refs=refs_list,
            loras=loras_list,
            init_video=init_mp4,
            denoise_strength=denoise,
            profile=False,
            ssd_streaming=False,
            preview_latent_dir=preview_latent_dir,
            allow_expanded_canvas=True,
            token_reduction=False,
            int8_row_fc2=False,
            use_slower_bf16_mlp=True,
            use_slower_bf16_qkv=True,
            use_slower_bf16_attention_output=True,
        )

        def _refine_progress(mp: dict[str, Any]) -> None:
            emit(
                "redetail_refine",
                message=f"ReDetail · refine {out_w}×{out_h} · denoise {denoise:.2f}",
                model_progress=mp,
            )

        engine.generate(req, on_progress=_refine_progress)

        # Prefer refined pixels; fall back to upscaled init if decode fails shape.
        try:
            refined, rw, rh = decode_clip_rgb(refined_mp4)
            if refined.shape[0] != n:
                if refined.shape[0] > n:
                    refined = refined[:n]
                else:
                    pad = np.repeat(refined[-1:], n - refined.shape[0], axis=0)
                    refined = np.concatenate([refined, pad], axis=0)
            if (rw, rh) != (out_w, out_h) and refined.shape[1:3] != (out_h, out_w):
                raise RedetailError(
                    f"refined spatial {refined.shape[1:3]} != target {out_h}×{out_w}"
                )
            silent = work / "redetail_silent.mp4"
            write_rgb_mp4(refined, silent, fps=FPS)
        except Exception as exc:
            log.warning("redetail decode refine failed (%s); remuxing init×2", exc)
            silent = init_mp4

        remux_audio(silent, video_path, output_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    emit(
        "redetail_done",
        message=f"ReDetail complete · {out_w}×{out_h}",
        path=str(output_path),
        width=out_w,
        height=out_h,
    )
    return output_path
