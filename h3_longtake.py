"""Long take refine: init-video re-denoise + temporal blend (LongTakeNoCuts Phase 1).

Ports the anti-photocopy idea from xyzDist/H3-LongTakeNoCuts onto h3.c:
decode → ``--init-video`` partial denoise → keyframe blend back into the
source RGB → remux source audio. Opt-in only; never runs unless enabled.
"""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

log = logging.getLogger("h3.longtake")

ProgressFn = Callable[[str, dict[str, Any]], None]

DEFAULT_LONGTAKE_DENOISE = 0.55
MIN_LONGTAKE_DENOISE = 0.1
MAX_LONGTAKE_DENOISE = 0.9
DEFAULT_KEYFRAMES = "0:0, 22:0, 44:1"
DEFAULT_INTERPOLATION = "linear"
FPS = 24


class LongTakeError(RuntimeError):
    """User-facing long-take refine failure."""


@dataclass
class LongTakeSettings:
    denoise: float = DEFAULT_LONGTAKE_DENOISE
    seed: int = 42
    keyframes: str = DEFAULT_KEYFRAMES
    interpolation: str = DEFAULT_INTERPOLATION  # linear | smooth


def clamp_longtake_denoise(value: float) -> float:
    return max(MIN_LONGTAKE_DENOISE, min(MAX_LONGTAKE_DENOISE, float(value)))


def parse_keyframes(
    k_str: str,
    total_frames: int,
    *,
    pixel_total_frames: int | None = None,
    interpolation: str = DEFAULT_INTERPOLATION,
) -> list[float]:
    """Port of H3BlendLatentsByFrames.parse_keyframes → per-frame blend factors.

    Keyframe string ``0:0, 22:0, 44:1`` means frame index → blend weight toward
    the refine clip (0 = keep source, 1 = full refine).
    """
    if total_frames <= 0:
        return []
    pixel_total = int(pixel_total_frames or total_frames)
    pairs: list[tuple[int, float]] = []
    for part in (k_str or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        try:
            pf, v = part.split(":", 1)
            pairs.append((int(pf.strip()), float(v.strip())))
        except ValueError:
            continue
    if not pairs:
        return [0.0] * total_frames
    pairs.sort(key=lambda x: x[0])
    ratio = total_frames / max(1, pixel_total)
    latent_pairs: list[tuple[int, float]] = []
    for p, v in pairs:
        lf = int(round(p * ratio))
        lf = max(0, min(total_frames - 1, lf))
        latent_pairs.append((lf, v))
    dedup: dict[int, float] = {}
    for f, v in latent_pairs:
        dedup[f] = v
    latent_pairs = sorted(dedup.items())
    if latent_pairs[0][0] > 0:
        latent_pairs.insert(0, (0, latent_pairs[0][1]))
    vals: list[float] = []
    interp = (interpolation or DEFAULT_INTERPOLATION).strip().lower()
    for i in range(total_frames):
        pf, pv = latent_pairs[0]
        nf, nv = latent_pairs[-1]
        for j in range(len(latent_pairs)):
            if latent_pairs[j][0] <= i:
                pf, pv = latent_pairs[j]
            if latent_pairs[j][0] >= i:
                nf, nv = latent_pairs[j]
                break
        if nf == pf:
            t = 0.0
        else:
            t = (i - pf) / (nf - pf)
        if interp == "smooth":
            t = 0.5 * (1.0 - math.cos(math.pi * t))
        vals.append(float(pv + (nv - pv) * t))
    return vals


def blend_rgb_by_frames(
    source: np.ndarray,
    refined: np.ndarray,
    factors: Sequence[float],
) -> np.ndarray:
    """``out_f = source*(1-f) + refined*f`` for each frame (uint8 RGB)."""
    if source.shape != refined.shape:
        raise LongTakeError(
            f"shape mismatch source {source.shape} vs refined {refined.shape}"
        )
    n = int(source.shape[0])
    if len(factors) != n:
        raise LongTakeError(f"factors length {len(factors)} != frames {n}")
    out = np.empty_like(source)
    src_f = source.astype(np.float32)
    ref_f = refined.astype(np.float32)
    for i, fv in enumerate(factors):
        f = float(max(0.0, min(1.0, fv)))
        if f <= 0.0:
            out[i] = source[i]
        elif f >= 1.0:
            out[i] = refined[i]
        else:
            mixed = src_f[i] * (1.0 - f) + ref_f[i] * f
            out[i] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    return out


def run_longtake_refine(
    video_path: str | Path,
    output_path: str | Path,
    *,
    prompt: str,
    engine: Any,
    settings: LongTakeSettings | None = None,
    on_progress: ProgressFn | None = None,
    refs: Sequence[Any] | None = None,
    loras: Sequence[Any] | None = None,
    quality: str = "fast",
    steps: int | None = None,
    layers: int | None = None,
    reuse: int | None = None,
    preview_latent_dir: Path | None = None,
) -> Path:
    """Full long-take refine. Requires h3 ``--init-video`` support."""
    from h3_facepass import decode_clip_rgb, remux_audio, write_rgb_mp4
    from h3_backend import GenerateRequest

    settings = settings or LongTakeSettings()
    denoise = clamp_longtake_denoise(settings.denoise)
    seed = int(settings.seed)
    video_path = Path(video_path)
    output_path = Path(output_path)
    refs_list = list(refs or [])
    loras_list = list(loras or [])
    mode = "ref2va" if refs_list else "t2va"

    def emit(phase: str, **extra: Any) -> None:
        if on_progress:
            on_progress(phase, extra)

    emit("longtake_refine", message=f"Long take refine · denoise {denoise:.2f} · {mode}")
    frames, width, height = decode_clip_rgb(video_path)
    n = int(frames.shape[0])
    factors = parse_keyframes(
        settings.keyframes,
        n,
        pixel_total_frames=n,
        interpolation=settings.interpolation,
    )

    work = Path(tempfile.mkdtemp(prefix="h3_longtake_"))
    try:
        init_mp4 = work / "init.mp4"
        refined_mp4 = work / "refined.mp4"
        # Re-encode source RGB so init-video sees a clean silent clip.
        write_rgb_mp4(frames, init_mp4, fps=FPS)

        # Few steps: inherit caller steps, but cap for a short refine when unset.
        refine_steps = int(steps) if steps is not None else 8

        req = GenerateRequest(
            prompt=prompt or "detailed cinematic shot, sharp subject",
            output_path=refined_mp4,
            width=width,
            height=height,
            num_frames=n,
            quality=str(quality or "fast"),
            steps=refine_steps,
            layers=layers,
            reuse=reuse,
            seed=seed,
            mode=mode,
            refs=refs_list,
            loras=loras_list,
            init_video=init_mp4,
            denoise_strength=denoise,
            profile=False,
            ssd_streaming=False,
            preview_latent_dir=preview_latent_dir,
        )

        def _refine_progress(mp: dict[str, Any]) -> None:
            emit(
                "longtake_refine",
                message=f"Long take refine · denoise {denoise:.2f} · {mode}",
                model_progress=mp,
            )

        engine.generate(req, on_progress=_refine_progress)

        refined, _, _ = decode_clip_rgb(refined_mp4)
        if refined.shape[0] != n:
            if refined.shape[0] > n:
                refined = refined[:n]
            else:
                pad = np.repeat(refined[-1:], n - refined.shape[0], axis=0)
                refined = np.concatenate([refined, pad], axis=0)
        if refined.shape[1:] != frames.shape[1:]:
            raise LongTakeError(
                f"refined spatial shape {refined.shape[1:]} != source {frames.shape[1:]}"
            )

        emit("longtake_blend", message="Blending refine into source…")
        blended = blend_rgb_by_frames(frames, refined, factors)
        silent = work / "longtake_silent.mp4"
        write_rgb_mp4(blended, silent, fps=FPS)
        remux_audio(silent, video_path, output_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    emit("longtake_done", message="Long take refine complete", path=str(output_path))
    return output_path
