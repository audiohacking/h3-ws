"""H3 Faces post-pass — Continuity ``MiniMaxH3FacePass`` orchestration.

Detect → crop → init-video re-denoise with per-latent soft-hold → composite.

Source of truth: roadmaus/ComfyUI-Continuity ``creator/families/h3/facepass.py``
(sha256 prefix ``4488570f``). Soft-hold is ``--denoise-strength`` (ceiling) plus
``--denoise-video-strengths`` (length = ``video_t``), not a collapsed window
denoise and never a post-step latent blend.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from h3_faces import (
    ABSTAIN_FACE_PX,
    DEFAULT_FACE_CANVAS,
    DEFAULT_FACE_DENOISE,
    FaceError,
    PASTE_DILATION,
    Box,
    clamp_face_denoise,
    crop_boxes,
    face_canvas_for_video,
    face_heights,
    feather_in_canvas,
    latent_strengths,
    paste_weights,
    should_abstain,
    snap_face_canvas,
    strengths,
    video_latent_t,
    window_weights,
    windows,
    write_strengths_file,
)
from h3_sam import sam3_ready, track_faces_in_clip

log = logging.getLogger("h3.facepass")

ProgressFn = Callable[[str, dict[str, Any]], None]
FPS = 24


@dataclass
class FacesSettings:
    canvas: int = DEFAULT_FACE_CANVAS
    denoise: float = DEFAULT_FACE_DENOISE
    seed: int = 42
    # Continuity never hard-skips close-ups — strengths() softens denoise.
    # Keep abstain as an opt-in product gate (default off).
    abstain: bool = False
    abstain_px: float = ABSTAIN_FACE_PX
    confirm_boxes: bool = False
    max_faces: int = 1  # v1: single track; multi-face later


@dataclass
class FaceInspectResult:
    frames: int
    width: int
    height: int
    backend: str
    boxes: list[Box]
    found: list[bool]
    heights: list[float]
    crops: list[Box]
    face_rects: list[Box]
    abstain: bool
    overlays: list[dict[str, Any]]


def decode_clip_rgb(path: str | Path) -> tuple[np.ndarray, int, int]:
    """Decode a clip to ``[T,H,W,3]`` uint8 RGB plus width/height."""
    from h3_media import decode_clip_rgb as _decode

    frames, w, h = _decode(path)
    return np.asarray(frames), int(w), int(h)


def warp_crop_window(
    frames: np.ndarray,
    crops: Sequence[Box],
    span: tuple[int, int],
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    """Bilinear crop+resize for one window → ``[N,H,W,3]`` uint8.

    Continuity ``facepass._crop``: one ``affine_grid`` + ``grid_sample`` per
    frame at **sub-pixel** coordinates (``align_corners=False``). Mapping the
    crop with OpenCV perspective corners at ``(W-1, H-1)`` quantises the box —
    Continuity calls that the largest source of frame-to-frame jitter once the
    trajectory is already smoothed.
    """
    import torch
    import torch.nn.functional as F

    start, end = span
    length = end - start
    source_h, source_w = int(frames.shape[1]), int(frames.shape[2])
    canvas_w, canvas_h = int(canvas_w), int(canvas_h)
    block = torch.from_numpy(np.asarray(frames[start:end])).float().div_(255.0)
    block = block[..., :3].movedim(-1, 1).contiguous()  # [N,3,H,W]
    theta = torch.zeros((length, 2, 3), dtype=torch.float32)
    for offset, index in enumerate(range(start, end)):
        x, y, box_w, box_h = crops[index]
        theta[offset, 0, 0] = float(box_w) / float(source_w)
        theta[offset, 0, 2] = (2.0 * float(x) + float(box_w)) / float(source_w) - 1.0
        theta[offset, 1, 1] = float(box_h) / float(source_h)
        theta[offset, 1, 2] = (2.0 * float(y) + float(box_h)) / float(source_h) - 1.0
    grid = F.affine_grid(
        theta, (length, 3, canvas_h, canvas_w), align_corners=False
    )
    # ``border`` = Continuity: off-frame crops get edge replicate, not black.
    sampled = F.grid_sample(
        block, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    out = sampled.movedim(1, -1).clamp(0, 1).mul(255).byte().cpu().numpy()
    return np.ascontiguousarray(out)


def write_rgb_mp4(frames: np.ndarray, dest: str | Path, fps: int = FPS) -> Path:
    """Write ``[T,H,W,3]`` uint8 RGB to an H.264 MP4 (silent)."""
    from h3_media import write_rgb_mp4 as _write

    return _write(frames, dest, fps=fps)


def remux_audio(video_no_audio: str | Path, audio_source: str | Path, dest: str | Path) -> Path:
    """Copy video from ``video_no_audio`` and audio from ``audio_source``."""
    from h3_media import remux_audio as _remux

    return _remux(video_no_audio, audio_source, dest)


def mux_window_audio(
    silent_video: str | Path,
    audio_source: str | Path,
    dest: str | Path,
    *,
    start_frame: int,
    end_frame: int,
    fps: int = FPS,
) -> Path:
    """Attach source audio for ``[start_frame, end_frame)`` onto a silent crop.

    Continuity encodes ``spill.sound_between`` into the window latent so the
    mouth still attends to speech while ``hold_audio`` keeps the waveform.

    Video is re-encoded as fixed CFR ``fps`` with stamped ``1/fps`` PTS so
    h3-av's time-based ``fps=24`` resample stays 1:1 with crop indices.
    Chip-agnostic (M3 Ultra / M5 / other) — collapsed PTS desync was first
    reported on M5 but the remux bug is not Metal-class-specific.
    """
    import av

    from h3_media import _add_h264_stream, _frame_for_h264_encode, _sane_video_fps

    dest = Path(dest)
    silent_video = Path(silent_video)
    audio_source = Path(audio_source)
    dest.parent.mkdir(parents=True, exist_ok=True)

    ain = av.open(str(audio_source))
    try:
        a_streams = [s for s in ain.streams if s.type == "audio"]
        if not a_streams:
            shutil.copy2(silent_video, dest)
            return dest
    finally:
        ain.close()

    rate = float(max(1, int(fps)))
    t0 = float(start_frame) / rate
    t1 = float(end_frame) / rate
    try:
        vin = av.open(str(silent_video))
        ain = av.open(str(audio_source))
        try:
            out = av.open(str(dest), mode="w")
            try:
                v_in = vin.streams.video[0]
                a_in = next(s for s in ain.streams if s.type == "audio")
                # Force product fps — never trust a broken average_rate from
                # a prior encode (same class as trim_media PTS collapse).
                encode_fps = float(max(1, int(round(_sane_video_fps(v_in, default=rate)))))
                if abs(encode_fps - rate) >= 1.0:
                    encode_fps = rate
                v_out = _add_h264_stream(
                    out,
                    fps=encode_fps,
                    width=int(v_in.width),
                    height=int(v_in.height),
                    preset="veryfast",
                    crf="18",
                    color_from=v_in,
                )
                v_out.options = {
                    **dict(getattr(v_out, "options", {}) or {}),
                    "bf": "0",
                    "g": "1",
                    "keyint_min": "1",
                    "sc_threshold": "0",
                    "x264-params": "keyint=1:min-keyint=1:scenecut=0:bframes=0",
                }
                a_out = out.add_stream("aac", rate=a_in.rate or 48000)
                try:
                    a_out.layout = a_in.layout
                except Exception:
                    pass

                for index, frame in enumerate(vin.decode(v_in)):
                    prepared = _frame_for_h264_encode(frame, fps=encode_fps, pts=index)
                    for packet in v_out.encode(prepared):
                        out.mux(packet)
                for packet in v_out.encode(None):
                    out.mux(packet)

                # Seek near the window; demux until past t1.
                try:
                    ain.seek(int(t0 * av.time_base), any_frame=False)
                except Exception:
                    pass
                for frame in ain.decode(a_in):
                    ts = float(frame.pts * a_in.time_base) if frame.pts is not None else 0.0
                    if ts + 1e-3 < t0:
                        continue
                    if ts >= t1:
                        break
                    for packet in a_out.encode(frame):
                        out.mux(packet)
                for packet in a_out.encode(None):
                    out.mux(packet)
            finally:
                out.close()
        finally:
            vin.close()
            ain.close()
    except Exception as exc:
        log.warning("window audio mux failed (%s); using silent crop", exc)
        shutil.copy2(silent_video, dest)
    return dest


# Photographic x264 round-trips land ~25–35 MAE; do not use MAE alone.
# Primary gate: mean |i − nearest_crop_index(got[i])| after fps=24 decode.
INIT_ROUNDTRIP_INDEX_DRIFT_MAX = 0.35
# Soft secondary: absurd MAE still fails even if drift luckily looks ok.
INIT_ROUNDTRIP_MAE_MAX = 55.0
# Pre-composite shift search. Soft-hold changes appearance, so require both
# a relative and absolute MAE win — avoids spurious δ on healthy windows.
TEMPORAL_LOCK_SHIFT_RANGE = 2
TEMPORAL_LOCK_IMPROVE_RATIO = 0.80
TEMPORAL_LOCK_MIN_IMPROVE = 5.0
# Wild frame-count mismatches vs the crop window → hard fail.
REFINED_FRAME_SLACK = 2


def decode_fps24_rgb(
    path: str | Path,
    *,
    width: int,
    height: int,
    max_frames: int,
    fps: float = FPS,
) -> np.ndarray:
    """Decode like h3-av / ``h3_ffmpeg_read_video_f32``: time-resample at *fps*.

    Returns ``[T,H,W,3]`` uint8 with ``T <= max_frames``. Used to verify what
    H3 will actually feed the VAE before a Faces window generate.
    """
    import av
    from PIL import Image

    path = Path(path)
    container = av.open(str(path))
    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise FaceError(f"no video stream in {path}")
        decoded: list[tuple[float, np.ndarray]] = []
        for frame in container.decode(stream):
            t = (
                float(frame.time)
                if frame.time is not None
                else (
                    float(frame.pts * stream.time_base)
                    if frame.pts is not None and stream.time_base
                    else 0.0
                )
            )
            arr = frame.to_ndarray(format="rgb24")
            decoded.append((t, arr))
        if not decoded:
            raise FaceError(f"no frames in {path}")
        interval = 1.0 / float(fps)
        last_t = decoded[-1][0]
        if last_t <= 0:
            last_t = (len(decoded) - 1) * interval
        timed: list[np.ndarray] = []
        idx = 0
        t = 0.0
        while len(timed) < int(max_frames) and (t <= last_t + interval or not timed):
            while idx + 1 < len(decoded) and decoded[idx + 1][0] <= t + 1e-9:
                idx += 1
            img = Image.fromarray(decoded[idx][1])
            if img.size != (int(width), int(height)):
                img = img.resize((int(width), int(height)), Image.Resampling.LANCZOS)
            timed.append(np.asarray(img, dtype=np.uint8))
            t += interval
            if idx == len(decoded) - 1 and t > last_t + interval:
                break
        if not timed:
            raise FaceError(f"fps={fps} resample produced no frames from {path}")
        return np.stack(timed[: int(max_frames)], axis=0)
    finally:
        container.close()


def init_roundtrip_mae(expected: np.ndarray, got: np.ndarray) -> float:
    """Mean abs error between crop RGB and what fps=24 decode returned."""
    n = min(int(expected.shape[0]), int(got.shape[0]))
    if n <= 0:
        return 255.0
    a = expected[:n].astype(np.float32)
    b = got[:n].astype(np.float32)
    if a.shape[1:] != b.shape[1:]:
        return 255.0
    return float(np.mean(np.abs(a - b)))


def init_roundtrip_index_drift(expected: np.ndarray, got: np.ndarray) -> float:
    """Mean |i − argmin_j MAE(got[i], expected[j])|.

    Collapsed/duplicated timelines map many decoded frames onto the same
    crop index → drift ≫ 0. Healthy CFR 1:1 stays near 0 even when x264
    MAE is ~30 on photographic content.
    """
    n = min(int(expected.shape[0]), int(got.shape[0]))
    if n <= 0:
        return float(n if n else 99.0)
    if expected.shape[1:] != got.shape[1:]:
        return 99.0
    exp = expected[:n].astype(np.float32)
    # Flatten spatial for faster pairwise MAE (still unique per ramp/frame).
    exp_flat = exp.reshape(n, -1)
    got_flat = got[:n].astype(np.float32).reshape(n, -1)
    drifts: list[float] = []
    for i in range(n):
        # Mean abs over pixels — argmin is the nearest crop frame.
        errs = np.mean(np.abs(got_flat[i][None, :] - exp_flat), axis=1)
        drifts.append(float(abs(i - int(np.argmin(errs)))))
    return float(sum(drifts) / len(drifts))


def verify_init_roundtrip(
    init_path: str | Path,
    crop_rgb: np.ndarray,
    *,
    fps: int = FPS,
    drift_max: float = INIT_ROUNDTRIP_INDEX_DRIFT_MAX,
    mae_max: float = INIT_ROUNDTRIP_MAE_MAX,
) -> tuple[bool, float]:
    """True when h3-style decode preserves crop frame order.

    Returns ``(ok, mae)``. *mae* is logged for humans; the pass/fail decision
    is primarily index drift (x264 of real faces sits ~25–35 MAE).
    """
    length, height, width = int(crop_rgb.shape[0]), int(crop_rgb.shape[1]), int(crop_rgb.shape[2])
    try:
        got = decode_fps24_rgb(
            init_path, width=width, height=height, max_frames=length, fps=float(fps)
        )
    except Exception as exc:
        log.warning("Faces init round-trip decode failed (%s)", exc)
        return False, 255.0
    if int(got.shape[0]) != length:
        log.warning(
            "Faces init round-trip frame count %d != crop %d",
            got.shape[0],
            length,
        )
        return False, 255.0
    mae = init_roundtrip_mae(crop_rgb, got)
    drift = init_roundtrip_index_drift(crop_rgb, got)
    ok = drift <= float(drift_max) and mae <= float(mae_max)
    if not ok:
        log.warning(
            "Faces init round-trip weak: drift=%.3f (max %.3f) mae=%.1f (max %.1f)",
            drift,
            drift_max,
            mae,
            mae_max,
        )
    return ok, mae


def align_refined_frame_count(
    refined: np.ndarray, length: int, *, slack: int = REFINED_FRAME_SLACK
) -> np.ndarray:
    """Force refined to exactly *length* frames; reject wild mismatches."""
    got = int(refined.shape[0])
    length = int(length)
    if got == length:
        return refined
    delta = abs(got - length)
    if delta > int(slack):
        raise FaceError(
            f"Faces refined window has {got} frames; expected {length} "
            f"(slack ±{slack}) — refusing to paste a desynced crop"
        )
    if got > length:
        # Prefer leading slice: H3 usually emits extras at the tail when
        # duration/fps drift; center-cut would misalign mouth with body.
        log.warning("Faces refined %d → trim to %d (leading)", got, length)
        return refined[:length]
    pad_n = length - got
    log.warning("Faces refined %d → pad %d (repeat last)", got, pad_n)
    pad = np.repeat(refined[-1:], pad_n, axis=0)
    return np.concatenate([refined, pad], axis=0)


def _face_mask_canvas(
    face_rects: Sequence[Box], canvas_w: int, canvas_h: int
) -> np.ndarray:
    """Per-frame soft face masks in canvas space ``[T,H,W]`` float32."""
    masks = np.zeros((len(face_rects), int(canvas_h), int(canvas_w)), dtype=np.float32)
    for i, rect in enumerate(face_rects):
        feather = feather_in_canvas(rect[3], canvas_h)
        masks[i] = _paste_mask(rect, canvas_w, canvas_h, feather)
    return masks


def masked_temporal_mae(
    refined: np.ndarray,
    crop: np.ndarray,
    face_rects: Sequence[Box],
    *,
    shift: int = 0,
) -> float:
    """Mean abs error under face masks with integer *shift* on refined."""
    n = min(int(refined.shape[0]), int(crop.shape[0]), len(face_rects))
    if n <= 0:
        return 255.0
    canvas_h, canvas_w = int(crop.shape[1]), int(crop.shape[2])
    masks = _face_mask_canvas(face_rects[:n], canvas_w, canvas_h)
    total = 0.0
    weight = 0.0
    for i in range(n):
        j = i + int(shift)
        if j < 0 or j >= int(refined.shape[0]):
            continue
        m = masks[i][..., None]
        w = float(m.sum()) + 1e-6
        err = np.abs(
            refined[j].astype(np.float32) - crop[i].astype(np.float32)
        )
        total += float((err * m).sum()) / 3.0
        weight += w
    if weight <= 0:
        return 255.0
    return total / weight


def temporal_lock_refined(
    refined: np.ndarray,
    crop: np.ndarray,
    face_rects: Sequence[Box],
    *,
    shift_range: int = TEMPORAL_LOCK_SHIFT_RANGE,
    improve_ratio: float = TEMPORAL_LOCK_IMPROVE_RATIO,
    min_improve: float = TEMPORAL_LOCK_MIN_IMPROVE,
    enabled: bool = True,
) -> tuple[np.ndarray, int, float]:
    """Search ``δ ∈ [-shift_range, +shift_range]``; apply best if it helps.

    Returns ``(aligned_refined, delta, best_mae)``. Positive *delta* means
    refined content is early vs crop (refined[i+δ] matches crop[i]).

    When *enabled* is False (healthy init round-trip), only frame-count
    alignment runs — soft-hold changes face appearance enough that a
    shift search can spuriously fire on noise.
    """
    length = int(crop.shape[0])
    refined = align_refined_frame_count(refined, length)
    zero = masked_temporal_mae(refined, crop, face_rects, shift=0)
    if not enabled:
        return refined, 0, zero
    best_shift = 0
    best_mae = zero
    for delta in range(-int(shift_range), int(shift_range) + 1):
        if delta == 0:
            continue
        mae = masked_temporal_mae(refined, crop, face_rects, shift=delta)
        if mae < best_mae:
            best_mae = mae
            best_shift = delta
    improved = (zero - best_mae) >= float(min_improve)
    relative = best_mae <= float(improve_ratio) * zero
    if best_shift != 0 and improved and relative:
        log.info(
            "Faces temporal lock: shift %+d (mae %.1f → %.1f)",
            best_shift,
            zero,
            best_mae,
        )
        # refined[i+δ] → slot i: roll content so index 0 gets former δ.
        out = np.empty_like(refined)
        for i in range(length):
            j = i + best_shift
            if 0 <= j < length:
                out[i] = refined[j]
            elif j < 0:
                out[i] = refined[0]
            else:
                out[i] = refined[-1]
        return out, best_shift, best_mae
    return refined, 0, zero


def _gaussian_blur_2d(mask: np.ndarray, feather: int) -> np.ndarray:
    import cv2

    if feather <= 0:
        return mask
    size = 2 * int(feather) + 1
    shortest = min(mask.shape[0], mask.shape[1])
    if shortest <= size:
        size = max(3, int(shortest / 2) | 1)
    sigma = max(size / 6.0, 0.5)
    return cv2.GaussianBlur(mask, (size, size), sigmaX=sigma, sigmaY=sigma)


def _paste_mask(face: Box, width: int, height: int, feather: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    x, y, bw, bh = face
    x -= PASTE_DILATION
    y -= PASTE_DILATION
    bw += 2 * PASTE_DILATION
    bh += 2 * PASTE_DILATION
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1 = min(width, int(round(x + bw)))
    y1 = min(height, int(round(y + bh)))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 1.0
    return np.clip(_gaussian_blur_2d(mask, feather), 0.0, 1.0)


def composite_window(
    base: np.ndarray,
    refined: np.ndarray,
    crops: Sequence[Box],
    face_rects: Sequence[Box],
    weights: Sequence[float],
) -> np.ndarray:
    """Paste refined crops into base frames (uint8 RGB arrays).

    Continuity ``facepass._composite``: inverse ``affine_grid`` warp of the
    crop + face mask (``align_corners=False``), colour-match inside the mask,
    then opacity blend. Must use the same continuous mapping as
    ``warp_crop_window`` or the paste crawls when crop size changes.
    """
    import torch
    import torch.nn.functional as F

    count = int(base.shape[0])
    height, width = int(base.shape[1]), int(base.shape[2])
    canvas_h, canvas_w = int(refined.shape[1]), int(refined.shape[2])

    base_t = torch.from_numpy(np.asarray(base)).float().div_(255.0)
    patch = (
        torch.from_numpy(np.asarray(refined))
        .float()
        .div_(255.0)
        .movedim(-1, 1)
        .contiguous()
    )

    theta = torch.zeros((count, 2, 3), dtype=torch.float32)
    masks: list[torch.Tensor] = []
    weight_list: list[float] = []
    for index in range(count):
        w = float(weights[index])
        weight_list.append(w)
        x, y, box_w, box_h = crops[index]
        # Continuity inverse: canvas → source (same formulas as facepass.py).
        theta[index, 0, 0] = float(width) / float(box_w)
        theta[index, 0, 2] = (float(width) - 2.0 * float(x)) / float(box_w) - 1.0
        theta[index, 1, 1] = float(height) / float(box_h)
        theta[index, 1, 2] = (float(height) - 2.0 * float(y)) / float(box_h) - 1.0
        feather = feather_in_canvas(box_h, canvas_h)
        mask_np = _paste_mask(face_rects[index], canvas_w, canvas_h, feather)
        masks.append(torch.from_numpy(mask_np).view(1, 1, canvas_h, canvas_w))

    grid = F.affine_grid(theta, (count, 3, height, width), align_corners=False)
    warped = F.grid_sample(
        patch, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    ).movedim(1, -1)
    mask = F.grid_sample(
        torch.cat(masks, dim=0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).movedim(1, -1).clamp(0, 1)

    total = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    base_mean = (base_t * mask).sum(dim=(1, 2), keepdim=True) / total
    patch_mean = (warped * mask).sum(dim=(1, 2), keepdim=True) / total
    base_sd = (
        (((base_t - base_mean) ** 2) * mask).sum(dim=(1, 2), keepdim=True) / total
    ).sqrt().clamp_min(1e-6)
    patch_sd = (
        (((warped - patch_mean) ** 2) * mask).sum(dim=(1, 2), keepdim=True) / total
    ).sqrt().clamp_min(1e-6)
    matched = ((warped - patch_mean) * (base_sd / patch_sd) + base_mean).clamp(0, 1)

    opacity = mask * torch.tensor(
        weight_list, dtype=torch.float32
    ).view(-1, 1, 1, 1)
    out = ((1.0 - opacity) * base_t + opacity * matched).clamp(0, 1)
    return (out * 255.0).byte().cpu().numpy()


def inspect_faces(
    video_path: str | Path,
    *,
    model_dir: Path | None = None,
    canvas: int = DEFAULT_FACE_CANVAS,
    allow_fallback: bool = True,
) -> FaceInspectResult:
    """Detect + plan geometry for Library Inspect faces (no H3 call)."""
    from h3_sam import _active_backend_name

    frames, width, height = decode_clip_rgb(video_path)
    boxes, found = track_faces_in_clip(
        list(frames),
        model_dir=model_dir,
        allow_fallback=allow_fallback and not sam3_ready(model_dir),
    )
    canvas = snap_face_canvas(canvas)
    crops, face_rects = crop_boxes(boxes, found, canvas, canvas)
    heights = face_heights(crops)
    abstain = should_abstain(heights)
    overlays: list[dict[str, Any]] = []
    for i, (box, ok) in enumerate(zip(boxes, found)):
        if not ok:
            continue
        overlays.append(
            {
                "frame": i,
                "box": {"x": box[0], "y": box[1], "w": box[2], "h": box[3]},
                "height_px": float(box[3]),
                "crop": {
                    "x": crops[i][0],
                    "y": crops[i][1],
                    "w": crops[i][2],
                    "h": crops[i][3],
                },
            }
        )
    return FaceInspectResult(
        frames=int(frames.shape[0]),
        width=width,
        height=height,
        backend=_active_backend_name(model_dir),
        boxes=list(boxes),
        found=list(found),
        heights=heights,
        crops=list(crops),
        face_rects=list(face_rects),
        abstain=abstain,
        overlays=overlays,
    )


def build_faces_window_request(
    *,
    prompt: str,
    output_path: Path,
    init_video: Path,
    strengths_file: Path,
    canvas: int,
    num_frames: int,
    ceiling: float,
    seed: int,
    face_mode: str,
    refs_list: Sequence[Any],
    loras_list: Sequence[Any],
    quality: str,
    steps: int | None,
    layers: int | None,
    reuse: int | None,
    preview_latent_dir: Path | None,
) -> Any:
    """Build the GenerateRequest for one Faces window (soft-hold contract)."""
    from h3_backend import GenerateRequest

    return GenerateRequest(
        prompt=prompt or "a person speaking, detailed face",
        output_path=output_path,
        width=canvas,
        height=canvas,
        num_frames=num_frames,
        quality=str(quality or "close"),
        steps=steps,
        layers=layers,
        reuse=reuse if reuse is not None else 1,
        seed=seed,
        mode=face_mode,
        refs=list(refs_list),
        loras=list(loras_list),
        init_video=init_video,
        denoise_strength=ceiling,
        denoise_video_strengths_file=strengths_file,
        profile=False,
        ssd_streaming=False,
        preview_latent_dir=preview_latent_dir,
        token_reduction=False,
        # Faces windows are always init-video soft-hold: request the close
        # BF16 stack here; ``H3Engine.generate`` still runs
        # ``apply_platform_generate_defaults`` so M5 vs M3 auto-select stays
        # the single source of truth for non-Faces jobs.
        int8_row_fc2=False,
        use_slower_bf16_mlp=True,
        use_slower_bf16_qkv=True,
        use_slower_bf16_attention_output=True,
        # Continuity face_payload: refs only — never FL2VA anchors.
        first_frame=None,
        last_frame=None,
    )


def run_faces_pass(
    video_path: str | Path,
    output_path: str | Path,
    *,
    prompt: str,
    engine: Any,
    settings: FacesSettings | None = None,
    model_dir: Path | None = None,
    on_progress: ProgressFn | None = None,
    confirmed_boxes: Sequence[Box] | None = None,
    refs: Sequence[Any] | None = None,
    loras: Sequence[Any] | None = None,
    quality: str = "close",
    steps: int | None = None,
    layers: int | None = None,
    reuse: int | None = None,
    preview_latent_dir: Path | None = None,
) -> Path:
    """Full Faces repair. Writes ``output_path`` (typically ``*_faces.mp4`` twin)."""
    settings = settings or FacesSettings()
    ceiling = clamp_face_denoise(settings.denoise)
    seed = int(settings.seed)
    video_path = Path(video_path)
    output_path = Path(output_path)
    refs_list = list(refs or [])
    loras_list = list(loras or [])
    face_mode = "ref2va" if refs_list else "t2va"

    def emit(phase: str, **extra: Any) -> None:
        if on_progress:
            on_progress(phase, extra)

    emit("faces_detect", message="Detecting faces (SAM 3.1)…")
    frames, width, height = decode_clip_rgb(video_path)
    # Face crop canvas always tracks the source video short edge (384–768).
    canvas = face_canvas_for_video(width, height, fallback=settings.canvas)
    n = int(frames.shape[0])
    emit(
        "faces_detect",
        message=f"Detecting faces (SAM 3.1) · canvas {canvas}p (matches {width}×{height})",
        canvas=canvas,
        width=width,
        height=height,
    )
    if not sam3_ready(model_dir):
        raise FaceError(
            "SAM 3.1 is required for Faces. Open Models and download "
            "SAM 3.1 (mlx-community/sam3.1-bf16), then retry."
        )
    try:
        boxes, found = track_faces_in_clip(
            list(frames), model_dir=model_dir, allow_fallback=False
        )
    except Exception as exc:
        raise FaceError(f"SAM face detect failed: {exc}") from exc
    if confirmed_boxes is not None and len(confirmed_boxes) == n:
        boxes = list(confirmed_boxes)
        found = [b[2] > 0 and b[3] > 0 for b in boxes]

    if not any(found):
        log.info("Faces: no face found — left as it is")
        emit("faces_skip", message="No face found (SAM 3.1) — left as it is")
        if video_path.resolve() != output_path.resolve():
            shutil.copy2(video_path, output_path)
        return output_path

    crops, face_rects = crop_boxes(boxes, found, canvas, canvas)
    heights = face_heights(crops)
    if settings.abstain and should_abstain(heights, settings.abstain_px):
        log.info(
            "Faces: face already ≥ %.0f px — left as it is (abstain)",
            settings.abstain_px,
        )
        emit(
            "faces_skip",
            message=f"Face already ≥ {settings.abstain_px:.0f}px — left as it is",
        )
        if video_path.resolve() != output_path.resolve():
            shutil.copy2(video_path, output_path)
        return output_path

    curve = strengths(heights)
    confidence = paste_weights(found)
    spans = windows(n)
    blend = window_weights(spans)
    from h3_backend import apple_silicon_class

    chip = apple_silicon_class()
    ref_note = f", {len(refs_list)} ref(s)" if refs_list else ", no refs (t2va)"
    emit(
        "faces_plan",
        message=(
            f"{len(spans)} window(s), canvas {canvas}, denoise ≤ {ceiling:.2f}, "
            f"{face_mode}{ref_note}, chip={chip}"
        ),
        windows=len(spans),
        canvas=canvas,
        denoise=ceiling,
        mode=face_mode,
        refs=len(refs_list),
        face_px_min=float(min(heights)),
        face_px_max=float(max(heights)),
        chip_class=chip,
    )
    log.info(
        "Faces pass: chip_class=%s (init CFR + round-trip are chip-agnostic; "
        "DiT BF16/int8 still auto-selected in h3_backend)",
        chip,
    )

    result = frames.copy()
    work = Path(tempfile.mkdtemp(prefix="h3_faces_"))
    try:
        for wi, span in enumerate(spans):
            start, end = span
            length = end - start
            vt = video_latent_t(length)
            latent_curve = latent_strengths(curve[start:end], vt)
            strengths_path = write_strengths_file(
                work / f"strengths_{wi:02d}.txt", latent_curve
            )
            window_label = (
                f"Faces window {wi + 1}/{len(spans)} · ceiling {ceiling:.2f} · "
                f"video_t={vt} · {face_mode}"
            )
            emit(
                "faces_window",
                message=window_label,
                index=wi,
                total=len(spans),
                start=start,
                end=end,
                strength=ceiling,
                video_t=vt,
                strengths=latent_curve,
            )
            crop_rgb = warp_crop_window(frames, crops, span, canvas, canvas)
            silent_init = work / f"init_silent_{wi:02d}.mp4"
            init_mp4 = work / f"init_{wi:02d}.mp4"
            refined_mp4 = work / f"refined_{wi:02d}.mp4"
            write_rgb_mp4(crop_rgb, silent_init)
            mux_window_audio(
                silent_init,
                video_path,
                init_mp4,
                start_frame=start,
                end_frame=end,
                fps=FPS,
            )
            # Gate: what h3's fps=24 resample will see must match crop indices.
            ok_mux, mae_mux = verify_init_roundtrip(init_mp4, crop_rgb, fps=FPS)
            init_for_h3 = init_mp4
            roundtrip_ok = ok_mux
            if not ok_mux:
                ok_silent, mae_silent = verify_init_roundtrip(
                    silent_init, crop_rgb, fps=FPS
                )
                if ok_silent:
                    log.warning(
                        "Faces window %d/%d: muxed init round-trip mae=%.1f — "
                        "falling back to silent CFR crop (mae=%.1f)",
                        wi + 1,
                        len(spans),
                        mae_mux,
                        mae_silent,
                    )
                    init_for_h3 = silent_init
                    roundtrip_ok = True
                else:
                    log.warning(
                        "Faces window %d/%d: init round-trip weak "
                        "(mux mae=%.1f, silent mae=%.1f) — continuing with "
                        "silent crop; temporal lock will try to recover",
                        wi + 1,
                        len(spans),
                        mae_mux,
                        mae_silent,
                    )
                    init_for_h3 = silent_init
                    roundtrip_ok = False
            else:
                log.info(
                    "Faces window %d/%d: init round-trip ok (mae=%.1f)",
                    wi + 1,
                    len(spans),
                    mae_mux,
                )

            req = build_faces_window_request(
                prompt=prompt,
                output_path=refined_mp4,
                init_video=init_for_h3,
                strengths_file=strengths_path,
                canvas=canvas,
                num_frames=length,
                ceiling=ceiling,
                seed=seed,
                face_mode=face_mode,
                refs_list=refs_list,
                loras_list=loras_list,
                quality=quality,
                steps=steps,
                layers=layers,
                reuse=reuse,
                preview_latent_dir=preview_latent_dir,
            )

            def _window_progress(
                mp: dict[str, Any],
                *,
                _label: str = window_label,
                _wi: int = wi,
                _n: int = len(spans),
                _start: int = start,
                _end: int = end,
            ) -> None:
                emit(
                    "faces_window",
                    message=_label,
                    index=_wi,
                    total=_n,
                    start=_start,
                    end=_end,
                    strength=ceiling,
                    model_progress=mp,
                )

            engine.generate(req, on_progress=_window_progress)

            refined, _, _ = decode_clip_rgb(refined_mp4)
            refined, shift, lock_mae = temporal_lock_refined(
                refined,
                crop_rgb,
                face_rects[start:end],
                # Shift search only when init timeline looked wrong — soft-hold
                # changes appearance enough to fake a δ on healthy windows.
                enabled=not roundtrip_ok,
            )
            if shift != 0:
                emit(
                    "faces_window",
                    message=f"{window_label} · temporal lock δ={shift:+d}",
                    index=wi,
                    total=len(spans),
                    start=start,
                    end=end,
                    strength=ceiling,
                    temporal_shift=shift,
                    temporal_mae=lock_mae,
                )

            piece = result[start:end]
            weights = [confidence[i] * blend[wi][i - start] for i in range(start, end)]
            result[start:end] = composite_window(
                piece, refined, crops[start:end], face_rects[start:end], weights
            )

        emit("faces_composite", message="Compositing and remuxing audio…")
        silent = work / "faces_silent.mp4"
        write_rgb_mp4(result, silent)
        remux_audio(silent, video_path, output_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    emit("faces_done", message="Faces pass complete", path=str(output_path))
    return output_path


def overlay_png(
    frame_rgb: np.ndarray,
    boxes: Sequence[Box],
    *,
    label: str | None = None,
) -> bytes:
    """Draw boxes on one frame and return PNG bytes."""
    import cv2

    img = frame_rgb.copy()
    for box in boxes:
        x, y, w, h = box
        if w <= 0 or h <= 0:
            continue
        p1 = (int(round(x)), int(round(y)))
        p2 = (int(round(x + w)), int(round(y + h)))
        cv2.rectangle(img, p1, p2, (80, 220, 120), 2)
        text = label or f"{h:.0f}px"
        cv2.putText(
            img,
            text,
            (p1[0], max(16, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 220, 120),
            1,
            cv2.LINE_AA,
        )
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encode failed")
    return bytes(buf)
