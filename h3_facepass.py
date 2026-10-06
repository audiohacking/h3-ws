"""H3 Faces post-pass: detect → crop → init-video re-denoise → composite.

Orchestrates Continuity's face repair on top of h3-ws: geometry from
``h3_faces``, detection from ``h3_sam``, engine repair via
``--init-video`` / ``--denoise-strength``, then feathered paste + remux.
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

from h3_faces import (
    ABSTAIN_FACE_PX,
    DEFAULT_FACE_CANVAS,
    DEFAULT_FACE_DENOISE,
    FaceError,
    PASTE_DILATION,
    Box,
    clamp_face_denoise,
    snap_face_canvas,
    crop_boxes,
    face_heights,
    feather_in_canvas,
    paste_weights,
    should_abstain,
    strengths,
    window_denoise,
    window_weights,
    windows,
)
from h3_sam import sam3_ready, track_faces_in_clip

log = logging.getLogger("h3.facepass")

ProgressFn = Callable[[str, dict[str, Any]], None]


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
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        frames: list[np.ndarray] = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
        if not frames:
            raise FaceError(f"no video frames in {path}")
        arr = np.stack(frames, axis=0)
        h, w = int(arr.shape[1]), int(arr.shape[2])
        return arr, w, h
    finally:
        container.close()


def warp_crop_window(
    frames: np.ndarray,
    crops: Sequence[Box],
    span: tuple[int, int],
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    """Bilinear crop+resize for one window → ``[N,H,W,3]`` uint8."""
    import cv2

    start, end = span
    out = np.zeros((end - start, canvas_h, canvas_w, 3), dtype=np.uint8)
    src_h, src_w = int(frames.shape[1]), int(frames.shape[2])
    for i, index in enumerate(range(start, end)):
        x, y, bw, bh = crops[index]
        # Source quad in pixel space → destination rectangle.
        src = np.float32(
            [
                [x, y],
                [x + bw, y],
                [x + bw, y + bh],
                [x, y + bh],
            ]
        )
        dst = np.float32(
            [
                [0, 0],
                [canvas_w - 1, 0],
                [canvas_w - 1, canvas_h - 1],
                [0, canvas_h - 1],
            ]
        )
        # Clamp source points into a border-extended sample by using
        # BORDER_REPLICATE via warpPerspective on a padded view when needed.
        M = cv2.getPerspectiveTransform(src, dst)
        out[i] = cv2.warpPerspective(
            frames[index],
            M,
            (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        # Silence unused when coords are fine; keep src dims referenced.
        _ = (src_h, src_w)
    return out


def write_rgb_mp4(frames: np.ndarray, dest: str | Path, fps: int = 24) -> Path:
    """Write ``[T,H,W,3]`` uint8 RGB to an H.264 MP4 (silent)."""
    import av

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    h, w = int(frames.shape[1]), int(frames.shape[2])
    out = av.open(str(dest), mode="w")
    try:
        stream = out.add_stream("libx264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        for frame in frames:
            vf = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(vf):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    finally:
        out.close()
    return dest


def remux_audio(video_no_audio: str | Path, audio_source: str | Path, dest: str | Path) -> Path:
    """Copy video from ``video_no_audio`` and audio stream(s) from ``audio_source``.

    When the source has no audio (or remux fails), fall back to copying the
    silent refined video so the Faces pass still lands a playable file.
    """
    import av

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    video_no_audio = Path(video_no_audio)
    audio_source = Path(audio_source)

    ain = av.open(str(audio_source))
    try:
        has_audio = any(s.type == "audio" for s in ain.streams)
    finally:
        ain.close()
    if not has_audio:
        shutil.copy2(video_no_audio, dest)
        return dest

    try:
        vin = av.open(str(video_no_audio))
        ain = av.open(str(audio_source))
        try:
            out = av.open(str(dest), mode="w")
            try:
                v_in = vin.streams.video[0]
                a_in = next(s for s in ain.streams if s.type == "audio")
                # Prefer stream copy; fall back to re-encode if template API differs.
                try:
                    v_out = out.add_stream(template=v_in)
                    a_out = out.add_stream(template=a_in)
                    copy_packets = True
                except TypeError:
                    rate = v_in.average_rate or 24
                    v_out = out.add_stream("libx264", rate=rate)
                    v_out.width = v_in.width
                    v_out.height = v_in.height
                    v_out.pix_fmt = "yuv420p"
                    a_out = out.add_stream("aac")
                    copy_packets = False
                if copy_packets:
                    for packet in vin.demux(v_in):
                        if packet.dts is None:
                            continue
                        packet.stream = v_out
                        out.mux(packet)
                    for packet in ain.demux(a_in):
                        if packet.dts is None:
                            continue
                        packet.stream = a_out
                        out.mux(packet)
                else:
                    for frame in vin.decode(v_in):
                        for packet in v_out.encode(frame):
                            out.mux(packet)
                    for packet in v_out.encode(None):
                        out.mux(packet)
                    for frame in ain.decode(a_in):
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
        log.warning("audio remux failed (%s); keeping silent refined video", exc)
        shutil.copy2(video_no_audio, dest)
    return dest


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
    """Paste refined crops into base frames (uint8 RGB arrays)."""
    import cv2

    count = base.shape[0]
    src_h, src_w = int(base.shape[1]), int(base.shape[2])
    canvas_h, canvas_w = int(refined.shape[1]), int(refined.shape[2])
    out = base.astype(np.float32) / 255.0
    patch = refined.astype(np.float32) / 255.0

    for i in range(count):
        w = float(weights[i])
        if w <= 0:
            continue
        x, y, bw, bh = crops[i]
        feather = feather_in_canvas(bh, canvas_h)
        mask_c = _paste_mask(face_rects[i], canvas_w, canvas_h, feather)

        # Warp refined crop (+ mask) back into source frame coordinates.
        src = np.float32(
            [
                [0, 0],
                [canvas_w - 1, 0],
                [canvas_w - 1, canvas_h - 1],
                [0, canvas_h - 1],
            ]
        )
        dst = np.float32(
            [
                [x, y],
                [x + bw, y],
                [x + bw, y + bh],
                [x, y + bh],
            ]
        )
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            patch[i], M, (src_w, src_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        mask = cv2.warpPerspective(
            mask_c, M, (src_w, src_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        mask = np.clip(mask, 0.0, 1.0)[..., None]

        # Colour-match inside the mask.
        total = float(mask.sum()) + 1e-6
        base_i = out[i]
        base_mean = (base_i * mask).sum(axis=(0, 1), keepdims=True) / total
        patch_mean = (warped * mask).sum(axis=(0, 1), keepdims=True) / total
        base_sd = np.sqrt((((base_i - base_mean) ** 2) * mask).sum(axis=(0, 1), keepdims=True) / total)
        patch_sd = np.sqrt((((warped - patch_mean) ** 2) * mask).sum(axis=(0, 1), keepdims=True) / total)
        base_sd = np.maximum(base_sd, 1e-6)
        patch_sd = np.maximum(patch_sd, 1e-6)
        matched = np.clip((warped - patch_mean) * (base_sd / patch_sd) + base_mean, 0.0, 1.0)

        opacity = mask * w
        out[i] = np.clip((1.0 - opacity) * base_i + opacity * matched, 0.0, 1.0)

    return (out * 255.0).astype(np.uint8)


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
    # Inspect may Haar-preview when SAM is missing; Faces repair never does.
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
    quality: str = "fast",
    steps: int | None = None,
    layers: int | None = None,
    reuse: int | None = None,
    preview_latent_dir: Path | None = None,
) -> Path:
    """Full Faces repair. Requires h3 ``--init-video`` support in the binary.

    When ``refs`` are provided (the generation's Ref2VA list), each crop window
    is re-drawn in Ref2VA mode so identity matches the same Picture N / Video N
    / Audio N conditioning Continuity uses for the face pass.
    """
    settings = settings or FacesSettings()
    canvas = snap_face_canvas(settings.canvas)
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
    n = int(frames.shape[0])
    # Continuity: SAM is required — no Haar substitute for the repair pass.
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
        # Continuity: leave the pass as it is (log, no invented repair).
        log.info("Faces: no face found — left as it is")
        emit(
            "faces_skip",
            message="No face found (SAM 3.1) — left as it is",
        )
        if Path(video_path).resolve() != Path(output_path).resolve():
            shutil.copy2(video_path, output_path)
        return output_path

    crops, face_rects = crop_boxes(boxes, found, canvas, canvas)
    heights = face_heights(crops)
    if settings.abstain and should_abstain(heights, settings.abstain_px):
        # Optional product gate — Continuity never abstains; strengths already
        # soften denoise on large faces. Default abstain is off.
        log.info(
            "Faces: face already ≥ %.0f px — left as it is (abstain)",
            settings.abstain_px,
        )
        emit(
            "faces_skip",
            message=(
                f"Face already ≥ {settings.abstain_px:.0f}px — left as it is"
            ),
        )
        if Path(video_path).resolve() != Path(output_path).resolve():
            shutil.copy2(video_path, output_path)
        return output_path

    curve = strengths(heights)
    confidence = paste_weights(found)
    spans = windows(n)
    blend = window_weights(spans)
    ref_note = f", {len(refs_list)} ref(s)" if refs_list else ", no refs (t2va)"
    emit(
        "faces_plan",
        message=(
            f"{len(spans)} window(s), canvas {canvas}, denoise ≤ {ceiling:.2f}, "
            f"{face_mode}{ref_note}"
        ),
        windows=len(spans),
        canvas=canvas,
        denoise=ceiling,
        mode=face_mode,
        refs=len(refs_list),
    )

    result = frames.copy()
    work = Path(tempfile.mkdtemp(prefix="h3_faces_"))
    try:
        for wi, span in enumerate(spans):
            start, end = span
            strength = window_denoise(ceiling, curve[start:end])
            window_label = (
                f"Faces window {wi + 1}/{len(spans)} · strength {strength:.2f} · {face_mode}"
            )
            emit(
                "faces_window",
                message=window_label,
                index=wi,
                total=len(spans),
                start=start,
                end=end,
                strength=strength,
            )
            crop_rgb = warp_crop_window(frames, crops, span, canvas, canvas)
            init_mp4 = work / f"init_{wi:02d}.mp4"
            refined_mp4 = work / f"refined_{wi:02d}.mp4"
            write_rgb_mp4(crop_rgb, init_mp4)

            from h3_backend import GenerateRequest

            req = GenerateRequest(
                prompt=prompt or "a person speaking, detailed face",
                output_path=refined_mp4,
                width=canvas,
                height=canvas,
                num_frames=end - start,
                quality=str(quality or "fast"),
                steps=steps,
                layers=layers,
                reuse=reuse,
                seed=seed,
                mode=face_mode,
                refs=refs_list,
                loras=loras_list,
                init_video=init_mp4,
                denoise_strength=strength,
                profile=False,
                ssd_streaming=False,
                preview_latent_dir=preview_latent_dir,
            )

            def _window_progress(
                mp: dict[str, Any],
                *,
                _label: str = window_label,
                _wi: int = wi,
                _n: int = len(spans),
                _strength: float = strength,
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
                    strength=_strength,
                    model_progress=mp,
                )

            engine.generate(req, on_progress=_window_progress)

            refined, _, _ = decode_clip_rgb(refined_mp4)
            if refined.shape[0] != end - start:
                # Trim or pad to window length (should already match).
                if refined.shape[0] > end - start:
                    refined = refined[: end - start]
                else:
                    pad = np.repeat(refined[-1:], end - start - refined.shape[0], axis=0)
                    refined = np.concatenate([refined, pad], axis=0)

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
