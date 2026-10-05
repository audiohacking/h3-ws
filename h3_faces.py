"""Face-pass geometry — Continuity `faces.py` ported for h3-ws.

Pure arithmetic: no torch, no numpy, no I/O. Plans where to crop, how hard to
re-denoise, and how to tile a pass into legal H3 windows (``5 + 17n``).

See Continuity-Mac ``creator/families/h3/faces.py`` for the design rationale.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

# Crop is this many face-heights tall (head + shoulders + context).
CROP_FACTOR = 3.0

POSITION_WINDOW = 21
SIZE_WINDOW = 51

# Trained length ceiling — one face window never asks for longer than this.
WINDOW_CAP = 362
WINDOW_OVERLAP = 17

STRENGTH_SMALL = 0.8
STRENGTH_LARGE = 0.35
FACE_PX_SMALL = 30.0
FACE_PX_LARGE = 120.0
STRENGTH_SMOOTH = 9

DETECT_INTERVAL = 4
BLIND_HOLD = DETECT_INTERVAL * 2
BLIND_FADE = DETECT_INTERVAL * 2

PASTE_DILATION = 24
PASTE_FEATHER = 24

# UI defaults (Continuity compile defaults).
DEFAULT_FACE_CANVAS = 512
MIN_FACE_CANVAS = 384
MAX_FACE_CANVAS = 768
DEFAULT_FACE_DENOISE = 0.45
MIN_FACE_DENOISE = 0.1
MAX_FACE_DENOISE = 0.9

# Skip the repair when every tracked face is already this tall (source px).
ABSTAIN_FACE_PX = FACE_PX_LARGE


class FaceError(ValueError):
    """A face pass that cannot be planned from what the pass actually holds."""


Box = tuple[float, float, float, float]  # x, y, w, h


def legal_frame_counts(cap: int = WINDOW_CAP) -> list[int]:
    """Legal H3 temporal lengths ``5 + 17n`` for ``n ≥ 0``, up to *cap*.

    Continuity's face tiler includes length 5; product generation snaps to ≥22,
    but window tiling of trimmed/merged passes still needs the full grid.
    """
    out: list[int] = []
    n = 0
    while True:
        frames = 5 + 17 * n
        if frames > int(cap):
            break
        out.append(frames)
        n += 1
    return out


def snap_face_canvas(edge: int) -> int:
    """Square canvas snapped to 32 px, clamped to Continuity's 384–768 range."""
    edge = int(edge)
    edge = max(MIN_FACE_CANVAS, min(MAX_FACE_CANVAS, edge))
    return max(MIN_FACE_CANVAS, (edge // 32) * 32)


def clamp_face_denoise(denoise: float) -> float:
    return max(MIN_FACE_DENOISE, min(MAX_FACE_DENOISE, float(denoise)))


def windows(
    frames: int,
    cap: int = WINDOW_CAP,
    overlap: int = WINDOW_OVERLAP,
) -> list[tuple[int, int]]:
    """``frames`` → legal-length half-open ``(start, end)`` windows.

    Never pads: the last window ends on the pass's true last frame.
    """
    frames = int(frames)
    overlap = int(overlap)
    usable = [n for n in legal_frame_counts(cap) if n <= min(frames, int(cap))]
    if not usable:
        raise FaceError(
            f"this pass is {frames} frames and the shortest generation H3 "
            f"accepts is {legal_frame_counts()[0]} — there is nothing "
            f"here to refine"
        )
    if usable[-1] == frames:
        return [(0, frames)]

    best: Optional[tuple[tuple[int, int], int, int]] = None
    for size in usable:
        stride = size - overlap
        if stride <= 0:
            continue
        count = max(1, -(-(frames - overlap) // stride))
        if count * size - (count - 1) * overlap < frames:
            continue
        cost = (count * size, count)
        if best is None or cost < best[0]:
            best = (cost, count, size)
    if best is None:
        raise FaceError(f"a {frames}-frame pass cannot be tiled with legal windows")

    _, count, size = best
    if count == 1:
        return [(0, size)]
    return [
        (
            round(index * (frames - size) / (count - 1)),
            round(index * (frames - size) / (count - 1)) + size,
        )
        for index in range(count)
    ]


def detect_frames(frames: int, interval: int = DETECT_INTERVAL) -> list[int]:
    """Which frames the detector runs on: every *interval*, and always the last."""
    frames = int(frames)
    if frames <= 0:
        raise FaceError("a face pass over no frames")
    interval = max(1, int(interval))
    sampled = list(range(0, frames, interval))
    if sampled[-1] != frames - 1:
        sampled.append(frames - 1)
    return sampled


def _iou(a: Box, b: Box) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(ax2, bx2), min(ay2, by2)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    union = a[2] * a[3] + b[2] * b[3] - overlap
    return overlap / union if union > 0 else 0.0


def pick(candidates: Sequence[Box], previous: Optional[Box]) -> Optional[Box]:
    """One face out of a frame's detections — stay on the previous track by IoU."""
    if not candidates:
        return None
    if previous is not None:
        best = max(candidates, key=lambda box: _iou(box, previous))
        if _iou(best, previous) > 0:
            return best
    return max(candidates, key=lambda box: box[2] * box[3])


def paste_weights(
    found: Sequence[bool],
    hold: int = BLIND_HOLD,
    fade: int = BLIND_FADE,
) -> list[float]:
    """Per-frame paste opacity from which frames detected a face."""
    seen = [index for index, ok in enumerate(found) if ok]
    if not seen:
        raise FaceError("no face was found anywhere in this pass")
    out: list[float] = []
    for index in range(len(found)):
        distance = min(abs(index - i) for i in seen)
        if distance <= hold:
            out.append(1.0)
        elif fade <= 0:
            out.append(0.0)
        else:
            out.append(max(0.0, 1.0 - (distance - hold) / float(fade)))
    return out


def smooth(values: Iterable[float], window: int) -> list[float]:
    """Gaussian low-pass with reflected edges."""
    values = [float(v) for v in values]
    window = int(window)
    if window <= 1 or len(values) < 3:
        return values
    window = min(window, len(values))
    if window % 2 == 0:
        window += 1
    if window < 3:
        return values

    half = window // 2
    sigma = max(window / 6.0, 0.5)
    kernel = [
        math.exp(-((i - half) ** 2) / (2.0 * sigma * sigma)) for i in range(window)
    ]
    total = sum(kernel)
    kernel = [k / total for k in kernel]

    padded = (
        [values[min(half - i, len(values) - 1)] for i in range(half)]
        + values
        + [values[max(len(values) - 2 - i, 0)] for i in range(half)]
    )
    return [
        sum(padded[i + k] * kernel[k] for k in range(window))
        for i in range(len(values))
    ]


def _fill_gaps(values: Sequence[float], found: Sequence[bool]) -> list[float]:
    seen = [i for i, ok in enumerate(found) if ok]
    if not seen:
        raise FaceError("no face was found anywhere in this pass")
    out: list[float] = []
    for index in range(len(values)):
        if found[index]:
            out.append(float(values[index]))
            continue
        before = [i for i in seen if i <= index]
        after = [i for i in seen if i >= index]
        if not before:
            out.append(float(values[after[0]]))
        elif not after:
            out.append(float(values[before[-1]]))
        else:
            low, high = before[-1], after[0]
            span = high - low
            weight = (index - low) / span if span else 0.0
            out.append(float(values[low]) * (1 - weight) + float(values[high]) * weight)
    return out


def crop_boxes(
    boxes: Sequence[Box],
    found: Sequence[bool],
    canvas_width: int,
    canvas_height: int,
    crop_factor: float = CROP_FACTOR,
) -> tuple[list[Box], list[Box]]:
    """Per-frame face boxes → smoothed crop boxes + face rects in canvas space."""
    if len(boxes) != len(found):
        raise FaceError(f"{len(boxes)} boxes for {len(found)} frames")
    if not boxes:
        raise FaceError("a face pass over no frames")

    centres_x = _fill_gaps([b[0] + b[2] / 2.0 for b in boxes], found)
    centres_y = _fill_gaps([b[1] + b[3] / 2.0 for b in boxes], found)
    heights = _fill_gaps([b[3] for b in boxes], found)
    widths = _fill_gaps([b[2] for b in boxes], found)

    centres_x = smooth(centres_x, POSITION_WINDOW)
    centres_y = smooth(centres_y, POSITION_WINDOW)
    heights = smooth(heights, SIZE_WINDOW)
    widths = smooth(widths, SIZE_WINDOW)

    aspect = float(canvas_width) / float(canvas_height)
    crops: list[Box] = []
    faces: list[Box] = []
    for cx, cy, fw, fh in zip(centres_x, centres_y, widths, heights):
        crop_h = max(float(fh) * float(crop_factor), 8.0)
        crop_w = crop_h * aspect
        crop_x = cx - crop_w / 2.0
        crop_y = cy - crop_h / 2.0
        crops.append((crop_x, crop_y, crop_w, crop_h))

        scale_x = float(canvas_width) / crop_w
        scale_y = float(canvas_height) / crop_h
        faces.append(
            (
                (cx - fw / 2.0 - crop_x) * scale_x,
                (cy - fh / 2.0 - crop_y) * scale_y,
                float(fw) * scale_x,
                float(fh) * scale_y,
            )
        )
    return crops, faces


def face_heights(crops: Sequence[Box], crop_factor: float = CROP_FACTOR) -> list[float]:
    return [float(crop[3]) / float(crop_factor) for crop in crops]


def strengths(
    heights: Sequence[float],
    small: float = STRENGTH_SMALL,
    large: float = STRENGTH_LARGE,
    px_small: float = FACE_PX_SMALL,
    px_large: float = FACE_PX_LARGE,
    smoothing: int = STRENGTH_SMOOTH,
) -> list[float]:
    """Per-frame denoise multipliers from source-pixel face height."""
    span = float(px_large) - float(px_small)
    out: list[float] = []
    for height in heights:
        if span <= 0:
            weight = 0.0
        else:
            weight = min(1.0, max(0.0, (float(height) - float(px_small)) / span))
        out.append(float(small) + (float(large) - float(small)) * weight)
    return [min(1.0, max(0.0, value)) for value in smooth(out, smoothing)]


def feather_in_canvas(
    crop_height: float,
    canvas_height: int,
    feather: int = PASTE_FEATHER,
) -> int:
    magnification = float(canvas_height) / max(float(crop_height), 1.0)
    return max(
        1,
        min(int(round(float(feather) * magnification)), int(canvas_height) // 3),
    )


def window_weights(spans: Sequence[tuple[int, int]]) -> list[list[float]]:
    """Per-window paste weights that cross-fade overlaps."""
    weights = [[1.0] * (end - start) for start, end in spans]
    for index in range(1, len(spans)):
        prev_start, prev_end = spans[index - 1]
        start, end = spans[index]
        shared = prev_end - start
        if shared <= 0:
            continue
        for offset in range(shared):
            ramp = (offset + 1) / (shared + 1)
            weights[index][offset] = ramp
            weights[index - 1][start - prev_start + offset] = 1.0 - ramp
    return weights


def should_abstain(heights: Sequence[float], threshold: float = ABSTAIN_FACE_PX) -> bool:
    """True when every tracked face is already large enough to leave alone."""
    if not heights:
        return True
    return min(float(h) for h in heights) >= float(threshold)


def window_denoise(
    ceiling: float,
    frame_strengths: Sequence[float],
    *,
    aggregate: str = "max",
) -> float:
    """Single denoise for a window until h3.c supports per-latent strength masks."""
    ceiling = clamp_face_denoise(ceiling)
    if not frame_strengths:
        return ceiling
    if aggregate == "mean":
        scale = sum(frame_strengths) / len(frame_strengths)
    else:
        scale = max(frame_strengths)
    return clamp_face_denoise(ceiling * float(scale))
