"""SAM 3.1 face detector for the H3 Faces pass (Apple Silicon).

Preferred backend: MLX SAM 3.1 weights under ``models/sam3.1/`` (HF download via
Models). Until those weights + an MLX runtime are available, falls back to
OpenCV Haar so Library "Inspect faces" and geometry plumbing stay testable.

Continuity uses Comfy's SAM3 with prompt ``"face"`` and threshold 0.35; we keep
the same prompt/threshold contract.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

from h3_faces import Box, DETECT_INTERVAL, detect_frames, pick

log = logging.getLogger("h3.sam")

DETECT_PROMPT = "face"
DETECT_THRESHOLD = 0.35
DETECT_EDGE = 1008

# Public MLX BF16 pack (SAM License). Continuity recommends sam3.1 multiplex;
# on Mac we prefer the mlx-community conversion.
SAM31_REPO = "mlx-community/sam3.1-bf16"
SAM31_DIRNAME = "sam3.1"


def sam3_root(model_dir: Path | None = None) -> Path:
    if model_dir is None:
        model_dir = Path(os.environ.get("H3_MODEL_DIR", "models/MiniMax-H3"))
    # Sibling of MiniMax-H3: models/sam3.1
    root = model_dir.parent if model_dir.name.upper().startswith("MINIMAX") else model_dir
    if (root / SAM31_DIRNAME).is_dir() or model_dir.name == "MiniMax-H3":
        return (model_dir.parent / SAM31_DIRNAME).resolve()
    return (Path("models") / SAM31_DIRNAME).resolve()


def sam3_ready(model_dir: Path | None = None) -> bool:
    root = sam3_root(model_dir)
    if not root.is_dir():
        return False
    # Any weight-ish file counts as "on disk".
    for pattern in ("*.safetensors", "*.npz", "config.json"):
        if any(root.rglob(pattern)):
            return True
    return False


def sam3_status(model_dir: Path | None = None) -> dict[str, Any]:
    root = sam3_root(model_dir)
    ready = sam3_ready(model_dir)
    size = 0
    if root.is_dir():
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    size += p.stat().st_size
                except OSError:
                    pass
    return {
        "id": "sam3",
        "label": "SAM 3.1 face detector",
        "present": ready,
        "path": str(root),
        "size_gib": round(size / (1024 ** 3), 2),
        "note": (
            "Open-vocabulary face detector for the Faces pass (~1–2 GB MLX BF16). "
            "Accept the SAM License on Hugging Face before download. "
            "Until present, Inspect faces can use a Haar fallback for geometry tests."
        ),
        "essential": False,
        "repo": SAM31_REPO,
        "backend": _active_backend_name(model_dir),
    }


def download_sam3(model_dir: Path | None = None) -> Path:
    """Snapshot-download SAM 3.1 MLX weights into models/sam3.1/."""
    from huggingface_hub import snapshot_download

    dest = sam3_root(model_dir)
    dest.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s → %s", SAM31_REPO, dest)
    snapshot_download(repo_id=SAM31_REPO, local_dir=str(dest))
    return dest


def _active_backend_name(model_dir: Path | None = None) -> str:
    if sam3_ready(model_dir) and _try_import_mlx_sam():
        return "sam3.1-mlx"
    return "haar-fallback"


def _try_import_mlx_sam() -> bool:
    try:
        import mlx  # noqa: F401
    except Exception:
        return False
    # Prefer mlx_cv; mlx_vlm SAM path is optional and currently fragile in some envs.
    try:
        import mlx_cv  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import mlx_vlm  # noqa: F401
        return True
    except Exception:
        return False


def _detect_haar(rgb) -> list[Box]:
    import cv2
    import numpy as np

    if hasattr(rgb, "shape") and len(rgb.shape) == 3 and rgb.shape[2] == 3:
        arr = np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.5 else arr.astype(np.uint8)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("expected HxWx3 RGB frame")
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    hits = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    return [(float(x), float(y), float(w), float(h)) for (x, y, w, h) in hits]


def _detect_mlx(rgb, threshold: float = DETECT_THRESHOLD) -> list[Box]:
    """Best-effort SAM 3.1 MLX detect. Raises if runtime is unavailable."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.5 else arr.astype(np.uint8)
    image = Image.fromarray(arr, mode="RGB")

    # Try mlx_cv first (App Automaton / mlx-cv SAM3Processor).
    try:
        from mlx_cv.models.sam3 import SAM3Processor  # type: ignore

        model = SAM3Processor.from_pretrained("sam3.1")
        prediction = model.predict(image, DETECT_PROMPT)
        boxes: list[Box] = []
        raw_boxes = getattr(prediction, "boxes", None) or prediction.get("boxes")  # type: ignore[union-attr]
        scores = getattr(prediction, "scores", None) or prediction.get("scores")  # type: ignore[union-attr]
        if raw_boxes is None:
            return []
        for i, box in enumerate(raw_boxes):
            score = float(scores[i]) if scores is not None else 1.0
            if score < threshold:
                continue
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes
    except Exception as exc:
        log.debug("mlx_cv SAM3 unavailable: %s", exc)

    raise RuntimeError(
        "SAM 3.1 MLX runtime is not available. Install mlx-cv (or a working "
        "mlx_vlm SAM build) and download the sam3 component from Models."
    )


def detect_faces_in_frame(
    rgb,
    *,
    model_dir: Path | None = None,
    threshold: float = DETECT_THRESHOLD,
    allow_fallback: bool = True,
) -> list[Box]:
    """Return ``(x, y, w, h)`` face boxes for one RGB frame."""
    if sam3_ready(model_dir) and _try_import_mlx_sam():
        try:
            return _detect_mlx(rgb, threshold=threshold)
        except Exception as exc:
            log.warning("SAM 3.1 detect failed (%s); fallback=%s", exc, allow_fallback)
            if not allow_fallback:
                raise
    if not allow_fallback:
        raise RuntimeError("SAM 3.1 weights/runtime required (fallback disabled)")
    return _detect_haar(rgb)


def track_faces_in_clip(
    frames: Sequence[Any],
    *,
    model_dir: Path | None = None,
    interval: int = DETECT_INTERVAL,
    threshold: float = DETECT_THRESHOLD,
    allow_fallback: bool = True,
    max_faces: int = 1,
    prefer_video_track: bool = True,
) -> tuple[list[Box], list[bool]]:
    """Detect on ``detect_frames`` indices and track with Continuity ``pick``.

    Returns per-frame boxes (placeholders when undetected) and a found mask.

    When SAM 3.1 MLX video multiplex is available, ``prefer_video_track`` uses
    that path; otherwise falls back to every-``interval`` image detect (Phase 0).
    ``max_faces`` > 1 keeps the largest N detections per sample (multi-face).
    """
    count = len(frames)
    if count <= 0:
        raise ValueError("no frames")
    if prefer_video_track and sam3_ready(model_dir) and _try_import_mlx_sam():
        try:
            return _track_mlx_video(
                frames,
                model_dir=model_dir,
                threshold=threshold,
                max_faces=max_faces,
            )
        except Exception as exc:
            log.debug("SAM 3.1 video track unavailable (%s); image interval", exc)

    indices = detect_frames(count, interval=interval)
    boxes: list[Box] = [(0.0, 0.0, 0.0, 0.0)] * count
    found = [False] * count
    previous: Box | None = None
    for index in indices:
        candidates = detect_faces_in_frame(
            frames[index],
            model_dir=model_dir,
            threshold=threshold,
            allow_fallback=allow_fallback,
        )
        if max_faces > 1 and candidates:
            candidates = sorted(candidates, key=lambda b: b[2] * b[3], reverse=True)[
                : max_faces
            ]
            # Multi-face v1: still track the largest only for geometry continuity.
            candidates = candidates[:1]
        chosen = pick(candidates, previous)
        if chosen is not None:
            boxes[index] = chosen
            found[index] = True
            previous = chosen
    return boxes, found


def _track_mlx_video(
    frames: Sequence[Any],
    *,
    model_dir: Path | None,
    threshold: float,
    max_faces: int,
) -> tuple[list[Box], list[bool]]:
    """Best-effort SAM 3.1 multiplex video track. Raises if unsupported."""
    # mlx_cv / mlx_vlm video multiplex APIs are still settling; keep the hook
    # and fall back by raising so callers use the image interval path.
    raise RuntimeError("SAM 3.1 MLX video multiplex not wired in this build")
