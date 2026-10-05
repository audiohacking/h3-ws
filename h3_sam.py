"""SAM 3.1 face detector for the H3 Faces pass (Apple Silicon).

Looks for weights already on disk (models/sam3.1, HF hub cache, Continuity/Comfy
multiplex) before any download. Haar fallback keeps Inspect usable without MLX.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional, Sequence

from h3_faces import Box, DETECT_INTERVAL, detect_frames, pick

log = logging.getLogger("h3.sam")

DETECT_PROMPT = "face"
DETECT_THRESHOLD = 0.35
DETECT_EDGE = 1008

SAM31_REPO = "mlx-community/sam3.1-bf16"
SAM31_DIRNAME = "sam3.1"
SAM31_LICENSE_URL = "https://huggingface.co/facebook/sam3"
_MIN_WEIGHT_BYTES = 100 * 1024 * 1024


class SamDownloadError(RuntimeError):
    """User-facing SAM download / license failure."""


def sam3_root(model_dir: Path | None = None) -> Path:
    if model_dir is None:
        model_dir = Path(os.environ.get("H3_MODEL_DIR", "models/MiniMax-H3"))
    if model_dir.name.upper().startswith("MINIMAX") or model_dir.name == "MiniMax-H3":
        return (model_dir.parent / SAM31_DIRNAME).resolve()
    return (Path("models") / SAM31_DIRNAME).resolve()


def _weight_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    hits: list[Path] = []
    for pattern in ("*.safetensors", "*.npz", "*.pt", "*.bin"):
        for path in root.rglob(pattern):
            if ".cache" in path.parts:
                continue
            try:
                if path.is_file() and path.stat().st_size >= _MIN_WEIGHT_BYTES:
                    hits.append(path)
            except OSError:
                continue
    return hits


def find_comfy_sam3_multiplex() -> Path | None:
    home = Path.home()
    for path in (
        home / "Documents" / "ComfyUI" / "models" / "checkpoints" / "sam3.1_multiplex.safetensors",
        home / "Documents" / "ComfyUI" / "models" / "checkpoints" / "sam3.1_multiplex_fp16.safetensors",
        home / "ComfyUI" / "models" / "checkpoints" / "sam3.1_multiplex.safetensors",
        home / "ComfyUI-Shared" / "models" / "checkpoints" / "sam3.1_multiplex.safetensors",
    ):
        try:
            if path.is_file() and path.stat().st_size >= _MIN_WEIGHT_BYTES:
                return path.resolve()
        except OSError:
            continue
    return None


def find_hf_cached_sam3() -> Path | None:
    from h3_paths import hf_hub_cache, hf_snapshot

    snap = hf_snapshot("models--" + SAM31_REPO.replace("/", "--"), hub=hf_hub_cache())
    if snap is None:
        return None
    return snap if _weight_files(snap) else None


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        return
    try:
        os.symlink(src, dest)
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dest)


def adopt_local_sam3(model_dir: Path | None = None) -> Path | None:
    """Point models/sam3.1 at an existing local weight (HF cache or Comfy)."""
    dest = sam3_root(model_dir)
    if _weight_files(dest):
        return dest

    snap = find_hf_cached_sam3()
    if snap is not None:
        dest.mkdir(parents=True, exist_ok=True)
        for src in snap.iterdir():
            if src.name.startswith("."):
                continue
            _link_or_copy(src, dest / src.name)
        if _weight_files(dest):
            return dest

    comfy = find_comfy_sam3_multiplex()
    if comfy is not None:
        _link_or_copy(comfy, dest / comfy.name)
        if _weight_files(dest):
            return dest
    return None


def sam3_ready(model_dir: Path | None = None) -> bool:
    if _weight_files(sam3_root(model_dir)):
        return True
    # Already on disk elsewhere — adopt once, then report ready.
    return adopt_local_sam3(model_dir) is not None


def download_sam3(model_dir: Path | None = None) -> Path:
    """Ensure weights under models/sam3.1/. Prefer local copies; HF only if needed."""
    dest = sam3_root(model_dir)
    adopted = adopt_local_sam3(model_dir)
    if adopted is not None:
        return adopted

    try:
        from huggingface_hub import get_token, snapshot_download
    except ImportError as exc:
        raise SamDownloadError("huggingface_hub is required to download SAM 3.1.") from exc

    token = get_token()
    if not token:
        raise SamDownloadError(
            f"HF login required for {SAM31_REPO}. "
            f"Run huggingface-cli login and accept the license at {SAM31_LICENSE_URL}."
        )

    dest.mkdir(parents=True, exist_ok=True)
    try:
        from h3_ssl import ensure_ssl_certs

        ensure_ssl_certs()
        snapshot_download(
            repo_id=SAM31_REPO,
            local_dir=str(dest),
            token=token,
        )
    except Exception as exc:
        raise SamDownloadError(
            f"SAM download failed: {exc}. "
            f"Accept the license at {SAM31_LICENSE_URL} and retry."
        ) from exc

    if not _weight_files(dest):
        raise SamDownloadError(f"Download finished but no weights under {dest}.")
    return dest


def sam3_status(model_dir: Path | None = None) -> dict[str, Any]:
    root = sam3_root(model_dir)
    ready = sam3_ready(model_dir)
    size = 0
    if root.is_dir():
        for p in root.rglob("*"):
            if p.is_file() and ".cache" not in p.parts:
                try:
                    size += p.stat().st_size
                except OSError:
                    pass
    if size < _MIN_WEIGHT_BYTES:
        comfy = find_comfy_sam3_multiplex()
        if comfy is not None:
            try:
                size = comfy.stat().st_size
            except OSError:
                pass

    return {
        "id": "sam3",
        "label": "SAM 3.1",
        "present": ready,
        "path": str(root if _weight_files(root) else (find_comfy_sam3_multiplex() or root)),
        "size_gib": round(size / (1024 ** 3), 2),
        "note": "Face detector for the Faces pass.",
        "essential": False,
        "repo": SAM31_REPO,
        "backend": _active_backend_name(model_dir),
    }


def _active_backend_name(model_dir: Path | None = None) -> str:
    if sam3_ready(model_dir) and _try_import_mlx_sam():
        return "sam3.1-mlx"
    if sam3_ready(model_dir):
        return "sam3-weights"
    return "haar-fallback"


def _try_import_mlx_sam() -> bool:
    try:
        import mlx  # noqa: F401
    except Exception:
        return False
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
    import numpy as np
    from PIL import Image

    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.5 else arr.astype(np.uint8)
    image = Image.fromarray(arr, mode="RGB")

    try:
        from mlx_cv.models.sam3 import SAM3Processor  # type: ignore

        local = sam3_root()
        model = (
            SAM3Processor.from_pretrained(str(local))
            if _weight_files(local)
            else SAM3Processor.from_pretrained("sam3.1")
        )
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

    raise RuntimeError("SAM 3.1 MLX runtime unavailable")


def detect_faces_in_frame(
    rgb,
    *,
    model_dir: Path | None = None,
    threshold: float = DETECT_THRESHOLD,
    allow_fallback: bool = True,
) -> list[Box]:
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
            log.debug("SAM video track unavailable (%s)", exc)

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
            candidates = sorted(candidates, key=lambda b: b[2] * b[3], reverse=True)[:1]
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
    raise RuntimeError("SAM 3.1 MLX video multiplex not wired in this build")
