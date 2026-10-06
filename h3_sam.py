"""SAM 3.1 face detector for the H3 Faces pass (Continuity parity).

Continuity's ``MiniMaxH3FacePass`` asks SAM3 for open-vocabulary ``"face"``
boxes (threshold 0.35, frames sampled every 4 + last), then ``faces.pick``
keeps one track. We mirror that with Apple Silicon ``mlx_vlm`` Sam3Predictor
against ``mlx-community/sam3.1-bf16``.

SAM is required to *run* Faces, but it is not a core Models essential — it is
fetched automatically the first time a Faces workflow needs it. Haar remains
Inspect-preview-only.

The ComfyUI ``sam3.1_multiplex.safetensors`` checkpoint is Continuity/Comfy's
torch format — not the MLX pack. Linking it alone must not mark SAM ready.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

from h3_faces import Box, DETECT_INTERVAL, detect_frames, pick

log = logging.getLogger("h3.sam")

# Continuity facepass.py constants — keep in lockstep.
DETECT_PROMPT = "face"
DETECT_THRESHOLD = 0.35
DETECT_EDGE = 1008

SAM31_REPO = "mlx-community/sam3.1-bf16"
SAM31_DIRNAME = "sam3.1"
SAM31_LICENSE_URL = "https://huggingface.co/facebook/sam3"
_MIN_WEIGHT_BYTES = 100 * 1024 * 1024
_MLX_WEIGHT_NAME = "model.safetensors"

_predictor_lock = threading.Lock()
_predictor: Any | None = None
_predictor_root: str | None = None


class SamDownloadError(RuntimeError):
    """User-facing SAM download / license failure."""


class SamDetectError(RuntimeError):
    """SAM weights or MLX runtime missing / failed — Faces cannot run."""


def sam3_root(model_dir: Path | None = None) -> Path:
    """Writable SAM 3.1 pack root (sibling of MiniMax-H3 under models/).

    Never resolves to a CWD-relative ``models/sam3.1`` — frozen apps run with
    an opaque working directory and must land weights under App Support.
    """
    if model_dir is None:
        from h3_paths import default_model_dir

        env = os.environ.get("H3_MODEL_DIR", "").strip()
        model_dir = Path(env).expanduser() if env else default_model_dir()
    else:
        model_dir = Path(model_dir).expanduser()
    # Standard layout: …/models/MiniMax-H3 → …/models/sam3.1
    if model_dir.name.upper().startswith("MINIMAX") or model_dir.name == "MiniMax-H3":
        return (model_dir.parent / SAM31_DIRNAME).resolve()
    if model_dir.name == "models":
        return (model_dir / SAM31_DIRNAME).resolve()
    # Custom model_dir: keep SAM next to it under a models/ sibling when possible.
    if model_dir.parent.name == "models":
        return (model_dir.parent / SAM31_DIRNAME).resolve()
    from h3_paths import writable_root

    return (writable_root() / "models" / SAM31_DIRNAME).resolve()


def mlx_weight_path(model_dir: Path | None = None) -> Path | None:
    """Path to the MLX ``model.safetensors`` if present and large enough."""
    root = sam3_root(model_dir)
    candidate = root / _MLX_WEIGHT_NAME
    try:
        if candidate.is_file() and candidate.stat().st_size >= _MIN_WEIGHT_BYTES:
            # Reject a mistaken Comfy multiplex symlink named model.safetensors.
            cfg = root / "config.json"
            if not cfg.is_file():
                return None
            return candidate.resolve()
    except OSError:
        return None
    return None


def _weight_files(root: Path) -> list[Path]:
    """Any large weight under root (status / size only — not readiness)."""
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
    """Comfy/Continuity torch multiplex — not usable as the MLX detector pack."""
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
    weight = snap / _MLX_WEIGHT_NAME
    cfg = snap / "config.json"
    try:
        if weight.is_file() and weight.stat().st_size >= _MIN_WEIGHT_BYTES and cfg.is_file():
            return snap
    except OSError:
        return None
    return None


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
    """Adopt an existing **MLX** SAM 3.1 pack into models/sam3.1/.

    Does not treat the Comfy multiplex checkpoint as ready — that is Continuity's
    torch format and cannot drive ``mlx_vlm`` Sam3Predictor.
    """
    dest = sam3_root(model_dir)
    if mlx_weight_path(model_dir) is not None:
        return dest

    snap = find_hf_cached_sam3()
    if snap is not None:
        dest.mkdir(parents=True, exist_ok=True)
        for src in snap.iterdir():
            if src.name.startswith("."):
                continue
            _link_or_copy(src, dest / src.name)
        if mlx_weight_path(model_dir) is not None:
            return dest
    return None


def _try_import_mlx_sam() -> bool:
    try:
        import mlx  # noqa: F401
        from mlx_vlm.models.sam3.generate import Sam3Predictor  # noqa: F401
        from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor  # noqa: F401
        from mlx_vlm.utils import load_model  # noqa: F401
    except Exception as exc:
        log.debug("mlx_vlm SAM import failed: %s", exc)
        return False
    return True


def sam3_ready(model_dir: Path | None = None) -> bool:
    """True only when MLX SAM 3.1 weights **and** runtime can run Faces detect."""
    if mlx_weight_path(model_dir) is None:
        adopt_local_sam3(model_dir)
    return mlx_weight_path(model_dir) is not None and _try_import_mlx_sam()


def download_sam3(model_dir: Path | None = None) -> Path:
    """Fetch ``mlx-community/sam3.1-bf16`` into models/sam3.1/ (MLX pack)."""
    dest = sam3_root(model_dir)
    if mlx_weight_path(model_dir) is not None:
        return dest

    adopted = adopt_local_sam3(model_dir)
    if adopted is not None and mlx_weight_path(model_dir) is not None:
        return adopted

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SamDownloadError("huggingface_hub is required to download SAM 3.1.") from exc

    dest.mkdir(parents=True, exist_ok=True)
    # Drop a mistaken Comfy multiplex symlink that blocks the real MLX file name.
    for stale in dest.glob("sam3.1_multiplex*.safetensors"):
        try:
            if stale.is_symlink() or stale.is_file():
                log.info("Removing non-MLX SAM weight from detector dir: %s", stale)
                stale.unlink()
        except OSError:
            pass

    try:
        from h3_ssl import ensure_ssl_certs

        ensure_ssl_certs()
        # mlx-community pack is typically public; token helps if the hub gates it.
        from huggingface_hub import get_token

        token = get_token()
        snapshot_download(
            repo_id=SAM31_REPO,
            local_dir=str(dest),
            token=token,
            allow_patterns=["config.json", "model.safetensors", "README.md", "*.json"],
        )
    except Exception as exc:
        raise SamDownloadError(
            f"SAM download failed: {exc}. "
            f"Need {SAM31_REPO} (MLX). Accept the SAM license at {SAM31_LICENSE_URL} "
            f"if the hub asks, then retry from Models."
        ) from exc

    if mlx_weight_path(model_dir) is None:
        raise SamDownloadError(
            f"Download finished but {dest / _MLX_WEIGHT_NAME} is missing. "
            f"Faces needs the MLX pack from {SAM31_REPO}, not the Comfy multiplex."
        )
    # Invalidate any cached predictor bound to a previous empty tree.
    global _predictor, _predictor_root
    with _predictor_lock:
        _predictor = None
        _predictor_root = None
    return dest


def sam3_status(model_dir: Path | None = None) -> dict[str, Any]:
    root = sam3_root(model_dir)
    ready = sam3_ready(model_dir)
    mlx = mlx_weight_path(model_dir)
    size = 0
    if mlx is not None:
        try:
            size = mlx.stat().st_size
        except OSError:
            size = 0
    elif root.is_dir():
        for p in root.rglob("*"):
            if p.is_file() and ".cache" not in p.parts:
                try:
                    size += p.stat().st_size
                except OSError:
                    pass

    note = (
        "Faces detector (open-vocab “face” boxes). Optional — auto-downloaded "
        f"when Faces runs. {SAM31_REPO} (~3.3 GB MLX)."
    )
    if not ready and find_comfy_sam3_multiplex() is not None:
        note += (
            " A ComfyUI multiplex checkpoint was found but cannot drive the "
            "Apple Silicon detector — download the MLX pack."
        )

    return {
        "id": "sam3",
        "label": "SAM 3.1 (Faces)",
        "present": ready,
        "path": str(mlx or root),
        "size_gib": round(size / (1024 ** 3), 2),
        "note": note,
        "essential": False,
        "repo": SAM31_REPO,
        "backend": _active_backend_name(model_dir),
        "runtime": _try_import_mlx_sam(),
    }


def _active_backend_name(model_dir: Path | None = None) -> str:
    if sam3_ready(model_dir):
        return "sam3.1-mlx"
    if mlx_weight_path(model_dir) is not None:
        return "sam3-weights-no-runtime"
    if find_comfy_sam3_multiplex() is not None:
        return "comfy-multiplex-only"
    return "missing"


def _get_predictor(model_dir: Path | None = None) -> Any:
    """Load and cache Sam3Predictor for Continuity-style ``"face"`` detection."""
    global _predictor, _predictor_root
    root = sam3_root(model_dir)
    weight = mlx_weight_path(model_dir)
    if weight is None:
        raise SamDetectError(
            f"SAM 3.1 MLX weights missing under {root}. "
            f"Open Models and download SAM 3.1 ({SAM31_REPO})."
        )
    if not _try_import_mlx_sam():
        from h3_paths import is_frozen

        if is_frozen():
            raise SamDetectError(
                "SAM 3.1 runtime is missing from this H3-WS build. "
                "Update to the latest release from GitHub Releases, then retry Faces."
            )
        raise SamDetectError(
            "mlx_vlm SAM 3.1 runtime unavailable. "
            "Install mlx-vlm (>=0.4.3) into the H3-WS Python environment "
            "(see requirements_macos.txt)."
        )

    key = str(root.resolve())
    with _predictor_lock:
        if _predictor is not None and _predictor_root == key:
            return _predictor

        import mlx.core as mx
        from mlx_vlm.models.sam3.generate import Sam3Predictor
        from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
        from mlx_vlm.utils import load_model

        log.info("Loading SAM 3.1 MLX detector from %s", root)
        # mlx_vlm.load_model requires a Path (it does model_path / "*.safetensors").
        model = load_model(Path(root))
        processor = Sam31Processor.from_pretrained(str(root))
        processor.image_size = DETECT_EDGE
        predictor = Sam3Predictor(model, processor, score_threshold=DETECT_THRESHOLD)
        # Warm text cache for the Continuity prompt once.
        try:
            predictor._get_input_embeddings(DETECT_PROMPT)
            mx.eval([])
        except Exception:
            pass
        _predictor = predictor
        _predictor_root = key
        return _predictor


def _rgb_to_pil(rgb):
    import numpy as np
    from PIL import Image

    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = (
            (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            if arr.max() <= 1.5
            else arr.astype(np.uint8)
        )
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected HxWx3 RGB frame")
    return Image.fromarray(arr, mode="RGB")


def _detect_haar(rgb) -> list[Box]:
    """Legacy OpenCV Haar — Inspect preview only; never used for Faces repair."""
    import cv2
    import numpy as np

    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = (
            (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            if arr.max() <= 1.5
            else arr.astype(np.uint8)
        )
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    hits = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
    )
    return [(float(x), float(y), float(w), float(h)) for (x, y, w, h) in hits]


def _detect_mlx(
    rgb,
    *,
    model_dir: Path | None = None,
    threshold: float = DETECT_THRESHOLD,
) -> list[Box]:
    """Continuity-equivalent open-vocab face boxes via mlx_vlm Sam3Predictor."""
    import numpy as np

    predictor = _get_predictor(model_dir)
    image = _rgb_to_pil(rgb)
    result = predictor.predict(image, DETECT_PROMPT, score_threshold=threshold)
    boxes_xyxy = getattr(result, "boxes", None)
    scores = getattr(result, "scores", None)
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return []
    out: list[Box] = []
    for i, box in enumerate(np.asarray(boxes_xyxy)):
        if scores is not None and float(np.asarray(scores).reshape(-1)[i]) < threshold:
            continue
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2 - x1, y2 - y1))
    return out


def detect_faces_in_frame(
    rgb,
    *,
    model_dir: Path | None = None,
    threshold: float = DETECT_THRESHOLD,
    allow_fallback: bool = False,
) -> list[Box]:
    """Detect faces in one RGB frame.

    Default ``allow_fallback=False`` — Faces repair requires SAM. Pass
    ``allow_fallback=True`` only for Library Inspect when SAM is missing.
    """
    try:
        return _detect_mlx(rgb, model_dir=model_dir, threshold=threshold)
    except SamDetectError:
        if not allow_fallback:
            raise
        log.warning("SAM unavailable; Haar preview only (Faces pass disabled)")
        return _detect_haar(rgb)
    except Exception as exc:
        if not allow_fallback:
            raise SamDetectError(f"SAM 3.1 detect failed: {exc}") from exc
        log.warning("SAM detect failed (%s); Haar preview only", exc)
        return _detect_haar(rgb)


def track_faces_in_clip(
    frames: Sequence[Any],
    *,
    model_dir: Path | None = None,
    interval: int = DETECT_INTERVAL,
    threshold: float = DETECT_THRESHOLD,
    allow_fallback: bool = False,
    max_faces: int = 1,
    prefer_video_track: bool = True,
) -> tuple[list[Box], list[bool]]:
    """Continuity sample schedule: every ``interval`` frames + last, then ``pick``.

    Video multiplex tracking is attempted when available; otherwise per-frame
    SAM detection on the Continuity sample indices (same as facepass._detect).
    """
    count = len(frames)
    if count <= 0:
        raise ValueError("no frames")

    if not allow_fallback and not sam3_ready(model_dir):
        raise SamDetectError(
            "SAM 3.1 is required for Faces. Download it from Models "
            f"({SAM31_REPO}), then retry."
        )

    if prefer_video_track and sam3_ready(model_dir):
        try:
            return _track_mlx_video(
                frames,
                model_dir=model_dir,
                threshold=threshold,
                max_faces=max_faces,
            )
        except Exception as exc:
            log.debug("SAM video track unavailable (%s); using sampled detect", exc)

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
        # Continuity hands every SAM box to faces.pick — IoU keeps the track;
        # largest only wins when nothing overlaps. Do not pre-truncate.
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
    """Optional full-clip multiplex track — raise to fall back to sampled detect."""
    raise RuntimeError("SAM 3.1 MLX video multiplex not wired; use sampled detect")
