"""TAEH3 live preview — Continuity / KJNodes ModelPreviewOverride parity.

h3.c ``--preview-latent DIR`` dumps mid-denoise video latents. This module:

1. Decodes the full latent through madebyollin's ``taeh3.safetensors`` (same as
   Continuity's ``preview_frames=1024`` full-clip path — taeh3 is causal).
2. Encodes a short looping clip (MP4 via PyAV libx264, WebP fallback) at the
   family's fps, capped to Continuity's default preview max edge (640).
3. Deletes latent dumps as they are consumed and removes all preview scratch
   when the watcher stops.

Missing weights or torch → generate still works; no preview events.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import struct
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from h3_paths import load_desktop_config, resource_root, writable_root

log = logging.getLogger("h3-preview")

H3_LATENT_MAGIC = b"H3L1"
TAEHV_MODULE = resource_root() / "third_party" / "taehv.py"

# Continuity defaults (creator/settings.py + models.PREVIEW_FRAMES / fps).
PREVIEW_MAX_PX = 640
PREVIEW_QUALITY = 80
PREVIEW_FPS = 24

PreviewCallback = Callable[[dict[str, Any]], None]

_decoder_lock = threading.Lock()
_decoder: Any | None = None
_decoder_device: str | None = None
_decoder_path: Path | None = None


def default_taeh3_path() -> Path:
    env = os.environ.get("H3_TAEH3", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    candidates: list[Path] = []
    cfg = load_desktop_config()
    configured = str(cfg.get("taeh3_path") or "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    repo = str(cfg.get("repo_root") or "").strip()
    if repo:
        candidates.append(Path(repo).expanduser() / "models" / "vae_approx" / "taeh3.safetensors")
    model = str(cfg.get("model_dir") or "").strip()
    if model:
        mp = Path(model).expanduser()
        if mp.name == "MiniMax-H3" and mp.parent.name == "models":
            candidates.append(mp.parent / "vae_approx" / "taeh3.safetensors")
    candidates.append(writable_root() / "models" / "vae_approx" / "taeh3.safetensors")
    candidates.append(resource_root() / "models" / "vae_approx" / "taeh3.safetensors")
    for path in candidates:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            continue
        if resolved.is_file() and resolved.stat().st_size > 1_000_000:
            return resolved
    return (writable_root() / "models" / "vae_approx" / "taeh3.safetensors").resolve()


# Back-compat alias for scripts/download_model.py status lines
DEFAULT_TAEH3 = default_taeh3_path()


def taeh3_available(path: Path | None = None) -> bool:
    """True when the weight file is on disk (download status / --status)."""
    p = Path(path) if path is not None else default_taeh3_path()
    return p.is_file() and p.stat().st_size > 1_000_000


def taeh3_decode_ready(path: Path | None = None) -> bool:
    """True when live preview can actually run — weights + torch + safetensors.

    Arming ``--preview-latent`` without these deps forces a one-shot h3 spawn
    and kills the warm FL2VA session for nothing.
    """
    if not taeh3_available(path):
        return False
    if not TAEHV_MODULE.is_file():
        return False
    try:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            return False
        if importlib.util.find_spec("safetensors") is None:
            return False
    except Exception:
        return False
    return True


def cleanup_preview_artifacts(*paths: Path | None) -> None:
    """Remove latent dump dirs and served preview files. Best-effort."""
    for path in paths:
        if path is None:
            continue
        p = Path(path)
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink(missing_ok=True)
        except OSError as exc:
            log.debug("preview cleanup %s: %s", p, exc)


def _pick_device() -> str:
    """CPU by default — h3.c owns Metal; don't fight it with MPS for previews."""
    forced = os.environ.get("H3_PREVIEW_DEVICE", "").strip().lower()
    if forced in {"cpu", "mps", "cuda"}:
        return forced
    return "cpu"


def _load_taehv(checkpoint: Path) -> Any:
    global _decoder, _decoder_device, _decoder_path
    with _decoder_lock:
        if (
            _decoder is not None
            and _decoder_path == checkpoint
            and _decoder_device is not None
        ):
            return _decoder
        if not TAEHV_MODULE.is_file():
            raise FileNotFoundError(
                f"vendored taehv missing at {TAEHV_MODULE} — "
                "fetch from https://github.com/madebyollin/taehv"
            )
        import importlib.util

        import torch

        spec = importlib.util.spec_from_file_location("taehv", TAEHV_MODULE)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {TAEHV_MODULE}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["taehv"] = mod
        spec.loader.exec_module(mod)

        device = _pick_device()
        model = mod.TAEHV(checkpoint_path=str(checkpoint), arch_name="taeh3")
        model = model.to(device).eval()
        _decoder = model
        _decoder_device = device
        _decoder_path = checkpoint
        log.info("taeh3 loaded on %s from %s", device, checkpoint)
        return model


def read_latent_bin(path: Path) -> tuple[dict[str, int], Any]:
    """Parse an H3L1 dump → meta dict + torch float32 tensor shaped C,T,H,W."""
    import numpy as np
    import torch

    raw = path.read_bytes()
    if len(raw) < 4 + 7 * 4 or raw[:4] != H3_LATENT_MAGIC:
        raise ValueError(f"bad latent header in {path}")
    version, step, total, channels, t, h, w = struct.unpack_from("<7I", raw, 4)
    if version != 1 or channels != 24:
        raise ValueError(f"unsupported latent version/channels in {path}")
    expected = channels * t * h * w
    payload = raw[4 + 7 * 4 :]
    if len(payload) != expected * 4:
        raise ValueError(
            f"latent size mismatch in {path}: got {len(payload)} bytes, "
            f"want {expected * 4} for {channels}x{t}x{h}x{w}"
        )
    arr = np.frombuffer(payload, dtype=np.float32).reshape(channels, t, h, w).copy()
    tensor = torch.from_numpy(arr)
    meta = {
        "step": int(step),
        "total": int(total),
        "channels": int(channels),
        "t": int(t),
        "h": int(h),
        "w": int(w),
    }
    return meta, tensor


def _pil_frames_from_latent(
    latent_cthw: Any,
    *,
    checkpoint: Path,
    max_edge: int = PREVIEW_MAX_PX,
) -> list[Any]:
    """Full-clip taeh3 decode → list of RGB PIL images (Continuity path)."""
    import torch
    from PIL import Image, ImageOps

    model = _load_taehv(checkpoint)
    device = _decoder_device or "cpu"
    # h3.c layout is C,T,H,W; TAEHV wants N,T,C,H,W.
    ntchw = latent_cthw.permute(1, 0, 2, 3).unsqueeze(0).to(device)
    with torch.inference_mode():
        rgb = model.decode_video(ntchw, parallel=True, show_progress_bar=False)
    # rgb: N,T,C,H,W in [0,1] — PIL wants H,W,C per frame.
    u8 = (
        rgb[0]
        .detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)  # T,H,W,C
        .cpu()
        .numpy()
    )
    frames = []
    for i in range(u8.shape[0]):
        img = Image.fromarray(u8[i], mode="RGB")
        if max_edge and max_edge > 0 and (img.width > max_edge or img.height > max_edge):
            img = ImageOps.contain(img, (max_edge, max_edge), Image.Resampling.LANCZOS)
        frames.append(img)
    return frames


def _encode_looping_mp4(frames: list[Any], out_path: Path, *, fps: int) -> Path:
    """Silent H.264 loop clip — Continuity stage plays this with autoplay+loop."""
    import av
    import numpy as np

    if not frames:
        raise ValueError("no frames to encode")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp4")
    rate = max(1, int(fps))
    container = av.open(str(tmp), mode="w")
    try:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = frames[0].width
        stream.height = frames[0].height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "28", "tune": "fastdecode"}
        for img in frames:
            frame = av.VideoFrame.from_ndarray(np.asarray(img.convert("RGB")), format="rgb24")
            frame = frame.reformat(format="yuv420p")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    tmp.replace(out_path)
    return out_path


def _encode_animated_webp(
    frames: list[Any], out_path: Path, *, fps: int, quality: int = PREVIEW_QUALITY
) -> Path:
    """KJNodes fallback when NVENC/MP4 is unavailable — browser loops in <img>."""
    if not frames:
        raise ValueError("no frames to encode")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.webp")
    duration_ms = max(1, int(round(1000 / max(1, fps))))
    frames[0].save(
        tmp,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        quality=quality,
        method=4,
    )
    tmp.replace(out_path)
    return out_path


def decode_latent_to_preview(
    latent_cthw: Any,
    *,
    out_path: Path,
    checkpoint: Path | None = None,
    fps: int = PREVIEW_FPS,
    max_edge: int = PREVIEW_MAX_PX,
    quality: int = PREVIEW_QUALITY,
) -> dict[str, Any]:
    """Full-clip taeh3 → looping preview file.

    Matches KJNodes encode order as closely as we can without NVENC:
    animated WebP first (what Continuity-on-Mac actually shows in ``<img>``),
    then silent H.264 MP4 for the stage's ``video/*`` loop path.
    """
    ckpt = Path(checkpoint) if checkpoint is not None else default_taeh3_path()
    if not taeh3_available(ckpt):
        raise FileNotFoundError(f"taeh3 weights not found at {ckpt}")
    frames = _pil_frames_from_latent(latent_cthw, checkpoint=ckpt, max_edge=max_edge)
    if not frames:
        raise RuntimeError("taeh3 returned no frames")

    out_path = Path(out_path)
    webp_path = out_path.with_suffix(".webp")
    try:
        _encode_animated_webp(frames, webp_path, fps=fps, quality=quality)
        return {
            "path": str(webp_path),
            "mime": "image/webp",
            "fps": fps,
            "frames": len(frames),
        }
    except Exception as exc:
        log.warning("Animated WebP encode failed (%s); trying MP4", exc)

    mp4_path = out_path.with_suffix(".mp4")
    _encode_looping_mp4(frames, mp4_path, fps=fps)
    return {
        "path": str(mp4_path),
        "mime": "video/mp4",
        "fps": fps,
        "frames": len(frames),
    }


def decode_dump_file(
    bin_path: Path,
    *,
    out_path: Path,
    checkpoint: Path | None = None,
    fps: int = PREVIEW_FPS,
) -> dict[str, Any]:
    meta, latent = read_latent_bin(bin_path)
    encoded = decode_latent_to_preview(
        latent, out_path=out_path, checkpoint=checkpoint, fps=fps
    )
    meta.update(encoded)
    return meta


def _unlink_consumed_bins(dump_dir: Path, keep_step: int | None = None) -> None:
    """Drop step_*.bin files once decoded (and any older than keep_step)."""
    for path in dump_dir.glob("step_*.bin"):
        try:
            if keep_step is None:
                path.unlink(missing_ok=True)
                continue
            # step_0007.bin → 7
            stem = path.stem  # step_0007
            num = int(stem.split("_")[-1])
            if num <= keep_step:
                path.unlink(missing_ok=True)
        except (OSError, ValueError):
            continue


class LatentPreviewWatcher:
    """Poll ``latest.json``, decode full-clip looping previews, scrub dumps."""

    def __init__(
        self,
        dump_dir: Path,
        out_path: Path,
        *,
        on_preview: PreviewCallback | None = None,
        checkpoint: Path | None = None,
        poll_s: float = 0.2,
        fps: int = PREVIEW_FPS,
    ) -> None:
        self.dump_dir = Path(dump_dir)
        # Stem only — encoder picks .mp4 / .webp.
        self.out_path = Path(out_path)
        self.on_preview = on_preview
        self.checkpoint = Path(checkpoint) if checkpoint else default_taeh3_path()
        self.poll_s = poll_s
        self.fps = fps
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_step: int | None = None
        self._enabled = taeh3_decode_ready(self.checkpoint)
        self.last_served: Path | None = None
        self.last_mime: str | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if not self._enabled:
            log.info("taeh3 absent — preview watcher idle")
            return
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="taeh3-preview", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 8.0, *, cleanup: bool = True) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        if cleanup:
            cleanup_preview_artifacts(self.dump_dir, self.last_served, self.out_path)
            # Clear any leftover encodings for this stem.
            parent = self.out_path.parent
            if parent.is_dir():
                for sib in parent.glob(self.out_path.stem + ".*"):
                    cleanup_preview_artifacts(sib)

    def _run(self) -> None:
        latest = self.dump_dir / "latest.json"
        while not self._stop.is_set():
            try:
                if latest.is_file():
                    info = json.loads(latest.read_text(encoding="utf-8"))
                    step = int(info.get("step") or 0)
                    if step and step != self._last_step:
                        name = str(info.get("file") or f"step_{step:04d}.bin")
                        bin_path = self.dump_dir / name
                        if bin_path.is_file():
                            # Advance even on failure so a bad step is not
                            # retried every poll for the whole denoise.
                            self._last_step = step
                            try:
                                self._decode_step(bin_path, info)
                            except Exception:
                                log.exception(
                                    "taeh3 preview decode failed at step %s", step
                                )
                                _unlink_consumed_bins(self.dump_dir, keep_step=step)
            except Exception:
                log.exception("taeh3 preview watcher tick failed")
            self._stop.wait(self.poll_s)

    def _decode_step(self, bin_path: Path, info: dict[str, Any]) -> None:
        meta = decode_dump_file(
            bin_path,
            out_path=self.out_path,
            checkpoint=self.checkpoint,
            fps=self.fps,
        )
        served = Path(str(meta["path"]))
        self.last_served = served
        self.last_mime = str(meta.get("mime") or "image/webp")
        step = int(meta.get("step") or info.get("step") or 0)
        total = int(meta.get("total") or info.get("total") or 0)
        # Drop consumed latent dumps so they don't pile up mid-run.
        _unlink_consumed_bins(self.dump_dir, keep_step=step)
        payload = {
            "step": step,
            "total": total,
            "path": str(served),
            "mime": self.last_mime,
            "fps": meta.get("fps", self.fps),
            "frames": meta.get("frames"),
        }
        if self.on_preview:
            self.on_preview(payload)
