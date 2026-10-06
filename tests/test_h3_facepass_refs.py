"""Faces re-draw must keep the generation's Ref2VA refs (Continuity parity)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from h3_backend import GenerateRequest, RefItem
from h3_facepass import FacesSettings, run_faces_pass


def _write_tiny_mp4(path: Path, frames: int = 22, size: int = 64) -> Path:
    from h3_facepass import write_rgb_mp4

    rgb = np.zeros((frames, size, size, 3), dtype=np.uint8)
    # Bright square "face" so Haar/track isn't required — we inject boxes.
    rgb[:, 20:44, 20:44] = 220
    write_rgb_mp4(rgb, path, fps=24)
    return path


def test_faces_pass_uses_ref2va_refs(tmp_path: Path) -> None:
    src = _write_tiny_mp4(tmp_path / "src.mp4")
    out = tmp_path / "out.mp4"
    ref_img = tmp_path / "face.png"
    # Minimal PNG via Pillow if available, else skip path existence for RefItem.
    try:
        from PIL import Image

        Image.new("RGB", (64, 64), (200, 160, 140)).save(ref_img)
    except ImportError:
        ref_img.write_bytes(b"\x89PNG\r\n\x1a\n")

    captured: list[GenerateRequest] = []

    class FakeEngine:
        def generate(self, req: GenerateRequest, *, on_progress=None) -> dict:
            captured.append(req)
            if on_progress:
                on_progress({"stage": "denoise", "step": 1, "total": 4, "pct": 25.0})
                on_progress({"stage": "denoise", "step": 4, "total": 4, "pct": 100.0})
            # Echo init video as "refined" so composite has frames.
            from shutil import copy2

            assert req.init_video is not None
            copy2(req.init_video, req.output_path)
            return {"ok": True}

    n = 22
    # Inject a tracked face on every frame (skip real detector).
    box = (20.0, 20.0, 24.0, 24.0)
    boxes = [box] * n

    refs = [RefItem(kind="image", path=ref_img, name="face.png")]

    # Patch tracker to return our boxes.
    import h3_facepass as fp

    original = fp.track_faces_in_clip

    def _fake_track(frames, **kwargs):
        count = len(frames)
        return [box] * count, [True] * count

    events: list[tuple[str, dict]] = []

    def _on_progress(phase: str, extra: dict) -> None:
        events.append((phase, extra))

    fp.track_faces_in_clip = _fake_track  # type: ignore[assignment]
    try:
        run_faces_pass(
            src,
            out,
            prompt="Picture 1 speaking",
            engine=FakeEngine(),
            settings=FacesSettings(canvas=384, denoise=0.45, seed=7, abstain=False),
            refs=refs,
            confirmed_boxes=boxes,
            on_progress=_on_progress,
            preview_latent_dir=tmp_path / "preview",
        )
    finally:
        fp.track_faces_in_clip = original  # type: ignore[assignment]

    assert out.is_file()
    assert captured, "engine.generate was never called"
    req = captured[0]
    assert req.mode == "ref2va"
    assert len(req.refs) == 1
    assert req.refs[0].path == ref_img
    assert req.init_video is not None
    assert req.denoise_strength is not None
    assert 0.0 < float(req.denoise_strength) < 1.0
    assert req.preview_latent_dir == tmp_path / "preview"
    window_events = [e for e in events if e[0] == "faces_window"]
    assert window_events, "faces_window progress never emitted"
    assert any(
        isinstance(extra.get("model_progress"), dict)
        and extra["model_progress"].get("step") == 4
        for _, extra in window_events
    ), "nested denoise progress was not forwarded"


def test_faces_pass_abstain_leaves_as_is(tmp_path: Path) -> None:
    src = _write_tiny_mp4(tmp_path / "src.mp4", size=256)
    out = tmp_path / "out.mp4"
    # Huge face → abstain copies source (Continuity-style leave-as-is).
    boxes = [(10.0, 10.0, 200.0, 200.0)] * 22
    engine = MagicMock()

    import h3_facepass as fp

    original = fp.track_faces_in_clip
    fp.track_faces_in_clip = lambda frames, **kw: (  # type: ignore[assignment]
        [(10.0, 10.0, 200.0, 200.0)] * len(frames),
        [True] * len(frames),
    )
    try:
        result = run_faces_pass(
            src,
            out,
            prompt="x",
            engine=engine,
            settings=FacesSettings(abstain=True),
            confirmed_boxes=boxes,
        )
    finally:
        fp.track_faces_in_clip = original  # type: ignore[assignment]

    assert result == out
    assert out.is_file()
    engine.generate.assert_not_called()
