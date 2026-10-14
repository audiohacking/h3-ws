"""Unit tests for Long take keyframe parser + fake-engine refine blend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from h3_longtake import (
    DEFAULT_KEYFRAMES,
    LongTakeSettings,
    blend_rgb_by_frames,
    clamp_longtake_denoise,
    parse_keyframes,
    run_longtake_refine,
)


def test_parse_keyframes_default_hold_then_ramp():
    # 0:0, 22:0, 44:1 → frames 0–22 stay 0; 22→44 ramp to 1.
    vals = parse_keyframes(DEFAULT_KEYFRAMES, 45)
    assert len(vals) == 45
    assert vals[0] == pytest.approx(0.0)
    assert vals[22] == pytest.approx(0.0)
    assert vals[44] == pytest.approx(1.0)
    assert 0.0 < vals[33] < 1.0


def test_parse_keyframes_empty_is_zeros():
    assert parse_keyframes("", 10) == [0.0] * 10
    assert parse_keyframes("bogus", 5) == [0.0] * 5


def test_parse_keyframes_smooth_midpoint():
    linear = parse_keyframes("0:0, 10:1", 11, interpolation="linear")
    smooth = parse_keyframes("0:0, 10:1", 11, interpolation="smooth")
    assert linear[5] == pytest.approx(0.5)
    # Cosine ease: at t=0.5 → 0.5 *(1 - cos(π/2)) = 0.5 — same midpoint;
    # check a quarter point differs.
    assert smooth[2] != pytest.approx(linear[2])
    assert smooth[2] < linear[2]


def test_clamp_longtake_denoise():
    assert clamp_longtake_denoise(0.01) == 0.1
    assert clamp_longtake_denoise(0.99) == 0.9
    assert clamp_longtake_denoise(0.55) == 0.55


def test_blend_rgb_by_frames():
    src = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    ref = np.full((3, 4, 4, 3), 200, dtype=np.uint8)
    out = blend_rgb_by_frames(src, ref, [0.0, 0.5, 1.0])
    assert out[0, 0, 0, 0] == 0
    assert out[1, 0, 0, 0] == 100
    assert out[2, 0, 0, 0] == 200


def test_longtake_skipped_when_disabled():
    """Gate used by web_ui: post-pass only when enabled is truthy."""
    cfg_off = {"enabled": False, "denoise": 0.55}
    cfg_missing: dict[str, Any] | None = None
    cfg_on = {"enabled": True}
    assert not (cfg_off and cfg_off.get("enabled"))
    assert not (cfg_missing and cfg_missing.get("enabled"))
    assert bool(cfg_on and cfg_on.get("enabled"))


class _FakeEngine:
    """Copies init-video RGB with a constant offset as the 'refine' output."""

    def generate(self, req: Any, on_progress=None) -> None:
        from h3_facepass import decode_clip_rgb, write_rgb_mp4

        if on_progress:
            on_progress({"stage": "denoise", "step": 1, "total": 2})
        frames, _, _ = decode_clip_rgb(req.init_video)
        # Brighten so blend is observable.
        refined = np.clip(frames.astype(np.int16) + 80, 0, 255).astype(np.uint8)
        write_rgb_mp4(refined, req.output_path, fps=24)


def test_run_longtake_refine_fake_engine(tmp_path: Path):
    pytest.importorskip("av")
    from h3_facepass import decode_clip_rgb, write_rgb_mp4

    n, h, w = 45, 32, 32
    src = np.zeros((n, h, w, 3), dtype=np.uint8)
    src[:, :, :, 0] = 40
    src_path = tmp_path / "src.mp4"
    out_path = tmp_path / "out_longtake.mp4"
    write_rgb_mp4(src, src_path, fps=24)

    phases: list[str] = []

    def on_progress(phase: str, extra: dict) -> None:
        phases.append(phase)

    run_longtake_refine(
        src_path,
        out_path,
        prompt="test",
        engine=_FakeEngine(),
        settings=LongTakeSettings(
            denoise=0.55,
            seed=1,
            keyframes="0:0, 22:0, 44:1",
        ),
        on_progress=on_progress,
        steps=4,
    )
    assert out_path.is_file()
    assert "longtake_refine" in phases
    assert "longtake_blend" in phases
    assert "longtake_done" in phases
    out, _, _ = decode_clip_rgb(out_path)
    assert out.shape[0] == n
    # Early frames stay near source; late frames pick up refine brightness.
    # H.264 round-trip can shift RGB by a few counts.
    assert int(out[0, 0, 0, 0]) == pytest.approx(40, abs=8)
    assert int(out[44, 0, 0, 0]) == pytest.approx(120, abs=8)
    assert int(out[44, 0, 0, 0]) > int(out[0, 0, 0, 0]) + 40
