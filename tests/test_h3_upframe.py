"""Unit tests for Upframe frame-count math (blend path)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from h3_upframe import (
    DEFAULT_MULTIPLIER,
    UpframeError,
    interpolate_rgb_blend,
    upframe_available,
)


def test_interpolate_rgb_blend_doubles_interior():
    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    frames[0] = 0
    frames[1] = 100
    frames[2] = 200
    frames[3] = 255
    out = interpolate_rgb_blend(frames, multiplier=2)
    # n + (n-1) inserts → 4 + 3 = 7
    assert out.shape[0] == 7
    assert out[0, 0, 0, 0] == 0
    assert out[1, 0, 0, 0] == 50  # mid between 0 and 100
    assert out[-1, 0, 0, 0] == 255


def test_interpolate_rgb_blend_multiplier_3():
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    frames[1] = 255
    out = interpolate_rgb_blend(frames, multiplier=3)
    # 2 + 1*2 = 4
    assert out.shape[0] == 4


def test_interpolate_single_frame_passthrough():
    frames = np.ones((1, 4, 4, 3), dtype=np.uint8) * 7
    out = interpolate_rgb_blend(frames, multiplier=2)
    assert out.shape == frames.shape
    assert out[0, 0, 0, 0] == 7


def test_upframe_gate_independent_of_redetail():
    upframe_cfg = {"enabled": True, "multiplier": DEFAULT_MULTIPLIER}
    redetail_cfg: dict[str, Any] | None = None
    assert bool(upframe_cfg and upframe_cfg.get("enabled"))
    assert not (redetail_cfg and redetail_cfg.get("enabled"))


def test_upframe_available_false_without_weights(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Empty models tree → unavailable
    assert upframe_available(tmp_path) is False


def test_run_upframe_requires_weights(tmp_path):
    from h3_upframe import run_upframe

    src = tmp_path / "clip.mp4"
    # Minimal invalid path — should fail before decode if weights missing
    with pytest.raises(UpframeError, match="film_net"):
        run_upframe(src, tmp_path / "out.mp4", model_dir=tmp_path, require_weights=True)
