"""Unit tests for ReDetail size math + orchestration gate."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from h3_media import redetail_output_size, validate_canvas
from h3_redetail import (
    DEFAULT_REDETAIL_DENOISE,
    clamp_redetail_denoise,
    redetail_available,
)


def test_redetail_available_default_on():
    assert redetail_available() is True


def test_clamp_redetail_denoise():
    assert clamp_redetail_denoise(0.01) == pytest.approx(0.15)
    assert clamp_redetail_denoise(0.99) == pytest.approx(0.85)
    assert clamp_redetail_denoise(DEFAULT_REDETAIL_DENOISE) == DEFAULT_REDETAIL_DENOISE


def test_redetail_output_size_doubles_and_snaps():
    assert redetail_output_size(256, 256) == (512, 512)
    assert redetail_output_size(768, 768) == (1536, 1536)
    w, h = redetail_output_size(1024, 576)
    assert w % 32 == 0 and h % 32 == 0
    assert (w, h) == validate_canvas(w, h)


def test_redetail_gate_independent_of_upframe():
    redetail_cfg = {"enabled": True}
    upframe_cfg: dict[str, Any] | None = None
    assert bool(redetail_cfg and redetail_cfg.get("enabled"))
    assert not (upframe_cfg and upframe_cfg.get("enabled"))


def test_upscale_rgb_bridge_shape():
    from h3_redetail import _upscale_rgb

    frames = np.zeros((3, 32, 48, 3), dtype=np.uint8)
    frames[:, :, :, 0] = 100
    out = _upscale_rgb(frames, 96, 64)
    assert out.shape == (3, 64, 96, 3)
    assert out.dtype == np.uint8
