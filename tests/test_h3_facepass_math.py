"""Unit tests for Faces init/denoise helpers (no GPU)."""

from __future__ import annotations

import math

from h3_faces import should_abstain, window_denoise


def denoise_start_step(steps: int, strength: float) -> int:
    """Mirror of h3_denoise_start_step in h3_host.c."""
    if steps < 1:
        return 0
    if not (strength > 0.0):
        return steps
    if strength >= 1.0:
        return 0
    start = int(round((1.0 - float(strength)) * float(steps)))
    return max(0, min(steps, start))


def test_denoise_start_step_full_noise():
    assert denoise_start_step(20, 1.0) == 0
    assert denoise_start_step(20, 0.999) == 0


def test_denoise_start_step_faces_default():
    # Continuity default ~0.45 → start about halfway through.
    assert denoise_start_step(20, 0.45) == 11
    assert denoise_start_step(8, 0.45) == 4


def test_denoise_start_step_zero():
    assert denoise_start_step(20, 0.0) == 20


def test_window_denoise_max():
    assert math.isclose(window_denoise(0.45, [0.8, 0.5, 0.35]), 0.45 * 0.8)


def test_abstain_closeup():
    assert should_abstain([120.0, 130.0])
    assert not should_abstain([40.0, 130.0])
