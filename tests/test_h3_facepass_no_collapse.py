"""Guard: Faces path must not revive the v0.0.14 collapsed-denoise failure."""

from __future__ import annotations

from pathlib import Path


def test_facepass_source_uses_strengths_file_not_window_denoise() -> None:
    src = Path(__file__).resolve().parents[1] / "h3_facepass.py"
    text = src.read_text(encoding="utf-8")
    assert "denoise_video_strengths_file" in text
    assert "write_strengths_file" in text
    assert "latent_strengths" in text
    assert "window_denoise" not in text
    assert "h3_blend_video_latent_strengths" not in text
