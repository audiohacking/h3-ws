"""Faces must replace the baseline in place — no *_faces.mp4 twin."""

from __future__ import annotations

from pathlib import Path

from web_ui import _promote_video_over_baseline


def test_promote_replaces_baseline_and_removes_work_file(tmp_path: Path) -> None:
    baseline = tmp_path / "clip.mp4"
    work = tmp_path / "clip.faces_work.mp4"
    baseline.write_bytes(b"BASELINE")
    work.write_bytes(b"FACES_FINAL")

    out = _promote_video_over_baseline(baseline, work)

    assert out == baseline
    assert baseline.read_bytes() == b"FACES_FINAL"
    assert not work.exists()
    assert not (tmp_path / "clip.promoting.mp4").exists()


def test_promote_idempotent_when_same_path(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"SAME")
    assert _promote_video_over_baseline(path, path).read_bytes() == b"SAME"
