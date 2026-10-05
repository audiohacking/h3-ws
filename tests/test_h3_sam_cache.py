"""SAM cache / readiness helpers (no network)."""

from __future__ import annotations

from pathlib import Path

from h3_sam import (
    _MIN_WEIGHT_BYTES,
    _weight_files,
    adopt_local_sam3,
    sam3_ready,
    sam3_root,
    sam3_status,
)


def test_config_only_is_not_ready(tmp_path: Path):
    models = tmp_path / "models"
    (models / "MiniMax-H3").mkdir(parents=True)
    root = models / "sam3.1"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    assert not _weight_files(root)


def test_sam3_ready_with_large_weight(tmp_path: Path):
    models = tmp_path / "models"
    (models / "MiniMax-H3").mkdir(parents=True)
    root = models / "sam3.1"
    root.mkdir()
    with (root / "model.safetensors").open("wb") as fh:
        fh.truncate(_MIN_WEIGHT_BYTES)
    assert sam3_ready(models / "MiniMax-H3")


def test_adopt_comfy_style_weight(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    (models / "MiniMax-H3").mkdir(parents=True)
    comfy = tmp_path / "comfy" / "sam3.1_multiplex.safetensors"
    comfy.parent.mkdir(parents=True)
    with comfy.open("wb") as fh:
        fh.truncate(_MIN_WEIGHT_BYTES)
    monkeypatch.setattr("h3_sam.find_comfy_sam3_multiplex", lambda: comfy)
    monkeypatch.setattr("h3_sam.find_hf_cached_sam3", lambda: None)
    assert adopt_local_sam3(models / "MiniMax-H3") is not None
    assert sam3_ready(models / "MiniMax-H3")
    st = sam3_status(models / "MiniMax-H3")
    assert st["present"] is True
    assert st["note"] == "Face detector for the Faces pass."


def test_sam3_root_sibling():
    root = sam3_root(Path("/tmp/models/MiniMax-H3"))
    assert root.name == "sam3.1"
