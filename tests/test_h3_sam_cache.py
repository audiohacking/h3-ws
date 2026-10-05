"""SAM cache / readiness helpers (no network)."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from h3_sam import (
    _MIN_WEIGHT_BYTES,
    _weight_files,
    adopt_local_sam3,
    mlx_weight_path,
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
    assert mlx_weight_path(models / "MiniMax-H3") is None


def test_comfy_multiplex_alone_is_not_ready(tmp_path: Path, monkeypatch):
    """Comfy torch multiplex must not enable Faces (wrong weight format)."""
    models = tmp_path / "models"
    (models / "MiniMax-H3").mkdir(parents=True)
    root = models / "sam3.1"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    comfy = root / "sam3.1_multiplex.safetensors"
    with comfy.open("wb") as fh:
        fh.truncate(_MIN_WEIGHT_BYTES)
    monkeypatch.setattr("h3_sam.find_hf_cached_sam3", lambda: None)
    monkeypatch.setattr("h3_sam._try_import_mlx_sam", lambda: True)
    assert adopt_local_sam3(models / "MiniMax-H3") is None
    assert not sam3_ready(models / "MiniMax-H3")
    st = sam3_status(models / "MiniMax-H3")
    assert st["present"] is False
    assert st["essential"] is True


def test_sam3_ready_with_mlx_pack(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    (models / "MiniMax-H3").mkdir(parents=True)
    root = models / "sam3.1"
    root.mkdir()
    (root / "config.json").write_text('{"architectures":["Sam3VideoModel"]}', encoding="utf-8")
    with (root / "model.safetensors").open("wb") as fh:
        fh.truncate(_MIN_WEIGHT_BYTES)
    monkeypatch.setattr("h3_sam._try_import_mlx_sam", lambda: True)
    assert mlx_weight_path(models / "MiniMax-H3") is not None
    assert sam3_ready(models / "MiniMax-H3")
    st = sam3_status(models / "MiniMax-H3")
    assert st["present"] is True
    assert st["backend"] == "sam3.1-mlx"


def test_sam3_ready_needs_runtime(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    (models / "MiniMax-H3").mkdir(parents=True)
    root = models / "sam3.1"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    with (root / "model.safetensors").open("wb") as fh:
        fh.truncate(_MIN_WEIGHT_BYTES)
    monkeypatch.setattr("h3_sam._try_import_mlx_sam", lambda: False)
    assert not sam3_ready(models / "MiniMax-H3")


def test_adopt_hf_mlx_snapshot(tmp_path: Path, monkeypatch):
    models = tmp_path / "models"
    (models / "MiniMax-H3").mkdir(parents=True)
    snap = tmp_path / "hf-snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}", encoding="utf-8")
    with (snap / "model.safetensors").open("wb") as fh:
        fh.truncate(_MIN_WEIGHT_BYTES)
    monkeypatch.setattr("h3_sam.find_hf_cached_sam3", lambda: snap)
    monkeypatch.setattr("h3_sam._try_import_mlx_sam", lambda: True)
    assert adopt_local_sam3(models / "MiniMax-H3") is not None
    assert sam3_ready(models / "MiniMax-H3")


def test_sam3_root_sibling():
    root = sam3_root(Path("/tmp/models/MiniMax-H3"))
    assert root.name == "sam3.1"
