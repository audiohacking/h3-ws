# Tiny smoke test for frozen-path helpers (no network, no PyInstaller).
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from h3_paths import (
    discover_existing_model_dirs,
    looks_like_minimax_h3,
    save_desktop_config,
    load_desktop_config,
)


class TestDesktopPaths(unittest.TestCase):
    def test_looks_like_minimax_h3(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(looks_like_minimax_h3(root))
            (root / "FL2VA").mkdir()
            self.assertTrue(looks_like_minimax_h3(root))

    def test_discover_respects_env(self) -> None:
        with TemporaryDirectory() as tmp:
            model = Path(tmp) / "MiniMax-H3"
            (model / "FL2VA" / "transformer").mkdir(parents=True)
            with mock.patch.dict("os.environ", {"H3_MODEL_DIR": str(model)}, clear=False):
                found = discover_existing_model_dirs(limit=5)
            self.assertTrue(any(p.resolve() == model.resolve() for p in found))

    def test_config_roundtrip(self) -> None:
        with TemporaryDirectory() as tmp:
            with mock.patch("h3_paths.writable_root", return_value=Path(tmp)):
                save_desktop_config({"model_dir": "/tmp/models/MiniMax-H3", "setup_done": True})
                cfg = load_desktop_config()
            self.assertEqual(cfg.get("model_dir"), "/tmp/models/MiniMax-H3")
            self.assertTrue(cfg.get("setup_done"))


if __name__ == "__main__":
    unittest.main()
