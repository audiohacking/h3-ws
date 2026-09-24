"""Continuity framing (h3_crop) — turn, mirror, window."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from h3_crop import Crop, apply_still, parse, pil, to_dict
from h3_backend import parse_refs_payload


class CropParseTests(unittest.TestCase):
    def test_identity_is_none(self) -> None:
        self.assertIsNone(parse({"x": 0, "y": 0, "w": 1, "h": 1}))

    def test_window(self) -> None:
        crop = parse({"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.4})
        assert crop is not None
        self.assertTrue(crop.windowed)
        self.assertEqual(to_dict(crop)["x"], 0.1)

    def test_turn_only(self) -> None:
        crop = parse({"turn": 90})
        assert crop is not None
        self.assertEqual(crop.turn, 90)
        self.assertFalse(crop.windowed)

    def test_ref_payload_crop_and_trim(self) -> None:
        items = parse_refs_payload(
            [
                {
                    "kind": "image",
                    "path": "/tmp/a.png",
                    "crop": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5, "turn": 90},
                },
                {
                    "kind": "audio",
                    "path": "/tmp/a.wav",
                    "trim": {"start": 1.0, "end": 5.0},
                },
                {"kind": "silent_video", "path": "/tmp/v.mp4"},
            ],
            validate=False,
        )
        self.assertEqual(items[0].crop["turn"], 90)
        self.assertEqual(items[1].trim_start, 1.0)
        self.assertEqual(items[1].trim_end, 5.0)


class CropPilTests(unittest.TestCase):
    def test_apply_still_turn_and_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.png"
            dest = Path(tmp) / "out.png"
            Image.new("RGB", (100, 50), (10, 20, 30)).save(src)
            apply_still(src, dest, Crop(x=0.1, y=0.1, w=0.5, h=0.5, turn=90))
            with Image.open(dest) as out:
                # 100x50 turned 90 → 50x100, then 0.5×0.5 window
                self.assertEqual(out.size, (25, 50))


if __name__ == "__main__":
    unittest.main()
