"""Unit tests for TAEH3 latent dump parsing (no full model required)."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from h3_preview import H3_LATENT_MAGIC, read_latent_bin, taeh3_available, taeh3_decode_ready


class TestLatentDump(unittest.TestCase):
    def test_read_latent_bin_roundtrip(self) -> None:
        c, t, h, w = 24, 3, 4, 4
        latent = np.linspace(-1, 1, c * t * h * w, dtype=np.float32).reshape(c, t, h, w)
        raw = H3_LATENT_MAGIC + struct.pack("<7I", 1, 7, 20, c, t, h, w) + latent.tobytes()
        path = Path(tempfile.mkdtemp()) / "step_0007.bin"
        path.write_bytes(raw)
        meta, tensor = read_latent_bin(path)
        self.assertEqual(meta["step"], 7)
        self.assertEqual(meta["total"], 20)
        self.assertEqual(tuple(tensor.shape), (c, t, h, w))
        self.assertTrue(np.allclose(tensor.numpy(), latent))

    def test_bad_magic_rejected(self) -> None:
        path = Path(tempfile.mkdtemp()) / "bad.bin"
        path.write_bytes(b"XXXX" + b"\0" * 40)
        with self.assertRaises(ValueError):
            read_latent_bin(path)

    def test_taeh3_availability_is_bool(self) -> None:
        self.assertIsInstance(taeh3_available(), bool)
        self.assertIsInstance(taeh3_decode_ready(), bool)
        # Weights alone are not enough to arm --preview-latent.
        if not taeh3_available():
            self.assertFalse(taeh3_decode_ready())


if __name__ == "__main__":
    unittest.main()
