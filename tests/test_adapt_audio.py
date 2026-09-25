"""PyAV adapt_audio_for_h3: stereo soundtrack for H3 ingest; silent video pass-through."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import av
import numpy as np

from h3_media import adapt_audio_for_h3, probe_media


def _write_video_only(path: Path, *, frames: int = 12, w: int = 64, h: int = 48) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as out:
        v = out.add_stream("libx264", rate=24)
        v.width = w
        v.height = h
        v.pix_fmt = "yuv420p"
        v.options = {"preset": "ultrafast", "crf": "28"}
        for i in range(frames):
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            rgb[:, :, 0] = (i * 20) % 256
            vf = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
            vf.pts = i
            for packet in v.encode(vf):
                out.mux(packet)
        for packet in v.encode(None):
            out.mux(packet)
    return path


def _write_mono_av(path: Path, *, frames: int = 12, w: int = 64, h: int = 48) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as out:
        v = out.add_stream("libx264", rate=24)
        v.width = w
        v.height = h
        v.pix_fmt = "yuv420p"
        v.options = {"preset": "ultrafast", "crf": "28"}
        a = out.add_stream("aac", rate=44100)
        a.layout = "mono"
        hop = 1024
        sample_pts = 0
        for i in range(frames):
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            rgb[:, :, 1] = (i * 20) % 256
            vf = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
            vf.pts = i
            for packet in v.encode(vf):
                out.mux(packet)
            t = (np.arange(hop, dtype=np.float32) + sample_pts) / 44100.0
            tone = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
            af = av.AudioFrame.from_ndarray(tone.reshape(1, -1), format="fltp", layout="mono")
            af.sample_rate = 44100
            af.pts = sample_pts
            for packet in a.encode(af):
                out.mux(packet)
            sample_pts += hop
        for packet in v.encode(None):
            out.mux(packet)
        for packet in a.encode(None):
            out.mux(packet)
    return path


class AdaptAudioTests(unittest.TestCase):
    def test_silent_video_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _write_video_only(Path(tmp) / "silent.mp4")
            dest = Path(tmp) / "out.mp4"
            out = adapt_audio_for_h3(src, dest)
            self.assertEqual(out, dest)
            info = probe_media(out)
            self.assertTrue(info["has_video"])
            self.assertFalse(info["has_audio"])

    def test_mono_becomes_stereo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _write_mono_av(Path(tmp) / "mono.mp4")
            before = probe_media(src)
            self.assertEqual(before.get("audio_channels"), 1)
            dest = Path(tmp) / "stereo.mp4"
            out = adapt_audio_for_h3(src, dest)
            info = probe_media(out)
            self.assertTrue(info["has_video"])
            self.assertTrue(info["has_audio"])
            self.assertEqual(info.get("audio_channels"), 2)
            self.assertEqual(info.get("audio_codec"), "aac")


if __name__ == "__main__":
    unittest.main()
