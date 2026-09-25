"""Upscale / trim / frame bake must work on PyAV 17 (no AudioFrame.reformat)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
from PIL import Image

from h3_media import trim_media, upscale_video


def _write_short_av_mp4(path: Path, *, frames: int = 8, w: int = 64, h: int = 48) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as out:
        v = out.add_stream("libx264", rate=24)
        v.width = w
        v.height = h
        v.pix_fmt = "yuv420p"
        v.options = {"preset": "ultrafast", "crf": "28"}
        a = out.add_stream("aac", rate=48000)
        a.layout = "stereo"
        hop = 1024
        sample_pts = 0
        for i in range(frames):
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            rgb[:, :, 0] = (i * 30) % 256
            rgb[:, :, 1] = 80
            rgb[:, :, 2] = 160
            vf = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
            vf.pts = i
            for packet in v.encode(vf):
                out.mux(packet)
            # ~frames/24 seconds of tone
            t = (np.arange(hop, dtype=np.float32) + sample_pts) / 48000.0
            tone = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
            planar = np.ascontiguousarray(np.stack([tone, tone], axis=0))
            af = av.AudioFrame.from_ndarray(planar, format="fltp", layout="stereo")
            af.sample_rate = 48000
            af.pts = sample_pts
            sample_pts += hop
            for packet in a.encode(af):
                out.mux(packet)
        for packet in v.encode(None):
            out.mux(packet)
        for packet in a.encode(None):
            out.mux(packet)
    return path


class UpscaleAudioTests(unittest.TestCase):
    def test_upscale_preserves_audio_on_pyav17(self) -> None:
        self.assertFalse(hasattr(av.AudioFrame, "reformat"))
        with tempfile.TemporaryDirectory() as tmp:
            src = _write_short_av_mp4(Path(tmp) / "in.mp4")
            dest = Path(tmp) / "out.mp4"
            out, ow, oh = upscale_video(src, dest, scale=2.0)
            self.assertEqual(out, dest)
            self.assertTrue(dest.is_file())
            self.assertGreater(dest.stat().st_size, 500)
            self.assertEqual((ow, oh), (128, 96))
            with av.open(str(dest)) as probe:
                kinds = {s.type for s in probe.streams}
                self.assertIn("video", kinds)
                self.assertIn("audio", kinds)
                v = next(s for s in probe.streams if s.type == "video")
                self.assertEqual(int(v.codec_context.width), 128)
                self.assertEqual(int(v.codec_context.height), 96)

    def test_trim_with_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _write_short_av_mp4(Path(tmp) / "in.mp4", frames=24)
            dest = Path(tmp) / "trim.mp4"
            out = trim_media(src, dest, start_s=0.1, end_s=0.5)
            self.assertTrue(out.is_file())
            with av.open(str(out)) as probe:
                self.assertTrue(any(s.type == "audio" for s in probe.streams))


if __name__ == "__main__":
    unittest.main()
