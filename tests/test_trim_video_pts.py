"""Regression: trim must stamp encode PTS in 1/fps (PyAV Time docs).

Bug: re-encoding decoded frames kept the source ``time_base`` while clearing
or rewriting ``pts``, so the timeline collapsed. ffmpeg ``fps=24`` then kept
1 frame → "reference videos require at least 5 decoded frames".
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from h3_media import (
    _sane_video_fps,
    _stamp_encode_video_pts,
    adapt_audio_for_h3,
    trim_media,
)


def _write_av(path: Path, *, frames: int = 48, fps: int = 24, w: int = 64, h: int = 48) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as out:
        v = out.add_stream("libx264", rate=fps)
        v.width = w
        v.height = h
        v.pix_fmt = "yuv420p"
        v.options = {"preset": "ultrafast", "crf": "28"}
        a = out.add_stream("aac", rate=48000)
        a.layout = "stereo"
        hop = 2000
        sample_pts = 0
        for i in range(frames):
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            rgb[:, :, 0] = (i * 5) % 256
            vf = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
            vf.pts = i
            for packet in v.encode(vf):
                out.mux(packet)
            t = (np.arange(hop, dtype=np.float32) + sample_pts) / 48000.0
            tone = (0.05 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
            stereo = np.stack([tone, tone], axis=0)
            af = av.AudioFrame.from_ndarray(stereo, format="fltp", layout="stereo")
            af.sample_rate = 48000
            af.pts = sample_pts
            for packet in a.encode(af):
                out.mux(packet)
            sample_pts += hop
        for packet in v.encode(None):
            out.mux(packet)
        for packet in a.encode(None):
            out.mux(packet)
    return path


def _ffmpeg_fps24_frame_count(path: Path, width: int, height: int, max_frames: int = 200) -> int:
    """Mirror h3.c h3_ffmpeg_read_video_f32 filter chain."""
    ffmpeg = "ffmpeg"
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"fps=24,scale={width}:{height}:flags=lanczos,setsar=1",
        "-frames:v",
        str(max_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        raise unittest.SkipTest("ffmpeg not on PATH") from None
    if proc.returncode != 0:
        raise AssertionError(proc.stderr.decode()[:400])
    frame_bytes = width * height * 3
    return len(proc.stdout) // frame_bytes


class TrimVideoPtsTests(unittest.TestCase):
    def test_sane_fps_clamps_absurd_rate(self) -> None:
        class Fake:
            average_rate = 6840

        self.assertEqual(_sane_video_fps(Fake()), 24.0)

    def test_stamp_encode_video_pts_sets_one_over_fps(self) -> None:
        frame = av.VideoFrame(64, 48, "yuv420p")
        # Simulate a decoded frame that still carries a fine source time_base.
        frame.pts = 512
        frame.time_base = Fraction(1, 15360)
        _stamp_encode_video_pts(frame, 3, 24.0)
        self.assertEqual(frame.pts, 3)
        self.assertEqual(frame.time_base, Fraction(1, 24))

    def test_trim_keeps_pts_span_and_fps24_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _write_av(Path(tmp) / "src.mp4", frames=48, fps=24)
            dest = Path(tmp) / "trim.mp4"
            trim_media(src, dest, start_s=0.25, end_s=1.5)
            self.assertTrue(dest.is_file())

            with av.open(str(dest)) as probe:
                vs = next(s for s in probe.streams if s.type == "video")
                times: list[float] = []
                for frame in probe.decode(video=0):
                    if frame.time is not None:
                        times.append(float(frame.time))
                self.assertGreaterEqual(len(times), 20)
                span = times[-1] - times[0]
                self.assertGreater(span, 0.5, f"PTS collapsed: span={span} times[0]={times[0]}")
                self.assertLessEqual(_sane_video_fps(vs), 120.0)

            n = _ffmpeg_fps24_frame_count(dest, 64, 48)
            self.assertGreaterEqual(
                n,
                5,
                f"h3-style fps=24 decode got {n} frames (need ≥5)",
            )

    def test_trim_then_adapt_still_decodable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Mono soundtrack forces adapt_audio_for_h3 to re-encode A+V.
            mono = Path(tmp) / "mono.mp4"
            with av.open(str(mono), mode="w") as out:
                v = out.add_stream("libx264", rate=24)
                v.width, v.height, v.pix_fmt = 64, 48, "yuv420p"
                v.options = {"preset": "ultrafast", "crf": "28"}
                a = out.add_stream("aac", rate=44100)
                a.layout = "mono"
                hop = 1024
                sample_pts = 0
                for i in range(36):
                    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
                    rgb[:, :, 1] = (i * 7) % 256
                    vf = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(
                        format="yuv420p"
                    )
                    vf.pts = i
                    for packet in v.encode(vf):
                        out.mux(packet)
                    t = (np.arange(hop, dtype=np.float32) + sample_pts) / 44100.0
                    tone = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
                    af = av.AudioFrame.from_ndarray(
                        tone.reshape(1, -1), format="fltp", layout="mono"
                    )
                    af.sample_rate = 44100
                    af.pts = sample_pts
                    for packet in a.encode(af):
                        out.mux(packet)
                    sample_pts += hop
                for packet in v.encode(None):
                    out.mux(packet)
                for packet in a.encode(None):
                    out.mux(packet)
            trimmed = Path(tmp) / "trim.mp4"
            trim_media(mono, trimmed, start_s=0.0, end_s=1.0)
            adapted = Path(tmp) / "adapted.mp4"
            adapt_audio_for_h3(trimmed, adapted)
            n = _ffmpeg_fps24_frame_count(adapted, 64, 48)
            self.assertGreaterEqual(n, 5, f"adapted trim fps=24 decode got {n}")


if __name__ == "__main__":
    unittest.main()
