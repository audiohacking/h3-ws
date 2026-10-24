"""Faces init CFR / fps=24 round-trip / temporal lock (M5 desync class)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import av
import numpy as np

from h3_facepass import (
    align_refined_frame_count,
    decode_fps24_rgb,
    init_roundtrip_index_drift,
    init_roundtrip_mae,
    mux_window_audio,
    temporal_lock_refined,
    verify_init_roundtrip,
    write_rgb_mp4,
)
from h3_faces import FaceError
from h3_media import _sane_video_fps, _stamp_encode_video_pts


def _ramp_frames(n: int, h: int = 64, w: int = 64) -> np.ndarray:
    """Unique solid color per frame index (encode-stable identity)."""
    out = np.zeros((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        out[i, :, :, 0] = (i * 11) % 256
        out[i, :, :, 1] = (i * 29) % 256
        out[i, :, :, 2] = (255 - i * 7) % 256
    return out


def _write_av_with_audio(
    path: Path, *, frames: int = 48, fps: int = 24, w: int = 64, h: int = 48
) -> Path:
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
            _stamp_encode_video_pts(vf, i, float(fps))
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


class FacesSyncPtsTests(unittest.TestCase):
    def test_write_rgb_mp4_stamps_cfr_pts_span(self) -> None:
        n = 22
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "silent.mp4"
            write_rgb_mp4(_ramp_frames(n), path, fps=24)
            with av.open(str(path)) as probe:
                times: list[float] = []
                for frame in probe.decode(video=0):
                    if frame.time is not None:
                        times.append(float(frame.time))
                self.assertEqual(len(times), n)
                span = times[-1] - times[0]
                self.assertGreater(span, 0.7, f"PTS collapsed: span={span}")
                self.assertAlmostEqual(span, (n - 1) / 24.0, delta=0.08)
                vs = next(s for s in probe.streams if s.type == "video")
                self.assertLessEqual(_sane_video_fps(vs), 120.0)

    def test_mux_window_audio_keeps_cfr_for_fps24(self) -> None:
        n = 22
        with tempfile.TemporaryDirectory() as tmp:
            silent = Path(tmp) / "silent.mp4"
            write_rgb_mp4(_ramp_frames(n), silent, fps=24)
            src = _write_av_with_audio(Path(tmp) / "src.mp4", frames=48, fps=24)
            dest = Path(tmp) / "muxed.mp4"
            mux_window_audio(silent, src, dest, start_frame=0, end_frame=n, fps=24)

            with av.open(str(dest)) as probe:
                times: list[float] = []
                for frame in probe.decode(video=0):
                    if frame.time is not None:
                        times.append(float(frame.time))
                self.assertEqual(len(times), n)
                span = times[-1] - times[0]
                self.assertGreater(span, 0.7, f"mux PTS collapsed: span={span}")
                self.assertAlmostEqual(span, (n - 1) / 24.0, delta=0.08)

            got = decode_fps24_rgb(dest, width=64, height=64, max_frames=n, fps=24.0)
            self.assertEqual(got.shape[0], n)
            crop = _ramp_frames(n)
            drift = init_roundtrip_index_drift(crop, got)
            self.assertLessEqual(drift, 0.35, f"fps=24 index drift={drift}")
            mae = init_roundtrip_mae(crop, got)
            self.assertLess(mae, 55.0, f"fps=24 round-trip mae={mae}")


class FacesRoundtripTests(unittest.TestCase):
    def test_verify_init_roundtrip_happy_path(self) -> None:
        n = 22
        crop = _ramp_frames(n)
        with tempfile.TemporaryDirectory() as tmp:
            silent = Path(tmp) / "silent.mp4"
            write_rgb_mp4(crop, silent, fps=24)
            src = _write_av_with_audio(Path(tmp) / "src.mp4", frames=48)
            muxed = Path(tmp) / "muxed.mp4"
            mux_window_audio(silent, src, muxed, start_frame=0, end_frame=n, fps=24)
            ok, mae = verify_init_roundtrip(muxed, crop, fps=24)
            self.assertTrue(ok, f"round-trip failed mae={mae}")

    def test_silent_init_alone_roundtrips(self) -> None:
        n = 22
        crop = _ramp_frames(n)
        with tempfile.TemporaryDirectory() as tmp:
            silent = Path(tmp) / "silent.mp4"
            write_rgb_mp4(crop, silent, fps=24)
            ok, mae = verify_init_roundtrip(silent, crop, fps=24)
            self.assertTrue(ok, f"silent round-trip failed mae={mae}")

    def test_photo_like_roundtrip_passes_index_drift(self) -> None:
        """x264 MAE ~30 on photo-like crops must still pass (drift gate)."""
        rng = np.random.default_rng(0)
        n, h, w = 22, 96, 96
        crop = np.zeros((n, h, w, 3), dtype=np.uint8)
        base = rng.integers(40, 200, size=(h, w, 3), dtype=np.uint8)
        for i in range(n):
            crop[i] = np.clip(
                base.astype(np.int16) + rng.integers(-8, 9, size=base.shape), 0, 255
            ).astype(np.uint8)
            crop[i, 60:80, 40:70] = (i * 9) % 256
        with tempfile.TemporaryDirectory() as tmp:
            silent = Path(tmp) / "silent.mp4"
            write_rgb_mp4(crop, silent, fps=24)
            src = _write_av_with_audio(Path(tmp) / "src.mp4", frames=48)
            muxed = Path(tmp) / "muxed.mp4"
            mux_window_audio(silent, src, muxed, start_frame=0, end_frame=n, fps=24)
            ok, mae = verify_init_roundtrip(muxed, crop, fps=24)
            self.assertTrue(ok, f"photo-like round-trip failed mae={mae}")
            self.assertGreater(mae, 18.0, "expected compression MAE > old 18 gate")


class FacesTemporalLockTests(unittest.TestCase):
    def test_align_rejects_wild_mismatch(self) -> None:
        refined = _ramp_frames(10)
        with self.assertRaises(FaceError):
            align_refined_frame_count(refined, 22, slack=2)

    def test_align_trims_small_overshoot(self) -> None:
        refined = _ramp_frames(24)
        out = align_refined_frame_count(refined, 22, slack=2)
        self.assertEqual(out.shape[0], 22)
        np.testing.assert_array_equal(out[0], refined[0])

    def test_temporal_lock_recovers_plus_one_shift(self) -> None:
        n = 22
        crop = _ramp_frames(n)
        # refined[i] = crop[i-1] content → content is late; matching crop[i]
        # needs refined[i+1] so best_shift = +1.
        refined = np.zeros_like(crop)
        refined[0] = crop[0]
        refined[1:] = crop[:-1]
        face_rects = [(16.0, 16.0, 32.0, 32.0)] * n
        locked, delta, mae = temporal_lock_refined(
            refined, crop, face_rects, enabled=True
        )
        self.assertEqual(delta, 1, f"expected δ=+1 got {delta} mae={mae}")
        self.assertEqual(locked.shape[0], n)
        # After +1 lock, locked[i] should equal former refined[i+1] ≈ crop[i]
        # for the interior (encode-free arrays → exact).
        np.testing.assert_array_equal(locked[5], crop[5])

    def test_temporal_lock_disabled_skips_shift(self) -> None:
        n = 22
        crop = _ramp_frames(n)
        refined = np.zeros_like(crop)
        refined[0] = crop[0]
        refined[1:] = crop[:-1]
        face_rects = [(16.0, 16.0, 32.0, 32.0)] * n
        locked, delta, _mae = temporal_lock_refined(
            refined, crop, face_rects, enabled=False
        )
        self.assertEqual(delta, 0)
        np.testing.assert_array_equal(locked[5], refined[5])


if __name__ == "__main__":
    unittest.main()
