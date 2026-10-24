"""Soft-hold contract: Faces windows must emit latent strengths, never collapse."""

from __future__ import annotations

from pathlib import Path
from shutil import copy2

import numpy as np

from h3_backend import GenerateRequest
from h3_facepass import FacesSettings, build_faces_window_request, run_faces_pass
from h3_faces import latent_strengths, strengths, video_latent_t, write_strengths_file


def test_video_latent_t_matches_h3_grid() -> None:
    assert video_latent_t(5) == 2
    assert video_latent_t(22) == 7
    assert video_latent_t(39) == 12


def test_latent_strengths_resamples_curve() -> None:
    curve = strengths([30.0] * 22)
    out = latent_strengths(curve, video_latent_t(22))
    assert len(out) == 7
    assert all(0.0 < v <= 1.0 for v in out)


def test_write_strengths_file(tmp_path: Path) -> None:
    path = write_strengths_file(tmp_path / "s.txt", [0.8, 0.5, 0.35])
    text = path.read_text(encoding="utf-8").strip()
    assert text == "0.800000,0.500000,0.350000"


def test_build_faces_window_request_soft_hold(tmp_path: Path) -> None:
    init = tmp_path / "init.mp4"
    init.write_bytes(b"x")
    strengths_path = write_strengths_file(tmp_path / "s.txt", [0.8, 0.5])
    req = build_faces_window_request(
        prompt="Picture 1",
        output_path=tmp_path / "out.mp4",
        init_video=init,
        strengths_file=strengths_path,
        canvas=512,
        num_frames=22,
        ceiling=0.45,
        seed=7,
        face_mode="t2va",
        refs_list=[],
        loras_list=[],
        quality="close",
        steps=8,
        layers=50,
        reuse=1,
        preview_latent_dir=None,
    )
    assert isinstance(req, GenerateRequest)
    assert req.init_video == init
    assert req.denoise_strength == 0.45
    assert req.denoise_video_strengths_file == strengths_path
    assert req.use_slower_bf16_mlp is True
    assert req.int8_row_fc2 is False
    assert req.token_reduction is False
    assert req.first_frame is None
    assert req.last_frame is None
    assert req.width == req.height == 512


def test_faces_pass_emits_strengths_file(tmp_path: Path) -> None:
    from h3_facepass import write_rgb_mp4

    frames = 22
    rgb = np.zeros((frames, 64, 64, 3), dtype=np.uint8)
    rgb[:, 20:44, 20:44] = 220
    src = tmp_path / "src.mp4"
    write_rgb_mp4(rgb, src)
    out = tmp_path / "out.mp4"
    box = (20.0, 20.0, 24.0, 24.0)
    captured: list[GenerateRequest] = []

    class FakeEngine:
        def generate(self, req: GenerateRequest, *, on_progress=None) -> dict:
            captured.append(req)
            assert req.denoise_video_strengths_file is not None
            assert Path(req.denoise_video_strengths_file).is_file()
            text = Path(req.denoise_video_strengths_file).read_text().strip()
            vals = [float(x) for x in text.split(",") if x]
            assert len(vals) == video_latent_t(int(req.num_frames))
            assert abs(float(req.denoise_strength or 0) - 0.45) < 1e-6
            copy2(req.init_video, req.output_path)
            return {"ok": True}

    import h3_facepass as fp

    original = fp.track_faces_in_clip
    original_ready = fp.sam3_ready
    fp.track_faces_in_clip = lambda frames, **kw: (  # type: ignore[assignment]
        [box] * len(frames),
        [True] * len(frames),
    )
    fp.sam3_ready = lambda model_dir=None: True  # type: ignore[assignment]
    try:
        run_faces_pass(
            src,
            out,
            prompt="x",
            engine=FakeEngine(),
            settings=FacesSettings(canvas=384, denoise=0.45, seed=7, abstain=False),
            confirmed_boxes=[box] * frames,
            quality="close",
        )
    finally:
        fp.track_faces_in_clip = original  # type: ignore[assignment]
        fp.sam3_ready = original_ready  # type: ignore[assignment]

    assert captured
    assert all(r.denoise_video_strengths_file is not None for r in captured)
    assert all(r.use_slower_bf16_mlp for r in captured)
