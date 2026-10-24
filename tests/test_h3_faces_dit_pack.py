"""Faces soft-hold must use DiT-aligned encode T, not raw encoder T.

Continuity/Comfy encode_temporal: 17-frame clips, token_drop=3 → video_t.
Raw h3 encoder (n+3)/4 then linear resample smears soft-hold (mashup).
"""

from __future__ import annotations


def video_latent_t(frame_count: int) -> int:
    n = int(frame_count)
    if n <= 5:
        return 2
    return ((n - 5) // 17) * 5 + 2


def encoder_latent_t(frame_count: int) -> int:
    return (int(frame_count) + 3) // 4 if frame_count > 0 else 0


def dit_encode_latent_t(frame_count: int) -> int:
    """Mirror h3_video_dit_encode_latent_t / Comfy encode_temporal."""
    if frame_count < 1:
        return 0
    clip_length = 17
    tokens_per_clip = 5
    token_drop = 3
    chunks = (frame_count + clip_length - 1) // clip_length
    tokens = chunks * tokens_per_clip
    return tokens if tokens <= token_drop else tokens - token_drop


def test_dit_pack_matches_video_t_on_legal_windows() -> None:
    for n in (5, 22, 39, 56, 73, 90, 107, 243, 362):
        assert dit_encode_latent_t(n) == video_latent_t(n), (n, dit_encode_latent_t(n), video_latent_t(n))


def test_raw_encoder_mismatches_dit_on_typical_faces_window() -> None:
    assert encoder_latent_t(22) == 6
    assert video_latent_t(22) == 7
    assert dit_encode_latent_t(22) == 7


def test_faces_request_still_sets_strengths_file(tmp_path) -> None:
    """Guard: Faces windows must keep denoise_video_strengths_file so h3.c
    takes the dit_temporal branch (not encode+resample)."""
    from pathlib import Path

    from h3_facepass import build_faces_window_request
    from h3_faces import write_strengths_file

    strengths = write_strengths_file(tmp_path / "strengths.txt", [0.8, 0.5])
    req = build_faces_window_request(
        prompt="x",
        output_path=tmp_path / "out.mp4",
        init_video=tmp_path / "init.mp4",
        strengths_file=strengths,
        canvas=512,
        num_frames=22,
        ceiling=0.45,
        seed=1,
        face_mode="t2va",
        refs_list=[],
        loras_list=[],
        quality="close",
        steps=8,
        layers=50,
        reuse=1,
        preview_latent_dir=None,
    )
    assert req.denoise_video_strengths_file == strengths
    assert req.init_video is not None
    assert float(req.denoise_strength or 0) == 0.45
