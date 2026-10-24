"""Chip-class auto-select: M3 Ultra must not inherit M5-only force paths incorrectly."""

from __future__ import annotations

from pathlib import Path

from h3_backend import (
    GenerateRequest,
    apple_silicon_class,
    apply_init_video_defaults,
    apply_platform_generate_defaults,
    m5_close_path_env,
    metal4_gpu,
)


def _req(**kwargs) -> GenerateRequest:
    base = dict(
        prompt="x",
        output_path=Path("/tmp/out.mp4"),
        width=512,
        height=512,
        num_frames=22,
        quality="fast",
    )
    base.update(kwargs)
    return GenerateRequest(**base)  # type: ignore[arg-type]


def test_apple_silicon_class_from_brand_string() -> None:
    assert apple_silicon_class("Apple M3 Ultra") == "m3"
    assert apple_silicon_class("Apple M4 Max") == "m3"
    assert apple_silicon_class("Apple M5") == "m5"
    assert apple_silicon_class("Apple M5 Pro") == "m5"
    assert apple_silicon_class("Intel Core") == "other"
    assert metal4_gpu("Apple M5") is True
    assert metal4_gpu("Apple M3 Ultra") is False


def test_m3_platform_defaults_noop_without_init_video() -> None:
    """M3/M4: do not force BF16 flags on a normal generate (already BF16-class)."""
    req = _req(int8_row_fc2=None, use_slower_bf16_mlp=False)
    apply_platform_generate_defaults(req, chip_class="m3", explicit_int8_row_fc2=None)
    assert req.use_slower_bf16_mlp is False
    assert req.int8_row_fc2 is None or req.int8_row_fc2 is False


def test_m5_platform_defaults_force_bf16_without_init_video() -> None:
    req = _req(int8_row_fc2=None, use_slower_bf16_mlp=False)
    apply_platform_generate_defaults(req, chip_class="m5", explicit_int8_row_fc2=None)
    assert req.use_slower_bf16_mlp is True
    assert req.use_slower_bf16_qkv is True
    assert req.use_slower_bf16_attention_output is True
    assert req.int8_row_fc2 is False


def test_m5_explicit_int8_speed_path_skips_bf16_force() -> None:
    req = _req(int8_row_fc2=True, use_slower_bf16_mlp=False)
    apply_platform_generate_defaults(req, chip_class="m5", explicit_int8_row_fc2=True)
    assert req.use_slower_bf16_mlp is False


def test_init_video_defaults_chip_agnostic() -> None:
    """Native canvas + BF16 for init-video on every chip (Faces / long-take / redetail)."""
    for klass in ("m3", "m5", "other"):
        req = _req(
            init_video=Path("/tmp/init.mp4"),
            quality="fast",
            width=512,
            height=512,
            use_slower_bf16_mlp=False,
            int8_row_fc2=None,
        )
        apply_platform_generate_defaults(req, chip_class=klass, explicit_int8_row_fc2=None)
        assert req.render_width == 512
        assert req.render_height == 512
        assert req.token_reduction is False
        assert req.use_slower_bf16_mlp is True
        assert req.int8_row_fc2 is False


def test_apply_init_video_defaults_alone() -> None:
    req = _req(init_video=Path("/tmp/init.mp4"), width=640, height=640)
    apply_init_video_defaults(req)
    assert req.render_width == 640
    assert req.render_height == 640


def test_m5_close_path_env_only_when_bf16() -> None:
    env = {"PATH": "/usr/bin"}
    cold = _req(use_slower_bf16_mlp=False)
    assert "H3_CPU_SAMPLER" not in m5_close_path_env(cold, env)
    hot = _req(use_slower_bf16_mlp=True)
    assert m5_close_path_env(hot, env).get("H3_CPU_SAMPLER") == "1"
