"""H3 canvas / duration snap helpers and PyAV media I/O.

All Python-side media (concat, last-frame, duration, and the h3.c ffmpeg CLI
shim in ``h3_av.py``) goes through PyAV. h3.c posix_spawns ``scripts/h3-av``
via ``H3_AV`` instead of a system ffmpeg install.
"""

from __future__ import annotations

import logging
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

log = logging.getLogger("h3")

FPS = 24
SPATIAL_ALIGN = 32
MIN_SPATIAL = 32
# One-pass / UI presets: H3-Base 768p-class training envelope.
ONE_PASS_MAX_PIXELS = 768 * 1344
# Engine + ReDetail refine may go to 2× of the largest one-pass canvas.
MAX_PIXELS = 1536 * 2688
MIN_FRAMES = 22  # 5 + 17*1


def snap_frames(raw: int) -> int:
    """Snap *up* to the next legal H3 temporal shape ``5 + 17n`` (n ≥ 1)."""
    n = max(1, int(raw))
    if n <= MIN_FRAMES:
        return MIN_FRAMES
    k = math.ceil((n - 5) / 17)
    k = max(1, k)
    return 5 + 17 * k


def frames_to_seconds(num_frames: int, fps: int = FPS) -> float:
    return float(num_frames) / float(fps)


# Canonical rounded lengths keep the historical H3-friendly frame counts;
# every other integer second (1–15) snaps up via ``5+17n``.
_UI_DURATION_CANONICAL: dict[int, int] = {
    1: 22,
    2: 56,
    5: 107,
    10: 243,
    15: 362,
}


def _frames_for_ui_seconds(rounded_s: int, fps: int = FPS) -> int:
    s = max(1, min(15, int(rounded_s)))
    if s in _UI_DURATION_CANONICAL:
        return _UI_DURATION_CANONICAL[s]
    return snap_frames(int(math.ceil(s * fps)))


# Full clip-length ladder shown in the Web UI stepper / picker.
UI_DURATION_FRAMES: dict[int, int] = {s: _frames_for_ui_seconds(s) for s in range(1, 16)}


def seconds_to_frames(seconds: float, fps: int = FPS) -> int:
    sec = float(seconds)
    for rounded, frames in UI_DURATION_FRAMES.items():
        actual = frames_to_seconds(frames, fps=fps)
        if abs(sec - rounded) < 0.05 or abs(sec - actual) < 0.05:
            return frames
    raw = int(math.ceil(sec * fps))
    return snap_frames(raw)


def snap_spatial(n: int, align: int = SPATIAL_ALIGN) -> int:
    n = int(n)
    if n < MIN_SPATIAL:
        return MIN_SPATIAL
    return max(MIN_SPATIAL, int(round(n / align) * align))


def validate_canvas(width: int, height: int) -> tuple[int, int]:
    w = snap_spatial(width)
    h = snap_spatial(height)
    if w * h > MAX_PIXELS:
        raise ValueError(
            f"canvas {w}×{h} ({w * h} px) exceeds H3 limit {MAX_PIXELS} "
            f"(1536×2688)"
        )
    return w, h


def redetail_output_size(width: int, height: int) -> tuple[int, int]:
    """2× spatial canvas for ReDetail refine, snapped to ×32 under MAX_PIXELS."""
    w = snap_spatial(int(width) * 2)
    h = snap_spatial(int(height) * 2)
    return validate_canvas(w, h)


def duration_preset(
    rounded_s: int, *, num_frames: int, note: str = ""
) -> dict[str, Any]:
    nf = snap_frames(int(num_frames))
    label = f"{rounded_s}s ({nf} frames @ {FPS} fps)"
    if note:
        label = f"{label} — {note}"
    return {
        "id": f"{rounded_s}s",
        "seconds": float(rounded_s),
        "num_frames": nf,
        "label": label,
    }


DURATION_PRESETS = [
    duration_preset(
        s,
        num_frames=UI_DURATION_FRAMES[s],
        note="dev" if s == 1 else "",
    )
    for s in range(1, 16)
]


def _canvas(
    preset_id: str,
    width: int,
    height: int,
    *,
    aspect: str,
    group: str,
    label: str,
    guidance: str,
    render: tuple[int, int] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": preset_id,
        "width": width,
        "height": height,
        "aspect": aspect,
        "group": group,
        "label": label,
        "guidance": guidance,
    }
    if render is not None:
        item["render_width"], item["render_height"] = render
    return item


# Output canvases: each side ×32, product ≤ 768×1344. Exact 16:9 at 768 short
# side is 1365×768 and does not fit; 1024×576 is the exact 16:9 under the cap,
# 1248×704 is the largest near-16:9. Same for 9:16. Internal render sizes are
# same-aspect DiT/VAE canvases that vImage-upscale (512 square only).
RESOLUTION_PRESETS = [
    _canvas(
        "256x256",
        256,
        256,
        aspect="1:1",
        group="Square",
        label="1:1 · 256 × 256 — preview",
        guidance="Native 8×8 token grid with automatic low-resolution RoPE. Keep token reduction off.",
    ),
    _canvas(
        "512x512",
        512,
        512,
        aspect="1:1",
        group="Square",
        label="1:1 · 512 × 512 — default",
        guidance="Balanced default resolution.",
    ),
    _canvas(
        "512x512-fast",
        512,
        512,
        aspect="1:1",
        group="Square",
        label="1:1 · 512 × 512 · 384 internal",
        guidance="Validated fast-quality scaling point: DiT/VAE at 384, output 512.",
        render=(384, 384),
    ),
    _canvas(
        "512x512-aggressive",
        512,
        512,
        aspect="1:1",
        group="Square",
        label="1:1 · 512 × 512 · 320 internal",
        guidance="Validated aggressive scaling point. Do not add token-reduction with layers 40 + reuse 3.",
        render=(320, 320),
    ),
    _canvas(
        "768x768",
        768,
        768,
        aspect="1:1",
        group="Square",
        label="1:1 · 768 × 768 — 768p",
        guidance="Validated close-quality square; substantially more expensive.",
    ),
    _canvas(
        "1024x576",
        1024,
        576,
        aspect="16:9",
        group="Landscape",
        label="16:9 · 1024 × 576 — YouTube / landscape",
        guidance="Exact 16:9. Largest exact 16:9 under the ×32 / 768×1344 cap (next step 1536×864 overflows).",
    ),
    _canvas(
        "1248x704",
        1248,
        704,
        aspect="16:9",
        group="Landscape",
        label="16:9 · 1248 × 704 — largest",
        guidance="Closest 16:9 under the pixel cap (1248/704 ≈ 1.773 vs 1.778). True 768p 16:9 would be 1365×768.",
    ),
    _canvas(
        "1344x768",
        1344,
        768,
        aspect="7:4",
        group="Landscape",
        label="7:4 · 1344 × 768 — max landscape",
        guidance="Released 768p-class landscape pixel-cap. 7:4, not 16:9.",
    ),
    _canvas(
        "1024x768",
        1024,
        768,
        aspect="4:3",
        group="Landscape",
        label="4:3 · 1024 × 768",
        guidance="Exact 4:3 768p-class canvas.",
    ),
    _canvas(
        "960x768",
        960,
        768,
        aspect="5:4",
        group="Landscape",
        label="5:4 · 960 × 768",
        guidance="Exact 5:4 (landscape complement of 4:5 social).",
    ),
    _canvas(
        "768x960",
        768,
        960,
        aspect="4:5",
        group="Portrait",
        label="4:5 · 768 × 960 — Instagram / social",
        guidance="Exact 4:5 with 768 short side. Feed-style portrait (not Stories/Reels 9:16).",
    ),
    _canvas(
        "768x1024",
        768,
        1024,
        aspect="3:4",
        group="Portrait",
        label="3:4 · 768 × 1024",
        guidance="Exact 3:4 768p-class canvas.",
    ),
    _canvas(
        "576x1024",
        576,
        1024,
        aspect="9:16",
        group="Portrait",
        label="9:16 · 576 × 1024 — Stories / Reels / TikTok",
        guidance="Exact 9:16. Largest exact 9:16 under the cap (next step 864×1536 overflows).",
    ),
    _canvas(
        "704x1248",
        704,
        1248,
        aspect="9:16",
        group="Portrait",
        label="9:16 · 704 × 1248 — largest",
        guidance="Closest 9:16 under the pixel cap. True 768p 9:16 would be 768×1365.",
    ),
    _canvas(
        "768x1344",
        768,
        1344,
        aspect="4:7",
        group="Portrait",
        label="4:7 · 768 × 1344 — max portrait",
        guidance="Released 768p-class portrait pixel-cap. 4:7, not 9:16.",
    ),
]

ALLOWED_OUTPUT_SIZES = frozenset(
    (int(p["width"]), int(p["height"])) for p in RESOLUTION_PRESETS
)

# Continuity H3 ``declare.RULES`` short-edge envelope (UI resolveCanvas).
NATIVE_SHORT_EDGE = 768
MIN_SHORT_EDGE = 384  # Continuity H3 declare.RULES
MAX_SHORT_EDGE = 2048
MIN_RATIO = 9 / 21
MAX_RATIO = 21 / 9
H3_ASPECTS: dict[str, float] = {
    "16:9": 16 / 9,
    "21:9": 21 / 9,
    "3:2": 3 / 2,
    "4:3": 4 / 3,
    "5:4": 5 / 4,
    "1:1": 1.0,
    "4:5": 4 / 5,
    "3:4": 3 / 4,
    "2:3": 2 / 3,
    "9:16": 9 / 16,
    "9:21": 9 / 21,
}


def clamp_ratio(ratio: float) -> tuple[float, bool]:
    r = float(ratio)
    if r < MIN_RATIO:
        return MIN_RATIO, True
    if r > MAX_RATIO:
        return MAX_RATIO, True
    return r, False


def resolve_canvas(ratio: float, short_edge: int) -> tuple[int, int]:
    """Continuity ``resolve_canvas`` — aspect + short edge → WxH (×32)."""
    clamped, _ = clamp_ratio(ratio)
    edge = max(MIN_SHORT_EDGE, min(MAX_SHORT_EDGE, int(short_edge)))
    max_pixels = ONE_PASS_MAX_PIXELS * (edge / float(NATIVE_SHORT_EDGE)) ** 2
    if clamped >= 1.0:
        width, height = edge * clamped, float(edge)
    else:
        width, height = float(edge), edge / clamped
    if width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        width, height = width * scale, height * scale
    width = float(snap_spatial(width))
    height = float(snap_spatial(height))
    while width * height > max_pixels and max(width, height) > SPATIAL_ALIGN:
        if width >= height:
            width -= SPATIAL_ALIGN
        else:
            height -= SPATIAL_ALIGN
    return int(width), int(height)


def require_ui_canvas(width: int, height: int) -> tuple[int, int]:
    """Accept Continuity free short-edge canvases (×32, ≤ engine MAX_PIXELS).

    Tagged ``RESOLUTION_PRESETS`` remain the documented defaults / migration
    targets; the composer short-edge slider is no longer locked to that list.
    """
    return validate_canvas(width, height)


def sanitize_filename(prompt: str, maxlen: int = 48) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", (prompt or "").strip())[:maxlen].strip("_")
    return slug.lower() or "clip"


def media_available() -> bool:
    try:
        import av  # noqa: F401

        return True
    except ImportError:
        return False


MIN_REF_AUDIO_S = 2.0
MAX_REF_AUDIO_S = 15.0
MAX_REF_AUDIO_TOTAL_S = 15.0
MAX_REF_AUDIO_CLIPS = 3
_DURATION_SLACK_S = 0.05


def media_shim_ok() -> tuple[bool, str]:
    """PyAV + h3-av shim (the tools h3.c will posix_spawn)."""
    from h3_paths import default_h3_av

    if not media_available():
        return False, "PyAV is required (pip install av) — h3.c media goes through h3-av"
    shim = default_h3_av()
    if not shim.is_file():
        return False, f"h3-av shim not found at {shim}"
    return True, str(shim)


def ffmpeg_ok() -> tuple[bool, str]:
    """Backward-compatible alias: the media stack is the PyAV shim."""
    return media_shim_ok()


def _make_audio_resampler(*, sample_fmt: str, layout: str = "stereo", rate: int = 48000):
    """PyAV ≥16 dropped ``AudioFrame.reformat`` — use AudioResampler instead."""
    from av.audio.resampler import AudioResampler

    return AudioResampler(format=sample_fmt, layout=layout, rate=rate)


def _sane_video_fps(stream: Any, default: float = float(FPS)) -> float:
    """Clamp container-reported fps into a range libx264 / h3.c can ingest.

    Broken or missing ``average_rate`` (or a prior encode that collapsed all
    frame PTS to 0) can report absurd rates like 6840. h3.c then applies
    ``fps=24`` and keeps only one frame → "at least 5 decoded frames".
    """
    rate = getattr(stream, "average_rate", None) or getattr(stream, "base_rate", None)
    try:
        fps = float(rate) if rate is not None else float(default)
    except (TypeError, ValueError, ZeroDivisionError):
        fps = float(default)
    if not math.isfinite(fps) or fps < 1.0 or fps > 120.0:
        return float(default)
    return fps


def _video_time_base(fps: float) -> Fraction:
    """``1/round(fps)`` stream/codec time base for fixed-fps encodes (PyAV Time docs)."""
    return Fraction(1, max(1, int(round(float(fps)))))


def _stamp_encode_video_pts(frame: Any, pts: int, fps: float) -> None:
    """Set frame PTS in encode time_base per https://pyav.org/docs/stable/api/time.html

    Decoded frames keep the *source* ``time_base``. Writing ``pts=0,1,2…``
    without resetting ``time_base`` (or clearing pts while leaving the source
    base) collapses the timeline so ffmpeg ``fps=24`` keeps ~1 frame.
    For fixed-fps content, PTS is the frame index in ``1/fps``.
    """
    tb = _video_time_base(fps)
    frame.pts = int(pts)
    frame.time_base = tb


def _copy_video_color_tags(dst_stream: Any, src_stream: Any | None) -> None:
    """Copy non-unspecified color tags so re-encoded refs keep BT.709 / MPEG range."""
    if src_stream is None:
        return
    src_ctx = getattr(src_stream, "codec_context", None)
    dst_ctx = getattr(dst_stream, "codec_context", None)
    if src_ctx is None or dst_ctx is None:
        return
    for attr in ("color_range", "colorspace", "color_primaries", "color_trc"):
        try:
            val = getattr(src_ctx, attr, None)
            if val is None:
                continue
            if int(val) == 0:  # UNSPECIFIED
                continue
            setattr(dst_ctx, attr, val)
        except (TypeError, ValueError, AttributeError):
            continue


def _add_h264_stream(
    out: Any,
    *,
    fps: float,
    width: int,
    height: int,
    preset: str = "veryfast",
    crf: str = "18",
    color_from: Any | None = None,
) -> Any:
    """libx264 stream with rate + time_base hint matching PyAV encoding guidance."""
    rate = max(1, int(round(float(fps))))
    stream = out.add_stream("libx264", rate=rate)
    stream.width = int(width)
    stream.height = int(height)
    stream.pix_fmt = "yuv420p"
    stream.options = {"preset": preset, "crf": crf}
    stream.time_base = _video_time_base(rate)
    _copy_video_color_tags(stream, color_from)
    return stream


def _frame_for_h264_encode(frame: Any, *, fps: float, pts: int) -> Any:
    """Prepare a decoded frame for libx264 without scrubbing color range.

    Already-``yuv420p`` frames are left in place (only PTS retagged). Other
    formats convert with ``src_color_range``/``dst_color_range`` preserved so
    swscale does not treat limited-range BT.709 as full-range (see PyAV
    ColorRange / issue #2149).
    """
    fmt = getattr(getattr(frame, "format", None), "name", None)
    if fmt != "yuv420p":
        kwargs: dict[str, Any] = {"format": "yuv420p"}
        src_range = getattr(frame, "color_range", None)
        try:
            if src_range is not None and int(src_range) != 0:
                kwargs["src_color_range"] = src_range
                kwargs["dst_color_range"] = src_range
        except (TypeError, ValueError):
            pass
        frame = frame.reformat(**kwargs)
    _stamp_encode_video_pts(frame, pts, fps)
    return frame


def _mux_audio_frame(out, stream, resampler, frame) -> None:
    """Resample one audio frame (or flush with ``frame=None``) and mux packets."""
    for converted in resampler.resample(frame):
        converted.pts = None
        for packet in stream.encode(converted):
            out.mux(packet)


def _flush_audio_encoder(out, stream, resampler) -> None:
    _mux_audio_frame(out, stream, resampler, None)
    for packet in stream.encode(None):
        out.mux(packet)


def probe_duration_seconds(path: Path | str) -> float | None:
    """Duration in seconds via PyAV. No ffprobe CLI."""
    info = probe_media(path)
    return info.get("duration_s")


def probe_media(path: Path | str) -> dict[str, Any]:
    """Lightweight PyAV probe: streams + duration. Never shells out to ffprobe."""
    src = Path(path)
    empty: dict[str, Any] = {
        "has_audio": False,
        "has_video": False,
        "duration_s": None,
        "audio_rate": None,
        "audio_channels": None,
        "audio_codec": None,
    }
    if not src.is_file() or not media_available():
        return empty
    try:
        import av

        container = av.open(str(src))
        try:
            has_audio = any(s.type == "audio" for s in container.streams)
            has_video = any(s.type == "video" for s in container.streams)
            duration_s: float | None = None
            if container.duration and container.duration > 0:
                duration_s = float(container.duration) / float(av.time_base)
            else:
                best = 0.0
                for stream in container.streams:
                    if stream.duration and stream.time_base and stream.duration > 0:
                        seconds = float(stream.duration * stream.time_base)
                        if seconds > best:
                            best = seconds
                duration_s = best or None
            audio_rate = audio_channels = None
            audio_codec = None
            a_stream = next((s for s in container.streams if s.type == "audio"), None)
            if a_stream is not None:
                ctx = a_stream.codec_context
                audio_codec = getattr(ctx, "name", None) if ctx is not None else None
                audio_rate = int(a_stream.rate) if a_stream.rate else None
                layout = getattr(a_stream, "layout", None)
                if layout is not None and getattr(layout, "nb_channels", None):
                    audio_channels = int(layout.nb_channels)
                elif ctx is not None and getattr(ctx, "channels", None):
                    audio_channels = int(ctx.channels)
            return {
                "has_audio": has_audio,
                "has_video": has_video,
                "duration_s": duration_s,
                "audio_rate": audio_rate,
                "audio_channels": audio_channels,
                "audio_codec": audio_codec,
            }
        finally:
            container.close()
    except Exception:
        return empty


def audio_is_h3_ready(path: Path | str) -> bool:
    """True when the file already has a stereo soundtrack PyAV/h3-av can ingest."""
    info = probe_media(path)
    if not info["has_audio"]:
        return False
    channels = info.get("audio_channels")
    rate = info.get("audio_rate")
    codec = (info.get("audio_codec") or "").lower()
    # H3 ultimately wants stereo @ 32 kHz f32; h3-av resamples, so any stereo
    # PCM/AAC/MP3 we can open is fine. Reject mono / multi / unknown layout so
    # adapt_audio_for_h3 normalizes them up front.
    if channels != 2:
        return False
    if rate is None or int(rate) < 16000:
        return False
    if codec in ("aac", "mp3", "pcm_s16le", "pcm_f32le", "flac", "opus", "vorbis"):
        return True
    # Unknown but stereo — still rewrite so decode path is deterministic.
    return False


def adapt_audio_for_h3(src: Path | str, dest: Path | str) -> Path:
    """Rewrite media so H3 ingestion always gets a stereo soundtrack via PyAV.

    * Video + audio → stream-copy video, re-encode audio as stereo AAC @ 48 kHz.
    * Audio-only → stereo PCM WAV @ 48 kHz (no AAC encoder dependency).
    * Video without audio → copy through unchanged (caller should use silent_video).

    Never shells out to ffmpeg — PyAV only.
    """
    import shutil

    import av
    from av.audio.frame import AudioFrame
    from av.video.frame import VideoFrame

    source = Path(src)
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(f"media not found: {source}")

    info = probe_media(source)
    if not info["has_audio"] and not info["has_video"]:
        raise ValueError(f"no audio/video streams in {source}")

    # Silent video: nothing to adapt — keep bytes identical.
    if info["has_video"] and not info["has_audio"]:
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return target

    # Already a stereo soundtrack we can open — skip the re-encode.
    if audio_is_h3_ready(source):
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        return target

    inp = av.open(str(source))
    try:
        has_video = any(s.type == "video" for s in inp.streams)
        if has_video:
            if target.suffix.lower() not in _VIDEO_EXTS_ADAPT:
                target = target.with_suffix(".mp4")
            out = av.open(str(target), mode="w")
        else:
            if target.suffix.lower() not in (".wav", ".mp3", ".m4a", ".aac"):
                target = target.with_suffix(".wav")
            out = av.open(str(target), mode="w", format="wav")
        try:
            v_out = None
            a_out = None
            a_resampler = None
            v_pts = 0
            fps = float(FPS)
            v_in = next((s for s in inp.streams if s.type == "video"), None)
            if v_in is not None:
                # Re-encode video (stream-copy + separate audio re-encode is
                # fragile across containers); keep veryfast/CRF so refs stay light.
                fps = _sane_video_fps(v_in)
                v_out = _add_h264_stream(
                    out,
                    fps=fps,
                    width=int(v_in.codec_context.width or 0),
                    height=int(v_in.codec_context.height or 0),
                    color_from=v_in,
                )
            if has_video:
                a_out = out.add_stream("aac", rate=48000)
                a_out.layout = "stereo"
                a_out.time_base = Fraction(1, 48000)
                a_resampler = _make_audio_resampler(
                    sample_fmt="fltp", layout="stereo", rate=48000
                )
            else:
                a_out = out.add_stream("pcm_s16le", rate=48000)
                a_out.layout = "stereo"
                a_out.time_base = Fraction(1, 48000)
                a_resampler = _make_audio_resampler(
                    sample_fmt="s16", layout="stereo", rate=48000
                )

            for frame in inp.decode():
                if isinstance(frame, VideoFrame) and v_out is not None:
                    frame = _frame_for_h264_encode(frame, fps=fps, pts=v_pts)
                    v_pts += 1
                    for packet in v_out.encode(frame):
                        out.mux(packet)
                elif isinstance(frame, AudioFrame) and a_out is not None and a_resampler is not None:
                    _mux_audio_frame(out, a_out, a_resampler, frame)

            if v_out is not None:
                for packet in v_out.encode():
                    out.mux(packet)
            if a_out is not None and a_resampler is not None:
                _flush_audio_encoder(out, a_out, a_resampler)
        finally:
            out.close()
    finally:
        inp.close()
    return target


# Suffixes we keep when adapting a video reference for H3.
_VIDEO_EXTS_ADAPT = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}


def resize_still_to_canvas(src: Path | str, dest: Path | str, width: int, height: int) -> Path:
    """Resize an image still to the output canvas. Images only."""
    import av

    source = Path(src)
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    inp = av.open(str(source))
    try:
        frame = next(inp.decode(video=0))
    finally:
        inp.close()
    frame = frame.reformat(width=int(width), height=int(height), format="rgb24")
    out = av.open(str(target), mode="w", format="image2")
    try:
        stream = out.add_stream("png", rate=1)
        stream.width = int(width)
        stream.height = int(height)
        stream.pix_fmt = "rgb24"
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    finally:
        out.close()
    return target


def crop_still(
    src: Path | str,
    dest: Path | str,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
) -> Path:
    """Crop an image using a normalized box in [0, 1]. Writes PNG.

    Prefer ``h3_crop.apply_still`` for Continuity framing (turn/mirror/window).
    """
    from h3_crop import Crop, apply_still

    return apply_still(src, dest, Crop(x=float(x), y=float(y), w=float(w), h=float(h)))


def frame_video(src: Path | str, dest: Path | str, crop: Any) -> Path:
    """Bake Continuity framing (turn/mirror/window) into every video frame."""
    import av
    import numpy as np
    from av.audio.frame import AudioFrame
    from av.video.frame import VideoFrame
    from PIL import Image

    from h3_crop import pil as crop_pil

    source = Path(src)
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    inp = av.open(str(source))
    try:
        v_in = next((s for s in inp.streams if s.type == "video"), None)
        if v_in is None:
            raise ValueError(f"no video stream in {source}")
        fps = _sane_video_fps(v_in)
        out = av.open(str(target), mode="w")
        try:
            v_out = None
            a_out = None
            a_resampler = None
            v_pts = 0
            if any(s.type == "audio" for s in inp.streams):
                a_out = out.add_stream("aac", rate=48000)
                a_out.layout = "stereo"
                a_out.time_base = Fraction(1, 48000)
                a_resampler = _make_audio_resampler(sample_fmt="fltp", layout="stereo", rate=48000)
            for frame in inp.decode():
                if isinstance(frame, VideoFrame):
                    img = Image.fromarray(frame.to_ndarray(format="rgb24"))
                    framed = crop_pil(img, crop)
                    arr = np.asarray(framed.convert("RGB"))
                    h, w = arr.shape[:2]
                    # yuv420p needs even dims
                    ew, eh = w - (w % 2), h - (h % 2)
                    if (ew, eh) != (w, h):
                        framed = framed.resize((max(2, ew), max(2, eh)), Image.Resampling.LANCZOS)
                        arr = np.asarray(framed)
                        h, w = arr.shape[:2]
                    if v_out is None:
                        v_out = _add_h264_stream(out, fps=fps, width=w, height=h)
                    new_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                    new_frame = new_frame.reformat(format="yuv420p")
                    _stamp_encode_video_pts(new_frame, v_pts, fps)
                    v_pts += 1
                    for packet in v_out.encode(new_frame):
                        out.mux(packet)
                elif isinstance(frame, AudioFrame) and a_out is not None and a_resampler is not None:
                    _mux_audio_frame(out, a_out, a_resampler, frame)
            if v_out is not None:
                for packet in v_out.encode():
                    out.mux(packet)
            if a_out is not None and a_resampler is not None:
                _flush_audio_encoder(out, a_out, a_resampler)
        finally:
            out.close()
    finally:
        inp.close()
    return target


def audio_peaks(src: Path | str, *, buckets: int = 400) -> dict[str, Any]:
    """Peak envelope for the Continuity trim waveform, plus duration / has_audio."""
    import av
    import numpy as np

    source = Path(src)
    if not source.is_file():
        return {"peaks": [], "duration": None, "has_audio": False}
    container = av.open(str(source))
    try:
        has_video = any(s.type == "video" for s in container.streams)
        has_audio = any(s.type == "audio" for s in container.streams)
        duration = None
        if container.duration and container.duration > 0:
            duration = float(container.duration) / float(av.time_base)
        peaks: list[float] = []
        if has_audio:
            samples: list[np.ndarray] = []
            for frame in container.decode(audio=0):
                arr = frame.to_ndarray()
                if arr.ndim > 1:
                    arr = arr.mean(axis=0)
                samples.append(np.asarray(arr, dtype=np.float32).ravel())
            if samples:
                wave = np.concatenate(samples)
                # Normalize to [-1, 1] peak envelope.
                if wave.dtype != np.float32:
                    wave = wave.astype(np.float32)
                if np.issubdtype(wave.dtype, np.integer):
                    wave = wave / np.iinfo(wave.dtype).max
                n = max(1, int(buckets))
                chunk = max(1, len(wave) // n)
                for i in range(n):
                    part = wave[i * chunk : (i + 1) * chunk]
                    peaks.append(float(np.max(np.abs(part))) if len(part) else 0.0)
                peak_max = max(peaks) or 1.0
                peaks = [min(1.0, p / peak_max) for p in peaks]
        return {"peaks": peaks, "duration": duration, "has_audio": has_audio, "has_video": has_video}
    finally:
        container.close()


def trim_media(
    src: Path | str,
    dest: Path | str,
    *,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> Path:
    """Trim audio and/or video to ``[start_s, end_s)``. Re-encodes for keyframe safety."""
    import av
    from av.audio.frame import AudioFrame
    from av.video.frame import VideoFrame

    source = Path(src)
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    start = max(0.0, float(start_s))
    end = float(end_s) if end_s is not None else None
    if end is not None and end <= start:
        raise ValueError("trim end must be greater than trim start")

    inp = av.open(str(source))
    try:
        has_video = any(s.type == "video" for s in inp.streams)
        has_audio = any(s.type == "audio" for s in inp.streams)
        if not has_video and not has_audio:
            raise ValueError(f"no audio/video streams in {source}")

        try:
            inp.seek(int(start * av.time_base))
        except Exception:
            pass

        if has_video:
            out = av.open(str(target), mode="w")
        else:
            # wav keeps trim simple without an AAC encoder dependency.
            if target.suffix.lower() not in (".wav", ".mp3", ".m4a"):
                target = target.with_suffix(".wav")
            out = av.open(str(target), mode="w", format="wav")
        try:
            v_out = None
            a_out = None
            a_resampler = None
            v_pts = 0
            fps = float(FPS)
            if has_video:
                v_in = next(s for s in inp.streams if s.type == "video")
                fps = _sane_video_fps(v_in)
                v_out = _add_h264_stream(
                    out,
                    fps=fps,
                    width=int(v_in.codec_context.width or 0),
                    height=int(v_in.codec_context.height or 0),
                    color_from=v_in,
                )
            if has_audio:
                if has_video:
                    a_out = out.add_stream("aac", rate=48000)
                    a_out.layout = "stereo"
                    a_out.time_base = Fraction(1, 48000)
                    a_resampler = _make_audio_resampler(
                        sample_fmt="fltp", layout="stereo", rate=48000
                    )
                else:
                    a_out = out.add_stream("pcm_s16le", rate=48000)
                    a_out.layout = "stereo"
                    a_out.time_base = Fraction(1, 48000)
                    a_resampler = _make_audio_resampler(
                        sample_fmt="s16", layout="stereo", rate=48000
                    )

            for frame in inp.decode():
                pts_s = float(frame.time) if frame.time is not None else 0.0
                if pts_s + 1e-3 < start:
                    continue
                if end is not None and pts_s >= end - 1e-6:
                    if isinstance(frame, VideoFrame):
                        break
                    continue
                if isinstance(frame, VideoFrame) and v_out is not None:
                    # PyAV Time docs: encode PTS in 1/fps; keep yuv420p pixels / color tags.
                    frame = _frame_for_h264_encode(frame, fps=fps, pts=v_pts)
                    v_pts += 1
                    for packet in v_out.encode(frame):
                        out.mux(packet)
                elif isinstance(frame, AudioFrame) and a_out is not None and a_resampler is not None:
                    _mux_audio_frame(out, a_out, a_resampler, frame)

            if v_out is not None:
                for packet in v_out.encode():
                    out.mux(packet)
            if a_out is not None and a_resampler is not None:
                _flush_audio_encoder(out, a_out, a_resampler)
        finally:
            out.close()
    finally:
        inp.close()
    return target


def upscale_video(
    src: Path | str,
    dest: Path | str,
    *,
    scale: float = 2.0,
    max_side: int = 2048,
) -> tuple[Path, int, int]:
    """Lanczos-upscale every video frame; re-encode audio. Returns (path, w, h)."""
    import av
    import numpy as np
    from av.audio.frame import AudioFrame
    from av.video.frame import VideoFrame
    from PIL import Image

    source = Path(src)
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    factor = max(1.0, float(scale))
    inp = av.open(str(source))
    try:
        v_in = next((s for s in inp.streams if s.type == "video"), None)
        if v_in is None:
            raise ValueError(f"no video stream in {source}")
        src_w = int(v_in.codec_context.width or 0)
        src_h = int(v_in.codec_context.height or 0)
        if src_w < 1 or src_h < 1:
            raise ValueError(f"invalid video size {src_w}x{src_h}")
        out_w = max(2, int(round(src_w * factor)))
        out_h = max(2, int(round(src_h * factor)))
        long_side = max(out_w, out_h)
        if long_side > max_side:
            shrink = max_side / long_side
            out_w = max(2, int(round(out_w * shrink)))
            out_h = max(2, int(round(out_h * shrink)))
        out_w -= out_w % 2
        out_h -= out_h % 2

        fps = _sane_video_fps(v_in)
        out = av.open(str(target), mode="w")
        try:
            v_out = _add_h264_stream(
                out, fps=fps, width=out_w, height=out_h, preset="medium", crf="17"
            )
            a_out = None
            a_resampler = None
            v_pts = 0
            if any(s.type == "audio" for s in inp.streams):
                a_out = out.add_stream("aac", rate=48000)
                a_out.layout = "stereo"
                a_out.time_base = Fraction(1, 48000)
                a_resampler = _make_audio_resampler(
                    sample_fmt="fltp", layout="stereo", rate=48000
                )

            for frame in inp.decode():
                if isinstance(frame, VideoFrame):
                    img = Image.fromarray(frame.to_ndarray(format="rgb24"))
                    scaled = img.resize((out_w, out_h), Image.Resampling.LANCZOS)
                    new_frame = av.VideoFrame.from_ndarray(np.asarray(scaled), format="rgb24")
                    new_frame = new_frame.reformat(format="yuv420p")
                    _stamp_encode_video_pts(new_frame, v_pts, fps)
                    v_pts += 1
                    for packet in v_out.encode(new_frame):
                        out.mux(packet)
                elif isinstance(frame, AudioFrame) and a_out is not None and a_resampler is not None:
                    _mux_audio_frame(out, a_out, a_resampler, frame)

            for packet in v_out.encode():
                out.mux(packet)
            if a_out is not None and a_resampler is not None:
                _flush_audio_encoder(out, a_out, a_resampler)
        finally:
            out.close()
    finally:
        inp.close()
    return target, out_w, out_h


def assert_audio_durations(seconds: list[float]) -> None:
    """Enforce h3.c audio-reference limits (2–15 s each, ≤3 clips, total ≤15 s)."""
    if len(seconds) > MAX_REF_AUDIO_CLIPS:
        raise ValueError(f"at most {MAX_REF_AUDIO_CLIPS} audio references")
    total = 0.0
    for duration in seconds:
        if duration + _DURATION_SLACK_S < MIN_REF_AUDIO_S:
            raise ValueError(
                f"audio reference is {duration:.2f}s; h3.c requires at least "
                f"{MIN_REF_AUDIO_S:.0f}s"
            )
        if duration - _DURATION_SLACK_S > MAX_REF_AUDIO_S:
            raise ValueError(
                f"audio reference is {duration:.2f}s; h3.c allows at most "
                f"{MAX_REF_AUDIO_S:.0f}s"
            )
        total += duration
    if total - _DURATION_SLACK_S > MAX_REF_AUDIO_TOTAL_S:
        raise ValueError(
            f"audio references total {total:.2f}s; h3.c caps the sum at "
            f"{MAX_REF_AUDIO_TOTAL_S:.0f}s"
        )


def extract_last_frame(video_path: str | Path, dest: str | Path) -> Path:
    """Write the last decoded frame of ``video_path`` as PNG to ``dest``."""
    import av
    from PIL import Image

    src = Path(video_path)
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(src))
    stream = container.streams.video[0]
    last = None
    for frame in container.decode(stream):
        last = frame
    container.close()
    if last is None:
        raise RuntimeError(f"no video frames in {src}")
    img = last.to_ndarray(format="rgb24")
    Image.fromarray(img).save(out)
    return out


def video_poster_path(video_path: str | Path) -> Path:
    """Sidecar JPEG used for library / timeline thumbs (next to the mp4)."""
    src = Path(video_path)
    return src.parent / ".thumbs" / f"{src.stem}.jpg"


def ensure_video_poster(
    video_path: str | Path,
    *,
    max_edge: int = 480,
    force: bool = False,
) -> Path | None:
    """Decode an early frame to JPEG so WebKit library thumbs aren't black.

    h3-av writes non-faststart MP4s (moov at end). ``<video preload=metadata>``
    then often paints a black poster in Safari / pywebview. A real JPEG sidesteps that.
    """
    if not media_available():
        return None
    src = Path(video_path)
    if not src.is_file():
        return None
    dest = video_poster_path(src)
    if dest.is_file() and dest.stat().st_size > 64 and not force:
        # Stale if the video was rewritten after the thumb.
        try:
            if dest.stat().st_mtime >= src.stat().st_mtime:
                return dest
        except OSError:
            return dest
    import av
    from PIL import Image

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.jpg")
    try:
        container = av.open(str(src))
        try:
            stream = container.streams.video[0]
            # Prefer a frame a little past t=0 (some encodes are near-black at 0).
            target_pts = None
            if stream.average_rate and stream.time_base:
                # ~0.12s in
                target_pts = int(0.12 / float(stream.time_base))
            chosen = None
            for i, frame in enumerate(container.decode(stream)):
                chosen = frame
                if target_pts is not None and frame.pts is not None and frame.pts >= target_pts:
                    break
                if i >= 8:
                    break
        finally:
            container.close()
        if chosen is None:
            return None
        img = Image.fromarray(chosen.to_ndarray(format="rgb24"))
        w, h = img.size
        edge = max(w, h)
        if edge > max_edge:
            scale = max_edge / float(edge)
            img = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
        img.save(tmp, format="JPEG", quality=82, optimize=True)
        tmp.replace(dest)
        return dest
    except Exception as exc:
        log.debug("poster for %s failed: %s", src, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def concat_mp4s(paths: list[Path], dest: Path) -> Path:
    """Stream-copy concatenate MP4s with PyAV. Falls back to re-encode if needed."""
    import av

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        import shutil

        shutil.copy2(paths[0], dest)
        return dest

    first = av.open(str(paths[0]))
    out = av.open(str(dest), mode="w")
    stream_map: dict[int, Any] = {}
    for i, stream in enumerate(first.streams):
        if stream.type not in ("video", "audio"):
            continue
        out_stream = out.add_stream(template=stream)
        stream_map[i] = out_stream
    first.close()

    try:
        for path in paths:
            src = av.open(str(path))
            for packet in src.demux():
                if packet.stream_index not in stream_map:
                    continue
                if packet.dts is None:
                    continue
                packet.stream = stream_map[packet.stream_index]
                out.mux(packet)
            src.close()
    finally:
        out.close()
    return dest


def decode_clip_rgb(path: str | Path) -> tuple[Any, int, int]:
    """Decode a clip to ``[T,H,W,3]`` uint8 RGB plus width/height."""
    import numpy as np
    import av

    container = av.open(str(path))
    try:
        frames: list[Any] = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
        if not frames:
            raise ValueError(f"no video frames in {path}")
        arr = np.stack(frames, axis=0)
        h, w = int(arr.shape[1]), int(arr.shape[2])
        return arr, w, h
    finally:
        container.close()


def write_rgb_mp4(frames: Any, dest: str | Path, fps: int = 24) -> Path:
    """Write ``[T,H,W,3]`` uint8 RGB to an H.264 MP4 (silent)."""
    import av
    import numpy as np

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    h, w = int(frames.shape[1]), int(frames.shape[2])
    out = av.open(str(dest), mode="w")
    try:
        stream = out.add_stream("libx264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, int(fps))
        stream.options = {
            "crf": "12",
            "preset": "veryfast",
            "bf": "0",
            "g": "1",
            "keyint_min": "1",
            "sc_threshold": "0",
            "x264-params": "keyint=1:min-keyint=1:scenecut=0:bframes=0",
        }
        for index, frame in enumerate(frames):
            vf = av.VideoFrame.from_ndarray(np.asarray(frame), format="rgb24")
            vf.pts = int(index)
            for packet in stream.encode(vf):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    finally:
        out.close()
    return dest


def remux_audio(
    video_no_audio: str | Path, audio_source: str | Path, dest: str | Path
) -> Path:
    """Copy video from ``video_no_audio`` and audio stream(s) from ``audio_source``."""
    import shutil

    import av

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    video_no_audio = Path(video_no_audio)
    audio_source = Path(audio_source)

    ain = av.open(str(audio_source))
    try:
        has_audio = any(s.type == "audio" for s in ain.streams)
    finally:
        ain.close()
    if not has_audio:
        shutil.copy2(video_no_audio, dest)
        return dest

    try:
        vin = av.open(str(video_no_audio))
        ain = av.open(str(audio_source))
        try:
            out = av.open(str(dest), mode="w")
            try:
                v_in = vin.streams.video[0]
                a_in = next(s for s in ain.streams if s.type == "audio")
                try:
                    v_out = out.add_stream(template=v_in)
                    a_out = out.add_stream(template=a_in)
                    copy_packets = True
                except TypeError:
                    rate = v_in.average_rate or 24
                    v_out = out.add_stream("libx264", rate=rate)
                    v_out.width = v_in.width
                    v_out.height = v_in.height
                    v_out.pix_fmt = "yuv420p"
                    a_out = out.add_stream("aac")
                    copy_packets = False
                if copy_packets:
                    for packet in vin.demux(v_in):
                        if packet.dts is None:
                            continue
                        packet.stream = v_out
                        out.mux(packet)
                    for packet in ain.demux(a_in):
                        if packet.dts is None:
                            continue
                        packet.stream = a_out
                        out.mux(packet)
                else:
                    for frame in vin.decode(v_in):
                        for packet in v_out.encode(frame):
                            out.mux(packet)
                    for packet in v_out.encode(None):
                        out.mux(packet)
                    for frame in ain.decode(a_in):
                        for packet in a_out.encode(frame):
                            out.mux(packet)
                    for packet in a_out.encode(None):
                        out.mux(packet)
            finally:
                out.close()
        finally:
            vin.close()
            ain.close()
    except Exception as exc:
        log.warning("audio remux failed (%s); keeping silent video", exc)
        shutil.copy2(video_no_audio, dest)
    return dest
