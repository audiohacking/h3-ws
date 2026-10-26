"""Export / import generation recipes as zip archives.

A recipe packs the composer snapshot (prompt, settings, refs) plus the media
files it referenced, and optionally the finished mp4. Used to swap jobs
between machines for debugging and iteration.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("h3-ws.recipe")

RECIPE_FORMAT = "h3-ws-recipe"
RECIPE_VERSION = 1
MANIFEST_NAME = "recipe.json"


def _safe_name(name: str, fallback: str = "asset") -> str:
    base = Path(str(name or fallback)).name.strip() or fallback
    cleaned = "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in base)
    return cleaned[:120] or fallback


def _collect_media_paths(generation: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Return (role, path) pairs from a generation snapshot."""
    if not isinstance(generation, dict):
        return []
    found: list[tuple[str, str]] = []
    for key in ("imagePath", "image_path"):
        raw = generation.get(key)
        if isinstance(raw, str) and raw.strip():
            found.append(("image", raw.strip()))
            break
    for key in ("endImagePath", "end_image_path"):
        raw = generation.get(key)
        if isinstance(raw, str) and raw.strip():
            found.append(("end_image", raw.strip()))
            break
    refs = generation.get("refs")
    if isinstance(refs, list):
        for index, item in enumerate(refs):
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            if isinstance(path, str) and path.strip():
                found.append((f"ref_{index}", path.strip()))
            audio = item.get("audioPath") or item.get("audio_path")
            if isinstance(audio, str) and audio.strip():
                found.append((f"ref_{index}_audio", audio.strip()))
    return found


def _resolve_media(
    raw: str,
    *,
    output_dir: Path,
    upload_dir: Path,
    frames_dir: Path | None = None,
) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    p = Path(text)
    if p.is_file():
        return p.resolve()
    name = Path(text).name
    candidates = [output_dir / name, upload_dir / name]
    if frames_dir is not None:
        candidates.append(frames_dir / name)
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


def build_recipe_zip(
    *,
    clip: Any,
    generation: dict[str, Any] | None,
    output_dir: Path,
    upload_dir: Path,
    frames_dir: Path | None = None,
    include_video: bool = True,
) -> bytes:
    """Build a .h3recipe.zip (standard zip) for one clip."""
    output_dir = Path(output_dir)
    upload_dir = Path(upload_dir)
    gen = dict(generation) if isinstance(generation, dict) else {}
    if not gen and getattr(clip, "generation", None):
        gen = dict(clip.generation)  # type: ignore[arg-type]

    media_map: dict[str, str] = {}  # role -> archive relative path
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        used_names: set[str] = set()

        def add_file(role: str, src: Path) -> str:
            archive_name = f"media/{role}_{_safe_name(src.name)}"
            # Avoid collisions when two roles share a basename.
            if archive_name in used_names:
                archive_name = f"media/{role}_{uuid.uuid4().hex[:8]}_{_safe_name(src.name)}"
            used_names.add(archive_name)
            zf.write(src, archive_name)
            media_map[role] = archive_name
            return archive_name

        for role, raw in _collect_media_paths(gen):
            resolved = _resolve_media(
                raw,
                output_dir=output_dir,
                upload_dir=upload_dir,
                frames_dir=frames_dir,
            )
            if resolved is None:
                log.warning("recipe export: missing media %s (%s)", role, raw)
                continue
            add_file(role, resolved)

        video_archive: str | None = None
        filename = str(getattr(clip, "filename", "") or "").strip()
        if include_video and filename:
            video_path = output_dir / filename
            if video_path.is_file():
                video_archive = add_file("video", video_path)

        manifest = {
            "format": RECIPE_FORMAT,
            "version": RECIPE_VERSION,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "clip": {
                "prompt": getattr(clip, "prompt", "") or "",
                "label": getattr(clip, "label", "") or "",
                "mode": getattr(clip, "mode", "") or "",
                "status": getattr(clip, "status", "") or "",
                "num_frames": getattr(clip, "num_frames", None),
                "width": getattr(clip, "width", None),
                "height": getattr(clip, "height", None),
                "seed": getattr(clip, "seed", None),
                "num_steps": getattr(clip, "num_steps", None),
                "layers": getattr(clip, "layers", None),
                "reuse": getattr(clip, "reuse", None),
                "duration_seconds": getattr(clip, "duration_seconds", None),
                "clip_count": getattr(clip, "clip_count", None),
                "quality": getattr(clip, "quality", None),
                "loras": getattr(clip, "loras", None),
            },
            "generation": gen,
            "media": media_map,
            "video": video_archive,
        }
        zf.writestr(
            MANIFEST_NAME,
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )

    return buffer.getvalue()


def _rewrite_generation_paths(
    generation: dict[str, Any],
    role_to_path: dict[str, Path],
) -> dict[str, Any]:
    """Copy generation and point media keys at freshly imported absolute paths."""
    out = json.loads(json.dumps(generation))  # deep copy via JSON
    if "image" in role_to_path:
        path = str(role_to_path["image"])
        if "imagePath" in out or "image_path" not in out:
            out["imagePath"] = path
        if "image_path" in out:
            out["image_path"] = path
    if "end_image" in role_to_path:
        path = str(role_to_path["end_image"])
        if "endImagePath" in out or "end_image_path" not in out:
            out["endImagePath"] = path
        if "end_image_path" in out:
            out["end_image_path"] = path
    refs = out.get("refs")
    if isinstance(refs, list):
        for index, item in enumerate(refs):
            if not isinstance(item, dict):
                continue
            key = f"ref_{index}"
            if key in role_to_path:
                item["path"] = str(role_to_path[key])
            audio_key = f"ref_{index}_audio"
            if audio_key in role_to_path:
                item["audioPath"] = str(role_to_path[audio_key])
                item.pop("audio_path", None)
            # Drop stale blob previews — client rebuilds them.
            item.pop("previewUrl", None)
            item.pop("preview_url", None)
    return out


def import_recipe_zip(
    raw: bytes,
    *,
    output_dir: Path,
    upload_dir: Path,
    project_id: str | None = None,
    make_clip: Callable[..., Any],
) -> dict[str, Any]:
    """Import a recipe zip.

    ``make_clip`` is called with keyword args for a new ClipRecord (or
    equivalent) and must return the stored clip object / dict.

    Returns ``{clip, generation, had_video}``.
    """
    output_dir = Path(output_dir)
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        names = set(zf.namelist())
        if MANIFEST_NAME not in names:
            raise ValueError("not an h3-ws recipe archive (missing recipe.json)")
        manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("invalid recipe.json")
        if manifest.get("format") != RECIPE_FORMAT:
            raise ValueError(
                f"unsupported recipe format: {manifest.get('format')!r}"
            )
        version = int(manifest.get("version") or 0)
        if version < 1 or version > RECIPE_VERSION:
            raise ValueError(f"unsupported recipe version: {version}")

        media = manifest.get("media") if isinstance(manifest.get("media"), dict) else {}
        generation_in = (
            dict(manifest["generation"])
            if isinstance(manifest.get("generation"), dict)
            else {}
        )
        clip_meta = (
            dict(manifest["clip"]) if isinstance(manifest.get("clip"), dict) else {}
        )

        role_to_path: dict[str, Path] = {}
        for role, archive_path in media.items():
            if role == "video":
                continue
            if not isinstance(archive_path, str) or archive_path not in names:
                log.warning("recipe import: missing archive member %s", archive_path)
                continue
            # Zip-slip guard
            member = Path(archive_path)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe archive path: {archive_path}")
            ext = Path(archive_path).suffix or ".bin"
            dest = upload_dir / f"{uuid.uuid4().hex}{ext}"
            with zf.open(archive_path) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            role_to_path[str(role)] = dest.resolve()

        had_video = False
        video_filename = ""
        video_archive = manifest.get("video") or media.get("video")
        if isinstance(video_archive, str) and video_archive in names:
            member = Path(video_archive)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe archive path: {video_archive}")
            video_filename = f"{uuid.uuid4().hex}.mp4"
            dest = output_dir / video_filename
            with zf.open(video_archive) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            had_video = dest.is_file() and dest.stat().st_size > 0
            if not had_video:
                video_filename = ""

        generation = _rewrite_generation_paths(generation_in, role_to_path)
        prompt = str(
            generation.get("prompt")
            or clip_meta.get("prompt")
            or ""
        ).strip()
        mode = str(
            generation.get("mode") or clip_meta.get("mode") or "t2va"
        ).strip() or "t2va"
        clip_id = str(uuid.uuid4())
        chain_id = str(uuid.uuid4())
        status = "done" if had_video else "done"
        label = str(clip_meta.get("label") or ("IMPORTED" if had_video else "RECIPE"))

        clip = make_clip(
            id=clip_id,
            prompt=prompt or "(imported recipe)",
            label=label[:32],
            video_url="",  # filled by caller via clip_url when filename set
            filename=video_filename,
            chain_id=chain_id,
            clip_index=0,
            mode=mode,
            status=status,
            created_at=datetime.now().isoformat(timespec="seconds"),
            project_id=project_id,
            generation=generation,
            num_frames=clip_meta.get("num_frames"),
            width=clip_meta.get("width"),
            height=clip_meta.get("height"),
            seed=clip_meta.get("seed"),
            num_steps=clip_meta.get("num_steps") or generation.get("numSteps"),
            layers=clip_meta.get("layers") or generation.get("layers"),
            reuse=clip_meta.get("reuse") or generation.get("reuse"),
            duration_seconds=clip_meta.get("duration_seconds"),
            clip_count=clip_meta.get("clip_count") or generation.get("clipMultiplier"),
            quality=clip_meta.get("quality") or generation.get("quality"),
            loras=clip_meta.get("loras"),
            bytes=(output_dir / video_filename).stat().st_size
            if had_video and video_filename
            else None,
        )
        return {
            "clip": clip,
            "generation": generation,
            "had_video": had_video,
            "clip_id": clip_id,
        }
