"""Generation recipe zip export / import round-trip."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from h3_recipe import RECIPE_FORMAT, build_recipe_zip, import_recipe_zip


def _clip(img: Path, vid: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        prompt="a cat walks",
        label="CURRENT",
        mode="ref2va",
        status="done",
        filename=vid.name if vid else "",
        generation={
            "prompt": "a cat walks",
            "mode": "ref2va",
            "quality": "fast",
            "resolutionId": "512x512",
            "durationId": "1s",
            "numSteps": 8,
            "layers": 50,
            "reuse": 1,
            "seed": "1",
            "loraPresetIds": [],
            "refs": [
                {
                    "id": "r1",
                    "kind": "image",
                    "path": str(img),
                    "name": "ref.png",
                    "enabled": True,
                }
            ],
            "imagePath": None,
            "endImagePath": None,
            "clipMultiplier": 1,
            "faces": True,
        },
        num_frames=22,
        width=512,
        height=512,
        seed=1,
        num_steps=8,
        layers=50,
        reuse=1,
        duration_seconds=1.0,
        clip_count=1,
        quality="fast",
        loras=None,
    )


def test_recipe_roundtrip_includes_refs_and_video(tmp_path: Path) -> None:
    out = tmp_path / "out"
    up = tmp_path / "up"
    out.mkdir()
    up.mkdir()
    img = up / "ref.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 32)
    vid = out / "clip.mp4"
    vid.write_bytes(b"fakevideo" * 50)
    clip = _clip(img, vid)

    blob = build_recipe_zip(
        clip=clip,
        generation=clip.generation,
        output_dir=out,
        upload_dir=up,
    )
    assert len(blob) > 100

    made: dict[str, SimpleNamespace] = {}

    def make_clip(**kwargs):
        c = SimpleNamespace(**kwargs)
        made["c"] = c
        return c

    result = import_recipe_zip(
        blob,
        output_dir=tmp_path / "imp_out",
        upload_dir=tmp_path / "imp_up",
        project_id="proj",
        make_clip=make_clip,
    )
    assert result["had_video"] is True
    gen = result["generation"]
    assert gen["prompt"] == "a cat walks"
    assert gen["faces"] is True
    ref_path = Path(gen["refs"][0]["path"])
    assert ref_path.is_file()
    assert ref_path.read_bytes().startswith(b"\x89PNG")
    assert made["c"].filename.endswith(".mp4")
    assert (tmp_path / "imp_out" / made["c"].filename).is_file()


def test_recipe_export_without_video_still_packs_refs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    up = tmp_path / "up"
    out.mkdir()
    up.mkdir()
    img = up / "anchor.png"
    img.write_bytes(b"IMGDATA")
    clip = _clip(img, vid=None)
    clip.filename = "missing.mp4"
    blob = build_recipe_zip(
        clip=clip,
        generation=clip.generation,
        output_dir=out,
        upload_dir=up,
        include_video=True,
    )

    def make_clip(**kwargs):
        return SimpleNamespace(**kwargs)

    result = import_recipe_zip(
        blob,
        output_dir=tmp_path / "o2",
        upload_dir=tmp_path / "u2",
        make_clip=make_clip,
    )
    assert result["had_video"] is False
    assert Path(result["generation"]["refs"][0]["path"]).is_file()


def test_bad_archive_rejected(tmp_path: Path) -> None:
    import io
    import json
    import zipfile

    import pytest

    def make_clip(**kwargs):
        return SimpleNamespace(**kwargs)

    with pytest.raises(zipfile.BadZipFile):
        import_recipe_zip(
            b"not-a-zip",
            output_dir=tmp_path,
            upload_dir=tmp_path,
            make_clip=make_clip,
        )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("recipe.json", json.dumps({"format": "nope", "version": 1}))
    with pytest.raises(ValueError, match="unsupported recipe format"):
        import_recipe_zip(
            buf.getvalue(),
            output_dir=tmp_path,
            upload_dir=tmp_path,
            make_clip=make_clip,
        )
    assert RECIPE_FORMAT == "h3-ws-recipe"
