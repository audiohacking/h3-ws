"""LoRA catalog and argv wiring. No Hugging Face traffic."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3_backend import GenerateRequest, LoraRef, build_h3_argv
from h3_lora import (
    BUILTIN_LORAS,
    TUTU_REPO,
    ensure_lora,
    find_cached_lora,
    lora_cached_path,
    lora_catalog,
    normalize_lora_spec,
    parse_lora_specs,
)
from h3_session import build_session_argv


class LoraCatalogTests(unittest.TestCase):
    def test_builtin_includes_tutu_step100(self) -> None:
        tutu = next(p for p in BUILTIN_LORAS if p["id"] == "tutu_20to8_nfe_step100")
        self.assertEqual(tutu["scale"], 0.8)
        self.assertEqual(tutu["steps"], 8)
        self.assertIn("tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA", tutu["spec"])
        # Empty disk tree → download recipes still appear in the catalog.
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"H3_LORA_DIR": tmp}, clear=False):
                with mock.patch("h3_lora._lora_search_roots", return_value=[Path(tmp)]):
                    catalog = lora_catalog(None)
                    ids = [p["id"] for p in catalog]
                    self.assertIn("tutu_20to8_nfe_step100", ids)

    def test_normalize_blob_to_resolve(self) -> None:
        spec = normalize_lora_spec(
            "https://huggingface.co/tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA/"
            "blob/main/comfyui/tutu.safetensors"
        )
        self.assertIn("/resolve/main/", spec)

    def test_parse_lora_specs_by_id(self) -> None:
        parsed = parse_lora_specs(
            [{"id": "tutu_20to8_nfe_step100", "scale": 0.8}],
            BUILTIN_LORAS,
        )
        self.assertEqual(len(parsed), 1)
        self.assertIn("step000100", parsed[0][0])
        self.assertEqual(parsed[0][1], 0.8)

    def test_build_argv_emits_lora_and_rejects_ssd(self) -> None:
        lora = LoraRef(
            spec="x",
            path=Path("/tmp/tutu.safetensors"),
            scale=0.8,
        )
        req = GenerateRequest(
            prompt="fox",
            output_path=Path("/tmp/out.mp4"),
            loras=[lora],
        )
        argv = build_h3_argv(h3_bin=Path("/opt/h3"), model_dir=Path("/m"), req=req)
        self.assertIn("--lora", argv)
        self.assertTrue(any(str(a).startswith("/tmp/tutu.safetensors") for a in argv))
        session = build_session_argv(h3_bin=Path("/opt/h3"), model_dir=Path("/m"), req=req)
        self.assertIn("--lora", session)
        with self.assertRaisesRegex(ValueError, "ssd-streaming"):
            build_h3_argv(
                h3_bin=Path("/opt/h3"),
                model_dir=Path("/m"),
                req=GenerateRequest(
                    prompt="fox",
                    output_path=Path("/tmp/out.mp4"),
                    loras=[lora],
                    ssd_streaming=True,
                ),
            )


class LoraCacheReuseTests(unittest.TestCase):
    def test_finds_nested_comfyui_layout(self) -> None:
        """Clone layout keeps Tutu under comfyui/; Turbo must not re-download."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            leaf = (
                "tutu-t8-minimax-h3-av-20to8-nfe-lora-step000100-bf16-comfyui.safetensors"
            )
            nested = (
                root
                / f"{TUTU_REPO.replace('/', '__')}"
                / "comfyui"
                / leaf
            )
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b"x" * 128)

            with mock.patch.dict(os.environ, {"H3_LORA_DIR": str(root)}, clear=False):
                hit = find_cached_lora(
                    TUTU_REPO,
                    f"comfyui/{leaf}",
                )
                self.assertIsNotNone(hit)
                self.assertEqual(hit.resolve(), nested.resolve())

                # Repo-only Turbo spec (no filename) must still hit the nested file.
                cached = lora_cached_path(TUTU_REPO)
                self.assertIsNotNone(cached)
                self.assertEqual(cached.resolve(), nested.resolve())

                result = ensure_lora(TUTU_REPO)
                self.assertTrue(result["cached"])
                self.assertEqual(Path(result["path"]).resolve(), nested.resolve())

    def test_scan_disk_loras_lists_all_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tao = root / "Robert__TaoMate" / "taomate_h3_3step_comfy.safetensors"
            style = root / "fal__Realism" / "people.safetensors"
            tao.parent.mkdir(parents=True)
            style.parent.mkdir(parents=True)
            tao.write_bytes(b"x" * 128)
            style.write_bytes(b"y" * 128)
            with mock.patch.dict(os.environ, {"H3_LORA_DIR": str(root)}, clear=False):
                with mock.patch("h3_lora._lora_search_roots", return_value=[root]):
                    with mock.patch("h3_lora.iter_hf_hub_lora_files", return_value=[]):
                        from h3_lora import scan_disk_loras

                        scanned = scan_disk_loras()
                        self.assertEqual(len(scanned), 2)
                        labels = " ".join(e["label"].lower() for e in scanned)
                        self.assertIn("taomate", labels)
                        catalog = lora_catalog(None)
                        local = [p for p in catalog if p.get("local")]
                        self.assertGreaterEqual(len(local), 2)
                        turbo = [p for p in catalog if p.get("turbo")]
                        self.assertTrue(any("taomate" in p["label"].lower() for p in turbo))


if __name__ == "__main__":
    unittest.main()
