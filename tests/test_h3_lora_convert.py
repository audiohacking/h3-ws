"""Diffusers→native H3 LoRA conversion (no network)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3_lora_convert import (  # noqa: E402
    convert_diffusers_h3_lora,
    is_diffusers_h3_lora,
    is_native_h3_lora,
)


def _save(path: Path, tensors: dict[str, np.ndarray]) -> None:
    from safetensors.numpy import save_file

    save_file(tensors, str(path))


def _keys(path: Path) -> list[str]:
    from safetensors import safe_open

    with safe_open(path, framework="np") as handle:
        return list(handle.keys())


class DiffusersConvertTests(unittest.TestCase):
    def test_fuses_qkv_and_renames_ff(self) -> None:
        rank, hidden, head_out, ffn = 4, 16, 8, 32
        tensors = {
            "transformer_blocks.0.attn.to_q.lora.down.weight": np.ones((rank, hidden), np.float32),
            "transformer_blocks.0.attn.to_q.lora.up.weight": np.ones((head_out, rank), np.float32),
            "transformer_blocks.0.attn.to_k.lora.down.weight": np.ones((rank, hidden), np.float32) * 2,
            "transformer_blocks.0.attn.to_k.lora.up.weight": np.ones((head_out, rank), np.float32) * 2,
            "transformer_blocks.0.attn.to_v.lora.down.weight": np.ones((rank, hidden), np.float32) * 3,
            "transformer_blocks.0.attn.to_v.lora.up.weight": np.ones((head_out, rank), np.float32) * 3,
            "transformer_blocks.0.attn.to_out.0.lora.down.weight": np.ones((rank, head_out), np.float32),
            "transformer_blocks.0.attn.to_out.0.lora.up.weight": np.ones((hidden, rank), np.float32),
            "transformer_blocks.0.ff.net.0.proj.lora_A.weight": np.ones((rank, hidden), np.float32),
            "transformer_blocks.0.ff.net.0.proj.lora_B.weight": np.arange(
                ffn * 2 * rank, dtype=np.float32
            ).reshape(ffn * 2, rank),
            "transformer_blocks.0.ff.net.2.lora_A.weight": np.ones((rank, ffn), np.float32),
            "transformer_blocks.0.ff.net.2.lora_B.weight": np.ones((hidden, rank), np.float32),
            "token_refiner.refiner_blocks.0.attn.to_q.lora.down.weight": np.ones((rank, hidden), np.float32),
            "token_refiner.refiner_blocks.0.attn.to_q.lora.up.weight": np.ones((head_out, rank), np.float32),
            "token_refiner.refiner_blocks.0.attn.to_k.lora.down.weight": np.ones((rank, hidden), np.float32),
            "token_refiner.refiner_blocks.0.attn.to_k.lora.up.weight": np.ones((head_out, rank), np.float32),
            "token_refiner.refiner_blocks.0.attn.to_v.lora.down.weight": np.ones((rank, hidden), np.float32),
            "token_refiner.refiner_blocks.0.attn.to_v.lora.up.weight": np.ones((head_out, rank), np.float32),
            "token_refiner.refiner_blocks.0.attn.to_out.0.lora.down.weight": np.ones((rank, head_out), np.float32),
            "token_refiner.refiner_blocks.0.attn.to_out.0.lora.up.weight": np.ones((hidden, rank), np.float32),
            "token_refiner.refiner_blocks.0.ff.net.0.proj.lora_A.weight": np.ones((rank, hidden), np.float32),
            "token_refiner.refiner_blocks.0.ff.net.0.proj.lora_B.weight": np.ones((ffn * 2, rank), np.float32),
            "token_refiner.refiner_blocks.0.ff.net.2.lora_A.weight": np.ones((rank, ffn), np.float32),
            "token_refiner.refiner_blocks.0.ff.net.2.lora_B.weight": np.ones((hidden, rank), np.float32),
        }
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "dmad_diffusers.safetensors"
            dst = Path(tmp) / "native" / "dmad_comfy.safetensors"
            _save(src, tensors)
            self.assertTrue(is_diffusers_h3_lora(_keys(src)))
            out = convert_diffusers_h3_lora(src, dst)
            self.assertEqual(out.resolve(), dst.resolve())
            keys = _keys(dst)
            self.assertTrue(is_native_h3_lora(keys))
            self.assertIn("diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight", keys)
            self.assertIn("diffusion_model.blocks.0.attn.out_proj.lora_A.weight", keys)
            self.assertIn("diffusion_model.blocks.0.mlp.fc1.lora_A.weight", keys)
            self.assertIn("diffusion_model.blocks.0.mlp.fc2.lora_A.weight", keys)
            self.assertIn("diffusion_model.token_refiner.blocks.0.attn.qkv_proj.lora_A.weight", keys)
            from safetensors import safe_open

            with safe_open(dst, framework="pt") as handle:
                a = handle.get_tensor(
                    "diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"
                ).float().numpy()
                b = handle.get_tensor(
                    "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"
                ).float().numpy()
                fc1_b = handle.get_tensor(
                    "diffusion_model.blocks.0.mlp.fc1.lora_B.weight"
                ).float().numpy()
            self.assertEqual(tuple(a.shape), (rank * 3, hidden))
            self.assertEqual(tuple(b.shape), (head_out * 3, rank * 3))
            # FC1 halves swapped: first row of original B becomes at half offset.
            original_b = tensors["transformer_blocks.0.ff.net.0.proj.lora_B.weight"]
            self.assertTrue(np.allclose(fc1_b[:ffn], original_b[ffn:]))
            self.assertTrue(np.allclose(fc1_b[ffn:], original_b[:ffn]))


if __name__ == "__main__":
    unittest.main()
