"""Convert Diffusers MiniMax-H3 LoRAs (split Q/K/V) to h3.c / Comfy native layout.

DMAD and other Diffusers adapters use ``attn.to_q|to_k|to_v`` and
``ff.net.{0,2}``. Native h3.c fuses ``attn.qkv_proj`` / ``attn.out_proj`` /
``mlp.fc1`` / ``mlp.fc2`` (same stems as TaoMate / Tutu Comfy packages).

QKV fusion uses a block-diagonal B and stacked A so ``B @ A`` still equals
``[Bq@Aq; Bk@Ak; Bv@Av]``. FC1 halves are swapped on B (Diffusers↔native
gate/value order — see community notes on Diffusers→Comfy H3 LoRAs).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger("h3")


def is_diffusers_h3_lora(keys: list[str]) -> bool:
    joined = " ".join(keys[:80])
    return (
        "transformer_blocks." in joined
        and ("attn.to_q" in joined or ".attn.to_q." in joined)
        and "attn.qkv_proj" not in joined
    )


def is_native_h3_lora(keys: list[str]) -> bool:
    return any("attn.qkv_proj" in k for k in keys[:120])


def _read_tensors(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load tensors as float32 numpy (BF16 via torch — numpy lacks bf16)."""
    from safetensors import safe_open

    meta: dict[str, str] = {}
    with safe_open(path, framework="pt") as handle:
        raw_meta = handle.metadata() or {}
        meta = {str(k): str(v) for k, v in raw_meta.items()}
        keys = list(handle.keys())
        tensors: dict[str, np.ndarray] = {}
        for key in keys:
            t = handle.get_tensor(key)
            tensors[key] = t.detach().float().cpu().numpy()
    return tensors, meta


def _write_tensors(path: Path, tensors: dict[str, np.ndarray], meta: dict[str, str]) -> None:
    """Write BF16 safetensors (matches TaoMate/Tutu / h3.c fuse path)."""
    import torch
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    torch_tensors = {
        key: torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)).to(
            torch.bfloat16
        )
        for key, value in tensors.items()
    }
    save_file(torch_tensors, str(path), metadata={k: str(v) for k, v in meta.items()})


def _pair_suffix(name: str) -> tuple[str, str] | None:
    for a_suf, b_suf in (
        (".lora.down.weight", ".lora.up.weight"),
        (".lora_A.weight", ".lora_B.weight"),
        (".lora.A.weight", ".lora.B.weight"),
        (".lora_down.weight", ".lora_up.weight"),
    ):
        if name.endswith(a_suf):
            return a_suf, b_suf
        if name.endswith(b_suf):
            return a_suf, b_suf
    return None


def _stem(name: str) -> str | None:
    pair = _pair_suffix(name)
    if not pair:
        return None
    a_suf, b_suf = pair
    if name.endswith(a_suf):
        return name[: -len(a_suf)]
    if name.endswith(b_suf):
        return name[: -len(b_suf)]
    return None


def _get_ab(
    tensors: dict[str, np.ndarray], stem: str
) -> tuple[np.ndarray, np.ndarray] | None:
    for a_suf, b_suf in (
        (".lora.down.weight", ".lora.up.weight"),
        (".lora_A.weight", ".lora_B.weight"),
        (".lora.A.weight", ".lora.B.weight"),
        (".lora_down.weight", ".lora_up.weight"),
    ):
        a = tensors.get(stem + a_suf)
        b = tensors.get(stem + b_suf)
        if a is not None and b is not None:
            return a.astype(np.float32, copy=False), b.astype(np.float32, copy=False)
    return None


def _native_prefix(diffusers_stem: str) -> str | None:
    m = re.match(
        r"^(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)$",
        diffusers_stem,
    )
    if not m:
        return None
    kind, index, rest = m.group(1), m.group(2), m.group(3)
    if kind == "transformer_blocks":
        base = f"diffusion_model.blocks.{index}."
    else:
        base = f"diffusion_model.token_refiner.blocks.{index}."
    if rest == "attn.to_out.0":
        return base + "attn.out_proj"
    if rest == "ff.net.0.proj":
        return base + "mlp.fc1"
    if rest == "ff.net.2":
        return base + "mlp.fc2"
    if rest in {"attn.to_q", "attn.to_k", "attn.to_v"}:
        return base + "attn.qkv_proj"
    return None


def _block_diag(mats: list[np.ndarray]) -> np.ndarray:
    rows = sum(m.shape[0] for m in mats)
    cols = sum(m.shape[1] for m in mats)
    out = np.zeros((rows, cols), dtype=np.float32)
    r = c = 0
    for m in mats:
        rr, cc = m.shape
        out[r : r + rr, c : c + cc] = m
        r += rr
        c += cc
    return out


def _swap_fc1_halves(b: np.ndarray) -> np.ndarray:
    """Swap gate/value row blocks on LoRA-B for Diffusers→native FC1."""
    if b.ndim != 2 or b.shape[0] % 2 != 0:
        return b
    half = b.shape[0] // 2
    return np.concatenate([b[half:], b[:half]], axis=0)


def convert_diffusers_h3_lora(
    source: Path,
    destination: Path | None = None,
    *,
    swap_fc1_halves: bool = True,
) -> Path:
    """Rewrite a Diffusers H3 LoRA into native Comfy/h3.c key layout.

    Returns the destination path (created or already native).
    """
    source = Path(source).expanduser().resolve()
    if destination is None:
        destination = source.parent / "native" / f"{source.stem}_comfy.safetensors"
    else:
        destination = Path(destination).expanduser().resolve()

    if destination.is_file() and destination.stat().st_size > 64:
        # Reuse prior conversion when newer than the source.
        if destination.stat().st_mtime >= source.stat().st_mtime:
            return destination

    tensors, meta = _read_tensors(source)
    keys = list(tensors.keys())
    if is_native_h3_lora(keys):
        log.info("LoRA already native: %s", source)
        return source
    if not is_diffusers_h3_lora(keys):
        raise ValueError(
            f"not a Diffusers MiniMax-H3 LoRA (no split to_q / transformer_blocks): {source}"
        )

    # Group by block stem for QKV fusion
    qkv_groups: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    singles: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    stems_seen: set[str] = set()
    for key in keys:
        stem = _stem(key)
        if stem is None or stem in stems_seen:
            continue
        stems_seen.add(stem)
        ab = _get_ab(tensors, stem)
        if ab is None:
            continue
        if stem.endswith(".attn.to_q") or stem.endswith(".attn.to_k") or stem.endswith(".attn.to_v"):
            block = stem.rsplit(".attn.to_", 1)[0]
            which = stem.rsplit(".", 1)[-1]  # to_q / to_k / to_v
            qkv_groups.setdefault(block, {})[which] = ab
        else:
            singles[stem] = ab

    out: dict[str, np.ndarray] = {}
    fused_qkv = 0
    for block, parts in sorted(qkv_groups.items()):
        if not all(k in parts for k in ("to_q", "to_k", "to_v")):
            missing = {"to_q", "to_k", "to_v"} - set(parts)
            raise ValueError(f"incomplete QKV LoRA at {block}: missing {missing}")
        a_q, b_q = parts["to_q"]
        a_k, b_k = parts["to_k"]
        a_v, b_v = parts["to_v"]
        # A [rank, in] stacked → [3*rank, in]; B block-diag → [3*out, 3*rank]
        a_f = np.concatenate([a_q, a_k, a_v], axis=0)
        b_f = _block_diag([b_q, b_k, b_v])
        native = _native_prefix(block + ".attn.to_q")
        if not native:
            raise ValueError(f"cannot map QKV block {block}")
        out[f"{native}.lora_A.weight"] = a_f
        out[f"{native}.lora_B.weight"] = b_f
        # alpha = fused rank (matches h3_lora.c default when alpha tensor absent)
        out[f"{native}.alpha"] = np.array([float(a_f.shape[0])], dtype=np.float32)
        fused_qkv += 1

    renamed = 0
    for stem, (a, b) in singles.items():
        native = _native_prefix(stem)
        if not native:
            log.warning("skip unmapped Diffusers LoRA stem: %s", stem)
            continue
        if native.endswith("mlp.fc1") and swap_fc1_halves:
            b = _swap_fc1_halves(b)
        out[f"{native}.lora_A.weight"] = a
        out[f"{native}.lora_B.weight"] = b
        out[f"{native}.alpha"] = np.array([float(a.shape[0])], dtype=np.float32)
        renamed += 1

    if not out:
        raise RuntimeError(f"conversion produced no tensors from {source}")

    meta = dict(meta)
    meta["h3_ws_converted_from"] = str(source.name)
    meta["h3_ws_layout"] = "native_comfy_qkv_proj"
    meta["h3_ws_fc1_halves_swapped"] = "1" if swap_fc1_halves else "0"
    _write_tensors(destination, out, meta)
    log.info(
        "Converted Diffusers H3 LoRA → native (%d QKV blocks, %d other) → %s",
        fused_qkv,
        renamed,
        destination,
    )
    return destination


def ensure_native_h3_lora(path: Path, *, swap_fc1_halves: bool = True) -> Path:
    """Return a path h3.c can fuse; convert Diffusers packs when needed."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    tensors, _ = _read_tensors(path)
    keys = list(tensors.keys())
    if is_native_h3_lora(keys):
        return path
    if is_diffusers_h3_lora(keys):
        return convert_diffusers_h3_lora(path, swap_fc1_halves=swap_fc1_halves)
    # Unknown layout — let h3.c try as-is.
    return path
