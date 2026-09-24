"""H3 DiT LoRA catalog, Hugging Face download, and h3.c --lora wiring."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from h3_paths import hf_hub_cache, hf_snapshot, load_desktop_config, writable_root

log = logging.getLogger("h3")

TAOMATE_REPO = "Robert1212star/TaoMate-H3-3Step-ComfyUI"
TAOMATE_FILE = "taomate_h3_3step_comfy.safetensors"
TAOMATE_GUIDANCE = (
    "TaoMate 3-step distill LoRA for MiniMax-H3 (ComfyUI package). "
    "Quality tiers collapse to 3 steps at strength ~0.8. Prefer this over "
    "older 8-step turbo recipes when you want the fastest turnaround."
)

TUTU_REPO = "tutututututu/Tutu-MiniMax-H3-AudioVideo-20to8-NFE-LoRA"
TUTU_GUIDANCE = (
    "Tutu 20→8 NFE LoRA for FL2VA. Trained for 8 Euler steps at strength 0.8. "
    "h3.c uses its own 8-step shifted schedule (not ComfyUI ManualSigmas). "
    "SSD streaming is off while a LoRA is enabled. Rebuild h3 after pull "
    "(scripts/build_h3.sh) so --lora is fused at DiT load."
)

# Continuity-style filename hints — scale / steps when scanning on-disk LoRAs.
# Only hints; any other .safetensors under models/loras still appears in the catalog.
_DISK_LORA_HINTS: tuple[dict[str, Any], ...] = (
    {
        "match": re.compile(r"taomate", re.I),
        "scale": 0.8,
        "steps": 3,
        "guidance": "TaoMate 3-step distill. Quality tiers collapse to 3 steps.",
        "turbo": True,
    },
    {
        "match": re.compile(r"tutu|20to8[_-]?nfe", re.I),
        "scale": 0.8,
        "steps": 8,
        "guidance": TUTU_GUIDANCE,
        "turbo": True,
    },
    {
        "match": re.compile(r"(?:lightx2v|[_-]turbo[_-]|\bdmd\b|8step).*(?:h3|minimax)|(?:h3|minimax).*(?:lightx2v|turbo|dmd|8step)", re.I),
        "scale": 0.6,
        "steps": 8,
        "guidance": "Turbo / distill adapter — try 6–8 steps at the suggested strength.",
        "turbo": True,
    },
    {
        "match": re.compile(r"(?:fasth3|3step|3[_-]?nfe).*(?:h3|minimax)|(?:h3|minimax).*(?:fasth3|3step|3[_-]?nfe)", re.I),
        "scale": 0.8,
        "steps": 3,
        "guidance": "Fast / low-NFE adapter — try 3–4 steps.",
        "turbo": True,
    },
)

BUILTIN_LORAS: list[dict[str, Any]] = [
    {
        "id": "taomate_h3_3step",
        "label": "TaoMate H3 3-step",
        "spec": (
            f"https://huggingface.co/{TAOMATE_REPO}/resolve/main/{TAOMATE_FILE}"
        ),
        "scale": 0.8,
        "steps": 3,
        "layers": 50,
        "reuse": 1,
        "guidance": TAOMATE_GUIDANCE,
        "turbo": True,
    },
    {
        "id": "tutu_20to8_nfe_step100",
        "label": "Tutu 20→8 NFE (step 100)",
        "spec": (
            f"https://huggingface.co/{TUTU_REPO}/resolve/main/"
            "comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000100-bf16-comfyui.safetensors"
        ),
        "scale": 0.8,
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "guidance": TUTU_GUIDANCE,
        "turbo": True,
    },
    {
        "id": "tutu_20to8_nfe_step200",
        "label": "Tutu 20→8 NFE (step 200)",
        "spec": (
            f"https://huggingface.co/{TUTU_REPO}/resolve/main/"
            "comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000200-bf16-comfyui.safetensors"
        ),
        "scale": 0.8,
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "guidance": TUTU_GUIDANCE,
        "turbo": True,
    },
    {
        "id": "tutu_20to8_nfe_step300",
        "label": "Tutu 20→8 NFE (step 300)",
        "spec": (
            f"https://huggingface.co/{TUTU_REPO}/resolve/main/"
            "comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000300-bf16-comfyui.safetensors"
        ),
        "scale": 0.8,
        "steps": 8,
        "layers": 50,
        "reuse": 1,
        "guidance": TUTU_GUIDANCE,
        "turbo": True,
    },
]


def lora_cache_dir() -> Path:
    env = os.environ.get("H3_LORA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    cfg = load_desktop_config()
    configured = str(cfg.get("lora_dir") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    # Prefer the clone's models/loras when the user pointed the app at a git checkout.
    repo = str(cfg.get("repo_root") or "").strip()
    if repo:
        candidate = Path(repo).expanduser() / "models" / "loras"
        if candidate.is_dir():
            return candidate.resolve()
    model = str(cfg.get("model_dir") or "").strip()
    if model:
        mp = Path(model).expanduser()
        if mp.name == "MiniMax-H3" and mp.parent.name == "models":
            candidate = mp.parent / "loras"
            if candidate.is_dir():
                return candidate.resolve()
    return writable_root() / "models" / "loras"


# Hub repo names that look like LoRA / turbo adapters (not the base MiniMax tree).
_HF_LORA_REPO_RE = re.compile(
    r"lora|turbo|tutu|taomate|nfe|distill|fasth3|lightx|20to8|3step|8step",
    re.I,
)
_HF_BASE_REPO_RE = re.compile(
    r"MiniMaxAI--MiniMax-H3$|Comfy-Org--MiniMax-H3$",
    re.I,
)
# TaoMate is ~2.4 GB; keep a little headroom, skip giant base DiT shards.
_MAX_LORA_BYTES = 3_600_000_000
_MIN_LORA_BYTES = 1_000_000


def _lora_search_roots() -> list[Path]:
    """Directories that may already hold downloaded LoRA files."""
    roots: list[Path] = []
    seen: set[str] = set()

    def push(path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return
        key = str(resolved)
        if key in seen or not resolved.is_dir():
            return
        seen.add(key)
        roots.append(resolved)

    push(lora_cache_dir())
    push(writable_root() / "models" / "loras")
    cfg = load_desktop_config()
    repo = str(cfg.get("repo_root") or "").strip()
    if repo:
        push(Path(repo) / "models" / "loras")
    model = str(cfg.get("model_dir") or "").strip()
    if model:
        mp = Path(model)
        if mp.name == "MiniMax-H3" and mp.parent.name == "models":
            push(mp.parent / "loras")
    home = Path.home()
    for base in (
        home / "Documents" / "git" / "h3-ws" / "models" / "loras",
        home / "git" / "h3-ws" / "models" / "loras",
        Path("/Users") / home.name / "Documents" / "git" / "h3-ws" / "models" / "loras",
        home / "Documents" / "ComfyUI" / "models" / "loras",
        home / "ComfyUI-Shared" / "models" / "loras",
    ):
        push(base)

    # Hugging Face hub snapshots that expose a loras/ folder, plus LoRA-named repos.
    hub = hf_hub_cache()
    if hub.is_dir():
        try:
            for child in hub.iterdir():
                if not child.name.startswith("models--"):
                    continue
                snap = hf_snapshot(child.name, hub=hub)
                if snap is None:
                    continue
                push(snap / "loras")
                push(snap / "models" / "loras")
                low = child.name.lower()
                if _HF_LORA_REPO_RE.search(low) and not _HF_BASE_REPO_RE.search(child.name):
                    push(snap)
        except OSError:
            pass
    return roots


def normalize_lora_spec(spec: str) -> str:
    raw = (spec or "").strip().strip("'\"")
    if not raw:
        return ""
    raw = raw.replace("/blob/", "/resolve/", 1)
    if raw.startswith("hf.co/"):
        raw = "https://huggingface.co/" + raw[len("hf.co/") :]
    if raw.startswith("huggingface.co/"):
        raw = "https://" + raw
    parsed = urlparse(raw)
    if parsed.netloc in {"hf.co", "www.hf.co"}:
        raw = f"https://huggingface.co{parsed.path}"
        if parsed.query:
            raw += f"?{parsed.query}"
    if re.fullmatch(r"[\w.-]+/[\w.-]+", raw):
        raw = f"https://huggingface.co/{raw}/resolve/main"
    return raw


def _parse_hf_resolve(url: str) -> tuple[str, str, str] | None:
    parsed = urlparse(url)
    if "huggingface.co" not in parsed.netloc:
        return None
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) >= 5 and parts[2] == "resolve":
        repo = f"{parts[0]}/{parts[1]}"
        revision = parts[3]
        filename = "/".join(parts[4:])
        return repo, revision, filename
    # Handle /repo/name/resolve/revision without filename
    if len(parts) == 4 and parts[2] == "resolve":
        return f"{parts[0]}/{parts[1]}", parts[3], ""
    if len(parts) == 2:
        return f"{parts[0]}/{parts[1]}", "main", ""
    return None


def _label_for_spec(spec: str) -> str:
    parsed = _parse_hf_resolve(spec)
    if parsed and parsed[2]:
        return Path(parsed[2]).name
    path = Path(spec)
    if path.suffix:
        return path.name
    return spec[-48:] if len(spec) > 48 else spec


def _is_usable(path: Path | None) -> bool:
    return bool(path and path.is_file() and path.stat().st_size > 64)


def _default_filename(repo: str) -> str:
    """When a repo-only Turbo/LoRA URL has no file path, pick the catalog default."""
    if repo == TUTU_REPO:
        return (
            "comfyui/tutu-t8-minimax-h3-av-20to8-nfe-lora-step000100-"
            "bf16-comfyui.safetensors"
        )
    # Fall back to the first builtin's leaf if somehow empty.
    parsed = _parse_hf_resolve(normalize_lora_spec(str(BUILTIN_LORAS[0]["spec"])))
    if parsed and parsed[2]:
        return parsed[2]
    return Path(BUILTIN_LORAS[0]["spec"]).name


def find_cached_lora(repo: str, filename: str) -> Path | None:
    """Return an on-disk LoRA matching repo/filename without downloading.

    Hugging Face layouts keep files under a nested path (e.g. ``comfyui/…``).
    Older flat-leaf copies and App Support caches are also accepted.
    """
    if not filename:
        filename = _default_filename(repo)
    repo_key = repo.replace("/", "__")
    leaf = Path(filename).name
    for root in _lora_search_roots():
        package = root / repo_key
        for candidate in (
            package / filename,  # nested: …/comfyui/foo.safetensors
            package / leaf,  # flat: …/foo.safetensors
        ):
            if _is_usable(candidate):
                return candidate
        if package.is_dir():
            for hit in package.rglob(leaf):
                if ".cache" in hit.parts:
                    continue
                if _is_usable(hit):
                    return hit
    return None


def resolve_lora_path(spec: str) -> Path:
    """Local path or Hugging Face download into models/loras/."""
    spec = normalize_lora_spec(spec)
    if not spec:
        raise ValueError("LoRA spec is empty")
    local = Path(spec).expanduser()
    if local.is_file():
        return local.resolve()
    parsed = _parse_hf_resolve(spec)
    if parsed is None:
        if spec.startswith(("http://", "https://")):
            raise ValueError(f"only Hugging Face LoRA URLs are supported: {spec}")
        raise FileNotFoundError(f"LoRA file not found: {spec}")
    repo, revision, filename = parsed
    if not filename:
        filename = _default_filename(repo)
    hit = find_cached_lora(repo, filename)
    if hit is not None:
        log.info("Reusing cached LoRA %s", hit)
        return hit
    dest_dir = lora_cache_dir() / repo.replace("/", "__")
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Keep the HF relative path so future lookups match the clone layout.
    dest = dest_dir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _is_usable(dest):
        return dest
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download LoRAs") from exc
    log.info("Downloading LoRA %s (%s) …", repo, filename)
    local_path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        revision=revision,
        local_dir=str(dest_dir),
    )
    path = Path(local_path).resolve()
    if path != dest and _is_usable(path) and not _is_usable(dest):
        try:
            dest.unlink(missing_ok=True)
            dest.symlink_to(path)
        except OSError:
            return path
    if not _is_usable(path):
        raise RuntimeError(f"downloaded LoRA is empty: {path}")
    return path


def lora_cached_path(spec: str) -> Path | None:
    spec = normalize_lora_spec(spec)
    local = Path(spec).expanduser()
    if local.is_file():
        return local.resolve()
    parsed = _parse_hf_resolve(spec)
    if parsed is None:
        return None
    repo, _rev, filename = parsed
    return find_cached_lora(repo, filename or _default_filename(repo))


def ensure_lora(spec: str) -> dict[str, Any]:
    normalized = normalize_lora_spec(spec)
    cached = lora_cached_path(normalized)
    if cached is not None:
        return {"ok": True, "spec": normalized, "path": str(cached), "cached": True}
    path = resolve_lora_path(normalized)
    return {"ok": True, "spec": normalized, "path": str(path), "cached": False}


def read_custom_loras(output_dir: Path) -> list[dict[str, Any]]:
    from web_ui import read_web_settings

    raw = read_web_settings(output_dir).get("custom_loras")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        spec = normalize_lora_spec(str(item.get("spec") or ""))
        lid = str(item.get("id") or "").strip()
        if not spec or not lid:
            continue
        try:
            scale = float(item.get("scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        out.append(
            {
                "id": lid,
                "label": str(item.get("label") or "").strip() or _label_for_spec(spec),
                "spec": spec,
                "scale": scale,
                "custom": True,
            }
        )
    return out


def write_custom_loras(output_dir: Path, entries: list[dict[str, Any]]) -> None:
    from web_ui import read_web_settings, write_web_settings

    data = read_web_settings(output_dir)
    data["custom_loras"] = [
        {
            "id": e["id"],
            "label": e.get("label") or _label_for_spec(str(e.get("spec") or "")),
            "spec": normalize_lora_spec(str(e.get("spec") or "")),
            "scale": float(e.get("scale") or 1.0),
        }
        for e in entries
        if e.get("id") and e.get("spec")
    ]
    write_web_settings(output_dir, data)


def _hint_for_lora_name(name: str) -> dict[str, Any]:
    for hint in _DISK_LORA_HINTS:
        if hint["match"].search(name):
            return hint
    return {}


def _hint_for_lora_path(path: Path) -> dict[str, Any]:
    """Match hints against filename and parent package (HF / clone layouts)."""
    hay = f"{path.name} {path.parent.name} {path.as_posix()}"
    return _hint_for_lora_name(hay)


def _human_lora_label(path: Path, root: Path) -> str:
    leaf = path.stem.replace("_", " ").replace("-", " ")
    try:
        rel = path.relative_to(root)
        package = rel.parts[0] if len(rel.parts) > 1 else ""
    except ValueError:
        package = path.parent.name
    package = package.replace("__", "/").replace("_", " ")
    if package and package.lower() not in leaf.lower():
        short = package.split("/")[-1] if "/" in package else package
        if len(short) > 28:
            short = short[:25] + "…"
        return f"{leaf} · {short}"
    return leaf


def _local_lora_id(path: Path) -> str:
    # Stable id from absolute path so selection survives restarts.
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", path.stem).strip("_").lower()[:40]
    return f"local_{stem}_{digest}"


def _entry_for_lora_path(path: Path, *, root: Path | None = None, source: str = "disk") -> dict[str, Any]:
    hint = _hint_for_lora_path(path)
    label_root = root if root is not None else path.parent
    entry: dict[str, Any] = {
        "id": _local_lora_id(path),
        "label": _human_lora_label(path, label_root),
        "spec": str(path.resolve()),
        "scale": float(hint.get("scale") or 0.8),
        "local": True,
        "cached": True,
        "path": str(path.resolve()),
        "source": source,
    }
    if hint.get("steps") is not None:
        entry["steps"] = int(hint["steps"])
    if hint.get("guidance"):
        entry["guidance"] = hint["guidance"]
    if hint.get("turbo"):
        entry["turbo"] = True
        entry["layers"] = 50
        entry["reuse"] = 1
    return entry


def _collect_loras_under(root: Path, *, source: str, min_bytes: int | None = None) -> dict[str, Path]:
    """Deduped leaf→path map for one tree (prefer nested comfyui/)."""
    chosen: dict[str, Path] = {}
    if not root.is_dir():
        return chosen
    floor = _MIN_LORA_BYTES if min_bytes is None and source == "hf" else (min_bytes if min_bytes is not None else 64)
    try:
        paths = sorted(root.rglob("*.safetensors"))
    except OSError:
        return chosen
    for path in paths:
        if ".cache" in path.parts or any(p.startswith(".") for p in path.parts):
            continue
        if not _is_usable(path):
            continue
        try:
            sz = path.stat().st_size
        except OSError:
            continue
        if sz < floor or sz > _MAX_LORA_BYTES:
            continue
        try:
            rel = path.relative_to(root)
            package = rel.parts[0] if len(rel.parts) > 1 else "_"
        except ValueError:
            package = path.parent.name
        key = f"{package}/{path.name}"
        prev = chosen.get(key)
        if prev is None:
            chosen[key] = path
            continue
        if "comfyui" in path.parts and "comfyui" not in prev.parts:
            chosen[key] = path
        elif path.stat().st_mtime > prev.stat().st_mtime and "comfyui" not in prev.parts:
            chosen[key] = path
    return chosen


def iter_hf_hub_lora_files() -> list[tuple[Path, Path]]:
    """``(file, repo_snap_root)`` for LoRA-sized weights in the HF hub cache."""
    hub = hf_hub_cache()
    if not hub.is_dir():
        return []
    out: list[tuple[Path, Path]] = []
    try:
        children = list(hub.iterdir())
    except OSError:
        return []
    for child in children:
        if not child.name.startswith("models--"):
            continue
        if _HF_BASE_REPO_RE.search(child.name):
            continue
        low = child.name.lower()
        # Prefer repos whose name says LoRA/turbo; also accept *minimax*h3* adapters.
        if not (_HF_LORA_REPO_RE.search(low) or ("minimax" in low and "h3" in low and "lora" in low)):
            # Still pick up a bare loras/ folder under any snapshot.
            snap = hf_snapshot(child.name, hub=hub)
            if snap is None:
                continue
            for sub in (snap / "loras", snap / "models" / "loras"):
                for path in _collect_loras_under(sub, source="hf").values():
                    out.append((path, sub))
            continue
        snap = hf_snapshot(child.name, hub=hub)
        if snap is None:
            continue
        for path in _collect_loras_under(snap, source="hf").values():
            # Skip obvious non-LoRA shard names inside adapter repos.
            leaf = path.name.lower()
            if leaf.startswith("model-") and "of-" in leaf:
                continue
            if leaf in {"model.safetensors", "diffusion_pytorch_model.safetensors"}:
                # Single-file adapters sometimes use this — keep if small enough already.
                pass
            out.append((path, snap))
    return out


def scan_disk_loras() -> list[dict[str, Any]]:
    """Every usable LoRA under models/loras **and** the Hugging Face hub cache.

    Mirrors Continuity: do not pin a fixed list — whatever the user already
    downloaded (checkout or ``~/.cache/huggingface/hub``) shows up.
    """
    seen_paths: set[str] = set()
    out: list[dict[str, Any]] = []

    for root in _lora_search_roots():
        # Skip whole HF hub snaps here — handled below with better labeling.
        if "huggingface" in str(root).lower() and "hub" in str(root).lower():
            if root.name != "loras":
                continue
        for path in _collect_loras_under(root, source="disk").values():
            key = str(path.resolve())
            if key in seen_paths:
                continue
            seen_paths.add(key)
            out.append(_entry_for_lora_path(path, root=root, source="disk"))

    for path, snap in iter_hf_hub_lora_files():
        key = str(path.resolve())
        if key in seen_paths:
            continue
        seen_paths.add(key)
        entry = _entry_for_lora_path(path, root=snap, source="hf")
        # Clarify hub origin in the label when not already obvious.
        if "hf" not in entry["label"].lower() and "hub" not in entry["label"].lower():
            entry["label"] = f"{entry['label']} · HF hub"
        out.append(entry)

    out.sort(key=lambda e: str(e.get("label") or "").lower())
    return out


def lora_catalog(output_dir: Path | None = None) -> list[dict[str, Any]]:
    """Built-ins (download recipes) + on-disk LoRAs + user-added customs."""
    presets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()

    # 1) Everything already on disk under models/loras (primary — no pinning).
    for entry in scan_disk_loras():
        presets.append(entry)
        seen_ids.add(str(entry["id"]))
        seen_paths.add(str(Path(entry["spec"]).resolve()))

    # 2) Built-in HF recipes — skip if that file is already listed from disk.
    for item in BUILTIN_LORAS:
        preset = dict(item)
        cached = lora_cached_path(str(preset["spec"]))
        preset["cached"] = cached is not None
        if cached is not None:
            key = str(cached.resolve())
            if key in seen_paths:
                # Disk entry already covers this recipe; keep turbo metadata
                # on the local card via hints instead of duplicating.
                continue
            seen_paths.add(key)
        if preset["id"] in seen_ids:
            continue
        presets.append(preset)
        seen_ids.add(str(preset["id"]))

    # 3) User-added custom specs (settings.json).
    if output_dir is not None:
        for entry in read_custom_loras(output_dir):
            if entry["id"] in seen_ids:
                continue
            cached = lora_cached_path(entry["spec"])
            if cached is not None and str(cached.resolve()) in seen_paths:
                continue
            entry["cached"] = cached is not None
            presets.append(entry)
            seen_ids.add(entry["id"])

    for preset in presets:
        if "cached" not in preset:
            preset["cached"] = lora_cached_path(str(preset["spec"])) is not None
    return presets


def materialize_loras(specs: list[tuple[str, float]]) -> list[dict[str, Any]]:
    """Download each spec and return ``{spec, path, scale}`` dicts."""
    out: list[dict[str, Any]] = []
    for spec, scale in specs:
        path = resolve_lora_path(spec)
        out.append({"spec": spec, "path": str(path), "scale": float(scale)})
    return out


def parse_lora_specs(raw: Any, catalog: list[dict[str, Any]] | None = None) -> list[tuple[str, float]]:
    """Body `loras` / `lora_specs`: [{id, spec, scale}] or [[spec, scale], ...]."""
    if raw is None:
        return []
    items = raw
    if isinstance(raw, dict):
        items = [raw]
    if not isinstance(items, list):
        raise ValueError("loras must be a list")
    catalog = catalog or []
    by_id = {str(p.get("id")): p for p in catalog}
    out: list[tuple[str, float]] = []
    for item in items:
        spec = ""
        scale = 1.0
        if isinstance(item, dict):
            lid = str(item.get("id") or "").strip()
            if lid and lid in by_id:
                spec = str(by_id[lid].get("spec") or "")
                scale = float(item.get("scale", by_id[lid].get("scale", 1.0)))
            else:
                spec = str(item.get("spec") or item.get("path") or "")
                scale = float(item.get("scale", 1.0))
        elif isinstance(item, (list, tuple)) and item:
            spec = str(item[0])
            scale = float(item[1]) if len(item) > 1 else 1.0
        elif isinstance(item, str):
            spec = item
        spec = normalize_lora_spec(spec)
        if not spec:
            continue
        out.append((spec, max(0.0, float(scale))))
    return out
