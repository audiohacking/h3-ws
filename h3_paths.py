"""Scratch paths for temp media — never macOS TMPDIR (/var/folders/...)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_APP_SUPPORT_NAME = "H3-WS"


def is_frozen() -> bool:
    """True when running inside a PyInstaller (or similar) bundle."""
    return bool(getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"))


def resource_root() -> Path:
    """Read-only tree: repo checkout, or PyInstaller ``_MEIPASS``."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent


def writable_root() -> Path:
    """User-writable data root (models, outputs, uploads, logs)."""
    env = os.environ.get("H3_WS_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if is_frozen():
        return (Path.home() / "Library" / "Application Support" / _APP_SUPPORT_NAME).resolve()
    return Path(__file__).resolve().parent


def ensure_writable_tree() -> Path:
    """Create App Support (or repo) subdirs used by the desktop/server."""
    root = writable_root()
    for rel in (
        "models/MiniMax-H3",
        "models/vae_approx",
        "models/loras",
        "web_outputs",
        "web_uploads",
        "logs",
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


# Back-compat: most call sites mean "resource root" (code + bundled assets).
REPO_ROOT = resource_root()

_configured_root: Path | None = None


def configure_scratch_root(root: Path | str | None) -> None:
    """Prefer ``output_dir/.scratch``, else ``/tmp/h3-ws``."""
    global _configured_root
    if root is None:
        _configured_root = None
        return
    _configured_root = Path(root).expanduser().resolve()


def scratch_root() -> Path:
    if _configured_root is not None:
        return _configured_root
    env = os.environ.get("H3_SCRATCH_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path("/tmp/h3-ws")


def ensure_scratch_root() -> Path:
    root = scratch_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def mk_scratch_dir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(ensure_scratch_root())))


def mk_scratch_file(prefix: str, suffix: str) -> tuple[int, str]:
    return tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=str(ensure_scratch_root()))


def default_h3_bin() -> Path:
    env = os.environ.get("H3_BIN", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    root = resource_root()
    # Prefer normal path; PyInstaller mangles "h3.c" → "h3__dot__c" for binary dests.
    for rel in (
        ("third_party", "h3.c", "h3"),
        ("third_party", "h3__dot__c", "h3"),
    ):
        candidate = root.joinpath(*rel)
        if candidate.is_file():
            return candidate
    return root / "third_party" / "h3.c" / "h3"


def h3_process_cwd(h3_bin: Path | str | None = None) -> Path:
    """h3.c opens ``h3_shaders.metal`` from the process working directory."""
    binary = Path(h3_bin) if h3_bin is not None else default_h3_bin()
    return binary.expanduser().resolve().parent


def debug_console() -> bool:
    """Dump h3 progress to the server console unless ``DEBUG=false``."""
    raw = os.environ.get("DEBUG")
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


_last_console_line = ""


def console_h3(message: str, *args: object) -> None:
    """Print an h3 line to the server log when console debug is on."""
    if not debug_console():
        return
    import logging

    text = message % args if args else message
    text = " ".join(str(text).split())
    if not text:
        return
    global _last_console_line
    if text == _last_console_line:
        return
    _last_console_line = text
    logging.getLogger("h3").info("%s", text[:300])


def default_model_dir() -> Path:
    env = os.environ.get("H3_MODEL_DIR", "").strip() or os.environ.get(
        "H3_WS_MODEL_DIR", ""
    ).strip()
    if env:
        return Path(env).expanduser().resolve()
    cfg = load_desktop_config()
    configured = str(cfg.get("model_dir") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return writable_root() / "models" / "MiniMax-H3"


def default_output_dir() -> Path:
    env = os.environ.get("H3_WS_OUTPUT_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    cfg = load_desktop_config()
    configured = str(cfg.get("output_dir") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return writable_root() / "web_outputs"


def default_upload_dir() -> Path:
    env = os.environ.get("H3_WS_UPLOAD_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    cfg = load_desktop_config()
    configured = str(cfg.get("upload_dir") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return writable_root() / "web_uploads"


def default_logs_dir() -> Path:
    env = os.environ.get("H3_WS_LOG_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    if is_frozen():
        return (Path.home() / "Library" / "Logs" / _APP_SUPPORT_NAME).resolve()
    return writable_root() / "logs"


def desktop_config_path() -> Path:
    return writable_root() / "config.json"


def load_desktop_config() -> dict:
    path = desktop_config_path()
    if not path.is_file():
        return {}
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_desktop_config(data: dict) -> Path:
    import json

    root = ensure_writable_tree()
    path = root / "config.json"
    merged = load_desktop_config()
    merged.update(data)
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return path


def looks_like_minimax_h3(path: Path) -> bool:
    """True when ``path`` looks like a usable MiniMax-H3 native tree."""
    p = Path(path)
    if not p.is_dir():
        return False
    # Either FL2VA or Ref2VA transformer present is enough to count as "already downloaded".
    markers = (
        p / "FL2VA" / "transformer",
        p / "FL2VA",
        p / "Ref2VA" / "transformer",
        p / "Ref2VA",
    )
    return any(m.is_dir() for m in markers)


def hf_hub_cache() -> Path:
    """Hugging Face hub cache root (``~/.cache/huggingface/hub`` by default)."""
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return Path(raw).expanduser().resolve()
    home = os.environ.get("HF_HOME", "").strip()
    if home:
        return (Path(home).expanduser() / "hub").resolve()
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def hf_snapshot(repo_dirname: str, *, hub: Path | None = None) -> Path | None:
    """Current snapshot dir for a ``models--org--name`` cache tree, or None."""
    root = (hub or hf_hub_cache()) / repo_dirname
    ref = root / "refs" / "main"
    if ref.is_file():
        try:
            snap = root / "snapshots" / ref.read_text(encoding="utf-8").strip()
        except OSError:
            snap = None
        if snap is not None and snap.is_dir():
            return snap
    snaps = root / "snapshots"
    if not snaps.is_dir():
        return None
    try:
        kids = [p for p in snaps.iterdir() if p.is_dir()]
    except OSError:
        return None
    if not kids:
        return None
    return max(kids, key=lambda p: p.stat().st_mtime)


def discover_hf_hub_model_dirs(*, limit: int = 8) -> list[Path]:
    """Native MiniMax-H3 trees already sitting in the Hugging Face hub cache."""
    found: list[Path] = []
    seen: set[str] = set()

    def push(candidate: Path) -> None:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            return
        key = str(resolved)
        if key in seen or not looks_like_minimax_h3(resolved):
            return
        seen.add(key)
        found.append(resolved)

    # Official native checkpoint + any models--*MiniMax*H3* snapshot that has FL2VA.
    for dirname in (
        "models--MiniMaxAI--MiniMax-H3",
        "models--Comfy-Org--MiniMax-H3",
    ):
        snap = hf_snapshot(dirname)
        if snap is None:
            continue
        push(snap)
        push(snap / "MiniMax-H3")
        if len(found) >= limit:
            return found[:limit]

    hub = hf_hub_cache()
    if hub.is_dir():
        try:
            for child in sorted(hub.iterdir()):
                if not child.name.startswith("models--"):
                    continue
                low = child.name.lower()
                if "minimax" not in low or "h3" not in low:
                    continue
                if "lora" in low:
                    continue
                snap = hf_snapshot(child.name, hub=hub)
                if snap is None:
                    continue
                push(snap)
                push(snap / "MiniMax-H3")
                if len(found) >= limit:
                    break
        except OSError:
            pass
    return found[:limit]


def discover_existing_model_dirs(*, limit: int = 12) -> list[Path]:
    """Find prior git-clone / custom MiniMax-H3 trees so the app never re-downloads."""
    found: list[Path] = []
    seen: set[str] = set()

    def push(candidate: Path) -> None:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            return
        key = str(resolved)
        if key in seen:
            return
        if not looks_like_minimax_h3(resolved):
            return
        seen.add(key)
        found.append(resolved)

    # Explicit env always first.
    for key in ("H3_MODEL_DIR", "H3_WS_MODEL_DIR"):
        raw = os.environ.get(key, "").strip()
        if raw:
            push(Path(raw))

    cfg = load_desktop_config()
    for key in ("model_dir", "repo_root"):
        raw = str(cfg.get(key) or "").strip()
        if not raw:
            continue
        p = Path(raw)
        push(p if p.name == "MiniMax-H3" else p / "models" / "MiniMax-H3")

    # Hugging Face hub cache — common download location on the machine.
    for p in discover_hf_hub_model_dirs(limit=limit):
        push(p)
        if len(found) >= limit:
            return found[:limit]

    # Common clone locations relative to the user's home / Documents / git trees.
    home = Path.home()
    candidates = [
        writable_root() / "models" / "MiniMax-H3",
        home / "Documents" / "git" / "h3-ws" / "models" / "MiniMax-H3",
        home / "git" / "h3-ws" / "models" / "MiniMax-H3",
        home / "src" / "h3-ws" / "models" / "MiniMax-H3",
        home / "h3-ws" / "models" / "MiniMax-H3",
        home / "Developer" / "h3-ws" / "models" / "MiniMax-H3",
        Path("/Users") / home.name / "Documents" / "git" / "h3-ws" / "models" / "MiniMax-H3",
    ]
    # Sibling of Application Support is unlikely; also scan ~/Documents/git/*/models/MiniMax-H3 lightly.
    git_root = home / "Documents" / "git"
    if git_root.is_dir():
        try:
            for child in sorted(git_root.iterdir()):
                push(child / "models" / "MiniMax-H3")
                if len(found) >= limit:
                    break
        except OSError:
            pass
    for c in candidates:
        push(c)
        if len(found) >= limit:
            break
    return found[:limit]


def apply_desktop_config_env() -> None:
    """Export persisted path overrides into the process environment."""
    cfg = load_desktop_config()
    mapping = {
        "model_dir": "H3_MODEL_DIR",
        "output_dir": "H3_WS_OUTPUT_DIR",
        "upload_dir": "H3_WS_UPLOAD_DIR",
        "data_dir": "H3_WS_DATA_DIR",
        "h3_bin": "H3_BIN",
        "lora_dir": "H3_LORA_DIR",
        "taeh3_path": "H3_TAEH3",
    }
    for cfg_key, env_key in mapping.items():
        val = str(cfg.get(cfg_key) or "").strip()
        if val and not os.environ.get(env_key, "").strip():
            os.environ[env_key] = val
    # repo_root → derive models/outputs/uploads/loras if not set explicitly
    repo = str(cfg.get("repo_root") or "").strip()
    if repo:
        root = Path(repo).expanduser()
        if not os.environ.get("H3_MODEL_DIR", "").strip() and looks_like_minimax_h3(
            root / "models" / "MiniMax-H3"
        ):
            os.environ["H3_MODEL_DIR"] = str((root / "models" / "MiniMax-H3").resolve())
        if not os.environ.get("H3_WS_OUTPUT_DIR", "").strip() and (root / "web_outputs").is_dir():
            os.environ["H3_WS_OUTPUT_DIR"] = str((root / "web_outputs").resolve())
        if not os.environ.get("H3_WS_UPLOAD_DIR", "").strip() and (root / "web_uploads").is_dir():
            os.environ["H3_WS_UPLOAD_DIR"] = str((root / "web_uploads").resolve())
        loras = root / "models" / "loras"
        if not os.environ.get("H3_LORA_DIR", "").strip() and loras.is_dir():
            os.environ["H3_LORA_DIR"] = str(loras.resolve())
        taeh3 = root / "models" / "vae_approx" / "taeh3.safetensors"
        if not os.environ.get("H3_TAEH3", "").strip() and taeh3.is_file():
            os.environ["H3_TAEH3"] = str(taeh3.resolve())


def default_h3_av() -> Path:
    env = os.environ.get("H3_AV", "").strip()
    if env:
        return Path(env).expanduser()
    # Frozen: prefer a writable wrapper that re-enters the app binary.
    if is_frozen():
        wrapper = writable_root() / "bin" / "h3-av"
        if wrapper.is_file():
            return wrapper
    return resource_root() / "scripts" / "h3-av"


def install_frozen_h3_av_wrapper() -> Path | None:
    """Write ``H3_AV`` shim that re-enters the frozen binary with ``--run-h3-av``."""
    if not is_frozen():
        return None
    bindir = writable_root() / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    wrapper = bindir / "h3-av"
    exe = Path(sys.executable).resolve()
    body = (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f'exec "{exe}" --run-h3-av "$@"\n'
    )
    try:
        wrapper.write_text(body, encoding="utf-8")
        wrapper.chmod(0o755)
    except OSError:
        return None
    os.environ["H3_AV"] = str(wrapper)
    return wrapper


_SHIM_NAMES = {"h3-av", "h3-ffmpeg", "h3-ffprobe", "h3_av.py"}


def _looks_like_media_shim(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if resolved.name in _SHIM_NAMES:
        return True
    try:
        if resolved == (resource_root() / "h3_av.py").resolve():
            return True
        if resolved == (resource_root() / "scripts" / "h3-av").resolve():
            return True
    except OSError:
        pass
    return False


def _force_pyav() -> bool:
    return os.environ.get("H3_AV_FORCE_PYAV", "").strip().lower() in {"1", "true", "yes"}


def real_ffmpeg() -> str | None:
    """System ffmpeg that actually starts, never the PyAV shim."""
    return _real_tool("ffmpeg")


def real_ffprobe() -> str | None:
    return _real_tool("ffprobe")


_tool_ok: dict[str, bool] = {}


def _tool_runs(path: str) -> bool:
    """Reject Homebrew binaries that crash on missing dylibs (dyld)."""
    cached = _tool_ok.get(path)
    if cached is not None:
        return cached
    try:
        proc = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=8,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        _tool_ok[path] = False
        return False
    text = f"{proc.stdout or ''}{proc.stderr or ''}"
    ok = proc.returncode == 0 and "Library not loaded" not in text and "dyld[" not in text
    if not ok:
        logging.getLogger("h3").warning(
            "skipping broken %s (%s)",
            path,
            "dyld" if "Library not loaded" in text or "dyld[" in text else f"exit {proc.returncode}",
        )
    _tool_ok[path] = ok
    return ok


def _real_tool(name: str) -> str | None:
    if _force_pyav():
        return None
    folders: list[str] = []
    for part in os.environ.get("PATH", "").split(os.pathsep):
        if part:
            folders.append(part)
    folders.extend(["/opt/homebrew/bin", "/usr/local/bin"])
    seen: set[str] = set()
    for folder in folders:
        candidate = Path(folder) / name
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if not os.path.isfile(candidate) or not os.access(candidate, os.X_OK):
            continue
        if _looks_like_media_shim(candidate):
            continue
        resolved = str(candidate)
        if not _tool_runs(resolved):
            continue
        return resolved
    return None


def ffmpeg_shim_bindir() -> Path:
    """Directory with ``ffmpeg`` / ``ffprobe`` names pointing at h3-av.

    Unpatched h3.c looks up those names on PATH. Patched builds prefer H3_AV.
    """
    # Always writable — frozen bundles cannot create symlinks under _MEIPASS.
    bindir = writable_root() / "bin" / ".h3-av-bin"
    bindir.mkdir(parents=True, exist_ok=True)
    shim = default_h3_av().resolve()
    for name in ("ffmpeg", "ffprobe"):
        link = bindir / name
        try:
            if link.is_symlink() and link.resolve() == shim:
                continue
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(shim)
        except OSError:
            continue
    return bindir


def h3_media_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """h3.c mux/decode: a working system ffmpeg, otherwise the PyAV shim.

    Homebrew binaries that crash on missing dylibs (``Library not loaded``)
    are skipped. ``H3_AV`` overrides both tools in patched h3.c, so it must
    be unset when the real binaries are used. A warm session captures this
    env at spawn — restart the server after muxer changes.
    """
    import sys

    env = dict(base if base is not None else os.environ)
    env["H3_PYTHON"] = sys.executable
    ffmpeg = real_ffmpeg()
    ffprobe = real_ffprobe()
    shim = str(default_h3_av())
    if ffmpeg and not _force_pyav():
        env.pop("H3_AV", None)
        env["H3_FFMPEG"] = ffmpeg
        env["H3_FFPROBE"] = ffprobe or shim
        return env
    env["H3_AV"] = shim
    env["H3_FFMPEG"] = shim
    env["H3_FFPROBE"] = shim
    bindir = str(ffmpeg_shim_bindir())
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    return env
