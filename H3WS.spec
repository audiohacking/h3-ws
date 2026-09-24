# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for H3-WS macOS .app (AceForge pattern)

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

block_cipher = None
spec_root = Path(SPECPATH)

icon_path = spec_root / "build" / "macos" / "H3-WS.icns"
web_dist = spec_root / "web" / "dist"
h3_dir = spec_root / "third_party" / "h3.c"
scripts_dir = spec_root / "scripts"
third_party = spec_root / "third_party"

datas = []
binaries = []

# Prebuilt Continuity UI
if web_dist.is_dir():
    datas.append((str(web_dist), "web/dist"))

# Native h3 binary + Metal shaders (cwd for h3 must be this folder).
# Use datas (not binaries) so PyInstaller does not mangle "h3.c" → "h3__dot__c".
if (h3_dir / "h3").is_file():
    datas.append((str(h3_dir / "h3"), "third_party/h3.c"))
if (h3_dir / "h3_shaders.metal").is_file():
    datas.append((str(h3_dir / "h3_shaders.metal"), "third_party/h3.c"))

# PyAV shim + download helper
if (scripts_dir / "h3-av").is_file():
    datas.append((str(scripts_dir / "h3-av"), "scripts"))
if (spec_root / "h3_av.py").is_file():
    datas.append((str(spec_root / "h3_av.py"), "."))
if (scripts_dir / "download_model.py").is_file():
    datas.append((str(scripts_dir / "download_model.py"), "scripts"))
if (third_party / "taehv.py").is_file():
    datas.append((str(third_party / "taehv.py"), "third_party"))

# VERSION stamp for About / diagnostics
version_file = spec_root / "VERSION"
if version_file.is_file():
    datas.append((str(version_file), "."))

# Native extension libs
for pkg in ("av", "tokenizers"):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception as exc:
        print(f"[H3WS.spec] collect_dynamic_libs({pkg}): {exc}")

for pkg in ("av", "certifi", "huggingface_hub"):
    try:
        datas += collect_data_files(pkg)
    except Exception as exc:
        print(f"[H3WS.spec] collect_data_files({pkg}): {exc}")

hiddenimports = [
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "fastapi",
    "starlette",
    "multipart",
    "sse_starlette",
    "websockets",
    "av",
    "PIL",
    "numpy",
    "huggingface_hub",
    "mcp",
    "webview",
    "server",
    "web_ui",
    "h3_backend",
    "h3_av",
    "h3_paths",
    "h3_bootstrap",
    "h3_preview",
    "h3_lora",
    "h3_media",
    "h3_crop",
    "h3_session",
    "h3_refine",
    "h3_update",
    "torch",
    "safetensors",
    *collect_submodules("uvicorn"),
    *collect_submodules("fastapi"),
    *collect_submodules("starlette"),
]

a = Analysis(
    ["h3_desktop.py"],
    pathex=[str(spec_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["build/macos/pyinstaller_hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Keep Analysis from importing broken/unrelated contrib hooks
        # (nltk → sklearn → pandas/numpy ABI mismatch on this machine).
        "nltk",
        "sklearn",
        "pandas",
        "matplotlib",
        "scipy",
        "tkinter.test",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="H3-WS_bin",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="H3-WS",
)

app = BUNDLE(
    coll,
    name="H3-WS.app",
    icon=str(icon_path) if icon_path.exists() else None,
    bundle_identifier="com.h3ws.app",
    info_plist={
        "CFBundleName": "H3-WS",
        "CFBundleDisplayName": "H3-WS",
        "CFBundleShortVersionString": "0.1.0",
        "CFBundleVersion": "0.1.0",
        "CFBundleExecutable": "H3-WS",
        "CFBundlePackageType": "APPL",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "13.0",
        "LSUIElement": False,
        "LSBackgroundOnly": False,
        "NSAppTransportSecurity": {
            "NSAllowsLocalNetworking": True,
        },
        "NSMicrophoneUsageDescription": "H3-WS may attach audio references for generation.",
        "NSUserNotificationsUsageDescription": "H3-WS can notify you when a video is ready or if generation fails.",
    },
)
