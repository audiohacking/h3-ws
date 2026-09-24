# h3-ws

> H3 powered videofentanyl

Local **MiniMax-H3** video+audio generation on Apple Silicon (Metal), driven by native [h3.c](https://github.com/antirez/h3.c).

<img width="900" height="581" alt="image" src="https://github.com/user-attachments/assets/8d26e446-f881-404a-86e7-d1e7b12fae93" />


## Download (macOS · Apple Silicon)

**End users:** install the prebuilt app from GitHub Releases — you do not need to clone or compile.

1. Open [**Releases**](https://github.com/audiohacking/h3-ws/releases) and download the latest **H3-WS-macOS.dmg** (or `.app` zip if attached).
2. Open the DMG and drag **H3-WS.app** into Applications.
3. First launch: **right-click → Open → Open** (ad-hoc signed / not notarized — Gatekeeper may warn; this is expected, not “damaged”).
4. **Weights:** on first launch, point the dialog at an existing `models/MiniMax-H3` tree from a previous clone if you have one (nothing is re-downloaded). Otherwise use the in-app **Models** panel to fetch FL2VA (~134 GB). Choice is saved in `~/Library/Application Support/H3-WS/config.json`.

Logs: `~/Library/Logs/H3-WS/`.

See [`ROADMAP.md`](ROADMAP.md) for the product plan. Contributors: [`DEV.md`](DEV.md) and the **Developers** section below.

## Using the app

The UI opens in **Ref2VA** mode (add an image / video / audio reference to enable Generate). Switch modes as needed.

Prompts should describe scene, action, camera, look, and audio. What you type is what H3 sees — there is no cloud rewriter.

| Mode | Use |
|------|-----|
| Text to video+audio | Prompt only |
| First / last / first+last frame | Image anchors |
| Ordered references (Ref2VA) | Image, silent video, video, video+audio, and/or audio |

First/last-frame anchors cannot be mixed with Ref2VA references. For references, prompt with `Picture 1`, `Video 1`, `Audio 1` in list order. Standalone audio must accompany an image or video (≤9 images, ≤3 videos, ≤3 audio).

**Quality:** Four-step · Aggressive · Fast (default) · Balanced · Close.

**Canvas** (dropdown, grouped): **1:1** 256×256 preview, 512×512 (safest), 512×512 with 384 or 320 internal, 768×768. **16:9** 1024×576 (exact) and 1248×704 (largest under the cap). **7:4** 1344×768 max landscape. **4:3** 1024×768. **5:4** 960×768. **4:5** 768×960 Instagram/social. **3:4** 768×1024. **9:16** 576×1024 (exact Stories/Reels/TikTok) and 704×1248 (largest). **4:7** 768×1344 max portrait. Each side is a multiple of 32; pixel count cannot exceed 768×1344. H3-Base is a 768p model; 512×512 is the usual working size.

Output is 24 fps video with 32 kHz stereo audio. Clip length is any integer **1s–15s** (each snapped up to a legal H3 frame count `5+17n` @ 24 fps).

---

## Developers

Building and running from a git checkout is for development, packaging, and CI. Day-to-day use should prefer the Release DMG above.

### Requirements

- Apple Silicon Mac (Metal)
- Python 3.12+ (`requirements.txt`; PyAV `av` for all media, including the h3.c mux/decode shim)
- Node.js 18+ (Web UI)
- Xcode command-line tools (`make`)
- Enough unified memory for the model (~40 GB peak). Use `--ssd-streaming` if you have under ~64 GB.

### Install from source

```bash
git clone https://github.com/audiohacking/h3-ws.git
cd h3-ws

uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install -r requirements.txt

./scripts/build_h3.sh
python scripts/download_model.py          # ~134 GB FL2VA only (enough to generate)
# python scripts/download_model.py --with-ref2va   # +~62 GB Ref2VA transformer

cd web && npm install && npm run build && cd ..
```

`third_party/h3.c` is a **vendored local fork** of [antirez/h3.c](https://github.com/antirez/h3.c) (PyAV shim, LoRA fold, INT8, `--preview-latent`). It lives in this repo — no git submodules. Do **not** `hf download MiniMaxAI/MiniMax-H3` without filters; use `scripts/download_model.py` instead.

### Run (dev server)

```bash
source .venv/bin/activate
python server.py
```

Then open **http://127.0.0.1:8765/**. Or double-click `Start H3-WS.command` / `H3-WS.command`. For an embedded window with crash-restart: `python h3_desktop.py` (needs `pywebview`; see `requirements_macos.txt`).

Default generate is 512×512, **1s** (22 frames), **fast** quality. Lower RAM if a run is killed for memory (slower denoise):

```bash
python server.py --ssd-streaming
```

### Package the macOS `.app` locally

```bash
pip install -r requirements_macos.txt
python -m PyInstaller H3WS.spec --clean --noconfirm
cp dist/H3-WS.app/Contents/MacOS/H3-WS_bin dist/H3-WS.app/Contents/MacOS/H3-WS
chmod +x dist/H3-WS.app/Contents/MacOS/H3-WS
./build/macos/codesign.sh dist/H3-WS.app
open dist/H3-WS.app
```

CI builds the DMG on **Release** publish or via **workflow_dispatch** (`.github/workflows/build-release.yml`) and uploads it to that Release.
