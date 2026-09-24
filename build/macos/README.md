# macOS build assets for H3-WS

Ported from the [AceForge](https://github.com/audiohacking/AceForge) release pipeline.

| File | Role |
|------|------|
| `H3-WS.icns` | App icon |
| `codesign.sh` | Ad-hoc (default) or Developer ID signing |
| `entitlements.plist` | Network, JIT, library validation off |
| `pyinstaller_hooks/` | Extra PyInstaller hooks if needed |

## Local build

```bash
# From repo root, after web/dist + h3 binary exist:
pip install -r requirements_macos.txt
python -m PyInstaller H3WS.spec --clean --noconfirm
cp dist/H3-WS.app/Contents/MacOS/H3-WS_bin dist/H3-WS.app/Contents/MacOS/H3-WS
chmod +x dist/H3-WS.app/Contents/MacOS/H3-WS
./build/macos/codesign.sh dist/H3-WS.app
open dist/H3-WS.app
```

Ad-hoc signing (`MACOS_SIGNING_IDENTITY=-`) avoids “app is damaged”. First launch on another Mac: **right-click → Open**.

## Existing model weights

On first launch the app asks for (or auto-detects) an existing `models/MiniMax-H3` tree from a git clone so users are never forced to re-download ~134 GB. Choice is stored in:

`~/Library/Application Support/H3-WS/config.json`

To reset setup: delete that file (or set `"setup_done": false`).
