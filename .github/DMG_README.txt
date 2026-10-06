H3-WS for Apple Silicon
=======================

1. Drag H3-WS.app into Applications.
2. First open: right-click the app → Open → Open
   (Gatekeeper warning is expected — this build is ad-hoc signed, not notarized.)
3. If you already downloaded MiniMax-H3 weights in a git clone, point the
   first-run dialog at that models/MiniMax-H3 folder. Nothing will be re-downloaded.
4. Otherwise the Models panel opens automatically — download FL2VA (~134 GB).
   Ref2VA (~62 GB) is required for reference mode (default).
5. Optional light downloads happen automatically when needed:
   - TAEH3 (~22 MB) for live denoise previews on first launch
   - SAM 3.1 (~3.3 GB) the first time you enable Faces

Logs: ~/Library/Logs/H3-WS/
Config: ~/Library/Application Support/H3-WS/config.json
Models: ~/Library/Application Support/H3-WS/models/  (or your chosen folder)
