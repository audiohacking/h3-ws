#!/bin/bash
# Fallback launcher if the .app is quarantined oddly — still prefers H3-WS.app.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -d "$DIR/H3-WS.app" ]]; then
  open "$DIR/H3-WS.app"
  exit 0
fi
if [[ -d "/Applications/H3-WS.app" ]]; then
  open "/Applications/H3-WS.app"
  exit 0
fi
echo "H3-WS.app not found next to this script or in /Applications."
exit 1
