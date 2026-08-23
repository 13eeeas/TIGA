#!/usr/bin/env bash
# TIGA Hunt — one-click launcher (starts services + opens portal)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "[ERROR] Run setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -c "from tools.launcher_util import ensure_services; ensure_services()"
sleep 2

if command -v xdg-open &>/dev/null; then
  xdg-open "http://127.0.0.1:7860/launcher"
elif command -v open &>/dev/null; then
  open "http://127.0.0.1:7860/launcher"
else
  echo "Open http://127.0.0.1:7860/launcher in your browser"
fi
