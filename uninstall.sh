#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo ""
echo "============================================================"
echo "  TIGA Hunt — Uninstall"
echo "============================================================"
echo ""
echo "This will:"
echo "  - Stop TIGA server and admin processes"
echo "  - Remove desktop shortcuts"
echo ""
echo "Your indexed data in tiga_work/ is kept by default."
echo ""

read -r -p "Continue with uninstall? [y/N] " ans
[[ "${ans,,}" == "y" ]] || { echo "Cancelled."; exit 0; }

REMOVE_VENV=""
REMOVE_DATA=""
read -r -p "Also remove .venv? [y/N] " ans
[[ "${ans,,}" == "y" ]] && REMOVE_VENV="--venv"
read -r -p "Also DELETE tiga_work index data? [y/N] " ans
[[ "${ans,,}" == "y" ]] && REMOVE_DATA="--data"

source .venv/bin/activate
python tiga.py uninstall --yes $REMOVE_VENV $REMOVE_DATA

echo ""
echo "Uninstall complete. You can delete this folder manually:"
echo "  $(pwd)"
