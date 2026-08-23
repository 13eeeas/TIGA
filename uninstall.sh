#!/usr/bin/env bash
# TIGA Hunt — Uninstall (Linux / macOS)
# Usage: bash uninstall.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "  TIGA Hunt — Uninstall"
echo "============================================================"
echo ""

read -rp "Also remove local data (tiga_work — indexes, config, logs)? [y/N] " ANSWER
REMOVE_DATA=0
if [[ "${ANSWER,,}" == "y" ]]; then
    REMOVE_DATA=1
fi

echo ""
echo "Stopping TIGA processes..."
pkill -f "tiga.py serve" 2>/dev/null || true
pkill -f "tiga.py ui" 2>/dev/null || true

echo "Removing virtual environment..."
rm -rf ".venv"

if [[ "$REMOVE_DATA" == "1" ]]; then
    echo "Removing local data..."
    rm -rf "tiga_work"
fi

echo ""
echo "============================================================"
echo "  TIGA Hunt uninstalled."
echo "  Source files remain in: $SCRIPT_DIR"
echo "  Delete this folder manually if you want a full removal."
echo "============================================================"
echo ""
