#!/usr/bin/env bash
# TIGA Hunt — Safe update from GitHub (Linux / macOS)
# Fast-forward only. Never reset, merge, stash, or discard office work.
# Usage: bash update.sh [--sha <commit>] [--require-health] [--dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=python3
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    PYTHON=python
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "[ERROR] Python 3 not found. Install Python, then rerun."
    echo "        Office data in tiga_work/ was not changed."
    exit 1
fi

# Run the updater with *system* Python (stdlib only). The office venv is
# used only inside the updater for pip install, so a broken venv cannot
# brick the next update.
exec "$PYTHON" "$SCRIPT_DIR/tools/safe_update.py" "$@"
