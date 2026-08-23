#!/usr/bin/env bash
# TIGA Hunt — One-Click Install and Run (Linux / macOS)
# Usage: bash install.sh
set -euo pipefail

BOLD="\033[1m"
GREEN="\033[32m"
RESET="\033[0m"

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}  TIGA Hunt — One-Click Install and Run${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "      First run — running setup..."
    bash setup.sh
else
    echo -e "${GREEN}[OK]${RESET}  Virtual environment ready"
fi

echo ""
echo "      Starting TIGA Hunt..."
if [[ -f "run.sh" ]]; then
    bash run.sh
else
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python tiga.py index
    python tiga.py serve &
    SERVER_PID=$!
    sleep 2
    python tiga.py ui &
    UI_PID=$!
    echo ""
    echo -e "${GREEN}TIGA Hunt is running${RESET}"
    echo "  UI:  http://localhost:8501"
    echo "  API: http://localhost:7860"
    echo "  Press Ctrl+C to stop"
    trap "kill $SERVER_PID $UI_PID 2>/dev/null" EXIT
    wait
fi
