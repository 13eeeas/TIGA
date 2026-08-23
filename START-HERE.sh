#!/usr/bin/env bash
# TIGA Hunt — one-click setup + POC test (Linux / macOS)
set -euo pipefail

cd "$(dirname "$0")"

echo ""
echo "============================================================"
echo "  TIGA Hunt — Setup + POC Test (one click)"
echo "============================================================"
echo ""

if [ ! -d ".venv" ]; then
    echo "[Step 1/3] Installing TIGA (first time — may take a while)..."
    bash setup.sh
else
    echo "[Step 1/3] Install — already done"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo ""
echo "[Step 2/3] Project folders..."
python tools/office_setup.py configure

echo ""
echo "[Step 3/3] POC retrieval test..."
python tiga.py poc-test run || true

echo ""
echo "  Export zip:  tiga_work/poc_test/exports/"
echo "  Daily use:   bash launcher.sh"
echo "  Re-test:     bash poc-test.sh"
