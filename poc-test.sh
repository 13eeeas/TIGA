#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "[ERROR] Run bash START-HERE.sh first."
    exit 1
fi

source .venv/bin/activate

if ! python tools/office_setup.py check; then
    python tools/office_setup.py configure
fi

echo ""
echo "TIGA POC Test — choose projects → index → stress → export"
echo ""
python tiga.py poc-test run
echo ""
echo "Export: tiga_work/poc_test/exports/"
