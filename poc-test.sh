#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
echo ""
echo "TIGA POC Test — choose projects → index → stress → export"
echo ""
python tiga.py poc-test run
