#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
python -c "from tools.launcher_util import ensure_services; ensure_services()"
sleep 3
python -c "from tools.launcher_util import open_portal; open_portal('admin', start=False)"
