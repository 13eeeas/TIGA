#!/usr/bin/env bash
# TIGA Hunt — Bootstrap Installer (Linux / macOS)
# Clones from GitHub, installs dependencies, and starts TIGA.
# Usage: bash bootstrap.sh [install-directory]
set -euo pipefail

DEFAULT_DIR="${HOME}/TIGA"
INSTALL_DIR="${1:-$DEFAULT_DIR}"
REPO="https://github.com/13eeeas/TIGA.git"

echo ""
echo "============================================================"
echo "  TIGA Hunt — Bootstrap Installer"
echo "  Clones from GitHub, installs, and starts TIGA"
echo "============================================================"
echo ""

command -v git >/dev/null 2>&1 || { echo "[ERROR] Git not found."; exit 1; }

if [[ -d "$INSTALL_DIR/.git" ]]; then
    echo "[INFO] TIGA already installed at $INSTALL_DIR"
    cd "$INSTALL_DIR"
    bash install.sh
    exit 0
fi

if [[ -e "$INSTALL_DIR" ]]; then
    echo "[ERROR] Path exists but is not a TIGA install: $INSTALL_DIR"
    exit 1
fi

echo "Cloning TIGA from GitHub to $INSTALL_DIR ..."
git clone "$REPO" "$INSTALL_DIR"
cd "$INSTALL_DIR"
echo ""
echo "[OK] Clone complete. Running install..."
bash install.sh
