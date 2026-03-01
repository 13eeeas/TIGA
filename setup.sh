#!/usr/bin/env bash
# TIGA Hunt — One-Click Installer (Linux / macOS)
# Usage: bash setup.sh
set -euo pipefail

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

ok()   { echo -e "${GREEN}[OK]${RESET}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }
err()  { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }
info() { echo -e "      $*"; }

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}  TIGA Hunt — One-Click Installer${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 1. Python check (3.10+)
# ---------------------------------------------------------------------------

PYTHON=python3
$PYTHON --version >/dev/null 2>&1 || PYTHON=python
$PYTHON --version >/dev/null 2>&1 || err "Python 3 not found. Install from https://www.python.org"

PY_VER=$($PYTHON --version 2>&1 | awk '{print $2}')
ok "Python $PY_VER"

# ---------------------------------------------------------------------------
# 2. Virtual environment
# ---------------------------------------------------------------------------

if [ ! -d ".venv" ]; then
    info "Creating virtual environment..."
    $PYTHON -m venv .venv || err "Failed to create .venv"
    ok ".venv created"
else
    ok ".venv already exists"
fi

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate

# ---------------------------------------------------------------------------
# 3. pip + dependencies
# ---------------------------------------------------------------------------

echo ""
info "Upgrading pip..."
pip install --upgrade pip --quiet

info "Installing TIGA dependencies (may take a few minutes)..."
pip install -r requirements.txt || err "pip install failed"
ok "Dependencies installed"

info "Installing sentence-transformers (cross-encoder reranker, optional)..."
pip install sentence-transformers --quiet && ok "sentence-transformers installed" \
    || warn "sentence-transformers skipped — reranker will be disabled"

# ---------------------------------------------------------------------------
# 4. Ollama check + install
# ---------------------------------------------------------------------------

echo ""
if ! command -v ollama &>/dev/null; then
    info "Ollama not found. Installing..."
    if [[ "$(uname)" == "Darwin" ]]; then
        if command -v brew &>/dev/null; then
            brew install ollama && ok "Ollama installed via Homebrew" || warn "Homebrew install failed"
        else
            info "Downloading Ollama for macOS..."
            curl -fsSL https://ollama.com/install.sh | sh && ok "Ollama installed" \
                || warn "Could not install Ollama. Install manually: https://ollama.com"
        fi
    else
        # Linux
        info "Downloading and running Ollama install script..."
        curl -fsSL https://ollama.com/install.sh | sh && ok "Ollama installed" \
            || warn "Could not install Ollama. Install manually: https://ollama.com"
    fi
else
    OL_VER=$(ollama --version 2>&1 | head -1)
    ok "Ollama found: $OL_VER"
fi

# ---------------------------------------------------------------------------
# 5. Pull Ollama models
# ---------------------------------------------------------------------------

echo ""
if command -v ollama &>/dev/null; then
    # Ensure Ollama server is running
    if ! ollama list &>/dev/null; then
        info "Starting Ollama server in background..."
        ollama serve &>/dev/null &
        sleep 3
    fi

    info "Pulling nomic-embed-text (embedding model, ~274 MB)..."
    ollama pull nomic-embed-text && ok "nomic-embed-text ready" \
        || warn "Could not pull nomic-embed-text. Run: ollama pull nomic-embed-text"

    info "Pulling mistral (chat model, ~4 GB)..."
    ollama pull mistral && ok "mistral ready" \
        || warn "Could not pull mistral. Run: ollama pull mistral"
else
    warn "Ollama not available — skipping model pull."
    info "Install Ollama from https://ollama.com then run:"
    info "  ollama pull nomic-embed-text"
    info "  ollama pull mistral"
fi

# ---------------------------------------------------------------------------
# 6. Default config
# ---------------------------------------------------------------------------

echo ""
if [ ! -f "tiga_work/config.yaml" ]; then
    info "Creating default config.yaml..."
    python tiga.py init && ok "config.yaml created at tiga_work/config.yaml" \
        || warn "Could not create config.yaml. Run: python tiga.py init"
else
    ok "config.yaml already exists"
fi

# ---------------------------------------------------------------------------
# 7. Work directories
# ---------------------------------------------------------------------------

python -c "from config import cfg; cfg.ensure_dirs(); print('[OK]  Work dirs ready')" 2>/dev/null \
    || warn "Could not create work dirs (config may need editing first)"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}  Setup Complete!${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""
echo "  NEXT STEPS:"
echo ""
echo "  1. Edit your archive paths in:"
echo "       tiga_work/config.yaml"
echo "     Set index_roots to your project folders, e.g.:"
echo "       index_roots:"
echo '         - "/mnt/projects/WOHA"'
echo ""
echo "  2. Start TIGA Hunt:"
echo "       source .venv/bin/activate"
echo "       python tiga.py serve"
echo ""
echo "  3. Open your browser to: http://localhost:7860"
echo ""
echo "  4. (Optional) Quick scan before indexing:"
echo '       python tiga.py scan "/mnt/projects/WOHA" --phases'
echo ""
