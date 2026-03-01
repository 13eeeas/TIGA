#!/usr/bin/env bash
# TIGA Hunt — Update from GitHub (Linux / macOS)
# Usage: bash update.sh
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}  TIGA Hunt — Update from GitHub${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""

# ---------------------------------------------------------------------------
# Git repo check
# ---------------------------------------------------------------------------

git rev-parse --git-dir >/dev/null 2>&1 || err "Not a git repository. Run from the TIGA folder."

CUR_BRANCH=$(git branch --show-current)
CUR_HASH=$(git rev-parse --short HEAD)
echo "      Current branch: $CUR_BRANCH @ $CUR_HASH"

# ---------------------------------------------------------------------------
# Check for uncommitted changes
# ---------------------------------------------------------------------------

STASHED=0
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    echo ""
    warn "You have uncommitted local changes:"
    git status --short
    echo ""
    read -rp "      Stash local changes before updating? [Y/n] " ANSWER
    if [[ "${ANSWER,,}" != "n" ]]; then
        git stash push -m "auto-stash before TIGA update $(date '+%Y-%m-%d %H:%M')"
        ok "Changes stashed. Run 'git stash pop' to restore them."
        STASHED=1
    else
        info "Proceeding without stashing. Merge conflicts may occur."
    fi
fi

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

echo ""
info "Fetching latest changes from origin..."
git fetch origin || {
    [[ $STASHED == 1 ]] && git stash pop --quiet 2>/dev/null || true
    err "Could not reach remote. Check network connection."
}

# ---------------------------------------------------------------------------
# Already up to date?
# ---------------------------------------------------------------------------

LOCAL_H=$(git rev-parse HEAD)
REMOTE_H=$(git rev-parse "origin/$CUR_BRANCH" 2>/dev/null || echo "")

if [[ "$LOCAL_H" == "$REMOTE_H" ]]; then
    ok "Already up to date — nothing to pull."
else
    # Show incoming commits
    echo ""
    info "Changes coming in:"
    git log --oneline HEAD.."origin/$CUR_BRANCH"
    echo ""

    # Pull
    if ! git pull origin "$CUR_BRANCH"; then
        warn "git pull failed — possible merge conflict."
        info "Resolve conflicts manually, then run: git pull origin $CUR_BRANCH"
        [[ $STASHED == 1 ]] && info "Your stash is still available: git stash pop"
        exit 1
    fi

    NEW_HASH=$(git rev-parse --short HEAD)
    ok "Updated to $NEW_HASH"
fi

# ---------------------------------------------------------------------------
# Restore stash
# ---------------------------------------------------------------------------

if [[ $STASHED == 1 ]]; then
    echo ""
    info "Restoring your stashed local changes..."
    git stash pop && ok "Local changes restored." \
        || warn "Stash pop had conflicts. Resolve manually: git stash pop"
fi

# ---------------------------------------------------------------------------
# Reinstall dependencies
# ---------------------------------------------------------------------------

echo ""
info "Updating dependencies..."
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install -r requirements.txt --quiet && ok "Dependencies up to date" \
        || warn "pip install had issues — check manually"
else
    warn "No .venv found — run setup.sh first."
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}  Update Complete!${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""
echo "  Restart TIGA Hunt to apply changes:"
echo "    python tiga.py serve"
echo ""
