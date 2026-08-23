"""
core/lifecycle.py — TIGA install/update/uninstall lifecycle helpers.

Used by server API endpoints and install scripts.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
GITHUB_REPO = "https://github.com/13eeeas/TIGA.git"
GITHUB_WEB = "https://github.com/13eeeas/TIGA"

_lifecycle_lock = threading.Lock()
_update_state: dict[str, Any] = {
    "running": False,
    "phase": "idle",
    "log": [],
    "error": None,
    "started_at": None,
    "finished_at": None,
    "result": None,
}


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with _lifecycle_lock:
        _update_state["log"].append(line)
        if len(_update_state["log"]) > 200:
            _update_state["log"] = _update_state["log"][-200:]


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd or REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args])


def _venv_python() -> Path:
    if platform.system() == "Windows":
        return REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    return REPO_ROOT / ".venv" / "bin" / "python"


def _venv_pip() -> list[str]:
    py = _venv_python()
    if py.exists():
        return [str(py), "-m", "pip"]
    return [sys.executable, "-m", "pip"]


def get_install_info() -> dict[str, Any]:
    """Return install metadata for the UI."""
    info: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "platform": platform.system(),
        "python": sys.version.split()[0],
        "venv_ready": _venv_python().exists(),
        "is_git_repo": (REPO_ROOT / ".git").exists(),
        "branch": None,
        "commit": None,
        "commit_short": None,
        "remote_url": None,
        "update_available": False,
        "commits_behind": 0,
        "github_url": GITHUB_WEB,
        "github_repo": GITHUB_REPO,
    }

    if not info["is_git_repo"]:
        return info

    branch = _git("branch", "--show-current")
    if branch.returncode == 0:
        info["branch"] = branch.stdout.strip() or "HEAD"

    commit = _git("rev-parse", "HEAD")
    if commit.returncode == 0:
        info["commit"] = commit.stdout.strip()
        info["commit_short"] = info["commit"][:7]

    remote = _git("remote", "get-url", "origin")
    if remote.returncode == 0:
        info["remote_url"] = remote.stdout.strip()

    behind = _commits_behind(info["branch"])
    info["commits_behind"] = behind
    info["update_available"] = behind > 0
    return info


def _commits_behind(branch: str | None) -> int:
    if not branch:
        return 0
    fetch = _run(["git", "fetch", "origin", branch], timeout=120)
    if fetch.returncode != 0:
        return 0
    local = _git("rev-parse", "HEAD")
    remote = _git("rev-parse", f"origin/{branch}")
    if local.returncode != 0 or remote.returncode != 0:
        return 0
    if local.stdout.strip() == remote.stdout.strip():
        return 0
    count = _git("rev-list", "--count", f"HEAD..origin/{branch}")
    if count.returncode == 0 and count.stdout.strip().isdigit():
        return int(count.stdout.strip())
    return 1


def get_update_status() -> dict[str, Any]:
    with _lifecycle_lock:
        return {
            "running": _update_state["running"],
            "phase": _update_state["phase"],
            "log": list(_update_state["log"]),
            "error": _update_state["error"],
            "started_at": _update_state["started_at"],
            "finished_at": _update_state["finished_at"],
            "result": _update_state["result"],
        }


def start_update(*, stash: bool = True) -> dict[str, Any]:
    """Kick off a background GitHub update. Returns immediately."""
    with _lifecycle_lock:
        if _update_state["running"]:
            return {"status": "busy", "message": "Update already in progress."}
        if not (REPO_ROOT / ".git").exists():
            return {"status": "error", "message": "Not a git repository — cannot update from GitHub."}
        _update_state.update({
            "running": True,
            "phase": "starting",
            "log": [],
            "error": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "result": None,
        })

    thread = threading.Thread(target=_run_update, args=(stash,), daemon=True)
    thread.start()
    return {"status": "started", "message": "Update started."}


def _run_update(stash: bool) -> None:
    try:
        with _lifecycle_lock:
            _update_state["phase"] = "checking"

        branch_res = _git("branch", "--show-current")
        branch = branch_res.stdout.strip() or "master"
        _log(f"Branch: {branch}")

        dirty = (
            _git("diff", "--quiet").returncode != 0
            or _git("diff", "--cached", "--quiet").returncode != 0
        )
        stashed = False
        if dirty:
            if stash:
                with _lifecycle_lock:
                    _update_state["phase"] = "stashing"
                _log("Stashing local changes...")
                stash_res = _git(
                    "stash", "push", "-m",
                    f"auto-stash before TIGA update {datetime.now().isoformat()}",
                )
                if stash_res.returncode == 0:
                    stashed = True
                    _log("Local changes stashed.")
                else:
                    _log(f"Stash warning: {stash_res.stderr.strip()}")
            else:
                _log("Proceeding with uncommitted local changes.")

        with _lifecycle_lock:
            _update_state["phase"] = "fetching"
        _log("Fetching from origin...")
        fetch = _git("fetch", "origin", timeout=180)
        if fetch.returncode != 0:
            raise RuntimeError(fetch.stderr.strip() or "git fetch failed")
        _log("Fetch complete.")

        local = _git("rev-parse", "HEAD").stdout.strip()
        remote = _git("rev-parse", f"origin/{branch}").stdout.strip()
        if local == remote:
            _log("Already up to date.")
            result = {"updated": False, "commit_short": local[:7], "commits_pulled": 0}
        else:
            with _lifecycle_lock:
                _update_state["phase"] = "pulling"
            _log("Pulling latest changes...")
            pull = _git("pull", "origin", branch, timeout=180)
            if pull.returncode != 0:
                raise RuntimeError(pull.stderr.strip() or pull.stdout.strip() or "git pull failed")
            for line in (pull.stdout or "").splitlines():
                if line.strip():
                    _log(line.strip())
            new_hash = _git("rev-parse", "--short", "HEAD").stdout.strip()
            _log(f"Updated to {new_hash}.")
            result = {"updated": True, "commit_short": new_hash, "commits_pulled": 1}

        if stashed:
            with _lifecycle_lock:
                _update_state["phase"] = "restoring"
            _log("Restoring stashed changes...")
            pop = _git("stash", "pop")
            if pop.returncode != 0:
                _log(f"Stash pop warning: {pop.stderr.strip() or pop.stdout.strip()}")
            else:
                _log("Stashed changes restored.")

        with _lifecycle_lock:
            _update_state["phase"] = "dependencies"
        _log("Updating Python dependencies...")
        pip = _run([*_venv_pip(), "install", "-r", "requirements.txt", "--quiet"], timeout=600)
        if pip.returncode != 0:
            _log(f"pip warning: {pip.stderr.strip() or pip.stdout.strip()}")
        else:
            _log("Dependencies up to date.")

        with _lifecycle_lock:
            _update_state["phase"] = "done"
            _update_state["result"] = result
            _update_state["error"] = None
    except Exception as exc:
        _log(f"ERROR: {exc}")
        with _lifecycle_lock:
            _update_state["phase"] = "failed"
            _update_state["error"] = str(exc)
    finally:
        with _lifecycle_lock:
            _update_state["running"] = False
            _update_state["finished_at"] = datetime.now(timezone.utc).isoformat()


def launch_uninstall(*, remove_data: bool = False, server_pid: int | None = None) -> dict[str, Any]:
    """
    Spawn a detached uninstall script that waits for the server to exit,
    then removes the virtualenv and optionally local data.
    """
    pid = server_pid or os.getpid()
    script_path = _write_uninstall_script(remove_data=remove_data, server_pid=pid)
    _spawn_detached(script_path)
    return {
        "status": "scheduled",
        "message": "Uninstall scheduled. TIGA will shut down shortly.",
        "remove_data": remove_data,
        "script": str(script_path),
    }


def _write_uninstall_script(*, remove_data: bool, server_pid: int) -> Path:
    logs_dir = REPO_ROOT / "tiga_work" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if platform.system() == "Windows":
        script = REPO_ROOT / f"_tiga_uninstall_{stamp}.bat"
        data_line = 'if exist "tiga_work" rmdir /s /q "tiga_work"' if remove_data else ""
        script.write_text(
            f"@echo off\r\n"
            f"title TIGA Uninstall\r\n"
            f"echo Waiting for TIGA to stop (PID {server_pid})...\r\n"
            f":waitloop\r\n"
            f'tasklist /FI "PID eq {server_pid}" 2>nul | find "{server_pid}" >nul\r\n'
            f"if not errorlevel 1 (\r\n"
            f"  timeout /t 2 /nobreak >nul\r\n"
            f"  goto waitloop\r\n"
            f")\r\n"
            f"echo Removing virtual environment...\r\n"
            f'if exist ".venv" rmdir /s /q ".venv"\r\n'
            f"{data_line}\r\n"
            f"echo TIGA uninstalled.\r\n"
            f'del /q "%~f0"\r\n',
            encoding="utf-8",
        )
        return script

    script = REPO_ROOT / f"_tiga_uninstall_{stamp}.sh"
    data_line = 'rm -rf "tiga_work"' if remove_data else ""
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'echo "Waiting for TIGA to stop (PID {server_pid})..."\n'
        f"while kill -0 {server_pid} 2>/dev/null; do sleep 2; done\n"
        'echo "Removing virtual environment..."\n'
        'rm -rf ".venv"\n'
        f"{data_line}\n"
        'echo "TIGA uninstalled."\n'
        f'rm -f "$0"\n',
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | 0o111)
    return script


def _spawn_detached(script_path: Path) -> None:
    if platform.system() == "Windows":
        subprocess.Popen(
            ["cmd", "/c", "start", "", "/MIN", str(script_path)],
            cwd=REPO_ROOT,
            creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
            close_fds=True,
        )
        return
    subprocess.Popen(
        ["/bin/bash", str(script_path)],
        cwd=REPO_ROOT,
        start_new_session=True,
        close_fds=True,
    )


def shutdown_server(delay_seconds: float = 1.0) -> None:
    """Exit the current process after a short delay (for uninstall handoff)."""

    def _exit() -> None:
        time.sleep(delay_seconds)
        os._exit(0)

    threading.Thread(target=_exit, daemon=True).start()
