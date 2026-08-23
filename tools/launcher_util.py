"""
tools/launcher_util.py — Desktop shortcuts, service checks, and uninstall helpers.

Used by:
  python tiga.py open
  python tiga.py shortcuts
  python tiga.py uninstall
  launcher.bat / launcher.sh
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _cfg():
    from config import cfg
    return cfg


def hunt_url(host: str = "127.0.0.1") -> str:
    c = _cfg()
    return f"http://{host}:{c.server_port}/"


def launcher_url(host: str = "127.0.0.1") -> str:
    c = _cfg()
    return f"http://{host}:{c.server_port}/launcher"


def admin_url(host: str = "127.0.0.1") -> str:
    c = _cfg()
    return f"http://{host}:{c.server_port + 1}/"


def is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def server_running() -> bool:
    return is_port_open(_cfg().server_port)


def admin_running() -> bool:
    return is_port_open(_cfg().server_port + 1)


def _python_exe() -> Path:
    return Path(sys.executable)


def _spawn_detached(args: list[str], title: str) -> None:
    repo = str(REPO_ROOT)
    if platform.system() == "Windows":
        cmd = " ".join(f'"{a}"' if " " in a else a for a in args)
        subprocess.Popen(
            ["cmd", "/c", "start", title, "cmd", "/k", f"cd /d {repo} && {cmd}"],
            cwd=repo,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
    else:
        subprocess.Popen(
            args,
            cwd=repo,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


def ensure_services(start_if_down: bool = True, wait_s: float = 4.0) -> dict[str, bool]:
    """Ensure Hunt server (+ admin) are reachable; optionally start them."""
    c = _cfg()
    py = str(_python_exe())
    status = {"server": server_running(), "admin": admin_running()}

    if start_if_down and not status["server"]:
        _spawn_detached([py, "tiga.py", "serve"], "TIGA-Server")
        status["server"] = _wait_for_port(c.server_port, wait_s)

    if start_if_down and not status["admin"]:
        _spawn_detached([py, "tiga.py", "ui"], "TIGA-Admin")
        status["admin"] = _wait_for_port(c.server_port + 1, wait_s)

    return status


def _wait_for_port(port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_port_open(port):
            return True
        time.sleep(0.25)
    return is_port_open(port)


def open_portal(target: str = "launcher", start: bool = True) -> None:
    if start:
        ensure_services(start_if_down=True)
    urls = {
        "hunt": hunt_url(),
        "launcher": launcher_url(),
        "admin": admin_url(),
    }
    url = urls.get(target, launcher_url())
    print(f"Opening {url}")
    webbrowser.open(url)


def _desktop_dir() -> Path:
    home = Path.home()
    if platform.system() == "Windows":
        return Path(os.environ.get("USERPROFILE", home)) / "Desktop"
    return home / "Desktop"


def _shortcut_names() -> tuple[str, str, str]:
    return ("TIGA Hunt", "TIGA Admin", "Uninstall TIGA")


def create_shortcuts() -> list[str]:
    """Create desktop shortcuts; returns paths created."""
    desktop = _desktop_dir()
    desktop.mkdir(parents=True, exist_ok=True)
    created: list[str] = []

    if platform.system() == "Windows":
        created.extend(_create_windows_shortcuts(desktop))
    else:
        created.extend(_create_desktop_entries(desktop))

    return created


def _create_windows_shortcuts(desktop: Path) -> list[str]:
    launcher = REPO_ROOT / "launcher.bat"
    uninstall = REPO_ROOT / "uninstall.bat"
    open_admin = REPO_ROOT / "open-admin.bat"
    names = _shortcut_names()
    targets = [launcher, open_admin, uninstall]
    icons = ["", "", ""]

    ps_lines = []
    for name, target, icon in zip(names, targets, icons):
        lnk = desktop / f"{name}.lnk"
        ps = (
            f"$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}');"
            f"$s.TargetPath = '{target}';"
            f"$s.WorkingDirectory = '{REPO_ROOT}';"
            f"$s.Description = '{name}';"
        )
        if icon:
            ps += f"$s.IconLocation = '{icon}';"
        ps += "$s.Save();"
        ps_lines.append(ps)

    script = "\n".join(ps_lines)
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=True,
        cwd=str(REPO_ROOT),
    )
    return [str(desktop / f"{n}.lnk") for n in names]


def _create_desktop_entries(desktop: Path) -> list[str]:
    launcher = REPO_ROOT / "launcher.sh"
    open_admin = REPO_ROOT / "open-admin.sh"
    uninstall = REPO_ROOT / "uninstall.sh"
    entries = [
        ("TIGA Hunt", launcher, "Open TIGA Hunt launcher portal"),
        ("TIGA Admin", open_admin, "Open TIGA admin panel"),
        ("Uninstall TIGA", uninstall, "Remove TIGA shortcuts and optional local data"),
    ]
    created: list[str] = []
    for name, script, comment in entries:
        path = desktop / f"{name}.desktop"
        content = (
            "[Desktop Entry]\n"
            f"Name={name}\n"
            f"Comment={comment}\n"
            f"Exec={script}\n"
            f"Path={REPO_ROOT}\n"
            "Terminal=true\n"
            "Type=Application\n"
            "Categories=Office;\n"
        )
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)
        created.append(str(path))
    return created


def remove_shortcuts() -> list[str]:
    removed: list[str] = []
    desktop = _desktop_dir()
    for name in _shortcut_names():
        if platform.system() == "Windows":
            path = desktop / f"{name}.lnk"
        else:
            path = desktop / f"{name}.desktop"
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def stop_tiga_processes() -> None:
    """Best-effort stop of serve/ui processes started from this repo."""
    if platform.system() == "Windows":
        subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.CommandLine -match 'tiga\\.py (serve|ui)' } | "
                "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }",
            ],
            cwd=str(REPO_ROOT),
        )
    else:
        subprocess.run(["pkill", "-f", "tiga.py serve"], check=False)
        subprocess.run(["pkill", "-f", "tiga.py ui"], check=False)


def uninstall(
    *,
    remove_desktop_shortcuts: bool = True,
    remove_venv: bool = False,
    remove_work_dir: bool = False,
    stop_services: bool = True,
) -> dict[str, list[str] | bool]:
    actions: dict[str, list[str] | bool] = {
        "shortcuts_removed": [],
        "venv_removed": False,
        "work_dir_removed": False,
        "services_stopped": False,
    }
    if stop_services:
        stop_tiga_processes()
        actions["services_stopped"] = True
    if remove_desktop_shortcuts:
        actions["shortcuts_removed"] = remove_shortcuts()
    if remove_venv:
        venv = REPO_ROOT / ".venv"
        if venv.exists():
            shutil.rmtree(venv)
            actions["venv_removed"] = True
    if remove_work_dir:
        from config import cfg
        wd = cfg.work_dir
        if wd.exists():
            shutil.rmtree(wd)
            actions["work_dir_removed"] = True
    validate_dir = REPO_ROOT / "tiga_work_validate"
    if remove_work_dir and validate_dir.exists():
        shutil.rmtree(validate_dir)
    return actions
