"""
tests/test_lifecycle.py — lifecycle module and API tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from core.db import get_connection
from core import lifecycle


@pytest.fixture
def db(tmp_path: Path):
    conn = get_connection(tmp_path / "db" / "tiga.db")
    yield conn
    conn.close()


@pytest.fixture
def client(db):
    from server import app, get_db
    app.dependency_overrides[get_db] = lambda: db
    with patch("server.ollama_available", return_value=False):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c
    app.dependency_overrides.clear()


def test_get_install_info_in_git_repo():
    with patch.object(lifecycle, "_commits_behind", return_value=0):
        info = lifecycle.get_install_info()
    assert "repo_root" in info
    assert "is_git_repo" in info
    assert info["github_url"] == "https://github.com/13eeeas/TIGA"


def test_lifecycle_info_endpoint(client) -> None:
    resp = client.get("/api/lifecycle/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "commit_short" in data or data.get("commit_short") is None
    assert data["github_url"]


def test_lifecycle_update_status_idle(client) -> None:
    resp = client.get("/api/lifecycle/update/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is False
    assert data["phase"]


def test_lifecycle_uninstall_requires_confirm(client) -> None:
    resp = client.post("/api/lifecycle/uninstall", json={
        "remove_data": False,
        "confirm": "wrong",
    })
    assert resp.status_code == 400


def test_lifecycle_uninstall_schedules(client) -> None:
    with patch("server.launch_uninstall", return_value={"status": "scheduled"}) as mock_launch:
        with patch("server.shutdown_server") as mock_shutdown:
            resp = client.post("/api/lifecycle/uninstall", json={
                "remove_data": False,
                "confirm": "UNINSTALL",
            })
    assert resp.status_code == 200
    assert resp.json()["status"] == "scheduled"
    mock_launch.assert_called_once()
    mock_shutdown.assert_called_once()


def test_start_update_busy_when_running():
    with patch.object(lifecycle, "_update_state", {
        "running": True,
        "phase": "pulling",
        "log": [],
        "error": None,
        "started_at": None,
        "finished_at": None,
        "result": None,
    }):
        result = lifecycle.start_update()
    assert result["status"] == "busy"


def test_write_uninstall_script_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "REPO_ROOT", tmp_path)
    script = lifecycle._write_uninstall_script(remove_data=True, server_pid=12345)
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert "12345" in text
    assert "tiga_work" in text


def test_write_uninstall_script_unix(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(lifecycle.platform, "system", lambda: "Linux")
    script = lifecycle._write_uninstall_script(remove_data=False, server_pid=999)
    assert script.suffix == ".sh"
    assert script.read_text(encoding="utf-8").count("999") >= 1
