from pathlib import Path

from core.db import get_connection, upsert_file
from core.progress import get_index_progress


def _file(file_id: str, path: str, status: str) -> dict:
    return {
        "file_id": file_id,
        "file_path": path,
        "file_name": Path(path).name,
        "extension": Path(path).suffix,
        "status": status,
    }


def test_progress_reports_per_project_queue_and_stalled_eta(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "tiga.db")
    root_a = tmp_path / "Project A"
    root_b = tmp_path / "Project B"
    upsert_file(conn, _file("a1", "\\\\?\\" + str(root_a / "one.pdf"), "DISCOVERED"))
    upsert_file(conn, _file("a2", str(root_a / "two.pdf"), "EXTRACTED"))
    upsert_file(conn, _file("b1", str(root_b / "done.pdf"), "INDEXED"))

    progress = get_index_progress(conn, [root_a, root_b])

    assert progress["overall"]["files_remaining"] == 2
    assert progress["overall"]["state"] == "stalled"
    assert progress["overall"]["eta"] is None
    assert progress["projects"][0]["files_remaining"] == 2
    assert progress["projects"][1]["state"] == "complete"
    conn.close()


def test_progress_uses_recent_transition_rates(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "tiga.db")
    root = tmp_path / "HICA"
    upsert_file(conn, _file("f1", str(root / "one.pdf"), "DISCOVERED"))
    upsert_file(conn, _file("f2", str(root / "two.pdf"), "EXTRACTED"))
    conn.executemany(
        "INSERT INTO events(file_id,event_type) VALUES (?,?)",
        [("f1", "EXTRACTED"), ("f2", "INDEXED")],
    )
    conn.commit()

    item = get_index_progress(conn, [root], window_minutes=60)["overall"]

    assert item["extract_rate_files_per_hour"] == 1
    assert item["index_rate_files_per_hour"] == 1
    assert item["eta_seconds"] == 10800
    assert item["eta"] == "3h"
    assert item["state"] == "running"
    conn.close()


def test_progress_matches_mapped_root_to_unc_storage_path(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "tiga.db")
    root = Path("F:/Shared/Projects/283 HICA")
    upsert_file(conn, _file("f1", "//EgnyteDrive/woha/Shared/Projects/283 HICA/a.pdf", "DISCOVERED"))

    project = get_index_progress(conn, [root])["projects"][0]

    assert project["files_remaining"] == 1
    assert project["discovered"] == 1
    conn.close()
