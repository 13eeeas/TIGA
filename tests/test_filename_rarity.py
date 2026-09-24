import sqlite3

from core.query import _filename_candidates
from core.retrieval_boost import distinctive_tokens


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE files (
            file_id INTEGER PRIMARY KEY,
            file_path TEXT,
            file_name TEXT,
            project_id TEXT,
            typology TEXT,
            is_latest INTEGER,
            is_superseded INTEGER
        )
        """
    )
    rows = [("Public Area Layout.pdf",) for _ in range(40)]
    rows.append(("VO-032_Changes to Electronic Washlet.xlsx",))
    rows.append(("AMB-INT-FS-002_Public Area Fixture Schedule.pdf",))
    for i, (name,) in enumerate(rows, 1):
        conn.execute(
            "INSERT INTO files VALUES (?,?,?,?,?,?,?)",
            (i, f"/hica/{name}", name, "283 HICA", "hospitality", 0, 0),
        )
    return conn


def test_covering_does_not_displace_a_rare_word() -> None:
    tokens = distinctive_tokens(
        "can you find the latest drawing covering Changes Electronic Washlet on this project"
    )
    assert "covering" not in [t.lower() for t in tokens]
    assert any(t.lower() == "washlet" for t in tokens)


def test_rare_token_finds_the_file_among_common_names() -> None:
    found = _filename_candidates(
        "can you find the latest drawing covering Changes Electronic Washlet on this project",
        {"project_id": "283 HICA"},
        _db(),
    )
    assert found
    assert "Washlet" in found[0]["file_name"]
