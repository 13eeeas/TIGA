from __future__ import annotations

from core.path_parser import parse_file_path


def _semantics() -> dict:
    return {
        "stage_synonyms": {
            "submission": ["Submission", "Authorities"],
            "client": ["Client"],
            "cad": ["CAD"],
        },
        "issued_synonyms": ["Issued", "To Client"],
        "received_synonyms": ["Received", "From Client"],
        "discipline_folders": {
            "Architecture": ["Architecture", "Arch"],
        },
        "canonical_categories": {
            "client": {"folder_names": ["Client"]},
            "submission": {"folder_names": ["Submission", "Authorities"]},
            "cad": {"folder_names": ["CAD"]},
        },
    }


def test_parse_handles_numbered_prefix_folders_for_stage_and_category() -> None:
    path = "/archive/272 Documents/07 Submission/Final PDF/board.pdf"
    root = "/archive"

    meta = parse_file_path(path, root, semantics=_semantics())

    assert meta["folder_stage"] == "submission"
    assert meta["canonical_category"] == "submission"


def test_parse_handles_project_prefixed_client_received_folders() -> None:
    path = "/archive/272 Documents/270 04 Client/Received/note.msg"
    root = "/archive"

    meta = parse_file_path(path, root, semantics=_semantics())

    assert meta["is_received"] == 1
    assert meta["canonical_category"] == "client"


def test_parse_handles_numbered_cad_folder() -> None:
    path = "/archive/272 CAD/09 CAD/A-101.dwg"
    root = "/archive"

    meta = parse_file_path(path, root, semantics=_semantics())

    assert meta["folder_stage"] == "cad"
    assert meta["canonical_category"] == "cad"
