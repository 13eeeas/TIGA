from core.typology_infer import (
    client_from_brief,
    infer_client,
    infer_location,
    infer_stage_from_paths,
    infer_typology,
)


def test_brief_names_a_school() -> None:
    text = (
        "The university campus includes a new business school for students "
        "and faculty, with academic teaching spaces."
    )
    assert infer_typology(text) == "education"


def test_project_name_fills_client_and_type() -> None:
    assert infer_typology("seminar room carpark", name="PP 2025 052 NUS BIZ3") == "education"
    assert infer_typology("", name="270 Keppel Bay Residential Devt") == "residential"
    assert infer_client("PP 2025 052 NUS BIZ3") == "NUS"
    assert client_from_brief(
        "Developer & Sub-lessee     :   OUE Capital Management Pte Ltd"
    ) == "OUE Capital Management Pte Ltd"
    block = """
DEVELOPER
12 Marina Boulevard
DBS Trustee Limited (in its capacity as trustee of OUE T2 Hotel Trust) c/o OUE Capital Management Pte. Ltd.
"""
    assert client_from_brief(block) == "OUE Capital Management Pte. Ltd"
    assert infer_location("270 Keppel Bay Residential Devt", "hotel at Changi Airport, Singapore") == (
        "Changi Airport, Singapore"
    )
    assert infer_stage_from_paths(["X/Stage 2/brief.pdf"] * 3) == "tender"


def test_thin_text_stays_blank() -> None:
    assert infer_typology("door schedule revision C") is None


def test_tied_cues_stay_blank() -> None:
    text = "hotel hotel school school"
    assert infer_typology(text) is None
