from core.card_llm import accept_card_fields, improve_project_card, parse_card_json


def test_parse_strips_thinking_and_keeps_json() -> None:
    raw = '<think>draft</think>\n{"client": "OUE", "typology": "hospitality", "blurb": "Hotel at Changi."}'
    parsed = parse_card_json(raw)
    assert parsed["client"] == "OUE"
    assert parsed["typology"] == "hospitality"


def test_accept_rejects_the_architect_as_client() -> None:
    kept = accept_card_fields(
        {
            "client": "WOHA Architects",
            "typology": "spaceship",
            "stage": "tender",
            "location": "Changi Airport, Singapore",
            "blurb": "An 8-storey hotel at Changi Airport for OUE.",
        }
    )
    assert "client" not in kept
    assert "typology" not in kept
    assert kept["stage"] == "tender"
    assert kept["location"] == "Changi Airport, Singapore"


def test_improve_uses_the_model_reply() -> None:
    def fake(_messages: list[dict[str, str]]) -> str:
        return json_reply()

    fields = improve_project_card(
        "283 HICA",
        "8 storey hotel at Changi Airport. Developer OUE Capital Management.",
        {"typology": "hospitality", "stage": "tender"},
        complete=fake,
    )
    assert fields["client"] == "OUE Capital Management"
    assert fields["blurb"].endswith(".")


def json_reply() -> str:
    return (
        '{"client": "OUE Capital Management", "typology": "hospitality", '
        '"stage": "tender", "location": "Changi Airport, Singapore", '
        '"blurb": "283 HICA is an 8-storey hotel at Changi Airport for OUE Capital Management."}'
    )
