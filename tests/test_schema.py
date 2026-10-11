import json
from pathlib import Path


def test_project_schema_is_valid_json_and_has_canonical_keys():
    schema_path = Path(__file__).parents[1] / "schemas" / "project-v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["format_version"]["const"] == 1
    assert set(schema["required"]) == {
        "format_version",
        "project",
        "inventory",
        "recordings",
        "audio",
        "alignment",
        "generation",
        "calibration",
    }

