from pathlib import Path

from qorl.util.io import display_path, write_json


def test_write_json_creates_parents_and_preserves_format(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "report.json"

    write_json(path, {"z": True, "a": "é"})

    assert path.read_text() == '{\n  "a": "\\u00e9",\n  "z": true\n}\n'
    assert list(path.parent.iterdir()) == [path]


def test_write_json_replaces_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text("old contents")
    values: list[str] = ["a", "b"]

    write_json(path, values)

    assert path.read_text() == '[\n  "a",\n  "b"\n]\n'
    assert list(tmp_path.iterdir()) == [path]


def test_display_path_is_relative_only_inside_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    external = tmp_path / "external.json"

    assert display_path(repository, repository / "report.json") == "report.json"
    assert display_path(repository, external) == str(external)
