from pathlib import Path
from unittest.mock import Mock

import pytest

from qorl import cli
from qorl.adapters.merge import OBJECT
from qorl.util.io import write_json


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_merges_default_or_relocated_base(
    merge_inputs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    explicit: bool,
) -> None:
    base, adapter = merge_inputs
    arguments = [
        "qorl",
        "model",
        "merge",
        "--adapter-path",
        str(adapter),
        "--output",
        str(tmp_path / "output"),
    ]
    if explicit:
        relocated = tmp_path / "relocated"
        base.rename(relocated)
        arguments.extend(["--base-model-name-or-path", str(relocated)])
    monkeypatch.setattr("sys.argv", arguments)
    assert cli.main() == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "output")
    assert (tmp_path / "output" / "qorl-merge.json").is_file()


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_uses_cached_revision_without_network(
    merge_inputs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    explicit: bool,
) -> None:
    base, adapter = merge_inputs
    revision = "a" * 40
    cache = tmp_path / "cache"
    snapshot = cache / "models--owner--model" / "snapshots" / revision
    snapshot.parent.mkdir(parents=True)
    base.rename(snapshot)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))
    arguments = [
        "qorl",
        "model",
        "merge",
        "--adapter-path",
        str(adapter),
        "--output",
        str(tmp_path / "output"),
    ]
    if explicit:
        arguments.extend(
            [
                "--base-model-name-or-path",
                "owner/model",
                "--base-model-revision",
                revision,
            ]
        )
    else:
        path = adapter / "adapter_config.json"
        config = OBJECT.validate_json(path.read_bytes())
        config.update(base_model_name_or_path="owner/model", revision=revision)
        write_json(path, config)
    execute = Mock(return_value=tmp_path / "output")
    monkeypatch.setattr(cli, "merge", execute)
    monkeypatch.setattr("sys.argv", arguments)
    assert cli.main() == 0
    execute.assert_called_once_with(
        snapshot.resolve(), adapter.resolve(), tmp_path / "output"
    )


@pytest.mark.parametrize(
    "arguments", [[], ["--adapter-path", "adapter"], ["--output", "model"]]
)
def test_merge_requires_both_paths(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.parser().parse_args(["model", "merge", *arguments])
    assert error.value.code == 2


def test_missing_cached_revision_fails_before_merge(
    merge_inputs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, adapter = merge_inputs
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "qorl",
            "model",
            "merge",
            "--adapter-path",
            str(adapter),
            "--output",
            str(tmp_path / "output"),
            "--base-model-name-or-path",
            "owner/model",
            "--base-model-revision",
            "b" * 40,
        ],
    )
    assert cli.main() == 1
    assert not (tmp_path / "output").exists()
