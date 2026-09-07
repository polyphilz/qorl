from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from qorl.model.files import (
    model_snapshot,
    model_weight_files,
    model_weights_sha256,
    validate_model_directory,
)
from qorl.model.schemas import ModelWeightIndex
from qorl.util.hashing import sha256_file, sha256_json


@pytest.mark.parametrize("cache_source", ["hub_cache", "hf_home", "default"])
def test_model_snapshot_finds_only_the_requested_cached_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_source: str
) -> None:
    home = tmp_path / "home"

    def fake_home(_cls: type[Path]) -> Path:
        return home

    monkeypatch.setattr(Path, "home", classmethod(fake_home))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    if cache_source == "hub_cache":
        cache = tmp_path / "hub-cache"
        monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))
        monkeypatch.setenv("HF_HOME", str(tmp_path / "unused-home"))
    elif cache_source == "hf_home":
        hf_home = tmp_path / "hf-home"
        cache = hf_home / "hub"
        monkeypatch.setenv("HF_HOME", str(hf_home))
    else:
        cache = home / ".cache/huggingface/hub"
    snapshot = cache / "models--organization--model/snapshots/pinned-revision"
    snapshot.mkdir(parents=True)

    assert model_snapshot("organization/model", "pinned-revision") == snapshot.resolve()
    with pytest.raises(RuntimeError, match="pinned model snapshot is missing"):
        model_snapshot("organization/model", "missing-revision")


def test_sharded_model_requires_every_shard(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "shard.safetensors"}})
    )
    with pytest.raises(ValueError, match="model shard"):
        validate_model_directory(tmp_path)
    (tmp_path / "shard.safetensors").write_bytes(b"shard")
    validate_model_directory(tmp_path)


def test_hugging_face_cache_shard_symlinks_are_valid(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    blob = tmp_path / "cached-blob"
    blob.write_bytes(b"cached weights")
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "shard.safetensors"}})
    )
    (snapshot / "shard.safetensors").symlink_to(blob)
    validate_model_directory(snapshot)


@pytest.mark.parametrize("filename", ["model.safetensors", "pytorch_model.bin"])
def test_single_weight_file_keeps_its_original_checksum(
    tmp_path: Path, filename: str
) -> None:
    weights = tmp_path / filename
    weights.write_bytes(b"original weights")
    assert model_weight_files(tmp_path) == [weights]
    assert model_weights_sha256(tmp_path) == sha256_file(weights)


@pytest.fixture(params=["model.safetensors", "pytorch_model.bin"])
def sharded_model(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    filename = str(request.param)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / f"first-{filename}").write_bytes(b"first shard")
    (base / f"second-{filename}").write_bytes(b"second shard")
    (base / f"{filename}.index.json").write_text(
        ModelWeightIndex(
            weight_map={
                "z": f"second-{filename}",
                "a": f"first-{filename}",
                "b": f"first-{filename}",
            }
        ).model_dump_json()
    )
    return base


def test_sharded_checksum_covers_the_index_and_each_distinct_shard(
    sharded_model: Path,
) -> None:
    validate_model_directory(sharded_model)
    files = model_weight_files(sharded_model)
    assert len(files) == 3
    assert files[0].name.endswith(".index.json")
    expected = sha256_json({path.name: sha256_file(path) for path in files})
    assert model_weights_sha256(sharded_model) == expected
    copy = sharded_model.parent / "copied"
    shutil.copytree(sharded_model, copy)
    assert model_weights_sha256(copy) == expected


@pytest.mark.parametrize("changed_file", [0, 1, 2])
def test_every_weight_file_affects_the_sharded_checksum(
    sharded_model: Path, changed_file: int
) -> None:
    before = model_weights_sha256(sharded_model)
    path = model_weight_files(sharded_model)[changed_file]
    path.write_bytes(path.read_bytes() + b" ")
    assert model_weights_sha256(sharded_model) != before


def test_missing_shard_fails_validation_and_hashing(sharded_model: Path) -> None:
    model_weight_files(sharded_model)[-1].unlink()
    with pytest.raises(ValueError, match="model shard"):
        validate_model_directory(sharded_model)
    with pytest.raises(ValueError, match="model shard"):
        model_weights_sha256(sharded_model)


@pytest.mark.parametrize(
    "shard", ["../outside.safetensors", "/outside.safetensors", "missing.safetensors"]
)
def test_invalid_shard_paths_are_rejected(tmp_path: Path, shard: str) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        ModelWeightIndex(weight_map={"weight": shard}).model_dump_json()
    )
    with pytest.raises(ValueError, match="model shard"):
        model_weights_sha256(tmp_path)


def test_safetensors_index_takes_precedence_over_alternate_weight_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        ModelWeightIndex(weight_map={"weight": "shard.safetensors"}).model_dump_json()
    )
    (tmp_path / "shard.safetensors").write_bytes(b"selected weights")
    (tmp_path / "model.safetensors").write_bytes(b"unindexed weights")
    (tmp_path / "pytorch_model.bin").write_bytes(b"alternate format")
    assert [path.name for path in model_weight_files(tmp_path)] == [
        "model.safetensors.index.json",
        "shard.safetensors",
    ]
    (tmp_path / "shard.safetensors").unlink()
    with pytest.raises(ValueError, match="model shard"):
        model_weights_sha256(tmp_path)
