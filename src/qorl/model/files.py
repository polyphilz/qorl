from __future__ import annotations

import os
from pathlib import Path

from qorl.model.schemas import ModelProvider, ModelSettings, ModelWeightIndex
from qorl.paths import REPOSITORY_ROOT
from qorl.util.hashing import sha256_file, sha256_json

MODEL_FILES = ("model.safetensors", "pytorch_model.bin")


def model_weight_files(path: Path) -> list[Path]:
    """Locate complete weights, preferring safetensors; include the shard index."""
    for filename in MODEL_FILES:
        index_path = path / f"{filename}.index.json"
        if index_path.is_file():
            index = ModelWeightIndex.model_validate_json(index_path.read_bytes())
            shards = sorted(set(index.weight_map.values()))
            for shard in shards:
                relative = Path(shard)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not (path / relative).is_file()
                ):
                    raise ValueError(
                        f"missing or invalid model shard: {shard} in {path}"
                    )
            return [index_path, *(path / shard for shard in shards)]
        if (path / filename).is_file():
            return [path / filename]
    raise ValueError(
        f"complete model weights missing (adapter-only directories are not bases): {path}"
    )


def model_weights_sha256(path: Path) -> str:
    """Hash one file directly, or the index and shards by relative filename/checksum."""
    files = model_weight_files(path)
    if len(files) == 1:
        return sha256_file(files[0])
    return sha256_json(
        {file.relative_to(path).as_posix(): sha256_file(file) for file in files}
    )


def validate_model_directory(path: Path) -> None:
    """Require model config and complete weight files without loading any tensors."""
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError(
            f"complete model directory requires config.json and weights: {path}"
        )
    model_weight_files(path)


def resolve_model(settings: ModelSettings) -> Path:
    """Resolve complete local weights or an already-cached pinned Hugging Face revision."""
    if settings.provider != ModelProvider.LOCAL:
        raise ValueError("hosted models do not have local model weights")
    path = (
        model_snapshot(settings.name_or_path, settings.revision)
        if settings.revision is not None
        else (REPOSITORY_ROOT / Path(settings.name_or_path).expanduser()).resolve()
    )
    validate_model_directory(path)
    return path


def model_snapshot(model: str, revision: str) -> Path:
    """Find an already-downloaded model revision in the Hugging Face cache."""
    cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache is None:
        home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        cache = str(home / "hub")
    snapshot = (
        Path(cache) / f"models--{model.replace('/', '--')}" / "snapshots" / revision
    )
    if not snapshot.is_dir():
        raise RuntimeError(f"pinned model snapshot is missing: {snapshot}")
    return snapshot.resolve()
