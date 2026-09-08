"""Checks for exported adapters and complete merged model artifacts."""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol

import safetensors
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizerBase,
)

from qorl.adapters.config import adapter_config
from qorl.adapters.schemas import (
    AdapterExportManifest,
    MergeArtifact,
    MergeLoraConfig,
    MergeManifest,
    TensorSignature,
)
from qorl.model.files import (
    model_weight_files,
    model_weights_sha256,
    validate_model_directory,
)
from qorl.model.schemas import ModelWeightIndex
from qorl.paths import REPOSITORY_ROOT
from qorl.util.hashing import sha256_file

ADAPTER_MANIFEST_FILE = "qorl-manifest.json"


class PretrainedLoader[Result](Protocol):
    """The local-only subset of Transformers' dynamic loading API."""

    def from_pretrained(self, path: Path, /, *, local_files_only: bool) -> Result: ...


CONFIG_LOADER: PretrainedLoader[PretrainedConfig] = AutoConfig
TOKENIZER_LOADER: PretrainedLoader[PreTrainedTokenizerBase] = AutoTokenizer


class TensorSlice(Protocol):
    def get_shape(self) -> list[int]: ...
    def get_dtype(self) -> str: ...


class TensorHeader(Protocol):
    def keys(self) -> list[str]: ...
    def get_slice(self, name: str) -> TensorSlice: ...


class TensorOpener(Protocol):
    def __call__(
        self, filename: Path, *, framework: str, device: str
    ) -> AbstractContextManager[TensorHeader]: ...


OPEN_TENSORS: TensorOpener = safetensors.safe_open


def tensor_signatures(model: Path) -> dict[str, TensorSignature]:
    """Check complete shard membership and headers without allocating model tensors."""
    signatures: dict[str, TensorSignature] = {}
    locations: dict[str, str] = {}
    files = model_weight_files(model)
    index_path = next(
        (file for file in files if file.name.endswith(".index.json")), None
    )
    for file in files:
        if file == index_path:
            continue
        if file.suffix != ".safetensors":
            raise RuntimeError("merge requires a safetensors base")
        with OPEN_TENSORS(file, framework="pt", device="cpu") as tensors:
            for key in sorted(tensors.keys()):
                if key in signatures:
                    raise RuntimeError(f"duplicate tensor across shards: {key}")
                view = tensors.get_slice(key)
                signatures[key] = TensorSignature(
                    shape=view.get_shape(), dtype=view.get_dtype()
                )
                locations[key] = file.relative_to(model).as_posix()
    if index_path is not None:
        index = ModelWeightIndex.model_validate_json(index_path.read_bytes())
        if index.weight_map != locations:
            raise RuntimeError("model shard index does not match its tensors")
    if not signatures:
        raise RuntimeError("model has no tensors")
    return signatures


def verify_adapter_base(adapter: Path, base: Path) -> str:
    """Use recorded checksums when available, so identical relocated bases work."""
    recorded = (
        REPOSITORY_ROOT
        / Path(adapter_config(adapter).base_model_name_or_path).expanduser()
    ).resolve()
    manifest_path = adapter / ADAPTER_MANIFEST_FILE
    manifest = (
        AdapterExportManifest.model_validate_json(manifest_path.read_bytes())
        if manifest_path.is_file()
        else None
    )
    expected = manifest.base_model_sha256 if manifest is not None else None
    if expected is None:
        try:
            expected = model_weights_sha256(recorded)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"adapter's recorded base model is missing or invalid: {recorded}"
            ) from error
    try:
        supplied = model_weights_sha256(base.resolve())
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"base model weights are missing or invalid: {base}"
        ) from error
    if supplied != expected:
        if base.resolve() == recorded:
            raise RuntimeError(
                "adapter manifest does not match its recorded base model"
            )
        raise RuntimeError(
            "supplied base model does not match the adapter's training base"
        )
    return expected


def verify_adapter_weights(adapter: Path) -> AdapterExportManifest:
    """Require the export's recorded tensor bytes before applying its updates."""
    manifest = AdapterExportManifest.model_validate_json(
        (adapter / ADAPTER_MANIFEST_FILE).read_bytes()
    )
    if sha256_file(adapter / "adapter_model.safetensors") != manifest.adapter_sha256:
        raise RuntimeError("adapter weights do not match the export manifest")
    if manifest.base_model_sha256 is None:
        raise RuntimeError("adapter export manifest has no base checksum")
    return manifest


def artifact_inventory(directory: Path) -> list[MergeArtifact]:
    """Inventory all published files except the inventory itself."""
    return [
        MergeArtifact(
            path=path.relative_to(directory).as_posix(),
            bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != directory / "qorl-merge.json"
    ]


def verify_merged_model(base: Path, adapter: Path, merged: Path) -> None:
    """Check input identity and every staged or published output file."""
    base_hash = verify_adapter_base(adapter, base)
    export = verify_adapter_weights(adapter)
    validate_model_directory(merged)
    manifest = MergeManifest.model_validate_json(
        (merged / "qorl-merge.json").read_bytes()
    )
    config = MergeLoraConfig.model_validate_json(
        (adapter / "adapter_config.json").read_bytes()
    )
    if (
        manifest.lora_rank != config.r
        or manifest.lora_alpha != config.lora_alpha
        or manifest.lora_scale != config.lora_alpha / config.r
        or manifest.merged_tensor_count * 2 != export.tensor_count
    ):
        raise RuntimeError("merged model scaling or tensor count differs from adapter")
    if (
        manifest.base_model_sha256 != base_hash
        or manifest.adapter_model_sha256
        != sha256_file(adapter / "adapter_model.safetensors")
        or manifest.adapter_config_sha256
        != sha256_file(adapter / "adapter_config.json")
        or manifest.adapter_manifest_sha256
        != sha256_file(adapter / ADAPTER_MANIFEST_FILE)
        or manifest.merged_model_sha256 != model_weights_sha256(merged)
    ):
        raise RuntimeError("merged model checksum mismatch")
    if manifest.artifacts != artifact_inventory(merged):
        raise RuntimeError("merged model artifact inventory differs")
    if tensor_signatures(base) != tensor_signatures(merged):
        raise RuntimeError(
            "merged model tensor names, shapes or dtypes differ from base"
        )
    CONFIG_LOADER.from_pretrained(merged, local_files_only=True)
    tokenizer = TOKENIZER_LOADER.from_pretrained(merged, local_files_only=True)
    if len(tokenizer) == 0:
        raise RuntimeError("merged model tokenizer is empty")
