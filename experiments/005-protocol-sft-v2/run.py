from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from qorl.adapters.model import (
    adapter_base_model,
    adapter_rank,
    model_snapshot,
    verify_adapter_base,
)
from qorl.adapters.verify import verify_merged_model
from qorl.agent import QoAgentConfig
from qorl.measure.schemas import RunStatus
from qorl.sft.schemas import (
    JSON_OBJECT_ADAPTER,
    DatasetConfig,
    JsonObject,
    PreparationReport,
    TrainingReport,
    load_json_object,
    load_record,
    require_float,
    require_int,
    require_object,
)
from qorl.util.hashing import sha256_file
from qorl.util.serving import ServedModel

ROOT = Path(__file__).resolve().parents[2]
DATASET = Path("outputs/sft/protocol-sft-v2")
RUN_NAME = "protocol-sft-train-v2"
RUN_DIR = Path("outputs/sft") / RUN_NAME
TEMPLATE = Path("experiments/005-protocol-sft-v2/train.toml.template")
RESOLVED_CONFIG = DATASET / "train.toml"
PRIME_RL_VERSION = "0.9.0"
DEFAULT_SAMPLER_ADAPTER = Path(
    "outputs/sft/protocol-sft-train-v1/checkpoints/step_159/adapter"
)
DEFAULT_SAMPLER_MODEL = Path("outputs/sft/protocol-sft-v2-sampler-pilot-sft")
GATE_MODEL = "qorl-protocol-sft-v2"
GATE_PORT = 8000
MODEL_STARTUP_TIMEOUT_SECONDS = 600
VLLM_GPU_MEMORY_UTILIZATION = 0.9


def run(command: list[str], repository: Path) -> None:
    try:
        subprocess.run(command, cwd=repository, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"command failed with status {error.returncode}: {' '.join(command)}"
        ) from error


def policy(
    repository: Path, config: DatasetConfig
) -> tuple[Path, QoAgentConfig, JsonObject]:
    value = require_object(
        load_json_object(repository / config.policy_config).get("policy"), "policy"
    )
    return model_snapshot(value), QoAgentConfig.from_dict(value), value


def merge_sampler(repository: Path, adapter: Path, output: Path) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not installed")
    adapter = (repository / adapter).resolve()
    snapshot = adapter_base_model(adapter, repository)
    output = (repository / output).resolve()
    run(
        [
            uv,
            "run",
            "--project",
            str(repository / "training"),
            "--frozen",
            "python",
            "-m",
            "qorl_training.adapters.merge",
            "--repository",
            str(repository),
            "--base",
            str(snapshot),
            "--adapter",
            str(adapter),
            "--output",
            str(output),
        ],
        repository,
    )
    verify_merged_model(snapshot, adapter, output, repository)
    return output / "qorl-merge.json"


def prepare(repository: Path) -> tuple[Path, PreparationReport]:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not installed")
    dataset_config = load_record(
        repository / "experiments/005-protocol-sft-v2/dataset.json", DatasetConfig
    )
    snapshot, policy_config, _ = policy(repository, dataset_config)
    dataset = repository / DATASET
    audit = dataset / "render-audit.json"
    run(
        [
            sys.executable,
            "-m",
            "qorl.sft.validate_dataset",
            "--dataset",
            str(dataset),
        ],
        repository,
    )
    prime = [uv, "run", "--project", str(repository / "training"), "--frozen"]
    run(
        [
            *prime,
            "python",
            "-m",
            "qorl_training.audit.dataset",
            "--model",
            str(snapshot),
            "--dataset",
            str(dataset / "prime"),
            "--output",
            str(audit),
            "--sequence-length",
            str(policy_config.context_length),
            "--seed",
            str(dataset_config.seed),
            "--splits",
            "train",
        ],
        repository,
    )
    rendered = load_json_object(audit)
    packed_rows = require_object(
        rendered.get("packed_rows"), "render audit packed_rows"
    )
    steps = require_int(packed_rows.get("train"), "render audit train rows")
    if require_int(rendered.get("truncated_examples"), "truncated_examples"):
        raise RuntimeError("render audit found truncated examples")
    template = (repository / TEMPLATE).read_text(encoding="utf-8")
    resolved = (
        template.replace("__MAX_STEPS__", str(steps))
        .replace("__SEQUENCE_LENGTH__", str(policy_config.context_length))
        .replace("__DATASET_SEED__", str(dataset_config.seed))
    )
    output = repository / RESOLVED_CONFIG
    output.write_text(resolved, encoding="utf-8")
    return output, PreparationReport(
        optimizer_steps=steps,
        render_audit=str(audit),
        resolved_config=str(output),
    )


def training_report(repository: Path, model: QoAgentConfig, steps: int) -> Path:
    dataset = repository / DATASET
    run_dir = repository / RUN_DIR
    records = [
        JSON_OBJECT_ADAPTER.validate_json(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    report = TrainingReport(
        status=RunStatus.PASSED,
        prime_rl_version=PRIME_RL_VERSION,
        model=model.model,
        model_revision=model.revision,
        optimizer_steps=steps,
        peak_gpu_memory_gib=max(
            require_float(item.get("perf/peak_memory"), "peak memory")
            for item in records
            if "perf/peak_memory" in item
        ),
        final_training_loss=next(
            require_float(item.get("loss/mean"), "training loss")
            for item in reversed(records)
            if "loss/mean" in item
        ),
        dataset_manifest_sha256=sha256_file(dataset / "manifest.json"),
        render_audit_sha256=sha256_file(dataset / "render-audit.json"),
        resolved_config_sha256=sha256_file(dataset / "train.toml"),
        adapter=f"checkpoints/step_{steps}/adapter",
        adapter_verification="adapter-verification.json",
    )
    output = run_dir / "training-report.json"
    output.write_text(json.dumps(report.to_wire(), indent=2, sort_keys=True) + "\n")
    return output


def trained_adapter(repository: Path) -> Path:
    run_dir = (repository / RUN_DIR).resolve()
    report = load_record(run_dir / "training-report.json", TrainingReport)
    relative = Path(report.adapter)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("training report contains an unsafe adapter path")
    adapter = (run_dir / relative).resolve()
    if not adapter.is_relative_to(run_dir):
        raise RuntimeError("training report adapter escapes its run directory")
    return adapter


def gate(repository: Path) -> Path:
    dataset_config = load_record(
        repository / "experiments/005-protocol-sft-v2/dataset.json", DatasetConfig
    )
    snapshot, policy_config, _ = policy(repository, dataset_config)
    adapter = trained_adapter(repository)
    verify_adapter_base(adapter, snapshot, repository)
    vllm = repository / ".venv-vllm/bin/vllm"
    if not vllm.is_file():
        raise RuntimeError(f"pinned evaluation vLLM is missing: {vllm}")
    output = repository / RUN_DIR / "live-gate.json"
    command = [
        str(vllm),
        "serve",
        str(snapshot),
        "--served-model-name",
        GATE_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(GATE_PORT),
        "--language-model-only",
        "--max-model-len",
        str(policy_config.context_length),
        "--max-num-seqs",
        str(dataset_config.gate.concurrency),
        "--gpu-memory-utilization",
        str(VLLM_GPU_MEMORY_UTILIZATION),
        "--generation-config",
        "vllm",
        "--enable-prefix-caching",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        policy_config.tool_call_parser,
        "--enable-lora",
        "--max-lora-rank",
        str(adapter_rank(adapter)),
        "--lora-modules",
        f"{GATE_MODEL}={adapter}",
        "--enforce-eager",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    with ServedModel(
        command,
        repository=repository,
        log_path=output.parent / "live-gate-vllm.log",
        health_url=f"http://127.0.0.1:{GATE_PORT}/health",
        startup_timeout=MODEL_STARTUP_TIMEOUT_SECONDS,
        environment=environment,
    ):
        run(
            [
                sys.executable,
                "-m",
                "qorl.sft.gate",
                "--repository",
                str(repository),
                "--model",
                GATE_MODEL,
                "--output",
                str(output),
            ],
            repository,
        )
    return output


def train(repository: Path) -> Path:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("tool-use SFT v2 requires a Linux x86_64 CUDA host")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not installed")
    config_path, preparation = prepare(repository)
    dataset_config = load_record(
        repository / "experiments/005-protocol-sft-v2/dataset.json", DatasetConfig
    )
    snapshot, model, _ = policy(repository, dataset_config)
    dataset = repository / DATASET
    output = repository / "outputs/sft"
    steps = preparation.optimizer_steps
    prime = [uv, "run", "--project", str(repository / "training"), "--frozen"]
    run(
        [
            *prime,
            "sft",
            "@",
            str(config_path),
            "--model.name",
            str(snapshot),
            "--tokenizer.name",
            str(snapshot),
            "--data.name",
            str(dataset / "prime"),
            "--output-dir",
            str(output),
        ],
        repository,
    )
    checkpoint = output / RUN_NAME / f"checkpoints/step_{steps}"
    adapter = checkpoint / "adapter"
    run(
        [
            *prime,
            "python",
            "-m",
            "qorl_training.adapters.export",
            "--checkpoint",
            str(checkpoint / "trainer"),
            "--model",
            str(snapshot),
            "--output",
            str(adapter),
        ],
        repository,
    )
    run(
        [
            *prime,
            "python",
            "-m",
            "qorl.adapters.verify",
            "--model",
            str(snapshot),
            "--adapter",
            str(adapter),
            "--audit",
            str(dataset / "render-audit.json"),
            "--output",
            str(output / RUN_NAME / "adapter-verification.json"),
        ],
        repository,
    )
    return training_report(repository, model, steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run protocol SFT v2.")
    parser.add_argument("--repository", type=Path, default=ROOT)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--merge-sampler", action="store_true")
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--train", action="store_true")
    action.add_argument("--gate", action="store_true")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_SAMPLER_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_SAMPLER_MODEL)
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    if arguments.merge_sampler:
        print(merge_sampler(repository, arguments.adapter, arguments.output))
    elif arguments.prepare:
        _, report = prepare(repository)
        print(json.dumps(report.to_wire(), indent=2, sort_keys=True))
    elif arguments.gate:
        print(gate(repository))
    else:
        print(train(repository))


if __name__ == "__main__":
    main()
