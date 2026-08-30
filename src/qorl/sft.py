from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PRIME_RL_VERSION = "0.9.0"
RUN_NAME = "protocol-sft-spike"


def run(command: list[str], repository: Path) -> None:
    try:
        subprocess.run(command, cwd=repository, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"command failed with status {error.returncode}: {' '.join(command)}"
        ) from error


def model_snapshot(repository: Path) -> tuple[Path, dict[str, Any]]:
    config = json.loads(
        (repository / "configs/evaluation/run-v1.json").read_text()
    )["policy"]
    model = config["model"]
    revision = config["revision"]
    cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache is None:
        home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        cache = str(home / "hub")
    snapshot = (
        Path(cache)
        / f"models--{model.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    if not snapshot.is_dir():
        raise RuntimeError(
            f"pinned model snapshot is not downloaded: {model}@{revision}"
        )
    return snapshot.resolve(), config


def compatibility_report(repository: Path, model: dict[str, Any]) -> Path:
    run_dir = repository / "outputs/sft" / RUN_NAME
    metrics_path = run_dir / "metrics.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    peak_memory = max(
        item["perf/peak_memory"]
        for item in metrics
        if "perf/peak_memory" in item
    )
    loss = next(item["loss/mean"] for item in reversed(metrics) if "loss/mean" in item)
    report = {
        "schema_version": 1,
        "status": "passed",
        "prime_rl_version": PRIME_RL_VERSION,
        "model": model["model"],
        "model_revision": model["revision"],
        "optimizer_steps": 1,
        "packed_sequence_length": 16384,
        "peak_gpu_memory_gib": peak_memory,
        "training_loss": loss,
        "render_audit": "../protocol-render-audit.json",
        "adapter": "checkpoints/step_1/adapter",
        "adapter_verification": "adapter-verification.json",
    }
    output = run_dir / "compatibility-report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output


def sft(repository: Path) -> Path:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("qorl sft requires a Linux x86_64 CUDA training host")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not installed")
    snapshot, model = model_snapshot(repository)
    training = repository / "training"
    dataset = repository / "outputs/sft/prime-protocol-demo"
    audit = repository / "outputs/sft/protocol-render-audit.json"
    output = repository / "outputs/sft"
    run_dir = output / RUN_NAME

    run(
        [
            sys.executable,
            "-m",
            "scripts.sft.export_prime_dataset",
            "--repository",
            str(repository),
            "--output",
            str(dataset),
        ],
        repository,
    )
    prime = [uv, "run", "--project", str(training), "--frozen"]
    run(
        [
            *prime,
            "python",
            str(repository / "scripts/sft/audit_prime_sample.py"),
            "--model",
            str(snapshot),
            "--dataset",
            str(dataset),
            "--output",
            str(audit),
        ],
        repository,
    )
    run(
        [
            *prime,
            "sft",
            "@",
            str(repository / "configs/training/protocol-sft-spike.toml"),
            "--model.name",
            str(snapshot),
            "--tokenizer.name",
            str(snapshot),
            "--data.name",
            str(dataset),
            "--output-dir",
            str(output),
        ],
        repository,
    )
    adapter = run_dir / "checkpoints/step_1/adapter"
    run(
        [
            *prime,
            "python",
            str(training / "export_adapter.py"),
            "--checkpoint",
            str(run_dir / "checkpoints/step_1/trainer"),
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
            str(repository / "scripts/sft/verify_adapter.py"),
            "--model",
            str(snapshot),
            "--adapter",
            str(adapter),
            "--audit",
            str(audit),
            "--output",
            str(run_dir / "adapter-verification.json"),
        ],
        repository,
    )
    return compatibility_report(repository, model)
