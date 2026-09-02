from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from qorl.adapters.model import model_snapshot


PRIME_RL_VERSION = "0.9.0"
RUN_NAME = "protocol-sft-train-v1"
DATASET = Path("outputs/sft/protocol-sft-v1")
CONFIG = Path("experiments/001-protocol-sft-v1/train.toml")


def run(command: list[str], repository: Path) -> None:
    try:
        subprocess.run(command, cwd=repository, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"command failed with status {error.returncode}: {' '.join(command)}"
        ) from error


def pinned_policy(repository: Path) -> tuple[Path, dict[str, Any]]:
    config = json.loads(
        (repository / "configs/policy/run-v1.json").read_text()
    )["policy"]
    return model_snapshot(config), config


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_report(
    repository: Path,
    model: dict[str, Any],
    steps: int,
) -> Path:
    dataset = repository / DATASET
    run_dir = repository / "outputs/sft" / RUN_NAME
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    report = {
        "schema_version": 1,
        "status": "passed",
        "prime_rl_version": PRIME_RL_VERSION,
        "model": model["model"],
        "model_revision": model["revision"],
        "optimizer_steps": steps,
        "packed_sequence_length": json.loads(
            (dataset / "render-audit.json").read_text()
        )["packed_sequence_length"],
        "peak_gpu_memory_gib": max(
            item["perf/peak_memory"]
            for item in metrics
            if "perf/peak_memory" in item
        ),
        "final_training_loss": next(
            item["loss/mean"]
            for item in reversed(metrics)
            if "loss/mean" in item
        ),
        "validation_losses": [
            {"step": item["step"], "loss": item["val/loss"]}
            for item in metrics
            if "val/loss" in item
        ],
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "render_audit_sha256": sha256(dataset / "render-audit.json"),
        "adapter": f"checkpoints/step_{steps}/adapter",
        "adapter_verification": "adapter-verification.json",
    }
    output = run_dir / "training-report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output


def sft(repository: Path) -> Path:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("qorl sft requires a Linux x86_64 CUDA training host")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not installed")

    snapshot, model = pinned_policy(repository)
    training = repository / "training"
    dataset = repository / DATASET
    audit = dataset / "render-audit.json"
    config_path = repository / CONFIG
    output = repository / "outputs/sft"

    run(
        [sys.executable, "-m", "qorl.sft.validate_dataset"],
        repository,
    )
    replay = json.loads((dataset / "replay-audit.json").read_text())
    if replay.get("status") != "passed":
        raise RuntimeError("protocol dataset replay audit has not passed")
    if replay.get("dataset_manifest_sha256") != sha256(dataset / "manifest.json"):
        raise RuntimeError("protocol dataset changed after its replay audit")

    prime = [uv, "run", "--project", str(training), "--frozen"]
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
        ],
        repository,
    )

    rendered = json.loads(audit.read_text())
    configured = tomllib.loads(config_path.read_text())
    steps = rendered["packed_rows"]["train"]
    if configured["data"]["seed"] != rendered["seed"]:
        raise RuntimeError("render audit and training shuffle seeds differ")
    if configured["data"]["seq_len"] != rendered["packed_sequence_length"]:
        raise RuntimeError("render audit and training sequence lengths differ")
    if configured["max_steps"] != steps:
        raise RuntimeError(
            f"training config max_steps={configured['max_steps']} but "
            f"one packed epoch requires {steps}"
        )
    if configured["val"]["interval"] != steps:
        raise RuntimeError("validation interval must equal the one-epoch step count")

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
            "--val.data.name",
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
            str(audit),
            "--output",
            str(output / RUN_NAME / "adapter-verification.json"),
        ],
        repository,
    )
    return training_report(repository, model, steps)
