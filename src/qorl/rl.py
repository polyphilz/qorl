from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path

from qorl.fixture import sha256_file
from qorl.sft import model_snapshot, run


CONFIG = Path("configs/training/rl-spike-v1.toml")
INVENTORY_CHECK = Path("scripts/rl/build_pilot_inventory.py")
MERGE_SCRIPT = Path("scripts/rl/merge_sft_adapter.py")
SFT_RUN = Path("outputs/sft/protocol-sft-train-v1")
MERGED_MODEL = Path("outputs/rl/protocol-sft-v1-merged")
RUN = Path("outputs/rl/rl-spike-v1")


def verify_merged_model(
    base: Path, adapter: Path, merged: Path
) -> None:
    manifest_path = merged / "qorl-merge.json"
    model_path = merged / "model.safetensors"
    if not manifest_path.is_file() or not model_path.is_file():
        raise RuntimeError(f"merged SFT model is incomplete: {merged}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "base_model_sha256": sha256_file(base / "model.safetensors"),
        "adapter_model_sha256": sha256_file(
            adapter / "adapter_model.safetensors"
        ),
        "adapter_config_sha256": sha256_file(
            adapter / "adapter_config.json"
        ),
        "merged_model_sha256": sha256_file(model_path),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("merged SFT model checksum mismatch")


def rl(repository: Path) -> Path:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("qorl rl requires a Linux x86_64 CUDA training host")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not installed")

    repository = repository.resolve()
    training = repository / "training"
    python_path = [repository / "src", training / "src"]
    existing_python_path = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [
            *(str(path) for path in python_path),
            *([existing_python_path] if existing_python_path else []),
        ]
    )
    prime = [
        uv,
        "run",
        "--project",
        str(training),
        "--frozen",
        "--no-sync",
    ]
    run(
        [sys.executable, str(repository / INVENTORY_CHECK), "--check"],
        repository,
    )

    base, _ = model_snapshot(repository)
    report_path = repository / SFT_RUN / "training-report.json"
    if not report_path.is_file():
        raise RuntimeError("protocol-SFT training report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError("protocol-SFT training did not pass")
    adapter = repository / SFT_RUN / report["adapter"]
    verification_path = repository / SFT_RUN / report["adapter_verification"]
    if not verification_path.is_file():
        raise RuntimeError("protocol-SFT adapter verification is missing")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if not (
        verification.get("adapter_reloaded")
        and verification.get("probabilities_changed")
    ):
        raise RuntimeError("protocol-SFT adapter verification did not pass")

    merged = repository / MERGED_MODEL
    if not merged.exists():
        run(
            [
                *prime,
                "python",
                str(repository / MERGE_SCRIPT),
                "--base",
                str(base),
                "--adapter",
                str(adapter),
                "--output",
                str(merged),
            ],
            repository,
        )
    verify_merged_model(base, adapter, merged)

    run(
        [
            *prime,
            "rl",
            "@",
            str(repository / CONFIG),
            "--model.name",
            str(merged),
        ],
        repository,
    )
    checkpoint = repository / RUN / "checkpoints/step_1"
    if not checkpoint.is_dir():
        raise RuntimeError("RL spike completed without a step-1 checkpoint")
    return checkpoint
