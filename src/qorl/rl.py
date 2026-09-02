from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path

from qorl.adapters.verify import verify_merged_model
from qorl.util.hashing import sha256_file
from qorl.sft import pinned_policy, run

CONFIG = Path("experiments/003-rl-pilot-v1/train.toml")
INVENTORY_CHECK = Path("scripts/rl/build_pilot_inventory.py")
MERGE_MODULE = "qorl_training.adapters.merge"
SFT_RUN = Path("outputs/sft/protocol-sft-train-v1")
MERGED_MODEL = Path("outputs/rl/protocol-sft-v1-merged")
PRE_RL_REPORT = Path("outputs/rl/qorl-rl-pilot-validation-v1/pre/report.json")
RUN = Path("outputs/rl/rl-pilot-v1")
FINAL_STEP = 12


def verify_pre_rl_validation(repository: Path, merged_model_sha256: str) -> None:
    path = repository / PRE_RL_REPORT
    if not path.is_file():
        raise RuntimeError("frozen pre-RL validation report is missing")
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    expected_hashes = {
        "config_sha256": sha256_file(
            repository / "experiments/003-rl-pilot-v1/validation.json"
        ),
        "selection_sha256": sha256_file(
            repository / "experiments/003-rl-pilot-v1/selection.json"
        ),
        "run_config_sha256": sha256_file(repository / "configs/policy/run-v1.json"),
    }
    if any(report.get(key) != value for key, value in expected_hashes.items()):
        raise RuntimeError("frozen pre-RL validation inputs have changed")
    if report.get("model", {}).get("model_safetensors_sha256") != merged_model_sha256:
        raise RuntimeError("frozen pre-RL validation used a different model")
    if not (
        report.get("status") == "completed"
        and report.get("phase") == "pre"
        and summary.get("completed_rollout_count") == 64
        and summary.get("orchestration_failure_count") == 0
        and summary.get("task_group_count") == 16
        and summary.get("nonzero_reward_variance_group_count") == 16
    ):
        raise RuntimeError("frozen pre-RL validation did not pass its gates")


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

    base, _ = pinned_policy(repository)
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
                "-m",
                MERGE_MODULE,
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
    merged_manifest = json.loads(
        (merged / "qorl-merge.json").read_text(encoding="utf-8")
    )
    verify_pre_rl_validation(repository, merged_manifest["merged_model_sha256"])

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
    checkpoint = repository / RUN / f"checkpoints/step_{FINAL_STEP}"
    if not checkpoint.is_dir():
        raise RuntimeError(f"RL pilot completed without a step-{FINAL_STEP} checkpoint")
    return checkpoint
