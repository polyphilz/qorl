from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qorl.db.fixture import DatabaseFixture, data_identity
from qorl.measure.schemas import RunStatus
from qorl.util.hashing import sha256_file
from qorl.util.io import write_json
from qorl.workload.taskset import TaskSet
from qorl.workload.timeouts import (
    GLOBAL_TIMEOUT_MS,
    TIMEOUT_FLOOR_MS,
    TIMEOUT_MULTIPLIER,
    CalibratedTimeouts,
    task_timeout_ms,
)

ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / "experiments/004-rl-run-v2/selection.json"
OUTPUT = ROOT / "experiments/004-rl-run-v2/timeouts.json"


def build(calibration: Path) -> dict[str, Any]:
    calibration = calibration.resolve()
    source_manifest_path = calibration / "calibration.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    task_set = TaskSet.load(ROOT, "ceb")
    fixture = DatabaseFixture.load(ROOT)
    selected = selection["splits"]["train"]
    if source_manifest.get("status") != RunStatus.COMPLETED:
        raise RuntimeError("source calibration is not complete")
    source_data_identity = source_manifest.get(
        "data_identity", source_manifest.get("database", {})
    )
    if data_identity(source_data_identity) != task_set.data_identity:
        raise RuntimeError("source calibration uses a different database")
    if source_manifest.get("selection", {}).get("sha256") != sha256_file(SELECTION):
        raise RuntimeError("source calibration uses a different selection")

    tasks: list[dict[str, Any]] = []
    for item in selected:
        path = calibration / "tasks" / f"{item['task_id']}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != RunStatus.COMPLETED:
            raise RuntimeError(f"calibration is incomplete: {item['task_id']}")
        summary = result["summary"]
        median = summary["median_execution_time_ms"]
        tasks.append(
            {
                "task_id": item["task_id"],
                "template_id": item["template_id"],
                "calibrated_default_median_ms": median,
                "timeout_ms": task_timeout_ms(median),
                "plan_sha256s": summary["plan_sha256s"],
            }
        )

    return {
        "schema_version": 1,
        "manifest_id": "qorl-rl-run-v2-timeouts-v1",
        "algorithm": {
            "global_cap_ms": GLOBAL_TIMEOUT_MS,
            "minimum_ms": TIMEOUT_FLOOR_MS,
            "multiplier": TIMEOUT_MULTIPLIER,
        },
        "selection": {
            "inventory_id": selection["inventory_id"],
            "path": str(SELECTION.relative_to(ROOT)),
            "sha256": sha256_file(SELECTION),
            "split": "train",
        },
        "data_identity": task_set.data_identity,
        "runtime_identity": source_manifest.get(
            "runtime_identity", fixture.runtime_identity
        ),
        "source_calibration": {
            "calibration_id": source_manifest["calibration_id"],
            "manifest_sha256": sha256_file(source_manifest_path),
            "derived_from": source_manifest.get("derived_from", []),
        },
        "task_count": len(tasks),
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.write == arguments.check:
        raise SystemExit("choose exactly one of --write or --check")
    if arguments.write:
        if arguments.calibration is None:
            raise SystemExit("--write requires --calibration")
        write_json(OUTPUT, build(arguments.calibration))
        print(OUTPUT)
        return
    if arguments.calibration is not None:
        raise SystemExit("--calibration is only used with --write")
    CalibratedTimeouts.load(ROOT, OUTPUT, TaskSet.load(ROOT, "ceb"))


if __name__ == "__main__":
    main()
