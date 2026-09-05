from __future__ import annotations

import argparse
from pathlib import Path

from qorl.measure.schemas import RunStatus
from qorl.sft.schemas import (
    JSON_OBJECT_ADAPTER,
    DatasetSelection,
    SourceCalibration,
    TimeoutAlgorithm,
    TimeoutManifest,
    TimeoutSelection,
    TimeoutTask,
    load_record,
    require_float,
    require_list,
    require_object,
    require_string,
)
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
SELECTION = ROOT / "experiments/005-protocol-sft-v2/selection.json"
OUTPUT = ROOT / "experiments/005-protocol-sft-v2/timeouts.json"
SPLIT = "sampling"


def build(calibration: Path) -> TimeoutManifest:
    calibration = calibration.resolve()
    source_manifest_path = calibration / "calibration.json"
    source_manifest = JSON_OBJECT_ADAPTER.validate_json(
        source_manifest_path.read_text(encoding="utf-8")
    )
    selection = load_record(SELECTION, DatasetSelection)
    task_set = TaskSet.load(ROOT, "ceb")
    selected = selection.splits.sampling
    if source_manifest.get("status") != RunStatus.COMPLETED:
        raise RuntimeError("source calibration is not complete")
    calibration_selection = require_object(
        source_manifest.get("selection"), "calibration selection"
    )
    if calibration_selection.get("sha256") != sha256_file(SELECTION):
        raise RuntimeError("source calibration uses a different selection")
    if calibration_selection.get("split") != SPLIT:
        raise RuntimeError("source calibration uses a different split")

    tasks: list[TimeoutTask] = []
    for item in selected:
        path = calibration / "tasks" / f"{item.task_id}.json"
        result = JSON_OBJECT_ADAPTER.validate_json(path.read_text(encoding="utf-8"))
        if result.get("status") != RunStatus.COMPLETED:
            raise RuntimeError(f"calibration is incomplete: {item.task_id}")
        summary = require_object(result.get("summary"), "calibration summary")
        median = require_float(
            summary.get("median_execution_time_ms"), "calibration median"
        )
        tasks.append(
            TimeoutTask(
                task_id=item.task_id,
                template_id=item.template_id,
                calibrated_default_median_ms=median,
                timeout_ms=task_timeout_ms(median),
                plan_sha256s=[
                    require_string(value, "plan fingerprint")
                    for value in require_list(
                        summary.get("plan_sha256s"), "calibration plan fingerprints"
                    )
                ],
            )
        )

    runtime_identity = source_manifest.get("runtime_identity")
    return TimeoutManifest(
        manifest_id="qorl-protocol-sft-v2-timeouts-v1",
        algorithm=TimeoutAlgorithm(
            global_cap_ms=GLOBAL_TIMEOUT_MS,
            minimum_ms=TIMEOUT_FLOOR_MS,
            multiplier=TIMEOUT_MULTIPLIER,
        ),
        selection=TimeoutSelection(
            inventory_id=selection.inventory_id,
            path=str(SELECTION.relative_to(ROOT)),
            sha256=sha256_file(SELECTION),
            split=SPLIT,
        ),
        data_identity=JSON_OBJECT_ADAPTER.validate_python(task_set.data_identity),
        runtime_identity=require_object(
            runtime_identity, "calibration runtime identity"
        ),
        source_calibration=SourceCalibration(
            calibration_id=require_string(
                source_manifest.get("calibration_id"), "calibration id"
            ),
            manifest_sha256=sha256_file(source_manifest_path),
            derived_from=[
                require_object(value, "derived calibration")
                for value in require_list(
                    source_manifest.get("derived_from", []), "derived calibrations"
                )
            ],
        ),
        task_count=len(tasks),
        tasks=tasks,
    )


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
        write_json(OUTPUT, build(arguments.calibration).to_wire())
        print(OUTPUT)
        return
    if arguments.calibration is not None:
        raise SystemExit("--calibration is only used with --write")
    CalibratedTimeouts.load(ROOT, OUTPUT, TaskSet.load(ROOT, "ceb"))


if __name__ == "__main__":
    main()
