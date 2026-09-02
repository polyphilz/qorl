from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qorl.db.fixture import DatabaseFixture, TaskSet, sha256_file
from qorl.db.worker import PostgresWorker
from qorl.measure.calibration import plan_sha256
from scripts.sft.build_protocol_dataset import PlanValidationEvaluator
from scripts.utils.protocol_dataset import load_documents


def candidate_actions(document: dict[str, Any]) -> list[dict[str, Any]]:
    actions = []
    for message in document["messages"]:
        if message["role"] != "assistant":
            continue
        function = message["tool_calls"][0]["function"]
        if function["name"] == "evaluate_candidate":
            actions.append(json.loads(function["arguments"])["action"])
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one protocol-SFT example per CEB template."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/sft/protocol-sft-v1")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sft/protocol-sft-v1/replay-audit.json"),
    )
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    dataset = arguments.dataset
    output = arguments.output
    if not dataset.is_absolute():
        dataset = repository / dataset
    if not output.is_absolute():
        output = repository / output

    fixture = DatabaseFixture.load(repository)
    task_set = TaskSet.load(repository, "ceb-v1", fixture.data_identity)
    tasks = {task["task_id"]: task for task in task_set.inventory["tasks"]}

    samples: dict[str, dict[str, Any]] = {}
    for _, document in load_documents(dataset):
        template = document["metadata"]["template_id"]
        samples.setdefault(template, document)

    records = []
    with PostgresWorker(fixture, "qorl-protocol-sft-replay") as worker:
        for template, document in sorted(samples.items()):
            task_id = document["metadata"]["task_id"]
            evaluator = PlanValidationEvaluator(worker, task_set, tasks[task_id])
            baseline = evaluator.start()
            recorded_default = document["evidence"]["default_plan"]["Plan"]

            if baseline["plan_sha256"] != plan_sha256(recorded_default):
                raise RuntimeError(f"{task_id}: default plan fingerprint changed")

            candidate_ids = []
            for action in candidate_actions(document):
                candidate = evaluator.evaluate(action)
                candidate_id = candidate["candidate_id"]
                recorded = document["evidence"]["candidates"][candidate_id]
                if not candidate["action_valid"]:
                    raise RuntimeError(f"{task_id}/{candidate_id}: action rejected")
                if not candidate["constraints_satisfied"]:
                    raise RuntimeError(
                        f"{task_id}/{candidate_id}: constraints not satisfied"
                    )
                if candidate["action"] != recorded["action"]:
                    raise RuntimeError(f"{task_id}/{candidate_id}: action changed")
                if candidate["plan_sha256"] != plan_sha256(
                    recorded["plain_explain"]["Plan"]
                ):
                    raise RuntimeError(
                        f"{task_id}/{candidate_id}: plan fingerprint changed"
                    )
                candidate_ids.append(candidate_id)

            records.append(
                {
                    "template_id": template,
                    "task_id": task_id,
                    "candidate_ids": candidate_ids,
                    "status": "passed",
                }
            )
            print(
                f"[{len(records)}/{len(samples)}] {template} "
                f"candidates={len(candidate_ids)}",
                flush=True,
            )

    report = {
        "schema_version": 1,
        "status": "passed",
        "selection": "lowest dataset ordinal per template",
        "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "data_identity": fixture.data_identity,
        "runtime_identity": fixture.runtime_identity,
        "templates": len(records),
        "candidates": sum(len(record["candidate_ids"]) for record in records),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {key: value for key, value in report.items() if key != "records"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
