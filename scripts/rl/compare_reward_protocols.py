from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qorl.calibration import write_json
from qorl.fixture import DatabaseFixture, TaskSet, sha256_file
from qorl.rollout import (
    RolloutEvaluator,
    TrainingRolloutEvaluatorV1,
)
from qorl.worker import ExplainResult, PostgresWorker


DEFAULT_CONFIG = Path("configs/training/reward-protocol-audit-v1.json")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_config(repository: Path, path: Path) -> tuple[Path, dict[str, Any]]:
    path = path if path.is_absolute() else repository / path
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise RuntimeError("unknown reward-protocol audit configuration")
    return path, config


def templates(repository: Path, config: dict[str, Any]) -> list[str]:
    selection = json.loads(
        (repository / config["training_selection"]).read_text(encoding="utf-8")
    )
    return list(selection["selection"]["template_order"])


def build_cases(
    repository: Path,
    config: dict[str, Any],
    trace_root: Path,
) -> dict[str, Any]:
    expected = templates(repository, config)
    choices: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    file_hashes: dict[Path, str] = {}
    paths = sorted(trace_root.glob("step_*/train/all/traces.jsonl"))
    if not paths:
        raise RuntimeError(f"no pilot traces found beneath {trace_root}")

    for path in paths:
        file_hashes[path] = sha256_file(path)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            row = json.loads(line)
            for trace in row.get("traces", []):
                result = trace.get("info", {}).get("qorl", {})
                template_id = result.get("template_id")
                final = result.get("final", {})
                candidates = result.get("candidates", [])
                if (
                    template_id not in expected
                    or final.get("status") != "completed"
                    or not candidates
                ):
                    continue
                actions = [candidate["action"] for candidate in candidates]
                trace_id = trace["id"]
                key = fingerprint(
                    f"{config['case_selection_salt']}:{result['task_id']}:{trace_id}"
                )
                choices[template_id].append(
                    (
                        key,
                        {
                            "task_id": result["task_id"],
                            "template_id": template_id,
                            "trace_id": trace_id,
                            "source_trace": str(path.relative_to(repository)),
                            "source_trace_sha256": file_hashes[path],
                            "source_line": line_number,
                            "original_rigorous_score": final["score"],
                            "actions_sha256": fingerprint(actions),
                            "actions": actions,
                        },
                    )
                )

    missing = [template_id for template_id in expected if not choices[template_id]]
    if missing:
        raise RuntimeError(f"no completed pilot trajectory for: {missing}")
    cases = [min(choices[template_id], key=lambda item: item[0])[1] for template_id in expected]
    if len(cases) != config["case_count"]:
        raise RuntimeError("reward audit case count differs from configuration")
    return {
        "schema_version": 1,
        "inventory_id": config["audit_id"] + "-cases",
        "selection": {
            "algorithm": (
                "lowest salted SHA-256 completed pilot trajectory per training template"
            ),
            "salt": config["case_selection_salt"],
        },
        "case_count": len(cases),
        "cases": cases,
    }


class CountingWorker:
    def __init__(self, worker: PostgresWorker) -> None:
        self.worker = worker
        self.explain_calls = 0
        self.explain_analyze_calls = 0

    def task_indexes(self, task: dict[str, Any]) -> dict[str, set[str]]:
        return self.worker.task_indexes(task)

    def explain(
        self,
        sql: str,
        timeout_ms: int,
        *,
        analyze: bool = False,
        hint: str = "",
    ) -> ExplainResult:
        if analyze:
            self.explain_analyze_calls += 1
        else:
            self.explain_calls += 1
        return self.worker.explain(sql, timeout_ms, analyze=analyze, hint=hint)


def replay(
    worker: PostgresWorker,
    task_set: TaskSet,
    task: dict[str, Any],
    actions: list[Any],
    protocol: str,
    seed: str,
) -> dict[str, Any]:
    counted = CountingWorker(worker)
    evaluator_type = (
        TrainingRolloutEvaluatorV1
        if protocol == "rl-training-v1"
        else RolloutEvaluator
    )
    evaluator = evaluator_type(  # type: ignore[arg-type]
        counted, task_set, task
    )
    baseline = evaluator.start()
    candidates = [evaluator.evaluate(action) for action in actions]
    final = evaluator.finish(random.Random(seed))
    return {
        "measurement_protocol": evaluator.measurement_protocol.manifest(),
        "default_median_execution_time_ms": baseline[
            "median_execution_time_ms"
        ],
        "candidates": [
            {
                "candidate_id": candidate["candidate_id"],
                "action_valid": candidate["action_valid"],
                "constraints_satisfied": candidate["constraints_satisfied"],
                "duplicate_of": candidate["duplicate_of"],
                "plan_sha256": candidate["plan_sha256"],
                "provisional_speedup": candidate["provisional_speedup"],
            }
            for candidate in candidates
        ],
        "final": final,
        "database_calls": {
            "explain": counted.explain_calls,
            "explain_analyze": counted.explain_analyze_calls,
        },
    }


def direction(value: float, material_ratio: float = 1.0) -> str:
    if value > material_ratio:
        return "improved"
    if value < 1 / material_ratio:
        return "regressed"
    return "neutral"


def sign(value: float) -> int:
    return (value > 0) - (value < 0)


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def summarize(results: list[dict[str, Any]], material_ratio: float) -> dict[str, Any]:
    comparable = [
        result
        for result in results
        if all(
            result[protocol]["final"]["status"] == "completed"
            for protocol in ("rl-training-v1", "rigorous-evaluation-v1")
        )
    ]
    cheap_scores = [
        float(result["rl-training-v1"]["final"]["score"])
        for result in comparable
    ]
    full_scores = [
        float(result["rigorous-evaluation-v1"]["final"]["score"])
        for result in comparable
    ]
    strict_agreements = [
        direction(cheap) == direction(full)
        for cheap, full in zip(cheap_scores, full_scores)
    ]
    material_agreements = [
        direction(cheap, material_ratio) == direction(full, material_ratio)
        for cheap, full in zip(cheap_scores, full_scores)
    ]
    opposite_material = [
        {
            direction(cheap, material_ratio),
            direction(full, material_ratio),
        }
        == {"improved", "regressed"}
        for cheap, full in zip(cheap_scores, full_scores)
    ]
    reward_agreements = [
        sign(float(result["rl-training-v1"]["final"]["trajectory_reward"]))
        == sign(
            float(
                result["rigorous-evaluation-v1"]["final"][
                    "trajectory_reward"
                ]
            )
        )
        for result in comparable
    ]
    winner_agreements = [
        result["rl-training-v1"]["final"]["winning_candidate_id"]
        == result["rigorous-evaluation-v1"]["final"][
            "winning_candidate_id"
        ]
        for result in comparable
    ]
    count = len(comparable)
    return {
        "case_count": len(results),
        "comparable_case_count": count,
        "strict_speedup_direction_agreement_count": sum(strict_agreements),
        "strict_speedup_direction_agreement_rate": (
            sum(strict_agreements) / count if count else None
        ),
        "material_speedup_ratio": material_ratio,
        "material_direction_agreement_count": sum(material_agreements),
        "material_direction_agreement_rate": (
            sum(material_agreements) / count if count else None
        ),
        "opposite_material_direction_count": sum(opposite_material),
        "reward_sign_agreement_count": sum(reward_agreements),
        "reward_sign_agreement_rate": (
            sum(reward_agreements) / count if count else None
        ),
        "winning_candidate_agreement_count": sum(winner_agreements),
        "winning_candidate_agreement_rate": (
            sum(winner_agreements) / count if count else None
        ),
        "log_speedup_pearson_correlation": correlation(
            [math.log(value) for value in cheap_scores],
            [math.log(value) for value in full_scores],
        ),
        "mean_absolute_log_speedup_error": (
            statistics.fmean(
                abs(math.log(cheap) - math.log(full))
                for cheap, full in zip(cheap_scores, full_scores)
            )
            if count
            else None
        ),
        "training_explain_analyze_calls": sum(
            result["rl-training-v1"]["database_calls"]["explain_analyze"]
            for result in results
        ),
        "rigorous_explain_analyze_calls": sum(
            result["rigorous-evaluation-v1"]["database_calls"][
                "explain_analyze"
            ]
            for result in results
        ),
    }


def run_audit(
    repository: Path,
    config_path: Path,
    config: dict[str, Any],
    case_path: Path,
    output_dir: Path,
) -> Path:
    case_manifest = json.loads(case_path.read_text(encoding="utf-8"))
    if case_manifest.get("inventory_id") != config["audit_id"] + "-cases":
        raise RuntimeError("unexpected reward-protocol case manifest")

    fixture = DatabaseFixture.load(repository)
    task_set = TaskSet.load(
        repository, config["task_set"], fixture.identity
    )
    tasks = {task["task_id"]: task for task in task_set.inventory["tasks"]}
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "config_sha256": sha256_file(config_path),
        "case_manifest_sha256": sha256_file(case_path),
        "snapshot_manifest_sha256": sha256_file(
            fixture.snapshot_manifest_path
        ),
        "results": [],
        "summary": None,
    }
    write_json(report_path, report)

    project_name = f"qorl-reward-audit-{os.getpid()}"
    with PostgresWorker(fixture, project_name) as worker:
        worker.capture_environment(output_dir, "pre")
        for index, case in enumerate(case_manifest["cases"]):
            task = tasks[case["task_id"]]
            order = list(config["protocols"])
            if index % 2:
                order.reverse()
            print(
                f"[{index + 1}/{case_manifest['case_count']}] "
                f"{task['task_id']} ({task['template_id']})",
                flush=True,
            )
            result: dict[str, Any] = {
                "task_id": task["task_id"],
                "template_id": task["template_id"],
                "trace_id": case["trace_id"],
                "actions_sha256": case["actions_sha256"],
                "protocol_order": order,
            }
            for protocol in order:
                result[protocol] = replay(
                    worker,
                    task_set,
                    task,
                    case["actions"],
                    protocol,
                    f"{config['pair_order_seed']}:{task['task_id']}:pairs",
                )
                final = result[protocol]["final"]
                label = (
                    f"{final['score']:.3f}x"
                    if final["status"] == "completed"
                    else final["status"]
                )
                print(f"  {protocol}: {label}", flush=True)
            report["results"].append(result)
            write_json(report_path, report)
        worker.capture_environment(output_dir, "post")

    report["status"] = "completed"
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    report["summary"] = summarize(
        report["results"], config["material_speedup_ratio"]
    )
    write_json(report_path, report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cheap RL rewards with rigorous evaluation rewards."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--build-cases", action="store_true")
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    repository = arguments.repository.resolve()
    config_path, config = load_config(repository, arguments.config)
    case_path = arguments.cases or Path(config["case_manifest"])
    case_path = case_path if case_path.is_absolute() else repository / case_path
    if arguments.build_cases:
        trace_root = arguments.trace_root or Path(config["source_traces"])
        trace_root = (
            trace_root if trace_root.is_absolute() else repository / trace_root
        )
        write_json(case_path, build_cases(repository, config, trace_root))
        print(case_path)
        return

    output_dir = arguments.output or Path("outputs/rl") / config["audit_id"]
    output_dir = (
        output_dir if output_dir.is_absolute() else repository / output_dir
    )
    print(
        run_audit(
            repository, config_path, config, case_path, output_dir
        )
    )


if __name__ == "__main__":
    main()
