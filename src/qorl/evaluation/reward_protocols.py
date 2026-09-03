from __future__ import annotations

import argparse
import contextlib
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

from qorl.db.fixture import DatabaseFixture
from qorl.db.pool import start_pool
from qorl.db.worker import ExplainResult, PostgresWorker
from qorl.measure.rollout import (
    MATERIAL_SHARED_READ_FRACTION,
    RolloutEvaluator,
    has_material_shared_reads,
    training_protocol,
)
from qorl.measure.schemas import (
    FinalStatus,
    MeasurementProtocolId,
    RunStatus,
    ScoreSource,
)
from qorl.util.hashing import sha256_file
from qorl.util.io import write_json
from qorl.workload.taskset import TaskSet

DEFAULT_CONFIG = Path("experiments/004-rl-run-v2/reward-protocol-audit/config.json")
MIN_CORRELATION_SAMPLES = 2


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
                    or final.get("status") != FinalStatus.COMPLETED
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
    cases = [
        min(choices[template_id], key=lambda item: item[0])[1]
        for template_id in expected
    ]
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
    if protocol == MeasurementProtocolId.RL_TRAINING_V1:
        evaluator = RolloutEvaluator(
            counted,
            task_set,
            task,
            measurement_protocol=training_protocol(
                MeasurementProtocolId.RL_TRAINING_V1
            ),
        )
    else:
        evaluator = RolloutEvaluator(counted, task_set, task)
    baseline = evaluator.start()
    candidates = [evaluator.evaluate(action) for action in actions]
    final = evaluator.finish(random.Random(seed))
    provisional_measurements = [
        measurement
        for candidate in candidates
        if candidate.duplicate_of is None
        for measurement in candidate.provisional_measurements
    ]
    final_measurements = (
        final.candidate_measurements
        if final.score_source == ScoreSource.INTERLEAVED_MEASUREMENT
        or final.status == FinalStatus.CANDIDATE_TIMEOUT
        else []
    )
    retained_candidate_measurements = provisional_measurements + final_measurements
    cold_read_repeat_count = (
        sum(candidate.cold_read_repeat_count for candidate in candidates)
        + final.cold_read_repeat_count
    )
    initial_candidate_measurement_count = (
        sum(
            len(candidate.provisional_measurements) + int(candidate.execution_timed_out)
            for candidate in candidates
            if candidate.duplicate_of is None
        )
        + len(final_measurements)
        + int(final.status == FinalStatus.CANDIDATE_TIMEOUT)
    )
    repeat_fraction = evaluator.measurement_protocol.cold_read_repeat_fraction
    retained_material_shared_read_count = sum(
        has_material_shared_reads(measurement, MATERIAL_SHARED_READ_FRACTION)
        for measurement in retained_candidate_measurements
    )
    return {
        "measurement_protocol": evaluator.measurement_protocol.manifest(),
        "default_median_execution_time_ms": baseline.median_execution_time_ms,
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "action_valid": candidate.action_valid,
                "constraints_satisfied": candidate.constraints_satisfied,
                "duplicate_of": candidate.duplicate_of,
                "plan_sha256": candidate.plan_sha256,
                "provisional_speedup": candidate.provisional_speedup,
                "cold_read_repeat_count": candidate.cold_read_repeat_count,
            }
            for candidate in candidates
        ],
        "final": final.to_wire(),
        "cold_read_guard": {
            "audit_fraction": MATERIAL_SHARED_READ_FRACTION,
            "repeat_fraction": repeat_fraction,
            "repeat_count": cold_read_repeat_count,
            "initial_candidate_measurement_count": (
                initial_candidate_measurement_count
            ),
            "retained_candidate_measurement_count": len(
                retained_candidate_measurements
            ),
            "retained_material_shared_read_count": (
                retained_material_shared_read_count
            ),
        },
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


def summarize_cold_reads(
    results: list[dict[str, Any]], protocol: MeasurementProtocolId
) -> dict[str, Any]:
    guards = [result[protocol]["cold_read_guard"] for result in results]
    repeat_count = sum(guard["repeat_count"] for guard in guards)
    initial_count = sum(
        guard["initial_candidate_measurement_count"] for guard in guards
    )
    retained_count = sum(
        guard["retained_candidate_measurement_count"] for guard in guards
    )
    retained_material_count = sum(
        guard["retained_material_shared_read_count"] for guard in guards
    )
    return {
        "guard_repeat_count": repeat_count,
        "guard_repeat_rate": (repeat_count / initial_count if initial_count else None),
        "initial_candidate_measurement_count": initial_count,
        "retained_candidate_measurement_count": retained_count,
        "retained_material_shared_read_count": retained_material_count,
        "retained_material_shared_read_rate": (
            retained_material_count / retained_count if retained_count else None
        ),
    }


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < MIN_CORRELATION_SAMPLES:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
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
            result[protocol]["final"]["status"] == FinalStatus.COMPLETED
            for protocol in (
                MeasurementProtocolId.RL_TRAINING_V1,
                MeasurementProtocolId.RIGOROUS_EVALUATION_V1,
            )
        )
    ]
    cheap_scores = [
        float(result["rl-training-v1"]["final"]["score"]) for result in comparable
    ]
    full_scores = [
        float(result["rigorous-evaluation-v1"]["final"]["score"])
        for result in comparable
    ]
    strict_agreements = [
        direction(cheap) == direction(full)
        for cheap, full in zip(cheap_scores, full_scores, strict=True)
    ]
    material_agreements = [
        direction(cheap, material_ratio) == direction(full, material_ratio)
        for cheap, full in zip(cheap_scores, full_scores, strict=True)
    ]
    opposite_material = [
        {
            direction(cheap, material_ratio),
            direction(full, material_ratio),
        }
        == {"improved", "regressed"}
        for cheap, full in zip(cheap_scores, full_scores, strict=True)
    ]
    reward_agreements = [
        sign(float(result["rl-training-v1"]["final"]["trajectory_reward"]))
        == sign(float(result["rigorous-evaluation-v1"]["final"]["trajectory_reward"]))
        for result in comparable
    ]
    winner_agreements = [
        result["rl-training-v1"]["final"]["winning_candidate_id"]
        == result["rigorous-evaluation-v1"]["final"]["winning_candidate_id"]
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
                for cheap, full in zip(cheap_scores, full_scores, strict=True)
            )
            if count
            else None
        ),
        "cold_read_audit": {
            "material_shared_read_fraction": MATERIAL_SHARED_READ_FRACTION,
            MeasurementProtocolId.RL_TRAINING_V1.value: summarize_cold_reads(
                results, MeasurementProtocolId.RL_TRAINING_V1
            ),
            MeasurementProtocolId.RIGOROUS_EVALUATION_V1.value: (
                summarize_cold_reads(
                    results, MeasurementProtocolId.RIGOROUS_EVALUATION_V1
                )
            ),
        },
        "training_explain_analyze_calls": sum(
            result["rl-training-v1"]["database_calls"]["explain_analyze"]
            for result in results
        ),
        "rigorous_explain_analyze_calls": sum(
            result["rigorous-evaluation-v1"]["database_calls"]["explain_analyze"]
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
    task_set = TaskSet.load(repository, config["task_set"], fixture.data_identity)
    tasks = {task["task_id"]: task for task in task_set.inventory["tasks"]}
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "status": RunStatus.RUNNING.value,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "completed_at_utc": None,
        "config_sha256": sha256_file(config_path),
        "case_manifest_sha256": sha256_file(case_path),
        "snapshot_manifest_sha256": sha256_file(fixture.snapshot_manifest_path),
        "data_identity": fixture.data_identity,
        "runtime_identity": fixture.runtime_identity,
        "results": [],
        "summary": None,
    }
    write_json(report_path, report)

    project_name = f"qorl-reward-audit-{os.getpid()}"
    with contextlib.closing(start_pool(fixture, project_name)) as pool:
        report["database_pool"] = pool.manifest()
        with pool.claim_worker() as slot:
            report["worker"] = slot.resources.manifest()
            write_json(report_path, report)
            worker = slot.worker
            slot.container.capture_environment(output_dir, "pre")
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
                        if final["status"] == FinalStatus.COMPLETED
                        else final["status"]
                    )
                    print(f"  {protocol}: {label}", flush=True)
                report["results"].append(result)
                write_json(report_path, report)
            slot.container.capture_environment(output_dir, "post")

    report["status"] = RunStatus.COMPLETED.value
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    report["summary"] = summarize(report["results"], config["material_speedup_ratio"])
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
        trace_root = trace_root if trace_root.is_absolute() else repository / trace_root
        write_json(case_path, build_cases(repository, config, trace_root))
        print(case_path)
        return

    output_dir = arguments.output or Path("outputs/rl") / config["audit_id"]
    output_dir = output_dir if output_dir.is_absolute() else repository / output_dir
    print(run_audit(repository, config_path, config, case_path, output_dir))


if __name__ == "__main__":
    main()
