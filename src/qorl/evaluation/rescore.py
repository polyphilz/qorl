#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from qorl.measure.schemas import (
    DUPLICATE_ATTEMPT_PENALTY,
    INVALID_ATTEMPT_PENALTY,
    FinalStatus,
)
from qorl.util.io import write_json

EXPECTED_GROUP_SIZE = 4


def trace_records(run_dir: Path, kind: str) -> Iterable[dict[str, Any]]:
    pattern = f"rollouts/step_*/train/{kind}/traces.jsonl"
    for path in run_dir.glob(pattern):
        step = int(next(part[5:] for part in path.parts if part.startswith("step_")))
        with path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                group_id = row["group"]["id"]
                for trace in row["traces"]:
                    yield {"step": step, "group_id": group_id, "trace": trace}


def qorl_result(record: dict[str, Any]) -> dict[str, Any]:
    return record["trace"]["info"]["qorl"]


def original_reward(record: dict[str, Any]) -> float:
    reward = record["trace"]["rewards"]["trajectory_reward"]
    return float(reward["score"])


def has_reward(record: dict[str, Any]) -> bool:
    reward = record["trace"].get("rewards", {}).get("trajectory_reward")
    return isinstance(reward, dict) and reward.get("score") is not None


def matches_default(record: dict[str, Any]) -> bool:
    result = qorl_result(record)
    final = result["final"]
    return (
        final["status"] == FinalStatus.COMPLETED
        and final.get("winning_plan_sha256") == result["default"]["plan_sha256"]
    )


def rescored_reward(record: dict[str, Any]) -> float:
    if not matches_default(record):
        return original_reward(record)
    final = qorl_result(record)["final"]
    return -INVALID_ATTEMPT_PENALTY * int(
        final["invalid_attempt_count"]
    ) - DUPLICATE_ATTEMPT_PENALTY * int(final["duplicate_attempt_count"])


def geometric_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(statistics.fmean(math.log(value) for value in values))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    original_scores = [
        row["original_score"] for row in rows if row["original_score"] is not None
    ]
    rescored_scores = [
        row["rescored_score"] for row in rows if row["rescored_score"] is not None
    ]
    matches = [row for row in rows if row["default_fingerprint_match"]]

    def advantage_summary(prefix: str) -> dict[str, float]:
        absolute = sum(abs(row[f"{prefix}_advantage"]) for row in rows)
        token_weighted = sum(
            abs(row[f"{prefix}_advantage"]) * row["output_tokens"] for row in rows
        )
        match_absolute = sum(abs(row[f"{prefix}_advantage"]) for row in matches)
        match_token_weighted = sum(
            abs(row[f"{prefix}_advantage"]) * row["output_tokens"] for row in matches
        )
        return {
            "scalar_absolute_mass": absolute,
            "token_weighted_absolute_mass": token_weighted,
            "default_match_scalar_absolute_mass": match_absolute,
            "default_match_token_weighted_absolute_mass": match_token_weighted,
        }

    original_advantage = advantage_summary("original")
    rescored_advantage = advantage_summary("rescored")
    return {
        "effective_rollout_count": len(rows),
        "task_group_count": len({row["group_id"] for row in rows}),
        "completed_rollout_count": len(original_scores),
        "default_fingerprint_match_count": len(matches),
        "default_fingerprint_match_rate": len(matches) / len(rows),
        "default_matches_previously_above_1x": sum(
            row["original_score"] > 1.0 for row in matches
        ),
        "default_matches_previously_below_1x": sum(
            row["original_score"] < 1.0 for row in matches
        ),
        "original": {
            "geometric_mean_speedup": geometric_mean(original_scores),
            "mean_reward": statistics.fmean(row["original_reward"] for row in rows),
            "advantage": original_advantage,
        },
        "rescored": {
            "geometric_mean_speedup": geometric_mean(rescored_scores),
            "mean_reward": statistics.fmean(row["rescored_reward"] for row in rows),
            "advantage": rescored_advantage,
        },
        "change": {
            "geometric_mean_speedup": (
                geometric_mean(rescored_scores) - geometric_mean(original_scores)
            ),
            "mean_reward": statistics.fmean(row["rescored_reward"] for row in rows)
            - statistics.fmean(row["original_reward"] for row in rows),
            "scalar_absolute_advantage_mass": (
                rescored_advantage["scalar_absolute_mass"]
                - original_advantage["scalar_absolute_mass"]
            ),
            "token_weighted_absolute_advantage_mass": (
                rescored_advantage["token_weighted_absolute_mass"]
                - original_advantage["token_weighted_absolute_mass"]
            ),
        },
    }


def rescore(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    effective = list(trace_records(run_dir, "effective"))
    all_by_group: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in trace_records(run_dir, "all"):
        trace_id = record["trace"]["id"]
        all_by_group[record["group_id"]][trace_id] = record

    rows: list[dict[str, Any]] = []
    maximum_advantage_error = 0.0
    for record in effective:
        whole_group = list(all_by_group[record["group_id"]].values())
        if len(whole_group) != EXPECTED_GROUP_SIZE:
            raise RuntimeError(
                f"group {record['group_id']} has {len(whole_group)} traces, "
                f"expected {EXPECTED_GROUP_SIZE}"
            )
        group = [item for item in whole_group if has_reward(item)]
        if not group:
            raise RuntimeError(f"group {record['group_id']} has no scored traces")
        original_mean = statistics.fmean(original_reward(item) for item in group)
        rescored_mean = statistics.fmean(rescored_reward(item) for item in group)
        original_advantage = original_reward(record) - original_mean
        stored_advantage = float(record["trace"]["info"]["advantage"])
        maximum_advantage_error = max(
            maximum_advantage_error,
            abs(original_advantage - stored_advantage),
        )
        result = qorl_result(record)
        final = result["final"]
        default_match = matches_default(record)
        rows.append(
            {
                "trace_id": record["trace"]["id"],
                "step": record["step"],
                "group_id": record["group_id"],
                "task_id": result["task_id"],
                "default_fingerprint_match": default_match,
                "original_score": (
                    float(final["score"])
                    if final["status"] == FinalStatus.COMPLETED
                    else None
                ),
                "rescored_score": (
                    1.0
                    if default_match
                    else float(final["score"])
                    if final["status"] == FinalStatus.COMPLETED
                    else None
                ),
                "original_reward": original_reward(record),
                "rescored_reward": rescored_reward(record),
                "original_advantage": original_advantage,
                "rescored_advantage": rescored_reward(record) - rescored_mean,
                "output_tokens": float(record["trace"]["metrics"]["output_tokens"]),
            }
        )

    rows.sort(key=lambda row: (row["step"], row["trace_id"]))
    summary = {
        "schema_version": 1,
        "analysis_id": "rl-run-v2-default-fingerprint-rescore-v1",
        "rule": (
            "For the same query and frozen benchmark environment, an exact "
            "default-plan fingerprint match has quality score 1.0x. Existing "
            "invalid and duplicate protocol costs remain unchanged."
        ),
        "source": str(run_dir),
        "stored_advantage_maximum_absolute_error": maximum_advantage_error,
        "summary": summarize(rows),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score exact default-plan matches in a completed RL run."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    rows, summary = rescore(args.run_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "summary.json", summary)
    with (args.output_dir / "effective-rollouts.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
