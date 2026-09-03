from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from qorl.agent import QoAgentConfig
from qorl.agent.types import StopReason, ToolName
from qorl.measure.schemas import RunStatus
from qorl.sft.assemble import action_families, canonical_json
from qorl.sft.schemas import (
    JSON_OBJECT_ADAPTER,
    ActionFamily,
    DatasetConfig,
    FilterManifest,
    FilterRecord,
    FilterSummary,
    JsonObject,
    SampleRecord,
    load_json_object,
    load_record,
    require_list,
    require_object,
)
from qorl.util.hashing import sha256_file
from qorl.util.io import write_json

FILTER_ID = "qorl-protocol-sft-v2-filter-v1"


def tool_calls(trace: JsonObject) -> tuple[list[tuple[str, JsonObject]], str | None]:
    calls: list[tuple[str, JsonObject]] = []
    transcript = require_list(trace.get("transcript"), "policy_trace.transcript")
    for raw_message in transcript:
        message = require_object(raw_message, "policy_trace.transcript[]")
        if message.get("role") != "assistant":
            continue
        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list) or len(raw_calls) != 1:
            return [], "not_exactly_one_tool_call"
        call = require_object(raw_calls[0], "tool_call")
        function = require_object(call.get("function"), "tool_call.function")
        name = function.get("name")
        if not isinstance(name, str):
            return [], "malformed_tool_call"
        arguments = function.get("arguments")
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            parsed_object = JSON_OBJECT_ADAPTER.validate_python(parsed)
        except (json.JSONDecodeError, ValueError):
            return [], "malformed_tool_arguments"
        calls.append((name, parsed_object))
    return calls, None


def rejection_reason(record: SampleRecord, context_length: int) -> str | None:
    if record.status != RunStatus.COMPLETED:
        return "rollout_failed"
    trace = record.policy_trace
    if trace is None:
        return "missing_policy_trace"
    calls, call_error = tool_calls(trace)
    if call_error is not None:
        return call_error
    names = [name for name, _ in calls]
    if ToolName.KEEP_DEFAULT in names:
        return "keep_default"
    if trace.get("stop_reason") != StopReason.MODEL_FINISH:
        return "missing_finish"
    if names.count(ToolName.EVALUATE_CANDIDATE) != 1:
        return "candidate_count"
    if not names or names[-2:] != [ToolName.EVALUATE_CANDIDATE, ToolName.FINISH]:
        return "decision_sequence"

    events = trace.get("tool_events")
    if not isinstance(events, list):
        return "missing_tool_events"
    for raw_event in events:
        event = require_object(raw_event, "policy_trace.tool_events[]")
        result = event.get("result")
        if isinstance(result, dict) and isinstance(result.get("error"), str):
            return "tool_error"

    seen_inspections: set[tuple[str, str]] = set()
    for name, arguments in calls[:-2]:
        if name == ToolName.GET_PLAN and arguments.get("candidate_id") != "default":
            return "invented_candidate_id"
        key = (name, canonical_json(arguments))
        if key in seen_inspections:
            return "repeated_inspection"
        seen_inspections.add(key)

    if len(record.candidates) != 1:
        return "candidate_record_count"
    candidate = record.candidates[0]
    if not candidate.action_valid:
        return "malformed_action"
    if not candidate.constraints_satisfied:
        return "constraints_not_satisfied"
    if candidate.duplicate_of is not None:
        return "default_duplicate"
    if candidate.plan_sha256 is None:
        return "missing_plan_fingerprint"

    responses = trace.get("model_responses")
    if not isinstance(responses, list):
        return "missing_model_responses"
    maximum_tokens = 0
    for raw_response in responses:
        response = require_object(raw_response, "policy_trace.model_responses[]")
        usage = response.get("usage")
        if not isinstance(usage, dict):
            continue
        total = usage.get("total_tokens", 0)
        if isinstance(total, int):
            maximum_tokens = max(maximum_tokens, total)
    if maximum_tokens >= context_length:
        return "context_limit"
    return None


def filter_records(
    samples: list[tuple[Path, SampleRecord]],
    context_length: int,
    syntax_examples_per_task: int,
) -> list[FilterRecord]:
    results: list[FilterRecord] = []
    seen: set[tuple[str, str]] = set()
    accepted_by_task: Counter[str] = Counter()
    for path, sample in sorted(
        samples, key=lambda item: (item[1].task_id, item[1].sample)
    ):
        reason = rejection_reason(sample, context_length)
        candidate = sample.candidates[0] if len(sample.candidates) == 1 else None
        fingerprint = candidate.plan_sha256 if candidate is not None else None
        if reason is None:
            if fingerprint is None:
                raise RuntimeError("accepted sample has no plan fingerprint")
            key = (sample.task_id, fingerprint)
            if key in seen:
                reason = "task_fingerprint_duplicate"
            else:
                seen.add(key)
        accepted = reason is None
        if accepted:
            accepted_by_task[sample.task_id] += 1
        action = (
            JSON_OBJECT_ADAPTER.validate_python(candidate.action)
            if accepted and candidate is not None
            else None
        )
        if accepted and action is None:
            raise RuntimeError("accepted sample has no PlanAction object")
        results.append(
            FilterRecord(
                task_id=sample.task_id,
                template_id=sample.template_id,
                sample=sample.sample,
                sample_path=path.as_posix(),
                accepted=accepted,
                rejection_reason=reason,
                plan_sha256=fingerprint if accepted else None,
                action_families=[
                    ActionFamily(value) for value in action_families(action)
                ]
                if action is not None
                else [],
                syntax_eligible=accepted
                and accepted_by_task[sample.task_id] <= syntax_examples_per_task,
                steered=sample.steered,
            )
        )
    return results


def summarize(
    records: list[FilterRecord], default_best_minimum_fingerprints: int
) -> FilterSummary:
    reasons = Counter(
        record.rejection_reason
        for record in records
        if not record.accepted and record.rejection_reason is not None
    )
    families = Counter(
        family
        for record in records
        if record.accepted
        for family in record.action_families
    )
    accepted = [record for record in records if record.accepted]
    tasks = Counter(record.task_id for record in accepted)
    return FilterSummary(
        rollouts=len(records),
        accepted_distinct_novel_candidates=len(accepted),
        accepted_yield=len(accepted) / len(records) if records else 0.0,
        tasks_with_accepted_candidate=len(tasks),
        tasks_reaching_default_best_budget=sum(
            count >= default_best_minimum_fingerprints for count in tasks.values()
        ),
        rejection_reasons=dict(sorted(reasons.items())),
        action_families=dict(sorted(families.items())),
        steered_accepted=sum(record.accepted and record.steered for record in records),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter protocol SFT v2 samples.")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/005-protocol-sft-v2/dataset.json"),
    )
    parser.add_argument(
        "--input", type=Path, default=Path("outputs/sft/protocol-sft-v2")
    )
    parser.add_argument("--split", choices=("sampling", "validation"), required=True)
    arguments = parser.parse_args()

    repository = arguments.repository.resolve()
    config_path = (repository / arguments.config).resolve()
    input_dir = (repository / arguments.input).resolve()
    config = load_record(config_path, DatasetConfig)
    policy = require_object(
        load_json_object(repository / config.policy_config).get("policy"), "policy"
    )
    context_length = QoAgentConfig.from_dict(policy).context_length
    paths = sorted((input_dir / "samples" / arguments.split).glob("*/sample-*.json"))
    samples = [
        (path.relative_to(input_dir), load_record(path, SampleRecord)) for path in paths
    ]
    records = filter_records(
        samples, context_length, config.assembly.maximum_syntax_examples_per_task
    )

    output_dir = input_dir / "filter" / arguments.split
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(record.to_wire(), sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
    )
    manifest = FilterManifest(
        filter_id=FILTER_ID,
        split=arguments.split,
        config_sha256=sha256_file(config_path),
        records_sha256=sha256_file(records_path),
        summary=summarize(records, config.labels.default_best_minimum_fingerprints),
    )
    write_json(output_dir / "manifest.json", manifest.to_wire())
    print(json.dumps(manifest.summary.to_wire(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
