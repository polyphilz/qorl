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
    ExampleSource,
    FileIdentity,
    FilterManifest,
    FilterRecord,
    FilterSummary,
    JsonObject,
    SampleRecord,
    SamplingMode,
    TeacherGenerationRecord,
    TeacherManifest,
    load_json_lines,
    load_json_object,
    load_record,
    require_list,
    require_object,
)
from qorl.util.hashing import sha256_file
from qorl.util.io import write_json

FILTER_ID = "qorl-protocol-sft-v2-filter-v2"


def load_filtered_sample(
    repository: Path,
    dataset: Path,
    record: FilterRecord,
    source: ExampleSource,
) -> SampleRecord:
    if source == ExampleSource.STUDENT:
        return load_record(dataset / record.sample_path, SampleRecord)
    generation = load_record(repository / record.sample_path, TeacherGenerationRecord)
    if generation.accepted_sample is None:
        raise RuntimeError("teacher filter record refers to a rejected generation")
    return generation.accepted_sample


def load_teacher_records(
    repository: Path, teacher_dir: Path
) -> tuple[TeacherManifest, list[tuple[Path, TeacherGenerationRecord]]]:
    manifest = load_record(teacher_dir / "manifest.json", TeacherManifest)
    paths = sorted((teacher_dir / "records").glob("*/*.json"))
    by_path = {path.relative_to(repository).as_posix(): path for path in paths}
    if set(by_path) != {record.path for record in manifest.records}:
        raise RuntimeError("teacher manifest record inventory differs")
    records: list[tuple[Path, TeacherGenerationRecord]] = []
    for identity in manifest.records:
        path = by_path[identity.path]
        if sha256_file(path) != identity.sha256:
            raise RuntimeError(f"teacher record checksum differs: {identity.path}")
        record = load_record(path, TeacherGenerationRecord)
        if (
            record.task_id != identity.task_id
            or record.template_id != identity.template_id
            or record.requested_family != identity.requested_family
            or (record.accepted_sample is not None) != identity.accepted
            or record.teacher != manifest.teacher
        ):
            raise RuntimeError(f"teacher record identity differs: {identity.path}")
        records.append((path, record))
    return manifest, records


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
    existing: list[FilterRecord] | None = None,
) -> list[FilterRecord]:
    results: list[FilterRecord] = []
    existing = existing or []
    seen = {
        (record.task_id, record.plan_sha256)
        for record in existing
        if record.accepted and record.plan_sha256 is not None
    }
    accepted_by_task = Counter(record.task_id for record in existing if record.accepted)
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
    parser.add_argument(
        "--split",
        choices=("sampling", "default_best", "teacher", "validation"),
        required=True,
    )
    parser.add_argument(
        "--teacher-records",
        type=Path,
        default=Path("experiments/005-protocol-sft-v2/teacher"),
    )
    arguments = parser.parse_args()

    repository = arguments.repository.resolve()
    config_path = (repository / arguments.config).resolve()
    input_dir = (repository / arguments.input).resolve()
    config = load_record(config_path, DatasetConfig)
    policy = require_object(
        load_json_object(repository / config.policy_config).get("policy"), "policy"
    )
    context_length = QoAgentConfig.from_dict(policy).context_length
    source = ExampleSource.STUDENT
    source_manifest: FileIdentity | None = None
    existing: list[FilterRecord] = []
    samples: list[tuple[Path, SampleRecord]]
    syntax_examples_per_task = config.assembly.maximum_syntax_examples_per_task
    if arguments.split == "teacher":
        source = ExampleSource.TEACHER
        teacher_dir = (repository / arguments.teacher_records).resolve()
        _, generation_records = load_teacher_records(repository, teacher_dir)
        teacher_manifest_path = teacher_dir / "manifest.json"
        source_manifest = FileIdentity(
            path=teacher_manifest_path.relative_to(repository).as_posix(),
            sha256=sha256_file(teacher_manifest_path),
        )
        samples = []
        for path, generation in generation_records:
            if generation.accepted_sample is not None:
                samples.append(
                    (path.relative_to(repository), generation.accepted_sample)
                )
    else:
        sample_split = (
            "sampling" if arguments.split == "default_best" else arguments.split
        )
        paths = sorted((input_dir / "samples" / sample_split).glob("*/sample-*.json"))
        samples = [
            (path.relative_to(input_dir), load_record(path, SampleRecord))
            for path in paths
        ]
        if arguments.split == "default_best":
            samples = [
                (path, sample)
                for path, sample in samples
                if sample.sampling_mode == SamplingMode.DEFAULT_BEST
            ]
            syntax_examples_per_task = 0
        sampling_manifest_name = (
            "default-best-manifest.json"
            if arguments.split == "default_best"
            else f"{arguments.split}-manifest.json"
        )
        sampling_manifest_path = input_dir / "sampling" / sampling_manifest_name
        source_manifest = FileIdentity(
            path=sampling_manifest_path.relative_to(repository).as_posix(),
            sha256=sha256_file(sampling_manifest_path),
        )
    if arguments.split in {"default_best", "teacher"}:
        existing = load_json_lines(
            input_dir / "filter/sampling/records.jsonl", FilterRecord
        )
    records = filter_records(
        samples, context_length, syntax_examples_per_task, existing
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
        source=source,
        source_manifest=source_manifest,
        config_sha256=sha256_file(config_path),
        records_sha256=sha256_file(records_path),
        summary=summarize(records, config.labels.default_best_minimum_fingerprints),
    )
    write_json(output_dir / "manifest.json", manifest.to_wire())
    print(json.dumps(manifest.summary.to_wire(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
