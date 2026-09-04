from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from qorl.agent.types import TURN_BUDGET_FIELD, ToolName
from qorl.measure.schemas import ToolResultStatus
from qorl.sft.assemble import canonical_json
from qorl.sft.filter import load_filtered_sample
from qorl.sft.schemas import (
    JSON_OBJECT_ADAPTER,
    JSON_OBJECT_LIST_ADAPTER,
    ActionFamily,
    CandidateEvidence,
    CandidateLabel,
    CandidateMeasurement,
    DatasetConfig,
    DatasetInputs,
    DatasetManifest,
    DatasetSelection,
    DatasetSelectionIdentity,
    DefaultBestProvenance,
    DemonstrationDocument,
    DemonstrationEvidence,
    DemonstrationIdentity,
    DemonstrationMetadata,
    DemonstrationProvenance,
    ExampleKind,
    ExampleSource,
    FileIdentity,
    FilterProvenance,
    FilterRecord,
    JsonObject,
    MeasurementManifest,
    MeasurementProvenance,
    PrimeArtifact,
    SampleRecord,
    TaskLabel,
    TeacherConfig,
    load_json_lines,
    load_record,
    require_list,
    require_object,
    require_string,
)
from qorl.sft.validate import validate_protocol_demo
from qorl.util.hashing import sha256_file
from qorl.util.io import write_json
from qorl.workload.taskset import TaskSet

DATASET_ID = "protocol-sft-v2"
KEEP_DEFAULT_CALL_ID = "sft-v2-keep-default"
TEACHER_ID = "iterated_rejection_sampling_v1"
FABLE_TEACHER_ID = "protocol-sft-v2-fable-5-1"
MEASUREMENT_MODE = "rejection_sampling_plan_validation"


@dataclass(frozen=True)
class SelectedExample:
    record: FilterRecord
    source: ExampleSource
    kind: ExampleKind
    measurement: CandidateMeasurement | DefaultBestProvenance | None


@dataclass(frozen=True)
class SourcedRecord:
    record: FilterRecord
    source: ExampleSource


def stable_rank(seed: int, label: str, item: SourcedRecord) -> str:
    record = item.record
    identity = (
        f"{seed}:{label}:{item.source.value}:{record.task_id}:"
        f"{record.plan_sha256}:{record.sample}"
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def load_measurements(
    dataset: Path,
) -> tuple[dict[tuple[str, str], CandidateMeasurement], dict[str, TaskLabel]]:
    manifest = load_record(dataset / "measurement.json", MeasurementManifest)
    if manifest.task_labels is None:
        raise RuntimeError("measurement manifest has no task labels")
    measurements = {
        (record.task_id, record.plan_sha256): record
        for path in sorted((dataset / "measurements").glob("*/*.json"))
        for record in [load_record(path, CandidateMeasurement)]
    }
    return measurements, manifest.task_labels


def tool_sequence(messages: list[JsonObject]) -> list[str]:
    names: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        calls = require_list(message.get("tool_calls"), "assistant.tool_calls")
        call = require_object(calls[0], "assistant.tool_calls[0]")
        function = require_object(call.get("function"), "assistant.tool_call.function")
        names.append(require_string(function.get("name"), "assistant.tool_call.name"))
    return names


def keep_default_messages(messages: list[JsonObject]) -> list[JsonObject]:
    decision_index: int | None = None
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            continue
        call = require_object(calls[0], "assistant.tool_calls[0]")
        function = require_object(call.get("function"), "assistant.tool_call.function")
        if function.get("name") == ToolName.EVALUATE_CANDIDATE:
            decision_index = index
            break
    if decision_index is None or decision_index + 1 >= len(messages):
        raise RuntimeError("sample transcript has no candidate decision")
    result_message = messages[decision_index + 1]
    content = require_string(result_message.get("content"), "candidate tool result")
    result = JSON_OBJECT_ADAPTER.validate_json(content)
    budget = require_object(result.get(TURN_BUDGET_FIELD), "candidate turn budget")
    keep_call: JsonObject = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": KEEP_DEFAULT_CALL_ID,
                "type": "function",
                "function": {"name": ToolName.KEEP_DEFAULT.value, "arguments": "{}"},
            }
        ],
    }
    keep_result: JsonObject = {
        "role": "tool",
        "tool_call_id": KEEP_DEFAULT_CALL_ID,
        "name": ToolName.KEEP_DEFAULT.value,
        "content": json.dumps(
            {"status": ToolResultStatus.KEPT_DEFAULT.value, TURN_BUDGET_FIELD: budget},
            sort_keys=True,
        ),
    }
    return [*messages[:decision_index], keep_call, keep_result]


def measurement_provenance(
    value: CandidateMeasurement | DefaultBestProvenance | None,
) -> MeasurementProvenance | DefaultBestProvenance | None:
    if value is None or isinstance(value, DefaultBestProvenance):
        return value
    if value.candidate_label is None or value.score_interval is None:
        raise RuntimeError("selected measurement has not been labeled")
    return MeasurementProvenance(
        plan_sha256=value.plan_sha256,
        candidate_label=value.candidate_label,
        score_interval=value.score_interval,
        attempt_count=len(value.attempts),
    )


def build_document(
    repository: Path,
    task_set: TaskSet,
    sample: SampleRecord,
    selected: SelectedExample,
    ordinal: int,
) -> DemonstrationDocument:
    trace = sample.policy_trace
    messages = sample.training_transcript
    if trace is None or messages is None or sample.default is None:
        raise RuntimeError("selected sample is incomplete")
    if selected.kind == ExampleKind.KEEP_DEFAULT:
        messages = keep_default_messages(messages)
    tasks = JSON_OBJECT_LIST_ADAPTER.validate_python(task_set.inventory["tasks"])
    task = next(item for item in tasks if item.get("task_id") == sample.task_id)
    candidates: dict[str, CandidateEvidence] = {}
    if selected.kind != ExampleKind.KEEP_DEFAULT:
        if len(sample.candidates) != 1:
            raise RuntimeError("selected sample has no single candidate")
        candidate = sample.candidates[0]
        if candidate.plain_explain is None:
            raise RuntimeError("selected candidate has no plan")
        candidates[candidate.candidate_id] = CandidateEvidence(
            action=JSON_OBJECT_ADAPTER.validate_python(candidate.action),
            plain_explain=JSON_OBJECT_ADAPTER.validate_python(candidate.plain_explain),
            pg_hint_plan=candidate.pg_hint_plan,
        )
    initial = require_object(
        trace.get("initial_observation"), "trace.initial_observation"
    )
    budget = require_object(initial.get("turn_budget"), "initial turn budget")
    document = DemonstrationDocument(
        messages=messages,
        tools=[
            require_object(item, "trace.tools[]")
            for item in require_list(trace.get("tools"), "trace.tools")
        ],
        metadata=DemonstrationMetadata(
            demonstration_id=f"{DATASET_ID}-{ordinal + 1:04d}",
            ordinal=ordinal,
            teacher=FABLE_TEACHER_ID
            if selected.source == ExampleSource.TEACHER
            else TEACHER_ID,
            task_set_id=task_set.task_set_id,
            task_id=sample.task_id,
            template_id=sample.template_id,
            partition=require_string(task.get("partition"), "task.partition"),
            sql_sha256=require_string(task.get("sql_sha256"), "task.sql_sha256"),
            data_identity=sample.data_identity,
            runtime_identity=sample.runtime_identity,
            in_author_unique_plans_subset=task.get("in_author_unique_plans_subset")
            is True,
            trace_seed=sample.seed,
            maximum_model_turns=require_positive_int(
                budget.get("total_model_turns"), "turn budget"
            ),
            candidate_count=0 if selected.kind == ExampleKind.KEEP_DEFAULT else 1,
            measurement_mode=MEASUREMENT_MODE,
            selection_used_speed=selected.kind
            in {ExampleKind.WIN, ExampleKind.KEEP_DEFAULT},
            example_kind=selected.kind,
            call_sequence=tool_sequence(messages),
        ),
        provenance=DemonstrationProvenance(
            source=selected.source,
            sample=sample.sample,
            sampler=sample.sampler,
            filter=FilterProvenance(
                accepted=selected.record.accepted,
                rejection_reason=selected.record.rejection_reason,
                syntax_eligible=selected.record.syntax_eligible,
                action_families=selected.record.action_families,
            ),
            budget=budget,
            measurement=measurement_provenance(selected.measurement),
        ),
        evidence=DemonstrationEvidence(
            default_plan=JSON_OBJECT_ADAPTER.validate_python(
                sample.default.plain_explain
            ),
            candidates=candidates,
        ),
    )
    validate_protocol_demo(document.to_wire(), repository)
    return document


def require_positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def candidate_measurement(
    record: FilterRecord,
    measurements: dict[tuple[str, str], CandidateMeasurement],
) -> CandidateMeasurement | None:
    return measurements.get((record.task_id, record.plan_sha256 or ""))


def measured_lower(measurement: CandidateMeasurement) -> float:
    if measurement.score_interval is None:
        raise RuntimeError("candidate measurement has no score interval")
    return measurement.score_interval.lower


def select_training(
    records: list[SourcedRecord],
    measurements: dict[tuple[str, str], CandidateMeasurement],
    task_labels: dict[str, TaskLabel],
    config: DatasetConfig,
    teacher_config: TeacherConfig,
) -> list[SelectedExample]:
    target = config.split_counts.train
    desired_wins = round(target * config.composition.win_share)
    desired_defaults = round(target * config.composition.keep_default_share)
    maximum_teacher_examples = int(target * teacher_config.maximum_teacher_share)
    wins = [
        (item, measurement)
        for item in records
        if item.source == ExampleSource.STUDENT
        and item.record.accepted
        and (measurement := candidate_measurement(item.record, measurements))
        is not None
        and measurement.candidate_label == CandidateLabel.WIN
        and measurement.score_interval is not None
    ]
    wins.sort(
        key=lambda item: (
            -measured_lower(item[1]),
            stable_rank(config.seed, "win", item[0]),
        )
    )
    default_samples: list[SourcedRecord] = []
    seen_default_tasks: set[str] = set()
    for item in sorted(
        records,
        key=lambda value: stable_rank(config.seed, "default", value),
    ):
        record = item.record
        if (
            item.source == ExampleSource.STUDENT
            and record.accepted
            and task_labels.get(record.task_id) == TaskLabel.DEFAULT_BEST
            and record.task_id not in seen_default_tasks
        ):
            default_samples.append(item)
            seen_default_tasks.add(record.task_id)

    selected: list[SelectedExample] = []
    selected_per_task: Counter[str] = Counter()
    family_totals: Counter[ActionFamily] = Counter()
    source_totals: Counter[ExampleSource] = Counter()

    def allowed(item: SourcedRecord) -> bool:
        record = item.record
        if len(selected) >= target:
            return False
        if (
            selected_per_task[record.task_id]
            >= config.assembly.maximum_examples_per_task
        ):
            return False
        return not (
            item.source == ExampleSource.TEACHER
            and source_totals[ExampleSource.TEACHER] >= maximum_teacher_examples
        )

    def add(
        item: SourcedRecord,
        kind: ExampleKind,
        measurement: CandidateMeasurement | DefaultBestProvenance | None,
    ) -> bool:
        if not allowed(item):
            return False
        record = item.record
        selected.append(SelectedExample(record, item.source, kind, measurement))
        selected_per_task[record.task_id] += 1
        source_totals[item.source] += 1
        if kind != ExampleKind.KEEP_DEFAULT:
            family_totals.update(record.action_families)
        return True

    def measured_kind(item: SourcedRecord) -> ExampleKind:
        if item.source == ExampleSource.TEACHER:
            return ExampleKind.SYNTAX
        measurement = candidate_measurement(item.record, measurements)
        selected_wins = sum(
            selected_item.kind == ExampleKind.WIN for selected_item in selected
        )
        if (
            measurement is not None
            and measurement.candidate_label == CandidateLabel.WIN
            and selected_wins < desired_wins
        ):
            return ExampleKind.WIN
        return ExampleKind.SYNTAX

    for item, measurement in wins:
        if sum(item.kind == ExampleKind.WIN for item in selected) >= desired_wins:
            break
        add(item, ExampleKind.WIN, measurement)
    for item in default_samples:
        if (
            sum(item.kind == ExampleKind.KEEP_DEFAULT for item in selected)
            >= desired_defaults
        ):
            break
        record = item.record
        task_measurements = [
            measurement
            for (task_id, _), measurement in measurements.items()
            if task_id == record.task_id and measurement.score_interval is not None
        ]
        if not task_measurements:
            continue
        add(
            item,
            ExampleKind.KEEP_DEFAULT,
            DefaultBestProvenance(
                task_label=TaskLabel.DEFAULT_BEST,
                measured_fingerprint_count=len(task_measurements),
                best_upper_speedup=max(
                    measurement.score_interval.upper
                    for measurement in task_measurements
                    if measurement.score_interval is not None
                ),
            ),
        )
    used = {(item.record.task_id, item.record.sample) for item in selected}
    syntax = [
        item
        for item in records
        if item.record.accepted
        and item.record.syntax_eligible
        and (item.record.task_id, item.record.sample) not in used
        and (
            (measurement := candidate_measurement(item.record, measurements)) is None
            or measurement.candidate_label != CandidateLabel.KNOWN_REGRESSION
        )
    ]
    syntax.sort(
        key=lambda item: (
            item.source == ExampleSource.TEACHER,
            stable_rank(config.seed, "syntax", item),
        )
    )
    for family in sorted(config.gate.required_action_families):
        if family_totals[family]:
            continue
        item = next(
            (
                candidate
                for candidate in syntax
                if candidate.source == ExampleSource.STUDENT
                and family in candidate.record.action_families
                and (candidate.record.task_id, candidate.record.sample) not in used
                and allowed(candidate)
            ),
            None,
        )
        if item is not None:
            kind = measured_kind(item)
            add(
                item,
                kind,
                candidate_measurement(item.record, measurements),
            )
            used.add((item.record.task_id, item.record.sample))

    for family in sorted(config.gate.required_action_families):
        if family_totals[family]:
            continue
        item = next(
            (
                candidate
                for candidate in syntax
                if candidate.source == ExampleSource.TEACHER
                and family in candidate.record.action_families
                and (candidate.record.task_id, candidate.record.sample) not in used
                and allowed(candidate)
            ),
            None,
        )
        if item is not None:
            kind = measured_kind(item)
            add(
                item,
                kind,
                candidate_measurement(item.record, measurements),
            )
            used.add((item.record.task_id, item.record.sample))

    for item in syntax:
        if item.source != ExampleSource.TEACHER:
            continue
        identity = (item.record.task_id, item.record.sample)
        kind = measured_kind(item)
        if identity not in used and add(
            item,
            kind,
            candidate_measurement(item.record, measurements),
        ):
            used.add(identity)

    missing_families = [
        family
        for family in config.gate.required_action_families
        if not family_totals[family]
    ]
    if missing_families:
        names = ", ".join(family.value for family in missing_families)
        raise RuntimeError(f"action families have no eligible example: {names}")

    for item in syntax:
        if len(selected) >= target:
            break
        if item.source != ExampleSource.STUDENT:
            continue
        identity = (item.record.task_id, item.record.sample)
        if identity not in used and add(
            item,
            ExampleKind.SYNTAX,
            candidate_measurement(item.record, measurements),
        ):
            used.add(identity)
    if len(selected) != target:
        raise RuntimeError(
            f"only {len(selected)} of {target} training documents are available"
        )
    return selected


def select_validation(
    records: list[FilterRecord], config: DatasetConfig
) -> list[SelectedExample]:
    eligible: dict[str, list[FilterRecord]] = defaultdict(list)
    for record in records:
        if record.accepted and record.syntax_eligible:
            eligible[record.task_id].append(record)
    selected = [
        min(
            task_records,
            key=lambda record: stable_rank(
                config.seed,
                "validation",
                SourcedRecord(record, ExampleSource.STUDENT),
            ),
        )
        for _, task_records in sorted(eligible.items())
    ]
    selected.sort(
        key=lambda record: stable_rank(
            config.seed,
            "validation",
            SourcedRecord(record, ExampleSource.STUDENT),
        )
    )
    if len(selected) < config.split_counts.validation:
        raise RuntimeError(
            f"only {len(selected)} of {config.split_counts.validation} validation tasks produced a document"
        )
    return [
        SelectedExample(record, ExampleSource.STUDENT, ExampleKind.SYNTAX, None)
        for record in selected[: config.split_counts.validation]
    ]


def write_prime(
    dataset: Path, documents: dict[str, list[DemonstrationDocument]]
) -> dict[str, PrimeArtifact]:
    prime = dataset / "prime"
    prime.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, PrimeArtifact] = {}
    for split, values in documents.items():
        rows: list[JsonObject] = []
        for document in values:
            row = JSON_OBJECT_ADAPTER.validate_python(
                {
                    "messages": document.messages,
                    "tools": json.dumps(document.tools, sort_keys=True),
                }
            )
            rows.append(row)
        encoded = "".join(canonical_json(row) + "\n" for row in rows).encode()
        path = prime / f"{split}.jsonl"
        path.write_bytes(encoded)
        artifacts[split] = PrimeArtifact(
            path=str(path.relative_to(dataset)),
            rows=len(rows),
            bytes=len(encoded),
            sha256=hashlib.sha256(encoded).hexdigest(),
        )
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble the protocol SFT v2 dataset from sampled traces."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/005-protocol-sft-v2/dataset.json"),
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/sft/protocol-sft-v2")
    )
    parser.add_argument(
        "--teacher-config",
        type=Path,
        default=Path("experiments/005-protocol-sft-v2/teacher.json"),
    )
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    dataset = (repository / arguments.dataset).resolve()
    config_path = (repository / arguments.config).resolve()
    config = load_record(config_path, DatasetConfig)
    teacher_config = load_record(
        (repository / arguments.teacher_config).resolve(), TeacherConfig
    )
    selection_path = (repository / config.selection).resolve()
    selection = load_record(selection_path, DatasetSelection)
    task_set = TaskSet.load(repository, "ceb-v1")
    measurements, task_labels = load_measurements(dataset)
    student_records = load_json_lines(
        dataset / "filter/sampling/records.jsonl", FilterRecord
    )
    default_best_filter = dataset / "filter/default_best/records.jsonl"
    if default_best_filter.is_file():
        student_records.extend(load_json_lines(default_best_filter, FilterRecord))
    teacher_records = load_json_lines(
        dataset / "filter/teacher/records.jsonl", FilterRecord
    )
    train_records = [
        *(SourcedRecord(record, ExampleSource.STUDENT) for record in student_records),
        *(SourcedRecord(record, ExampleSource.TEACHER) for record in teacher_records),
    ]
    validation_records = load_json_lines(
        dataset / "filter/validation/records.jsonl", FilterRecord
    )
    sampling_ids = {record.task_id for record in selection.splits.sampling}
    live_gate_ids = {record.task_id for record in selection.splits.live_gate}
    validation_ids = {record.task_id for record in selection.splits.validation}
    if sampling_ids & live_gate_ids:
        raise RuntimeError("sampling and live-gate task selections overlap")
    if any(item.record.task_id not in sampling_ids for item in train_records):
        raise RuntimeError("training filter contains a task outside the sampling split")
    if any(record.task_id not in validation_ids for record in validation_records):
        raise RuntimeError("validation filter contains a task outside its frozen split")
    selections = {
        "train": select_training(
            train_records, measurements, task_labels, config, teacher_config
        ),
        "validation": select_validation(validation_records, config),
    }
    documents: dict[str, list[DemonstrationDocument]] = defaultdict(list)
    ordinal = 0
    for split in ("train", "validation"):
        directory = dataset / "demonstrations" / split
        directory.mkdir(parents=True, exist_ok=True)
        for selected in selections[split]:
            sample = load_filtered_sample(
                repository, dataset, selected.record, selected.source
            )
            document = build_document(repository, task_set, sample, selected, ordinal)
            path = directory / f"{ordinal + 1:04d}-{sample.task_id}.json"
            write_json(path, document.to_wire())
            documents[split].append(document)
            ordinal += 1

    prime_artifacts = write_prime(dataset, documents)
    composition = {
        split: dict(
            sorted(
                Counter(document.metadata.example_kind for document in values).items()
            )
        )
        for split, values in documents.items()
    }
    templates = {
        split: dict(
            sorted(
                Counter(document.metadata.template_id for document in values).items()
            )
        )
        for split, values in documents.items()
    }
    families = Counter(
        family
        for selected in selections["train"]
        if selected.kind != ExampleKind.KEEP_DEFAULT
        for family in selected.record.action_families
    )
    sources = Counter(selected.source for selected in selections["train"])
    frozen_validation_task_ids = [
        document.metadata.task_id for document in documents["validation"]
    ]
    if (
        len(frozen_validation_task_ids) != len(set(frozen_validation_task_ids))
        or set(frozen_validation_task_ids) != validation_ids
    ):
        raise RuntimeError("validation documents do not cover the frozen task cohort")
    identities: list[DemonstrationIdentity] = []
    for split, values in documents.items():
        for document in values:
            validation = JSON_OBJECT_ADAPTER.validate_python(
                validate_protocol_demo(document.to_wire(), repository)
            )
            identities.append(
                DemonstrationIdentity(
                    demonstration_id=document.metadata.demonstration_id,
                    partition=split,
                    task_id=document.metadata.task_id,
                    template_id=document.metadata.template_id,
                    path=f"demonstrations/{split}/{document.metadata.ordinal + 1:04d}-{document.metadata.task_id}.json",
                    canonical_sha256=require_string(
                        validation.get("canonical_sha256"), "canonical_sha256"
                    ),
                )
            )
    manifest = DatasetManifest(
        dataset_id=DATASET_ID,
        seed=config.seed,
        config=FileIdentity(
            path=str(config_path.relative_to(repository)),
            sha256=sha256_file(config_path),
        ),
        selection=DatasetSelectionIdentity(
            path=str(selection_path.relative_to(repository)),
            sha256=sha256_file(selection_path),
            rl_v3_excluded_train_task_count=len(sampling_ids | live_gate_ids),
            sampling_live_gate_disjoint=True,
        ),
        task_set_id=task_set.task_set_id,
        counts={split: len(values) for split, values in documents.items()},
        composition=composition,
        templates=templates,
        train_action_families=dict(sorted(families.items())),
        train_example_sources=dict(sorted(sources.items())),
        prime_artifacts=prime_artifacts,
        inputs=DatasetInputs(
            sampling_manifest_sha256=sha256_file(
                dataset / "sampling/sampling-manifest.json"
            ),
            default_best_sampling_manifest_sha256=sha256_file(
                dataset / "sampling/default-best-manifest.json"
            )
            if (dataset / "sampling/default-best-manifest.json").is_file()
            else None,
            validation_sampling_manifest_sha256=sha256_file(
                dataset / "sampling/validation-manifest.json"
            ),
            sampling_filter_manifest_sha256=sha256_file(
                dataset / "filter/sampling/manifest.json"
            ),
            default_best_filter_manifest_sha256=sha256_file(
                dataset / "filter/default_best/manifest.json"
            )
            if (dataset / "filter/default_best/manifest.json").is_file()
            else None,
            teacher_filter_manifest_sha256=sha256_file(
                dataset / "filter/teacher/manifest.json"
            ),
            validation_filter_manifest_sha256=sha256_file(
                dataset / "filter/validation/manifest.json"
            ),
            measurement_manifest_sha256=sha256_file(dataset / "measurement.json"),
        ),
        frozen_validation_task_ids=frozen_validation_task_ids,
        demonstrations=identities,
    )
    write_json(dataset / "manifest.json", manifest.to_wire())
    print(
        json.dumps(
            {
                "counts": manifest.counts,
                "composition": JSON_OBJECT_ADAPTER.validate_python(
                    manifest.to_wire()["composition"]
                ),
                "train_action_families": JSON_OBJECT_ADAPTER.validate_python(
                    manifest.to_wire()["train_action_families"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


def validate_dataset(repository: Path, dataset: Path) -> JsonObject:
    manifest_path = dataset / "manifest.json"
    manifest = load_record(manifest_path, DatasetManifest)
    if manifest.dataset_id != DATASET_ID:
        raise RuntimeError("unexpected protocol SFT v2 dataset ID")
    counts: Counter[str] = Counter()
    for record in manifest.demonstrations:
        document = load_record(dataset / record.path, DemonstrationDocument)
        validation = JSON_OBJECT_ADAPTER.validate_python(
            validate_protocol_demo(document.to_wire(), repository)
        )
        if validation.get("canonical_sha256") != record.canonical_sha256:
            raise RuntimeError(f"demonstration checksum differs: {record.path}")
        counts[record.partition] += 1
    if dict(counts) != manifest.counts:
        raise RuntimeError("dataset demonstration counts differ")
    for split, artifact in manifest.prime_artifacts.items():
        path = dataset / artifact.path
        if sha256_file(path) != artifact.sha256:
            raise RuntimeError(f"Prime {split} checksum differs")
        if len(path.read_text(encoding="utf-8").splitlines()) != artifact.rows:
            raise RuntimeError(f"Prime {split} row count differs")
    return {
        "dataset_id": DATASET_ID,
        "demonstrations": len(manifest.demonstrations),
        "counts": dict(sorted(counts.items())),
        "manifest_sha256": sha256_file(manifest_path),
    }


if __name__ == "__main__":
    main()
