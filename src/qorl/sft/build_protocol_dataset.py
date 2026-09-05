from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from qorl.sft.assemble import canonical_json
from qorl.sft.filter import load_filtered_sample
from qorl.sft.schemas import (
    JSON_OBJECT_ADAPTER,
    JSON_OBJECT_LIST_ADAPTER,
    CandidateEvidence,
    DatasetConfig,
    DatasetInputs,
    DatasetManifest,
    DatasetSelection,
    DatasetSelectionIdentity,
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
    PrimeArtifact,
    SampleRecord,
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
STUDENT_TEACHER_ID = "iterated_rejection_sampling_v1"
FABLE_TEACHER_ID = "protocol-sft-v2-fable-5-1"
MEASUREMENT_MODE = "rejection_sampling_plan_validation"


@dataclass(frozen=True)
class SelectedExample:
    record: FilterRecord
    source: ExampleSource
    kind: ExampleKind


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


def require_positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


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
    tasks = JSON_OBJECT_LIST_ADAPTER.validate_python(task_set.inventory["tasks"])
    task = next(item for item in tasks if item.get("task_id") == sample.task_id)
    candidates: dict[str, CandidateEvidence] = {}
    if selected.kind == ExampleKind.SYNTAX:
        if len(sample.candidates) != 1:
            raise RuntimeError("selected plan sample has no single candidate")
        candidate = sample.candidates[0]
        if candidate.plain_explain is None:
            raise RuntimeError("selected candidate has no plan")
        candidates[candidate.candidate_id] = CandidateEvidence(
            action=JSON_OBJECT_ADAPTER.validate_python(candidate.action),
            plain_explain=JSON_OBJECT_ADAPTER.validate_python(candidate.plain_explain),
            pg_hint_plan=candidate.pg_hint_plan,
        )
    elif sample.candidates:
        raise RuntimeError("selected keep-default sample submitted a candidate")
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
            else STUDENT_TEACHER_ID,
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
            candidate_count=1 if selected.kind == ExampleKind.SYNTAX else 0,
            measurement_mode=MEASUREMENT_MODE,
            selection_used_speed=False,
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


def select_training(
    records: list[SourcedRecord], config: DatasetConfig
) -> list[SelectedExample]:
    keep_default_target = config.assembly.keep_default_examples
    plan_target = config.split_counts.train - keep_default_target
    per_task: Counter[str] = Counter()
    selected: list[SelectedExample] = []

    plan_records = sorted(
        (
            item
            for item in records
            if item.record.accepted and item.record.syntax_eligible
        ),
        key=lambda item: (
            item.source == ExampleSource.TEACHER,
            item.record.task_id,
            item.record.sample,
        ),
    )
    for item in plan_records:
        if per_task[item.record.task_id] >= config.assembly.maximum_examples_per_task:
            continue
        selected.append(SelectedExample(item.record, item.source, ExampleKind.SYNTAX))
        per_task[item.record.task_id] += 1
    if len(selected) != plan_target:
        raise RuntimeError(
            f"expected {plan_target} novel-plan documents, found {len(selected)}"
        )

    keep_default_records = sorted(
        (
            item
            for item in records
            if item.source == ExampleSource.STUDENT
            and item.record.rejection_reason == "keep_default"
        ),
        key=lambda item: stable_rank(config.seed, "keep_default", item),
    )
    kept = 0
    for item in keep_default_records:
        if kept >= keep_default_target:
            break
        if per_task[item.record.task_id] >= config.assembly.maximum_examples_per_task:
            continue
        selected.append(
            SelectedExample(item.record, item.source, ExampleKind.KEEP_DEFAULT)
        )
        per_task[item.record.task_id] += 1
        kept += 1
    if kept != keep_default_target:
        raise RuntimeError(
            f"expected {keep_default_target} keep-default documents, found {kept}"
        )
    if len(selected) != config.split_counts.train:
        raise RuntimeError(
            f"expected {config.split_counts.train} training documents, found {len(selected)}"
        )
    return selected


def write_prime(dataset: Path, documents: list[DemonstrationDocument]) -> PrimeArtifact:
    prime = dataset / "prime"
    prime.mkdir(parents=True, exist_ok=True)
    rows = [
        JSON_OBJECT_ADAPTER.validate_python(
            {
                "messages": document.messages,
                "tools": json.dumps(document.tools, sort_keys=True),
            }
        )
        for document in documents
    ]
    encoded = "".join(canonical_json(row) + "\n" for row in rows).encode()
    path = prime / "train.jsonl"
    path.write_bytes(encoded)
    return PrimeArtifact(
        path=str(path.relative_to(dataset)),
        rows=len(rows),
        bytes=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble the language-only protocol SFT v2 dataset."
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
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    dataset = (repository / arguments.dataset).resolve()
    config_path = (repository / arguments.config).resolve()
    config = load_record(config_path, DatasetConfig)
    selection_path = (repository / config.selection).resolve()
    selection = load_record(selection_path, DatasetSelection)
    task_set = TaskSet.load(repository, "ceb")

    student_records = load_json_lines(
        dataset / "filter/sampling/records.jsonl", FilterRecord
    )
    teacher_records = load_json_lines(
        dataset / "filter/teacher/records.jsonl", FilterRecord
    )
    records = [
        *(SourcedRecord(record, ExampleSource.STUDENT) for record in student_records),
        *(SourcedRecord(record, ExampleSource.TEACHER) for record in teacher_records),
    ]
    sampling_ids = {record.task_id for record in selection.splits.sampling}
    live_gate_ids = {record.task_id for record in selection.splits.live_gate}
    if sampling_ids & live_gate_ids:
        raise RuntimeError("sampling and live-gate task selections overlap")
    if any(item.record.task_id not in sampling_ids for item in records):
        raise RuntimeError("training filter contains a task outside the sampling split")

    selected = select_training(records, config)
    documents: list[DemonstrationDocument] = []
    identities: list[DemonstrationIdentity] = []
    directory = dataset / "demonstrations/train"
    directory.mkdir(parents=True, exist_ok=True)
    for ordinal, example in enumerate(selected):
        sample = load_filtered_sample(
            repository, dataset, example.record, example.source
        )
        document = build_document(repository, task_set, sample, example, ordinal)
        relative = Path(f"demonstrations/train/{ordinal + 1:04d}-{sample.task_id}.json")
        write_json(dataset / relative, document.to_wire())
        validation = JSON_OBJECT_ADAPTER.validate_python(
            validate_protocol_demo(document.to_wire(), repository)
        )
        identities.append(
            DemonstrationIdentity(
                demonstration_id=document.metadata.demonstration_id,
                partition="train",
                task_id=document.metadata.task_id,
                template_id=document.metadata.template_id,
                path=relative.as_posix(),
                canonical_sha256=require_string(
                    validation.get("canonical_sha256"), "canonical_sha256"
                ),
            )
        )
        documents.append(document)

    prime_artifact = write_prime(dataset, documents)
    families = Counter(
        family
        for example in selected
        if example.kind == ExampleKind.SYNTAX
        for family in example.record.action_families
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
        counts={"train": len(documents)},
        composition={
            "train": dict(
                sorted(
                    Counter(
                        document.metadata.example_kind for document in documents
                    ).items()
                )
            )
        },
        templates={
            "train": dict(
                sorted(
                    Counter(
                        document.metadata.template_id for document in documents
                    ).items()
                )
            )
        },
        train_action_families=dict(sorted(families.items())),
        train_example_sources=dict(
            sorted(Counter(example.source for example in selected).items())
        ),
        prime_artifacts={"train": prime_artifact},
        inputs=DatasetInputs(
            sampling_manifest_sha256=sha256_file(
                dataset / "sampling/sampling-manifest.json"
            ),
            sampling_filter_manifest_sha256=sha256_file(
                dataset / "filter/sampling/manifest.json"
            ),
            teacher_filter_manifest_sha256=sha256_file(
                dataset / "filter/teacher/manifest.json"
            ),
        ),
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
                "train_example_sources": JSON_OBJECT_ADAPTER.validate_python(
                    manifest.to_wire()["train_example_sources"]
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
