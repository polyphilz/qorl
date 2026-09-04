from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from qorl.db.exceptions import WorkerError
from qorl.db.fixture import DatabaseFixture
from qorl.db.pool import WorkerPool, WorkerSlot
from qorl.measure.rollout import RolloutEvaluator, training_protocol
from qorl.measure.run import TaskRun
from qorl.measure.schemas import MeasurementProtocolId, RunStatus
from qorl.sft.filter import load_filtered_sample
from qorl.sft.schemas import (
    JSON_OBJECT_ADAPTER,
    JSON_OBJECT_LIST_ADAPTER,
    CandidateLabel,
    CandidateMeasurement,
    DatasetConfig,
    ExampleSource,
    FilterRecord,
    JsonObject,
    MeasurementAttempt,
    MeasurementFailure,
    MeasurementManifest,
    MeasurementSummary,
    PipelineError,
    SampleRecord,
    ScoreInterval,
    TaskLabel,
    load_json_lines,
    load_record,
    require_object,
    require_string,
)
from qorl.util.hashing import sha256_file
from qorl.util.io import utc_now, write_json
from qorl.workload.taskset import TaskSet
from qorl.workload.timeouts import CalibratedTimeouts

MEASUREMENT_ID = "qorl-protocol-sft-v2-measurement-v1"
DEFAULT_BEST_TASKS_FILE = "default-best-tasks.json"


@dataclass(frozen=True)
class MeasurementRequest:
    record: FilterRecord
    source: ExampleSource
    sample: SampleRecord
    attempt: int


def measurement_seed(
    dataset_seed: int, task_id: str, fingerprint: str, attempt: int
) -> int:
    digest = hashlib.sha256(
        f"{dataset_seed}:{task_id}:{fingerprint}:measurement:{attempt}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def fingerprint(record: FilterRecord) -> str:
    if record.plan_sha256 is None:
        raise RuntimeError("accepted filter record has no plan fingerprint")
    return record.plan_sha256


def measurement_path(output: Path, record: FilterRecord) -> Path:
    return output / "measurements" / record.task_id / f"{fingerprint(record)}.json"


def measure_once(
    pool: WorkerPool,
    task_set: TaskSet,
    task: JsonObject,
    request: MeasurementRequest,
    timeouts: CalibratedTimeouts,
    dataset_seed: int,
) -> tuple[WorkerSlot, MeasurementAttempt]:
    with pool.claim_worker() as slot:
        evaluator = RolloutEvaluator(
            slot.worker,
            task_set,
            task,
            measurement_protocol=training_protocol(
                MeasurementProtocolId.RL_TRAINING_V2
            ),
            calibrated_timeout=timeouts.task(request.record.task_id),
            timeout_manifest_id=timeouts.manifest["manifest_id"],
            max_candidates=1,
        )
        baseline = evaluator.start()
        if len(request.sample.candidates) != 1:
            raise RuntimeError("accepted sample does not contain exactly one candidate")
        action = request.sample.candidates[0].action
        candidate = evaluator.evaluate(action)
        outcome = evaluator.finish(
            random.Random(
                measurement_seed(
                    dataset_seed,
                    request.record.task_id,
                    fingerprint(request.record),
                    request.attempt,
                )
            )
        )
        return slot, MeasurementAttempt(
            attempt=request.attempt,
            completed_at_utc=utc_now(),
            worker=JSON_OBJECT_ADAPTER.validate_python(slot.resources.manifest()),
            baseline=baseline,
            candidate=candidate,
            outcome=outcome,
        )


def candidate_scores(measurement: CandidateMeasurement) -> tuple[float, float]:
    scores = [attempt.outcome.score for attempt in measurement.attempts]
    if not scores:
        raise RuntimeError("candidate measurement has no completed attempts")
    return min(scores), max(scores)


def needs_band_remeasurement(
    measurement: CandidateMeasurement, config: DatasetConfig
) -> bool:
    if len(measurement.attempts) > 1 or measurement.failed_attempts:
        return False
    score, _ = candidate_scores(measurement)
    return (
        config.labels.remeasure_lower_speedup
        <= score
        <= config.labels.remeasure_upper_speedup
    )


def label_measurements(
    measurements: list[CandidateMeasurement], config: DatasetConfig
) -> tuple[list[CandidateMeasurement], dict[str, TaskLabel]]:
    labeled: list[CandidateMeasurement] = []
    by_task: dict[str, list[CandidateMeasurement]] = defaultdict(list)
    for measurement in measurements:
        lower, upper = candidate_scores(measurement)
        if measurement.failed_attempts:
            label = CandidateLabel.AMBIGUOUS
        elif lower > config.labels.win_speedup:
            label = CandidateLabel.WIN
        elif upper < config.labels.default_best_maximum_speedup:
            label = CandidateLabel.KNOWN_REGRESSION
        else:
            label = CandidateLabel.AMBIGUOUS
        updated = measurement.model_copy(
            update={
                "score_interval": ScoreInterval(lower=lower, upper=upper),
                "candidate_label": label,
            }
        )
        labeled.append(updated)
        by_task[updated.task_id].append(updated)

    task_labels: dict[str, TaskLabel] = {}
    minimum = config.labels.default_best_minimum_fingerprints
    for task_id, records in by_task.items():
        best = max(records, key=lambda record: candidate_scores(record)[1])
        if any(record.candidate_label == CandidateLabel.WIN for record in records):
            task_labels[task_id] = TaskLabel.KNOWN_WIN
        elif len(records) < minimum:
            task_labels[task_id] = TaskLabel.INSUFFICIENT_FINGERPRINTS
        elif best.failed_attempts:
            task_labels[task_id] = TaskLabel.AMBIGUOUS
        elif candidate_scores(best)[1] <= config.labels.default_best_maximum_speedup:
            task_labels[task_id] = TaskLabel.DEFAULT_BEST
        else:
            task_labels[task_id] = TaskLabel.AMBIGUOUS
    return labeled, task_labels


def summarize(
    measurements: list[CandidateMeasurement],
    task_labels: dict[str, TaskLabel],
    failures: list[MeasurementFailure],
) -> MeasurementSummary:
    candidate_labels = Counter(
        record.candidate_label
        for record in measurements
        if record.candidate_label is not None
    )
    task_counts = Counter(task_labels.values())
    return MeasurementSummary(
        measured_candidates=len(measurements),
        remeasured_candidates=sum(len(record.attempts) > 1 for record in measurements),
        failed_attempts=len(failures),
        candidate_labels=dict(sorted(candidate_labels.items())),
        task_labels=dict(sorted(task_counts.items())),
    )


def select_default_best_search_tasks(
    measurements: list[CandidateMeasurement], config: DatasetConfig
) -> list[str]:
    by_task: dict[str, list[CandidateMeasurement]] = defaultdict(list)
    for measurement in measurements:
        if measurement.score_interval is not None:
            by_task[measurement.task_id].append(measurement)
    eligible = [
        task_id
        for task_id, records in by_task.items()
        if len(records)
        >= config.sampling.default_best_search_minimum_measured_fingerprints
        and max(
            record.score_interval.upper
            for record in records
            if record.score_interval is not None
        )
        <= config.sampling.default_best_search_maximum_observed_speedup
    ]
    eligible.sort(
        key=lambda task_id: hashlib.sha256(
            f"{config.seed}:default-best-search:{task_id}".encode()
        ).hexdigest()
    )
    return eligible[: config.sampling.default_best_search_task_limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure accepted protocol SFT v2 candidates."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/005-protocol-sft-v2/dataset.json"),
    )
    parser.add_argument(
        "--timeouts",
        type=Path,
        default=Path("experiments/005-protocol-sft-v2/timeouts.json"),
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("outputs/sft/protocol-sft-v2")
    )
    arguments = parser.parse_args()

    repository = arguments.repository.resolve()
    dataset = (repository / arguments.dataset).resolve()
    config_path = (repository / arguments.config).resolve()
    timeout_path = (repository / arguments.timeouts).resolve()
    config = load_record(config_path, DatasetConfig)
    filter_paths = [
        (ExampleSource.STUDENT, dataset / "filter/sampling/records.jsonl"),
        *(
            [(ExampleSource.STUDENT, dataset / "filter/default_best/records.jsonl")]
            if (dataset / "filter/default_best/records.jsonl").is_file()
            else []
        ),
        (ExampleSource.TEACHER, dataset / "filter/teacher/records.jsonl"),
    ]
    filter_records_sha256 = hashlib.sha256(
        "\n".join(
            f"{source.value}:{path.parent.name}:{sha256_file(path)}"
            for source, path in filter_paths
        ).encode()
    ).hexdigest()
    accepted = [
        (record, source)
        for source, path in filter_paths
        for record in load_json_lines(path, FilterRecord)
        if record.accepted
    ]
    records: list[tuple[FilterRecord, ExampleSource]] = []
    per_task: Counter[str] = Counter()
    for record, source in accepted:
        if per_task[record.task_id] < config.measurement.maximum_candidates_per_task:
            records.append((record, source))
            per_task[record.task_id] += 1

    fixture = DatabaseFixture.load(repository)
    task_set = TaskSet.load(repository, "ceb-v1", fixture.data_identity)
    task_values = JSON_OBJECT_LIST_ADAPTER.validate_python(task_set.inventory["tasks"])
    tasks = {
        require_string(task.get("task_id"), "task.task_id"): task
        for task in task_values
    }
    timeouts = CalibratedTimeouts.load(
        repository, timeout_path, task_set, fixture.runtime_identity
    )
    manifest_path = dataset / "measurement.json"
    previous = (
        load_record(manifest_path, MeasurementManifest)
        if manifest_path.is_file()
        else None
    )
    if previous is not None and previous.dataset_config_sha256 != sha256_file(
        config_path
    ):
        raise RuntimeError("measurement output belongs to different experiment inputs")
    failures = list(previous.failures) if previous is not None else []
    manifest = MeasurementManifest(
        measurement_id=MEASUREMENT_ID,
        status=RunStatus.RUNNING,
        started_at_utc=previous.started_at_utc
        if previous
        else datetime.now(UTC).isoformat(),
        completed_at_utc=None,
        dataset_config_sha256=sha256_file(config_path),
        filter_records_sha256=filter_records_sha256,
        timeouts=JSON_OBJECT_ADAPTER.validate_python(timeouts.identity()),
        database_pool=previous.database_pool if previous else None,
        summary=None,
        task_labels=None,
        failures=failures,
    )
    manifest_wire = manifest.to_wire()
    write_json(manifest_path, manifest_wire)
    run = TaskRun(
        fixture,
        f"qorl-sft-v2-measure-{os.getpid()}",
        dataset,
        manifest_path,
        manifest_wire,
        pool_field="database_pool",
        environment_dir=dataset / "measurement-environment",
    )

    measurements: dict[tuple[str, str], CandidateMeasurement] = {}
    samples: dict[tuple[str, str], SampleRecord] = {}
    sources: dict[tuple[str, str], ExampleSource] = {}
    for record, source in records:
        key = (record.task_id, fingerprint(record))
        sources[key] = source
        samples[key] = load_filtered_sample(repository, dataset, record, source)
        path = measurement_path(dataset, record)
        if path.is_file():
            measurement = load_record(path, CandidateMeasurement)
            if (
                measurement.source != source
                or measurement.sample_path != record.sample_path
            ):
                raise RuntimeError(
                    f"candidate measurement source differs: {record.task_id}"
                )
            measurements[key] = measurement

    def execute(
        pool: WorkerPool, request: MeasurementRequest
    ) -> tuple[WorkerSlot, MeasurementAttempt]:
        return measure_once(
            pool,
            task_set,
            tasks[request.record.task_id],
            request,
            timeouts,
            config.seed,
        )

    def record_failure(request: MeasurementRequest, error: BaseException) -> None:
        failure = MeasurementFailure(
            task_id=request.record.task_id,
            plan_sha256=fingerprint(request.record),
            attempt=request.attempt,
            error=PipelineError(type=type(error).__name__, message=str(error)),
        )
        failures.append(failure)
        key = (request.record.task_id, fingerprint(request.record))
        existing = measurements.get(key)
        if existing is not None:
            updated = existing.model_copy(
                update={"failed_attempts": [*existing.failed_attempts, failure.error]}
            )
            measurements[key] = updated
            write_json(measurement_path(dataset, request.record), updated.to_wire())
        manifest_wire["failures"] = [item.to_wire() for item in failures]
        run.write()
        print(
            f"[measurement failed] {request.record.task_id} attempt={request.attempt}: {error}",
            flush=True,
        )

    with run:
        initial = [
            MeasurementRequest(
                record,
                source,
                samples[(record.task_id, fingerprint(record))],
                1,
            )
            for record, source in records
            if (record.task_id, fingerprint(record)) not in measurements
        ]
        for completion in run.map(
            initial,
            execute,
            concurrency=config.measurement.concurrency,
            handled_errors=(WorkerError,),
        ):
            if completion.error is not None:
                record_failure(completion.item, completion.error)
                continue
            if completion.result is None:
                raise RuntimeError("candidate measurement returned no result")
            _, attempt = completion.result
            request = completion.item
            measurement = CandidateMeasurement(
                task_id=request.record.task_id,
                template_id=request.record.template_id,
                source=request.source,
                plan_sha256=fingerprint(request.record),
                sample_path=request.record.sample_path,
                attempts=[attempt],
                failed_attempts=[],
            )
            measurements[(measurement.task_id, measurement.plan_sha256)] = measurement
            write_json(measurement_path(dataset, request.record), measurement.to_wire())
            print(
                f"[measure {completion.ordinal}/{len(initial)}] {measurement.task_id} {attempt.outcome.score:.3f}x",
                flush=True,
            )

        remeasure_records = [
            record
            for record, _ in records
            if (record.task_id, fingerprint(record)) in measurements
            and needs_band_remeasurement(
                measurements[(record.task_id, fingerprint(record))], config
            )
        ]
        by_task: dict[str, list[FilterRecord]] = defaultdict(list)
        for record, _ in records:
            if (record.task_id, fingerprint(record)) in measurements:
                by_task[record.task_id].append(record)
        already = {
            (record.task_id, fingerprint(record)) for record in remeasure_records
        }
        for task_records in by_task.values():
            if len(task_records) < config.labels.default_best_minimum_fingerprints:
                continue
            best = max(
                task_records,
                key=lambda record: candidate_scores(
                    measurements[(record.task_id, fingerprint(record))]
                )[1],
            )
            key = (best.task_id, fingerprint(best))
            if (
                key not in already
                and len(measurements[key].attempts) == 1
                and not measurements[key].failed_attempts
            ):
                remeasure_records.append(best)
                already.add(key)
        remeasure = [
            MeasurementRequest(
                record,
                sources[(record.task_id, fingerprint(record))],
                samples[(record.task_id, fingerprint(record))],
                2,
            )
            for record in remeasure_records
        ]
        for completion in run.map(
            remeasure,
            execute,
            concurrency=config.measurement.concurrency,
            handled_errors=(WorkerError,),
        ):
            if completion.error is not None:
                record_failure(completion.item, completion.error)
                continue
            if completion.result is None:
                raise RuntimeError("candidate remeasurement returned no result")
            _, attempt = completion.result
            request = completion.item
            key = (request.record.task_id, fingerprint(request.record))
            existing = measurements[key]
            updated = existing.model_copy(
                update={"attempts": [*existing.attempts, attempt]}
            )
            measurements[key] = updated
            write_json(measurement_path(dataset, request.record), updated.to_wire())
            print(
                f"[remeasure {completion.ordinal}/{len(remeasure)}] {updated.task_id} {attempt.outcome.score:.3f}x",
                flush=True,
            )

    values, task_labels = label_measurements(list(measurements.values()), config)
    by_key = {(value.task_id, value.plan_sha256): value for value in values}
    for record, _ in records:
        key = (record.task_id, fingerprint(record))
        if key in by_key:
            write_json(measurement_path(dataset, record), by_key[key].to_wire())
    summary = summarize(values, task_labels, failures)
    write_json(
        dataset / DEFAULT_BEST_TASKS_FILE,
        select_default_best_search_tasks(values, config),
    )
    final = manifest.model_copy(
        update={
            "status": RunStatus.COMPLETED_WITH_FAILURES
            if failures
            else RunStatus.COMPLETED,
            "completed_at_utc": utc_now(),
            "database_pool": require_object(
                manifest_wire.get("database_pool"), "database pool"
            ),
            "summary": summary,
            "task_labels": task_labels,
            "failures": failures,
        }
    )
    write_json(manifest_path, final.to_wire())
    print(json.dumps(summary.to_wire(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
