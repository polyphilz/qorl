"""Estimate no-op timing errors from saved calibration trials; never execute SQL."""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
from collections import Counter
from pathlib import Path

from pydantic import TypeAdapter

from qorl.measure.schemas import (
    CalibratedTask,
    CalibrationReport,
    CalibrationTaskResult,
    RunStatus,
)
from scripts.analysis.schemas import (
    AnalysisReport,
    AnalysisSettings,
    CalibrationAnalysis,
    CalibrationComparison,
    NoiseSummary,
    NoopEstimate,
    TaskAnalysis,
    TaskComparison,
)

TASK_RECORD: TypeAdapter[CalibrationTaskResult] = TypeAdapter(CalibrationTaskResult)
MAX_COMPARISONS_PER_TASK = 1_000_000
CAVEATS = [
    "Estimates use repeated default-plan executions, not live candidate rollouts "
    "or final training advantages. Candidate noise and ML-process load can differ.",
    "Overlapping windows and label assignments reuse the same observations. "
    "Comparison count is not an independent sample size or evidence of precision.",
    "Pair-label symmetry is imposed by this estimator; it does not establish "
    "that live measurement errors are unbiased.",
    "Rates are threshold crossings of raw log ratios, before native speedup "
    "sharing, clipping, peer references, and protocol penalties.",
    "Task summaries weight eligible queries equally. They describe that task "
    "set, not necessarily the query distribution in a training run.",
]


def percentile(values: list[float], fraction: float) -> float:
    """Linear interpolation at (n - 1) * fraction, including singleton samples."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    left = math.floor(index)
    weight = index - left
    return ordered[left] * (1 - weight) + ordered[math.ceil(index)] * weight


def estimate_noop_error(times: list[float], settings: AnalysisSettings) -> NoopEstimate:
    """Enumerate consecutive 2k-trial windows and both labels of each pair."""
    k = settings.paired_measurements
    windows = len(times) - 2 * k + 1
    if windows < 1:
        raise ValueError(f"need at least {2 * k} measured trials")
    if any(not math.isfinite(value) or value <= 0 for value in times):
        raise ValueError("execution times must be finite and strictly positive")
    if (
        k >= MAX_COMPARISONS_PER_TASK.bit_length()
        or windows * (1 << k) > MAX_COMPARISONS_PER_TASK
    ):
        raise ValueError(
            f"exact enumeration exceeds {MAX_COMPARISONS_PER_TASK:,} comparisons "
            "per task; use fewer pairs or fewer calibration trials"
        )
    errors: list[float] = []
    for start in range(windows):
        pairs = [times[start + 2 * i : start + 2 * i + 2] for i in range(k)]
        for labels in itertools.product((0, 1), repeat=k):
            default = [pair[label] for pair, label in zip(pairs, labels, strict=True)]
            candidate = [
                pair[1 - label] for pair, label in zip(pairs, labels, strict=True)
            ]
            errors.append(
                abs(
                    math.log(statistics.median(default))
                    - math.log(statistics.median(candidate))
                )
            )
    return NoopEstimate(
        window_count=windows,
        comparison_count=len(errors),
        estimated_noop_error_rate=sum(error > settings.tau for error in errors)
        / len(errors),
        absolute_log_ratio_p95=percentile(errors, 0.95),
    )


def analyze_task(
    task: CalibrationTaskResult, settings: AnalysisSettings
) -> TaskAnalysis:
    times = [item.execution_time_ms for item in task.measurements]
    plans = sorted({item.plan_sha256 for item in task.measurements})
    completed = isinstance(task, CalibratedTask)
    positive = bool(times) and all(value > 0 for value in times)
    if not completed:
        reason = "failed_task"
    elif not positive:
        reason = "nonpositive_execution_time"
    elif len(plans) != 1:
        reason = "plan_changed"
    elif len(times) < 2 * settings.paired_measurements:
        reason = "insufficient_measurements"
    else:
        reason = None
    return TaskAnalysis(
        task_id=task.task_id,
        template_id=task.template_id,
        source_status=task.status.value,
        worker_slot=task.worker.slot,
        measurement_count=len(times),
        plan_sha256s=plans,
        warmup_stable=task.summary.warmup_stable if completed else None,
        median_execution_time_ms=statistics.median(times) if positive else None,
        coefficient_of_variation=(
            statistics.stdev(times) / statistics.mean(times)
            if positive and len(times) > 1
            else None
        ),
        exclusion_reason=reason,
        estimate=estimate_noop_error(times, settings) if reason is None else None,
    )


def analyze_calibration(
    directory: Path, settings: AnalysisSettings
) -> CalibrationAnalysis:
    directory = directory.resolve()
    manifest_path = directory / "calibration.json"
    try:
        manifest = CalibrationReport.model_validate_json(manifest_path.read_bytes())
    except ValueError as error:
        raise ValueError(f"{manifest_path}: {error}") from error
    if manifest.status not in (RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_FAILURES):
        raise ValueError(
            f"{directory}: calibration must have finished (got {manifest.status})"
        )
    records: dict[str, CalibrationTaskResult] = {}
    for path in sorted((directory / "tasks").glob("*.json")):
        try:
            task = TASK_RECORD.validate_json(path.read_bytes())
            if task.task_id in records:
                raise ValueError(f"duplicate task ID {task.task_id}")
            if path.stem != task.task_id:
                raise ValueError("filename must match task_id")
            runs = [item.run for item in task.measurements]
            if runs != list(range(1, len(runs) + 1)):
                raise ValueError(
                    "measured run numbers must be consecutive and ordered from 1"
                )
            if isinstance(task, CalibratedTask) and (
                task.summary.measurement_count != len(runs)
                or len(runs) != manifest.measurement.num_trials
            ):
                raise ValueError(
                    "measured trial count disagrees with task summary or manifest"
                )
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
        records[task.task_id] = task
    completed = sum(isinstance(task, CalibratedTask) for task in records.values())
    if (
        len(records) != manifest.task_count
        or completed != manifest.completed_task_count
        or len(records) - completed != manifest.failed_task_count
    ):
        raise ValueError(f"{directory}: task files disagree with manifest task counts")
    tasks = [analyze_task(records[task_id], settings) for task_id in sorted(records)]
    rates = [
        task.estimate.estimated_noop_error_rate
        for task in tasks
        if task.estimate is not None
    ]
    reasons = Counter(task.exclusion_reason for task in tasks if task.exclusion_reason)
    return CalibrationAnalysis(
        source_directory=str(directory),
        source_manifest=manifest,
        tasks=tasks,
        summary=NoiseSummary(
            task_count=len(tasks),
            eligible_task_count=len(rates),
            excluded_task_count=len(tasks) - len(rates),
            exclusions_by_reason=dict(sorted(reasons.items())),
            mean_estimated_noop_error_rate=statistics.mean(rates) if rates else None,
            p90_estimated_noop_error_rate=percentile(rates, 0.9) if rates else None,
        ),
    )


def compare_calibrations(
    primary: CalibrationAnalysis, comparison: CalibrationAnalysis
) -> CalibrationComparison:
    if primary.source_manifest.benchmark_id != comparison.source_manifest.benchmark_id:
        raise ValueError("comparisons require the same benchmark")
    if (
        primary.source_manifest.plan_fingerprint_version
        != comparison.source_manifest.plan_fingerprint_version
    ):
        raise ValueError("comparisons require the same plan fingerprint version")
    left = {task.task_id: task for task in primary.tasks}
    right = {task.task_id: task for task in comparison.tasks}
    rows: list[TaskComparison] = []
    excluded: list[str] = []
    for task_id in sorted(left.keys() & right.keys()):
        a, b = left[task_id], right[task_id]
        if a.estimate is None or b.estimate is None:
            excluded.append(task_id)
            continue
        assert (
            a.median_execution_time_ms is not None
            and b.median_execution_time_ms is not None
        )
        assert (
            a.coefficient_of_variation is not None
            and b.coefficient_of_variation is not None
        )
        rows.append(
            TaskComparison(
                task_id=task_id,
                plan_changed=a.plan_sha256s != b.plan_sha256s,
                primary_error_rate=a.estimate.estimated_noop_error_rate,
                comparison_error_rate=b.estimate.estimated_noop_error_rate,
                error_rate_delta_pp=100
                * (
                    b.estimate.estimated_noop_error_rate
                    - a.estimate.estimated_noop_error_rate
                ),
                median_latency_ratio=b.median_execution_time_ms
                / a.median_execution_time_ms,
                cv_delta_pp=100
                * (b.coefficient_of_variation - a.coefficient_of_variation),
                absolute_log_ratio_p95_delta=(
                    b.estimate.absolute_log_ratio_p95
                    - a.estimate.absolute_log_ratio_p95
                ),
            )
        )
    deltas = [row.error_rate_delta_pp for row in rows]
    same_plan_deltas = [row.error_rate_delta_pp for row in rows if not row.plan_changed]
    return CalibrationComparison(
        primary_only_task_ids=sorted(left.keys() - right.keys()),
        comparison_only_task_ids=sorted(right.keys() - left.keys()),
        common_excluded_task_ids=excluded,
        common_eligible_task_count=len(rows),
        same_plan_task_count=len(same_plan_deltas),
        mean_error_rate_delta_pp=statistics.mean(deltas) if deltas else None,
        median_error_rate_delta_pp=statistics.median(deltas) if deltas else None,
        same_plan_mean_error_rate_delta_pp=(
            statistics.mean(same_plan_deltas) if same_plan_deltas else None
        ),
        tasks=rows,
    )


def number(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.2f}%" if percent else f"{value:.4g}"


def render_report(report: AnalysisReport) -> str:
    lines = [
        "# Calibration noise analysis",
        "",
        f"Estimator: `{report.estimator}` v{report.estimator_version}; "
        f"pairs: {report.settings.paired_measurements}; tau: {report.settings.tau}.",
        "",
        "## Interpretation",
        "",
        *[f"- {caveat}" for caveat in report.caveats],
    ]
    for label, analysis in (
        ("Primary", report.primary),
        ("Comparison", report.comparison),
    ):
        if analysis is None:
            continue
        summary = analysis.summary
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                f"Source: `{analysis.source_directory}`",
                "",
                f"Eligible tasks: {summary.eligible_task_count}/{summary.task_count}; "
                f"excluded: {summary.excluded_task_count}.",
                f"Mean estimated no-op error rate: {number(summary.mean_estimated_noop_error_rate, percent=True)}; "
                f"p90 across tasks: {number(summary.p90_estimated_noop_error_rate, percent=True)}.",
                "",
                "Tasks sorted by estimated error rate, highest first. Counts exclude warmups.",
                "",
                "| Task | Worker | Trials | Median ms | CV | Estimated no-op error | p95 abs(log ratio) | Plan count | Exclusion |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        ordered = sorted(
            analysis.tasks,
            key=lambda task: (
                -(task.estimate.estimated_noop_error_rate if task.estimate else -1),
                task.task_id,
            ),
        )
        for task in ordered:
            estimate = task.estimate
            lines.append(
                f"| {task.task_id} | {task.worker_slot} | {task.measurement_count} | "
                f"{number(task.median_execution_time_ms)} | {number(task.coefficient_of_variation, percent=True)} | "
                f"{number(estimate.estimated_noop_error_rate if estimate else None, percent=True)} | "
                f"{number(estimate.absolute_log_ratio_p95 if estimate else None)} | "
                f"{len(task.plan_sha256s)} | {task.exclusion_reason or '—'} |"
            )
    paired = report.paired_comparison
    if paired is not None:
        lines.extend(
            [
                "",
                "## Paired comparison",
                "",
                "All deltas are comparison minus primary; negative error deltas and latency ratios below 1 improve.",
                f"Common eligible tasks: {paired.common_eligible_task_count}; same plan: {paired.same_plan_task_count}.",
                f"Mean error change: {number(paired.mean_error_rate_delta_pp)} percentage points; "
                f"median: {number(paired.median_error_rate_delta_pp)}; "
                f"same-plan mean: {number(paired.same_plan_mean_error_rate_delta_pp)}.",
                f"Primary-only IDs: {', '.join(paired.primary_only_task_ids) or 'none'}.",
                f"Comparison-only IDs: {', '.join(paired.comparison_only_task_ids) or 'none'}.",
                f"Common excluded IDs: {', '.join(paired.common_excluded_task_ids) or 'none'}.",
                "Changed-plan rows include a plan-choice change; they do not isolate execution noise at a fixed plan.",
                "",
                "| Task | Primary error | Comparison error | Change (pp) | Latency ratio | Plan changed |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in paired.tasks:
            lines.append(
                f"| {row.task_id} | {number(row.primary_error_rate, percent=True)} | "
                f"{number(row.comparison_error_rate, percent=True)} | {number(row.error_rate_delta_pp)} | "
                f"{number(row.median_latency_ratio)} | {'yes' if row.plan_changed else 'no'} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "calibration_directory", type=Path, help="contains calibration.json and tasks/"
    )
    parser.add_argument(
        "--compare", type=Path, help="optional second calibration directory"
    )
    parser.add_argument("--paired-measurements", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="writes noise.json and noise.md"
    )
    args = parser.parse_args(argv)
    try:
        settings = AnalysisSettings(
            paired_measurements=args.paired_measurements, tau=args.tau
        )
        primary = analyze_calibration(args.calibration_directory, settings)
        comparison = (
            analyze_calibration(args.compare, settings) if args.compare else None
        )
        report = AnalysisReport(
            settings=settings,
            caveats=CAVEATS,
            primary=primary,
            comparison=comparison,
            paired_comparison=compare_calibrations(primary, comparison)
            if comparison
            else None,
        )
        output = args.output_dir.resolve()
        for source in (primary, comparison):
            if source and output.is_relative_to(
                Path(source.source_directory) / "tasks"
            ):
                raise ValueError(
                    "output directory must not be inside source task records"
                )
        rendered = render_report(report)
        output.mkdir(parents=True, exist_ok=True)
        (output / "noise.json").write_text(report.model_dump_json(indent=2) + "\n")
        (output / "noise.md").write_text(rendered)
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(rendered, end="")
    print(f"\nReports: {output / 'noise.json'} and {output / 'noise.md'}")


if __name__ == "__main__":
    main()
