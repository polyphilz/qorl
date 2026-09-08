"""Saved output of the calibration noise analysis script."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from qorl.measure.schemas import CalibrationReport


class AnalysisModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class AnalysisSettings(AnalysisModel):
    paired_measurements: int = Field(default=3, ge=1)
    tau: float = Field(default=0.05, ge=0)


class NoopEstimate(AnalysisModel):
    window_count: int
    comparison_count: int
    estimated_noop_error_rate: float
    absolute_log_ratio_p95: float


class TaskAnalysis(AnalysisModel):
    task_id: str
    template_id: str
    source_status: str
    worker_slot: int
    measurement_count: int
    plan_sha256s: list[str]
    warmup_stable: bool | None
    median_execution_time_ms: float | None
    coefficient_of_variation: float | None
    exclusion_reason: (
        Literal[
            "failed_task",
            "nonpositive_execution_time",
            "plan_changed",
            "insufficient_measurements",
        ]
        | None
    )
    estimate: NoopEstimate | None


class NoiseSummary(AnalysisModel):
    task_count: int
    eligible_task_count: int
    excluded_task_count: int
    exclusions_by_reason: dict[str, int]
    mean_estimated_noop_error_rate: float | None
    p90_estimated_noop_error_rate: float | None


class CalibrationAnalysis(AnalysisModel):
    source_directory: str
    source_manifest: CalibrationReport
    summary: NoiseSummary
    tasks: list[TaskAnalysis]


class TaskComparison(AnalysisModel):
    task_id: str
    plan_changed: bool
    primary_error_rate: float
    comparison_error_rate: float
    error_rate_delta_pp: float
    median_latency_ratio: float
    cv_delta_pp: float
    absolute_log_ratio_p95_delta: float


class CalibrationComparison(AnalysisModel):
    primary_only_task_ids: list[str]
    comparison_only_task_ids: list[str]
    common_excluded_task_ids: list[str]
    common_eligible_task_count: int
    same_plan_task_count: int
    mean_error_rate_delta_pp: float | None
    median_error_rate_delta_pp: float | None
    same_plan_mean_error_rate_delta_pp: float | None
    tasks: list[TaskComparison]


class AnalysisReport(AnalysisModel):
    schema_version: Literal[1] = 1
    estimator: Literal["adjacent_windows"] = "adjacent_windows"
    estimator_version: Literal[1] = 1
    settings: AnalysisSettings
    caveats: list[str]
    primary: CalibrationAnalysis
    comparison: CalibrationAnalysis | None = None
    paired_comparison: CalibrationComparison | None = None
