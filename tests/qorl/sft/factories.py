from __future__ import annotations

import json

from qorl.measure.schemas import (
    Baseline,
    Candidate,
    FinalStatus,
    MeasurementProtocolId,
    MeasurementStatus,
    Outcome,
    RunStatus,
)
from qorl.sft.schemas import (
    JSON_OBJECT_ADAPTER,
    CandidateMeasurement,
    ExampleSource,
    JsonObject,
    MeasurementAttempt,
    SampleRecord,
    SamplerIdentity,
    SamplingMode,
)


def assistant(name: str, arguments: JsonObject) -> JsonObject:
    return JSON_OBJECT_ADAPTER.validate_python(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call-{name}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, sort_keys=True),
                    },
                }
            ],
        }
    )


def messages() -> list[JsonObject]:
    values = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "observation"},
        assistant("get_plan", {"candidate_id": "default"}),
        {
            "role": "tool",
            "tool_call_id": "call-get_plan",
            "name": "get_plan",
            "content": "{}",
        },
        assistant("evaluate_candidate", {"action": {"version": 1}}),
        {
            "role": "tool",
            "tool_call_id": "call-evaluate_candidate",
            "name": "evaluate_candidate",
            "content": json.dumps(
                {
                    "candidate_id": "candidate-01",
                    "_turn_budget": {
                        "current_turn": 2,
                        "total_model_turns": 64,
                    },
                }
            ),
        },
        assistant("finish", {}),
        {
            "role": "tool",
            "tool_call_id": "call-finish",
            "name": "finish",
            "content": "{}",
        },
    ]
    return [JSON_OBJECT_ADAPTER.validate_python(value) for value in values]


def candidate(plan_sha256: str = "novel-plan") -> Candidate:
    return Candidate(
        candidate_id="candidate-01",
        action={"version": 1, "settings": {"enable_hashjoin": False}},
        action_valid=True,
        constraints_satisfied=True,
        compiled_hint="Set(enable_hashjoin off)",
        duplicate_of=None,
        plan_sha256=plan_sha256,
        plain_explain={"Plan": {"Node Type": "Seq Scan"}},
        compact_plan={"Node Type": "Seq Scan"},
        provisional_measurements=[],
        provisional_speedup=None,
        measurement_status=MeasurementStatus.NOT_MEASURED,
        errors_or_diagnostics=[],
        pg_hint_plan={},
        attempts_remaining=0,
    )


def baseline() -> Baseline:
    return Baseline(
        plan_sha256="default-plan",
        plain_explain={"Plan": {"Node Type": "Seq Scan"}},
        median_execution_time_ms=None,
        compact_plan={"Node Type": "Seq Scan"},
    )


def sample(sample_number: int = 1) -> SampleRecord:
    transcript = messages()
    trace = JSON_OBJECT_ADAPTER.validate_python(
        {
            "stop_reason": "model_finish",
            "transcript": transcript,
            "model_responses": [{"usage": {"total_tokens": 100}}],
            "tool_events": [],
            "tools": [],
            "initial_observation": {"turn_budget": {"total_model_turns": 64}},
        }
    )
    return SampleRecord(
        status=RunStatus.COMPLETED,
        completed_at_utc="2026-09-03T00:00:00+00:00",
        task_id="task-1",
        template_id="template-1",
        sample=sample_number,
        seed=sample_number,
        sampling_mode=SamplingMode.NORMAL,
        steered=False,
        guidance=None,
        worker={},
        data_identity={},
        runtime_identity={},
        sampler=SamplerIdentity(
            model="teacher", manifest_sha256="sha", server_identity={}
        ),
        default=baseline(),
        candidates=[candidate()],
        policy_trace=trace,
        training_transcript=transcript,
        error=None,
    )


def measurement(task_id: str, plan_sha256: str, *scores: float) -> CandidateMeasurement:
    attempts = [
        MeasurementAttempt(
            attempt=index,
            completed_at_utc="2026-09-03T00:00:00+00:00",
            worker={},
            baseline=baseline(),
            candidate=candidate(plan_sha256),
            outcome=Outcome(
                measurement_protocol_id=MeasurementProtocolId.RL_TRAINING_V2,
                status=FinalStatus.COMPLETED,
                winning_candidate_id="candidate-01",
                score=score,
                trajectory_reward=0.0,
                invalid_attempt_count=0,
                duplicate_attempt_count=0,
                timeout_attempt_count=0,
            ),
        )
        for index, score in enumerate(scores, start=1)
    ]
    return CandidateMeasurement(
        task_id=task_id,
        template_id="template-1",
        source=ExampleSource.STUDENT,
        plan_sha256=plan_sha256,
        sample_path=f"samples/{task_id}.json",
        attempts=attempts,
        failed_attempts=[],
    )
