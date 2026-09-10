"""Run final selection through the real shared agent, evaluator and native harness."""

import asyncio
import json
import re
import subprocess
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest
import verifiers.v1 as vf
from prime_rl.orchestrator.algo.qorl_anchored_grpo import decision_from_final
from tests.qorl.agent.test_agent import reply
from tests.qorl.evaluation.conftest import EvaluationActivity
from tests.qorl.evaluation.test_evaluate import run_evaluation, saved_rollouts

from qorl.agent.schemas import AgentTrace
from qorl.evaluation.evaluate import summarize_performance
from qorl.experiment.schemas import EvaluationExperimentConfig
from qorl.model.client import JSON_OBJECT, HttpTransport
from qorl.model.schemas import JsonObject, JsonValue
from qorl.postgres.client import PostgresClient
from qorl.postgres.config import PostgresConfig
from qorl.rl import runtime as shared_runtime
from qorl.rl.harness import QorlHarness
from qorl.rl.reward import scalar_reward, training_speedup
from qorl.rl.runtime import QorlRuntime
from qorl.rl.schemas import QorlHarnessConfig, QorlTaskData, ScalarRewardSettings
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool.schemas import PoolConfig

REJECTED_FINISHES: dict[str, JsonValue] = {
    "rejected_unknown": {"selected_candidate_id": "invented"},
    "rejected_null": {"selected_candidate_id": None},
    "rejected_number": {"selected_candidate_id": 1},
    "rejected_empty": {"selected_candidate_id": ""},
    "rejected_extra": {"selected_candidate_id": "candidate-01", "oops": True},
    "rejected_primitive": 42,
    "rejected_envelope": None,
    "truncated_unknown": {"selected_candidate_id": "invented"},
    "truncated_null": {"selected_candidate_id": None},
    "truncated_primitive": 42,
    "truncated_valid": {"selected_candidate_id": "candidate-01"},
}


@pytest.mark.parametrize("runner", ["evaluation", "native"])
@pytest.mark.parametrize(
    "scenario",
    [
        "planning_timeout",
        "execution_timeout",
        "all_timeouts",
        "planning_success",
        "execution_success",
        "earlier",
        "middle",
        "slower",
        "five_third",
        "five_slower",
        "omitted",
        "all_rejected",
        "ineligible",
        "corrected",
        "forced_multiple",
        "forced_sole",
        "forced_none",
        *REJECTED_FINISHES,
        "rejected_context",
        "rejected_output",
        "rejected_turns",
        "ignored_finish",
        "ignored_get_plan",
        "default_valid",
        "default_invalid",
        "default_timeout",
        "default_corrected",
        *[f"none_{name}" for name in REJECTED_FINISHES if name.startswith("rejected_")],
    ],
)
def test_selected_outcome_through_active_consumers(
    runner: str,
    scenario: str,
    tmp_path: Path,
    repository_root: Path,
    activity: EvaluationActivity,
    evaluation_config: EvaluationExperimentConfig,
    pool_config: PoolConfig,
    postgres_config: PostgresConfig,
    benchmark_task_sets: dict[str, TaskSet],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = (
        5
        if scenario.startswith("five_")
        else 3
        if scenario in {"earlier", "middle", "slower"}
        else 1
        if scenario.startswith(("rejected_", "ignored_", "truncated_"))
        or scenario
        in {"omitted", "all_rejected", "corrected", "forced_sole", "forced_none"}
        else 2
    )
    actions: list[JsonObject] = [
        {"version": 1, "settings": {"seq_page_cost": float(index + 2)}}
        for index in range(count)
    ]
    if scenario in {"all_rejected", "default_invalid"} or scenario.startswith("none_"):
        actions = [{"version": 2} for _ in range(count)]
    if scenario == "ineligible":
        actions[0] = {"version": 2}
    if scenario.startswith("five_"):
        actions[1] = {"version": 2}
    if scenario == "forced_none":
        actions = []
    responses = [
        reply("evaluate_candidate", json.dumps({"action": action}))
        for action in actions
    ]
    chosen = (
        5
        if scenario == "five_slower"
        else 3
        if scenario == "five_third"
        else 2
        if scenario in {"planning_success", "execution_success", "middle"}
        else 3
        if scenario == "slower"
        else 1
    )
    if scenario in {
        "corrected",
        "rejected_context",
        "rejected_output",
        "rejected_turns",
        "default_corrected",
    }:
        responses.append(reply("finish", '{"selected_candidate_id":"invented"}'))
    responses.append(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "out of time"},
                    "finish_reason": "stop"
                    if scenario == "rejected_turns"
                    else "length",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        if scenario.startswith("forced_")
        or scenario in {"rejected_context", "rejected_output", "rejected_turns"}
        else reply(
            "finish",
            json.dumps({"selected_candidate_id": "default"})
            if scenario.startswith("default_")
            else json.dumps(REJECTED_FINISHES[scenario.removeprefix("none_")])
            if scenario.startswith("none_")
            else "{}"
            if scenario in {"omitted", "all_rejected", "corrected"}
            else json.dumps(
                REJECTED_FINISHES.get(
                    scenario, {"selected_candidate_id": f"candidate-{chosen:02d}"}
                )
            ),
        )
    )
    if scenario.startswith("truncated_"):
        choices = responses[-1]["choices"]
        assert isinstance(choices, list) and isinstance(choices[0], dict)
        choices[0]["finish_reason"] = "length"
    if scenario.startswith("ignored_"):
        response = reply("finish", "{}")
        choices = response["choices"]
        assert isinstance(choices, list)
        choice = JSON_OBJECT.validate_python(choices[0])
        message = JSON_OBJECT.validate_python(choice["message"])
        calls = message["tool_calls"]
        assert isinstance(calls, list)
        calls.append(
            {
                "id": "ignored",
                "type": "function",
                "function": {
                    "name": scenario.removeprefix("ignored_"),
                    "arguments": "{}",
                },
            }
        )
        choice["message"] = message
        response["choices"] = [choice]
        responses[-1] = response
    agent = evaluation_config.agent.model_copy(
        update={"candidate_attempts": count, "maximum_model_turns": len(responses)}
    )
    config = evaluation_config.model_copy(update={"agent": agent})
    original_command = activity.command
    feedback_calls: dict[int, int] = {}
    final_hints: list[int] = []

    def command(command: list[str], query: str) -> subprocess.CompletedProcess[str]:
        cost = re.search(r"Set\(seq_page_cost (\d+(?:\.\d+)?)\)", query)
        candidate = int(float(cost[1])) - 1 if cost else None
        analyze = "ANALYZE" in query
        if candidate is not None and analyze:
            feedback_calls[candidate] = feedback_calls.get(candidate, 0) + 1
            if not responses:
                final_hints.append(candidate)
        timed_out = candidate is not None and (
            scenario in {"all_timeouts", "default_timeout"}
            or (
                candidate == 1
                and (
                    scenario in {"planning_timeout", "planning_success"}
                    or (
                        scenario in {"execution_timeout", "execution_success"}
                        and analyze
                        and feedback_calls[candidate] == 2
                    )
                )
            )
        )
        if timed_out:
            activity.queries.append(query)
            return subprocess.CompletedProcess(
                command, 1, "", "canceling statement due to statement timeout"
            )
        result = original_command(command, query)
        if candidate is not None:
            document: list[JsonObject] = [
                {
                    "Plan": {
                        "Node Type": "Materialize",
                        "Plans": [{"Node Type": "Result", "Plan Rows": 1}],
                    },
                    "Execution Time": float(candidate),
                    "Planning Time": 1.0,
                }
            ]
            return subprocess.CompletedProcess(
                command, 0, json.dumps(document), result.stderr
            )
        return result

    def request(
        transport: HttpTransport, path: str, body: JsonObject | None = None
    ) -> JsonObject:
        assert body is not None
        activity.requests.append(body)
        if path == "../tokenize":
            if scenario == "rejected_context" and len(responses) == 1:
                responses.clear()
                return {
                    "count": config.model.context_length + 1,
                    "max_model_len": config.model.context_length,
                }
            return {"count": 10, "max_model_len": config.model.context_length}
        assert path == "chat/completions"
        return responses.pop(0)

    monkeypatch.setattr(activity, "command", command)
    monkeypatch.setattr(HttpTransport, "request", request)
    if runner == "evaluation":
        run_evaluation(
            tmp_path / "evaluation",
            config,
            pool_config,
            postgres_config,
            benchmark_task_sets["job"],
        )
        result = saved_rollouts(tmp_path / "evaluation")[0]
        record, trace = result.rollout, result.trace
        assert trace is not None
    else:
        defaults = tomllib.loads(
            (repository_root / "configs/defaults/000-rl.toml").read_text()
        )
        harness_config = QorlHarnessConfig.model_validate(
            {
                key: defaults[key]
                for key in ("agent", "measurement", "rl", "model", "inference")
            }
        ).model_copy(update={"agent": agent})
        runtime = QorlRuntime(
            benchmark_task_sets["job"], pool_config, "selection", postgres_config
        )
        worker = runtime.workers[0]
        worker.client = PostgresClient(command, runtime.settings, runtime.indexes)
        monkeypatch.setattr(shared_runtime, "current", Mock(return_value=runtime))
        native_trace = Mock(spec=vf.Trace, id="selection", info={})
        task = runtime.task_set.tasks[0]
        result = asyncio.run(
            QorlHarness(harness_config).launch(
                Mock(spec=vf.ModelContext, model="training-model"),
                native_trace,
                Mock(),
                "http://proxy.test/v1",
                "secret",
                {},
                QorlTaskData(task_id=task.task_id, template_id=task.template_id),
            )
        )
        assert result.exit_code == 0
        # The native record adds resource identity; the shared record owns outcomes.
        from qorl.rl.schemas import RlRolloutRecord

        record = RlRolloutRecord.model_validate_json(
            json.dumps(native_trace.info["qorl"])
        )
        trace = AgentTrace.model_validate(native_trace.info["qorl_policy"])
    assert not responses
    assert record.failure is None and record.final is not None
    assert record.selection == trace.selection
    assert record.selection is not None
    if scenario.startswith("default_"):
        assert record.final.kind == "kept_default"
        assert record.final.speedup == 1.0
        assert record.final.selected_candidate_id is None
        assert record.default is not None
        assert record.final.selected_plan_sha256 == record.default.plan_sha256
        assert record.final.timing_reuse_key == record.default.timing_reuse_key
        assert len(record.candidates) == count
        assert not final_hints
        assert record.execution_counts is not None
        assert record.execution_counts.final_paired == 0
        assert (
            scalar_reward(
                record,
                ScalarRewardSettings(
                    invalid_attempt_penalty=0.1,
                    duplicate_attempt_penalty=0.02,
                    timeout_attempt_penalty=0.1,
                    no_valid_candidate_reward=-0.7,
                ),
            )
            == 0.0
        )
        assert decision_from_final(record.final.to_wire()).kind == "keep_default"
        assert training_speedup(record) == 1.0
    elif scenario.startswith("none_"):
        assert record.final.kind == "no_valid_candidate"
        assert not final_hints
        assert record.selection.status == "accepted"
        assert (
            record.selection.rejections[-1].arguments
            == REJECTED_FINISHES[scenario.removeprefix("none_")]
        )
    elif scenario in {"planning_timeout", "execution_timeout", "all_timeouts"}:
        assert record.final.kind == "timed_out" and record.final.speedup is None
        assert record.final.timeout_ms == record.candidates[0].timeout_ms
        assert record.final.selected_candidate_id == "candidate-01"
        assert (
            not final_hints
            and record.execution_counts is not None
            and record.execution_counts.final_paired == 0
        )
        if scenario == "execution_timeout":
            feedback = record.candidates[0].execution_feedback
            assert feedback is not None and len(feedback.warmups) == 1
    elif scenario in {"ineligible", "forced_multiple"} or scenario.startswith(
        ("rejected_", "truncated_")
    ):
        assert record.final.kind == "selection_failed"
        assert not final_hints
        assert summarize_performance([record]).selection_failure_count == 1
        assert training_speedup(record) is None
        rewards = ScalarRewardSettings(
            invalid_attempt_penalty=0.1,
            duplicate_attempt_penalty=0.02,
            timeout_attempt_penalty=0.1,
            no_valid_candidate_reward=-0.7,
        )
        assert scalar_reward(record, rewards) == -0.7
    elif scenario in {"all_rejected", "forced_none"}:
        assert record.final.kind == "no_valid_candidate" and not final_hints
    else:
        assert record.final.kind == "measured"
        assert record.final.selected_candidate_id == f"candidate-{chosen:02d}"
        assert record.final.speedup == 100.0 / chosen
        assert final_hints == [chosen] * 4
        if scenario.startswith("five_"):
            assert len(record.candidates) == 5
            assert not record.candidates[1].selection_eligible
            assert record.execution_counts is not None
            assert record.execution_counts.initial_default == (
                record.measurement.default_warmups
                + record.measurement.default_measurements
            )
            assert record.execution_counts.candidate_feedback == 8
            assert record.execution_counts.final_paired == 8
            performance = summarize_performance([record])
            assert performance.selected_candidate_positions == {chosen: 1}
            assert performance.earlier_candidate_selection_count == (chosen == 3)
    if scenario in {"corrected", "default_corrected"}:
        assert (
            trace.selection.status == "accepted"
            and len(trace.selection.rejections) == 1
        )
    if scenario.startswith("ignored_"):
        assert trace.selection.status == "accepted"
        assert trace.stop_reason == "model_finish"
        assert trace.tool_events[-1].result["error"] == "call one tool at a time"
    if scenario in REJECTED_FINISHES:
        assert trace.selection.rejections[-1].arguments == REJECTED_FINISHES[scenario]
        assert trace.selection.rejections[-1].diagnostics
    if scenario.startswith("truncated_"):
        assert trace.stop_reason == "model_output_limit"
        assert len(trace.tool_events) == len(actions)  # No truncated tool was executed.
    if actions:
        history = JSON_OBJECT.validate_python(
            trace.tool_events[len(actions) - 1].result["_candidate_history"]
        )
        assert history["selectable_candidate_ids"] == [
            item.candidate_id for item in record.candidates if item.selection_eligible
        ]
    assert type(record).model_validate_json(record.model_dump_json()) == record
