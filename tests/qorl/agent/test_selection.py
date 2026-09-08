"""The same explicit choice works with and without execution feedback."""

import json
import random
from pathlib import Path

import pytest
from pydantic import TypeAdapter
from tests.qorl.agent.test_agent import ScriptedTransport, policy, reply
from tests.qorl.measure.test_feedback import ENABLED, InspectionWorker
from tests.qorl.measure.test_rollout import SETTINGS, TASK, Sql

from qorl.agent.types import InspectionExecutor
from qorl.measure.rollout import RolloutEvaluator
from qorl.measure.schemas import RolloutMeasurementSettings
from qorl.measure.validation import PlanValidationEvaluator
from qorl.model.client import JSON_OBJECT
from qorl.model.schemas import ToolDefinition


@pytest.mark.parametrize("measurement", [None, SETTINGS, ENABLED])
@pytest.mark.parametrize("chosen", [1, 2])
@pytest.mark.parametrize("planning_timeout", [False, True])
def test_selection_is_shared_with_plan_only_and_feedback_disabled(
    repository_root: Path,
    measurement: RolloutMeasurementSettings | None,
    chosen: int,
    planning_timeout: bool,
) -> None:
    worker = InspectionWorker(repository_root)
    if planning_timeout:
        worker.failure = "candidate_plan"
    run: PlanValidationEvaluator[InspectionExecutor]
    if measurement is None:
        run = PlanValidationEvaluator(
            worker, Sql(), TASK, default_timeout_ms=5000, max_candidates=2
        )
    else:
        run = RolloutEvaluator(
            worker, Sql(), TASK, measurement=measurement, max_candidates=2
        )
    run.start()
    transport = ScriptedTransport(
        [
            reply(
                "evaluate_candidate",
                json.dumps(
                    {"action": {"version": 1, "settings": {"seq_page_cost": value}}}
                ),
            )
            for value in (2.0, 3.0)
        ]
        + [
            reply(
                "finish",
                json.dumps({"selected_candidate_id": f"candidate-{chosen:02d}"}),
            )
        ]
    )
    trace = policy(transport, attempts=2, turns=3).search(run)
    assert trace.selection.status == "accepted"
    assert trace.selection.selected_candidate_id == f"candidate-{chosen:02d}"
    assert [item.selection_eligible for item in run.candidates] == [True, True]
    history = JSON_OBJECT.validate_python(
        trace.tool_events[1].result["_candidate_history"]
    )
    assert history["selectable_candidate_ids"] == ["candidate-01", "candidate-02"]
    assert history["attempts_remaining"] == 0
    assert transport.requests[-1]["tools"] == [trace.tools[-1].model_dump(mode="json")]
    if isinstance(run, RolloutEvaluator):
        before = run.execution_counts.final_paired
        final = run.finish(
            random.Random(0),
            selected_candidate_id=trace.selection.selected_candidate_id,
        )
        assert final.selected_candidate_id == f"candidate-{chosen:02d}"
        if planning_timeout and chosen == 1:
            assert final.kind == "timed_out"
            assert run.execution_counts.final_paired == before
        else:
            assert final.kind == "measured" and run.execution_counts.final_paired == 8
    else:
        assert not any(call.analyze for call in worker.calls)
        assert all(item.execution_feedback is None for item in run.candidates)


def test_last_turn_reserves_a_terminal_tool(repository_root: Path) -> None:
    worker = InspectionWorker(repository_root)
    run = PlanValidationEvaluator[InspectionExecutor](
        worker, Sql(), TASK, default_timeout_ms=5000, max_candidates=5
    )
    run.start()
    transport = ScriptedTransport(
        [reply("evaluate_candidate", '{"action":{"version":1}}'), reply("finish")]
    )
    trace = policy(transport, attempts=5, turns=2).search(run)
    tools = TypeAdapter(list[ToolDefinition]).validate_python(
        transport.requests[-1]["tools"]
    )
    assert [tool.function.name for tool in tools] == ["finish"]
    assert len(run.candidates) == 1 and trace.selection.status == "accepted"
