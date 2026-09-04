from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from tests.qorl.sft.factories import baseline, candidate, sample

from qorl.agent.client import ModelError, ModelRequestError
from qorl.agent.types import ToolName
from qorl.sft.schemas import (
    JSON_OBJECT_ADAPTER,
    ActionFamily,
    JsonObject,
    SampleRecord,
    TeacherAttemptStatus,
    TeacherConfig,
    load_record,
)
from qorl.sft.teacher import (
    PrefixCandidate,
    TeacherDecision,
    TeacherTask,
    assistant_tool_name,
    eligible_for_family,
    generate_attempts,
    generation_prompt,
    generation_targets,
    inspection_prefix,
    require_separate_memoize_output,
    response_decision,
    scripted_messages,
    teacher_request,
)


class ScriptedTeacherClient:
    def __init__(self, responses: list[JsonObject | ModelError]) -> None:
        self.responses = responses
        self.requests: list[JsonObject] = []

    def models(self) -> JsonObject:
        raise RuntimeError("not used")

    def version(self) -> JsonObject:
        raise RuntimeError("not used")

    def chat(self, body: JsonObject) -> JsonObject:
        self.requests.append(body)
        response = self.responses.pop(0)
        if isinstance(response, ModelError):
            raise response
        return response


@pytest.fixture
def teacher_config(repository_root: Path) -> TeacherConfig:
    return load_record(
        repository_root / "experiments/005-protocol-sft-v2/teacher.json",
        TeacherConfig,
    )


def prefix() -> PrefixCandidate:
    record = sample()
    messages = inspection_prefix(record)
    assert messages is not None
    trace = record.policy_trace
    assert trace is not None
    record = record.model_copy(
        update={
            "policy_trace": {
                **trace,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": ToolName.GET_PLAN.value,
                            "description": "inspect",
                            "parameters": {"type": "object"},
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": ToolName.EVALUATE_CANDIDATE.value,
                            "description": "evaluate",
                            "parameters": {"type": "object"},
                        },
                    },
                ],
            }
        }
    )
    return PrefixCandidate(
        path=Path("sample.json"),
        sample=record,
        messages=messages,
        assistant_turns=1,
    )


def test_inspection_prefix_stops_before_the_student_decision() -> None:
    record = sample()

    messages = inspection_prefix(record)

    assert messages is not None
    assert len(messages) == 4
    assert assistant_tool_name(messages[-2]) == ToolName.GET_PLAN.value


def test_inspection_prefix_rejects_a_rollout_with_no_inspection() -> None:
    record = sample()
    trace = record.policy_trace
    assert trace is not None
    transcript = record.training_transcript
    assert transcript is not None
    trace = {**trace, "transcript": [*transcript[:2], *transcript[4:]]}

    assert inspection_prefix(record.model_copy(update={"policy_trace": trace})) is None


def test_teacher_request_uses_fable_auto_tool_choice_without_a_seed(
    teacher_config: TeacherConfig,
) -> None:
    request = teacher_request(teacher_config, ActionFamily.LEADING, prefix(), None)

    assert request["model"] == "claude-fable-5-1"
    assert request["max_tokens"] == 16_384
    assert "temperature" not in request
    assert request["tool_choice"] == "auto"
    assert "seed" not in request
    assert len(request["tools"]) == 2
    assert request["messages"][-1]["role"] == "user"
    assert "every query alias exactly once" in generation_prompt(
        ActionFamily.LEADING, None
    )
    prompt = generation_prompt(ActionFamily.JOIN, None)
    assert "action argument as a JSON object" in prompt
    assert "most selective filtered relation" in prompt
    assert "complete leaf-alias set exactly equals that target" in prompt
    assert "never force a nested loop over a large unfiltered relation" in prompt


def test_full_teacher_run_excludes_memoize_but_allows_a_separate_override(
    teacher_config: TeacherConfig,
) -> None:
    assert generation_targets(teacher_config, smoke=False) == {
        ActionFamily.PARALLEL: 4,
        ActionFamily.LEADING: 28,
        ActionFamily.JOIN: 8,
    }
    assert generation_targets(
        teacher_config,
        smoke=False,
        requested=(ActionFamily.MEMOIZE,),
    ) == {ActionFamily.MEMOIZE: 4}


def test_explicit_memoize_run_requires_a_separate_output() -> None:
    default = Path("teacher")

    with pytest.raises(ValueError, match="non-default --output"):
        require_separate_memoize_output(
            default,
            default,
            (ActionFamily.MEMOIZE,),
        )

    require_separate_memoize_output(
        Path("teacher-memoize"),
        default,
        (ActionFamily.MEMOIZE,),
    )


def teacher_response(arguments: str) -> JsonObject:
    return JSON_OBJECT_ADAPTER.validate_python(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "teacher-call",
                                "type": "function",
                                "function": {
                                    "name": ToolName.EVALUATE_CANDIDATE.value,
                                    "arguments": arguments,
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )


def test_response_decision_preserves_raw_tool_arguments() -> None:
    arguments = '{ "action": {"version": 1, "settings": {}}}'

    decision = response_decision(teacher_response(arguments))

    assert decision.action == {"version": 1, "settings": {}}
    calls = decision.assistant_message["tool_calls"]
    assert isinstance(calls, list)
    assert calls[0]["function"]["arguments"] == arguments
    assert decision.assistant_message["content"] == ""


def replayed_sample(decision_action: JsonObject, *, valid: bool) -> SampleRecord:
    replayed_candidate = candidate().model_copy(
        update={
            "action": decision_action,
            "constraints_satisfied": valid,
            "errors_or_diagnostics": []
            if valid
            else ["leading must contain every query relation exactly once"],
        }
    )
    return sample().model_copy(update={"candidates": [replayed_candidate]})


def test_generation_retries_with_validator_feedback_and_accepts(
    teacher_config: TeacherConfig,
) -> None:
    first_arguments = '{"action":{"version":1,"leading":{"left":"a","right":"b"}}}'
    second_arguments = '{"action":{"version":1,"leading":{"left":"a","right":"c"}}}'
    client = ScriptedTeacherClient(
        [teacher_response(first_arguments), teacher_response(second_arguments)]
    )
    replay_count = 0

    def replay(decision: TeacherDecision) -> SampleRecord:
        nonlocal replay_count
        replay_count += 1
        return replayed_sample(decision.action, valid=replay_count == 2)

    result = generate_attempts(
        ActionFamily.LEADING,
        teacher_config,
        prefix(),
        client,
        replay,
        20_480,
        3,
    )

    assert [attempt.status for attempt in result.attempts] == [
        TeacherAttemptStatus.VALIDATION_ERROR,
        TeacherAttemptStatus.ACCEPTED,
    ]
    assert result.accepted is not None
    second_prompt = client.requests[1]["messages"][-1]["content"]
    assert isinstance(second_prompt, str)
    assert "leading must contain every query relation exactly once" in second_prompt


def test_generation_stops_the_task_after_a_replay_error(
    teacher_config: TeacherConfig,
) -> None:
    arguments = '{"action":{"version":1,"leading":{"left":"a","right":"b"}}}'
    client = ScriptedTeacherClient(
        [teacher_response(arguments), teacher_response(arguments)]
    )

    def replay(_decision: TeacherDecision) -> SampleRecord:
        raise RuntimeError("database worker stopped")

    result = generate_attempts(
        ActionFamily.LEADING,
        teacher_config,
        prefix(),
        client,
        replay,
        20_480,
        3,
    )

    assert len(result.attempts) == 1
    assert result.attempts[0].status == TeacherAttemptStatus.REPLAY_ERROR
    assert len(client.requests) == 1


def test_generation_stops_the_run_after_a_fatal_provider_error(
    teacher_config: TeacherConfig,
) -> None:
    client = ScriptedTeacherClient([ModelRequestError("unsupported parameter")])

    def replay(_decision: TeacherDecision) -> SampleRecord:
        raise AssertionError("a rejected request cannot be replayed")

    with pytest.raises(ModelRequestError, match="unsupported parameter"):
        generate_attempts(
            ActionFamily.LEADING,
            teacher_config,
            prefix(),
            client,
            replay,
            20_480,
            3,
        )

    assert len(client.requests) == 1


def test_generation_records_provider_and_response_errors_before_accepting(
    teacher_config: TeacherConfig,
) -> None:
    malformed = JSON_OBJECT_ADAPTER.validate_python(
        {"choices": [{"message": {"role": "assistant", "content": "try a setting"}}]}
    )
    arguments = '{"action":{"version":1,"leading":{"left":"a","right":"b"}}}'
    client = ScriptedTeacherClient(
        [ModelError("temporary outage"), malformed, teacher_response(arguments)]
    )
    pauses: list[float] = []

    def replay(decision: TeacherDecision) -> SampleRecord:
        return replayed_sample(decision.action, valid=True)

    result = generate_attempts(
        ActionFamily.LEADING,
        teacher_config,
        prefix(),
        client,
        replay,
        20_480,
        3,
        pauses.append,
    )

    assert [attempt.status for attempt in result.attempts] == [
        TeacherAttemptStatus.PROVIDER_ERROR,
        TeacherAttemptStatus.RESPONSE_ERROR,
        TeacherAttemptStatus.ACCEPTED,
    ]
    final_prompt = client.requests[-1]["messages"][-1]["content"]
    assert isinstance(final_prompt, str)
    assert "exactly one tool call" in final_prompt
    assert pauses == [5]


def test_generation_rejects_an_action_missing_the_requested_family(
    teacher_config: TeacherConfig,
) -> None:
    arguments = '{"action":{"version":1,"settings":{"enable_hashjoin":false}}}'
    client = ScriptedTeacherClient([teacher_response(arguments)])

    def replay(decision: TeacherDecision) -> SampleRecord:
        return replayed_sample(decision.action, valid=True)

    result = generate_attempts(
        ActionFamily.LEADING,
        teacher_config,
        prefix(),
        client,
        replay,
        20_480,
        1,
    )

    assert result.accepted is None
    assert len(result.attempts) == 1
    assert result.attempts[0].status == TeacherAttemptStatus.VALIDATION_ERROR
    assert result.attempts[0].rejection_reason == "requested_family_missing"


def test_scripted_replay_keeps_the_teacher_decision_turn() -> None:
    arguments = '{ "action": {"version": 1}}'
    decision = response_decision(teacher_response(arguments))

    messages = scripted_messages(prefix(), decision)

    assert messages[-2] == decision.assistant_message
    assert messages[-2]["tool_calls"][0]["function"]["arguments"] == arguments
    assert messages[-1]["content"] == ""


@pytest.mark.parametrize(
    ("family", "node_type", "expected"),
    [
        (ActionFamily.MEMOIZE, "Memoize", True),
        (ActionFamily.MEMOIZE, "Hash Join", False),
        (ActionFamily.PARALLEL, "Gather", True),
        (ActionFamily.PARALLEL, "Gather Merge", True),
        (ActionFamily.PARALLEL, "Hash Join", False),
    ],
)
def test_specialized_teacher_families_require_relevant_default_nodes(
    family: ActionFamily, node_type: str, expected: bool
) -> None:
    candidate_prefix = prefix()
    candidate_prefix = replace(
        candidate_prefix,
        sample=candidate_prefix.sample.model_copy(
            update={
                "default": baseline().model_copy(
                    update={"compact_plan": {"Node Type": node_type}}
                )
            }
        ),
    )
    task = TeacherTask(
        task=JSON_OBJECT_ADAPTER.validate_python(
            {"task_id": "task-1", "template_id": "ceb-3a"}
        ),
        prefix=candidate_prefix,
    )

    assert eligible_for_family(family, task) is expected
