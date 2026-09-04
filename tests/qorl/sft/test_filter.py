from __future__ import annotations

from pathlib import Path

from tests.qorl.sft.factories import sample

from qorl.sft.filter import filter_records
from qorl.sft.schemas import JSON_OBJECT_ADAPTER


def test_filter_accepts_one_novel_candidate_and_rejects_repeated_inspection() -> None:
    first = sample()
    repeated = sample(2)
    trace = repeated.policy_trace
    assert trace is not None
    messages = trace["transcript"]
    assert isinstance(messages, list)
    messages[4:4] = [messages[2], messages[3]]

    records = filter_records(
        [(Path("first.json"), first), (Path("repeated.json"), repeated)],
        context_length=20_480,
        syntax_examples_per_task=2,
    )

    assert records[0].accepted is True
    assert records[1].rejection_reason == "repeated_inspection"


def test_filter_rejects_a_tool_error() -> None:
    value = sample()
    trace = value.policy_trace
    assert trace is not None
    trace["tool_events"] = JSON_OBJECT_ADAPTER.validate_python(
        {
            "events": [
                {"name": "get_plan", "result": {"error": "candidate was not issued"}}
            ]
        }
    )["events"]

    records = filter_records(
        [(Path("sample.json"), value)],
        context_length=20_480,
        syntax_examples_per_task=2,
    )

    assert records[0].rejection_reason == "tool_error"


def test_followup_filter_rejects_a_fingerprint_from_the_initial_pass() -> None:
    initial = filter_records(
        [(Path("initial.json"), sample())],
        context_length=20_480,
        syntax_examples_per_task=2,
    )

    followup = filter_records(
        [(Path("followup.json"), sample(7))],
        context_length=20_480,
        syntax_examples_per_task=0,
        existing=initial,
    )

    assert followup[0].rejection_reason == "task_fingerprint_duplicate"
    assert followup[0].syntax_eligible is False
