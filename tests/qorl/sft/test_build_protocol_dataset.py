from __future__ import annotations

import json
from pathlib import Path

from tests.qorl.sft.factories import measurement, messages

from qorl.sft.build_protocol_dataset import (
    SourcedRecord,
    keep_default_messages,
    select_training,
    select_validation,
)
from qorl.sft.schemas import (
    ActionFamily,
    CandidateLabel,
    DatasetConfig,
    ExampleKind,
    ExampleSource,
    FilterRecord,
    TeacherConfig,
    load_record,
)


def filter_record(task_id: str, family: ActionFamily, sample: int = 1) -> FilterRecord:
    return FilterRecord(
        task_id=task_id,
        template_id="template-1",
        sample=sample,
        sample_path=f"{task_id}-{sample}.json",
        accepted=True,
        rejection_reason=None,
        plan_sha256=f"{task_id}-{sample}",
        action_families=[family],
        syntax_eligible=True,
        steered=False,
    )


def test_keep_default_graft_preserves_the_inspection_prefix() -> None:
    original = messages()

    grafted = keep_default_messages(original)

    assert grafted[:4] == original[:4]
    assert len(grafted) == 6
    calls = grafted[-2]["tool_calls"]
    assert isinstance(calls, list)
    assert calls[0]["function"]["name"] == "keep_default"
    content = grafted[-1]["content"]
    assert isinstance(content, str)
    assert json.loads(content)["status"] == "kept_default"


def test_validation_selects_one_document_per_frozen_task() -> None:
    records = [
        FilterRecord(
            task_id=task_id,
            template_id="template-1",
            sample=sample,
            sample_path=f"{task_id}-{sample}.json",
            accepted=True,
            rejection_reason=None,
            plan_sha256=f"{task_id}-{sample}",
            action_families=[ActionFamily.SETTING],
            syntax_eligible=True,
            steered=False,
        )
        for task_id in ("task-1", "task-2")
        for sample in (1, 2)
    ]
    config = load_record(
        Path("experiments/005-protocol-sft-v2/dataset.json"), DatasetConfig
    )
    config = config.model_copy(
        update={
            "split_counts": config.split_counts.model_copy(update={"validation": 2})
        }
    )

    selected = select_validation(records, config)

    assert len(selected) == 2
    assert {item.record.task_id for item in selected} == {"task-1", "task-2"}


def test_training_uses_teacher_for_missing_coverage_then_fills_from_students() -> None:
    config = load_record(
        Path("experiments/005-protocol-sft-v2/dataset.json"), DatasetConfig
    )
    config = config.model_copy(
        update={
            "split_counts": config.split_counts.model_copy(update={"train": 4}),
            "composition": config.composition.model_copy(
                update={"win_share": 0.5, "keep_default_share": 0.0}
            ),
            "gate": config.gate.model_copy(
                update={
                    "required_action_families": [
                        ActionFamily.SETTING,
                        ActionFamily.PARALLEL,
                    ]
                }
            ),
        }
    )
    teacher_config = load_record(
        Path("experiments/005-protocol-sft-v2/teacher.json"), TeacherConfig
    ).model_copy(update={"maximum_teacher_share": 0.25})
    records = [
        *(
            SourcedRecord(
                filter_record(f"student-{index}", ActionFamily.SETTING),
                ExampleSource.STUDENT,
            )
            for index in range(3)
        ),
        SourcedRecord(
            filter_record("teacher-parallel", ActionFamily.PARALLEL),
            ExampleSource.TEACHER,
        ),
        SourcedRecord(
            filter_record("teacher-leading", ActionFamily.LEADING),
            ExampleSource.TEACHER,
        ),
    ]

    teacher_win = measurement("teacher-parallel", "teacher-parallel-1", 1.5).model_copy(
        update={
            "source": ExampleSource.TEACHER,
            "candidate_label": CandidateLabel.WIN,
        }
    )

    selected = select_training(
        records,
        {("teacher-parallel", "teacher-parallel-1"): teacher_win},
        {},
        config,
        teacher_config,
    )

    assert len(selected) == 4
    assert sum(item.source == ExampleSource.TEACHER for item in selected) == 1
    assert any(
        item.source == ExampleSource.TEACHER
        and ActionFamily.PARALLEL in item.record.action_families
        and item.kind == ExampleKind.SYNTAX
        for item in selected
    )
