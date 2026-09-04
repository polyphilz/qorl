from pathlib import Path

from qorl.sft.build_protocol_dataset import SourcedRecord, select_training
from qorl.sft.schemas import (
    ActionFamily,
    DatasetConfig,
    ExampleKind,
    ExampleSource,
    FilterRecord,
    load_record,
)


def filter_record(
    task_id: str,
    family: ActionFamily,
    sample: int = 1,
    *,
    keep_default: bool = False,
) -> FilterRecord:
    return FilterRecord(
        task_id=task_id,
        template_id="template-1",
        sample=sample,
        sample_path=f"{task_id}-{sample}.json",
        accepted=not keep_default,
        rejection_reason="keep_default" if keep_default else None,
        plan_sha256=None if keep_default else f"{task_id}-{sample}",
        action_families=[] if keep_default else [family],
        syntax_eligible=not keep_default,
        steered=False,
    )


def test_training_uses_all_novel_plans_and_natural_keep_default_examples() -> None:
    config = load_record(
        Path("experiments/005-protocol-sft-v2/dataset.json"), DatasetConfig
    )
    config = config.model_copy(
        update={
            "split_counts": config.split_counts.model_copy(update={"train": 4}),
            "assembly": config.assembly.model_copy(update={"keep_default_examples": 1}),
        }
    )
    records = [
        SourcedRecord(
            filter_record("student-1", ActionFamily.SETTING),
            ExampleSource.STUDENT,
        ),
        SourcedRecord(
            filter_record("student-2", ActionFamily.SCAN), ExampleSource.STUDENT
        ),
        SourcedRecord(
            filter_record("teacher-1", ActionFamily.LEADING), ExampleSource.TEACHER
        ),
        SourcedRecord(
            filter_record("default-1", ActionFamily.SETTING, keep_default=True),
            ExampleSource.STUDENT,
        ),
    ]

    selected = select_training(records, config)

    assert len(selected) == 4
    assert [item.kind for item in selected].count(ExampleKind.SYNTAX) == 3
    assert [item.kind for item in selected].count(ExampleKind.KEEP_DEFAULT) == 1
    assert [item.source for item in selected].count(ExampleSource.TEACHER) == 1


def test_training_enforces_the_per_query_cap() -> None:
    config = load_record(
        Path("experiments/005-protocol-sft-v2/dataset.json"), DatasetConfig
    )
    config = config.model_copy(
        update={
            "split_counts": config.split_counts.model_copy(update={"train": 3}),
            "assembly": config.assembly.model_copy(update={"keep_default_examples": 1}),
        }
    )
    records = [
        SourcedRecord(
            filter_record("task-1", ActionFamily.SETTING, sample),
            ExampleSource.STUDENT,
        )
        for sample in (1, 2, 3)
    ]
    records.extend(
        [
            SourcedRecord(
                filter_record("task-1", ActionFamily.SETTING, 4, keep_default=True),
                ExampleSource.STUDENT,
            ),
            SourcedRecord(
                filter_record("task-2", ActionFamily.SETTING, 1, keep_default=True),
                ExampleSource.STUDENT,
            ),
        ]
    )

    selected = select_training(records, config)

    assert [item.record.task_id for item in selected].count("task-1") == 2
    assert any(item.record.task_id == "task-2" for item in selected)
