from __future__ import annotations

import json
from pathlib import Path

from tests.qorl.sft.factories import messages

from qorl.sft.build_protocol_dataset import keep_default_messages, select_validation
from qorl.sft.schemas import (
    ActionFamily,
    DatasetConfig,
    FilterRecord,
    load_record,
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
