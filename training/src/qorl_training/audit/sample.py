from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from prime_rl.utils.chat_template import deserialize_tool_calls, normalize_messages
from renderers import Qwen35RendererConfig
from renderers.base import build_training_sample, create_renderer
from transformers import AutoTokenizer


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Prime-RL rendering, supervision, and truncation."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=16384)
    arguments = parser.parse_args()

    row = json.loads((arguments.dataset / "train.jsonl").read_text().splitlines()[0])
    messages = deserialize_tool_calls(
        normalize_messages(row["messages"], default_role="assistant")
    )
    tools = json.loads(row["tools"])
    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    renderer_config = Qwen35RendererConfig(enable_thinking=False)
    renderer = create_renderer(tokenizer, renderer_config)
    rendered = renderer.render(messages, tools=tools)
    sample = build_training_sample(
        renderer,
        messages,
        tools=tools,
        ensure_final_stop=True,
    )

    if len(rendered.token_ids) != len(rendered.message_indices):
        fail("renderer token attribution length mismatch")
    if len(sample.token_ids) != len(sample.loss_mask):
        fail("training token and loss-mask lengths differ")
    if len(sample.token_ids) - 1 > arguments.sequence_length:
        fail(
            "sample would be truncated: "
            f"tokens_after_shift={len(sample.token_ids) - 1} "
            f"limit={arguments.sequence_length}"
        )

    trainable_by_role: Counter[str] = Counter()
    rendered_by_role: Counter[str] = Counter()
    trainable_by_message: Counter[int] = Counter()
    for index, message_index in enumerate(rendered.message_indices):
        if message_index < 0:
            continue
        role = messages[message_index]["role"]
        rendered_by_role[role] += 1
        if sample.loss_mask[index]:
            trainable_by_role[role] += 1
            trainable_by_message[message_index] += 1

    forbidden = {
        role: trainable_by_role[role]
        for role in ("system", "user", "tool")
        if trainable_by_role[role]
    }
    if forbidden:
        fail(f"non-assistant tokens contribute to loss: {forbidden}")
    assistant_indexes = [
        index for index, message in enumerate(messages) if message["role"] == "assistant"
    ]
    if any(message.get("reasoning_content") for message in messages):
        fail("demonstration contains reasoning despite thinking being disabled")
    missing = [index for index in assistant_indexes if not trainable_by_message[index]]
    if missing:
        fail(f"assistant messages have no supervised tokens: {missing}")
    last_trainable = max(
        index for index, enabled in enumerate(sample.loss_mask) if enabled
    )
    if sample.token_ids[last_trainable] not in renderer.get_stop_token_ids():
        fail("final supervised assistant turn has no renderer stop token")

    tool_call_token_id = tokenizer.convert_tokens_to_ids("<tool_call>")
    first_tool_call = next(
        index
        for index, (token_id, enabled) in enumerate(
            zip(sample.token_ids, sample.loss_mask, strict=True)
        )
        if enabled and token_id == tool_call_token_id
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "renderer": {
            "name": "qwen3.5",
            "enable_thinking": False,
        },
        "message_count": len(messages),
        "assistant_message_count": len(assistant_indexes),
        "assistant_reasoning_content_present": False,
        "rendered_tokens": len(rendered.token_ids),
        "training_tokens_before_shift": len(sample.token_ids),
        "training_tokens_after_shift": len(sample.token_ids) - 1,
        "packed_sequence_length": arguments.sequence_length,
        "truncated": False,
        "trainable_tokens": sum(sample.loss_mask),
        "rendered_tokens_by_role": dict(sorted(rendered_by_role.items())),
        "trainable_tokens_by_role": dict(sorted(trainable_by_role.items())),
        "assistant_trainable_tokens": {
            str(index): trainable_by_message[index] for index in assistant_indexes
        },
        "final_stop_token_id": sample.token_ids[last_trainable],
        "token_ids_sha256": hashlib.sha256(
            json.dumps(sample.token_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "loss_mask_sha256": hashlib.sha256(
            bytes(int(value) for value in sample.loss_mask)
        ).hexdigest(),
        "probability_probe": {
            "description": "first supervised <tool_call> token",
            "prompt_token_ids": sample.token_ids[:first_tool_call],
            "target_token": "<tool_call>",
            "target_token_id": sample.token_ids[first_tool_call],
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {
        key: value for key, value in report.items() if key != "probability_probe"
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
