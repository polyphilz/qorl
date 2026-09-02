from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from prime_rl.utils.chat_template import deserialize_tool_calls, normalize_messages
from renderers import Qwen35RendererConfig
from renderers.base import build_training_sample, create_renderer
from transformers import AutoTokenizer


def distribution(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "maximum": ordered[-1],
    }


def fail(message: str) -> None:
    raise RuntimeError(message)


def packed_rows(lengths: list[int], sequence_length: int, seed: int) -> int:
    """Match Prime-RL's seeded shuffle and sequential CatDataset packing."""
    current = 0
    rows = 0
    for index in np.random.default_rng(seed).permutation(len(lengths)):
        length = lengths[int(index)]
        if length > sequence_length:
            fail(f"example has {length} tokens; limit is {sequence_length}")
        if current and current + length > sequence_length:
            rows += 1
            current = 0
        current += length
        if current == sequence_length:
            rows += 1
            current = 0
    return rows + bool(current)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit every rendered protocol-SFT example and loss mask."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=20480)
    parser.add_argument("--seed", type=int, default=20260830)
    arguments = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(arguments.model)
    renderer = create_renderer(
        tokenizer, Qwen35RendererConfig(enable_thinking=False)
    )
    role_tokens: Counter[str] = Counter()
    trainable_role_tokens: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    rendered_lengths: list[int] = []
    trainable_lengths: list[int] = []
    split_lengths: dict[str, list[int]] = {"train": [], "validation": []}
    records: list[dict[str, Any]] = []
    token_hash = hashlib.sha256()
    mask_hash = hashlib.sha256()
    truncated: list[dict[str, Any]] = []
    probability_probe: dict[str, Any] | None = None

    for partition in ("train", "validation"):
        path = arguments.dataset / f"{partition}.jsonl"
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            row = json.loads(line)
            messages = deserialize_tool_calls(
                normalize_messages(row["messages"], default_role="assistant")
            )
            if any(message.get("reasoning_content") for message in messages):
                fail(f"{partition}[{index}] contains reasoning content")
            tools = json.loads(row["tools"])
            rendered = renderer.render(messages, tools=tools)
            sample = build_training_sample(
                renderer, messages, tools=tools, ensure_final_stop=True
            )
            if len(rendered.token_ids) != len(rendered.message_indices):
                fail(f"{partition}[{index}] token attribution mismatch")
            if len(sample.token_ids) != len(sample.loss_mask):
                fail(f"{partition}[{index}] loss-mask length mismatch")

            assistant_indexes = {
                message_index
                for message_index, message in enumerate(messages)
                if message["role"] == "assistant"
            }
            supervised_assistants: set[int] = set()
            for token_index, message_index in enumerate(rendered.message_indices):
                if message_index < 0:
                    continue
                role = messages[message_index]["role"]
                role_tokens[role] += 1
                if sample.loss_mask[token_index]:
                    trainable_role_tokens[role] += 1
                    if role == "assistant":
                        supervised_assistants.add(message_index)
            if supervised_assistants != assistant_indexes:
                fail(f"{partition}[{index}] has an unsupervised assistant turn")

            last_trainable = max(
                position
                for position, enabled in enumerate(sample.loss_mask)
                if enabled
            )
            if sample.token_ids[last_trainable] not in renderer.get_stop_token_ids():
                fail(f"{partition}[{index}] final assistant turn has no stop token")

            if probability_probe is None and partition == "train":
                tool_call_token = tokenizer.convert_tokens_to_ids("<tool_call>")
                first_tool_call = next(
                    position
                    for position, (token, enabled) in enumerate(
                        zip(sample.token_ids, sample.loss_mask, strict=True)
                    )
                    if enabled and token == tool_call_token
                )
                probability_probe = {
                    "description": "first supervised <tool_call> token",
                    "prompt_token_ids": sample.token_ids[:first_tool_call],
                    "target_token": "<tool_call>",
                    "target_token_id": sample.token_ids[first_tool_call],
                }

            after_shift = len(sample.token_ids) - 1
            record = {
                "partition": partition,
                "index": index,
                "tokens_after_shift": after_shift,
                "trainable_tokens": sum(sample.loss_mask),
                "truncated": after_shift > arguments.sequence_length,
            }
            records.append(record)
            if record["truncated"]:
                truncated.append(record)
            split_counts[partition] += 1
            rendered_lengths.append(after_shift)
            split_lengths[partition].append(after_shift)
            trainable_lengths.append(record["trainable_tokens"])
            token_hash.update(
                json.dumps(sample.token_ids, separators=(",", ":")).encode()
            )
            mask_hash.update(bytes(int(value) for value in sample.loss_mask))

    forbidden = {
        role: count
        for role, count in trainable_role_tokens.items()
        if role != "assistant" and count
    }
    if forbidden:
        fail(f"non-assistant tokens contribute to loss: {forbidden}")

    report = {
        "schema_version": 1,
        "renderer": {"name": "qwen3.5", "enable_thinking": False},
        "packed_sequence_length": arguments.sequence_length,
        "seed": arguments.seed,
        "counts": dict(sorted(split_counts.items())),
        "packed_rows": {
            partition: packed_rows(
                lengths, arguments.sequence_length, arguments.seed
            )
            for partition, lengths in split_lengths.items()
        },
        "rendered_tokens_after_shift": distribution(rendered_lengths),
        "trainable_tokens": distribution(trainable_lengths),
        "rendered_tokens_by_role": dict(sorted(role_tokens.items())),
        "trainable_tokens_by_role": dict(sorted(trainable_role_tokens.items())),
        "truncated_examples": len(truncated),
        "token_ids_sha256": token_hash.hexdigest(),
        "loss_masks_sha256": mask_hash.hexdigest(),
        "probability_probe": probability_probe,
        "records": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    hidden = {"probability_probe", "records"}
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in hidden},
            indent=2,
        )
    )
    if truncated:
        fail(f"{len(truncated)} examples exceed the sequence length")


if __name__ == "__main__":
    main()
