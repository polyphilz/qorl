from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QoAgentConfig:
    model: str
    revision: str
    vllm_version: str
    use_flashinfer_sampler: bool
    enable_prefix_caching: bool
    base_url: str
    context_length: int
    maximum_model_turns: int
    request_timeout_seconds: int
    seed: int
    sampling: dict[str, int | float]
    thinking: bool
    tool_call_parser: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> QoAgentConfig:
        if value.get("type") != "qo_agent":
            raise ValueError("qo-agent policy type is incorrect")
        return cls(
            model=value["model"],
            revision=value["revision"],
            vllm_version=value["vllm_version"],
            use_flashinfer_sampler=value["use_flashinfer_sampler"],
            enable_prefix_caching=value["enable_prefix_caching"],
            base_url=value["base_url"],
            context_length=value["context_length"],
            maximum_model_turns=value["maximum_model_turns"],
            request_timeout_seconds=value["request_timeout_seconds"],
            seed=value["seed"],
            sampling=value["sampling"],
            thinking=value["thinking"],
            tool_call_parser=value["tool_call_parser"],
        )
