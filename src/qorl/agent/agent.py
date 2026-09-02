from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from qorl.agent.client import ModelError, OpenAIModelClient
from qorl.agent.config import QoAgentConfig
from qorl.agent.protocol import AgentProtocol
from qorl.agent.prompts import SYSTEM_PROMPT
from qorl.agent.tools import AgentEnvironment
from qorl.measure.rollout import RolloutEvaluator


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class QoAgentPolicy:
    def __init__(
        self,
        config: QoAgentConfig,
        client: OpenAIModelClient | None = None,
    ) -> None:
        self.config = config
        self.client = client or OpenAIModelClient(
            config.base_url, config.request_timeout_seconds
        )
        self.server_identity: dict[str, Any] | None = None

    def preflight(self) -> dict[str, Any]:
        response = self.client.models()
        models = [
            item
            for item in response.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        model_ids = sorted(item["id"] for item in models)
        if self.config.model not in model_ids:
            raise ModelError(
                f"model server does not advertise {self.config.model}: {model_ids}"
            )
        model = next(item for item in models if item["id"] == self.config.model)
        effective_context_model = model
        if model.get("max_model_len") is None and model.get("parent"):
            effective_context_model = next(
                (item for item in models if item["id"] == model["parent"]),
                model,
            )
        if (
            effective_context_model.get("max_model_len")
            != self.config.context_length
        ):
            raise ModelError(
                "model context length mismatch: "
                f"expected={self.config.context_length} "
                f"actual={model.get('max_model_len')} "
                f"parent={model.get('parent')} "
                "parent_actual="
                f"{effective_context_model.get('max_model_len')}"
            )
        version = self.client.version()
        if version.get("version") != self.config.vllm_version:
            raise ModelError(
                "vLLM version mismatch: "
                f"expected={self.config.vllm_version} actual={version.get('version')}"
            )
        self.server_identity = {
            "base_url": self.config.base_url,
            "advertised_models": model_ids,
            "model": model,
            "effective_context_model": effective_context_model,
            "vllm": version,
        }
        return self.server_identity

    def manifest(self) -> dict[str, Any]:
        return {
            "type": "qo_agent",
            **asdict(self.config),
            "system_prompt": SYSTEM_PROMPT,
            "system_prompt_sha256": hashlib.sha256(
                SYSTEM_PROMPT.encode()
            ).hexdigest(),
            "server_identity": self.server_identity,
        }

    def request_body(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        task_id: str,
        turn: int,
    ) -> dict[str, Any]:
        body = {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "parallel_tool_calls": False,
            **self.config.sampling,
            "chat_template_kwargs": {"enable_thinking": self.config.thinking},
        }
        if self.config.seed is not None:
            seed_bytes = hashlib.sha256(
                f"{self.config.seed}:{task_id}:{turn}".encode("utf-8")
            ).digest()
            body["seed"] = int.from_bytes(seed_bytes[:4], "big")
        return body

    def search(self, evaluator: RolloutEvaluator) -> dict[str, Any]:
        completion_reserve = max(
            256, int(self.config.sampling.get("max_tokens", 0))
        )
        protocol = AgentProtocol.from_evaluator(
            evaluator,
            self.config.maximum_model_turns,
            self.config.context_length,
            completion_reserve,
        )
        messages = protocol.initial_messages()
        responses: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        environment = AgentEnvironment(evaluator)
        stop_reason = "model_turn_limit"
        context_estimate_tokens: int | None = None

        for turn in range(1, self.config.maximum_model_turns + 1):
            available_tools = protocol.available_tools(
                turn, len(evaluator.candidates)
            )
            response = self.client.chat(
                self.request_body(
                    messages, available_tools, evaluator.task["task_id"], turn
                )
            )
            responses.append(response)
            try:
                raw_message = response["choices"][0]["message"]
                assistant = {
                    key: raw_message[key]
                    for key in ("role", "content", "tool_calls")
                    if key in raw_message
                }
            except (KeyError, IndexError, TypeError) as error:
                raise ModelError(
                    "model server returned an invalid chat response"
                ) from error
            messages.append(assistant)
            calls = assistant.get("tool_calls") or []
            if not calls:
                messages.append(
                    {"role": "user", "content": "Call exactly one available tool."}
                )
                events.append({"turn": turn, "error": "model emitted no tool call"})
                continue

            should_finish = False
            terminal_tool: str | None = None
            pending_tool_tokens = 0
            for index, call in enumerate(calls):
                call_id = call.get("id", f"turn-{turn}-call-{index + 1}")
                name = call.get("function", {}).get("name", "")
                raw_arguments = call.get("function", {}).get("arguments", "{}")
                try:
                    arguments = (
                        json.loads(raw_arguments)
                        if isinstance(raw_arguments, str)
                        else raw_arguments
                    )
                except json.JSONDecodeError:
                    arguments = raw_arguments
                result, finished = (
                    environment.execute(name, arguments)
                    if index == 0
                    else ({"error": "call one tool at a time"}, False)
                )
                budget = protocol.budget(turn)
                result = (
                    {**result, "_turn_budget": budget}
                    if isinstance(result, dict)
                    else {"result": result, "_turn_budget": budget}
                )
                if index == 0:
                    if name == "evaluate_candidate" and "candidate_id" in result:
                        label = (
                            f"{result['provisional_speedup']:.3f}x"
                            if result["constraints_satisfied"]
                            else "invalid"
                        )
                        print(f"  {result['candidate_id']}: {label}", flush=True)
                    elif name != "finish":
                        print(f"  turn-{turn:02d}: {name}", flush=True)
                content = json.dumps(result, sort_keys=True)
                # A byte-fallback tokenizer cannot produce more tokens than
                # input bytes. Include a small allowance for message framing.
                pending_tool_tokens += len(content.encode("utf-8")) + 64
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": content,
                    }
                )
                events.append(
                    {
                        "turn": turn,
                        "tool_call_id": call_id,
                        "name": name,
                        "result": result,
                    }
                )
                should_finish = should_finish or finished
                if finished:
                    terminal_tool = name
            if should_finish:
                stop_reason = f"model_{terminal_tool}"
                break
            response_usage = response.get("usage", {})
            total_tokens = response_usage.get("total_tokens")
            if not isinstance(total_tokens, int):
                prompt_tokens = response_usage.get("prompt_tokens")
                completion_tokens = response_usage.get("completion_tokens")
                if isinstance(prompt_tokens, int) and isinstance(
                    completion_tokens, int
                ):
                    total_tokens = prompt_tokens + completion_tokens
            if isinstance(total_tokens, int):
                context_estimate_tokens = total_tokens + pending_tool_tokens
                if (
                    context_estimate_tokens + completion_reserve
                    >= self.config.context_length
                ):
                    stop_reason = "context_budget"
                    break
            else:
                # Continuing without server-reported usage would make the next
                # request's fit unknowable. End this rollout cleanly instead.
                stop_reason = "missing_token_usage"
                break

        usage: dict[str, int] = {}
        for response in responses:
            for name, value in response.get("usage", {}).items():
                if isinstance(value, int):
                    usage[name] = usage.get(name, 0) + value
        return {
            "stop_reason": stop_reason,
            "initial_observation": protocol.observation,
            "tools": protocol.tools,
            "tools_sha256": sha256_json(protocol.tools),
            "transcript": messages,
            "model_responses": responses,
            "tool_events": events,
            "usage": usage,
            "context_estimate_tokens": context_estimate_tokens,
        }
