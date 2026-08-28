from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from qorl.action import BOOLEAN_SETTINGS, INTEGER_SETTINGS, NUMERIC_SETTINGS
from qorl.agent.client import ModelError, OpenAIModelClient
from qorl.agent.config import QoAgentConfig
from qorl.agent.prompts import SYSTEM_PROMPT
from qorl.agent.tools import AgentEnvironment, agent_tools
from qorl.rollout import MAX_CANDIDATES, RolloutEvaluator


RESERVED_DECISION_TURNS = MAX_CANDIDATES + 1


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
        if model.get("max_model_len") != self.config.context_length:
            raise ModelError(
                "model context length mismatch: "
                f"expected={self.config.context_length} "
                f"actual={model.get('max_model_len')}"
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
        seed_bytes = hashlib.sha256(
            f"{self.config.seed}:{task_id}:{turn}".encode("utf-8")
        ).digest()
        return {
            "model": self.config.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "seed": int.from_bytes(seed_bytes[:4], "big"),
            **self.config.sampling,
            "chat_template_kwargs": {"enable_thinking": self.config.thinking},
        }

    def search(self, evaluator: RolloutEvaluator) -> dict[str, Any]:
        aliases = sorted(evaluator.catalog.relations)
        tools = agent_tools(aliases)
        decision_tools = [
            tool
            for tool in tools
            if tool["function"]["name"] in {"evaluate_candidate", "finish"}
        ]
        inspection_tools = [
            tool for tool in tools if tool["function"]["name"] != "finish"
        ]
        evaluate_tool = [
            tool
            for tool in decision_tools
            if tool["function"]["name"] == "evaluate_candidate"
        ]
        finish_tool = [
            tool for tool in decision_tools if tool["function"]["name"] == "finish"
        ]
        inspection_turn_limit = min(
            len(aliases) * 3,
            max(0, self.config.maximum_model_turns - RESERVED_DECISION_TURNS),
        )
        expected_path = (
            evaluator.worker.fixture.repository
            / "docker/postgres/benchmark-v1.expected.json"
        )
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        configurable_settings = set(BOOLEAN_SETTINGS)
        configurable_settings.update(NUMERIC_SETTINGS, INTEGER_SETTINGS)
        observation = {
            "task_id": evaluator.task["task_id"],
            "objective": "minimize measured warm-cache execution time",
            "sql": evaluator.sql,
            "relations": evaluator.task["relations"],
            "join_edges": evaluator.task["join_edges"],
            "indexes": {
                alias: sorted(indexes)
                for alias, indexes in sorted(evaluator.catalog.indexes.items())
            },
            "postgresql_server_version_num": evaluator.worker.fixture.snapshot[
                "postgresql"
            ]["server_version_num"],
            "planner_settings": {
                name: expected["settings"][name]
                for name in sorted(configurable_settings)
            },
            "default_plan": evaluator.default["compact_plan"],
            "default_median_execution_time_ms": evaluator.default[
                "median_execution_time_ms"
            ],
            "candidate_attempts": MAX_CANDIDATES,
            "candidate_timeout_ms": evaluator.timeout_ms,
            "turn_budget": {
                "total_model_turns": self.config.maximum_model_turns,
                "maximum_inspection_turns": inspection_turn_limit,
                "reserved_final_turns": RESERVED_DECISION_TURNS,
                "reserved_for": {
                    "candidate_evaluations": MAX_CANDIDATES,
                    "finish": 1,
                },
            },
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(observation, sort_keys=True)},
        ]
        responses: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        environment = AgentEnvironment(evaluator)
        stop_reason = "model_turn_limit"

        for turn in range(1, self.config.maximum_model_turns + 1):
            if len(evaluator.candidates) >= MAX_CANDIDATES:
                available_tools = finish_tool
            elif turn > inspection_turn_limit:
                available_tools = (
                    decision_tools if evaluator.candidates else evaluate_tool
                )
            else:
                available_tools = (
                    tools if evaluator.candidates else inspection_tools
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
                budget = {
                    "current_turn": turn,
                    "turns_remaining": self.config.maximum_model_turns - turn,
                    "unrestricted_turns_remaining": max(
                        0, inspection_turn_limit - turn
                    ),
                    "reserved_final_turns": RESERVED_DECISION_TURNS,
                }
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
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(result, sort_keys=True),
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
            if should_finish:
                stop_reason = "model_finish"
                break

        usage: dict[str, int] = {}
        for response in responses:
            for name, value in response.get("usage", {}).items():
                if isinstance(value, int):
                    usage[name] = usage.get(name, 0) + value
        return {
            "stop_reason": stop_reason,
            "initial_observation": observation,
            "tools": tools,
            "tools_sha256": sha256_json(tools),
            "transcript": messages,
            "model_responses": responses,
            "tool_events": events,
            "usage": usage,
        }
