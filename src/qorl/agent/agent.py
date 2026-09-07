"""One provider-independent conversation loop over QORL's existing tools and budgets."""

import hashlib
import json

from pydantic import JsonValue, TypeAdapter

from qorl.agent.interface import AGENT_INTERFACE_VERSION, AgentInterface
from qorl.agent.schemas import AgentSettings, AgentTrace, ToolEvent
from qorl.agent.tool_runtime import AgentEnvironment
from qorl.agent.types import (
    TERMINAL_STOP_REASON,
    TURN_BUDGET_FIELD,
    AgentEvaluator,
    StopReason,
    ToolName,
)
from qorl.model.client import JSON_OBJECT, ModelClient
from qorl.model.exceptions import ContextBudgetError
from qorl.model.schemas import (
    GenerationRequest,
    Message,
    MessageRole,
    TokenUsage,
    ToolDefinition,
)
from qorl.util.hashing import sha256_json

SEED_BYTES = 4
JSON_VALUE: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def turn_seed(seed: int | None, task_id: str, turn: int) -> int | None:
    """Derive the same turn seed across experiments without using names or timestamps."""
    if seed is None:
        return None
    digest = hashlib.sha256(f"{seed}:{task_id}:{turn}".encode()).digest()
    return int.from_bytes(digest[:SEED_BYTES], "big")


def total_usage(responses: list[TokenUsage]) -> TokenUsage:
    """An aggregate count is unknown if any constituent count is unknown."""

    def total(values: list[int | None]) -> int | None:
        return (
            None
            if any(value is None for value in values)
            else sum(value for value in values if value is not None)
        )

    return TokenUsage(
        prompt_tokens=total([usage.prompt_tokens for usage in responses]),
        completion_tokens=total([usage.completion_tokens for usage in responses]),
        reasoning_tokens=total([usage.reasoning_tokens for usage in responses]),
        cached_tokens=total([usage.cached_tokens for usage in responses]),
    )


class QoAgentPolicy:
    """Apply agent rules to a supplied model client; serving and identity live outside."""

    def __init__(
        self,
        client: ModelClient,
        settings: AgentSettings,
        *,
        context_length: int,
        max_tokens: int,
        seed: int | None,
    ) -> None:
        if not 0 < max_tokens <= context_length:
            raise ValueError("completion allowance must fit the context length")
        self.client = client
        self.settings = settings
        self.context_length = context_length
        self.max_tokens = max_tokens
        self.seed = seed
        self.trace: AgentTrace | None = None

    def search(self, evaluator: AgentEvaluator) -> AgentTrace:
        """Retain completed turns and tool results even if the rollout is interrupted."""
        self.trace = None
        if self.settings.candidate_attempts != evaluator.max_candidates:
            raise ValueError("agent and evaluator candidate limits must agree")
        interface = AgentInterface.from_evaluator(
            evaluator,
            self.settings.maximum_model_turns,
            self.context_length,
            self.max_tokens,
            inspection_turns_per_alias=self.settings.inspection_turns_per_alias,
        )
        trace = AgentTrace(
            agent_interface_version=AGENT_INTERFACE_VERSION,
            seed=self.seed,
            initial_observation=JSON_OBJECT.validate_python(interface.observation),
            tools=[ToolDefinition.model_validate(tool) for tool in interface.tools],
            tools_sha256=sha256_json(interface.tools),
            transcript=[
                Message.model_validate(message)
                for message in interface.initial_messages()
            ],
        )
        self.trace = trace
        environment = AgentEnvironment(evaluator)
        for turn in range(1, self.settings.maximum_model_turns + 1):
            evaluator.check_cancelled()
            available_names = interface.available_tool_names(
                turn, len(evaluator.candidates)
            )
            request = GenerationRequest(
                messages=list(trace.transcript),
                tools=[
                    tool
                    for tool in trace.tools
                    if tool.function.name in available_names
                ],
                seed=turn_seed(self.seed, evaluator.task.task_id, turn),
            )
            try:
                response = self.client.generate(request)
            except ContextBudgetError as error:
                trace.stop_reason = StopReason.CONTEXT_BUDGET
                trace.prompt_tokens = error.prompt_tokens
                break
            trace.model_responses.append(response)
            trace.transcript.append(response.message)
            trace.usage = total_usage([item.usage for item in trace.model_responses])
            trace.prompt_tokens = response.prompt_tokens
            evaluator.check_cancelled()
            if response.truncated:
                trace.stop_reason = StopReason.MODEL_OUTPUT_LIMIT
                break
            calls = response.message.tool_calls or []
            if not calls:
                trace.transcript.append(
                    Message(
                        role=MessageRole.USER,
                        content="Call exactly one available tool.",
                    )
                )
                continue
            terminal_tool: ToolName | None = None
            for index, call in enumerate(calls):
                evaluator.check_cancelled()
                name = call.function.name
                try:
                    arguments = JSON_VALUE.validate_json(call.function.arguments)
                except ValueError:
                    arguments = call.function.arguments
                if index != 0:
                    raw_result, finished = {"error": "call one tool at a time"}, False
                elif name not in available_names:
                    raw_result, finished = (
                        {"error": "tool is not available for this turn"},
                        False,
                    )
                else:
                    raw_result, finished = environment.execute(name, arguments)
                result = JSON_VALUE.validate_python(raw_result)
                body = JSON_OBJECT.validate_python(
                    {
                        **(result if isinstance(result, dict) else {"result": result}),
                        TURN_BUDGET_FIELD: interface.budget(turn),
                    }
                )
                trace.transcript.append(
                    Message(
                        role=MessageRole.TOOL,
                        tool_call_id=call.id,
                        name=name,
                        content=json.dumps(body, sort_keys=True),
                    )
                )
                trace.tool_events.append(
                    ToolEvent(turn=turn, tool_call_id=call.id, name=name, result=body)
                )
                if index == 0:
                    if name == ToolName.EVALUATE_CANDIDATE and "candidate_id" in body:
                        label = (
                            "validated" if body["constraints_satisfied"] else "invalid"
                        )
                        print(f"  {body['candidate_id']}: {label}", flush=True)
                    elif name != ToolName.FINISH:
                        print(f"  turn-{turn:02d}: {name}", flush=True)
                if finished:
                    terminal_tool = ToolName(name)
                evaluator.check_cancelled()
            if terminal_tool is not None:
                trace.stop_reason = TERMINAL_STOP_REASON[terminal_tool]
                break
        else:
            trace.stop_reason = StopReason.MODEL_TURN_LIMIT
        return trace
