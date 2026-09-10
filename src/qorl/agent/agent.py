"""One provider-independent conversation loop over QORL's existing tools and budgets."""

import hashlib
import json

from pydantic import JsonValue, TypeAdapter

from qorl.agent.feedback import candidate_history
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
from qorl.model.exceptions import ContextBudgetError, ModelResponseError
from qorl.model.schemas import (
    GenerationRequest,
    Message,
    MessageRole,
    TokenUsage,
)
from qorl.util.hashing import sha256_json

SEED_BYTES = 4
JSON_VALUE: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def parse_arguments(raw: str) -> JsonValue:
    try:
        return JSON_VALUE.validate_json(raw)
    except ValueError:
        return raw


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

    def search(
        self,
        evaluator: AgentEvaluator,
        *,
        log_label: str | None = None,
    ) -> AgentTrace:
        """Retain completed turns and tool results even if the rollout is interrupted."""
        self.trace = None
        progress_label = evaluator.task.task_id if log_label is None else log_label
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
            selection=evaluator.selection,
            initial_observation=interface.observation,
            tools=interface.tools,
            tools_sha256=sha256_json(
                [tool.model_dump(mode="json") for tool in interface.tools]
            ),
            transcript=interface.initial_messages(),
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
                tools=interface.available_tools(turn, len(evaluator.candidates)),
                seed=turn_seed(self.seed, evaluator.task.task_id, turn),
            )
            trace.model_requests.append(request.model_copy(deep=True))
            try:
                response = self.client.generate(request)
            except ContextBudgetError as error:
                trace.stop_reason = StopReason.CONTEXT_BUDGET
                trace.prompt_tokens = error.prompt_tokens
                break
            except ModelResponseError as error:
                trace.model_failures.append(error.evidence)
                trace.usage = total_usage(
                    [
                        *(item.usage for item in trace.model_responses),
                        error.evidence.usage,
                    ]
                )
                raise
            trace.model_responses.append(response)
            trace.transcript.append(response.message)
            trace.usage = total_usage([item.usage for item in trace.model_responses])
            trace.prompt_tokens = response.prompt_tokens
            evaluator.check_cancelled()
            if response.truncated:
                for call in response.message.tool_calls or []:
                    if call.function.name == ToolName.FINISH:
                        evaluator.reject_selection(
                            parse_arguments(call.function.arguments),
                            [
                                "arguments: finish response exceeded the output limit; selection was not accepted"
                            ],
                        )
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
                arguments = parse_arguments(call.function.arguments)
                if index != 0:
                    raw_result, finished = {"error": "call one tool at a time"}, False
                elif name not in available_names:
                    guidance = (
                        "tool is not available for this turn; available tools: "
                        + ", ".join(sorted(available_names))
                    )
                    if len(evaluator.candidates) >= evaluator.max_candidates:
                        guidance += (
                            ". No candidate attempts remain; finish is available. "
                            "Choose an eligible issued ID from _candidate_history "
                            'or explicit "default". If no submitted candidate is '
                            "eligible, finish({}) ends with no_valid_candidate."
                        )
                    raw_result, finished = (
                        {"error": guidance},
                        False,
                    )
                else:
                    raw_result, finished = environment.execute(name, arguments)
                if name == ToolName.FINISH and (
                    index != 0 or name not in available_names
                ):
                    evaluator.reject_selection(arguments, [str(raw_result["error"])])
                result = JSON_VALUE.validate_python(raw_result)
                body = JSON_OBJECT.validate_python(
                    {
                        **(result if isinstance(result, dict) else {"result": result}),
                        "_candidate_history": candidate_history(
                            evaluator.candidates,
                            evaluator.default,
                            evaluator.max_candidates - len(evaluator.candidates),
                            evaluator.selection,
                        ).model_dump(mode="json"),
                        TURN_BUDGET_FIELD: interface.budget(turn).model_dump(
                            mode="json"
                        ),
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
                        print(
                            f"[{progress_label}] {body['candidate_id']}: {label}",
                            flush=True,
                        )
                    elif name != ToolName.FINISH:
                        print(f"[{progress_label}] turn-{turn:02d}: {name}", flush=True)
                if finished:
                    terminal_tool = ToolName(name)
                evaluator.check_cancelled()
            if terminal_tool is not None:
                trace.stop_reason = TERMINAL_STOP_REASON[terminal_tool]
                break
        else:
            trace.stop_reason = StopReason.MODEL_TURN_LIMIT
        evaluator.resolve_selection()
        return trace
