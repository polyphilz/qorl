"""Conversation budgets, independent of the model provider."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from qorl.agent.observation import AgentObservation
from qorl.agent.types import StopReason
from qorl.model.schemas import (
    GenerationResponse,
    JsonObject,
    Message,
    TokenUsage,
    ToolDefinition,
)


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_attempts: int = Field(ge=1)
    maximum_model_turns: int = Field(ge=1)
    inspection_turns_per_alias: int = Field(ge=0)


class ToolEvent(BaseModel):
    """One attempted tool call and the exact result added to the conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn: int
    tool_call_id: str
    name: str
    result: JsonObject


class AgentTrace(BaseModel):
    """Conversation evidence, including provider continuation and partial progress."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    agent_interface_version: int
    seed: int | None
    stop_reason: StopReason | None = None
    initial_observation: AgentObservation
    tools: list[ToolDefinition]
    tools_sha256: str
    transcript: list[Message]
    model_responses: list[GenerationResponse] = Field(
        default_factory=list[GenerationResponse]
    )
    tool_events: list[ToolEvent] = Field(default_factory=list[ToolEvent])
    usage: TokenUsage = TokenUsage()
    prompt_tokens: int | None = None
