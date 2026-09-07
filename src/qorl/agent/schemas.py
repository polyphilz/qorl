"""Conversation budgets, independent of the model provider."""

from pydantic import BaseModel, ConfigDict, Field


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_attempts: int = Field(ge=1)
    maximum_model_turns: int = Field(ge=1)
    inspection_turns_per_alias: int = Field(ge=0)
