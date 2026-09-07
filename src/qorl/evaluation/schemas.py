"""Settings for independent live evaluation rollouts."""

from pydantic import BaseModel, ConfigDict, Field


class EvaluationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rollouts_per_task: int = Field(ge=1)
