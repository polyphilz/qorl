import pytest
from pydantic import ValidationError

from qorl.agent.schemas import AgentSettings


def test_candidate_budget_does_not_restrict_schema_to_one_candidate() -> None:
    settings = AgentSettings(
        candidate_attempts=5, maximum_model_turns=64, inspection_turns_per_alias=3
    )
    assert settings.candidate_attempts == 5


def test_negative_inspection_budget_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentSettings(
            candidate_attempts=1, maximum_model_turns=64, inspection_turns_per_alias=-1
        )
