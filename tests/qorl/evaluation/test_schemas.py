import pytest
from pydantic import ValidationError

from qorl.evaluation.schemas import EvaluationSettings


@pytest.mark.parametrize("count", [0, -1])
def test_evaluation_requires_positive_rollout_count(count: int) -> None:
    with pytest.raises(ValidationError):
        EvaluationSettings(rollouts_per_task=count)
