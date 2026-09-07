import pytest
from pydantic import ValidationError

from qorl.model.schemas import ModelSettings, ModelWeightIndex


def test_model_identity_requires_context_and_known_provider() -> None:
    with pytest.raises(ValidationError):
        ModelSettings.model_validate({"provider": "made-up", "name_or_path": "model"})


def test_weight_index_requires_nonempty_shard_map() -> None:
    with pytest.raises(ValidationError):
        ModelWeightIndex(weight_map={})
