import pytest
from pydantic import ValidationError

from qorl.adapters.schemas import LoraSettings


@pytest.mark.parametrize("rank,dropout", [(0, 0.0), (16, -0.1), (16, 1.1)])
def test_lora_rank_and_dropout_bounds(rank: int, dropout: float) -> None:
    with pytest.raises(ValidationError):
        LoraSettings(rank=rank, alpha=32, dropout=dropout, target_modules=["q_proj"])
