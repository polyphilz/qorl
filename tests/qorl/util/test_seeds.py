import os
import subprocess
import sys

from qorl.util.seeds import derive_seed


def test_seed_is_stable_and_unsigned_32_bit() -> None:
    result = derive_seed(42, "task-selection", "ceb", "ceb-1a")
    assert result == 1_100_150_485
    assert result == derive_seed(42, "task-selection", "ceb", "ceb-1a")
    assert 0 <= result < 2**32


def test_seed_separates_purposes_and_identifier_boundaries() -> None:
    seeds = {
        derive_seed(42, "selection", "ab", "c"),
        derive_seed(42, "selection", "a", "bc"),
        derive_seed(42, "selection", "c", "ab"),
        derive_seed(42, "rollout", "ab", "c"),
        derive_seed(43, "selection", "ab", "c"),
    }
    assert len(seeds) == 5


def test_seed_does_not_depend_on_python_hash_randomization() -> None:
    code = (
        "from qorl.util.seeds import derive_seed; "
        "print(derive_seed(42, 'task-selection', 'ceb', 'ceb-1a'))"
    )
    first = subprocess.check_output(
        [sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": "1"}
    )
    second = subprocess.check_output(
        [sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": "2"}
    )
    assert first == second
    assert int(first) == derive_seed(42, "task-selection", "ceb", "ceb-1a")
