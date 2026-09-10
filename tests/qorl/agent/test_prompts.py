import pytest

from qorl.agent.prompts import system_prompt


@pytest.mark.parametrize(
    ("attempts", "label"), [(1, "candidate evaluation"), (5, "candidate evaluations")]
)
def test_prompt_budget_and_join_contract(attempts: int, label: str) -> None:
    prompt = system_prompt(attempts)
    assert prompt.startswith("You are qo-agent.")
    assert f"You may submit up to {attempts} {label}; you do not need\n" in prompt
    assert (
        "to use every attempt. Reserve one model turn for a terminal tool call."
        in prompt
    )
    assert (
        "Calling finish or keep_default does not consume a candidate attempt." in prompt
    )
    assert (
        "Each joins[].relations value must contain the complete set of leaf aliases\n"
        "beneath one internal node of the candidate plan.\n"
        "With leading, use only internal-node sets created by that tree.\n"
        "Without leading, use subtree sets visible in the initial plan summary or get_plan.\n"
        "Omit empty constraints."
    ) in prompt
    assert (
        "Otherwise finish by selecting the eligible candidate you expect to execute fastest.\n"
        "You may select an earlier candidate.\n"
    ) in prompt
    assert 'finish with selected_candidate_id="default"' in prompt
    assert "no valid candidate unless you select default" in prompt


@pytest.mark.parametrize("attempts", [0, -1])
def test_prompt_requires_a_candidate_budget(attempts: int) -> None:
    with pytest.raises(ValueError, match="candidate_attempts must be at least 1"):
        system_prompt(attempts)
