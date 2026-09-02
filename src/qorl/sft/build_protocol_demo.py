from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

from qorl.agent.protocol import AgentProtocol
from qorl.agent.tool_runtime import AgentEnvironment
from qorl.db.fixture import DatabaseFixture
from qorl.db.pool import start_pool
from qorl.workload.taskset import TaskSet
from qorl.measure.rollout import RolloutEvaluator
from qorl.plans.verify import plan_join_tree
from qorl.sft.validate import validate_protocol_demo


DEMONSTRATION_ID = "protocol-demo-v1"
TASK_ID = "ceb-4a-4a434"
MAXIMUM_MODEL_TURNS = 64
CALL_SEQUENCE = ["get_plan", "evaluate_candidate", "get_plan", "finish"]


def leading_action(plan: dict[str, Any]) -> dict[str, Any]:
    """Steer PostgreSQL back to its own deterministic default join tree."""
    tree = plan_join_tree(plan)
    if tree is None or isinstance(tree, str):
        raise RuntimeError("default plan does not contain a complete join tree")

    def encode(node: str | tuple[Any, Any]) -> str | dict[str, Any]:
        if isinstance(node, str):
            return node
        return {"left": encode(node[0]), "right": encode(node[1])}

    return {"version": 1, "leading": encode(tree)}


def call_tool(
    messages: list[dict[str, Any]],
    environment: AgentEnvironment,
    protocol: AgentProtocol,
    turn: int,
    name: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    if name not in protocol.available_tool_names(
        turn, len(environment.evaluator.candidates)
    ):
        raise RuntimeError(f"{name} is unavailable on turn {turn}")
    call_id = f"call-{turn:04d}"
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, sort_keys=True),
                    },
                }
            ],
        }
    )
    result, finished = environment.execute(name, arguments)
    if not isinstance(result, dict):
        result = {"result": result}
    result = {**result, "_turn_budget": protocol.budget(turn)}
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": json.dumps(result, sort_keys=True),
        }
    )
    return result, finished


def build_demo(repository: Path) -> dict[str, Any]:
    fixture = DatabaseFixture.load(repository)
    task_set = TaskSet.load(repository, "ceb-v1", fixture.data_identity)
    task = next(
        (item for item in task_set.inventory["tasks"] if item["task_id"] == TASK_ID),
        None,
    )
    if task is None:
        raise RuntimeError(f"missing pinned CEB task: {TASK_ID}")

    with contextlib.closing(
        start_pool(fixture, "qorl-protocol-demo")
    ) as pool, pool.claim_worker() as slot:
        worker = slot.worker
        evaluator = RolloutEvaluator(worker, task_set, task)
        evaluator.start()
        protocol = AgentProtocol.from_evaluator(evaluator, MAXIMUM_MODEL_TURNS)
        environment = AgentEnvironment(evaluator)
        messages = protocol.initial_messages()

        call_tool(messages, environment, protocol, 1, "get_plan", {"candidate_id": "default"})
        action = leading_action(evaluator.default["plain_explain"]["Plan"])
        candidate, _ = call_tool(
            messages,
            environment,
            protocol,
            2,
            "evaluate_candidate",
            {"action": action},
        )
        candidate_id = candidate.get("candidate_id")
        if not candidate.get("constraints_satisfied") or not isinstance(
            candidate_id, str
        ):
            raise RuntimeError(f"deterministic candidate failed: {candidate}")
        call_tool(
            messages,
            environment,
            protocol,
            3,
            "get_plan",
            {"candidate_id": candidate_id},
        )
        _, finished = call_tool(
            messages, environment, protocol, 4, "finish", {}
        )
        if not finished:
            raise RuntimeError("finish did not end the demonstration")

        measured_candidate = evaluator.candidates[0]
        return {
            "schema_version": 1,
            "messages": messages,
            "tools": protocol.tools,
            "metadata": {
                "demonstration_id": DEMONSTRATION_ID,
                "teacher": "postgres_default_join_tree",
                "task_set_id": task_set.task_set_id,
                "task_id": task["task_id"],
                "template_id": task["template_id"],
                "partition": task["partition"],
                "sql_sha256": task["sql_sha256"],
                "data_identity": fixture.data_identity,
                "runtime_identity": fixture.runtime_identity,
                "maximum_model_turns": MAXIMUM_MODEL_TURNS,
                "call_sequence": CALL_SEQUENCE,
            },
            "evidence": {
                "default_plan": evaluator.default["plain_explain"],
                "candidates": {
                    candidate_id: {
                        "action": measured_candidate["action"],
                        "plain_explain": measured_candidate["plain_explain"],
                        "pg_hint_plan": measured_candidate["pg_hint_plan"],
                    }
                },
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one live, deterministic CEB protocol demonstration."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sft/protocol-demo-v1.json"),
    )
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    output = arguments.output
    if not output.is_absolute():
        output = repository / output

    document = build_demo(repository)
    summary = validate_protocol_demo(document, repository)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
