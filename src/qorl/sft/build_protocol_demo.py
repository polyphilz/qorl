from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

from qorl.agent.interface import AgentInterface
from qorl.agent.tool_runtime import AgentEnvironment
from qorl.agent.types import ToolName
from qorl.measure.timeouts import DEFAULT_STATEMENT_TIMEOUT_MS
from qorl.measure.validation import PlanValidationEvaluator
from qorl.paths import REPOSITORY_ROOT
from qorl.plans.verify import plan_join_tree
from qorl.postgres.config import PostgresConfig
from qorl.sft.schemas import require_object
from qorl.sft.validate import PROTOCOL_DEMO_SCHEMA_VERSION, validate_protocol_demo
from qorl.taskset.schemas import Task
from qorl.taskset.taskset import TaskSet
from qorl.worker_pool.config import load_pool_config
from qorl.worker_pool.containers import start_pool
from qorl.worker_pool.schemas import PoolConfig

DEMONSTRATION_ID = "protocol-demo-v2"
TASK_ID = "ceb-4a-4a434"
MAXIMUM_MODEL_TURNS = 64
CALL_SEQUENCE = [
    ToolName.GET_PLAN.value,
    ToolName.EVALUATE_CANDIDATE.value,
    ToolName.GET_PLAN.value,
    ToolName.FINISH.value,
]


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
    interface: AgentInterface,
    turn: int,
    name: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    if name not in interface.available_tool_names(
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
    result = {**result, "_turn_budget": interface.budget(turn)}
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": json.dumps(result, sort_keys=True),
        }
    )
    return result, finished


def build_demo(
    repository: Path, *, postgres_config: PostgresConfig, pool_config: PoolConfig
) -> dict[str, Any]:
    task_set = TaskSet.load(repository, "ceb")
    task = next(
        (item.model_dump() for item in task_set.tasks if item.task_id == TASK_ID),
        None,
    )
    if task is None:
        raise RuntimeError(f"missing pinned CEB task: {TASK_ID}")

    with (
        contextlib.closing(
            start_pool(
                "qorl-protocol-demo",
                repository / "data/imdb.tar.gz",
                postgres_config=postgres_config,
                pool_config=pool_config,
            )
        ) as pool,
        pool.claim_worker() as slot,
    ):
        worker = slot.client
        evaluator = PlanValidationEvaluator(
            worker,
            task_set,
            Task.model_validate(task),
            default_timeout_ms=DEFAULT_STATEMENT_TIMEOUT_MS,
            max_candidates=1,
        )
        evaluator.start()
        interface = AgentInterface.from_evaluator(evaluator, MAXIMUM_MODEL_TURNS)
        environment = AgentEnvironment(evaluator)
        messages = interface.initial_messages()

        call_tool(
            messages,
            environment,
            interface,
            1,
            ToolName.GET_PLAN.value,
            {"candidate_id": "default"},
        )
        if evaluator.default is None:
            raise RuntimeError("rollout baseline has not been started")
        action = leading_action(
            require_object(evaluator.default.plain_explain["Plan"], "default plan")
        )
        candidate, _ = call_tool(
            messages,
            environment,
            interface,
            2,
            ToolName.EVALUATE_CANDIDATE.value,
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
            interface,
            3,
            ToolName.GET_PLAN.value,
            {"candidate_id": candidate_id},
        )
        _, finished = call_tool(
            messages, environment, interface, 4, ToolName.FINISH.value, {}
        )
        if not finished:
            raise RuntimeError("finish did not end the demonstration")

        measured_candidate = evaluator.candidates[0]
        return {
            "schema_version": PROTOCOL_DEMO_SCHEMA_VERSION,
            "messages": messages,
            "tools": interface.tools,
            "metadata": {
                "demonstration_id": DEMONSTRATION_ID,
                "teacher": "postgres_default_join_tree",
                "task_set_id": task_set.task_set_id,
                "task_id": task["task_id"],
                "template_id": task["template_id"],
                "partition": task["partition"],
                "sql_sha256": task["sql_sha256"],
                "data_identity": {"fixture_id": task_set.fixture_id},
                "runtime_identity": {"postgres_config_id": postgres_config.config_id},
                "maximum_model_turns": MAXIMUM_MODEL_TURNS,
                "call_sequence": CALL_SEQUENCE,
            },
            "evidence": {
                "default_plan": evaluator.default.plain_explain,
                "candidates": {
                    candidate_id: {
                        "action": measured_candidate.action,
                        "plain_explain": measured_candidate.plain_explain,
                        "plan_sha256": measured_candidate.plan_sha256,
                        "pg_hint_plan": measured_candidate.pg_hint_plan,
                    }
                },
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one live, deterministic CEB demonstration."
    )
    parser.add_argument("--repository", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--postgres-config", type=Path, required=True)
    parser.add_argument("--pool-config", type=Path, required=True)
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

    document = build_demo(
        repository,
        postgres_config=PostgresConfig.load(arguments.postgres_config),
        pool_config=load_pool_config(arguments.pool_config),
    )
    summary = validate_protocol_demo(document, repository)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
