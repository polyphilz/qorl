from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from qorl.plans.action import ActionError, TaskCatalog, compile_action
from qorl.agent.protocol import AgentProtocol
from qorl.agent.tool_runtime import AgentEnvironment
from qorl.agent.tools import candidate_feedback
from qorl.db.fixture import DatabaseFixture
from qorl.db.pool import start_pool
from qorl.db.worker import PostgresWorker, WorkerError
from qorl.plans.fingerprint import plan_sha256
from qorl.plans.random_tree import random_join_tree
from qorl.workload.taskset import TaskSet
from qorl.measure.rollout import MAX_CANDIDATES, RolloutEvaluator
from qorl.plans.verify import (
    JOIN_METHODS,
    SCAN_METHODS,
    compact_plan,
    contains_node,
    hint_status,
    index_names,
    nodes,
    plan_join_tree,
    relation_set,
    verify_action,
)
from qorl.sft.assemble import (
    DATASET_ID,
    DATASET_SEED,
    SPLIT_COUNTS,
    canonical_json,
    finalize_dataset,
    ranked_tasks,
    select_tasks,
)
from qorl.sft.validate import validate_protocol_demo


MAXIMUM_MODEL_TURNS = 64
MAX_ACTION_ATTEMPTS = 4
RECIPES = (
    "direct",
    "default_plan",
    "relation_size",
    "indexes",
    "column_stats",
    "focused",
    "schema",
    "extended_stats",
    "plan_aware",
)


class PlanValidationEvaluator(RolloutEvaluator):
    """Exercise the live planning path without executing benchmark queries."""

    def start(self) -> dict[str, Any]:
        plain = self.worker.explain(self.sql, self.global_timeout_ms)
        fingerprint = plan_sha256(plain.document["Plan"])
        self.default = {
            "plan_sha256": fingerprint,
            "plain_explain": plain.document,
            "compact_plan": compact_plan(plain.document["Plan"]),
            "median_execution_time_ms": None,
        }
        self.by_fingerprint[fingerprint] = {"candidate_id": "default"}
        return self.default

    def evaluate(self, raw_action: Any) -> dict[str, Any]:
        if not self.default:
            raise RuntimeError("rollout baseline has not been started")
        if len(self.candidates) >= MAX_CANDIDATES:
            raise RuntimeError("rollout candidate budget is exhausted")
        candidate_id = f"candidate-{len(self.candidates) + 1:02d}"
        try:
            action, hint = compile_action(raw_action, self.catalog)
        except ActionError as error:
            return self.invalid_candidate(candidate_id, raw_action, str(error))

        try:
            plain = self.worker.explain(self.sql, self.global_timeout_ms, hint=hint)
        except WorkerError as error:
            return self.invalid_candidate(
                candidate_id, action, str(error), hint=hint, action_valid=True
            )

        verification = verify_action(
            action, plain.document["Plan"], plain.hint_diagnostics
        )
        diagnostics = hint_status(plain.hint_diagnostics)
        if not verification.valid:
            return self.invalid_candidate(
                candidate_id,
                action,
                "; ".join(verification.errors),
                hint=hint,
                action_valid=True,
                pg_hint_plan=diagnostics,
            )

        fingerprint = plan_sha256(plain.document["Plan"])
        duplicate = self.by_fingerprint.get(fingerprint)
        result = {
            "candidate_id": candidate_id,
            "action": action,
            "action_valid": True,
            "constraints_satisfied": True,
            "compiled_hint": hint,
            "duplicate_of": duplicate["candidate_id"] if duplicate else None,
            "plan_sha256": fingerprint,
            "plain_explain": plain.document,
            "compact_plan": compact_plan(plain.document["Plan"]),
            "provisional_measurements": [],
            "provisional_speedup": None,
            "measurement_status": "not_measured",
            "errors_or_diagnostics": [],
            "pg_hint_plan": diagnostics,
            "attempts_remaining": MAX_CANDIDATES - len(self.candidates) - 1,
        }
        self.candidates.append(result)
        if duplicate is None:
            self.by_fingerprint[fingerprint] = {"candidate_id": candidate_id}
        return result


def trace_seed(task_id: str) -> int:
    digest = hashlib.sha256(f"{DATASET_SEED}:{task_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def encode_tree(node: str | tuple[Any, Any]) -> str | dict[str, Any]:
    if isinstance(node, str):
        return node
    return {"left": encode_tree(node[0]), "right": encode_tree(node[1])}


def default_derived_actions(
    plan: dict[str, Any], catalog: TaskCatalog, rng: random.Random
) -> list[tuple[str, dict[str, Any]]]:
    """Build valid-looking directives from facts visible in the default plan."""
    actions: list[tuple[str, dict[str, Any]]] = []
    tree = plan_join_tree(plan)
    leading = (
        {"version": 1, "leading": encode_tree(tree)}
        if tree is not None and not isinstance(tree, str)
        else None
    )
    if leading:
        actions.append(("leading", leading))

    joins = sorted(
        (
            node
            for node in nodes(plan)
            if node.get("Node Type") in JOIN_METHODS
        ),
        key=lambda node: tuple(sorted(relation_set(node))),
    )
    join_action: dict[str, Any] | None = None
    memoize_action: dict[str, Any] | None = None
    if joins:
        join = joins[rng.randrange(len(joins))]
        relations = sorted(relation_set(join))
        join_action = {
            "version": 1,
            "joins": [
                {
                    "relations": relations,
                    "force": JOIN_METHODS[join["Node Type"]],
                }
            ],
        }
        actions.append(("join", join_action))
        memoize_action = {
            "version": 1,
            "joins": [
                {
                    "relations": relations,
                    "memoize": (
                        "force" if contains_node(join, "Memoize") else "forbid"
                    ),
                }
            ],
        }
        actions.append(("memoize", memoize_action))

        rows = max(1, int(join.get("Plan Rows", 1)))
        actions.append(
            (
                "rows",
                {
                    "version": 1,
                    "row_corrections": [
                        {
                            "relations": relations,
                            "mode": "absolute",
                            "value": rows,
                        }
                    ],
                },
            )
        )

    scans = sorted(
        (
            node
            for node in nodes(plan)
            if node.get("Node Type") in SCAN_METHODS
            and isinstance(node.get("Alias"), str)
        ),
        key=lambda node: node["Alias"],
    )
    scan_action: dict[str, Any] | None = None
    if scans:
        scan = scans[rng.randrange(len(scans))]
        method = SCAN_METHODS[scan["Node Type"]]
        item: dict[str, Any] = {"relation": scan["Alias"], "force": method}
        used_indexes = sorted(index_names(scan))
        if method in {"index", "index_only", "bitmap"} and used_indexes:
            item["indexes"] = used_indexes
        scan_action = {"version": 1, "scans": [item]}
        actions.append(("scan", scan_action))

        nonparallel = next(
            (node for node in scans if not node.get("Parallel Aware")), None
        )
        if nonparallel:
            actions.append(
                (
                    "parallel",
                    {
                        "version": 1,
                        "parallel": [
                            {
                                "relation": nonparallel["Alias"],
                                "workers": 0,
                                "mode": "soft",
                            }
                        ],
                    },
                )
            )

        for node in scans:
            alias = node["Alias"]
            unused = sorted(catalog.indexes.get(alias, frozenset()) - index_names(node))
            if unused:
                actions.append(
                    (
                        "index_exclusion",
                        {
                            "version": 1,
                            "disabled_indexes": [
                                {"relation": alias, "indexes": [unused[0]]}
                            ],
                        },
                    )
                )
                break

    settings = [
        {"enable_memoize": True},
        {"random_page_cost": 4.0},
        {"join_collapse_limit": 8},
    ]
    rng.shuffle(settings)
    for values in settings:
        actions.append(("setting", {"version": 1, "settings": values}))
    setting_action = {
        "version": 1,
        "settings": settings[0],
    }

    if leading and join_action:
        actions.append(
            (
                "leading_join",
                {**leading, "joins": join_action["joins"]},
            )
        )
    if leading and scan_action:
        actions.append(
            (
                "leading_scan",
                {**leading, "scans": scan_action["scans"]},
            )
        )
    if leading:
        actions.append(
            (
                "leading_setting",
                {**leading, "settings": setting_action["settings"]},
            )
        )

    normalized: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for name, action in actions:
        try:
            value, _ = compile_action(action, catalog)
        except ActionError:
            continue
        encoded = canonical_json(value)
        if encoded not in seen:
            normalized.append((name, value))
            seen.add(encoded)
    return normalized


def inspection_calls(
    recipe: str, task: dict[str, Any], catalog: TaskCatalog
) -> list[tuple[str, dict[str, Any]]]:
    aliases = sorted(catalog.relations)
    indexed = next(
        (alias for alias in aliases if catalog.indexes.get(alias)), aliases[0]
    )
    edge = task["join_edges"][0].split("=", 1)[0]
    stats_alias = edge.split(":", 1)[0]
    stats_column = edge.rsplit(".", 1)[1]
    calls = {
        "direct": [],
        "default_plan": [("get_plan", {"candidate_id": "default"})],
        "relation_size": [("get_relation_size", {"relation": aliases[0]})],
        "indexes": [("list_indexes", {"relation": indexed})],
        "column_stats": [
            (
                "get_column_stats",
                {"relation": stats_alias, "column": stats_column},
            )
        ],
        "focused": [
            ("get_relation_size", {"relation": aliases[0]}),
            ("list_indexes", {"relation": indexed}),
        ],
        "schema": [("describe_table", {"relation": aliases[0]})],
        "extended_stats": [("get_extended_stats", {"relation": aliases[0]})],
        "plan_aware": [("get_plan", {"candidate_id": "default"})],
    }
    return calls[recipe]


def record_tool(
    messages: list[dict[str, Any]],
    protocol: AgentProtocol,
    turn: int,
    name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> None:
    call_id = f"call-{turn:04d}"
    messages.extend(
        [
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
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(
                    {**result, "_turn_budget": protocol.budget(turn)},
                    sort_keys=True,
                ),
            },
        ]
    )


def execute_inspection(
    messages: list[dict[str, Any]],
    environment: AgentEnvironment,
    protocol: AgentProtocol,
    turn: int,
    name: str,
    arguments: dict[str, Any],
) -> None:
    result, finished = environment.execute(name, arguments)
    if finished or not isinstance(result, dict) or "error" in result:
        raise RuntimeError(f"inspection failed: {name} {result}")
    record_tool(messages, protocol, turn, name, arguments, result)


def evaluate_action(
    evaluator: RolloutEvaluator,
    action: dict[str, Any],
    seen_actions: set[str],
    *,
    require_novel_plan: bool,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    try:
        normalized, _ = compile_action(action, evaluator.catalog)
    except ActionError:
        return None
    encoded = canonical_json(normalized)
    if encoded in seen_actions:
        return None

    candidate = evaluator.evaluate(normalized)
    if not candidate["action_valid"] or not candidate["constraints_satisfied"]:
        evaluator.candidates.pop()
        return None
    if require_novel_plan and candidate.get("duplicate_of") is not None:
        evaluator.candidates.pop()
        return None
    seen_actions.add(encoded)
    feedback = candidate_feedback(candidate)
    feedback["measurement_status"] = candidate["measurement_status"]
    return feedback, candidate


def choose_candidate(
    evaluator: RolloutEvaluator,
    derived: list[tuple[str, dict[str, Any]]],
    seen_actions: set[str],
    rng: random.Random,
    ordinal: int,
    slot: int,
) -> tuple[str, dict[str, Any], dict[str, Any], int]:
    rejected = 0
    if slot == 1:
        for _ in range(MAX_ACTION_ATTEMPTS):
            action = {"version": 1, "leading": random_join_tree(evaluator.catalog, rng)}
            evaluated = evaluate_action(
                evaluator, action, seen_actions, require_novel_plan=True
            )
            if evaluated:
                feedback, candidate = evaluated
                return "novel_leading", feedback, candidate, rejected
            rejected += 1

    if not derived:
        raise RuntimeError("default plan supplied no derivable actions")
    offset = (ordinal * 5 + slot) % len(derived)
    for name, action in derived[offset:] + derived[:offset]:
        evaluated = evaluate_action(
            evaluator, action, seen_actions, require_novel_plan=False
        )
        if evaluated:
            feedback, candidate = evaluated
            return name, feedback, candidate, rejected
        rejected += 1
    raise RuntimeError("could not produce another valid, distinct PlanAction")


def build_document(
    worker: PostgresWorker,
    task_set: TaskSet,
    task: dict[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    evaluator = PlanValidationEvaluator(worker, task_set, task)
    evaluator.start()
    protocol = AgentProtocol.from_evaluator(evaluator, MAXIMUM_MODEL_TURNS)
    environment = AgentEnvironment(evaluator)
    messages = protocol.initial_messages()
    rng = random.Random(trace_seed(task["task_id"]))
    recipe = RECIPES[ordinal % len(RECIPES)]
    requested_candidates = ordinal % 5 + 1
    call_sequence: list[str] = []
    turn = 1

    for name, arguments in inspection_calls(recipe, task, evaluator.catalog):
        execute_inspection(messages, environment, protocol, turn, name, arguments)
        call_sequence.append(name)
        turn += 1

    plan = evaluator.default["plain_explain"]["Plan"]
    derived = default_derived_actions(plan, evaluator.catalog, rng)
    seen_actions: set[str] = set()
    evidence: dict[str, dict[str, Any]] = {}
    strategies: list[str] = []
    rejected_actions = 0
    for slot in range(requested_candidates):
        strategy, feedback, candidate, rejected = choose_candidate(
            evaluator, derived, seen_actions, rng, ordinal, slot
        )
        rejected_actions += rejected
        candidate_id = feedback["candidate_id"]
        record_tool(
            messages,
            protocol,
            turn,
            "evaluate_candidate",
            {"action": candidate["action"]},
            feedback,
        )
        call_sequence.append("evaluate_candidate")
        strategies.append(strategy)
        evidence[candidate_id] = {
            "action": candidate["action"],
            "plain_explain": candidate["plain_explain"],
            "pg_hint_plan": candidate["pg_hint_plan"],
        }
        turn += 1

        if recipe == "plan_aware" and slot == 0:
            arguments = {"candidate_id": candidate_id}
            execute_inspection(
                messages, environment, protocol, turn, "get_plan", arguments
            )
            call_sequence.append("get_plan")
            turn += 1

    result, finished = environment.execute("finish", {})
    if not finished or result != {"status": "finished"}:
        raise RuntimeError("finish did not terminate the demonstration")
    record_tool(messages, protocol, turn, "finish", {}, result)
    call_sequence.append("finish")

    document = {
        "schema_version": 1,
        "messages": messages,
        "tools": protocol.tools,
        "metadata": {
            "demonstration_id": f"{DATASET_ID}-{ordinal + 1:04d}",
            "ordinal": ordinal,
            "teacher": "validated_plan_action_generator_v1",
            "task_set_id": task_set.task_set_id,
            "task_id": task["task_id"],
            "template_id": task["template_id"],
            "partition": task["partition"],
            "sql_sha256": task["sql_sha256"],
            "data_identity": worker.fixture.data_identity,
            "runtime_identity": worker.fixture.runtime_identity,
            "in_author_unique_plans_subset": task[
                "in_author_unique_plans_subset"
            ],
            "trace_seed": trace_seed(task["task_id"]),
            "maximum_model_turns": MAXIMUM_MODEL_TURNS,
            "inspection_recipe": recipe,
            "candidate_count": requested_candidates,
            "candidate_strategies": strategies,
            "rejected_generator_actions": rejected_actions,
            "measurement_mode": "plan_validation_only",
            "selection_used_speed": False,
            "call_sequence": call_sequence,
        },
        "evidence": {
            "default_plan": evaluator.default["plain_explain"],
            "candidates": evidence,
        },
    }
    validate_protocol_demo(document, worker.fixture.repository)
    return document


def existing_documents(
    repository: Path, output_dir: Path
) -> dict[int, tuple[Path, dict[str, Any]]]:
    existing: dict[int, tuple[Path, dict[str, Any]]] = {}
    for partition in SPLIT_COUNTS:
        for path in (output_dir / "demonstrations" / partition).glob("*.json"):
            document = json.loads(path.read_text(encoding="utf-8"))
            validate_protocol_demo(document, repository)
            ordinal = document["metadata"]["ordinal"]
            if ordinal in existing:
                raise RuntimeError(f"duplicate demonstration ordinal: {ordinal}")
            existing[ordinal] = (path, document)
    return existing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the deterministic 256/64 CEB protocol-SFT dataset."
    )
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sft/protocol-sft-v1"),
    )
    arguments = parser.parse_args()
    repository = arguments.repository.resolve()
    output_dir = arguments.output
    if not output_dir.is_absolute():
        output_dir = repository / output_dir

    fixture = DatabaseFixture.load(repository)
    task_set = TaskSet.load(repository, "ceb-v1", fixture.data_identity)
    selected = {
        partition: select_tasks(
            task_set.inventory["tasks"], partition, count, DATASET_SEED
        )
        for partition, count in SPLIT_COUNTS.items()
    }
    planned = [
        task
        for partition in ("train", "validation")
        for task in selected[partition]
    ]
    ranked = {
        partition: ranked_tasks(
            task_set.inventory["tasks"], partition, DATASET_SEED
        )
        for partition in SPLIT_COUNTS
    }
    existing = existing_documents(repository, output_dir)
    used_task_ids = {
        document["metadata"]["task_id"] for _, document in existing.values()
    }
    failure_path = output_dir / "generation-failures.jsonl"
    failures = (
        [json.loads(line) for line in failure_path.read_text().splitlines()]
        if failure_path.is_file()
        else []
    )
    attempted_task_ids = used_task_ids | {
        failure["task_id"] for failure in failures
    }
    missing = [ordinal for ordinal in range(len(planned)) if ordinal not in existing]

    if missing:
        with contextlib.closing(
            start_pool(fixture, "qorl-protocol-sft-data")
        ) as pool, pool.claim_worker() as slot:
            worker = slot.worker
            for ordinal in missing:
                requested = planned[ordinal]
                partition = requested["partition"]
                template = requested["template_id"]
                candidates = [
                    requested,
                    *(
                        task
                        for task in ranked[partition][template]
                        if task["task_id"] != requested["task_id"]
                    ),
                ]
                built = None
                for task in candidates:
                    if task["task_id"] in attempted_task_ids:
                        continue
                    attempted_task_ids.add(task["task_id"])
                    try:
                        built = build_document(worker, task_set, task, ordinal)
                    except Exception as error:
                        failure = {
                            "task_id": task["task_id"],
                            "template_id": template,
                            "error": str(error),
                        }
                        failures.append(failure)
                        failure_path.parent.mkdir(parents=True, exist_ok=True)
                        with failure_path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(failure, sort_keys=True) + "\n")
                        print(f"  rejected {task['task_id']}: {error}", flush=True)
                        continue
                    break
                if built is None:
                    raise RuntimeError(f"could not fill ordinal {ordinal} ({template})")
                path = (
                    output_dir
                    / "demonstrations"
                    / partition
                    / f"{built['metadata']['task_id']}.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(built, indent=2, sort_keys=True) + "\n")
                used_task_ids.add(built["metadata"]["task_id"])
                print(
                    f"[{ordinal + 1}/{len(planned)}] "
                    f"{built['metadata']['task_id']} "
                    f"candidates={built['metadata']['candidate_count']} "
                    f"recipe={built['metadata']['inspection_recipe']}",
                    flush=True,
                )

    manifest = finalize_dataset(repository, output_dir, failures)
    print(
        json.dumps(
            {
                "output": str(output_dir),
                "counts": manifest["counts"],
                "statistics": manifest["statistics"],
                "generation_failures": len(failures),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
