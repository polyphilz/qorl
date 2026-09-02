from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from qorl.action import TaskCatalog, compile_action
from qorl.agent.prompts import SYSTEM_PROMPT
from qorl.agent.protocol import AgentProtocol, RESERVED_DECISION_TURNS
from qorl.agent.tools import agent_tools
from qorl.calibration import plan_sha256
from qorl.fixture import TaskSet
from qorl.plan import verify_action
from qorl.rollout import MAX_CANDIDATES


class DemoValidationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DemoValidationError(message)


def parse_json(value: Any, label: str) -> Any:
    require(isinstance(value, str), f"{label} must be JSON text")
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise DemoValidationError(f"{label} is invalid JSON") from error


def hint_diagnostics(status: dict[str, str] | None) -> str:
    require(status is not None, "candidate is missing pg_hint_plan diagnostics")
    return (
        "HintStateDump: "
        f"{{used hints:{status['used']}}}, "
        f"{{not used hints:{status['not_used']}}}, "
        f"{{duplicate hints:{status['duplicate']}}}, "
        f"{{error hints:{status['error']}}}"
    )


def validate_protocol_demo(
    document: dict[str, Any], repository: Path
) -> dict[str, Any]:
    """Validate one frozen tool-use demonstration without executing SQL."""
    require(document.get("schema_version") == 1, "unsupported demo schema")
    metadata = document.get("metadata")
    messages = document.get("messages")
    tools = document.get("tools")
    evidence = document.get("evidence")
    require(isinstance(metadata, dict), "metadata must be an object")
    require(isinstance(messages, list), "messages must be a list")
    require(isinstance(tools, list), "tools must be a list")
    require(isinstance(evidence, dict), "evidence must be an object")

    task_set = TaskSet.load(repository, metadata.get("task_set_id"))
    task = next(
        (
            item
            for item in task_set.inventory["tasks"]
            if item["task_id"] == metadata.get("task_id")
        ),
        None,
    )
    require(task is not None, "demo task is absent from its task inventory")
    partition = metadata.get("partition", task["partition"])
    require(
        partition in {"train", "validation"},
        "demo partition must be train or validation",
    )
    require(task["partition"] == partition, "demo task partition mismatch")
    require(
        metadata.get("database") == task_set.inventory["database"],
        "demo database identity differs from its task inventory",
    )
    sql = task_set.load_sql(task)

    require(len(messages) >= 4, "demo has no tool interaction")
    require(
        messages[0] == {"role": "system", "content": SYSTEM_PROMPT},
        "system message differs from the live agent prompt",
    )
    require(messages[1].get("role") == "user", "second message must be user")
    observation = parse_json(messages[1].get("content"), "initial observation")
    require(isinstance(observation, dict), "initial observation must be an object")
    require(observation.get("task_id") == task["task_id"], "task ID mismatch")
    require(observation.get("sql") == sql, "task SQL mismatch")
    require(observation.get("relations") == task["relations"], "relations mismatch")
    require(observation.get("join_edges") == task["join_edges"], "join edges mismatch")

    aliases = sorted(relation["alias"] for relation in task["relations"])
    require(tools == agent_tools(aliases), "tool schemas differ from the live agent")
    maximum_turns = metadata.get("maximum_model_turns")
    require(
        isinstance(maximum_turns, int) and maximum_turns > 0,
        "maximum_model_turns must be positive",
    )
    inspection_limit = min(
        len(aliases) * 3,
        max(0, maximum_turns - RESERVED_DECISION_TURNS),
    )
    protocol = AgentProtocol(
        maximum_model_turns=maximum_turns,
        inspection_turn_limit=inspection_limit,
        observation=observation,
        tools=tools,
    )
    require(
        observation.get("turn_budget")
        == {
            "total_model_turns": maximum_turns,
            "maximum_inspection_turns": inspection_limit,
            "reserved_final_turns": RESERVED_DECISION_TURNS,
            "reserved_for": {
                "candidate_evaluations": MAX_CANDIDATES,
                "finish_or_keep_default": 1,
            },
        },
        "turn budget differs from the live agent protocol",
    )
    indexes = observation.get("indexes")
    require(isinstance(indexes, dict), "initial observation has invalid indexes")
    catalog = TaskCatalog.from_task(
        task, {alias: set(names) for alias, names in indexes.items()}
    )

    tail = messages[2:]
    require(len(tail) % 2 == 0, "assistant/tool messages are not paired")
    issued_candidates: list[str] = []
    normalized_actions: set[str] = set()
    call_ids: set[str] = set()
    inspections: set[tuple[str, str]] = set()
    call_sequence: list[str] = []
    finished = False
    kept_default = False

    for turn, offset in enumerate(range(0, len(tail), 2), start=1):
        assistant = tail[offset]
        tool_result = tail[offset + 1]
        require(assistant.get("role") == "assistant", f"turn {turn}: not assistant")
        calls = assistant.get("tool_calls")
        require(isinstance(calls, list) and len(calls) == 1, f"turn {turn}: expected one tool call")
        call = calls[0]
        require(call.get("type") == "function", f"turn {turn}: invalid call type")
        call_id = call.get("id")
        require(isinstance(call_id, str) and call_id, f"turn {turn}: invalid call ID")
        require(call_id not in call_ids, f"turn {turn}: duplicate call ID")
        call_ids.add(call_id)
        function = call.get("function")
        require(isinstance(function, dict), f"turn {turn}: missing function")
        name = function.get("name")
        arguments = parse_json(function.get("arguments"), f"turn {turn} arguments")
        require(isinstance(arguments, dict), f"turn {turn}: arguments must be an object")
        require(
            name in protocol.available_tool_names(turn, len(issued_candidates)),
            f"turn {turn}: {name} was not available",
        )
        call_sequence.append(name)

        require(tool_result.get("role") == "tool", f"turn {turn}: missing tool result")
        require(tool_result.get("tool_call_id") == call_id, f"turn {turn}: call ID mismatch")
        require(tool_result.get("name") == name, f"turn {turn}: tool name mismatch")
        result = parse_json(tool_result.get("content"), f"turn {turn} result")
        require(isinstance(result, dict), f"turn {turn}: result must be an object")
        require(result.get("_turn_budget") == protocol.budget(turn), f"turn {turn}: budget mismatch")
        require("error" not in result, f"turn {turn}: tool returned an error")

        if name == "evaluate_candidate":
            expected_id = f"candidate-{len(issued_candidates) + 1:02d}"
            require(result.get("candidate_id") == expected_id, f"turn {turn}: candidate ID mismatch")
            raw_action = arguments.get("action")
            action, hint = compile_action(raw_action, catalog)
            encoded_action = json.dumps(
                action, sort_keys=True, separators=(",", ":")
            )
            require(
                encoded_action not in normalized_actions,
                f"turn {turn}: duplicate normalized action",
            )
            normalized_actions.add(encoded_action)
            require(result.get("action_valid") is True, f"turn {turn}: invalid action")
            require(result.get("constraints_satisfied") is True, f"turn {turn}: unsatisfied constraints")
            require(result.get("compiled_hint") == hint, f"turn {turn}: hint mismatch")
            if metadata.get("measurement_mode") == "plan_validation_only":
                require(
                    result.get("measurement_status") == "not_measured",
                    f"turn {turn}: unexpected measurement status",
                )
                require(
                    result.get("provisional_speedup") is None,
                    f"turn {turn}: plan-only demo contains a speedup",
                )
                require(
                    result.get("planning_time_ms") == []
                    and result.get("execution_time_ms") == [],
                    f"turn {turn}: plan-only demo contains timings",
                )

            candidate_evidence = evidence.get("candidates", {}).get(expected_id)
            require(isinstance(candidate_evidence, dict), f"turn {turn}: missing candidate evidence")
            require(candidate_evidence.get("action") == action, f"turn {turn}: normalized action mismatch")
            plan = candidate_evidence.get("plain_explain")
            require(isinstance(plan, dict) and isinstance(plan.get("Plan"), dict), f"turn {turn}: invalid plan evidence")
            fingerprint = plan_sha256(plan["Plan"])
            require(result.get("plan_sha256") == fingerprint, f"turn {turn}: plan checksum mismatch")
            verification = verify_action(
                action,
                plan["Plan"],
                hint_diagnostics(candidate_evidence.get("pg_hint_plan")),
            )
            require(verification.valid, "; ".join(verification.errors))
            issued_candidates.append(expected_id)
        elif name == "get_plan":
            candidate_id = arguments.get("candidate_id")
            require(
                candidate_id == "default" or candidate_id in issued_candidates,
                f"turn {turn}: candidate ID was not issued",
            )
            key = (name, json.dumps(arguments, sort_keys=True))
            require(key not in inspections, f"turn {turn}: repeated inspection")
            inspections.add(key)
            expected_plan = (
                evidence.get("default_plan")
                if candidate_id == "default"
                else evidence["candidates"][candidate_id]["plain_explain"]
            )
            actual_plan = {key: value for key, value in result.items() if key != "_turn_budget"}
            require(actual_plan == expected_plan, f"turn {turn}: plan result differs from evidence")
        elif name == "finish":
            require(not arguments, f"turn {turn}: finish takes no arguments")
            require(result.get("status") == "finished", f"turn {turn}: finish failed")
            require(offset == len(tail) - 2, "finish must be the final call")
            finished = True
        elif name == "keep_default":
            require(
                not arguments, f"turn {turn}: keep_default takes no arguments"
            )
            require(
                result.get("status") == "kept_default",
                f"turn {turn}: keep_default failed",
            )
            require(
                offset == len(tail) - 2,
                "keep_default must be the final call",
            )
            kept_default = True
        else:
            key = (name, json.dumps(arguments, sort_keys=True))
            require(key not in inspections, f"turn {turn}: repeated inspection")
            inspections.add(key)

    require(finished or kept_default, "demo has no terminal decision")
    require(not (finished and kept_default), "demo has two terminal decisions")
    if kept_default:
        require(
            not issued_candidates,
            "keep_default demo must not submit a candidate",
        )
    else:
        require(
            1 <= len(issued_candidates) <= MAX_CANDIDATES,
            f"demo must submit between 1 and {MAX_CANDIDATES} candidates",
        )
    if "candidate_count" in metadata:
        require(
            metadata["candidate_count"] == len(issued_candidates),
            "candidate count differs from metadata",
        )
    require(
        call_sequence == metadata.get("call_sequence"),
        "call sequence differs from metadata",
    )
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return {
        "task_id": task["task_id"],
        "turn_count": len(call_sequence),
        "candidate_ids": issued_candidates,
        "terminal_decision": "keep_default" if kept_default else "finish",
        "call_sequence": call_sequence,
        "canonical_sha256": hashlib.sha256(encoded).hexdigest(),
    }
