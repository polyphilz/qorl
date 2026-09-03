from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from qorl.agent.prompts import system_prompt
from qorl.agent.tools import agent_tools
from qorl.agent.types import InspectionExecutor, ToolName
from qorl.measure.rollout import MAX_CANDIDATES, RolloutEvaluator
from qorl.plans.schemas import BOOLEAN_SETTINGS, INTEGER_SETTINGS, NUMERIC_SETTINGS

INSPECTION_TURNS_PER_ALIAS = 3


@dataclass(frozen=True)
class AgentInterface:
    """The exact observation, tools, and turn rules shown to the model."""

    maximum_model_turns: int
    inspection_turn_limit: int
    observation: dict[str, Any]
    tools: list[dict[str, Any]]

    @classmethod
    def from_evaluator(
        cls,
        evaluator: RolloutEvaluator[InspectionExecutor],
        maximum_model_turns: int,
        context_length: int | None = None,
        completion_reserve: int | None = None,
    ) -> AgentInterface:
        aliases = sorted(evaluator.catalog.relations)
        tools = agent_tools(aliases)
        candidate_attempts = getattr(evaluator, "max_candidates", MAX_CANDIDATES)
        reserved_decision_turns = candidate_attempts + 1
        inspection_turn_limit = min(
            len(aliases) * INSPECTION_TURNS_PER_ALIAS,
            max(0, maximum_model_turns - reserved_decision_turns),
        )
        settings = set(BOOLEAN_SETTINGS) | set(NUMERIC_SETTINGS) | set(INTEGER_SETTINGS)
        if evaluator.default is None:
            raise RuntimeError("rollout baseline has not been started")
        observation = {
            "task_id": evaluator.task["task_id"],
            "objective": "minimize measured warm-cache execution time",
            "sql": evaluator.sql,
            "relations": evaluator.task["relations"],
            "join_edges": evaluator.task["join_edges"],
            "indexes": {
                alias: sorted(indexes)
                for alias, indexes in sorted(evaluator.catalog.indexes.items())
            },
            "postgresql_server_version_num": evaluator.worker.fixture.snapshot[
                "postgresql"
            ]["server_version_num"],
            "planner_settings": evaluator.worker.settings(settings),
            "default_plan": evaluator.default.compact_plan,
            "default_median_execution_time_ms": (
                evaluator.default.median_execution_time_ms
            ),
            "candidate_attempts": candidate_attempts,
            "candidate_timeout_ms": evaluator.timeout_ms,
            "turn_budget": {
                "total_model_turns": maximum_model_turns,
                "maximum_inspection_turns": inspection_turn_limit,
                "reserved_final_turns": reserved_decision_turns,
                "reserved_for": {
                    "candidate_evaluations": candidate_attempts,
                    "finish_or_keep_default": 1,
                },
            },
        }
        if context_length is not None and completion_reserve is not None:
            observation["context_budget"] = {
                "maximum_tokens": context_length,
                "reserved_for_next_completion": completion_reserve,
            }
        return cls(
            maximum_model_turns=maximum_model_turns,
            inspection_turn_limit=inspection_turn_limit,
            observation=observation,
            tools=tools,
        )

    def initial_messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": system_prompt(self.candidate_attempts)},
            {
                "role": "user",
                "content": json.dumps(self.observation, sort_keys=True),
            },
        ]

    def available_tools(self, turn: int, candidate_count: int) -> list[dict[str, Any]]:
        names = self.available_tool_names(turn, candidate_count)
        return [tool for tool in self.tools if tool["function"]["name"] in names]

    def available_tool_names(self, turn: int, candidate_count: int) -> set[str]:
        if candidate_count >= self.candidate_attempts:
            return {ToolName.FINISH.value}
        if turn > self.inspection_turn_limit:
            return (
                {ToolName.EVALUATE_CANDIDATE.value, ToolName.FINISH.value}
                if candidate_count
                else {
                    ToolName.EVALUATE_CANDIDATE.value,
                    ToolName.KEEP_DEFAULT.value,
                }
            )
        names = {tool["function"]["name"] for tool in self.tools}
        if not candidate_count:
            names.remove(ToolName.FINISH.value)
        else:
            names.remove(ToolName.KEEP_DEFAULT.value)
        return names

    def budget(self, turn: int) -> dict[str, int]:
        return {
            "current_turn": turn,
            "turns_remaining": self.maximum_model_turns - turn,
            "unrestricted_turns_remaining": max(0, self.inspection_turn_limit - turn),
            "reserved_final_turns": self.reserved_decision_turns,
        }

    @property
    def candidate_attempts(self) -> int:
        return int(
            self.observation["turn_budget"]["reserved_for"]["candidate_evaluations"]
        )

    @property
    def reserved_decision_turns(self) -> int:
        return int(self.observation["turn_budget"]["reserved_final_turns"])
