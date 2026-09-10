from __future__ import annotations

import json
from dataclasses import dataclass

from qorl.agent.presentation import plan_view
from qorl.agent.prompts import system_prompt
from qorl.agent.schemas import (
    AgentObservation,
    ContextBudget,
    DecisionTurns,
    RemainingTurnBudget,
    ResourceLimits,
    TurnBudget,
)
from qorl.agent.tools import agent_tools
from qorl.agent.types import AgentEvaluator, ToolName
from qorl.measure.rollout import RolloutEvaluator
from qorl.model.schemas import Message, MessageRole, ToolDefinition
from qorl.postgres.schemas import PlannerSettings, PostgresResourceLimits

INSPECTION_TURNS_PER_ALIAS = 3
AGENT_INTERFACE_VERSION = 7


@dataclass(frozen=True)
class AgentInterface:
    """The exact observation, tools, and turn rules shown to the model."""

    maximum_model_turns: int
    inspection_turn_limit: int
    observation: AgentObservation
    tools: list[ToolDefinition]

    @classmethod
    def from_evaluator(
        cls,
        evaluator: AgentEvaluator,
        maximum_model_turns: int,
        context_length: int | None = None,
        completion_reserve: int | None = None,
        *,
        inspection_turns_per_alias: int = INSPECTION_TURNS_PER_ALIAS,
    ) -> AgentInterface:
        aliases = sorted(evaluator.catalog.relations)
        candidate_attempts = evaluator.max_candidates
        reserved_decision_turns = candidate_attempts + 1
        inspection_turn_limit = min(
            len(aliases) * inspection_turns_per_alias,
            max(0, maximum_model_turns - reserved_decision_turns),
        )
        if evaluator.default is None:
            raise RuntimeError("rollout baseline has not been started")
        postgres_settings = evaluator.worker.settings
        measurement = (
            evaluator.measurement if isinstance(evaluator, RolloutEvaluator) else None
        )
        tools = agent_tools(
            aliases,
            execution_feedback=measurement is not None
            and measurement.candidate_feedback_measurements > 0,
        )
        observation = AgentObservation(
            task_id=evaluator.task.task_id,
            sql=evaluator.sql,
            relations=evaluator.task.relations,
            join_edges=evaluator.task.join_edges,
            indexes={
                alias: sorted(indexes)
                for alias, indexes in sorted(evaluator.catalog.indexes.items())
            },
            planner_settings=PlannerSettings.model_validate(
                postgres_settings.model_dump(include=set(PlannerSettings.model_fields))
            ),
            default_plan=plan_view(
                evaluator.default.plain_explain["Plan"], summary=True
            ),
            resource_limits=ResourceLimits(
                worker=evaluator.worker.allocation,
                worker_source="startup_verified_container_limits"
                if evaluator.worker.allocation is not None
                else "unavailable",
                postgres=PostgresResourceLimits.model_validate(
                    postgres_settings.model_dump(
                        include=set(PostgresResourceLimits.model_fields)
                    )
                ),
            ),
            default_median_execution_time_ms=(
                evaluator.default.median_execution_time_ms
            ),
            candidate_attempts=candidate_attempts,
            candidate_timeout_ms=evaluator.timeout_ms,
            candidate_feedback_warmups=measurement.candidate_feedback_warmups
            if measurement is not None
            else 0,
            candidate_feedback_measurements=measurement.candidate_feedback_measurements
            if measurement is not None
            else 0,
            execution_mode=(
                "execution_feedback"
                if measurement.candidate_feedback_measurements
                else "final_only"
            )
            if measurement is not None
            else "plan_only",
            turn_budget=TurnBudget(
                total_model_turns=maximum_model_turns,
                maximum_inspection_turns=inspection_turn_limit,
                reserved_final_turns=reserved_decision_turns,
                reserved_for=DecisionTurns(candidate_evaluations=candidate_attempts),
            ),
        )
        if context_length is not None and completion_reserve is not None:
            observation.context_budget = ContextBudget(
                maximum_tokens=context_length,
                reserved_for_next_completion=completion_reserve,
            )
        return cls(
            maximum_model_turns=maximum_model_turns,
            inspection_turn_limit=inspection_turn_limit,
            observation=observation,
            tools=tools,
        )

    def initial_messages(self) -> list[Message]:
        return [
            Message(
                role=MessageRole.SYSTEM, content=system_prompt(self.candidate_attempts)
            ),
            Message(
                role=MessageRole.USER,
                content=json.dumps(
                    self.observation.model_dump(mode="json"), sort_keys=True
                ),
            ),
        ]

    def available_tools(self, turn: int, candidate_count: int) -> list[ToolDefinition]:
        names = self.available_tool_names(turn, candidate_count)
        return [tool for tool in self.tools if tool.function.name in names]

    def available_tool_names(self, turn: int, candidate_count: int) -> set[str]:
        if turn == self.maximum_model_turns:
            return (
                {ToolName.FINISH.value}
                if candidate_count
                else {ToolName.KEEP_DEFAULT.value}
            )
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
        names = {tool.function.name for tool in self.tools}
        if not candidate_count:
            names.remove(ToolName.FINISH.value)
        else:
            names.remove(ToolName.KEEP_DEFAULT.value)
        return names

    def budget(self, turn: int) -> RemainingTurnBudget:
        return RemainingTurnBudget(
            current_turn=turn,
            turns_remaining=self.maximum_model_turns - turn,
            unrestricted_turns_remaining=max(0, self.inspection_turn_limit - turn),
            reserved_final_turns=self.reserved_decision_turns,
        )

    @property
    def candidate_attempts(self) -> int:
        return self.observation.turn_budget.reserved_for.candidate_evaluations

    @property
    def reserved_decision_turns(self) -> int:
        return self.observation.turn_budget.reserved_final_turns
