"""Validated execution; conversation availability belongs to AgentInterface."""

from pydantic import TypeAdapter, ValidationError

from qorl.agent.feedback import execution_observation
from qorl.agent.presentation import plan_view
from qorl.agent.tool_schemas import (
    CandidateArguments,
    ColumnArguments,
    PlanArguments,
    RelationArguments,
    ToolArguments,
    argument_errors,
)
from qorl.agent.types import AgentEvaluator, ToolName
from qorl.measure.schemas import ToolResultStatus
from qorl.model.schemas import JsonObject, JsonValue
from qorl.postgres.inspection import (
    RelationMetadata,
    column_statistics,
    inspect_relation,
)

OBJECT: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


class AgentEnvironment:
    def __init__(self, evaluator: AgentEvaluator) -> None:
        self.evaluator = evaluator
        self.tables = {item.alias: item.table for item in evaluator.task.relations}
        self.metadata: dict[str, RelationMetadata] = {}

    def table(self, relation: str) -> str:
        if relation not in self.tables:
            raise ValueError("relation must be an alias in this query")
        return self.tables[relation]

    def relation_context(self, alias: str, table: str) -> JsonObject:
        return OBJECT.validate_python(
            {
                "relation": alias,
                "schema": "public",
                "table": table,
                "query_aliases": [
                    key for key, value in self.tables.items() if value == table
                ],
            }
        )

    def get_plan(self, arguments: PlanArguments) -> JsonObject:
        if arguments.candidate_id == "default":
            if self.evaluator.default is None:
                raise RuntimeError("rollout baseline has not been started")
            plan = self.evaluator.default.plain_explain
        else:
            candidate = next(
                (
                    item
                    for item in self.evaluator.candidates
                    if item.candidate_id == arguments.candidate_id
                ),
                None,
            )
            if candidate is None:
                raise ValueError("candidate_id was not issued by the server")
            plan = candidate.plain_explain
            execution = execution_observation(
                candidate,
                self.evaluator.default,
                self.evaluator.candidates,
                node_id=arguments.node_id,
                summary=False,
            )
            if execution is not None:
                if execution.plan is None and plan is not None:
                    execution.plan = plan_view(plan["Plan"], node_id=arguments.node_id)
                return OBJECT.validate_python(execution.model_dump(mode="json"))
        if plan is None:
            raise ValueError("candidate has no PostgreSQL plan")
        return OBJECT.validate_python(
            plan_view(plan["Plan"], node_id=arguments.node_id).model_dump(mode="json")
        )

    def execute(self, name: str, arguments: JsonValue) -> tuple[JsonObject, bool]:
        if name == ToolName.EVALUATE_CANDIDATE:
            if self.evaluator.kept_default:
                return {"error": "rollout already kept the default plan"}, False
            if len(self.evaluator.candidates) >= self.evaluator.max_candidates:
                return {"error": "rollout candidate budget is exhausted"}, False
            # Invalid submissions still consume an attempt through existing validation.
            try:
                action = CandidateArguments.model_validate(arguments).action
            except ValidationError as error:
                rejected = self.evaluator.reject_attempt(
                    arguments, argument_errors(error)
                )
                return OBJECT.validate_python(rejected.feedback()), False
            candidate = self.evaluator.evaluate(action)
            result = OBJECT.validate_python(candidate.feedback())
            execution = execution_observation(
                candidate, self.evaluator.default, self.evaluator.candidates
            )
            if execution is not None:
                result["execution_feedback"] = OBJECT.validate_python(
                    execution.model_dump(mode="json")
                )
            return result, False
        try:
            if name == ToolName.KEEP_DEFAULT:
                ToolArguments.model_validate(arguments)
                if self.evaluator.candidates:
                    raise ValueError(
                        "keep_default must be selected before submitting a candidate"
                    )
                if self.evaluator.kept_default:
                    raise ValueError("rollout already kept the default plan")
                return OBJECT.validate_python(self.evaluator.keep_default()), True
            if name == ToolName.FINISH:
                ToolArguments.model_validate(arguments)
                if not self.evaluator.candidates:
                    raise ValueError(
                        "finish requires a candidate; use keep_default before submitting one"
                    )
                return {"status": ToolResultStatus.FINISHED.value}, True
            if name == ToolName.GET_PLAN:
                return self.get_plan(PlanArguments.model_validate(arguments)), False
            if name == ToolName.INSPECT_RELATION:
                request = RelationArguments.model_validate(arguments)
                table = self.table(request.relation)
                if table not in self.metadata:
                    self.metadata[table] = inspect_relation(
                        self.evaluator.worker.admin_sql, table
                    )
                return self.relation_context(
                    request.relation, table
                ) | OBJECT.validate_python(
                    self.metadata[table].model_dump(mode="json")
                ), False
            if name == ToolName.GET_COLUMN_STATS:
                request = ColumnArguments.model_validate(arguments)
                table = self.table(request.relation)
                result = column_statistics(
                    self.evaluator.worker.admin_sql, table, request.columns
                )
                return self.relation_context(
                    request.relation, table
                ) | OBJECT.validate_python(result.model_dump(mode="json")), False
            raise ValueError("unknown tool")
        except ValidationError as error:
            return {"error": "; ".join(argument_errors(error))}, False
        except ValueError as error:
            return {"error": str(error)[:2048]}, False
