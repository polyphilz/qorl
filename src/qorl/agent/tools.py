"""Six query-scoped tool contracts, using the execution argument schemas."""

from pydantic import TypeAdapter

from qorl.agent.tool_schemas import (
    ColumnArguments,
    PlanArguments,
    RelationArguments,
    ToolArguments,
)
from qorl.agent.types import ToolName
from qorl.model.schemas import JsonObject, ToolDefinition, ToolFunction
from qorl.plans.schemas import PlanAction

OBJECT: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)


def function(
    name: ToolName, description: str, parameters: JsonObject
) -> ToolDefinition:
    return ToolDefinition(
        function=ToolFunction(
            name=name.value, description=description, parameters=parameters
        )
    )


def agent_tools(
    relations: list[str],
    *,
    execution_feedback: bool,
) -> list[ToolDefinition]:
    def relation_schema(arguments: type[RelationArguments]) -> JsonObject:
        schema = arguments.model_json_schema()
        schema["properties"]["relation"]["enum"] = relations
        return OBJECT.validate_python(schema)

    action = PlanAction.tool_schema(relations)
    definitions = action.pop("$defs")
    evaluation_description = "Submit one self-contained PlanAction for plain-EXPLAIN validation and return validation diagnostics."
    if execution_feedback:
        evaluation_description += " For valid candidates, also return available execution observations, including timings and observed plan diagnostics."
    return [
        function(
            ToolName.INSPECT_RELATION,
            "Inspect a query alias: physical table, columns/types/nullability, index definitions, estimated rows and bytes. Extended-statistics objects describe columns/kinds, not measured dependencies or frequencies. One inspection turn; omitted metadata is labelled.",
            relation_schema(RelationArguments),
        ),
        function(
            ToolName.GET_COLUMN_STATS,
            "Inspect 1-8 columns in one turn. Negative n_distinct is a fraction of estimated table rows; correlation describes physical heap order. Common values/frequencies retain their first 16 aligned entries. Histograms retain up to 16 evenly spaced original bounds spanning both endpoints, with original zero-based positions. Oversized selected values are explicitly omitted. Missing columns and missing statistics are distinct.",
            relation_schema(ColumnArguments),
        ),
        function(
            ToolName.GET_PLAN,
            "Inspect default estimates or an issued candidate's stored plan. If candidate execution evidence exists, show its last measured sample (or last warmup), including sourced reuse, estimated versus observed rows, loops, buffers, spills and workers where present. Costs are planner units, not milliseconds. At most 32 nodes with complete leaf aliases; use node_id for omitted subtrees. Never reruns SQL; default inspection stays estimate-only.",
            OBJECT.validate_python(PlanArguments.model_json_schema()),
        ),
        function(
            ToolName.EVALUATE_CANDIDATE,
            evaluation_description,
            OBJECT.validate_python(
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"action": action},
                    "required": ["action"],
                    "$defs": definitions,
                }
            ),
        ),
        function(
            ToolName.KEEP_DEFAULT,
            "End the rollout and keep PostgreSQL's default. Available only before submitting a candidate.",
            OBJECT.validate_python(ToolArguments.model_json_schema()),
        ),
        function(
            ToolName.FINISH,
            "End the search. The current one-candidate measured workflow finalizes its sole eligible candidate.",
            OBJECT.validate_python(ToolArguments.model_json_schema()),
        ),
    ]
