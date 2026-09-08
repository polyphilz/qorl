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


def agent_tools(relations: list[str]) -> list[ToolDefinition]:
    def relation_schema(arguments: type[RelationArguments]) -> JsonObject:
        schema = arguments.model_json_schema()
        schema["properties"]["relation"]["enum"] = relations
        return OBJECT.validate_python(schema)

    action = PlanAction.tool_schema(relations)
    definitions = action.pop("$defs")
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
            "Inspect estimates for default or an issued candidate. Costs are planner units, not milliseconds. Returns at most 32 nodes with complete leaf aliases, conditions, widths, sort and parallel details. Use a returned node_id for a subtree when nodes are omitted. No query execution.",
            OBJECT.validate_python(PlanArguments.model_json_schema()),
        ),
        function(
            ToolName.EVALUATE_CANDIDATE,
            "Submit one self-contained PlanAction for plain-EXPLAIN validation. Candidate execution latency is measured after the conversation in the current final-only workflow.",
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
