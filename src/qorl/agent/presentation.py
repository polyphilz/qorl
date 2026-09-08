"""Bounded plan views distinguish estimates from retained execution observations."""

import json

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from qorl.model.schemas import JsonObject
from qorl.postgres.exceptions import PostgresError

SUMMARY_NODES = 16
DETAIL_NODES = 32
PLAN_BYTES = 32_768
FIELD_BYTES = 2_048
OBJECT: TypeAdapter[JsonObject] = TypeAdapter(JsonObject)
CHILDREN = TypeAdapter(list[JsonObject])
SUMMARY_FIELDS = ("Node Type", "Join Type", "Relation Name", "Alias", "Plan Rows")
DETAIL_FIELDS = (
    *SUMMARY_FIELDS,
    "Startup Cost",
    "Total Cost",
    "Plan Width",
    "Index Name",
    "Scan Direction",
    "Index Cond",
    "Recheck Cond",
    "Filter",
    "Join Filter",
    "Hash Cond",
    "Merge Cond",
    "Sort Key",
    "Presorted Key",
    "Group Key",
    "Strategy",
    "Partial Mode",
    "Parallel Aware",
    "Async Capable",
    "Workers Planned",
    "Parent Relationship",
    "Subplan Name",
    "Inner Unique",
)
OBSERVED_FIELDS = (
    "Actual Rows",
    "Actual Loops",
    "Shared Hit Blocks",
    "Shared Read Blocks",
    "Shared Dirtied Blocks",
    "Shared Written Blocks",
    "Local Hit Blocks",
    "Local Read Blocks",
    "Local Dirtied Blocks",
    "Local Written Blocks",
    "Temp Read Blocks",
    "Temp Written Blocks",
    "Sort Method",
    "Sort Space Used",
    "Sort Space Type",
    "Full-sort Groups",
    "Pre-sorted Groups",
    "Cache Hits",
    "Cache Misses",
    "Cache Evictions",
    "Cache Overflows",
    "HashAgg Batches",
    "Hash Buckets",
    "Original Hash Buckets",
    "Hash Batches",
    "Original Hash Batches",
    "Peak Memory Usage",
    "Disk Usage",
    "Workers Launched",
    "Workers",
)


class PlanNode(BaseModel):
    node_id: str
    parent_id: str | None
    child_ids: list[str]
    leaf_aliases: list[str]
    estimates: JsonObject
    observed: JsonObject | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    omitted_fields: list[str] = Field(default_factory=list)


class PlanView(BaseModel):
    source: str = "plain_explain"
    cost_unit: str = "planner_units_not_milliseconds"
    row_kind: str = "estimated"
    root_node_id: str
    nodes: list[PlanNode]
    omitted_nodes: int


def plan_view(
    document: object,
    *,
    node_id: str = "0",
    summary: bool = False,
    observed: bool = False,
) -> PlanView:
    try:
        root = OBJECT.validate_python(document)
    except ValidationError as error:
        raise PostgresError("invalid stored plan metadata") from error
    pending: list[tuple[str, str | None, JsonObject]] = [("0", None, root)]
    ordered: list[tuple[str, str | None, JsonObject, list[str]]] = []
    leaves: dict[str, set[str]] = {}
    while pending:
        key, parent, plan = pending.pop()
        try:
            children = CHILDREN.validate_python(plan.get("Plans", []))
        except ValidationError as error:
            raise PostgresError("invalid stored plan children") from error
        child_ids = [f"{key}.{index}" for index in range(len(children))]
        ordered.append((key, parent, plan, child_ids))
        pending.extend(
            reversed(list(zip(child_ids, [key] * len(children), children, strict=True)))
        )
    for key, _, plan, children in reversed(ordered):
        alias = plan.get("Alias")
        leaves[key] = ({alias} if isinstance(alias, str) else set()).union(
            *(leaves[child] for child in children)
        )
    if node_id not in leaves:
        raise ValueError("node_id is not present in this plan")
    selected = [
        row for row in ordered if row[0] == node_id or row[0].startswith(node_id + ".")
    ]
    view = PlanView(root_node_id=node_id, nodes=[], omitted_nodes=len(selected))
    if observed:
        view.source = "explain_analyze_timing_off"
        view.row_kind = "estimated_and_observed"
    for key, parent, plan, children in selected[
        : SUMMARY_NODES if summary else DETAIL_NODES
    ]:
        values: JsonObject = {}
        omitted: list[str] = []
        for field in SUMMARY_FIELDS if summary else DETAIL_FIELDS:
            if field not in plan:
                continue
            if len(json.dumps(plan[field]).encode()) > FIELD_BYTES:
                omitted.append(field)
            else:
                values[field] = plan[field]
        node = PlanNode(
            node_id=key,
            parent_id=parent,
            child_ids=children,
            leaf_aliases=sorted(leaves[key]),
            estimates=values,
            omitted_fields=omitted,
        )
        if observed:
            actual: JsonObject = {}
            for field in OBSERVED_FIELDS:
                if field not in plan:
                    continue
                if len(json.dumps(plan[field]).encode()) > FIELD_BYTES:
                    node.omitted_fields.append(field)
                else:
                    actual[field] = plan[field]
            node.observed = actual
        view.nodes.append(node)
        view.omitted_nodes -= 1
        if len(json.dumps(view.model_dump(mode="json")).encode()) > PLAN_BYTES:
            view.nodes.pop()
            view.omitted_nodes += 1
            break
    if not view.nodes:
        raise PostgresError("plan node metadata exceeds the presentation limit")
    return view
