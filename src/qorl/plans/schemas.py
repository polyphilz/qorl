from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, TypeGuard, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.json_schema import WithJsonSchema

from qorl.plans.catalog import MIN_JOIN_RELATIONS, TaskCatalog
from qorl.plans.exceptions import ActionError


class JoinMethod(StrEnum):
    HASH = "hash"
    MERGE = "merge"
    NESTLOOP = "nestloop"


class ScanMethod(StrEnum):
    SEQ = "seq"
    INDEX = "index"
    INDEX_ONLY = "index_only"
    BITMAP = "bitmap"


class MemoizeMode(StrEnum):
    AUTO = "auto"
    FORCE = "force"
    FORBID = "forbid"


class RowMode(StrEnum):
    ABSOLUTE = "absolute"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"


class ParallelMode(StrEnum):
    SOFT = "soft"
    HARD = "hard"


AUTO = "auto"
ACTION_SCHEMA_VERSION = 1
MAX_PARALLEL_WORKERS = 2
MAX_COST_SETTING = 1_000_000.0
MAX_EFFECTIVE_CACHE_PAGES = 4_194_304
MIN_ROW_MULTIPLIER = 0.001
MAX_ROW_MULTIPLIER = 1_000.0
MIN_ABSOLUTE_ROW_COUNT = 1.0
MAX_ABSOLUTE_ROW_COUNT = 1_000_000_000_000.0

JOIN_METHODS = {method.value for method in JoinMethod}
SCAN_METHODS = {method.value for method in ScanMethod}

BOOLEAN_SETTINGS = {
    "enable_bitmapscan",
    "enable_gathermerge",
    "enable_group_by_reordering",
    "enable_hashagg",
    "enable_hashjoin",
    "enable_incremental_sort",
    "enable_indexonlyscan",
    "enable_indexscan",
    "enable_material",
    "enable_memoize",
    "enable_mergejoin",
    "enable_nestloop",
    "enable_parallel_hash",
    "enable_self_join_elimination",
    "enable_seqscan",
    "enable_sort",
}

NUMERIC_SETTINGS = {
    "seq_page_cost": (0.0, MAX_COST_SETTING),
    "random_page_cost": (0.0, MAX_COST_SETTING),
    "cpu_tuple_cost": (0.0, MAX_COST_SETTING),
    "cpu_index_tuple_cost": (0.0, MAX_COST_SETTING),
    "cpu_operator_cost": (0.0, MAX_COST_SETTING),
    "parallel_setup_cost": (0.0, MAX_COST_SETTING),
    "parallel_tuple_cost": (0.0, MAX_COST_SETTING),
}

INTEGER_SETTINGS = {
    "effective_cache_size": (1, MAX_EFFECTIVE_CACHE_PAGES),
}

JoinForce = Annotated[
    JoinMethod | Literal["auto"],
    WithJsonSchema({"type": "string", "enum": [method.value for method in JoinMethod]}),
]
JoinForbid = Annotated[
    list[JoinMethod],
    WithJsonSchema(
        {
            "type": "array",
            "items": {"type": "string", "enum": sorted(JOIN_METHODS)},
            "minItems": 1,
            "uniqueItems": True,
        }
    ),
]
MemoizeRequest = Annotated[
    MemoizeMode,
    WithJsonSchema(
        {
            "type": "string",
            "enum": [MemoizeMode.FORCE.value, MemoizeMode.FORBID.value],
        }
    ),
]
ScanForce = Annotated[
    ScanMethod | Literal["auto"],
    WithJsonSchema({"type": "string", "enum": sorted(SCAN_METHODS)}),
]
ScanForbid = Annotated[
    list[ScanMethod],
    WithJsonSchema(
        {
            "type": "array",
            "items": {"type": "string", "enum": sorted(SCAN_METHODS)},
            "minItems": 1,
            "uniqueItems": True,
        }
    ),
]
RowCorrectionMode = Annotated[
    RowMode,
    WithJsonSchema({"type": "string", "enum": [mode.value for mode in RowMode]}),
]
RowCorrectionValue = Annotated[
    int | float,
    WithJsonSchema({"type": "number"}),
]
ParallelRequestMode = Annotated[
    ParallelMode,
    WithJsonSchema({"type": "string", "enum": [mode.value for mode in ParallelMode]}),
]
PlannerBoolean = Annotated[bool | None, WithJsonSchema({"type": "boolean"})]
PlannerCost = Annotated[
    int | float | None,
    Field(ge=0, le=MAX_COST_SETTING, allow_inf_nan=False),
    WithJsonSchema({"type": "number", "minimum": 0.0, "maximum": MAX_COST_SETTING}),
]
EffectiveCachePages = Annotated[
    int | None,
    Field(ge=1, le=MAX_EFFECTIVE_CACHE_PAGES),
    WithJsonSchema(
        {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_EFFECTIVE_CACHE_PAGES,
        }
    ),
]


def _json_object(value: object) -> TypeGuard[dict[str, Any]]:
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) for key in cast(dict[object, object], value))


def _json_list(value: object) -> TypeGuard[list[Any]]:
    return isinstance(value, list)


def _catalog(info: ValidationInfo) -> TaskCatalog:
    context: dict[str, Any] = info.context or {}
    catalog = context.get("catalog")
    if not isinstance(catalog, TaskCatalog):
        raise ValueError("a task catalog is required")
    return catalog


def _enum[EnumT: StrEnum](value: object, enum_type: type[EnumT]) -> EnumT:
    if not isinstance(value, str):
        raise ValueError(f"must be one of {sorted(item.value for item in enum_type)}")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(
            f"must be one of {sorted(item.value for item in enum_type)}"
        ) from error


def _enum_list[EnumT: StrEnum](value: object, enum_type: type[EnumT]) -> list[EnumT]:
    if value is None:
        return []
    if not _json_list(value) or any(not isinstance(item, str) for item in value):
        raise ValueError(
            f"must contain only {sorted(item.value for item in enum_type)}"
        )
    if len(value) != len(set(value)):
        raise ValueError("contains duplicate methods")
    try:
        return [enum_type(item) for item in value]
    except ValueError as error:
        raise ValueError(
            f"must contain only {sorted(item.value for item in enum_type)}"
        ) from error


class ActionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @model_validator(mode="before")
    @classmethod
    def reject_unknown_fields(cls, value: object) -> object:
        if _json_object(value):
            unknown = set(value) - set(cls.model_fields)
            if unknown:
                raise ValueError(f"has unknown fields: {sorted(unknown)}")
        return value


class JoinNode(ActionModel):
    left: str | JoinNode
    right: str | JoinNode


class JoinConstraint(ActionModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        json_schema_extra={
            "anyOf": [
                {"required": ["force"]},
                {"required": ["forbid"]},
                {"required": ["memoize"]},
            ]
        },
    )

    relations: list[str] = Field(
        min_length=MIN_JOIN_RELATIONS,
        json_schema_extra={"uniqueItems": True},
    )
    force: JoinForce = AUTO
    forbid: JoinForbid = Field(
        default_factory=list[JoinMethod],
        json_schema_extra={"minItems": 1, "uniqueItems": True},
    )
    memoize: MemoizeRequest = MemoizeMode.AUTO

    @field_validator("force", mode="before")
    @classmethod
    def validate_force(cls, value: object) -> JoinMethod | Literal["auto"]:
        if value == AUTO:
            return AUTO
        return _enum(value, JoinMethod)

    @field_validator("forbid", mode="before")
    @classmethod
    def validate_forbid(cls, value: object) -> list[JoinMethod]:
        return _enum_list(value, JoinMethod)

    @field_validator("memoize", mode="before")
    @classmethod
    def validate_memoize(cls, value: object) -> MemoizeMode:
        return _enum(value, MemoizeMode)

    @model_validator(mode="after")
    def validate_constraint(self, info: ValidationInfo) -> JoinConstraint:
        catalog = _catalog(info)
        relations = catalog.require_relations(self.relations, "relations")
        forbidden = set(self.forbid)
        if forbidden == set(JoinMethod):
            raise ValueError("forbid cannot disable every join method")
        if self.force in forbidden:
            raise ValueError(f"both forces and forbids {self.force}")
        if self.force == AUTO and not forbidden and self.memoize == MemoizeMode.AUTO:
            raise ValueError("does not request any steering")
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "forbid", sorted(self.forbid, key=str))
        return self


class ScanConstraint(ActionModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        json_schema_extra={
            "anyOf": [
                {"required": ["force"]},
                {"required": ["forbid"]},
            ]
        },
    )

    relation: str
    force: ScanForce = AUTO
    forbid: ScanForbid = Field(
        default_factory=list[ScanMethod],
        json_schema_extra={"minItems": 1, "uniqueItems": True},
    )
    indexes: list[str] = Field(
        default_factory=list,
        description="Use only with force=index, index_only, or bitmap.",
        json_schema_extra={"uniqueItems": True},
    )

    @field_validator("force", mode="before")
    @classmethod
    def validate_force(cls, value: object) -> ScanMethod | Literal["auto"]:
        if value == AUTO:
            return AUTO
        return _enum(value, ScanMethod)

    @field_validator("forbid", mode="before")
    @classmethod
    def validate_forbid(cls, value: object) -> list[ScanMethod]:
        return _enum_list(value, ScanMethod)

    @field_validator("indexes", mode="before")
    @classmethod
    def default_indexes(cls, value: object) -> object:
        return [] if value is None else value

    @model_validator(mode="after")
    def validate_constraint(self, info: ValidationInfo) -> ScanConstraint:
        catalog = _catalog(info)
        relation = catalog.require_relation(self.relation, "relation")
        indexes = catalog.require_indexes(self.indexes, relation, "indexes")
        forbidden = set(self.forbid)
        if forbidden == set(ScanMethod):
            raise ValueError("forbid cannot disable every scan method")
        if self.force in forbidden:
            raise ValueError(f"both forces and forbids {self.force}")
        if indexes and self.force not in {
            ScanMethod.INDEX,
            ScanMethod.INDEX_ONLY,
            ScanMethod.BITMAP,
        }:
            raise ValueError("indexes requires an index-based forced scan")
        if self.force == AUTO and not forbidden:
            raise ValueError("does not request any steering")
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "indexes", indexes)
        object.__setattr__(self, "forbid", sorted(self.forbid, key=str))
        return self


class DisabledIndexes(ActionModel):
    relation: str
    indexes: list[str] = Field(min_length=1, json_schema_extra={"uniqueItems": True})

    @model_validator(mode="after")
    def validate_indexes(self, info: ValidationInfo) -> DisabledIndexes:
        catalog = _catalog(info)
        relation = catalog.require_relation(self.relation, "relation")
        indexes = catalog.require_indexes(self.indexes, relation, "indexes")
        if not indexes:
            raise ValueError("indexes must contain at least one index")
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "indexes", indexes)
        return self


class RowCorrection(ActionModel):
    relations: list[str] = Field(
        min_length=MIN_JOIN_RELATIONS,
        json_schema_extra={"uniqueItems": True},
    )
    mode: RowCorrectionMode
    value: RowCorrectionValue

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, value: object) -> RowMode:
        return _enum(value, RowMode)

    @model_validator(mode="after")
    def validate_correction(self, info: ValidationInfo) -> RowCorrection:
        catalog = _catalog(info)
        relations = catalog.require_relations(self.relations, "relations")
        if isinstance(self.value, bool) or not math.isfinite(self.value):
            raise ValueError("value must be finite and numeric")
        upper = (
            MAX_ROW_MULTIPLIER
            if self.mode == RowMode.MULTIPLY
            else MAX_ABSOLUTE_ROW_COUNT
        )
        lower = (
            MIN_ROW_MULTIPLIER
            if self.mode == RowMode.MULTIPLY
            else MIN_ABSOLUTE_ROW_COUNT
        )
        if not lower <= self.value <= upper:
            raise ValueError(f"value is outside [{lower}, {upper}]")
        object.__setattr__(self, "relations", relations)
        return self


class ParallelRequest(ActionModel):
    relation: str
    workers: int = Field(ge=0, le=MAX_PARALLEL_WORKERS)
    mode: ParallelRequestMode = ParallelMode.SOFT

    @field_validator("workers", mode="before")
    @classmethod
    def validate_workers(cls, value: object) -> object:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= MAX_PARALLEL_WORKERS
        ):
            raise ValueError(
                f"must be an integer from 0 through {MAX_PARALLEL_WORKERS}"
            )
        return value

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, value: object) -> ParallelMode:
        return _enum(value, ParallelMode)

    @model_validator(mode="after")
    def validate_relation(self, info: ValidationInfo) -> ParallelRequest:
        catalog = _catalog(info)
        object.__setattr__(
            self, "relation", catalog.require_relation(self.relation, "relation")
        )
        return self


class PlannerSettings(ActionModel):
    enable_bitmapscan: PlannerBoolean = None
    enable_gathermerge: PlannerBoolean = None
    enable_group_by_reordering: PlannerBoolean = None
    enable_hashagg: PlannerBoolean = None
    enable_hashjoin: PlannerBoolean = None
    enable_incremental_sort: PlannerBoolean = None
    enable_indexonlyscan: PlannerBoolean = None
    enable_indexscan: PlannerBoolean = None
    enable_material: PlannerBoolean = None
    enable_memoize: PlannerBoolean = None
    enable_mergejoin: PlannerBoolean = None
    enable_nestloop: PlannerBoolean = None
    enable_parallel_hash: PlannerBoolean = None
    enable_self_join_elimination: PlannerBoolean = None
    enable_seqscan: PlannerBoolean = None
    enable_sort: PlannerBoolean = None

    cpu_index_tuple_cost: PlannerCost = None
    cpu_operator_cost: PlannerCost = None
    cpu_tuple_cost: PlannerCost = None
    parallel_setup_cost: PlannerCost = None
    parallel_tuple_cost: PlannerCost = None
    random_page_cost: PlannerCost = None
    seq_page_cost: PlannerCost = None

    effective_cache_size: EffectiveCachePages = None

    @model_validator(mode="before")
    @classmethod
    def reject_unknown_or_null(cls, value: object) -> object:
        if value is None:
            return {}
        if not _json_object(value):
            return value
        null = next((name for name, setting in value.items() if setting is None), None)
        if null is not None:
            raise ValueError(f"{null} must not be null")
        return value


class PlanAction(ActionModel):
    version: Literal[1] = Field(description="PlanAction schema version.")
    leading: JoinNode | None = Field(
        default=None,
        description="Binary join order containing every query alias once.",
    )
    joins: list[JoinConstraint] = Field(
        default_factory=list[JoinConstraint],
        description=(
            "Join-method constraints. Each relations value must equal all leaf "
            "aliases beneath one internal node of the candidate plan. When leading "
            "is present, use only sets created by that tree."
        ),
    )
    scans: list[ScanConstraint] = Field(
        default_factory=list[ScanConstraint],
        description="Scan-method and index constraints by query alias.",
    )
    disabled_indexes: list[DisabledIndexes] = Field(
        default_factory=list[DisabledIndexes],
        description="Indexes PostgreSQL must not use by query alias.",
    )
    row_corrections: list[RowCorrection] = Field(
        default_factory=list[RowCorrection],
        description="Planner row-estimate corrections for alias sets.",
    )
    parallel: list[ParallelRequest] = Field(
        default_factory=list[ParallelRequest],
        description="Parallel worker requests by query alias.",
    )
    settings: PlannerSettings = Field(
        default_factory=PlannerSettings,
        description="Safe per-query PostgreSQL planner settings.",
    )

    _sequence_fields: ClassVar[tuple[str, ...]] = (
        "joins",
        "scans",
        "disabled_indexes",
        "row_corrections",
        "parallel",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_nullable_collections(cls, value: object) -> object:
        if not _json_object(value):
            return value
        normalized = dict(value)
        for name in cls._sequence_fields:
            if normalized.get(name) is None:
                normalized[name] = []
        if normalized.get("settings") is None:
            normalized["settings"] = {}
        return normalized

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        if isinstance(value, bool) or value != ACTION_SCHEMA_VERSION:
            raise ValueError(f"must equal {ACTION_SCHEMA_VERSION}")
        return value

    @field_validator("leading", mode="before")
    @classmethod
    def validate_leading(cls, value: object, info: ValidationInfo) -> object:
        if value is None:
            return value
        if isinstance(value, str):
            raise ValueError("must contain at least two relations")

        catalog = _catalog(info)
        leaves: list[str] = []

        def visit(node: object, path: str) -> frozenset[str]:
            if isinstance(node, str):
                relation = catalog.require_relation(node, path)
                leaves.append(relation)
                return frozenset({relation})
            if not _json_object(node):
                raise ValueError(f"{path} must be a relation or join object")
            unknown = set(node) - {"left", "right"}
            if unknown:
                raise ValueError(f"{path} has unknown fields: {sorted(unknown)}")
            for side in ("left", "right"):
                if side not in node:
                    raise ValueError(f"{path}.{side} is required")
            left = visit(node["left"], f"{path}.left")
            right = visit(node["right"], f"{path}.right")
            if not any(catalog.adjacency[relation] & right for relation in left):
                raise ValueError(f"{path} joins disconnected subtrees")
            return left | right

        visit(value, "leading")
        if len(leaves) != len(set(leaves)):
            raise ValueError("leading contains duplicate relations")
        if frozenset(leaves) != catalog.relations:
            raise ValueError("leading must contain every query relation exactly once")
        return value

    @model_validator(mode="after")
    def validate_action(self, info: ValidationInfo) -> PlanAction:
        catalog = _catalog(info)
        self._reject_duplicate_targets(self.joins, "joins")
        self._reject_duplicate_targets(self.row_corrections, "row_corrections")
        self._reject_duplicate_relations(self.scans, "scans")
        self._reject_duplicate_relations(self.disabled_indexes, "disabled_indexes")
        self._reject_duplicate_relations(self.parallel, "parallel")
        self._validate_cross_family_conflicts(catalog)

        object.__setattr__(self, "joins", sorted(self.joins, key=lambda x: x.relations))
        object.__setattr__(self, "scans", sorted(self.scans, key=lambda x: x.relation))
        object.__setattr__(
            self,
            "disabled_indexes",
            sorted(self.disabled_indexes, key=lambda x: x.relation),
        )
        object.__setattr__(
            self,
            "row_corrections",
            sorted(self.row_corrections, key=lambda x: x.relations),
        )
        object.__setattr__(
            self, "parallel", sorted(self.parallel, key=lambda x: x.relation)
        )
        return self

    @staticmethod
    def _reject_duplicate_targets(items: list[Any], field: str) -> None:
        seen: dict[tuple[str, ...], int] = {}
        for index, item in enumerate(items):
            target = tuple(item.relations)
            if target in seen:
                raise ValueError(f"{field}[{index}] duplicates {field}[{seen[target]}]")
            seen[target] = index

    @staticmethod
    def _reject_duplicate_relations(items: list[Any], field: str) -> None:
        seen: dict[str, int] = {}
        for index, item in enumerate(items):
            if item.relation in seen:
                raise ValueError(
                    f"{field}[{index}] duplicates {field}[{seen[item.relation]}]"
                )
            seen[item.relation] = index

    def _validate_cross_family_conflicts(self, catalog: TaskCatalog) -> None:
        settings = self.settings.model_dump(exclude_none=True)
        join_setting = {
            JoinMethod.HASH: "enable_hashjoin",
            JoinMethod.MERGE: "enable_mergejoin",
            JoinMethod.NESTLOOP: "enable_nestloop",
        }
        scan_setting = {
            ScanMethod.SEQ: "enable_seqscan",
            ScanMethod.INDEX: "enable_indexscan",
            ScanMethod.INDEX_ONLY: "enable_indexonlyscan",
            ScanMethod.BITMAP: "enable_bitmapscan",
        }
        for item in self.joins:
            setting = (
                join_setting.get(item.force)
                if isinstance(item.force, JoinMethod)
                else None
            )
            if setting and settings.get(setting) is False:
                raise ValueError(
                    f"action both forces {item.force} and disables {setting}"
                )
        for item in self.scans:
            setting = (
                scan_setting.get(item.force)
                if isinstance(item.force, ScanMethod)
                else None
            )
            if setting and settings.get(setting) is False:
                raise ValueError(
                    f"action both forces {item.force} and disables {setting}"
                )
            disabled = next(
                (
                    set(target.indexes)
                    for target in self.disabled_indexes
                    if target.relation == item.relation
                ),
                set[str](),
            )
            if disabled & set(item.indexes):
                raise ValueError(
                    f"action both forces and disables an index on {item.relation}"
                )
            if (
                item.force
                in {ScanMethod.INDEX, ScanMethod.INDEX_ONLY, ScanMethod.BITMAP}
                and disabled
                and disabled == set(catalog.indexes.get(item.relation, ()))
            ):
                raise ValueError(
                    f"action disables every index for forced scan on {item.relation}"
                )

    @classmethod
    def from_raw(cls, value: object, catalog: TaskCatalog) -> PlanAction:
        try:
            return cls.model_validate(value, context={"catalog": catalog})
        except ValidationError as error:
            raise ActionError(_validation_message(error)) from error

    def to_wire(self) -> dict[str, Any]:
        value = self.model_dump(mode="json", exclude_none=True)
        settings = value.get("settings", {})
        value["settings"] = dict(sorted(settings.items()))
        return {
            name: item
            for name, item in value.items()
            if name == "version" or item not in ([], {})
        }

    def compile(self) -> str:
        action = self.to_wire()
        hints: list[str] = []
        if self.leading is not None:
            hints.append(f"Leading({_render_leading(self.leading)})")

        join_names = {
            JoinMethod.HASH: "HashJoin",
            JoinMethod.MERGE: "MergeJoin",
            JoinMethod.NESTLOOP: "NestLoop",
        }
        for join in self.joins:
            relations = " ".join(join.relations)
            if join.force != AUTO:
                hints.append(f"{join_names[join.force]}({relations})")
            for method in join.forbid:
                hints.append(f"No{join_names[method]}({relations})")
            if join.memoize != MemoizeMode.AUTO:
                prefix = "" if join.memoize == MemoizeMode.FORCE else "No"
                hints.append(f"{prefix}Memoize({relations})")

        scan_names = {
            ScanMethod.SEQ: "SeqScan",
            ScanMethod.INDEX: "IndexScan",
            ScanMethod.INDEX_ONLY: "IndexOnlyScan",
            ScanMethod.BITMAP: "BitmapScan",
        }
        for scan in self.scans:
            arguments = " ".join([scan.relation, *scan.indexes])
            if scan.force != AUTO:
                hints.append(f"{scan_names[scan.force]}({arguments})")
            for method in scan.forbid:
                hints.append(f"No{scan_names[method]}({scan.relation})")

        for item in self.disabled_indexes:
            arguments = " ".join([item.relation, *item.indexes])
            hints.append(f"DisableIndex({arguments})")

        correction_prefix = {
            RowMode.ABSOLUTE: "#",
            RowMode.ADD: "+",
            RowMode.SUBTRACT: "-",
            RowMode.MULTIPLY: "*",
        }
        for item in self.row_corrections:
            relations = " ".join(item.relations)
            correction = correction_prefix[item.mode] + _format_number(item.value)
            hints.append(f"Rows({relations} {correction})")

        for item in self.parallel:
            hints.append(f"Parallel({item.relation} {item.workers} {item.mode})")

        for name, value in action.get("settings", {}).items():
            rendered = (
                ("on" if value else "off")
                if isinstance(value, bool)
                else _format_number(value)
            )
            hints.append(f"Set({name} {rendered})")

        return f"/*+ {' '.join(hints)} */" if hints else ""

    @classmethod
    def tool_schema(cls, relations: list[str] | None = None) -> dict[str, Any]:
        schema = _strip_schema_presentation(cls.model_json_schema())
        expected_settings = (
            BOOLEAN_SETTINGS | NUMERIC_SETTINGS.keys() | INTEGER_SETTINGS.keys()
        )
        if set(PlannerSettings.model_fields) != expected_settings:
            raise RuntimeError("PlannerSettings and the setting allowlist differ")
        if relations is not None:
            _set_relation_enums(schema, relations)
        return schema


def _format_location(location: tuple[int | str, ...]) -> str:
    result = ""
    union_branches = {"bool", "float", "int", "none", "str"}
    for item in (item for item in location if item not in union_branches):
        if isinstance(item, int):
            result += f"[{item}]"
        else:
            result += ("." if result else "") + item
    return result


def _validation_message(error: ValidationError) -> str:
    detail = error.errors(include_url=False)[0]
    location = _format_location(detail["loc"])
    error_type = detail["type"]
    context = detail.get("ctx") or {}

    structural = {
        "missing": "is required",
        "model_type": "must be an object",
        "dict_type": "must be an object",
        "list_type": "must be a list",
        "string_type": "must be a string",
        "int_type": "must be an integer",
        "int_from_float": "must be an integer",
        "bool_type": "must be boolean",
        "float_type": "must be numeric",
        "finite_number": "must be finite and numeric",
        "extra_forbidden": "is not allowed",
    }.get(error_type)
    numeric_field = location.rsplit(".", 1)[-1]
    if (
        location.endswith(".value") or numeric_field in NUMERIC_SETTINGS
    ) and error_type in {
        "float_type",
        "int_from_float",
        "int_type",
    }:
        structural = "must be numeric"
    if structural is not None:
        return f"{location or 'action'} {structural}"

    if error_type == "too_short":
        minimum = context.get("min_length")
        noun = "relations" if location.endswith("relations") else "items"
        if minimum == 1:
            noun = noun.removesuffix("s")
        return f"{location or 'action'} must contain at least {minimum} {noun}"
    if error_type == "greater_than_equal":
        return f"{location or 'action'} must be at least {context.get('ge')}"
    if error_type == "less_than_equal":
        return f"{location or 'action'} must be at most {context.get('le')}"
    if error_type == "literal_error":
        return f"{location or 'action'} has an unsupported value"

    message = detail["msg"].removeprefix("Value error, ")
    if not location:
        if any(
            message.startswith((f"{field}[", f"{field}."))
            for field in PlanAction.model_fields
        ):
            return message
        return message if message.startswith("action ") else f"action {message}"
    if message == location or message.startswith((f"{location} ", f"{location}.")):
        return message
    field = message.split(" ", 1)[0]
    if field in {
        "forbid",
        "force",
        "indexes",
        "memoize",
        "mode",
        "relation",
        "relations",
        "value",
    } or (location == "settings" and field in PlannerSettings.model_fields):
        return f"{location}.{message}"
    return f"{location} {message}"


def _format_number(value: int | float) -> str:
    return format(value, ".15g")


def _render_leading(tree: JoinNode) -> str:
    def render(node: str | JoinNode) -> str:
        if isinstance(node, str):
            return node
        return f"({render(node.left)} {render(node.right)})"

    return render(tree)


def _strip_schema_presentation(value: Any) -> Any:
    if _json_list(value):
        return [_strip_schema_presentation(item) for item in value]
    if not _json_object(value):
        return value
    result = {
        key: _strip_schema_presentation(item)
        for key, item in value.items()
        if key not in {"title", "default"}
    }
    variants = result.get("anyOf")
    if _json_list(variants):
        non_null = [
            item
            for item in variants
            if not (_json_object(item) and item.get("type") == "null")
        ]
        if len(non_null) == 1 and len(non_null) != len(variants):
            result.pop("anyOf")
            if _json_object(non_null[0]):
                result.update(non_null[0])
    return result


def _set_relation_enums(schema: dict[str, Any], relations: list[str]) -> None:
    relation = {"type": "string", "enum": relations}
    definitions = schema["$defs"]
    definitions["JoinConstraint"]["properties"]["relations"]["items"] = relation
    definitions["ScanConstraint"]["properties"]["relation"] = relation
    definitions["DisabledIndexes"]["properties"]["relation"] = relation
    definitions["RowCorrection"]["properties"]["relations"]["items"] = relation
    definitions["ParallelRequest"]["properties"]["relation"] = relation
    for side in ("left", "right"):
        definitions["JoinNode"]["properties"][side]["anyOf"][0] = relation
