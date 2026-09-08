"""Argument contracts shared by advertised tools and execution."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic.json_schema import SkipJsonSchema

from qorl.model.schemas import JsonObject, JsonValue

MAX_COLUMNS = 8
Identifier = Annotated[str, Field(pattern=r"^[a-z_][a-z0-9_]*$", max_length=63)]


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RelationArguments(ToolArguments):
    relation: Identifier


class ColumnArguments(RelationArguments):
    columns: list[Identifier] = Field(
        min_length=1, max_length=MAX_COLUMNS, json_schema_extra={"uniqueItems": True}
    )

    @field_validator("columns")
    @classmethod
    def unique_columns(cls, columns: list[str]) -> list[str]:
        if len(set(columns)) != len(columns):
            raise ValueError("columns must not contain duplicates")
        return columns


class PlanArguments(ToolArguments):
    candidate_id: str = Field(min_length=1, max_length=64)
    node_id: str = Field(default="0", pattern=r"^0(?:\.[0-9]+)*$", max_length=128)


class CandidateArguments(ToolArguments):
    action: JsonValue


def omit_selection_default(schema: JsonObject) -> None:
    # Omission permits automatic selection; an explicit null is not a candidate ID.
    schema.pop("default", None)


class FinishArguments(ToolArguments):
    selected_candidate_id: (
        Annotated[str, Field(min_length=1, max_length=64)] | SkipJsonSchema[None]
    ) = Field(
        default=None,
        json_schema_extra=omit_selection_default,
    )

    @field_validator("selected_candidate_id")
    @classmethod
    def supplied_id_is_not_null(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("must name an eligible candidate when supplied")
        return value


def argument_errors(error: ValidationError) -> list[str]:
    """Field-first diagnostics without Python types or submitted-value dumps."""
    messages: list[str] = []
    for detail in error.errors(include_url=False, include_input=False)[:8]:
        path = ".".join(str(part)[:64] for part in detail["loc"]) or "arguments"
        kind = detail["type"]
        if kind == "missing":
            message = "required"
        elif kind == "extra_forbidden":
            message = "not allowed"
        elif kind == "model_type":
            message = "must be an object"
        elif kind == "string_pattern_mismatch":
            message = "must match the advertised format"
        else:
            message = detail["msg"].removeprefix("Value error, ")
        messages.append(f"{path}: {message}"[:256])
    return messages
