"""SQL relation identities and their alias-independent topology representation."""

from dataclasses import dataclass
from pathlib import Path
from typing import NewType

from pydantic import BaseModel, ConfigDict

RelationAlias = NewType("RelationAlias", str)
TableName = NewType("TableName", str)
CanonicalNode = NewType("CanonicalNode", str)
CanonicalEdge = NewType("CanonicalEdge", str)
TopologyHash = NewType("TopologyHash", str)

type TablesByAlias = dict[RelationAlias, TableName]
type AliasesByTable = dict[TableName, list[RelationAlias]]
type AliasNeighbors = dict[RelationAlias, set[RelationAlias]]
type AliasOrdering = tuple[RelationAlias, ...]
type CanonicalLabels = dict[RelationAlias, CanonicalNode]
type TopologyGroups = dict[TopologyHash, list[Path]]


@dataclass(frozen=True)
class JoinEdge:
    """An undirected connection between two relation aliases."""

    left: RelationAlias
    right: RelationAlias

    @classmethod
    def between(cls, left: RelationAlias, right: RelationAlias) -> "JoinEdge":
        return cls(min(left, right), max(left, right))


@dataclass(frozen=True)
class JoinTopology:
    tables_by_alias: TablesByAlias
    edges: set[JoinEdge]


@dataclass(frozen=True)
class TableInstances:
    """The aliases referring to one physical table within a query."""

    table: TableName
    aliases: AliasOrdering


class CanonicalTopology(BaseModel):
    """Sorted node and edge labels serialized as the topology hash input."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    relations: list[CanonicalNode]
    join_edges: list[CanonicalEdge]
