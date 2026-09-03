from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypeGuard, cast

from qorl.plans.exceptions import ActionError

IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
MIN_JOIN_RELATIONS = 2


def _string_list(value: object) -> TypeGuard[list[str]]:
    if not isinstance(value, list):
        return False
    return all(isinstance(item, str) for item in cast(list[object], value))


@dataclass(frozen=True)
class TaskCatalog:
    relations: frozenset[str]
    adjacency: dict[str, frozenset[str]]
    indexes: dict[str, frozenset[str]]

    @classmethod
    def from_task(
        cls,
        task: dict[str, Any],
        indexes: dict[str, set[str]] | None = None,
    ) -> TaskCatalog:
        relations = frozenset(item["alias"] for item in task["relations"])
        adjacency: dict[str, set[str]] = {relation: set() for relation in relations}
        for edge in task["join_edges"]:
            left, right = edge.split("=", 1)
            left_alias = left.split(":", 1)[0]
            right_alias = right.split(":", 1)[0]
            adjacency[left_alias].add(right_alias)
            adjacency[right_alias].add(left_alias)
        supplied = indexes or {}
        return cls(
            relations=relations,
            adjacency={key: frozenset(value) for key, value in adjacency.items()},
            indexes={key: frozenset(value) for key, value in supplied.items()},
        )

    def require_relation(self, value: object, label: str) -> str:
        if not isinstance(value, str) or value not in self.relations:
            raise ActionError(f"{label} is not a relation in this query")
        return value

    def require_relations(self, values: object, label: str) -> list[str]:
        if not _string_list(values) or len(values) < MIN_JOIN_RELATIONS:
            raise ActionError(f"{label} must contain at least two relations")
        if len(values) != len(set(values)):
            raise ActionError(f"{label} contains duplicate relations")
        unknown = set(values) - self.relations
        if unknown:
            raise ActionError(f"{label} contains unknown relations: {sorted(unknown)}")
        selected = set(values)
        visited: set[str] = set()
        frontier = [values[0]]
        while frontier:
            relation = frontier.pop()
            if relation in visited:
                continue
            visited.add(relation)
            frontier.extend((self.adjacency[relation] & selected) - visited)
        if visited != selected:
            raise ActionError(f"{label} is not connected in the query join graph")
        return sorted(values)

    def require_indexes(self, value: object, relation: str, label: str) -> list[str]:
        if value is None:
            indexes: list[str] = []
        elif _string_list(value):
            indexes = value
        else:
            raise ActionError(f"{label} must be a list")
        if any(not IDENTIFIER.fullmatch(index) for index in indexes):
            raise ActionError(f"{label} contains an invalid index name")
        if len(indexes) != len(set(indexes)):
            raise ActionError(f"{label} contains duplicate indexes")
        unknown = set(indexes) - self.indexes.get(relation, frozenset())
        if unknown:
            raise ActionError(f"{label} contains unknown indexes: {sorted(unknown)}")
        return sorted(indexes)
