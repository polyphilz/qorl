"""Parse and fingerprint query join structures for data preparation."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from typing import Any, Iterable


TABLE_TERM = re.compile(
    r"^\s*(?P<table>[a-z_][a-z0-9_]*)\s+(?:AS\s+)?"
    r"(?P<alias>[a-z_][a-z0-9_]*)\s*$",
    re.IGNORECASE,
)
JOIN_PREDICATE = re.compile(
    r"\b(?P<left_alias>[a-z_][a-z0-9_]*)\."
    r"(?P<left_column>[a-z_][a-z0-9_]*)\s*=\s*"
    r"(?P<right_alias>[a-z_][a-z0-9_]*)\."
    r"(?P<right_column>[a-z_][a-z0-9_]*)\b",
    re.IGNORECASE,
)
INVENTORY_JOIN_EDGE = re.compile(
    r"^(?P<left_alias>[a-z_][a-z0-9_]*):[a-z_][a-z0-9_]*\."
    r"(?P<left_column>[a-z_][a-z0-9_]*)="
    r"(?P<right_alias>[a-z_][a-z0-9_]*):[a-z_][a-z0-9_]*\."
    r"(?P<right_column>[a-z_][a-z0-9_]*)$"
)

JoinPredicate = tuple[str, str, str, str]


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_graph_sha256(
    aliases: dict[str, str],
    join_predicates: Iterable[JoinPredicate],
    *,
    include_columns: bool,
) -> str:
    join_predicates = tuple(join_predicates)
    aliases_by_table: dict[str, list[str]] = {}
    for alias, table in aliases.items():
        aliases_by_table.setdefault(table, []).append(alias)

    permutation_count = math.prod(
        math.factorial(len(table_aliases))
        for table_aliases in aliases_by_table.values()
    )
    if permutation_count > 1_000_000:
        raise RuntimeError(
            f"join graph has too many alias permutations: {permutation_count}"
        )

    assignment_groups: list[list[dict[str, str]]] = []
    for table, table_aliases in sorted(aliases_by_table.items()):
        table_aliases = sorted(table_aliases)
        assignments: list[dict[str, str]] = []
        for ordering in itertools.permutations(table_aliases):
            assignments.append(
                {
                    alias: table if len(ordering) == 1 else f"{table}#{index}"
                    for index, alias in enumerate(ordering, start=1)
                }
            )
        assignment_groups.append(assignments)

    canonical_encodings: list[str] = []
    for assignment_group in itertools.product(*assignment_groups):
        node_names = {
            alias: node_name
            for assignment in assignment_group
            for alias, node_name in assignment.items()
        }
        canonical_edges: set[str] = set()
        for left_alias, left_column, right_alias, right_column in join_predicates:
            if include_columns:
                left = f"{node_names[left_alias]}.{left_column}"
                right = f"{node_names[right_alias]}.{right_column}"
            else:
                left = node_names[left_alias]
                right = node_names[right_alias]
            first, second = sorted((left, right))
            canonical_edges.add(f"{first}={second}")
        canonical_encodings.append(
            json.dumps(
                {
                    "relations": sorted(node_names.values()),
                    "join_edges": sorted(canonical_edges),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    return sha256_bytes(min(canonical_encodings).encode("utf-8"))


def canonical_join_graph_sha256(
    aliases: dict[str, str],
    join_predicates: Iterable[JoinPredicate],
) -> str:
    """Hash table instances and column-level equality joins, ignoring aliases."""
    return _canonical_graph_sha256(
        aliases,
        join_predicates,
        include_columns=True,
    )


def canonical_join_topology_sha256(
    aliases: dict[str, str],
    join_predicates: Iterable[JoinPredicate],
) -> str:
    """Hash the table-colored relation graph, ignoring aliases and columns."""
    return _canonical_graph_sha256(
        aliases,
        join_predicates,
        include_columns=False,
    )


def extract_join_structure(sql: str, query_name: str) -> dict[str, Any]:
    from_match = re.search(r"\bFROM\s+(.*?)\s+WHERE\b", sql, re.IGNORECASE | re.DOTALL)
    if not from_match:
        raise RuntimeError(f"cannot find FROM/WHERE clauses in {query_name}")

    aliases: dict[str, str] = {}
    for term in from_match.group(1).split(","):
        match = TABLE_TERM.fullmatch(term)
        if not match:
            raise RuntimeError(f"cannot parse FROM term in {query_name}: {term!r}")
        table = match["table"].lower()
        alias = match["alias"].lower()
        if alias in aliases:
            raise RuntimeError(f"duplicate table alias in {query_name}: {alias}")
        aliases[alias] = table

    join_edges: set[str] = set()
    join_predicates: set[JoinPredicate] = set()
    for match in JOIN_PREDICATE.finditer(sql):
        left_alias = match["left_alias"].lower()
        right_alias = match["right_alias"].lower()
        if left_alias not in aliases or right_alias not in aliases:
            continue
        if left_alias == right_alias:
            continue
        left_column = match["left_column"].lower()
        right_column = match["right_column"].lower()
        left = f"{left_alias}:{aliases[left_alias]}.{left_column}"
        right = f"{right_alias}:{aliases[right_alias]}.{right_column}"
        first, second = sorted((left, right))
        join_edges.add(f"{first}={second}")
        first_predicate, second_predicate = sorted(
            (
                (left_alias, left_column),
                (right_alias, right_column),
            )
        )
        join_predicates.add((*first_predicate, *second_predicate))

    relations = [
        {"alias": alias, "table": table}
        for alias, table in sorted(aliases.items())
    ]
    tables = sorted(set(aliases.values()))
    edges = sorted(join_edges)
    if len(relations) < 2 or not edges:
        raise RuntimeError(f"query has no usable join graph: {query_name}")

    adjacency = {alias: set() for alias in aliases}
    for left_alias, _left_column, right_alias, _right_column in join_predicates:
        adjacency[left_alias].add(right_alias)
        adjacency[right_alias].add(left_alias)
    visited: set[str] = set()
    frontier = [next(iter(aliases))]
    while frontier:
        alias = frontier.pop()
        if alias in visited:
            continue
        visited.add(alias)
        frontier.extend(adjacency[alias] - visited)
    if visited != set(aliases):
        raise RuntimeError(
            f"query join graph is disconnected: {query_name} "
            f"unreachable={sorted(set(aliases) - visited)}"
        )

    return {
        "tables": tables,
        "relations": relations,
        "join_edges": edges,
        "table_count": len(tables),
        "relation_count": len(relations),
        "join_predicate_count": len(edges),
        "join_graph_sha256": canonical_join_graph_sha256(
            aliases, join_predicates
        ),
    }


def task_join_fingerprints(task: dict[str, Any]) -> tuple[str, str]:
    aliases = {
        relation["alias"]: relation["table"]
        for relation in task["relations"]
    }
    predicates: set[JoinPredicate] = set()
    for edge in task["join_edges"]:
        match = INVENTORY_JOIN_EDGE.fullmatch(edge)
        if not match:
            raise RuntimeError(f"cannot parse inventory join edge: {edge}")
        first, second = sorted(
            (
                (match["left_alias"], match["left_column"]),
                (match["right_alias"], match["right_column"]),
            )
        )
        predicates.add((*first, *second))
    return (
        canonical_join_graph_sha256(aliases, predicates),
        canonical_join_topology_sha256(aliases, predicates),
    )
