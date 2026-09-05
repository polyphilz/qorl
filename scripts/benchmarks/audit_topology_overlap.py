"""Compare CEB and JOB join topologies, ignoring aliases, columns and filters."""

from __future__ import annotations

import itertools
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from qorl.util.hashing import sha256_bytes
from scripts.benchmarks.schemas import (
    AliasesByTable,
    AliasNeighbors,
    AliasOrdering,
    CanonicalEdge,
    CanonicalLabels,
    CanonicalNode,
    CanonicalTopology,
    JoinEdge,
    JoinTopology,
    RelationAlias,
    TableInstances,
    TableName,
    TablesByAlias,
    TopologyGroups,
    TopologyHash,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
JOB_SQL_DIRECTORY = REPOSITORY_ROOT / "benchmarks/job/queries"
CEB_SQL_DIRECTORY = REPOSITORY_ROOT / "benchmarks/ceb/queries"
TABLE_TERM = re.compile(
    r"^\s*(?P<table>[a-z_][a-z0-9_]*)\s+(?:AS\s+)?"
    r"(?P<alias>[a-z_][a-z0-9_]*)\s*$",
    re.IGNORECASE,
)
JOIN_PREDICATE = re.compile(
    r"\b(?P<left>[a-z_][a-z0-9_]*)\.[a-z_][a-z0-9_]*\s*=\s*"
    r"(?P<right>[a-z_][a-z0-9_]*)\.[a-z_][a-z0-9_]*\b",
    re.IGNORECASE,
)
MAX_ALIAS_PERMUTATIONS = 1_000_000
MIN_JOIN_RELATIONS = 2


def topology_sha256(topology: JoinTopology) -> TopologyHash:
    """Hash table instances and adjacency independently of their alias names."""
    aliases_by_table: AliasesByTable = defaultdict(list)
    for alias, table in topology.tables_by_alias.items():
        aliases_by_table[table].append(alias)

    tables = [
        TableInstances(table, tuple(sorted(aliases)))
        for table, aliases in sorted(aliases_by_table.items())
    ]

    permutation_count = math.prod(
        math.factorial(len(table.aliases)) for table in tables
    )
    if permutation_count > MAX_ALIAS_PERMUTATIONS:
        raise RuntimeError(
            f"join topology has too many alias permutations: {permutation_count}"
        )

    alias_orderings: list[Iterator[AliasOrdering]] = [
        itertools.permutations(table.aliases) for table in tables
    ]
    encodings: list[str] = []
    for combination in itertools.product(*alias_orderings):
        labels: CanonicalLabels = {}
        for table, ordering in zip(tables, combination, strict=True):
            for index, alias in enumerate(ordering, start=1):
                labels[alias] = CanonicalNode(
                    table.table if len(ordering) == 1 else f"{table.table}#{index}"
                )
        edges: set[CanonicalEdge] = {
            CanonicalEdge("=".join(sorted((labels[edge.left], labels[edge.right]))))
            for edge in topology.edges
        }
        canonical = CanonicalTopology(
            relations=sorted(labels.values()), join_edges=sorted(edges)
        )
        encodings.append(
            json.dumps(
                canonical.model_dump(),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return TopologyHash(sha256_bytes(min(encodings).encode("utf-8")))


def extract_topology(sql: str, query_name: str) -> JoinTopology:
    """Read the connected equality-join topology of a comma-joined SQL query."""
    from_match = re.search(r"\bFROM\s+(.*?)\s+WHERE\b", sql, re.IGNORECASE | re.DOTALL)
    if not from_match:
        raise RuntimeError(f"cannot find FROM/WHERE clauses in {query_name}")

    tables_by_alias: TablesByAlias = {}
    for term in from_match.group(1).split(","):
        match = TABLE_TERM.fullmatch(term)
        if not match:
            raise RuntimeError(f"cannot parse FROM term in {query_name}: {term!r}")
        alias = RelationAlias(match["alias"].lower())
        if alias in tables_by_alias:
            raise RuntimeError(f"duplicate table alias in {query_name}: {alias}")
        tables_by_alias[alias] = TableName(match["table"].lower())

    edges: set[JoinEdge] = set()
    for match in JOIN_PREDICATE.finditer(sql):
        left = RelationAlias(match["left"].lower())
        right = RelationAlias(match["right"].lower())
        if left in tables_by_alias and right in tables_by_alias and left != right:
            edges.add(JoinEdge.between(left, right))
    if len(tables_by_alias) < MIN_JOIN_RELATIONS or not edges:
        raise RuntimeError(f"query has no usable join topology: {query_name}")

    adjacency: AliasNeighbors = {alias: set() for alias in tables_by_alias}
    for edge in edges:
        adjacency[edge.left].add(edge.right)
        adjacency[edge.right].add(edge.left)
    visited: set[RelationAlias] = set()
    frontier: list[RelationAlias] = [next(iter(tables_by_alias))]
    while frontier:
        alias = frontier.pop()
        if alias not in visited:
            visited.add(alias)
            frontier.extend(adjacency[alias] - visited)
    if visited != set(tables_by_alias):
        raise RuntimeError(
            f"query join topology is disconnected: {query_name} "
            f"unreachable={sorted(set(tables_by_alias) - visited)}"
        )
    return JoinTopology(tables_by_alias, edges)


def read_topologies(directory: Path) -> TopologyGroups:
    """Group SQL files by their alias-independent join topology."""
    paths = sorted(directory.rglob("*.sql"))
    if not paths:
        raise RuntimeError(f"no SQL files found under {directory}")
    topologies: TopologyGroups = defaultdict(list)
    for path in paths:
        topology = extract_topology(path.read_text(encoding="utf-8"), str(path))
        topologies[topology_sha256(topology)].append(path)
    return dict(topologies)


def main() -> None:
    """Compare the checked-in JOB and CEB SQL directories and print any matches."""
    job = read_topologies(JOB_SQL_DIRECTORY)
    ceb = read_topologies(CEB_SQL_DIRECTORY)
    shared = sorted(job.keys() & ceb.keys())
    print(
        f"JOB: {sum(map(len, job.values()))} SQL files, {len(job)} distinct topologies"
    )
    print(
        f"CEB: {sum(map(len, ceb.values()))} SQL files, {len(ceb)} distinct topologies"
    )
    print(f"Shared topologies: {len(shared)}")
    for topology in shared:
        job_paths = ", ".join(
            str(path.relative_to(JOB_SQL_DIRECTORY)) for path in job[topology]
        )
        ceb_paths = ", ".join(
            str(path.relative_to(CEB_SQL_DIRECTORY)) for path in ceb[topology]
        )
        print(f"  JOB: {job_paths}")
        print(f"  CEB: {ceb_paths}")


if __name__ == "__main__":
    main()
