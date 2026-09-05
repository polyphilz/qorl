from scripts.benchmarks.schemas import (
    CanonicalEdge,
    CanonicalNode,
    CanonicalTopology,
    JoinEdge,
    RelationAlias,
)


def test_join_edges_are_undirected_and_deduplicate() -> None:
    left, right = RelationAlias("t"), RelationAlias("mi")
    forward = JoinEdge.between(left, right)
    reverse = JoinEdge.between(right, left)

    assert forward == reverse
    assert len({forward, reverse}) == 1
    assert forward.left == right
    assert forward.right == left


def test_canonical_topology_serializes_labels_as_strings() -> None:
    topology = CanonicalTopology(
        relations=[CanonicalNode("movie_info"), CanonicalNode("title")],
        join_edges=[CanonicalEdge("movie_info=title")],
    )

    assert topology.model_dump_json() == (
        '{"relations":["movie_info","title"],"join_edges":["movie_info=title"]}'
    )
