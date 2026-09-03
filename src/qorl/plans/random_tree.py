from __future__ import annotations

import random
from typing import Any

from qorl.plans.action import TaskCatalog

SWAP_CHILDREN_PROBABILITY = 0.5


def random_join_tree(catalog: TaskCatalog, rng: random.Random) -> str | dict[str, Any]:
    components: list[tuple[set[str], str | dict[str, Any]]] = [
        ({relation}, relation) for relation in sorted(catalog.relations)
    ]
    while len(components) > 1:
        pairs = [
            (left, right)
            for left in range(len(components))
            for right in range(left + 1, len(components))
            if any(
                catalog.adjacency[relation] & components[right][0]
                for relation in components[left][0]
            )
        ]
        left_index, right_index = rng.choice(pairs)
        left_relations, left_tree = components[left_index]
        right_relations, right_tree = components[right_index]
        if rng.random() < SWAP_CHILDREN_PROBABILITY:
            left_tree, right_tree = right_tree, left_tree
        merged = (
            left_relations | right_relations,
            {"left": left_tree, "right": right_tree},
        )
        components.pop(right_index)
        components.pop(left_index)
        components.append(merged)
    return components[0][1]
