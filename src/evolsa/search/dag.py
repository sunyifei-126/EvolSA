"""A small, deterministic directed acyclic search graph."""

from __future__ import annotations

from collections import defaultdict
from types import MappingProxyType
from typing import Iterable, Mapping

from evolsa.core import ExecutionStatus, SearchNode


class SearchDAG:
    """Append-only DAG with lineage, neighborhood, and incumbent queries."""

    def __init__(self, nodes: Iterable[SearchNode] = ()) -> None:
        self._nodes: dict[str, SearchNode] = {}
        self._children: dict[str, list[str]] = defaultdict(list)
        self._order: dict[str, int] = {}
        for node in nodes:
            self.add(node)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    @property
    def nodes(self) -> Mapping[str, SearchNode]:
        return MappingProxyType(self._nodes)

    def get(self, node_id: str) -> SearchNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown search node: {node_id}") from exc

    def add(self, node: SearchNode) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"duplicate search node: {node.node_id}")
        if len(node.parent_ids) != len(set(node.parent_ids)):
            raise ValueError("a search node cannot repeat a parent")
        missing = [parent_id for parent_id in node.parent_ids if parent_id not in self._nodes]
        if missing:
            raise ValueError(f"parents must exist before child insertion: {missing}")
        if node.parent_ids:
            expected_generation = 1 + max(self._nodes[parent_id].generation for parent_id in node.parent_ids)
            if node.generation != expected_generation:
                raise ValueError(
                    f"node generation {node.generation} does not follow parent generation {expected_generation - 1}"
                )
        elif node.generation != 0:
            raise ValueError("root nodes must have generation zero")
        self._order[node.node_id] = len(self._nodes)
        self._nodes[node.node_id] = node
        self._children.setdefault(node.node_id, [])
        for parent_id in node.parent_ids:
            self._children[parent_id].append(node.node_id)

    add_node = add

    @property
    def roots(self) -> tuple[SearchNode, ...]:
        return tuple(node for node in self._nodes.values() if not node.parent_ids)

    @property
    def leaves(self) -> tuple[SearchNode, ...]:
        return tuple(node for node in self._nodes.values() if not self._children[node.node_id])

    def children(self, node_id: str) -> tuple[SearchNode, ...]:
        self.get(node_id)
        return tuple(self._nodes[child_id] for child_id in self._children[node_id])

    def ancestors(self, node_id: str, limit: int | None = None) -> tuple[SearchNode, ...]:
        self.get(node_id)
        seen: set[str] = set()
        pending = list(self._nodes[node_id].parent_ids)
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(self._nodes[current].parent_ids)
        ordered = sorted((self._nodes[item] for item in seen), key=lambda item: self._order[item.node_id], reverse=True)
        return tuple(ordered if limit is None else ordered[:limit])

    def siblings(self, node_id: str) -> tuple[SearchNode, ...]:
        node = self.get(node_id)
        parents = set(node.parent_ids)
        if not parents:
            return ()
        return tuple(
            candidate
            for candidate in self._nodes.values()
            if candidate.node_id != node_id and parents.intersection(candidate.parent_ids)
        )

    @property
    def eligible_nodes(self) -> tuple[SearchNode, ...]:
        return tuple(
            node
            for node in self._nodes.values()
            if node.score.valid and node.observation.status is ExecutionStatus.SUCCESS
        )

    @property
    def failed_nodes(self) -> tuple[SearchNode, ...]:
        return tuple(
            node
            for node in self._nodes.values()
            if node.observation.status is not ExecutionStatus.SUCCESS
            or bool(node.observation.error_signature)
        )

    @property
    def best_node(self) -> SearchNode | None:
        eligible = self.eligible_nodes
        return (
            max(eligible, key=lambda node: (node.score.utility, -self._order[node.node_id]))
            if eligible
            else None
        )
