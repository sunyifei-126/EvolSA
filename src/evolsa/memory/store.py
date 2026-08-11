"""Frontis-style deterministic Experience Card and Experience Board storage."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from evolsa.core import (
    ExecutionStatus,
    ExperienceBoard,
    ExperienceCard,
    MetricDirection,
    SearchNode,
)


_SPACE = re.compile(r"\s+")


def _successful(node: SearchNode) -> bool:
    return node.score.valid and node.observation.status is ExecutionStatus.SUCCESS


def _error_signature(node: SearchNode) -> str:
    if node.observation.error_signature:
        return node.observation.error_signature.strip()
    if node.observation.status is ExecutionStatus.SUCCESS:
        return ""
    detail = node.observation.logs or node.observation.output
    detail = _SPACE.sub(" ", detail.strip().splitlines()[0] if detail.strip() else "")
    return f"{node.observation.status.value}:{detail[:160]}".rstrip(":")


def _minmax_quality(nodes: Sequence[SearchNode]) -> Mapping[str, float]:
    if not nodes:
        return {}
    directions = {node.score.direction for node in nodes}
    if len(directions) != 1:
        raise ValueError("all nodes in one experience store must use the same metric direction")
    low = min(node.score.primary for node in nodes)
    high = max(node.score.primary for node in nodes)
    if high == low:
        return {node.node_id: 0.5 for node in nodes}
    direction = nodes[0].score.direction
    if direction is MetricDirection.HIGHER_IS_BETTER:
        return {node.node_id: (node.score.primary - low) / (high - low) for node in nodes}
    return {node.node_id: (high - node.score.primary) / (high - low) for node in nodes}


class ExperienceStore:
    """Append-only node experience with deterministic global aggregation.

    Cards retain the search signals observed when their node was appended.
    The board is recomputed from the complete current population, matching the
    node-level versus task-global distinction in Frontis Tables 6 and 7.
    """

    def __init__(self, nodes: Iterable[SearchNode] = ()) -> None:
        self._nodes: dict[str, SearchNode] = {}
        self._cards: dict[str, ExperienceCard] = {}
        self._steps: dict[str, int] = {}
        self._parent_selection_weights: dict[str, float] = {}
        for node in nodes:
            self.add(node)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._nodes

    @property
    def nodes(self) -> Mapping[str, SearchNode]:
        return MappingProxyType(self._nodes)

    @property
    def cards(self) -> Mapping[str, ExperienceCard]:
        return MappingProxyType(self._cards)

    def get_node(self, node_id: str) -> SearchNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown experience node: {node_id}") from exc

    def get_card(self, node_id: str) -> ExperienceCard:
        try:
            return self._cards[node_id]
        except KeyError as exc:
            raise KeyError(f"unknown experience card: {node_id}") from exc

    def add(self, node: SearchNode) -> ExperienceCard:
        """Create and append one immutable experience card from ``SearchNode``."""

        if node.node_id in self._nodes:
            raise ValueError(f"duplicate experience node: {node.node_id}")
        missing = [parent_id for parent_id in node.parent_ids if parent_id not in self._nodes]
        if missing:
            raise ValueError(f"parents must be recorded before their child: {missing}")
        if self._nodes:
            direction = next(iter(self._nodes.values())).score.direction
            if node.score.direction is not direction:
                raise ValueError("all nodes in one experience store must use the same metric direction")

        prior_nodes = tuple(self._nodes.values())
        family_count_before = sum(item.method_family == node.method_family for item in prior_nodes)
        parent_utilities = [self._nodes[parent_id].score.utility for parent_id in node.parent_ids]
        delta = node.score.utility - max(parent_utilities) if parent_utilities else 0.0
        novelty = 1.0 / math.sqrt(1.0 + family_count_before)

        rank = 1 + sum(
            _successful(item) and item.score.utility > node.score.utility for item in prior_nodes
        )
        previous_best = max(
            (item.score.utility for item in prior_nodes if _successful(item)),
            default=-math.inf,
        )
        current_best = _successful(node) and node.score.utility > previous_best

        population = (*prior_nodes, node)
        quality = _minmax_quality(population)
        raw_progress: dict[str, float] = {}
        for candidate in population:
            candidate_parents = [
                self._nodes[parent_id].score.utility
                for parent_id in candidate.parent_ids
                if parent_id in self._nodes
            ]
            raw_progress[candidate.node_id] = (
                max(0.0, candidate.score.utility - max(candidate_parents))
                if candidate_parents
                else 0.0
            )
        max_progress = max(raw_progress.values(), default=0.0)
        normalized_progress = raw_progress[node.node_id] / max_progress if max_progress > 0 else 0.0
        selection_utility = quality[node.node_id] + normalized_progress + novelty

        metadata = node.artifact.metadata
        change_tags_value = metadata.get("change_tags", ())
        if isinstance(change_tags_value, str):
            change_tags = (change_tags_value,)
        else:
            change_tags = tuple(str(tag) for tag in change_tags_value)
        card = ExperienceCard(
            node_id=node.node_id,
            step_id=len(self._nodes),
            operator_id=node.operator_id,
            parent_node_ids=node.parent_ids,
            generation_id=node.generation,
            score=node.score.primary,
            utility=node.score.utility,
            status=node.observation.status,
            error_signature=_error_signature(node),
            runtime_seconds=node.observation.runtime_seconds,
            token_count=node.observation.token_usage,
            cost=node.observation.cost,
            method_family=node.method_family,
            family_count_before=family_count_before,
            delta_vs_parent=delta,
            novelty_score=novelty,
            rank=rank,
            current_best=current_best,
            selection_utility=selection_utility,
            plan=node.plan,
            analysis=node.observation.logs,
            method_overview=str(metadata.get("method_overview", "")),
            parent_comparison_experience=str(metadata.get("parent_comparison_experience", "")),
            change_tags=change_tags,
            valid=node.score.valid,
            validity_reason=node.score.reason,
            guardrails=dict(node.score.guardrails),
            artifact_id=node.artifact.artifact_id,
            task_id=node.task_id,
            split=node.split,
            seed=node.seed,
            evaluator_id=node.evaluator_id,
            model_id=node.model_id,
            budget=node.budget,
        )
        self._nodes[node.node_id] = node
        self._cards[node.node_id] = card
        self._steps[node.node_id] = card.step_id
        return card

    add_node = add
    record = add

    def extend(self, nodes: Iterable[SearchNode]) -> tuple[ExperienceCard, ...]:
        return tuple(self.add(node) for node in nodes)

    def set_parent_selection_weights(self, weights: Mapping[str, float]) -> None:
        unknown = set(weights) - set(self._nodes)
        if unknown:
            raise ValueError(f"selection weights reference unknown nodes: {sorted(unknown)}")
        if any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("selection weights must be finite and non-negative")
        self._parent_selection_weights = dict(weights)

    @property
    def board(self) -> ExperienceBoard:
        return self.build_board()

    @property
    def experience_board(self) -> ExperienceBoard:
        return self.build_board()

    def build_board(self) -> ExperienceBoard:
        """Aggregate all cards into a deterministic Table 7-style board."""

        if not self._nodes:
            return ExperienceBoard(
                num_nodes=0,
                best_node_id="",
                best_score=0.0,
                current_best_family="",
                method_family_stats={},
                family_best_nodes={},
                underexplored_families=(),
                score_history=(),
                recent_delta_trend=0.0,
                repeated_errors={},
                status_count={},
                operator_counts={},
                parent_graph={},
                novelty_by_node={},
                rank_by_node={},
                runtime_stats={
                    "count": 0.0,
                    "total_seconds": 0.0,
                    "mean_seconds": 0.0,
                    "max_seconds": 0.0,
                    "total_tokens": 0.0,
                    "total_cost": 0.0,
                },
                parent_selection_weights={},
            )

        nodes = tuple(self._nodes.values())
        cards = tuple(self._cards.values())
        valid_nodes = [node for node in nodes if _successful(node)]
        best_node = max(valid_nodes, key=lambda item: (item.score.utility, -self._steps[item.node_id])) if valid_nodes else None

        by_family: dict[str, list[SearchNode]] = defaultdict(list)
        for node in nodes:
            by_family[node.method_family].append(node)
        family_stats: dict[str, Mapping[str, float | int | str]] = {}
        family_best_nodes: dict[str, str] = {}
        for family in sorted(by_family):
            members = by_family[family]
            valid = [member for member in members if _successful(member)]
            failures = len(members) - len(valid)
            family_best = (
                max(valid, key=lambda item: (item.score.utility, -self._steps[item.node_id]))
                if valid
                else max(members, key=lambda item: (item.score.utility, -self._steps[item.node_id]))
            )
            family_best_nodes[family] = family_best.node_id
            family_stats[family] = {
                "count": len(members),
                "valid_count": len(valid),
                "failure_count": failures,
                "failure_rate": failures / len(members),
                "best_score": family_best.score.primary,
                "best_node_id": family_best.node_id,
            }
        minimum_family_count = min(len(members) for members in by_family.values())
        underexplored = tuple(
            family for family in sorted(by_family) if len(by_family[family]) == minimum_family_count
        )

        error_counts = Counter(card.error_signature for card in cards if card.error_signature)
        repeated_errors = dict(sorted(error_counts.items()))
        status_count = Counter(card.status.value for card in cards)
        operator_counts = Counter(card.operator_id for card in cards)
        current_ranks = {
            node.node_id: 1
            + sum(_successful(other) and other.score.utility > node.score.utility for other in nodes)
            for node in nodes
        }
        recent_cards = cards[-5:]
        total_runtime = sum(card.runtime_seconds for card in cards)
        return ExperienceBoard(
            num_nodes=len(nodes),
            best_node_id=best_node.node_id if best_node else "",
            best_score=best_node.score.primary if best_node else 0.0,
            current_best_family=best_node.method_family if best_node else "",
            method_family_stats=family_stats,
            family_best_nodes=family_best_nodes,
            underexplored_families=underexplored,
            score_history=tuple(card.score for card in cards),
            recent_delta_trend=(
                sum(card.delta_vs_parent for card in recent_cards) / len(recent_cards)
                if recent_cards
                else 0.0
            ),
            repeated_errors=repeated_errors,
            status_count=dict(sorted(status_count.items())),
            operator_counts=dict(sorted(operator_counts.items())),
            parent_graph={node.node_id: node.parent_ids for node in nodes},
            novelty_by_node={card.node_id: card.novelty_score for card in cards},
            rank_by_node=current_ranks,
            runtime_stats={
                "count": float(len(cards)),
                "total_seconds": total_runtime,
                "mean_seconds": total_runtime / len(cards),
                "max_seconds": max(card.runtime_seconds for card in cards),
                "total_tokens": float(sum(card.token_count for card in cards)),
                "total_cost": sum(card.cost for card in cards),
            },
            parent_selection_weights=dict(self._parent_selection_weights),
        )

    def ancestors(self, node_id: str, limit: int | None = None) -> tuple[SearchNode, ...]:
        self.get_node(node_id)
        seen: set[str] = set()
        pending = list(self._nodes[node_id].parent_ids)
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(self._nodes[current].parent_ids)
        ordered = sorted((self._nodes[item] for item in seen), key=lambda item: self._steps[item.node_id], reverse=True)
        return tuple(ordered if limit is None else ordered[:limit])

    def siblings(self, node_id: str) -> tuple[SearchNode, ...]:
        node = self.get_node(node_id)
        parent_ids = set(node.parent_ids)
        if not parent_ids:
            return ()
        return tuple(
            candidate
            for candidate in self._nodes.values()
            if candidate.node_id != node_id and parent_ids.intersection(candidate.parent_ids)
        )

    def current_selection_utilities(self) -> Mapping[str, float]:
        """Return current equal-weight quality/progress/novelty utilities."""

        nodes = tuple(self._nodes.values())
        quality = _minmax_quality(nodes)
        raw_progress = {card.node_id: max(0.0, card.delta_vs_parent) for card in self._cards.values()}
        max_progress = max(raw_progress.values(), default=0.0)
        return {
            node.node_id: quality[node.node_id]
            + (raw_progress[node.node_id] / max_progress if max_progress > 0 else 0.0)
            + self._cards[node.node_id].novelty_score
            for node in nodes
        }

    def retrieve(self, operator: Any, parents: Sequence[str | SearchNode] = ()) -> Mapping[str, Any]:
        """Convenience facade over :class:`OperationConditionedRetriever`."""

        from .retrieval import OperationConditionedRetriever

        return OperationConditionedRetriever(self).retrieve(operator, parents)
