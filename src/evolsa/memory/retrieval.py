"""Bounded operation-triggered retrieval over structured experience."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from evolsa.core import ExperienceCard, SearchNode
from evolsa.operators.runtime import operation_family_for

from .store import ExperienceStore


class OperationConditionedRetriever:
    """Apply the bounded default retrieval policy from Frontis Table 8."""

    def __init__(
        self,
        store: ExperienceStore,
        *,
        improve_ancestors: int = 3,
        improve_siblings: int = 3,
        crossover_ancestors: int = 2,
        crossover_siblings: int = 2,
        debug_related: int = 3,
    ) -> None:
        limits = (
            improve_ancestors,
            improve_siblings,
            crossover_ancestors,
            crossover_siblings,
            debug_related,
        )
        if any(limit < 0 for limit in limits):
            raise ValueError("retrieval limits must be non-negative")
        self.store = store
        self.improve_ancestors = improve_ancestors
        self.improve_siblings = improve_siblings
        self.crossover_ancestors = crossover_ancestors
        self.crossover_siblings = crossover_siblings
        self.debug_related = debug_related

    def _resolve(self, parents: Sequence[str | SearchNode]) -> tuple[SearchNode, ...]:
        resolved: list[SearchNode] = []
        for parent in parents:
            node_id = parent.node_id if isinstance(parent, SearchNode) else str(parent)
            resolved.append(self.store.get_node(node_id))
        return tuple(resolved)

    def _cards(self, nodes: Sequence[SearchNode]) -> tuple[ExperienceCard, ...]:
        return tuple(self.store.get_card(node.node_id) for node in nodes)

    def _top_siblings(self, node_id: str, limit: int) -> tuple[SearchNode, ...]:
        utilities = self.store.current_selection_utilities()
        siblings = self.store.siblings(node_id)
        return tuple(
            sorted(
                siblings,
                key=lambda node: (
                    utilities[node.node_id],
                    self.store.get_card(node.node_id).step_id,
                    node.node_id,
                ),
                reverse=True,
            )[:limit]
        )

    def retrieve(
        self,
        operator: Any,
        parents: Sequence[str | SearchNode] = (),
    ) -> Mapping[str, Any]:
        operation = operation_family_for(operator)
        if operation == "draft":
            # Deliberately omit the board as well: Draft starts an independent branch.
            return {"schema_version": "evolsa.experience-view/v1", "operation": "draft", "memory": ()}
        resolved = self._resolve(parents)
        if operation == "improve":
            if len(resolved) != 1:
                raise ValueError("Improve retrieval requires exactly one selected parent")
            parent = resolved[0]
            ancestors = self.store.ancestors(parent.node_id, self.improve_ancestors)
            siblings = self._top_siblings(parent.node_id, self.improve_siblings)
            return {
                "schema_version": "evolsa.experience-view/v1",
                "operation": "improve",
                "parent": self.store.get_card(parent.node_id),
                "ancestors": self._cards(ancestors),
                "siblings": self._cards(siblings),
                "board": self.store.board,
            }
        if operation == "crossover":
            if len(resolved) != 2 or resolved[0].node_id == resolved[1].node_id:
                raise ValueError("Crossover retrieval requires two distinct parents")
            branches = []
            for parent in resolved:
                ancestors = self.store.ancestors(parent.node_id, self.crossover_ancestors)
                siblings = self._top_siblings(parent.node_id, self.crossover_siblings)
                branches.append(
                    {
                        "parent": self.store.get_card(parent.node_id),
                        "ancestors": self._cards(ancestors),
                        "siblings": self._cards(siblings),
                    }
                )
            board = self.store.board
            families = (resolved[0].method_family, resolved[1].method_family)
            return {
                "schema_version": "evolsa.experience-view/v1",
                "operation": "crossover",
                "branches": tuple(branches),
                "family_stats": {
                    family: board.method_family_stats.get(family, {}) for family in dict.fromkeys(families)
                },
                "repeated_errors": board.repeated_errors,
                "complementarity": {
                    "families": families,
                    "different_method_families": families[0] != families[1],
                },
            }
        if len(resolved) != 1:
            raise ValueError("Debug retrieval requires exactly one current buggy node")
        current = resolved[0]
        current_card = self.store.get_card(current.node_id)
        prior_cards = [
            card
            for card in self.store.cards.values()
            if card.step_id < current_card.step_id and card.node_id != current.node_id
        ]
        same_error = (
            [card for card in prior_cards if card.error_signature == current_card.error_signature]
            if current_card.error_signature
            else []
        )
        same_error.sort(key=lambda card: card.step_id, reverse=True)
        selected = same_error[: self.debug_related]
        selected_ids = {card.node_id for card in selected}
        if len(selected) < self.debug_related:
            fallback = sorted(prior_cards, key=lambda card: card.step_id, reverse=True)
            selected.extend(
                card
                for card in fallback
                if card.node_id not in selected_ids
            )
            selected = selected[: self.debug_related]
        board = self.store.board
        return {
            "schema_version": "evolsa.experience-view/v1",
            "operation": "debug",
            "error_signature": current_card.error_signature,
            "repeated_error_count": board.repeated_errors.get(current_card.error_signature, 0),
            "current": current_card,
            "related": tuple(selected),
        }

    retrieve_for = retrieve
