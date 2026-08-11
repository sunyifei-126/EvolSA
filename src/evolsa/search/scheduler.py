"""Direction-aware quality-progress-novelty parent scheduling."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from evolsa.core import MetricDirection, SearchNode
from evolsa.operators.runtime import operation_family_for


@dataclass(frozen=True, slots=True)
class SchedulerWeights:
    quality: float = 1.0
    progress: float = 1.0
    novelty: float = 1.0

    def __post_init__(self) -> None:
        values = (self.quality, self.progress, self.novelty)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("scheduler weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one scheduler weight must be positive")


@dataclass(frozen=True, slots=True)
class NodeSelectionScore:
    node_id: str
    quality: float
    progress: float
    novelty: float
    utility: float
    probability: float


@dataclass(frozen=True, slots=True)
class SelectionTrace:
    operation: str
    candidates: tuple[NodeSelectionScore, ...]
    selected_node_ids: tuple[str, ...]
    replacement: bool


class QualityProgressNoveltyScheduler:
    """Frontis-style non-greedy scheduler with reproducible sampling."""

    def __init__(
        self,
        *,
        weights: SchedulerWeights | None = None,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> None:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero")
        self.weights = weights or SchedulerWeights()
        self.temperature = temperature
        self.seed = seed
        self._rng = random.Random(seed)
        self.last_trace: SelectionTrace | None = None

    def reset(self, seed: int | None = None) -> None:
        """Reset sampling to a reproducible state."""

        if seed is not None:
            self.seed = seed
        self._rng.seed(self.seed)
        self.last_trace = None

    @staticmethod
    def _validate(candidates: Sequence[SearchNode]) -> None:
        if not candidates:
            raise ValueError("at least one parent candidate is required")
        ids = [node.node_id for node in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("parent candidates must have unique node IDs")
        if len({node.score.direction for node in candidates}) != 1:
            raise ValueError("candidate metrics must have one consistent direction")

    def score(
        self,
        candidates: Sequence[SearchNode],
        *,
        population: Sequence[SearchNode] | None = None,
        family_counts: Mapping[str, int] | None = None,
    ) -> tuple[NodeSelectionScore, ...]:
        """Calculate normalized factors, utility, and temperature softmax.

        ``family_counts`` maps a method family to the number of *other*
        previously recorded nodes in that family.  When omitted it is derived
        from ``population`` (or from the candidates themselves).
        """

        candidates = tuple(candidates)
        self._validate(candidates)
        population = tuple(population) if population is not None else candidates
        if any(node.score.direction is not candidates[0].score.direction for node in population):
            raise ValueError("population and candidate metrics must have one consistent direction")
        lookup = {node.node_id: node for node in population}
        for node in candidates:
            lookup.setdefault(node.node_id, node)

        direction = candidates[0].score.direction
        raw_scores = [node.score.primary for node in candidates]
        low, high = min(raw_scores), max(raw_scores)
        if high == low:
            quality = {node.node_id: 0.5 for node in candidates}
        elif direction is MetricDirection.HIGHER_IS_BETTER:
            quality = {node.node_id: (node.score.primary - low) / (high - low) for node in candidates}
        else:
            quality = {node.node_id: (high - node.score.primary) / (high - low) for node in candidates}

        raw_progress: dict[str, float] = {}
        for node in candidates:
            parent_utilities = [
                lookup[parent_id].score.utility for parent_id in node.parent_ids if parent_id in lookup
            ]
            raw_progress[node.node_id] = (
                max(0.0, node.score.utility - max(parent_utilities)) if parent_utilities else 0.0
            )
        maximum_progress = max(raw_progress.values(), default=0.0)
        progress = {
            node.node_id: raw_progress[node.node_id] / maximum_progress if maximum_progress > 0 else 0.0
            for node in candidates
        }

        if family_counts is None:
            population_counts: dict[str, int] = {}
            for item in population:
                population_counts[item.method_family] = population_counts.get(item.method_family, 0) + 1
            family_counts = {
                family: max(0, count - 1) for family, count in population_counts.items()
            }
        if any(count < 0 for count in family_counts.values()):
            raise ValueError("family counts must be non-negative")
        novelty = {
            node.node_id: 1.0 / math.sqrt(1.0 + family_counts.get(node.method_family, 0))
            for node in candidates
        }

        utilities = {
            node.node_id: self.weights.quality * quality[node.node_id]
            + self.weights.progress * progress[node.node_id]
            + self.weights.novelty * novelty[node.node_id]
            for node in candidates
        }
        scaled = [utilities[node.node_id] / self.temperature for node in candidates]
        offset = max(scaled)
        exponentials = [math.exp(value - offset) for value in scaled]
        denominator = sum(exponentials)
        probabilities = [value / denominator for value in exponentials]
        return tuple(
            NodeSelectionScore(
                node_id=node.node_id,
                quality=quality[node.node_id],
                progress=progress[node.node_id],
                novelty=novelty[node.node_id],
                utility=utilities[node.node_id],
                probability=probability,
            )
            for node, probability in zip(candidates, probabilities, strict=True)
        )

    def sample(
        self,
        candidates: Sequence[SearchNode],
        *,
        count: int = 1,
        replace: bool = False,
        population: Sequence[SearchNode] | None = None,
        family_counts: Mapping[str, int] | None = None,
    ) -> tuple[SearchNode, ...]:
        """Sample parents using softmax weights, optionally without replacement."""

        candidates = tuple(candidates)
        if count < 0:
            raise ValueError("sample count must be non-negative")
        if count == 0:
            return ()
        if not replace and count > len(candidates):
            raise ValueError("cannot sample more distinct parents than candidates")
        scores = self.score(candidates, population=population, family_counts=family_counts)
        by_id = {node.node_id: node for node in candidates}
        # Softmax probabilities preserve the same relative weights and avoid a
        # second, potentially overflowing exponentiation during sampling.
        weighted = [(score.node_id, score.probability) for score in scores]
        selected: list[SearchNode] = []
        for _ in range(count):
            total = sum(weight for _, weight in weighted)
            threshold = self._rng.random() * total
            cumulative = 0.0
            selected_index = len(weighted) - 1
            for index, (_, weight) in enumerate(weighted):
                cumulative += weight
                if threshold < cumulative:
                    selected_index = index
                    break
            node_id, _ = weighted[selected_index]
            selected.append(by_id[node_id])
            if not replace:
                weighted.pop(selected_index)
        self.last_trace = SelectionTrace(
            operation="sample",
            candidates=scores,
            selected_node_ids=tuple(node.node_id for node in selected),
            replacement=replace,
        )
        return tuple(selected)

    def select(
        self,
        operator: object,
        candidates: Sequence[SearchNode],
        *,
        population: Sequence[SearchNode] | None = None,
        family_counts: Mapping[str, int] | None = None,
    ) -> tuple[SearchNode, ...]:
        """Select the arity required by a seed operation.

        Crossover always samples two distinct nodes without replacement.
        Draft never selects inherited parents.
        """

        operation = operation_family_for(operator)
        if operation == "draft":
            self.last_trace = SelectionTrace(operation, (), (), False)
            return ()
        count = 2 if operation == "crossover" else 1
        selected = self.sample(
            candidates,
            count=count,
            replace=False,
            population=population,
            family_counts=family_counts,
        )
        assert self.last_trace is not None
        self.last_trace = SelectionTrace(
            operation=operation,
            candidates=self.last_trace.candidates,
            selected_node_ids=self.last_trace.selected_node_ids,
            replacement=False,
        )
        return selected

    evaluate = score
    select_parents = select
