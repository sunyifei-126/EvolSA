"""Deterministic trace mining for candidate improvement operators.

The miner deliberately uses only structured :class:`ExperienceCard` fields.  It
does not ask a model to reinterpret a trace, which keeps the evidence used to
create an operator reproducible and auditable.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from statistics import fmean
from typing import Iterable

from evolsa.core import ExecutionStatus, ExperienceCard
from evolsa.core.hashing import sha256_json


_TAG_SEPARATOR = re.compile(r"[^a-z0-9._-]+")


class PatternKind(str, Enum):
    """Kinds of evidence that can seed a synthesized operator."""

    HIGH_YIELD = "high_yield"
    FAILURE = "failure"


@dataclass(frozen=True, slots=True)
class TracePattern:
    """A repeated, content-addressed pattern found in execution traces."""

    kind: PatternKind
    change_tags: tuple[str, ...]
    support: int
    mean_delta: float
    success_count: int
    failure_count: int
    evidence_ids: tuple[str, ...]
    source_operators: tuple[str, ...]
    method_families: tuple[str, ...]
    error_signatures: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    pattern_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PatternKind(self.kind))
        if self.support < 1 or self.support != len(self.evidence_ids):
            raise ValueError("pattern support must equal the number of evidence cards")
        if self.success_count < 0 or self.failure_count < 0:
            raise ValueError("pattern outcome counters cannot be negative")
        if self.success_count + self.failure_count != self.support:
            raise ValueError("pattern outcome counters must add up to support")
        if not self.change_tags or not math.isfinite(self.mean_delta):
            raise ValueError("a trace pattern requires tags and a finite mean delta")

        for field_name in (
            "change_tags",
            "evidence_ids",
            "source_operators",
            "method_families",
            "error_signatures",
            "preconditions",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))

        payload = {
            "kind": self.kind,
            "change_tags": self.change_tags,
            "support": self.support,
            "mean_delta": self.mean_delta,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "evidence_ids": self.evidence_ids,
            "source_operators": self.source_operators,
            "method_families": self.method_families,
            "error_signatures": self.error_signatures,
            "preconditions": self.preconditions,
        }
        digest = sha256_json(payload)
        if self.pattern_id and self.pattern_id != digest:
            raise ValueError("pattern_id does not match pattern content")
        object.__setattr__(self, "pattern_id", digest)

    @property
    def source(self) -> str:
        return "trajectory_mining" if self.kind is PatternKind.HIGH_YIELD else "failure_mining"


def _normalise_tag(value: str) -> str:
    tag = _TAG_SEPARATOR.sub("-", value.strip().lower()).strip("-")
    return tag or "untagged"


def _card_is_valid(card: ExperienceCard) -> bool:
    """Consume only explicit evaluator validity and successful execution."""

    return card.valid is True and card.status is ExecutionStatus.SUCCESS


class TraceMiner:
    """Mine repeated high-return changes and repeated failure modes.

    A card contributes to a high-yield pattern only if it is valid and its
    parent-relative utility delta reaches ``minimum_delta``.  Invalid and
    non-successful cards are considered only by the failure miner.  This
    separation prevents a large score on an invalid run from seeding an
    operator.
    """

    def __init__(
        self,
        *,
        minimum_support: int = 2,
        minimum_delta: float = 0.0,
        include_tag_combinations: bool = True,
    ) -> None:
        if minimum_support < 2:
            raise ValueError("minimum_support must be at least 2")
        if not math.isfinite(minimum_delta):
            raise ValueError("minimum_delta must be finite")
        self.minimum_support = minimum_support
        self.minimum_delta = minimum_delta
        self.include_tag_combinations = include_tag_combinations

    def mine(self, cards: Iterable[ExperienceCard]) -> tuple[TracePattern, ...]:
        materialized = tuple(cards)
        patterns = (*self.mine_high_yield(materialized), *self.mine_failures(materialized))
        return tuple(sorted(patterns, key=lambda item: (item.kind.value, item.change_tags, item.pattern_id)))

    def mine_high_yield(self, cards: Iterable[ExperienceCard]) -> tuple[TracePattern, ...]:
        groups: dict[tuple[str, ...], list[ExperienceCard]] = defaultdict(list)
        for card in cards:
            if not _card_is_valid(card) or card.delta_vs_parent < self.minimum_delta:
                continue
            for key in self._keys(card):
                groups[key].append(card)

        patterns: list[TracePattern] = []
        for tags, evidence in groups.items():
            if len(evidence) < self.minimum_support:
                continue
            patterns.append(self._make_pattern(PatternKind.HIGH_YIELD, tags, evidence))
        return tuple(sorted(patterns, key=lambda item: (-item.mean_delta, -item.support, item.change_tags)))

    def mine_failures(self, cards: Iterable[ExperienceCard]) -> tuple[TracePattern, ...]:
        groups: dict[tuple[str, ...], list[ExperienceCard]] = defaultdict(list)
        for card in cards:
            if _card_is_valid(card):
                continue
            signature = card.error_signature or card.status.value
            # Mine the error signature independently of change tags so the
            # same failure can be recognized across otherwise different
            # attempted strategies.
            keys = {
                *self._keys(card),
                (f"error-{_normalise_tag(signature)}",),
            }
            for key in sorted(keys):
                groups[key].append(card)

        patterns: list[TracePattern] = []
        for tags, evidence in groups.items():
            if len(evidence) < self.minimum_support:
                continue
            patterns.append(self._make_pattern(PatternKind.FAILURE, tags, evidence))
        return tuple(sorted(patterns, key=lambda item: (-item.support, item.change_tags)))

    def _keys(self, card: ExperienceCard) -> tuple[tuple[str, ...], ...]:
        tags = tuple(sorted({_normalise_tag(tag) for tag in card.change_tags if tag.strip()}))
        if not tags:
            return ()
        keys: list[tuple[str, ...]] = [(tag,) for tag in tags]
        if self.include_tag_combinations and len(tags) > 1:
            keys.append(tags)
        return tuple(keys)

    def _make_pattern(
        self,
        kind: PatternKind,
        tags: tuple[str, ...],
        evidence: list[ExperienceCard],
    ) -> TracePattern:
        ordered = tuple(sorted(evidence, key=lambda item: (item.step_id, item.node_id)))
        successes = sum(_card_is_valid(card) for card in ordered)
        method_counts = Counter(card.method_family for card in ordered if card.method_family)
        dominant_methods = tuple(
            method for method, _ in sorted(method_counts.items(), key=lambda item: (-item[1], item[0]))
        )
        errors = tuple(sorted({card.error_signature or card.status.value for card in ordered if not _card_is_valid(card)}))
        preconditions = tuple(f"method_family:{method}" for method in dominant_methods[:2])
        return TracePattern(
            kind=kind,
            change_tags=tags,
            support=len(ordered),
            mean_delta=fmean(card.delta_vs_parent for card in ordered),
            success_count=successes,
            failure_count=len(ordered) - successes,
            evidence_ids=tuple(card.card_id for card in ordered),
            source_operators=tuple(sorted({card.operator_id for card in ordered})),
            method_families=dominant_methods,
            error_signatures=errors,
            preconditions=preconditions,
        )


__all__ = ["PatternKind", "TraceMiner", "TracePattern"]
