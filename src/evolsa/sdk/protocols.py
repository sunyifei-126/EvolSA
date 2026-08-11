"""Minimal task, model, and operator extension protocols."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from evolsa.core import (
    Applicability,
    Artifact,
    Budget,
    Observation,
    OperatorPassport,
    ScoreCard,
    SearchNode,
    TaskState,
)


@runtime_checkable
class TaskAdapter(Protocol):
    """Connect a deterministic evaluator-backed task to EvolSA."""

    name: str

    def prepare(self, seed: int, split: str = "mine") -> TaskState: ...

    def seed_artifact(self, state: TaskState) -> Artifact: ...

    def execute(self, state: TaskState, artifact: Artifact, budget: Budget) -> Observation: ...

    def score(self, state: TaskState, observation: Observation) -> ScoreCard: ...


@runtime_checkable
class ModelBackend(Protocol):
    """Model-independent generation interface used by L0/L1 operators."""

    model_id: str

    def generate(self, messages: Sequence[Mapping[str, str]], budget: Budget) -> str: ...


@runtime_checkable
class Operator(Protocol):
    """Executable improvement operation with a versioned passport."""

    passport: OperatorPassport

    def applies(self, state: TaskState, parents: Sequence[SearchNode]) -> Applicability: ...

    def build_context(
        self,
        state: TaskState,
        parents: Sequence[SearchNode],
        experience_view: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def propose(
        self,
        context: Mapping[str, Any],
        model: ModelBackend,
        budget: Budget,
    ) -> Artifact: ...


class ContextRenderer(Protocol):
    def render(self, operator_id: str, evidence: Mapping[str, Any]) -> Mapping[str, Any]: ...
