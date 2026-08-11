"""Runtime capabilities for executable operators.

Operation families are deliberately runtime metadata rather than passport
identity.  A mined operator such as ``normalize-punctuation`` can therefore
behave as an improve operation without overloading its name or changing the
public passport schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from evolsa.core import Applicability, Artifact, Budget, OperatorPassport, SearchNode, TaskState
from evolsa.sdk.protocols import ModelBackend, Operator


OPERATION_FAMILIES = frozenset({"draft", "improve", "debug", "crossover"})
_PARENT_COUNTS = {"draft": 0, "improve": 1, "debug": 1, "crossover": 2}


def _validated_operation_family(value: object) -> str:
    if type(value) is not str:
        raise ValueError("operation_family must be a string")
    family = value.strip().lower()
    if family not in OPERATION_FAMILIES:
        raise ValueError(
            f"unsupported operation_family {value!r}; expected one of "
            f"{sorted(OPERATION_FAMILIES)}"
        )
    return family


def operation_family_for(operator: object) -> str:
    """Return an operator's explicit runtime family.

    Built-in seed operators already expose ``operation``.  Evolved or loaded
    operators must expose ``operation_family`` directly or be wrapped with
    :func:`bind_operation_family`.  Passport names are intentionally never
    inspected: an operator's identity is not a trustworthy capability signal.

    Exact family strings remain accepted for the scheduler/retriever facade so
    existing calls such as ``retrieve("improve", ...)`` keep working.
    """

    if isinstance(operator, str):
        return _validated_operation_family(operator)
    family = getattr(operator, "operation_family", None)
    if family is None:
        family = getattr(operator, "operation", None)
    if family is None:
        # Preserve capability metadata through simple delegating wrappers
        # (for example deterministic test/demo implementations around a seed
        # operator). This still follows an explicit runtime capability and
        # never derives behavior from passport identity.
        for attribute in ("operator", "_delegate"):
            delegate = getattr(operator, attribute, None)
            if delegate is not None and delegate is not operator:
                try:
                    return operation_family_for(delegate)
                except ValueError:
                    pass
    if family is None:
        raise ValueError(
            "operator has no explicit operation family; wrap evolved or loaded "
            "operators with bind_operation_family(operator, family)"
        )
    return _validated_operation_family(family)


@dataclass(frozen=True, slots=True)
class OperationFamilyBinding:
    """Bind an arbitrary executable operator to a search capability family."""

    operator: Operator
    operation_family: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_family",
            _validated_operation_family(self.operation_family),
        )

    @property
    def passport(self) -> OperatorPassport:
        return self.operator.passport

    @property
    def parent_count(self) -> int:
        return _PARENT_COUNTS[self.operation_family]

    def applies(self, state: TaskState, parents: Sequence[SearchNode]) -> Applicability:
        distinct_parent_ids = {parent.node_id for parent in parents}
        arity_matches = len(parents) == self.parent_count
        distinct_matches = (
            self.operation_family != "crossover" or len(distinct_parent_ids) == self.parent_count
        )
        if not arity_matches or not distinct_matches:
            return Applicability(
                applies=False,
                confidence=0.0,
                reasons=(
                    f"{self.operation_family} requires exactly "
                    f"{self.parent_count} distinct parent node(s)",
                ),
            )
        return self.operator.applies(state, parents)

    def build_context(
        self,
        state: TaskState,
        parents: Sequence[SearchNode],
        experience_view: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        applicability = self.applies(state, parents)
        if not applicability.applies:
            raise ValueError("; ".join(applicability.reasons))
        context = dict(self.operator.build_context(state, parents, experience_view))
        declared = context.get("operation_family")
        if declared is not None and declared != self.operation_family:
            raise ValueError("operator context conflicts with its bound operation_family")
        context["operation_family"] = self.operation_family
        return context

    def propose(
        self,
        context: Mapping[str, Any],
        model: ModelBackend,
        budget: Budget,
    ) -> Artifact:
        return self.operator.propose(context, model, budget)


def bind_operation_family(operator: Operator, operation_family: str) -> OperationFamilyBinding:
    """Return a runtime wrapper suitable for SearchEngine registration."""

    return OperationFamilyBinding(operator=operator, operation_family=operation_family)


__all__ = [
    "OPERATION_FAMILIES",
    "OperationFamilyBinding",
    "bind_operation_family",
    "operation_family_for",
]
