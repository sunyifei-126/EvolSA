"""Synthesis of auditable L0 operator drafts from mined patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass

from evolsa.core import Budget, EffectScope, OperatorVerification, PROTECTED_SCOPES
from evolsa.core.hashing import sha256_json

from .mining import PatternKind, TracePattern


_NAME_PART = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class OperatorDraft:
    """Intermediate, model-independent representation of an L0 operator."""

    name: str
    version: str
    intent: str
    source: str
    inputs: tuple[str, ...]
    preconditions: tuple[str, ...]
    expected_effects: tuple[EffectScope, ...]
    forbidden_effects: tuple[EffectScope, ...]
    context_policy: str
    prompt_template: str
    budget: Budget
    verification: OperatorVerification
    parents: tuple[str, ...]
    known_failures: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    draft_id: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.intent or not self.source or not self.prompt_template:
            raise ValueError("operator draft identity, source, intent, and prompt are required")
        for field_name in (
            "inputs",
            "preconditions",
            "expected_effects",
            "forbidden_effects",
            "parents",
            "known_failures",
            "evidence_ids",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
        digest = sha256_json({
            "name": self.name,
            "version": self.version,
            "intent": self.intent,
            "source": self.source,
            "inputs": self.inputs,
            "preconditions": self.preconditions,
            "expected_effects": self.expected_effects,
            "forbidden_effects": self.forbidden_effects,
            "context_policy": self.context_policy,
            "prompt_template": self.prompt_template,
            "budget": self.budget,
            "verification": self.verification,
            "parents": self.parents,
            "known_failures": self.known_failures,
            "evidence_ids": self.evidence_ids,
        })
        if self.draft_id and self.draft_id != digest:
            raise ValueError("draft_id does not match draft content")
        object.__setattr__(self, "draft_id", digest)


class OperatorSynthesizer:
    """Abstract a mined pattern into a conservative L0 prompt operator."""

    def __init__(
        self,
        *,
        baseline_id: str = "improve@1.0.0",
        version: str = "0.1.0",
        budget: Budget | None = None,
        minimum_tasks: int = 4,
    ) -> None:
        self.baseline_id = baseline_id
        self.version = version
        self.budget = budget or Budget(max_tokens=4096, max_exec_seconds=120.0, max_attempts=1)
        self.minimum_tasks = minimum_tasks

    def synthesize(
        self,
        pattern: TracePattern,
        *,
        name: str | None = None,
        parents: tuple[str, ...] | None = None,
    ) -> OperatorDraft:
        operator_name = name or self._name_for(pattern)
        is_failure = pattern.kind is PatternKind.FAILURE
        intent = (
            f"diagnose and repair the repeated failure pattern: {', '.join(pattern.change_tags)}"
            if is_failure
            else f"apply the repeated high-yield change pattern: {', '.join(pattern.change_tags)}"
        )
        known_failures = pattern.error_signatures if is_failure else tuple(pattern.error_signatures)
        effective_parents = tuple(parents) if parents is not None else pattern.source_operators
        preconditions = (
            *pattern.preconditions,
            *(f"observed_failure:{error}" for error in pattern.error_signatures if is_failure),
            f"mined_support:{pattern.support}",
        )
        prompt = self._prompt_for(pattern, intent)
        return OperatorDraft(
            name=operator_name,
            version=self.version,
            intent=intent,
            source=f"{pattern.source}:{pattern.pattern_id}",
            inputs=("task_spec", "candidate", "execution_trace", "experience_view"),
            preconditions=preconditions,
            expected_effects=(EffectScope.TASK_ARTIFACT,),
            forbidden_effects=tuple(sorted(PROTECTED_SCOPES, key=lambda item: item.value)),
            context_policy="operation-conditioned-evidence-v1",
            prompt_template=prompt,
            budget=self.budget,
            verification=OperatorVerification(
                baseline=self.baseline_id,
                primary_metric="paired_lift",
                guardrails=("valid_rate", "cost", "regression", "transfer"),
                minimum_tasks=self.minimum_tasks,
            ),
            parents=effective_parents,
            known_failures=known_failures,
            evidence_ids=pattern.evidence_ids,
        )

    def synthesize_all(self, patterns: tuple[TracePattern, ...]) -> tuple[OperatorDraft, ...]:
        return tuple(self.synthesize(pattern) for pattern in patterns)

    @staticmethod
    def _name_for(pattern: TracePattern) -> str:
        stem = "repair" if pattern.kind is PatternKind.FAILURE else "apply"
        tag_part = "-".join(pattern.change_tags[:3])
        slug = _NAME_PART.sub("-", f"{stem}-{tag_part}".lower()).strip("-")
        return slug[:64] or f"{stem}-mined-pattern"

    @staticmethod
    def _prompt_for(pattern: TracePattern, intent: str) -> str:
        failures = ", ".join(pattern.error_signatures) or "none observed"
        return (
            "You are an evidence-derived improvement operator.\n"
            f"Intent: {intent}.\n"
            f"Observed change tags: {', '.join(pattern.change_tags)}.\n"
            f"Evidence support: {pattern.support}; mean parent-relative lift: {pattern.mean_delta:.8g}.\n"
            f"Known failure signatures: {failures}.\n"
            "Respect the task specification and change only the candidate artifact. "
            "Do not alter evaluators, sandbox policy, budgets, logs, promotion gates, or release permissions.\n"
            "Use the supplied structured context, explain the intended repair internally, and return only the complete improved artifact."
        )


__all__ = ["OperatorDraft", "OperatorSynthesizer"]
