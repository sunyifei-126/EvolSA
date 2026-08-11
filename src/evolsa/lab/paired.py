"""Matched-budget paired experiment records and report aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from statistics import fmean
from typing import Iterable, Mapping, Sequence

from evolsa.core import Budget, LabReport, PairedOutcome

from .statistics import paired_bootstrap_ci


def budgets_match(candidate: Budget, baseline: Budget) -> bool:
    """Return whether two declared budget ceilings are exactly matched."""

    return candidate == baseline


def validate_matched_budget(
    candidate: Budget,
    baseline: Budget,
    *,
    raise_on_mismatch: bool = False,
) -> bool:
    """Validate budget equality, optionally raising with a useful diagnostic."""

    matched = budgets_match(candidate, baseline)
    if not matched and raise_on_mismatch:
        differing = [
            name
            for name in ("max_tokens", "max_exec_seconds", "max_attempts", "max_cost")
            if getattr(candidate, name) != getattr(baseline, name)
        ]
        raise ValueError(f"candidate and baseline budgets differ: {', '.join(differing)}")
    return matched


@dataclass(frozen=True, slots=True)
class PairedTrial:
    """One causal comparison with its matching conditions recorded."""

    outcome: PairedOutcome
    candidate_budget: Budget
    baseline_budget: Budget
    candidate_parent_id: str = ""
    baseline_parent_id: str = ""
    candidate_model_id: str = ""
    baseline_model_id: str = ""
    candidate_sampling: Mapping[str, object] = field(default_factory=dict)
    baseline_sampling: Mapping[str, object] = field(default_factory=dict)
    split: str = ""
    evaluator_id: str = ""

    @property
    def matched_budget(self) -> bool:
        return budgets_match(self.candidate_budget, self.baseline_budget)

    @property
    def matched_conditions(self) -> bool:
        parent_match = bool(self.candidate_parent_id) and self.candidate_parent_id == self.baseline_parent_id
        model_match = bool(self.candidate_model_id) and self.candidate_model_id == self.baseline_model_id
        return (
            self.matched_budget
            and bool(self.split)
            and bool(self.evaluator_id)
            and parent_match
            and model_match
            and dict(self.candidate_sampling) == dict(self.baseline_sampling)
        )


def _budget_evidence_matches(
    outcomes: Sequence[PairedOutcome],
    candidate_budgets: Sequence[Budget],
    baseline_budgets: Sequence[Budget],
) -> bool:
    if len(candidate_budgets) != len(outcomes) or len(baseline_budgets) != len(outcomes):
        raise ValueError("each paired outcome requires candidate and baseline budget evidence")
    return all(
        budgets_match(candidate, baseline)
        for candidate, baseline in zip(candidate_budgets, baseline_budgets, strict=True)
    )


def jointly_valid_outcomes(outcomes: Iterable[PairedOutcome]) -> tuple[PairedOutcome, ...]:
    """Return pairs whose candidate and baseline both passed evaluation.

    Invalid scores have no stable metric meaning and therefore cannot
    contribute to lift or confidence intervals.  The complete outcome set is
    still retained in :class:`LabReport` for validity-rate, cost, and audit
    accounting.
    """

    return tuple(
        outcome
        for outcome in outcomes
        if outcome.candidate_valid and outcome.baseline_valid
    )


def aggregate_lab_report(
    operator_id: str,
    baseline_id: str,
    outcomes: Iterable[PairedOutcome],
    *,
    candidate_budgets: Sequence[Budget] | None = None,
    baseline_budgets: Sequence[Budget] | None = None,
    matched_budget: bool | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
    bootstrap_confidence: float = 0.95,
    cost_penalty: float = 1.0,
) -> LabReport:
    """Aggregate paired outcomes into a content-addressed :class:`LabReport`.

    ``matched_budget`` defaults to false unless complete budget evidence is
    supplied.  An explicit true value cannot override contradictory budget
    objects.  This is intentional fail-closed behavior.
    """

    materialized = tuple(outcomes)
    if not operator_id or not baseline_id or operator_id == baseline_id:
        raise ValueError("operator and distinct baseline identifiers are required")
    if not materialized:
        raise ValueError("a lab report requires at least one paired outcome")
    if cost_penalty < 0.0 or not math.isfinite(cost_penalty):
        raise ValueError("cost_penalty must be finite and non-negative")

    has_candidate_budgets = candidate_budgets is not None
    has_baseline_budgets = baseline_budgets is not None
    if has_candidate_budgets != has_baseline_budgets:
        raise ValueError("candidate and baseline budget evidence must be supplied together")
    evidence_match: bool | None = None
    if candidate_budgets is not None and baseline_budgets is not None:
        candidate_evidence = tuple(candidate_budgets)
        baseline_evidence = tuple(baseline_budgets)
        evidence_match = _budget_evidence_matches(materialized, candidate_evidence, baseline_evidence)
        enriched: list[PairedOutcome] = []
        for outcome, candidate_budget, baseline_budget in zip(
            materialized, candidate_evidence, baseline_evidence, strict=True
        ):
            if outcome.candidate_budget is not None and outcome.candidate_budget != candidate_budget:
                raise ValueError("candidate budget contradicts paired outcome evidence")
            if outcome.baseline_budget is not None and outcome.baseline_budget != baseline_budget:
                raise ValueError("baseline budget contradicts paired outcome evidence")
            enriched.append(
                replace(
                    outcome,
                    candidate_budget=candidate_budget,
                    baseline_budget=baseline_budget,
                )
            )
        materialized = tuple(enriched)
    else:
        evidence_match = all(outcome.budget_evidence_matches for outcome in materialized)
    # A caller-supplied boolean can make evidence stricter, never create it.
    budget_match = evidence_match and matched_budget is not False
    conditions_match = all(outcome.condition_evidence_matches for outcome in materialized)

    valid_pairs = jointly_valid_outcomes(materialized)
    if not valid_pairs:
        raise ValueError("a lab report requires at least one jointly valid paired outcome")

    interval = paired_bootstrap_ci(
        valid_pairs,
        confidence=bootstrap_confidence,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    candidate_valid_count = sum(outcome.candidate_valid for outcome in materialized)
    baseline_valid_count = sum(outcome.baseline_valid for outcome in materialized)
    cost_adjusted = fmean(
        (outcome.lift if outcome.candidate_valid and outcome.baseline_valid else 0.0)
        - cost_penalty * (outcome.candidate_cost - outcome.baseline_cost)
        for outcome in materialized
    )
    worst_regression = max(
        max(0.0, outcome.regression, -outcome.lift)
        for outcome in materialized
    )
    valid_transfer_tasks = {
        outcome.task_id
        for outcome in materialized
        if outcome.task_id and outcome.candidate_valid and outcome.baseline_valid
    }
    return LabReport(
        operator_id=operator_id,
        baseline_id=baseline_id,
        outcomes=materialized,
        mean_lift=interval.mean,
        lcb95=interval.lower,
        ucb95=interval.upper,
        valid_rate=candidate_valid_count / len(materialized),
        baseline_valid_rate=baseline_valid_count / len(materialized),
        cost_adjusted_lift=cost_adjusted,
        transfer_coverage=len(valid_transfer_tasks),
        worst_regression=worst_regression,
        matched_budget=budget_match,
        matched_conditions=conditions_match,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence=bootstrap_confidence,
        cost_penalty=cost_penalty,
    )


class LabReportBuilder:
    """Reusable deterministic report builder for a candidate/baseline pair."""

    def __init__(
        self,
        operator_id: str,
        baseline_id: str,
        *,
        bootstrap_samples: int = 10_000,
        bootstrap_seed: int = 0,
        bootstrap_confidence: float = 0.95,
        cost_penalty: float = 1.0,
    ) -> None:
        self.operator_id = operator_id
        self.baseline_id = baseline_id
        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_seed = bootstrap_seed
        self.bootstrap_confidence = bootstrap_confidence
        self.cost_penalty = cost_penalty

    def build(self, trials: Iterable[PairedTrial]) -> LabReport:
        materialized = tuple(trials)
        if not materialized:
            raise ValueError("at least one paired trial is required")
        # Full matching conditions are retained on the trial, but LabReport's
        # matched_budget field intentionally represents only the budget gate.
        enriched = tuple(
            replace(
                trial.outcome,
                split=trial.split,
                evaluator_id=trial.evaluator_id,
                candidate_parent_id=trial.candidate_parent_id,
                baseline_parent_id=trial.baseline_parent_id,
                candidate_model_id=trial.candidate_model_id,
                baseline_model_id=trial.baseline_model_id,
                candidate_sampling=dict(trial.candidate_sampling),
                baseline_sampling=dict(trial.baseline_sampling),
                candidate_budget=trial.candidate_budget,
                baseline_budget=trial.baseline_budget,
            )
            for trial in materialized
        )
        return aggregate_lab_report(
            self.operator_id,
            self.baseline_id,
            enriched,
            candidate_budgets=tuple(trial.candidate_budget for trial in materialized),
            baseline_budgets=tuple(trial.baseline_budget for trial in materialized),
            bootstrap_samples=self.bootstrap_samples,
            bootstrap_seed=self.bootstrap_seed,
            bootstrap_confidence=self.bootstrap_confidence,
            cost_penalty=self.cost_penalty,
        )


build_lab_report = aggregate_lab_report


__all__ = [
    "LabReportBuilder",
    "PairedTrial",
    "aggregate_lab_report",
    "budgets_match",
    "build_lab_report",
    "jointly_valid_outcomes",
    "validate_matched_budget",
]
