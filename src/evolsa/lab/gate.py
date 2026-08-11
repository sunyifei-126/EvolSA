"""External, fail-closed promotion policy for candidate operators."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Iterable

from evolsa.core import (
    GateResult,
    LabReport,
    OperatorPassport,
    PromotionDecision,
    PromotionPolicy,
)
from evolsa.core.hashing import sha256_json

from .statistics import paired_bootstrap_ci
from .paired import jointly_valid_outcomes


_SUPPORTED_PRIMARY_METRIC = "paired_lift"
_SUPPORTED_GUARDRAILS = frozenset({"valid_rate", "cost", "regression", "transfer"})


class PromotionGate:
    """Evaluate immutable lab evidence against a fixed promotion policy."""

    def __init__(self, policy: PromotionPolicy | None = None) -> None:
        self.policy = policy or PromotionPolicy()

    def evaluate(
        self,
        report: LabReport,
        *,
        operator: OperatorPassport | str | None = None,
        security_issues: Iterable[str] = (),
        integrity_issues: Iterable[str] = (),
    ) -> GateResult:
        policy = self.policy
        expected_operator_id = operator.operator_id if isinstance(operator, OperatorPassport) else operator
        policy_hash = sha256_json(self.policy)
        safety = tuple(str(issue).strip() for issue in security_issues if str(issue).strip())
        integrity = [str(issue).strip() for issue in integrity_issues if str(issue).strip()]
        integrity.extend(self._integrity_findings(report, expected_operator_id, operator, policy))

        if safety or integrity:
            reasons = tuple(
                [*(f"security_issue:{issue}" for issue in safety), *(f"integrity_issue:{issue}" for issue in integrity)]
            )
            return GateResult(
                decision=PromotionDecision.QUARANTINE,
                operator_id=report.operator_id,
                report_id=report.report_id,
                reasons=reasons,
                policy_hash=policy_hash,
            )

        evidence_gaps: list[str] = []
        failures: list[str] = []
        if policy.require_matched_budget and not report.matched_budget:
            evidence_gaps.append("matched_budget_required")
        if policy.require_matched_conditions and not report.matched_conditions:
            evidence_gaps.append("matched_conditions_required")
        if report.transfer_coverage < policy.minimum_transfer_coverage:
            evidence_gaps.append(
                f"transfer_coverage:{report.transfer_coverage}<{policy.minimum_transfer_coverage}"
            )
        if not policy.baseline_id or not policy.baseline_content_hash:
            evidence_gaps.append("policy_baseline_identity_required")
        if not policy.required_trials:
            evidence_gaps.append("policy_required_trials_missing")
        elif tuple(sorted((item.task_id, item.seed) for item in report.outcomes)) != tuple(
            sorted(policy.required_trials)
        ):
            evidence_gaps.append("required_trial_allocation_mismatch")
        if any(item.split != policy.required_split for item in report.outcomes):
            evidence_gaps.append(f"required_split_mismatch:{policy.required_split}")
        if not policy.required_evaluator_id:
            evidence_gaps.append("policy_required_evaluator_missing")
        elif any(item.evaluator_id != policy.required_evaluator_id for item in report.outcomes):
            evidence_gaps.append("required_evaluator_mismatch")
        if report.bootstrap_samples != policy.bootstrap_samples:
            evidence_gaps.append(
                f"bootstrap_samples:{report.bootstrap_samples}!={policy.bootstrap_samples}"
            )
        if report.bootstrap_seed != policy.bootstrap_seed:
            evidence_gaps.append(f"bootstrap_seed:{report.bootstrap_seed}!={policy.bootstrap_seed}")
        if not math.isclose(
            report.bootstrap_confidence,
            policy.bootstrap_confidence,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            evidence_gaps.append(
                "bootstrap_confidence:"
                f"{report.bootstrap_confidence:.12g}!={policy.bootstrap_confidence:.12g}"
            )
        if report.lcb95 <= policy.minimum_lcb_lift:
            failures.append(f"lcb95:{report.lcb95:.12g}<={policy.minimum_lcb_lift:.12g}")
        minimum_valid_rate = report.baseline_valid_rate - policy.valid_rate_tolerance
        if report.valid_rate < minimum_valid_rate:
            failures.append(f"valid_rate:{report.valid_rate:.12g}<{minimum_valid_rate:.12g}")
        if report.cost_adjusted_lift <= policy.minimum_cost_adjusted_lift:
            failures.append(
                f"cost_adjusted_lift:{report.cost_adjusted_lift:.12g}<={policy.minimum_cost_adjusted_lift:.12g}"
            )
        if report.worst_regression > policy.maximum_regression:
            failures.append(
                f"worst_regression:{report.worst_regression:.12g}>{policy.maximum_regression:.12g}"
            )

        if failures:
            decision = PromotionDecision.REJECT
            reasons = tuple((*failures, *evidence_gaps))
        elif evidence_gaps:
            decision = PromotionDecision.REVISE
            reasons = tuple(evidence_gaps)
        else:
            decision = PromotionDecision.PROMOTE
            reasons = ("all promotion gates passed",)
        return GateResult(
            decision=decision,
            operator_id=report.operator_id,
            report_id=report.report_id,
            reasons=reasons,
            policy_hash=policy_hash,
        )

    assess = evaluate
    __call__ = evaluate

    @staticmethod
    def _integrity_findings(
        report: LabReport,
        expected_operator_id: str | None,
        operator: OperatorPassport | str | None,
        policy: PromotionPolicy,
    ) -> list[str]:
        findings: list[str] = []
        if expected_operator_id is not None and expected_operator_id != report.operator_id:
            findings.append("operator_id_mismatch")
        if not report.operator_id or not report.baseline_id or report.operator_id == report.baseline_id:
            findings.append("invalid_operator_or_baseline_identity")
        if policy.baseline_id and report.baseline_id != policy.baseline_id:
            findings.append("policy_baseline_id_mismatch")
        if not report.outcomes:
            findings.append("missing_paired_outcomes")
            return findings
        if isinstance(operator, OperatorPassport):
            if report.baseline_id != operator.verification.baseline:
                findings.append("verification_baseline_mismatch")
            if policy.baseline_id and operator.verification.baseline != policy.baseline_id:
                findings.append("verification_policy_baseline_mismatch")
            if len(report.outcomes) < operator.verification.minimum_tasks:
                findings.append("verification_minimum_tasks_not_met")
            if operator.verification.primary_metric != _SUPPORTED_PRIMARY_METRIC:
                findings.append("unsupported_verification_primary_metric")
            if not set(operator.verification.guardrails) <= _SUPPORTED_GUARDRAILS:
                findings.append("unsupported_verification_guardrail")
        trial_identities = tuple((item.task_id, item.seed) for item in report.outcomes)
        if len(trial_identities) != len(set(trial_identities)):
            findings.append("duplicate_task_seed_pair")
        if any(item.direction is not policy.metric_direction for item in report.outcomes):
            findings.append("metric_direction_policy_mismatch")
        if not math.isclose(
            report.cost_penalty,
            policy.cost_penalty,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            findings.append("cost_penalty_policy_mismatch")
        metrics = (
            report.mean_lift,
            report.lcb95,
            report.ucb95,
            report.valid_rate,
            report.baseline_valid_rate,
            report.cost_adjusted_lift,
            report.worst_regression,
        )
        if not all(math.isfinite(metric) for metric in metrics):
            findings.append("non_finite_report_metric")
        if not 0.0 <= report.valid_rate <= 1.0 or not 0.0 <= report.baseline_valid_rate <= 1.0:
            findings.append("invalid_valid_rate")
        if not report.lcb95 <= report.mean_lift <= report.ucb95:
            findings.append("confidence_interval_does_not_bound_mean")

        valid_pairs = jointly_valid_outcomes(report.outcomes)
        if not valid_pairs:
            findings.append("missing_jointly_valid_paired_outcomes")
            return findings
        observed_mean = fmean(outcome.lift for outcome in valid_pairs)
        if not math.isclose(report.mean_lift, observed_mean, rel_tol=1e-12, abs_tol=1e-12):
            findings.append("mean_lift_does_not_match_outcomes")
        try:
            interval = paired_bootstrap_ci(
                valid_pairs,
                confidence=report.bootstrap_confidence,
                samples=report.bootstrap_samples,
                seed=report.bootstrap_seed,
            )
        except ValueError:
            findings.append("invalid_bootstrap_configuration")
        else:
            if not math.isclose(report.lcb95, interval.lower, rel_tol=1e-12, abs_tol=1e-12):
                findings.append("lcb95_does_not_match_outcomes")
            if not math.isclose(report.ucb95, interval.upper, rel_tol=1e-12, abs_tol=1e-12):
                findings.append("ucb95_does_not_match_outcomes")
        observed_cost_adjusted = fmean(
            (outcome.lift if outcome.candidate_valid and outcome.baseline_valid else 0.0)
            - policy.cost_penalty * (outcome.candidate_cost - outcome.baseline_cost)
            for outcome in report.outcomes
        )
        if not math.isclose(
            report.cost_adjusted_lift,
            observed_cost_adjusted,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            findings.append("cost_adjusted_lift_does_not_match_outcomes")
        observed_candidate_rate = sum(item.candidate_valid for item in report.outcomes) / len(report.outcomes)
        observed_baseline_rate = sum(item.baseline_valid for item in report.outcomes) / len(report.outcomes)
        if not math.isclose(report.valid_rate, observed_candidate_rate, rel_tol=1e-12, abs_tol=1e-12):
            findings.append("candidate_valid_rate_does_not_match_outcomes")
        if not math.isclose(report.baseline_valid_rate, observed_baseline_rate, rel_tol=1e-12, abs_tol=1e-12):
            findings.append("baseline_valid_rate_does_not_match_outcomes")

        valid_tasks = {
            item.task_id
            for item in report.outcomes
            if item.task_id and item.candidate_valid and item.baseline_valid
        }
        if report.transfer_coverage != len(valid_tasks):
            findings.append("transfer_coverage_does_not_match_valid_evidence")
        observed_budget_match = all(item.budget_evidence_matches for item in report.outcomes)
        if report.matched_budget is not observed_budget_match:
            findings.append("matched_budget_does_not_match_trial_evidence")
        observed_conditions_match = all(item.condition_evidence_matches for item in report.outcomes)
        if report.matched_conditions is not observed_conditions_match:
            findings.append("matched_conditions_do_not_match_trial_evidence")
        observed_regression = max(max(0.0, item.regression, -item.lift) for item in report.outcomes)
        if report.worst_regression + 1e-12 < observed_regression:
            findings.append("worst_regression_understates_outcomes")
        return findings


__all__ = ["PromotionGate"]
