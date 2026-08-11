from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from evolsa.core import (
    Budget,
    ExecutionStatus,
    ExperienceCard,
    GateResult,
    OperatorLifecycle,
    PairedOutcome,
    PromotionDecision,
    PromotionPolicy,
)
from evolsa.foundry import OperatorCompiler, OperatorSynthesizer, PatternKind, TraceMiner
from evolsa.lab import (
    LabReportBuilder,
    PairedTrial,
    PromotionGate,
    aggregate_lab_report,
    paired_bootstrap_ci,
    validate_matched_budget,
)
from evolsa.kernel import TrustKernel, sign_passport
from evolsa.operators import seed_operators
from evolsa.registry import (
    LifecycleTransitionError,
    LocalRegistry,
    RegistryConflictError,
    RegistryIntegrityError,
)


def make_card(
    node_id: str,
    *,
    delta: float,
    status: ExecutionStatus = ExecutionStatus.SUCCESS,
    tags: tuple[str, ...] = ("residual slicing",),
    error: str = "",
) -> ExperienceCard:
    return ExperienceCard(
        node_id=node_id,
        step_id=int(node_id.removeprefix("n")),
        operator_id="improve@1.0.0",
        parent_node_ids=("root",),
        generation_id=1,
        score=1.0 + delta,
        utility=1.0 + delta,
        status=status,
        error_signature=error,
        runtime_seconds=1.0,
        token_count=10,
        cost=0.1,
        method_family="analysis",
        family_count_before=1,
        delta_vs_parent=delta,
        novelty_score=0.5,
        rank=1,
        current_best=delta > 0,
        selection_utility=1.0,
        change_tags=tags,
        valid=status is ExecutionStatus.SUCCESS,
    )


def make_outcomes(lifts: tuple[float, ...]) -> tuple[PairedOutcome, ...]:
    return tuple(
        PairedOutcome(
            task_id=f"held-out-{index}",
            seed=index,
            candidate_score=1.0 + lift,
            baseline_score=1.0,
            candidate_valid=True,
            baseline_valid=True,
            candidate_cost=1.0,
            baseline_cost=1.0,
            split="promotion",
            evaluator_id="test-evaluator-v1",
            candidate_parent_id="same-parent",
            baseline_parent_id="same-parent",
            candidate_model_id="same-model",
            baseline_model_id="same-model",
            candidate_sampling={"temperature": 0},
            baseline_sampling={"temperature": 0},
        )
        for index, lift in enumerate(lifts)
    )


def make_policy(
    outcomes: tuple[PairedOutcome, ...],
    *,
    baseline_id: str,
    baseline_content_hash: str = "a" * 64,
    bootstrap_samples: int,
    bootstrap_seed: int = 0,
    minimum_lcb_lift: float = 0.0,
    minimum_cost_adjusted_lift: float = 0.0,
    minimum_transfer_coverage: int | None = None,
    cost_penalty: float = 1.0,
) -> PromotionPolicy:
    return PromotionPolicy(
        minimum_lcb_lift=minimum_lcb_lift,
        minimum_cost_adjusted_lift=minimum_cost_adjusted_lift,
        maximum_regression=0.0,
        minimum_transfer_coverage=(
            len(outcomes)
            if minimum_transfer_coverage is None
            else minimum_transfer_coverage
        ),
        baseline_id=baseline_id,
        baseline_content_hash=baseline_content_hash,
        required_trials=tuple((outcome.task_id, outcome.seed) for outcome in outcomes),
        required_split="promotion",
        required_evaluator_id="test-evaluator-v1",
        metric_direction=outcomes[0].direction,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_confidence=0.95,
        cost_penalty=cost_penalty,
    )


class FoundryTests(unittest.TestCase):
    def test_mines_success_and_failure_patterns_from_structured_fields(self) -> None:
        cards = (
            make_card("n1", delta=0.3),
            make_card("n2", delta=0.2),
            make_card("n3", delta=9.0, status=ExecutionStatus.INVALID, error="schema-error"),
            make_card(
                "n4",
                delta=-0.1,
                status=ExecutionStatus.ERROR,
                tags=("different attempt",),
                error="schema-error",
            ),
        )
        patterns = TraceMiner(minimum_support=2, minimum_delta=0.1).mine(cards)
        success = next(pattern for pattern in patterns if pattern.kind is PatternKind.HIGH_YIELD)
        failure = next(
            pattern
            for pattern in patterns
            if pattern.kind is PatternKind.FAILURE and pattern.change_tags == ("error-schema-error",)
        )

        self.assertEqual(success.change_tags, ("residual-slicing",))
        self.assertEqual(success.support, 2)
        self.assertAlmostEqual(success.mean_delta, 0.25)
        self.assertEqual(failure.support, 2)
        self.assertEqual(failure.error_signatures, ("schema-error",))
        self.assertNotEqual(success.pattern_id, failure.pattern_id)

    def test_synthesizes_passport_with_lineage_and_executes_callback(self) -> None:
        cards = (make_card("n1", delta=0.3), make_card("n2", delta=0.2))
        pattern = TraceMiner(minimum_support=2, minimum_delta=0.1).mine_high_yield(
            cards
        )[0]
        draft = OperatorSynthesizer(minimum_tasks=2).synthesize(pattern)
        operator = OperatorCompiler().compile(
            draft,
            callback=lambda context, model, budget: {"improved": True, "task": context["task"]["task_id"]},
        )

        self.assertEqual(operator.passport.evidence_ids, (cards[0].card_id, cards[1].card_id))
        self.assertEqual(operator.passport.parents, ("improve@1.0.0",))
        self.assertIn("trajectory_mining", draft.source)
        self.assertEqual(operator.passport.generator_type, "prompt_template")

        from evolsa.core import TaskState

        artifact = operator.propose(
            {"task": {"task_id": "task-a"}},
            model=object(),  # callback operators do not invoke the model
            budget=Budget(max_tokens=100, max_exec_seconds=10.0),
        )
        self.assertIn('"improved":true', artifact.content)
        self.assertEqual(artifact.created_by, operator.passport.operator_id)

    def test_success_status_with_failed_guardrail_cannot_seed_high_yield(self) -> None:
        unsafe = (
            replace(
                make_card("n1", delta=9.0),
                valid=False,
                validity_reason="guardrail",
                card_id="",
            ),
            replace(
                make_card("n2", delta=9.0),
                valid=False,
                validity_reason="guardrail",
                card_id="",
            ),
        )
        self.assertEqual(
            TraceMiner(minimum_support=2, minimum_delta=0.1).mine_high_yield(unsafe),
            (),
        )


class LabTests(unittest.TestCase):
    def test_lower_is_better_lift_is_direction_aware(self) -> None:
        outcome = PairedOutcome(
            task_id="latency-task",
            seed=0,
            candidate_score=2.0,
            baseline_score=5.0,
            candidate_valid=True,
            baseline_valid=True,
            candidate_cost=0.0,
            baseline_cost=0.0,
            direction="lower_is_better",
        )
        self.assertEqual(outcome.lift, 3.0)

    def test_forged_confidence_and_cost_aggregates_are_quarantined(self) -> None:
        budget = Budget(max_tokens=8, max_exec_seconds=1.0, max_attempts=1)
        outcomes = tuple(
            replace(outcome, candidate_cost=100.0, baseline_cost=0.0)
            for outcome in make_outcomes((0.0, 0.0, 0.0, 0.004))
        )
        report = aggregate_lab_report(
            "candidate@0.1.0",
            "baseline@0.1.0",
            outcomes,
            candidate_budgets=(budget,) * 4,
            baseline_budgets=(budget,) * 4,
            bootstrap_samples=500,
            bootstrap_seed=9,
        )
        forged = replace(
            report,
            lcb95=0.0005,
            cost_adjusted_lift=0.001,
            report_id="",
        )
        result = PromotionGate(
            make_policy(
                outcomes,
                baseline_id="baseline@0.1.0",
                bootstrap_samples=500,
                bootstrap_seed=9,
            )
        ).evaluate(forged, operator="candidate@0.1.0")
        self.assertEqual(result.decision, PromotionDecision.QUARANTINE)
        self.assertTrue(any("lcb95_does_not_match" in reason for reason in result.reasons))

    def test_invalid_scores_cannot_inflate_promotion_evidence(self) -> None:
        budget = Budget(max_tokens=8, max_exec_seconds=1.0, max_attempts=1)
        valid = make_outcomes((0.0, 0.0, 0.0, 0.0))
        invalid = tuple(
            replace(
                outcome,
                task_id=f"invalid-{index}",
                seed=index + 10,
                candidate_score=1_000_000.0,
                baseline_score=0.0,
                candidate_valid=False,
                baseline_valid=False,
            )
            for index, outcome in enumerate(make_outcomes((0.0, 0.0, 0.0, 0.0)))
        )
        outcomes = (*valid, *invalid)
        report = aggregate_lab_report(
            "candidate@0.1.0",
            "baseline@0.1.0",
            outcomes,
            candidate_budgets=(budget,) * len(outcomes),
            baseline_budgets=(budget,) * len(outcomes),
            bootstrap_samples=200,
        )
        self.assertEqual(report.mean_lift, 0.0)
        self.assertEqual(report.lcb95, 0.0)
        self.assertEqual(report.cost_adjusted_lift, 0.0)
        decision = PromotionGate(
            make_policy(
                outcomes,
                baseline_id="baseline@0.1.0",
                bootstrap_samples=200,
                minimum_lcb_lift=0.01,
                minimum_cost_adjusted_lift=0.01,
                minimum_transfer_coverage=4,
            )
        ).evaluate(report, operator="candidate@0.1.0")
        self.assertEqual(decision.decision, PromotionDecision.REJECT)

    def test_paired_bootstrap_is_seeded_and_resamples_differences(self) -> None:
        candidates = (1.2, 1.4, 0.9, 1.1)
        baselines = (1.0, 1.0, 1.0, 1.0)
        first = paired_bootstrap_ci(candidates, baselines, samples=2_000, seed=17)
        second = paired_bootstrap_ci(candidates, baselines, samples=2_000, seed=17)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.mean, 0.15)
        lower, upper = first
        self.assertLessEqual(lower, first.mean)
        self.assertGreaterEqual(upper, first.mean)

    def test_matched_budget_and_lab_report_aggregation(self) -> None:
        outcomes = make_outcomes((0.2, 0.3, 0.25, 0.4))
        budget = Budget(max_tokens=1000, max_exec_seconds=30.0, max_attempts=1, max_cost=2.0)
        self.assertTrue(validate_matched_budget(budget, budget))
        self.assertFalse(validate_matched_budget(budget, replace(budget, max_tokens=1001)))

        trials = tuple(
            PairedTrial(
                outcome=outcome,
                candidate_budget=budget,
                baseline_budget=budget,
                candidate_parent_id="same-parent",
                baseline_parent_id="same-parent",
                candidate_model_id="same-model",
                baseline_model_id="same-model",
                candidate_sampling={"temperature": 0},
                baseline_sampling={"temperature": 0},
                split="promotion",
                evaluator_id="test-evaluator-v1",
            )
            for outcome in outcomes
        )
        report = LabReportBuilder(
            "candidate@0.1.0", "improve@1.0.0", bootstrap_samples=2_000, bootstrap_seed=11
        ).build(trials)
        self.assertTrue(report.matched_budget)
        self.assertEqual(report.transfer_coverage, 4)
        self.assertAlmostEqual(report.mean_lift, 0.2875)
        self.assertAlmostEqual(report.cost_adjusted_lift, report.mean_lift)

    def test_missing_or_contradictory_budget_evidence_fails_closed(self) -> None:
        outcomes = make_outcomes((0.3, 0.3, 0.3, 0.3))
        no_evidence = aggregate_lab_report(
            "candidate@0.1.0", "improve@1.0.0", outcomes, bootstrap_samples=100
        )
        self.assertFalse(no_evidence.matched_budget)

        budget = Budget(max_tokens=100)
        mismatch = aggregate_lab_report(
            "candidate@0.1.0",
            "improve@1.0.0",
            outcomes,
            candidate_budgets=(budget,) * 4,
            baseline_budgets=(replace(budget, max_tokens=101),) * 4,
            matched_budget=True,
            bootstrap_samples=100,
        )
        self.assertFalse(mismatch.matched_budget)

    def test_gate_promotes_only_complete_safe_evidence(self) -> None:
        outcomes = make_outcomes((0.3, 0.3, 0.3, 0.3))
        budget = Budget(max_tokens=100)
        report = aggregate_lab_report(
            "candidate@0.1.0",
            "improve@1.0.0",
            outcomes,
            candidate_budgets=(budget,) * 4,
            baseline_budgets=(budget,) * 4,
            bootstrap_samples=200,
        )
        gate = PromotionGate(
            make_policy(
                outcomes,
                baseline_id="improve@1.0.0",
                bootstrap_samples=200,
                minimum_lcb_lift=0.1,
                minimum_cost_adjusted_lift=0.1,
            )
        )
        self.assertEqual(gate.evaluate(report).decision, PromotionDecision.PROMOTE)
        self.assertEqual(
            gate.evaluate(report, security_issues=("forbidden-scope-write",)).decision,
            PromotionDecision.QUARANTINE,
        )
        self.assertEqual(
            gate.evaluate(replace(report, matched_budget=False, report_id="")).decision,
            PromotionDecision.QUARANTINE,
        )

        losing = aggregate_lab_report(
            "candidate@0.1.0",
            "improve@1.0.0",
            make_outcomes((-0.2, -0.1, -0.3, -0.2)),
            candidate_budgets=(budget,) * 4,
            baseline_budgets=(budget,) * 4,
            bootstrap_samples=200,
        )
        self.assertEqual(gate.evaluate(losing).decision, PromotionDecision.REJECT)

    def test_candidate_cannot_choose_weak_statistics_cost_or_protocol(self) -> None:
        budget = Budget(max_tokens=100)
        outcomes = make_outcomes((0.2, 0.2, 0.2, 0.2))
        weak_statistics = aggregate_lab_report(
            "candidate@0.1.0",
            "improve@1.0.0",
            outcomes,
            candidate_budgets=(budget,) * 4,
            baseline_budgets=(budget,) * 4,
            bootstrap_samples=1,
            bootstrap_seed=3,
        )
        protected_policy = make_policy(
            outcomes,
            baseline_id="improve@1.0.0",
            bootstrap_samples=2_000,
            bootstrap_seed=7,
            minimum_lcb_lift=0.05,
            minimum_cost_adjusted_lift=0.05,
        )
        weak_result = PromotionGate(protected_policy).evaluate(weak_statistics)
        self.assertIsNot(weak_result.decision, PromotionDecision.PROMOTE)
        self.assertTrue(any("bootstrap_samples" in reason for reason in weak_result.reasons))

        no_cost_penalty = aggregate_lab_report(
            "candidate@0.1.0",
            "improve@1.0.0",
            tuple(
                replace(outcome, candidate_cost=100.0, baseline_cost=0.0)
                for outcome in outcomes
            ),
            candidate_budgets=(budget,) * 4,
            baseline_budgets=(budget,) * 4,
            bootstrap_samples=2_000,
            bootstrap_seed=7,
            cost_penalty=0.0,
        )
        cost_result = PromotionGate(protected_policy).evaluate(no_cost_penalty)
        self.assertEqual(cost_result.decision, PromotionDecision.QUARANTINE)
        self.assertTrue(
            any("cost_penalty_policy_mismatch" in reason for reason in cost_result.reasons)
        )

        mining_outcomes = tuple(replace(outcome, split="mine") for outcome in outcomes)
        mining_report = aggregate_lab_report(
            "candidate@0.1.0",
            "improve@1.0.0",
            mining_outcomes,
            candidate_budgets=(budget,) * 4,
            baseline_budgets=(budget,) * 4,
            bootstrap_samples=2_000,
            bootstrap_seed=7,
        )
        split_result = PromotionGate(protected_policy).evaluate(mining_report)
        self.assertIsNot(split_result.decision, PromotionDecision.PROMOTE)
        self.assertTrue(any("required_split_mismatch" in reason for reason in split_result.reasons))

    def test_direction_duplicate_trials_and_unknown_guardrails_fail_closed(self) -> None:
        budget = Budget(max_tokens=100)
        outcomes = make_outcomes((0.2, 0.2, 0.2, 0.2))
        flipped = tuple(replace(outcome, direction="lower_is_better") for outcome in outcomes)
        flipped_report = aggregate_lab_report(
            "candidate@0.1.0",
            "improve@1.0.0",
            flipped,
            candidate_budgets=(budget,) * 4,
            baseline_budgets=(budget,) * 4,
            bootstrap_samples=200,
        )
        policy = make_policy(
            outcomes,
            baseline_id="improve@1.0.0",
            bootstrap_samples=200,
            minimum_transfer_coverage=4,
        )
        self.assertEqual(
            PromotionGate(policy).evaluate(flipped_report).decision,
            PromotionDecision.QUARANTINE,
        )

        duplicated = (*outcomes[:3], replace(outcomes[3], task_id=outcomes[0].task_id, seed=0))
        duplicate_report = aggregate_lab_report(
            "candidate@0.1.0",
            "improve@1.0.0",
            duplicated,
            candidate_budgets=(budget,) * 4,
            baseline_budgets=(budget,) * 4,
            bootstrap_samples=200,
        )
        self.assertEqual(
            PromotionGate(policy).evaluate(duplicate_report).decision,
            PromotionDecision.QUARANTINE,
        )

        improve = next(
            operator.passport for operator in seed_operators() if operator.passport.name == "improve"
        )
        unknown_contract = replace(
            improve,
            verification=replace(
                improve.verification,
                guardrails=(*improve.verification.guardrails, "unimplemented_safety_oracle"),
            ),
        )
        valid_report = aggregate_lab_report(
            unknown_contract.operator_id,
            improve.verification.baseline,
            outcomes,
            candidate_budgets=(budget,) * 4,
            baseline_budgets=(budget,) * 4,
            bootstrap_samples=200,
        )
        baseline = next(
            operator.passport
            for operator in seed_operators()
            if operator.passport.operator_id == improve.verification.baseline
        )
        contract_policy = make_policy(
            outcomes,
            baseline_id=baseline.operator_id,
            baseline_content_hash=baseline.content_hash,
            bootstrap_samples=200,
        )
        result = PromotionGate(contract_policy).evaluate(
            valid_report,
            operator=unknown_contract,
        )
        self.assertEqual(result.decision, PromotionDecision.QUARANTINE)
        self.assertTrue(
            any("unsupported_verification_guardrail" in reason for reason in result.reasons)
        )


class RegistryTests(unittest.TestCase):
    release_secret = b"registry-test-release-secret"
    release_actor = "registry-test-reviewer"

    def test_lifecycle_transition_drops_stale_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            registry = LocalRegistry(temporary)
            passport = replace(self._passport(), signature="hmac-sha256:test:" + "a" * 64)
            registry.register(passport)
            registry.transition(passport.operator_id, OperatorLifecycle.SHADOW)
            self.assertEqual(registry.get(passport.operator_id).signature, "")

    def _operator(self, version: str = "0.1.0"):
        pattern = TraceMiner(minimum_support=2, minimum_delta=0.1).mine_high_yield(
            (make_card("n1", delta=0.3), make_card("n2", delta=0.2))
        )[0]
        draft = OperatorSynthesizer(version=version, minimum_tasks=2).synthesize(
            pattern, name="residual-slice"
        )
        return OperatorCompiler().compile(draft)

    def _passport(self, version: str = "0.1.0"):
        return self._operator(version).passport

    def _promotion(self, passport):
        budget = Budget(max_tokens=32, max_exec_seconds=1.0, max_attempts=1)
        outcomes = make_outcomes((0.2, 0.2))
        report = aggregate_lab_report(
            passport.operator_id,
            passport.verification.baseline,
            outcomes,
            candidate_budgets=(budget, budget),
            baseline_budgets=(budget, budget),
            bootstrap_samples=100,
            bootstrap_seed=3,
        )
        baseline = next(
            operator.passport
            for operator in seed_operators()
            if operator.passport.operator_id == passport.verification.baseline
        )
        policy = make_policy(
            outcomes,
            baseline_id=baseline.operator_id,
            baseline_content_hash=baseline.content_hash,
            bootstrap_samples=100,
            bootstrap_seed=3,
            minimum_lcb_lift=0.1,
            minimum_cost_adjusted_lift=0.1,
            minimum_transfer_coverage=2,
        )
        return PromotionGate(policy).evaluate(report, operator=passport), report, policy

    def _make_stable(self, registry: LocalRegistry, operator):
        passport = operator.passport
        baseline = next(
            seed.passport
            for seed in seed_operators()
            if seed.passport.operator_id == passport.verification.baseline
        )
        registry.register(baseline)
        registry.register_executable(operator)
        registry.transition(passport.operator_id, OperatorLifecycle.SHADOW)
        registry.transition(passport.operator_id, OperatorLifecycle.CANDIDATE)
        candidate = registry.get(passport.operator_id)
        gate, report, policy = self._promotion(candidate)
        record = registry.transition(
            passport.operator_id,
            OperatorLifecycle.STABLE,
            gate_result=gate,
            lab_report=report,
            promotion_policy=policy,
        )
        stable = registry.get(passport.operator_id)
        signed = sign_passport(stable, "release", self.release_secret)
        kernel = TrustKernel(
            {"release": self.release_secret},
            release_signers={"release"},
            release_principals={self.release_actor},
        )
        return record, signed, kernel

    def test_content_addressing_lifecycle_install_and_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = LocalRegistry(directory)
            first = self._operator("0.1.0")
            first_record, first_signed, kernel = self._make_stable(registry, first)
            self.assertTrue(registry.object_path(first_record.content_hash).is_file())
            registry.install(
                first.passport.operator_id,
                signed_passport=first_signed,
                trust_kernel=kernel,
                actor=self.release_actor,
            )

            second = self._operator("0.2.0")
            second_record, second_signed, kernel = self._make_stable(registry, second)
            self.assertEqual(
                tuple(
                    record.lifecycle
                    for record in registry.history(second.passport.operator_id)
                ),
                (
                    OperatorLifecycle.PROPOSED,
                    OperatorLifecycle.SHADOW,
                    OperatorLifecycle.CANDIDATE,
                    OperatorLifecycle.STABLE,
                ),
            )
            self.assertRegex(second_record.gate_report_id, r"^[0-9a-f]{64}$")
            registry.install(
                second.passport.operator_id,
                signed_passport=second_signed,
                trust_kernel=kernel,
                actor=self.release_actor,
            )
            self.assertEqual(
                registry.installed(
                    "residual-slice",
                    trust_kernel=kernel,
                    actor=self.release_actor,
                ).version,
                "0.2.0",
            )

            rolled_back = registry.rollback(
                "residual-slice",
                trust_kernel=kernel,
                actor=self.release_actor,
            )
            self.assertEqual(rolled_back.operator_id, first.passport.operator_id)
            self.assertEqual(
                registry.resolve(
                    "residual-slice",
                    trust_kernel=kernel,
                    actor=self.release_actor,
                ).version,
                "0.1.0",
            )

    def test_registry_rejects_ungated_stable_transition_and_version_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = LocalRegistry(directory)
            passport = self._passport()
            registry.register(passport)
            registry.transition(passport.operator_id, OperatorLifecycle.SHADOW)
            registry.transition(passport.operator_id, OperatorLifecycle.CANDIDATE)
            with self.assertRaises(LifecycleTransitionError):
                registry.transition(passport.operator_id, OperatorLifecycle.STABLE)

            conflicting = replace(passport, intent="different immutable content")
            with self.assertRaises(RegistryConflictError):
                registry.register(conflicting)

    def test_quarantine_invalidates_an_active_installation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = LocalRegistry(directory)
            operator = self._operator()
            _, signed, kernel = self._make_stable(registry, operator)
            registry.install(
                operator.passport.operator_id,
                signed_passport=signed,
                trust_kernel=kernel,
                actor=self.release_actor,
            )
            registry.quarantine(operator.passport.operator_id)
            with self.assertRaises(LifecycleTransitionError):
                registry.installed(
                    operator.passport.name,
                    trust_kernel=kernel,
                    actor=self.release_actor,
                )

    def test_registry_detects_object_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = LocalRegistry(directory)
            passport = self._passport()
            record = registry.register(passport)
            path = registry.object_path(record.content_hash)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["passport"]["intent"] = "tampered"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RegistryIntegrityError):
                registry.get(passport.operator_id)


if __name__ == "__main__":
    unittest.main()
