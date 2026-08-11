from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

import evolsa.registry.local as registry_local
from evolsa.core import (
    Budget,
    EffectScope,
    GateResult,
    OperatorLifecycle,
    OperatorPassport,
    OperatorVerification,
    PairedOutcome,
    PromotionDecision,
    PromotionPolicy,
)
from evolsa.core.hashing import sha256_json, sha256_text
from evolsa.foundry import CallbackOperator
from evolsa.kernel import TrustKernel, sign_passport
from evolsa.lab import PromotionGate, aggregate_lab_report
from evolsa.registry import (
    LifecycleTransitionError,
    LocalRegistry,
    RegistryIntegrityError,
)


_SECRET = b"registry-evidence-test-release-key"
_PROMPT = "Return a JSON object containing the improved artifact."
_BUDGET = Budget(max_tokens=128, max_exec_seconds=5, max_attempts=1, max_cost=1)


class _Model:
    def generate(self, messages, budget):
        del messages, budget
        return '{"content":"restart-safe"}'


def _operator(version: str = "0.1.0") -> CallbackOperator:
    passport = OperatorPassport(
        name="evidence-op",
        version=version,
        intent="Improve an artifact under bounded, externally evaluated conditions.",
        inputs=("task", "artifact"),
        preconditions=("artifact_exists",),
        expected_effects=(EffectScope.TASK_ARTIFACT,),
        forbidden_effects=(
            EffectScope.EVALUATOR,
            EffectScope.SANDBOX,
            EffectScope.BUDGET,
            EffectScope.PROMOTION_GATE,
            EffectScope.EVENT_LOG,
            EffectScope.RELEASE_PERMISSION,
        ),
        context_policy="minimal-v1",
        generator_type="prompt_template",
        generator_ref=f"sha256:{sha256_text(_PROMPT)}",
        budget=_BUDGET,
        verification=OperatorVerification(
            baseline="baseline@1.0.0",
            minimum_tasks=4,
        ),
        evidence_ids=("e" * 64,),
    )
    return CallbackOperator(passport=passport, prompt_template=_PROMPT)


def _report(
    operator_id: str,
    *,
    candidate_score: float = 1.0,
    baseline_score: float = 0.75,
):
    outcomes = tuple(
        PairedOutcome(
            task_id=f"promotion-{seed}",
            seed=seed,
            candidate_score=candidate_score,
            baseline_score=baseline_score,
            candidate_valid=True,
            baseline_valid=True,
            candidate_cost=0.1,
            baseline_cost=0.1,
            split="promotion",
            evaluator_id="registry-test-evaluator-v1",
            candidate_parent_id=f"parent-{seed}",
            baseline_parent_id=f"parent-{seed}",
            candidate_model_id="fixed-model-v1",
            baseline_model_id="fixed-model-v1",
            candidate_sampling={"temperature": 0},
            baseline_sampling={"temperature": 0},
        )
        for seed in range(4)
    )
    return aggregate_lab_report(
        operator_id,
        "baseline@1.0.0",
        outcomes,
        candidate_budgets=(_BUDGET,) * 4,
        baseline_budgets=(_BUDGET,) * 4,
        bootstrap_samples=200,
        bootstrap_seed=13,
    )


def _policy() -> PromotionPolicy:
    baseline = _baseline()
    return PromotionPolicy(
        minimum_lcb_lift=0.1,
        minimum_cost_adjusted_lift=0.1,
        maximum_regression=0.0,
        minimum_transfer_coverage=4,
        baseline_id=baseline.operator_id,
        baseline_content_hash=baseline.content_hash,
        required_trials=tuple((f"promotion-{seed}", seed) for seed in range(4)),
        required_split="promotion",
        required_evaluator_id="registry-test-evaluator-v1",
        bootstrap_samples=200,
        bootstrap_seed=13,
        bootstrap_confidence=0.95,
        cost_penalty=1.0,
    )


def _kernel() -> TrustKernel:
    return TrustKernel(
        {"release": _SECRET},
        release_signers={"release"},
        release_principals={"reviewer"},
    )


def _baseline() -> OperatorPassport:
    return replace(
        _operator().passport,
        name="baseline",
        version="1.0.0",
        intent="Trusted fixed baseline for matched promotion trials.",
        parents=(),
        verification=OperatorVerification(
            baseline="draft@1.0.0",
            minimum_tasks=1,
        ),
        evidence_ids=("b" * 64,),
        signature="",
    )


def _candidate(registry: LocalRegistry, operator: CallbackOperator) -> OperatorPassport:
    registry.register(_baseline())
    registry.register_executable(operator)
    registry.transition(operator.passport.operator_id, OperatorLifecycle.SHADOW)
    registry.transition(operator.passport.operator_id, OperatorLifecycle.CANDIDATE)
    return registry.get(operator.passport.operator_id)


def _promote(registry: LocalRegistry, candidate: OperatorPassport):
    report = _report(candidate.operator_id)
    policy = _policy()
    gate = PromotionGate(policy).evaluate(report, operator=candidate)
    record = registry.promote(candidate.operator_id, gate, report, policy)
    return record, report, policy, gate


class RegistryEvidenceTests(unittest.TestCase):
    def test_process_lock_fallback_works_without_fcntl(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            registry_local, "_fcntl", None
        ):
            first = LocalRegistry(directory)
            second = LocalRegistry(directory)
            operator = _operator()
            first.register_executable(operator)
            self.assertEqual(second.get(operator.passport.operator_id), operator.passport)

    def test_forged_gate_cannot_create_stable_operator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = LocalRegistry(directory)
            candidate = _candidate(registry, _operator())
            report = _report(candidate.operator_id, candidate_score=0.5)
            policy = _policy()
            forged = GateResult(
                decision=PromotionDecision.PROMOTE,
                operator_id=candidate.operator_id,
                report_id=report.report_id,
                reasons=("all promotion gates passed",),
                policy_hash=sha256_json(policy),
            )
            with self.assertRaises(LifecycleTransitionError):
                registry.promote(candidate.operator_id, forged, report, policy)
            self.assertIs(registry.get(candidate.operator_id).lifecycle, OperatorLifecycle.CANDIDATE)

    def test_receipt_tampering_is_rejected_before_install(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = LocalRegistry(directory)
            candidate = _candidate(registry, _operator())
            record, _, _, _ = _promote(registry, candidate)
            stable = registry.get(candidate.operator_id)
            signed = sign_passport(stable, "release", _SECRET)
            receipt_path = registry.promotion_receipt_path(record.promotion_receipt_id)
            payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            payload["lab_report"]["mean_lift"] = 99.0
            receipt_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RegistryIntegrityError):
                registry.install(
                    candidate.operator_id,
                    signed_passport=signed,
                    trust_kernel=_kernel(),
                    actor="reviewer",
                )

    def test_restart_loads_and_executes_signed_installed_prompt_operator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = LocalRegistry(directory)
            candidate = _candidate(registry, _operator())
            record, _, _, _ = _promote(registry, candidate)
            self.assertEqual(
                registry.promotion_receipt(record.promotion_receipt_id).stable_content_hash,
                record.content_hash,
            )
            signed = sign_passport(registry.get(candidate.operator_id), "release", _SECRET)
            registry.install(
                candidate.operator_id,
                signed_passport=signed,
                trust_kernel=_kernel(),
                actor="reviewer",
            )

            reopened = LocalRegistry(directory)
            loaded = reopened.load_installed_operator(
                "evidence-op",
                trust_kernel=_kernel(),
                actor="reviewer",
            )
            self.assertEqual(loaded.passport.signature, signed.signature)
            artifact = loaded.propose({}, _Model(), _BUDGET)
            self.assertEqual(artifact.content, "restart-safe")
            self.assertEqual(artifact.created_by, candidate.operator_id)

    def test_generator_tampering_is_rejected_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = LocalRegistry(directory)
            candidate = _candidate(registry, _operator())
            _promote(registry, candidate)
            stable = registry.get(candidate.operator_id)
            registry.install(
                candidate.operator_id,
                signed_passport=sign_passport(stable, "release", _SECRET),
                trust_kernel=_kernel(),
                actor="reviewer",
            )
            generator_path = registry.generator_path(sha256_text(_PROMPT))
            payload = json.loads(generator_path.read_text(encoding="utf-8"))
            payload["content"] = "tampered prompt"
            generator_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(RegistryIntegrityError):
                LocalRegistry(directory).load_installed_operator(
                    "evidence-op",
                    trust_kernel=_kernel(),
                    actor="reviewer",
                )

    def test_install_requires_exact_signed_stable_passport(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = LocalRegistry(directory)
            candidate = _candidate(registry, _operator())
            _promote(registry, candidate)
            stable = registry.get(candidate.operator_id)
            with self.assertRaises(LifecycleTransitionError):
                registry.install(
                    candidate.operator_id,
                    signed_passport=stable,
                    trust_kernel=_kernel(),
                    actor="reviewer",
                )
            wrong_signature = sign_passport(
                replace(stable, intent="different stable object"),
                "release",
                _SECRET,
            )
            with self.assertRaises(LifecycleTransitionError):
                registry.install(
                    candidate.operator_id,
                    signed_passport=wrong_signature,
                    trust_kernel=_kernel(),
                    actor="reviewer",
                )

    def test_registry_reads_verify_index_hash_and_evidence_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = LocalRegistry(directory)
            baseline = _baseline()
            registry.register(baseline)
            index = json.loads(registry.index_path.read_text(encoding="utf-8"))
            index["operators"][baseline.operator_id]["evidence_ids"] = ["0" * 64]
            registry.index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaises(RegistryIntegrityError):
                registry.get(baseline.operator_id)
            with self.assertRaises(RegistryIntegrityError):
                registry.list()

        with tempfile.TemporaryDirectory() as directory:
            registry = LocalRegistry(directory)
            baseline = _baseline()
            registry.register(baseline)
            index = json.loads(registry.index_path.read_text(encoding="utf-8"))
            index["operators"][baseline.operator_id]["content_hash"] = "f" * 64
            registry.index_path.write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaises(RegistryIntegrityError):
                registry.list()

    def test_restart_rejects_a_forged_persisted_passport_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = LocalRegistry(directory)
            candidate = _candidate(registry, _operator())
            _promote(registry, candidate)
            stable = registry.get(candidate.operator_id)
            registry.install(
                candidate.operator_id,
                signed_passport=sign_passport(stable, "release", _SECRET),
                trust_kernel=_kernel(),
                actor="reviewer",
            )
            installation_path = registry.installations_dir / "evidence-op.json"
            installation = json.loads(installation_path.read_text(encoding="utf-8"))
            signature = installation["active"]["passport_signature"]
            installation["active"]["passport_signature"] = (
                signature[:-1] + ("0" if signature[-1] != "0" else "1")
            )
            installation_path.write_text(json.dumps(installation), encoding="utf-8")
            with self.assertRaises(RegistryIntegrityError):
                LocalRegistry(directory).load_installed_operator(
                    "evidence-op",
                    trust_kernel=_kernel(),
                    actor="reviewer",
                )

    def test_promotion_requires_registered_baseline_but_replays_historical_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = LocalRegistry(directory)
            operator = _operator()
            registry.register_executable(operator)
            registry.transition(operator.passport.operator_id, OperatorLifecycle.SHADOW)
            registry.transition(operator.passport.operator_id, OperatorLifecycle.CANDIDATE)
            candidate = registry.get(operator.passport.operator_id)
            report = _report(candidate.operator_id)
            policy = _policy()
            gate = PromotionGate(policy).evaluate(report, operator=candidate)
            self.assertEqual(gate.decision, PromotionDecision.PROMOTE)
            with self.assertRaises(LifecycleTransitionError):
                registry.promote(candidate.operator_id, gate, report, policy)

        with tempfile.TemporaryDirectory() as directory:
            registry = LocalRegistry(directory)
            candidate = _candidate(registry, _operator())
            baseline = _baseline()
            registry.transition(baseline.operator_id, OperatorLifecycle.REJECTED)
            report = _report(candidate.operator_id)
            policy = _policy()
            gate = PromotionGate(policy).evaluate(report, operator=candidate)
            with self.assertRaises(LifecycleTransitionError):
                registry.promote(candidate.operator_id, gate, report, policy)

        with tempfile.TemporaryDirectory() as directory:
            registry = LocalRegistry(directory)
            candidate = _candidate(registry, _operator())
            _promote(registry, candidate)
            baseline = _baseline()
            registry.transition(baseline.operator_id, OperatorLifecycle.SHADOW)
            self.assertEqual(
                registry.get(candidate.operator_id).lifecycle,
                OperatorLifecycle.STABLE,
            )


if __name__ == "__main__":
    unittest.main()
