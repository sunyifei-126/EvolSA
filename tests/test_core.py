from __future__ import annotations

import unittest

from evolsa.core import (
    Artifact,
    Budget,
    EffectScope,
    LabReport,
    OperatorPassport,
    OperatorVerification,
    PairedOutcome,
    RunManifest,
)
from evolsa.core.serde import canonical_json


class CoreContractTests(unittest.TestCase):
    def test_run_manifest_orders_protected_scopes_deterministically(self) -> None:
        manifest = RunManifest(
            run_id="run",
            mode="rsi",
            task_pack="pack",
            seeds=(0,),
            budget=Budget(),
            model_id="model",
            operator_ids=("draft@1.0.0",),
        )
        labels = [scope.value for scope in manifest.protected_scopes]
        self.assertEqual(labels, sorted(labels))

    def test_artifacts_are_content_addressed(self) -> None:
        first = Artifact(content="x")
        second = Artifact(content="x")
        other_type = Artifact(content="x", media_type="text/plain")
        self.assertEqual(first.artifact_id, second.artifact_id)
        self.assertNotEqual(first.artifact_id, other_type.artifact_id)
        with self.assertRaisesRegex(ValueError, "does not match"):
            Artifact(content="x", artifact_id="0" * 64)

    def test_passports_cannot_claim_protected_effects(self) -> None:
        kwargs = dict(
            name="x-op", version="0.1.0", intent="perform a bounded task change",
            inputs=("candidate",), preconditions=(), forbidden_effects=(EffectScope.PROMOTION_GATE,),
            context_policy="bounded", generator_type="prompt_template", generator_ref="x.txt",
            budget=Budget(), verification=OperatorVerification(baseline="improve@1.0.0"),
        )
        with self.assertRaisesRegex(ValueError, "protected effects"):
            OperatorPassport(expected_effects=(EffectScope.EVALUATOR,), **kwargs)
        passport = OperatorPassport(expected_effects=(EffectScope.TASK_ARTIFACT,), **kwargs)
        self.assertEqual(len(passport.content_hash), 64)

    def test_canonical_json_rejects_non_finite_numbers(self) -> None:
        with self.assertRaises(ValueError):
            canonical_json({"score": float("nan")})

    def test_lab_report_id_binds_raw_outcomes(self) -> None:
        outcome = PairedOutcome("t", 1, 0.8, 0.6, True, True, 1.0, 1.0)
        report = LabReport(
            operator_id="new@0.1.0", baseline_id="improve@1.0.0", outcomes=(outcome,),
            mean_lift=0.2, lcb95=0.1, ucb95=0.3, valid_rate=1.0,
            baseline_valid_rate=1.0, cost_adjusted_lift=0.2, transfer_coverage=1,
            worst_regression=0.0, matched_budget=True,
        )
        self.assertEqual(len(report.report_id), 64)
        with self.assertRaisesRegex(ValueError, "does not match"):
            LabReport(
                operator_id=report.operator_id, baseline_id=report.baseline_id,
                outcomes=report.outcomes, mean_lift=report.mean_lift, lcb95=report.lcb95,
                ucb95=report.ucb95, valid_rate=report.valid_rate,
                baseline_valid_rate=report.baseline_valid_rate,
                cost_adjusted_lift=report.cost_adjusted_lift,
                transfer_coverage=report.transfer_coverage,
                worst_regression=report.worst_regression, matched_budget=True,
                report_id="f" * 64,
            )


if __name__ == "__main__":
    unittest.main()
