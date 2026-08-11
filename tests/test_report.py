from __future__ import annotations

import unittest

from evolsa.core import (
    Budget,
    EffectScope,
    ExecutionStatus,
    ExperienceCard,
    GateResult,
    LabReport,
    OperatorPassport,
    OperatorVerification,
    PairedOutcome,
    PromotionDecision,
    PromotionPolicy,
)
from evolsa.core.hashing import sha256_json
from evolsa.report import render_lineage_html, render_review_markdown


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.passport = OperatorPassport(
            name="normalize-punctuation",
            version="0.1.0",
            intent="normalize punctuation before routing",
            inputs=("task_state", "candidate"),
            preconditions=("punctuation_errors_observed",),
            expected_effects=(EffectScope.TASK_ARTIFACT,),
            forbidden_effects=(EffectScope.EVALUATOR, EffectScope.PROMOTION_GATE),
            context_policy="branch-local-v1",
            generator_type="prompt_template",
            generator_ref="operators/normalize-punctuation.txt",
            budget=Budget(max_tokens=100, max_attempts=1),
            verification=OperatorVerification(baseline="improve@1.0.0", minimum_tasks=1),
        )
        self.card = ExperienceCard(
            node_id="n1", step_id=1, operator_id=self.passport.operator_id, parent_node_ids=("n0",),
            generation_id=1, score=0.9, utility=0.9, status=ExecutionStatus.SUCCESS,
            error_signature="", runtime_seconds=0.1, token_count=20, cost=0.0,
            method_family="normalization", family_count_before=0, delta_vs_parent=0.2,
            novelty_score=1.0, rank=1, current_best=True, selection_utility=2.2,
            valid=True,
        )
        outcome = PairedOutcome("t", 1, 0.9, 0.7, True, True, 1.0, 1.0)
        self.report = LabReport(
            operator_id=self.passport.operator_id, baseline_id="improve@1.0.0", outcomes=(outcome,),
            mean_lift=0.2, lcb95=0.1, ucb95=0.3, valid_rate=1.0, baseline_valid_rate=1.0,
            cost_adjusted_lift=0.2, transfer_coverage=1, worst_regression=0.0, matched_budget=True,
        )
        policy = PromotionPolicy(minimum_transfer_coverage=1)
        self.gate = GateResult(PromotionDecision.PROMOTE, self.passport.operator_id, self.report.report_id,
                               ("all gates passed",), sha256_json(policy))

    def test_html_is_self_contained_and_escapes_metadata(self) -> None:
        rendered = render_lineage_html(run_id="demo", cards=(self.card,), passports=(self.passport,),
                                       reports=(self.report,), gates=(self.gate,), metadata={"x": "</script><b>"})
        self.assertIn("Operator lineage", rendered)
        self.assertIn("Operator parent lineage graph", rendered)
        self.assertIn("<svg", rendered)
        self.assertIn('id="arrow"', rendered)
        self.assertIn("normalize-punctuation@0.1.0", rendered)
        self.assertNotIn("</script><b>", rendered)
        self.assertNotIn("<script", rendered)

    def test_markdown_is_evidence_carrying(self) -> None:
        rendered = render_review_markdown(run_id="demo", cards=(self.card,), reports=(self.report,), gates=(self.gate,))
        self.assertIn("mean lift +0.2000", rendered)
        self.assertIn("**PROMOTE**", rendered)


if __name__ == "__main__":
    unittest.main()
