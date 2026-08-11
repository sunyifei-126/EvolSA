from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from evolsa.cli import main
from evolsa.core import Budget, LabReport, PairedOutcome
from evolsa.core.serde import pretty_json
from evolsa.core.hashing import sha256_json
from evolsa.cli import _official_promotion_policy
from evolsa.lab import aggregate_lab_report


def _invoke(*arguments: str) -> tuple[int, dict[str, object] | None, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        status = main(arguments)
    payload = json.loads(stdout.getvalue()) if stdout.getvalue() else None
    return status, payload, stderr.getvalue()


def _report() -> LabReport:
    budget = Budget(max_tokens=100, max_exec_seconds=5, max_attempts=1)
    outcomes = tuple(
        PairedOutcome(
            task_id=f"promotion-{index}",
            seed=index,
            candidate_score=1.0,
            baseline_score=0.8,
            candidate_valid=True,
            baseline_valid=True,
            candidate_cost=0.0,
            baseline_cost=0.0,
            split="promotion",
            evaluator_id="test-evaluator-v1",
            candidate_parent_id="same-parent",
            baseline_parent_id="same-parent",
            candidate_model_id="same-model",
            baseline_model_id="same-model",
            candidate_sampling={"temperature": 0},
            baseline_sampling={"temperature": 0},
        )
        for index in range(4)
    )
    return aggregate_lab_report(
        "hyphen-to-space@1.0.0",
        "seed-policy@1.0.0",
        outcomes,
        candidate_budgets=(budget,) * 4,
        baseline_budgets=(budget,) * 4,
        bootstrap_samples=200,
    )


def _write_report_bundle(root: Path, report: LabReport) -> None:
    (root / "lab-report.json").write_text(pretty_json(report) + "\n", encoding="utf-8")
    (root / "operator-passport.json").write_text(
        json.dumps({"name": "hyphen-to-space", "version": "1.0.0"}),
        encoding="utf-8",
    )


class CLIReleaseHardeningTests(unittest.TestCase):
    def test_search_runs_all_fixed_seed_operators_and_improves_score(self) -> None:
        status, payload, error = _invoke("search", "--seed", "3")
        self.assertEqual(status, 0, error)
        assert payload is not None
        self.assertEqual(payload["schema_version"], "evolsa.search-result/v1")
        self.assertEqual(payload["node_count"], 5)
        self.assertEqual(
            payload["operation_plan"],
            ["draft", "improve", "debug", "crossover"],
        )
        self.assertEqual(payload["best_score"], 1.0)
        self.assertEqual(payload["nodes"][1]["status"], "invalid")
        self.assertEqual(payload["nodes"][3]["operator_id"], "debug@1.0.0")
        self.assertIn("hyphen_to_space", payload["best_artifact"]["normalizers"])

    def test_promote_parser_has_no_threshold_or_unmatched_escape_hatches(self) -> None:
        for option in ("--minimum-lcb-lift", "--allow-unmatched-budget"):
            stderr = io.StringIO()
            with self.assertRaisesRegex(SystemExit, "2"), redirect_stderr(stderr):
                main(("promote", "/not/used", option, "-999"))
            self.assertIn("unrecognized arguments", stderr.getvalue())

    def test_unsealed_report_produces_advisory_decision_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_report_bundle(root, _report())
            status, payload, error = _invoke("promote", str(root))
        self.assertEqual(status, 0, error)
        assert payload is not None
        self.assertEqual(payload["evidence_mode"], "advisory-unsealed")
        self.assertIn("advisory_decision", payload)
        self.assertNotIn("decision", payload)
        self.assertIs(payload["replay_verified"], False)
        self.assertTrue(payload["policy"]["require_matched_budget"])
        self.assertGreaterEqual(payload["policy"]["minimum_lcb_lift"], 0.05)

    def test_mine_and_lab_replay_before_reading_sealed_run_evidence(self) -> None:
        receipt = {"replay_verified": True, "semantics_replayed": True}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text("{}", encoding="utf-8")

            mine_order: list[str] = []
            replay = Mock(side_effect=lambda path: mine_order.append("replay") or receipt)
            with (
                patch("evolsa.cli._load_demo_api", return_value=(Mock(), Mock(), replay)),
                patch(
                    "evolsa.cli._read_cards",
                    side_effect=lambda path: mine_order.append("read") or (),
                ),
            ):
                status, payload, error = _invoke("mine", str(root))
            self.assertEqual(status, 0, error)
            self.assertEqual(mine_order, ["replay", "read"])
            self.assertTrue(payload["replay_verified"])

            lab_order: list[str] = []
            replay = Mock(side_effect=lambda path: lab_order.append("replay") or receipt)
            with (
                patch("evolsa.cli._load_demo_api", return_value=(Mock(), Mock(), replay)),
                patch(
                    "evolsa.cli._read_lab_report",
                    side_effect=lambda path: lab_order.append("read") or _report(),
                ),
            ):
                status, payload, error = _invoke("lab", str(root))
            self.assertEqual(status, 0, error)
            self.assertEqual(lab_order, ["replay", "read"])
            self.assertTrue(payload["replay_verified"])

    def test_official_promote_reports_replayed_run_bound_gate(self) -> None:
        report = _report()
        receipt = {"replay_verified": True, "semantics_replayed": True}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            _write_report_bundle(root, report)
            (root / "gate-result.json").write_text(
                json.dumps(
                    {
                        "decision": "promote",
                        "operator_id": report.operator_id,
                        "report_id": report.report_id,
                        "reasons": ["bound gate result"],
                        "policy_hash": sha256_json(_official_promotion_policy()),
                    }
                ),
                encoding="utf-8",
            )
            replay = Mock(return_value=receipt)
            with patch("evolsa.cli._load_demo_api", return_value=(Mock(), Mock(), replay)):
                status, payload, error = _invoke("promote", str(root))
        self.assertEqual(status, 0, error)
        replay.assert_called_once_with(root.resolve())
        assert payload is not None
        self.assertEqual(payload["decision"], "promote")
        self.assertEqual(payload["decision_source"], "run-bound-gate")
        self.assertTrue(payload["replay_verified"])
        self.assertNotIn("advisory_decision", payload)

    def test_partial_run_is_not_downgraded_to_advisory_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "run.json").write_text("{}", encoding="utf-8")
            status, payload, error = _invoke("lab", str(root))
        self.assertEqual(status, 2)
        self.assertIsNone(payload)
        self.assertIn("manifest.json is missing", error)


if __name__ == "__main__":
    unittest.main()
