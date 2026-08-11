from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from evolsa.cli import main
from evolsa.core import Budget, ExecutionStatus, ExperienceCard, LabReport, PairedOutcome
from evolsa.core.serde import pretty_json, to_primitive
from evolsa.lab import aggregate_lab_report


def _invoke(*arguments: str) -> tuple[int, dict[str, object] | None, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        status = main(arguments)
    payload = json.loads(stdout.getvalue()) if stdout.getvalue() else None
    return status, payload, stderr.getvalue()


def _card(node_id: str, step: int) -> ExperienceCard:
    return ExperienceCard(
        node_id=node_id,
        step_id=step,
        operator_id="improve@1.0.0",
        parent_node_ids=("seed",),
        generation_id=1,
        score=1.0,
        utility=1.0,
        status=ExecutionStatus.SUCCESS,
        error_signature="",
        runtime_seconds=0.01,
        token_count=4,
        cost=0.0,
        method_family="normalization",
        family_count_before=step - 1,
        delta_vs_parent=0.2,
        novelty_score=0.5,
        rank=1,
        current_best=True,
        selection_utility=1.1,
        change_tags=("hyphen_to_space",),
        valid=True,
    )


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


class CLICommandsTests(unittest.TestCase):
    def test_version(self) -> None:
        stdout = io.StringIO()
        with self.assertRaisesRegex(SystemExit, "0"), redirect_stdout(stdout):
            main(("--version",))
        self.assertRegex(stdout.getvalue(), r"^evolsa \d+\.\d+\.\d+")

    def test_demo_and_rsi_invoke_the_offline_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for command in ("demo", "rsi"):
                output = Path(temporary) / command
                runner = Mock(return_value={"run_id": command, "decision": "promote"})
                with patch("evolsa.cli._load_demo_api", return_value=(runner, Mock(), Mock())):
                    status, payload, error = _invoke(command, "--output", str(output))
                self.assertEqual(status, 0, error)
                self.assertEqual(payload, {"decision": "promote", "run_id": command})
                runner.assert_called_once_with(output.resolve())

    def test_demo_refuses_to_overwrite_nonempty_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            (output / "evidence.json").write_text("{}", encoding="utf-8")
            loader = Mock()
            with patch("evolsa.cli._load_demo_api", loader):
                status, payload, error = _invoke("demo", "--output", str(output))
            self.assertEqual(status, 2)
            self.assertIsNone(payload)
            self.assertIn("refusing to overwrite", error)
            loader.assert_not_called()

    def test_inspect_and_replay_delegate_to_integrity_api(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            inspect = Mock(return_value={"run_id": "demo", "decision": "promote"})
            replay = Mock(return_value={"run_id": "demo", "replay_verified": True})
            with patch("evolsa.cli._load_demo_api", return_value=(Mock(), inspect, replay)):
                inspect_status, inspect_payload, _ = _invoke("inspect", str(run_dir))
                replay_status, replay_payload, _ = _invoke("replay", str(run_dir))
            self.assertEqual(inspect_status, 0)
            self.assertEqual(inspect_payload["decision"], "promote")
            self.assertEqual(replay_status, 0)
            self.assertIs(replay_payload["replay_verified"], True)
            inspect.assert_called_once_with(run_dir.resolve())
            replay.assert_called_once_with(run_dir.resolve())

    def test_init_creates_a_bounded_project_and_will_not_replace_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "project"
            status, payload, error = _invoke("init", "--output", str(output))
            self.assertEqual(status, 0, error)
            self.assertTrue(payload["initialized"])
            config = json.loads((output / "evolsa.json").read_text(encoding="utf-8"))
            self.assertEqual(config["schema_version"], "evolsa.project/v1")
            self.assertIn("promotion_gate", config["protected_scopes"])

            status, payload, error = _invoke("init", "--output", str(output))
            self.assertEqual(status, 2)
            self.assertIsNone(payload)
            self.assertIn("non-empty", error)

    def test_registry_list_reports_an_empty_registry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "registry"
            status, payload, error = _invoke("registry", "list", "--root", str(root))
            self.assertEqual(status, 0, error)
            self.assertEqual(payload["count"], 0)
            self.assertEqual(payload["operators"], [])

    def test_mine_reconstructs_cards_and_discovers_a_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "cards.json").write_text(
                pretty_json([_card("candidate-1", 1), _card("candidate-2", 2)]) + "\n",
                encoding="utf-8",
            )
            status, payload, error = _invoke("mine", str(run_dir))
            self.assertEqual(status, 0, error)
            self.assertEqual(payload["card_count"], 2)
            self.assertEqual(payload["pattern_count"], 1)
            self.assertEqual(payload["patterns"][0]["change_tags"], ["hyphen_to_space"])

    def test_lab_validates_content_address_and_unsealed_promote_is_advisory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            report = _report()
            (run_dir / "lab-report.json").write_text(pretty_json(report) + "\n", encoding="utf-8")
            (run_dir / "operator-passport.json").write_text(
                json.dumps({"name": "hyphen-to-space", "version": "1.0.0"}),
                encoding="utf-8",
            )

            status, payload, error = _invoke("lab", str(run_dir))
            self.assertEqual(status, 0, error)
            self.assertEqual(payload["report_id"], report.report_id)
            self.assertEqual(payload["paired_tasks"], 4)

            status, payload, error = _invoke("promote", str(run_dir))
            self.assertEqual(status, 0, error)
            self.assertEqual(payload["advisory_decision"], "quarantine")
            self.assertNotIn("decision", payload)
            self.assertEqual(payload["evidence_mode"], "advisory-unsealed")
            self.assertEqual(payload["report_id"], report.report_id)

    def test_corrupt_lab_evidence_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            payload = to_primitive(_report())
            payload["report_id"] = "0" * 64
            (run_dir / "lab-report.json").write_text(json.dumps(payload), encoding="utf-8")
            status, output, error = _invoke("lab", str(run_dir))
            self.assertNotEqual(status, 0)
            self.assertIsNone(output)
            self.assertIn("report_id does not match", error)


if __name__ == "__main__":
    unittest.main()
