from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evolsa.demo.incident_routing import DeterministicModel, IncidentRoutingAdapter
from evolsa.demo.run import (
    _DEMO_LAB_BUDGET,
    _DEMO_TRAJECTORY_BUDGET,
    _evaluate_operator_from_parent,
    _evaluate_paired_operators,
    _evaluated_seed_parent,
    _generation_zero_improve,
    _sealed_trajectory_evidence,
    DemoError,
    RunIntegrityError,
    replay_run,
    run_offline_rsi_demo,
)
from evolsa.kernel import EventLog, TrustKernel
from evolsa.operators import ImproveOperator
from evolsa.registry import LocalRegistry
from evolsa.schemas import operator_passport_document, operator_passport_from_document


class OfflineRSIDemoTests(unittest.TestCase):
    def test_complete_loop_discovers_promotes_installs_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            summary = run_offline_rsi_demo(output)

            self.assertEqual(summary["decision"], "promote")
            self.assertTrue(summary["replay_verified"])
            self.assertGreater(summary["candidate_score"], summary["baseline_score"])
            self.assertEqual(summary["event_count"], 13)
            self.assertTrue(summary["integrity_checked"])
            self.assertTrue(summary["semantics_replayed"])
            self.assertFalse(summary["authenticated"])
            self.assertGreaterEqual(summary["artifact_count"], 2)
            for name in (
                "manifest.json",
                "cards.json",
                "patterns.json",
                "operator-passport.json",
                "lab-report.json",
                "gate-result.json",
                "validation-results.json",
                "sealed-results.json",
                "split-fingerprints.json",
                "review.md",
                "index.html",
            ):
                self.assertTrue((output / name).is_file(), name)

            cards = json.loads((output / "cards.json").read_text(encoding="utf-8"))["cards"]
            self.assertEqual(len(cards), 20)
            self.assertEqual(
                {card["operator_id"].split("@", 1)[0] for card in cards if "@" in card["operator_id"]},
                {"draft", "improve", "debug", "crossover"},
            )
            patterns = json.loads((output / "patterns.json").read_text(encoding="utf-8"))["patterns"]
            learned = [
                pattern
                for pattern in patterns
                if pattern["kind"] == "high_yield"
                and pattern["change_tags"] == ["hyphen_to_space"]
            ]
            self.assertEqual(len(learned), 1)
            self.assertEqual(learned[0]["support"], 4)

            promotion = json.loads(
                (output / "promotion-trials.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(promotion["receipts"]), 4)
            for receipt in promotion["receipts"]:
                baseline = receipt["baseline"]
                candidate = receipt["candidate"]
                self.assertEqual(baseline["operator_id"], "improve@1.0.0")
                self.assertEqual(baseline["artifact_created_by"], "improve@1.0.0")
                self.assertEqual(
                    baseline["parent_node_ids"],
                    candidate["parent_node_ids"],
                )
                self.assertEqual(
                    baseline["parent_node_ids"],
                    [receipt["parent"]["node_id"]],
                )
                self.assertEqual(receipt["budget"], promotion["budget"])
                self.assertEqual(
                    receipt["model_generate_calls"],
                    {"baseline": 1, "candidate": 1},
                )

            sealed = json.loads(
                (output / "sealed-results.json").read_text(encoding="utf-8")
            )
            rsi = sealed["rsi_evidence"]
            self.assertTrue(rsi["matched_task_seed_budget_trajectories"])
            self.assertTrue(rsi["matched_model_generate_calls"])
            self.assertTrue(rsi["persistent_operator_installation"])
            self.assertGreater(rsi["improvement_acceleration"], 0.0)
            self.assertEqual(
                rsi["metric"],
                "cumulative_best_utility_auc_per_matched_operation",
            )
            for outcome in sealed["outcomes"]:
                generation_0 = outcome["generation_0"]
                generation_1 = outcome["generation_1"]
                self.assertTrue(outcome["baseline_valid"])
                self.assertTrue(outcome["candidate_valid"])
                self.assertEqual(generation_0["run_budget"], generation_1["run_budget"])
                self.assertEqual(generation_0["usage"], generation_1["usage"])
                self.assertEqual(
                    generation_0["model_usage"],
                    {"generate_calls": 4},
                )
                self.assertEqual(generation_0["model_usage"], generation_1["model_usage"])
                self.assertEqual(len(generation_0["trajectory"]), 5)
                self.assertEqual(len(generation_1["trajectory"]), 5)
                self.assertEqual(
                    generation_0["trajectory"][0]["node_id"],
                    generation_1["trajectory"][0]["node_id"],
                )
                self.assertEqual(
                    generation_0["operation_plan"][1:],
                    generation_1["operation_plan"][1:],
                )
                self.assertEqual(
                    generation_0["operation_plan"][0]["operator_id"],
                    "improve@1.0.0",
                )
                self.assertEqual(
                    generation_1["operation_plan"][0]["operator_id"],
                    summary["operator_id"],
                )
                for generation in (generation_0, generation_1):
                    expected_auc = sum(
                        step["best_utility"] for step in generation["trajectory"][1:]
                    ) / 4
                    self.assertAlmostEqual(
                        generation["best_utility_auc_per_attempt"],
                        expected_auc,
                    )
                    self.assertEqual(
                        [step["cumulative_usage"]["attempts"] for step in generation["trajectory"]],
                        [1, 2, 3, 4, 5],
                    )
                self.assertGreater(
                    generation_1["best_utility_auc_per_attempt"],
                    generation_0["best_utility_auc_per_attempt"],
                )

            kernel = TrustKernel(
                {"offline-demo-kernel": b"evolsa-offline-demo-key-v1"},
                release_signers={"offline-demo-kernel"},
                release_principals={"offline-demo-release"},
            )
            installed = LocalRegistry(output / "registry").installed(
                "normalize-hyphenated-incidents",
                trust_kernel=kernel,
                actor="offline-demo-release",
            )
            self.assertEqual(installed.operator_id, summary["operator_id"])
            event_types = [record.event_type for record in EventLog(output / "events.jsonl").verify()]
            self.assertEqual(event_types[-1], "run_sealed")
            self.assertIn("matched_budget_lab_completed", event_types)
            self.assertIn("operator_installed", event_types)
            self.assertEqual(replay_run(output)["report_id"], summary["report_id"])

            loaded = LocalRegistry(output / "registry").load_installed_operator(
                "normalize-hyphenated-incidents",
                trust_kernel=kernel,
                actor="offline-demo-release",
            )
            paired_model = DeterministicModel()
            pair = _evaluate_paired_operators(
                adapter=IncidentRoutingAdapter(),
                model=paired_model,
                operator=loaded,
                split="promotion",
                seed=0,
                budget=_DEMO_LAB_BUDGET,
            )
            self.assertEqual(paired_model.generate_calls, 2)
            self.assertEqual(pair.baseline_model_generate_calls, 1)
            self.assertEqual(pair.candidate_model_generate_calls, 1)

            trajectory_model = DeterministicModel()
            _sealed_trajectory_evidence(
                IncidentRoutingAdapter(),
                trajectory_model,
                loaded,
                (0,),
                _DEMO_TRAJECTORY_BUDGET,
                persistent_installation=True,
            )
            self.assertEqual(trajectory_model.generate_calls, 8)

    def test_promotion_baseline_is_the_canonical_improve_runtime(self) -> None:
        adapter = IncidentRoutingAdapter()
        model = DeterministicModel()
        state = adapter.prepare(0, "promotion")
        parent = _evaluated_seed_parent(
            adapter=adapter,
            model=model,
            state=state,
            budget=_DEMO_LAB_BUDGET,
        )
        baseline = _generation_zero_improve()

        self.assertIs(type(baseline), ImproveOperator)
        before = model.generate_calls
        result = _evaluate_operator_from_parent(
            adapter=adapter,
            model=model,
            operator=baseline,
            state=state,
            parent=parent,
            budget=_DEMO_LAB_BUDGET,
        )
        self.assertEqual(model.generate_calls - before, 1)
        self.assertTrue(result.score.valid)
        self.assertEqual(result.score.primary, 0.8)
        self.assertEqual(result.artifact.created_by, ImproveOperator().passport.operator_id)

    def test_demo_is_deterministic_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            run_offline_rsi_demo(first)
            run_offline_rsi_demo(second)
            for name in (
                "manifest.json",
                "events.jsonl",
                "events.jsonl.head",
                "cards.json",
                "patterns.json",
                "lab-report.json",
                "gate-result.json",
                "sealed-results.json",
                "operator-passport.json",
            ):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)
            with self.assertRaisesRegex(DemoError, "not empty"):
                run_offline_rsi_demo(first)

    def test_replay_detects_evidence_event_and_artifact_tampering(self) -> None:
        cases = ("lab", "event", "artifact")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "run"
                run_offline_rsi_demo(output)
                if case == "lab":
                    path = output / "lab-report.json"
                    path.write_text(path.read_text(encoding="utf-8").replace('"mean_lift": 0.199', '"mean_lift": 0.299', 1), encoding="utf-8")
                elif case == "event":
                    path = output / "events.jsonl"
                    path.write_text(path.read_text(encoding="utf-8").replace("run_started", "run_changed", 1), encoding="utf-8")
                else:
                    blob = next(path for path in (output / "artifacts" / "sha256").rglob("*") if path.is_file())
                    blob.write_bytes(blob.read_bytes() + b"tamper")
                with self.assertRaises(RunIntegrityError):
                    replay_run(output)


class PassportWireContractTests(unittest.TestCase):
    def test_public_schema_document_round_trips_installed_passport(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            run_offline_rsi_demo(output)
            evidence = json.loads((output / "operator-passport.json").read_text(encoding="utf-8"))
            document = evidence["passport"]
            self.assertIn("generator", document)
            self.assertNotIn("generator_type", document)
            passport = operator_passport_from_document(document)
            self.assertEqual(operator_passport_document(passport), document)
            self.assertEqual(passport.content_hash, evidence["content_hash"])


if __name__ == "__main__":
    unittest.main()
