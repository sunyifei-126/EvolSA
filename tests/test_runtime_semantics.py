from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Callable

from evolsa.core.hashing import sha256_json
from evolsa.core.serde import canonical_json, pretty_json
from evolsa.demo import RunIntegrityError, replay_run, run_offline_rsi_demo


def _locally_reseal(
    root: Path,
    mutate: Callable[[list[dict[str, object]], dict[str, object]], None],
) -> None:
    """Simulate a hostile local user who can rewrite every local hash."""

    event_path = root / "events.jsonl"
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(events, manifest)
    previous = "0" * 64
    for sequence, event in enumerate(events):
        event["sequence"] = sequence
        event["previous_hash"] = previous
        event["event_hash"] = sha256_json(
            {key: value for key, value in event.items() if key != "event_hash"}
        )
        previous = str(event["event_hash"])
    event_path.write_text(
        "\n".join(canonical_json(event) for event in events) + "\n",
        encoding="utf-8",
    )
    terminal = events[-1]
    (root / "events.jsonl.head").write_text(
        pretty_json(
            {
                "sequence": terminal["sequence"],
                "event_hash": terminal["event_hash"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest["event_head"] = terminal["event_hash"]
    manifest_path.write_text(pretty_json(manifest) + "\n", encoding="utf-8")


def _reseal_indexed_files(root: Path, *relative_paths: str) -> None:
    digests = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in relative_paths
    }

    def mutate(events: list[dict[str, object]], manifest: dict[str, object]) -> None:
        terminal_payload = events[-1]["payload"]
        assert isinstance(terminal_payload, dict)
        terminal_files = terminal_payload["evidence_files"]
        manifest_files = manifest["evidence_files"]
        assert isinstance(terminal_files, dict) and isinstance(manifest_files, dict)
        terminal_files.update(digests)
        manifest_files.update(digests)

    _locally_reseal(root, mutate)


class ReplaySemanticSecurityTests(unittest.TestCase):
    def test_locally_resealed_fake_tasks_fail_semantic_replay(self) -> None:
        """A self-consistent local hash rewrite is not accepted as real evidence."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            run_offline_rsi_demo(root)

            sealed_path = root / "sealed-results.json"
            sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
            forged = dict(sealed["outcomes"][0])
            forged.update(
                {
                    "task_id": "forged-task",
                    "seed": 0,
                    "baseline_score": -999.0,
                    "candidate_score": 999.0,
                    "lift": 1998.0,
                    "candidate_valid": False,
                }
            )
            sealed["outcomes"] = [dict(forged) for _ in range(4)]
            sealed["baseline_mean"] = -999.0
            sealed["candidate_mean"] = 999.0
            sealed["mean_lift"] = 1998.0
            sealed_path.write_text(pretty_json(sealed) + "\n", encoding="utf-8")
            sealed_digest = hashlib.sha256(sealed_path.read_bytes()).hexdigest()

            event_path = root / "events.jsonl"
            events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
            terminal = events[-1]
            terminal["payload"]["evidence_files"]["sealed-results.json"] = sealed_digest
            terminal["event_hash"] = sha256_json(
                {key: value for key, value in terminal.items() if key != "event_hash"}
            )
            events[-1] = terminal
            event_path.write_text(
                "\n".join(canonical_json(event) for event in events) + "\n",
                encoding="utf-8",
            )
            (root / "events.jsonl.head").write_text(
                pretty_json(
                    {
                        "sequence": terminal["sequence"],
                        "event_hash": terminal["event_hash"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["evidence_files"]["sealed-results.json"] = sealed_digest
            manifest["event_head"] = terminal["event_hash"]
            manifest_path.write_text(pretty_json(manifest) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                RunIntegrityError,
                "sealed outcomes do not reproduce",
            ):
                replay_run(root)

    def test_locally_resealed_trajectory_cannot_forge_acceleration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            run_offline_rsi_demo(root)
            sealed_path = root / "sealed-results.json"
            sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
            generation_1 = sealed["outcomes"][0]["generation_1"]
            generation_1["trajectory"][1]["best_utility"] = 999.0
            generation_1["best_utility_auc_per_attempt"] = 999.0
            sealed["rsi_evidence"]["improvement_acceleration"] = 999.0
            sealed_path.write_text(pretty_json(sealed) + "\n", encoding="utf-8")
            _reseal_indexed_files(root, "sealed-results.json")

            with self.assertRaisesRegex(
                RunIntegrityError,
                "sealed outcomes do not reproduce",
            ):
                replay_run(root)

    def test_locally_resealed_run_contract_cannot_weaken_protected_protocol(self) -> None:
        variants = (
            ("protected scopes", lambda run: run.__setitem__("protected_scopes", ["evaluator"])),
            ("budget", lambda run: run["budget"].__setitem__("max_tokens", 999_999)),
            ("operator set", lambda run: run.__setitem__("operator_ids", ["seed"])),
        )
        for label, mutate_run in variants:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "run"
                run_offline_rsi_demo(root)
                run_path = root / "run.json"
                run = json.loads(run_path.read_text(encoding="utf-8"))
                mutate_run(run)
                run_path.write_text(pretty_json(run) + "\n", encoding="utf-8")
                _reseal_indexed_files(root, "run.json")
                with self.assertRaisesRegex(
                    RunIntegrityError,
                    "v1 offline RSI protocol|operator set",
                ):
                    replay_run(root)

    def test_registry_state_cannot_be_consumed_outside_the_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            run_offline_rsi_demo(root)

            def mutate(events: list[dict[str, object]], manifest: dict[str, object]) -> None:
                terminal_payload = events[-1]["payload"]
                assert isinstance(terminal_payload, dict)
                terminal_files = terminal_payload["evidence_files"]
                manifest_files = manifest["evidence_files"]
                assert isinstance(terminal_files, dict) and isinstance(manifest_files, dict)
                for name in tuple(manifest_files):
                    if str(name).startswith("registry/"):
                        manifest_files.pop(name)
                        terminal_files.pop(name)

            _locally_reseal(root, mutate)
            with self.assertRaisesRegex(RunIntegrityError, "sealed evidence file set"):
                replay_run(root)

    def test_artifact_occurrence_provenance_is_semantically_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            run_offline_rsi_demo(root)
            index_path = root / "artifact-index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            first = next(iter(index["artifacts"].values()))["occurrences"][0]
            first.update(
                {
                    "created_by": "forged-operator@9.9.9",
                    "parent_ids": ["f" * 64],
                    "metadata": {"forged": True},
                }
            )
            index_path.write_text(pretty_json(index) + "\n", encoding="utf-8")
            _reseal_indexed_files(root, "artifact-index.json")
            with self.assertRaisesRegex(RunIntegrityError, "artifact occurrence provenance"):
                replay_run(root)

    def test_user_visible_reports_are_rebuilt_from_verified_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            run_offline_rsi_demo(root)
            (root / "review.md").write_text(
                "FORGED REPORT: candidate_score=999; authenticated=true\n",
                encoding="utf-8",
            )
            (root / "index.html").write_text(
                "<h1>FORGED REPORT: authenticated=true</h1>",
                encoding="utf-8",
            )
            _reseal_indexed_files(root, "review.md", "index.html")
            with self.assertRaisesRegex(RunIntegrityError, "rendered reports"):
                replay_run(root)

    def test_every_event_payload_is_semantically_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            run_offline_rsi_demo(root)

            def mutate(events: list[dict[str, object]], manifest: dict[str, object]) -> None:
                del manifest
                payload = events[1]["payload"]
                assert isinstance(payload, dict)
                payload.update(
                    {
                        "best_score": 999.0,
                        "best_node_id": "f" * 64,
                        "node_ids": ["f" * 64],
                    }
                )

            _locally_reseal(root, mutate)
            with self.assertRaisesRegex(RunIntegrityError, "event payload or timestamp"):
                replay_run(root)


if __name__ == "__main__":
    unittest.main()
