from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

from evolsa.core import (
    Budget,
    BudgetUsage,
    EffectScope,
    ExecutionStatus,
    OperatorLifecycle,
    OperatorPassport,
    OperatorVerification,
)
from evolsa.kernel import (
    BudgetExceededError,
    BudgetLedger,
    ContentAddressedStore,
    ContentHashMismatchError,
    EffectAuthorizationError,
    EventLog,
    EventLogIntegrityError,
    PassportIntegrityError,
    ReleaseAuthorizationError,
    SandboxLimits,
    SandboxViolation,
    SubprocessSandbox,
    TrustKernel,
    UnknownReservationError,
    UnsafeContentPathError,
    sign_passport,
)


class BudgetLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = BudgetLedger(
            Budget(max_tokens=100, max_exec_seconds=10.0, max_attempts=3, max_cost=2.0)
        )

    def test_reserve_settle_and_cancel(self) -> None:
        first = self.ledger.reserve(
            BudgetUsage(tokens=40, exec_seconds=2.0, attempts=1, cost=0.5)
        )
        second = self.ledger.reserve(
            BudgetUsage(tokens=20, exec_seconds=1.0, attempts=1, cost=0.25)
        )
        self.assertEqual(self.ledger.reserved.tokens, 60)

        snapshot = self.ledger.settle(
            first, BudgetUsage(tokens=35, exec_seconds=2.5, attempts=1, cost=0.4)
        )
        self.assertEqual(snapshot.used.tokens, 35)
        self.assertEqual(snapshot.reserved.tokens, 20)
        self.assertEqual(snapshot.remaining.tokens, 45)
        self.ledger.cancel(second)
        self.assertEqual(self.ledger.reserved, BudgetUsage())

    def test_failed_operations_are_atomic(self) -> None:
        reservation = self.ledger.reserve(
            BudgetUsage(tokens=90, exec_seconds=1.0, attempts=1, cost=0.1)
        )
        before = self.ledger.snapshot
        with self.assertRaises(BudgetExceededError):
            self.ledger.reserve(BudgetUsage(tokens=11))
        self.assertEqual(self.ledger.snapshot, before)

        with self.assertRaises(BudgetExceededError):
            self.ledger.settle(reservation, BudgetUsage(tokens=101, attempts=1))
        self.assertEqual(self.ledger.snapshot, before)
        self.assertTrue(self.ledger.has_reservation(reservation))

    def test_reservations_are_single_use(self) -> None:
        reservation = self.ledger.reserve(BudgetUsage(tokens=1, attempts=1))
        self.ledger.settle(reservation)
        with self.assertRaises(UnknownReservationError):
            self.ledger.settle(reservation)
        with self.assertRaises(UnknownReservationError):
            self.ledger.cancel(reservation)

    def test_concurrent_reservations_cannot_overbook(self) -> None:
        ledger = BudgetLedger(Budget(max_tokens=4, max_exec_seconds=4, max_attempts=4, max_cost=4))
        barrier = threading.Barrier(8)
        accepted = []
        denied = []
        lock = threading.Lock()

        def reserve() -> None:
            barrier.wait()
            try:
                reservation = ledger.reserve(
                    BudgetUsage(tokens=1, exec_seconds=1, attempts=1, cost=1)
                )
                with lock:
                    accepted.append(reservation)
            except BudgetExceededError:
                with lock:
                    denied.append(True)

        threads = [threading.Thread(target=reserve) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(accepted), 4)
        self.assertEqual(len(denied), 4)
        self.assertEqual(ledger.snapshot.committed.tokens, 4)

    def test_zero_token_and_cost_limits_are_unmetered(self) -> None:
        ledger = BudgetLedger(Budget(max_tokens=0, max_exec_seconds=5, max_attempts=2, max_cost=0))
        ledger.charge(BudgetUsage(tokens=10_000, exec_seconds=1, attempts=1, cost=42.0))
        self.assertEqual(ledger.used.tokens, 10_000)
        self.assertEqual(ledger.used.cost, 42.0)
        self.assertEqual(ledger.remaining.tokens, 0)
        self.assertEqual(ledger.remaining.cost, 0.0)


class ContentAddressedStoreTests(unittest.TestCase):
    def test_round_trip_deduplication_and_verify_all(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ContentAddressedStore(directory)
            digest = store.put_text("evolsa")
            self.assertEqual(store.put_text("evolsa"), digest)
            self.assertEqual(store.get_text(digest), "evolsa")
            self.assertTrue(store.exists(digest))
            self.assertEqual(store.verify_all(), (digest,))

    def test_rejects_traversal_and_claimed_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ContentAddressedStore(directory)
            with self.assertRaises(UnsafeContentPathError):
                store.path_for("../outside")
            with self.assertRaises(UnsafeContentPathError):
                store.path_for("A" * 64)
            with self.assertRaises(ContentHashMismatchError):
                store.put_bytes(b"actual", expected_digest="0" * 64)

    def test_detects_tampered_blob(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ContentAddressedStore(directory)
            digest = store.put_bytes(b"trusted")
            store.path_for(digest).write_bytes(b"tampered")
            with self.assertRaises(ContentHashMismatchError):
                store.get_bytes(digest)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks unavailable")
    def test_rejects_symlink_redirect(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            store = ContentAddressedStore(directory)
            digest = "ab" + "0" * 62
            prefix = Path(directory) / "sha256" / "ab"
            os.symlink(outside, prefix)
            with self.assertRaises(UnsafeContentPathError):
                store.path_for(digest)


class EventLogTests(unittest.TestCase):
    def test_append_chain_and_reopen_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            log = EventLog(path)
            first = log.append("run.started", {"run_id": "r1"})
            second = log.append("artifact.created", {"digest": "abc"})
            self.assertEqual(first.sequence, 0)
            self.assertEqual(second.previous_hash, first.event_hash)
            with self.assertRaises(TypeError):
                first.payload["run_id"] = "mutated"  # type: ignore[index]
            reopened = EventLog(path)
            self.assertEqual(tuple(reopened), (first, second))
            self.assertEqual(reopened.last_hash, second.event_hash)

    def test_detects_record_tampering_on_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            log = EventLog(path)
            log.append("score", {"value": 1})
            record = json.loads(path.read_text(encoding="utf-8"))
            record["payload"]["value"] = 999
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(EventLogIntegrityError):
                EventLog(path)

    def test_durable_head_detects_suffix_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            log = EventLog(path)
            log.append("one", {})
            log.append("two", {})
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            path.write_text(lines[0], encoding="utf-8")
            with self.assertRaises(EventLogIntegrityError):
                EventLog(path)

    def test_concurrent_append_remains_linear(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            logs = [EventLog(path) for _ in range(4)]
            threads = [
                threading.Thread(target=log.append, args=("worker", {"index": index}))
                for index, log in enumerate(logs)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            records = EventLog(path).verify()
            self.assertEqual([record.sequence for record in records], list(range(4)))


class SubprocessSandboxTests(unittest.TestCase):
    def test_success_has_minimal_environment_and_honest_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            os.environ["EVOLSA_PARENT_SECRET"] = "must-not-leak"
            try:
                sandbox = SubprocessSandbox(directory, allowed_environment={"SAFE_VALUE"})
                observation = sandbox.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import os; "
                            "print(os.getenv('SAFE_VALUE')); "
                            "print(os.getenv('EVOLSA_PARENT_SECRET'))"
                        ),
                    ],
                    env={"SAFE_VALUE": "visible"},
                )
            finally:
                del os.environ["EVOLSA_PARENT_SECRET"]
            self.assertEqual(observation.status, ExecutionStatus.SUCCESS)
            self.assertEqual(observation.output.splitlines(), ["visible", "None"])
            self.assertEqual(observation.metadata["isolation"], "subprocess")
            self.assertFalse(observation.metadata["container_isolation"])
            self.assertFalse(observation.metadata["network_isolation"])

    def test_shell_and_environment_are_fail_closed(self) -> None:
        with SubprocessSandbox() as sandbox:
            with self.assertRaises(SandboxViolation):
                sandbox.run("echo unsafe")  # type: ignore[arg-type]
            with self.assertRaises(SandboxViolation):
                sandbox.run(["echo", "unsafe"], shell=True)
            with self.assertRaises(SandboxViolation):
                sandbox.run(["/bin/sh", "-c", "echo unsafe"])
            with self.assertRaises(SandboxViolation):
                sandbox.run([sys.executable, "-c", "pass"], env={"SECRET": "x"})

    def test_timeout_returns_structured_observation(self) -> None:
        with SubprocessSandbox(
            limits=SandboxLimits(default_timeout_seconds=0.05, max_memory_bytes=None)
        ) as sandbox:
            observation = sandbox.run([sys.executable, "-c", "import time; time.sleep(2)"])
        self.assertEqual(observation.status, ExecutionStatus.TIMEOUT)
        self.assertEqual(observation.error_signature, "timeout")
        self.assertLess(observation.runtime_seconds, 1.0)

    def test_nonzero_exit_and_missing_executable_are_structured(self) -> None:
        with SubprocessSandbox() as sandbox:
            failed = sandbox.run([sys.executable, "-c", "raise SystemExit(7)"])
            missing = sandbox.run(["/definitely/missing/evolsa-command"])
        self.assertEqual(failed.status, ExecutionStatus.ERROR)
        self.assertEqual(failed.error_signature, "exit:7")
        self.assertEqual(missing.status, ExecutionStatus.INVALID)
        self.assertTrue(missing.error_signature.startswith("spawn:"))

    def test_cwd_cannot_escape_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            sandbox = SubprocessSandbox(directory)
            with self.assertRaises(SandboxViolation):
                sandbox.run([sys.executable, "-c", "pass"], cwd=outside)

    def test_budget_tightens_timeout(self) -> None:
        with SubprocessSandbox(limits=SandboxLimits(max_memory_bytes=None)) as sandbox:
            before = time.monotonic()
            observation = sandbox.run(
                [sys.executable, "-c", "import time; time.sleep(2)"],
                budget=Budget(max_exec_seconds=0.05),
            )
            elapsed = time.monotonic() - before
        self.assertEqual(observation.status, ExecutionStatus.TIMEOUT)
        self.assertLess(elapsed, 1.0)


class TrustKernelTests(unittest.TestCase):
    release_secret = b"release-secret-at-least-16-bytes"
    dev_secret = b"developer-secret-at-least-16-bytes"

    def passport(
        self,
        *,
        lifecycle: OperatorLifecycle = OperatorLifecycle.STABLE,
        evidence_ids: tuple[str, ...] = ("0" * 64,),
    ) -> OperatorPassport:
        return OperatorPassport(
            name="improve-artifact",
            version="1.0.0",
            intent="improve the task artifact",
            inputs=("task", "artifact"),
            preconditions=("artifact_exists",),
            expected_effects=(EffectScope.TASK_ARTIFACT,),
            forbidden_effects=(
                EffectScope.SANDBOX,
                EffectScope.BUDGET,
                EffectScope.EVENT_LOG,
                EffectScope.PROMOTION_GATE,
                EffectScope.RELEASE_PERMISSION,
            ),
            context_policy="minimal-v1",
            generator_type="prompt_template",
            generator_ref="operators/improve.txt",
            budget=Budget(max_tokens=100, max_exec_seconds=10, max_attempts=1, max_cost=1),
            verification=OperatorVerification(baseline="improve@0.1.0"),
            lifecycle=lifecycle,
            evidence_ids=evidence_ids,
        )

    def kernel(self) -> TrustKernel:
        return TrustKernel(
            {"release": self.release_secret, "dev": self.dev_secret},
            release_signers={"release"},
            release_principals={"human-reviewer"},
        )

    def test_valid_signature_hash_effect_and_release(self) -> None:
        passport = sign_passport(self.passport(), "release", self.release_secret)
        decision = self.kernel().authorize_execution(
            passport,
            [EffectScope.TASK_ARTIFACT],
            expected_hash=passport.content_hash,
        )
        self.assertEqual(decision.signer_id, "release")
        self.assertEqual(decision.authorized_effects, (EffectScope.TASK_ARTIFACT,))
        release = self.kernel().authorize_release(
            passport, "human-reviewer", expected_hash=passport.content_hash
        )
        self.assertTrue(release.release_authorized)

    def test_missing_forged_and_stale_signatures_fail_closed(self) -> None:
        kernel = self.kernel()
        with self.assertRaises(PassportIntegrityError):
            kernel.verify_passport(self.passport())
        signed = sign_passport(self.passport(), "release", self.release_secret)
        tampered = replace(signed, intent="different intent")
        with self.assertRaises(PassportIntegrityError):
            kernel.verify_passport(tampered)
        with self.assertRaises(PassportIntegrityError):
            kernel.verify_passport(signed, expected_hash="0" * 64)

    def test_undeclared_forbidden_and_protected_effects_are_denied(self) -> None:
        signed = sign_passport(self.passport(), "release", self.release_secret)
        kernel = self.kernel()
        with self.assertRaises(EffectAuthorizationError):
            kernel.authorize_execution(signed, [EffectScope.OPERATOR])
        with self.assertRaises(EffectAuthorizationError):
            kernel.authorize_execution(signed, [EffectScope.SANDBOX])

        # TrustKernel repeats model-layer validation; it cannot be bypassed by
        # mutating a frozen object through low-level reflection.
        forged = replace(signed)
        object.__setattr__(forged, "expected_effects", (EffectScope.BUDGET,))
        with self.assertRaises(EffectAuthorizationError):
            kernel.verify_passport(forged)

    def test_release_requires_stable_evidence_release_key_and_actor(self) -> None:
        kernel = self.kernel()
        candidate = sign_passport(
            self.passport(lifecycle=OperatorLifecycle.CANDIDATE), "release", self.release_secret
        )
        with self.assertRaises(ReleaseAuthorizationError):
            kernel.authorize_release(candidate, "human-reviewer")

        no_evidence = sign_passport(
            self.passport(evidence_ids=()), "release", self.release_secret
        )
        with self.assertRaises(ReleaseAuthorizationError):
            kernel.authorize_release(no_evidence, "human-reviewer")

        dev_signed = sign_passport(self.passport(), "dev", self.dev_secret)
        with self.assertRaises(ReleaseAuthorizationError):
            kernel.authorize_release(dev_signed, "human-reviewer")

        release_signed = sign_passport(self.passport(), "release", self.release_secret)
        with self.assertRaises(ReleaseAuthorizationError):
            kernel.authorize_release(release_signed, "model-agent")


if __name__ == "__main__":
    unittest.main()
