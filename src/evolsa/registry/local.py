"""Content-addressed, atomic local operator registry."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

try:  # pragma: no cover - exercised by import tests on Windows.
    import fcntl as _fcntl
except ImportError:  # Windows has no fcntl; the process lock remains effective.
    _fcntl = None

from evolsa.core import (
    Budget,
    GateResult,
    LabReport,
    MetricDirection,
    OperatorLifecycle,
    OperatorPassport,
    PairedOutcome,
    PromotionDecision,
    PromotionPolicy,
)
from evolsa.core.serde import canonical_json, pretty_json, to_primitive
from evolsa.core.hashing import sha256_json, sha256_text
from evolsa.lab import PromotionGate
from evolsa.schemas import operator_passport_from_document


REGISTRY_SCHEMA = "evolsa.registry/v1"
OBJECT_SCHEMA = "evolsa.operator-object/v1"
INSTALL_SCHEMA = "evolsa.installation/v1"
GENERATOR_SCHEMA = "evolsa.generator-object/v1"
PROMOTION_RECEIPT_SCHEMA = "evolsa.promotion-receipt/v1"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SIGNATURE = re.compile(r"^hmac-sha256:[A-Za-z0-9][A-Za-z0-9._-]{0,127}:[0-9a-f]{64}$")


_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[Path, threading.RLock] = {}


def _process_lock_for(root: Path) -> threading.RLock:
    """Return a per-registry lock shared by every instance in this process."""

    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(root, threading.RLock())


class RegistryError(RuntimeError):
    """Base registry failure."""


class RegistryConflictError(RegistryError):
    """An operator ID was reused for different immutable content."""


class RegistryIntegrityError(RegistryError):
    """Stored content failed schema or content-address verification."""


class LifecycleTransitionError(RegistryError):
    """An invalid or unevidenced lifecycle transition was requested."""


@dataclass(frozen=True, slots=True)
class RegistryRecord:
    operator_id: str
    content_hash: str
    lifecycle: OperatorLifecycle
    evidence_ids: tuple[str, ...]
    gate_report_id: str = ""
    promotion_receipt_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))


@dataclass(frozen=True, slots=True)
class Installation:
    name: str
    operator_id: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class PromotionReceipt:
    """Immutable evidence bundle authorizing one candidate-to-stable transition."""

    receipt_id: str
    operator_id: str
    candidate_content_hash: str
    stable_content_hash: str
    lab_report: LabReport
    promotion_policy: PromotionPolicy
    gate_result: GateResult


_ALLOWED_TRANSITIONS: Mapping[OperatorLifecycle, frozenset[OperatorLifecycle]] = {
    OperatorLifecycle.PROPOSED: frozenset(
        {OperatorLifecycle.SHADOW, OperatorLifecycle.REJECTED, OperatorLifecycle.QUARANTINED}
    ),
    OperatorLifecycle.SHADOW: frozenset(
        {OperatorLifecycle.CANDIDATE, OperatorLifecycle.REJECTED, OperatorLifecycle.QUARANTINED}
    ),
    OperatorLifecycle.CANDIDATE: frozenset(
        {
            OperatorLifecycle.SHADOW,
            OperatorLifecycle.STABLE,
            OperatorLifecycle.REJECTED,
            OperatorLifecycle.QUARANTINED,
        }
    ),
    OperatorLifecycle.STABLE: frozenset(
        {OperatorLifecycle.DEPRECATED, OperatorLifecycle.QUARANTINED}
    ),
    OperatorLifecycle.REJECTED: frozenset(),
    OperatorLifecycle.QUARANTINED: frozenset(),
    OperatorLifecycle.DEPRECATED: frozenset(),
}


def _validate_component(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"unsafe {label}: {value!r}")
    return value


def _strict_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise RegistryIntegrityError(f"{label} must be a JSON string")
    return value


def _strict_integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise RegistryIntegrityError(f"{label} must be a JSON integer")
    return value


def _strict_number(value: Any, label: str) -> float | int:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise RegistryIntegrityError(f"{label} must be a finite JSON number")
    return value


def _strict_boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise RegistryIntegrityError(f"{label} must be a JSON boolean")
    return value


def _strict_string_array(value: Any, label: str) -> tuple[str, ...]:
    if type(value) is not list or not all(type(item) is str for item in value):
        raise RegistryIntegrityError(f"{label} must be an array of strings")
    return tuple(value)


def _passport_from_primitive(value: Mapping[str, Any]) -> OperatorPassport:
    try:
        document = dict(value)
        generator_type = document.pop("generator_type")
        generator_ref = document.pop("generator_ref")
        document["generator"] = {"type": generator_type, "ref": generator_ref}
        return operator_passport_from_document(document)
    except (KeyError, TypeError, ValueError) as error:
        raise RegistryIntegrityError(f"invalid operator object: {error}") from error


def _budget_from_primitive(value: Any, label: str) -> Budget:
    if not isinstance(value, Mapping) or set(value) != {
        "max_tokens",
        "max_exec_seconds",
        "max_attempts",
        "max_cost",
    }:
        raise RegistryIntegrityError(f"{label} must be a budget object")
    try:
        return Budget(
            max_tokens=_strict_integer(value["max_tokens"], f"{label}.max_tokens", minimum=0),
            max_exec_seconds=_strict_number(
                value["max_exec_seconds"], f"{label}.max_exec_seconds"
            ),
            max_attempts=_strict_integer(
                value["max_attempts"], f"{label}.max_attempts", minimum=1
            ),
            max_cost=_strict_number(value["max_cost"], f"{label}.max_cost"),
        )
    except (KeyError, TypeError, ValueError, RegistryIntegrityError) as error:
        raise RegistryIntegrityError(f"invalid {label}: {error}") from error


def _paired_outcome_from_primitive(value: Any) -> PairedOutcome:
    expected_fields = {
        "task_id",
        "seed",
        "candidate_score",
        "baseline_score",
        "candidate_valid",
        "baseline_valid",
        "candidate_cost",
        "baseline_cost",
        "regression",
        "direction",
        "split",
        "evaluator_id",
        "candidate_parent_id",
        "baseline_parent_id",
        "candidate_model_id",
        "baseline_model_id",
        "candidate_sampling",
        "baseline_sampling",
        "candidate_budget",
        "baseline_budget",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise RegistryIntegrityError("promotion receipt outcomes must be objects")
    try:
        candidate_sampling = value.get("candidate_sampling")
        baseline_sampling = value.get("baseline_sampling")
        if candidate_sampling is not None and not isinstance(candidate_sampling, Mapping):
            raise TypeError("candidate_sampling must be an object or null")
        if baseline_sampling is not None and not isinstance(baseline_sampling, Mapping):
            raise TypeError("baseline_sampling must be an object or null")
        candidate_budget = value.get("candidate_budget")
        baseline_budget = value.get("baseline_budget")
        return PairedOutcome(
            task_id=_strict_string(value["task_id"], "outcome.task_id"),
            seed=_strict_integer(value["seed"], "outcome.seed", minimum=0),
            candidate_score=_strict_number(value["candidate_score"], "outcome.candidate_score"),
            baseline_score=_strict_number(value["baseline_score"], "outcome.baseline_score"),
            candidate_valid=_strict_boolean(value["candidate_valid"], "outcome.candidate_valid"),
            baseline_valid=_strict_boolean(value["baseline_valid"], "outcome.baseline_valid"),
            candidate_cost=_strict_number(value["candidate_cost"], "outcome.candidate_cost"),
            baseline_cost=_strict_number(value["baseline_cost"], "outcome.baseline_cost"),
            regression=_strict_number(value["regression"], "outcome.regression"),
            direction=MetricDirection(value.get("direction", MetricDirection.HIGHER_IS_BETTER.value)),
            split=_strict_string(value["split"], "outcome.split", allow_empty=True),
            evaluator_id=_strict_string(
                value["evaluator_id"], "outcome.evaluator_id", allow_empty=True
            ),
            candidate_parent_id=_strict_string(
                value["candidate_parent_id"], "outcome.candidate_parent_id", allow_empty=True
            ),
            baseline_parent_id=_strict_string(
                value["baseline_parent_id"], "outcome.baseline_parent_id", allow_empty=True
            ),
            candidate_model_id=_strict_string(
                value["candidate_model_id"], "outcome.candidate_model_id", allow_empty=True
            ),
            baseline_model_id=_strict_string(
                value["baseline_model_id"], "outcome.baseline_model_id", allow_empty=True
            ),
            candidate_sampling=dict(candidate_sampling) if candidate_sampling is not None else None,
            baseline_sampling=dict(baseline_sampling) if baseline_sampling is not None else None,
            candidate_budget=(
                _budget_from_primitive(candidate_budget, "candidate_budget")
                if candidate_budget is not None
                else None
            ),
            baseline_budget=(
                _budget_from_primitive(baseline_budget, "baseline_budget")
                if baseline_budget is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError, RegistryIntegrityError) as error:
        raise RegistryIntegrityError(f"invalid paired promotion outcome: {error}") from error


def _lab_report_from_primitive(value: Any) -> LabReport:
    expected_fields = {
        "operator_id",
        "baseline_id",
        "outcomes",
        "mean_lift",
        "lcb95",
        "ucb95",
        "valid_rate",
        "baseline_valid_rate",
        "cost_adjusted_lift",
        "transfer_coverage",
        "worst_regression",
        "matched_budget",
        "matched_conditions",
        "bootstrap_samples",
        "bootstrap_seed",
        "bootstrap_confidence",
        "cost_penalty",
        "report_id",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise RegistryIntegrityError("promotion receipt lab_report must be an object")
    raw_outcomes = value.get("outcomes")
    if not isinstance(raw_outcomes, list):
        raise RegistryIntegrityError("promotion receipt lab_report outcomes must be an array")
    try:
        return LabReport(
            operator_id=_strict_string(value["operator_id"], "report.operator_id"),
            baseline_id=_strict_string(value["baseline_id"], "report.baseline_id"),
            outcomes=tuple(_paired_outcome_from_primitive(item) for item in raw_outcomes),
            mean_lift=_strict_number(value["mean_lift"], "report.mean_lift"),
            lcb95=_strict_number(value["lcb95"], "report.lcb95"),
            ucb95=_strict_number(value["ucb95"], "report.ucb95"),
            valid_rate=_strict_number(value["valid_rate"], "report.valid_rate"),
            baseline_valid_rate=_strict_number(
                value["baseline_valid_rate"], "report.baseline_valid_rate"
            ),
            cost_adjusted_lift=_strict_number(
                value["cost_adjusted_lift"], "report.cost_adjusted_lift"
            ),
            transfer_coverage=_strict_integer(
                value["transfer_coverage"], "report.transfer_coverage", minimum=0
            ),
            worst_regression=_strict_number(
                value["worst_regression"], "report.worst_regression"
            ),
            matched_budget=_strict_boolean(value["matched_budget"], "report.matched_budget"),
            matched_conditions=_strict_boolean(
                value["matched_conditions"], "report.matched_conditions"
            ),
            bootstrap_samples=_strict_integer(
                value["bootstrap_samples"], "report.bootstrap_samples", minimum=1
            ),
            bootstrap_seed=_strict_integer(value["bootstrap_seed"], "report.bootstrap_seed"),
            bootstrap_confidence=_strict_number(
                value["bootstrap_confidence"], "report.bootstrap_confidence"
            ),
            cost_penalty=_strict_number(value["cost_penalty"], "report.cost_penalty"),
            report_id=_strict_string(value["report_id"], "report.report_id"),
        )
    except (KeyError, TypeError, ValueError, RegistryIntegrityError) as error:
        raise RegistryIntegrityError(f"invalid promotion lab report: {error}") from error


def _promotion_policy_from_primitive(value: Any) -> PromotionPolicy:
    expected_fields = {
        "minimum_lcb_lift",
        "valid_rate_tolerance",
        "minimum_cost_adjusted_lift",
        "maximum_regression",
        "minimum_transfer_coverage",
        "require_matched_budget",
        "require_matched_conditions",
        "baseline_id",
        "baseline_content_hash",
        "required_trials",
        "required_split",
        "required_evaluator_id",
        "metric_direction",
        "bootstrap_samples",
        "bootstrap_seed",
        "bootstrap_confidence",
        "cost_penalty",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise RegistryIntegrityError("promotion receipt promotion_policy must be an object")
    try:
        raw_trials = value["required_trials"]
        if not isinstance(raw_trials, list):
            raise RegistryIntegrityError("policy.required_trials must be an array")
        required_trials: list[tuple[str, int]] = []
        for index, trial in enumerate(raw_trials):
            if not isinstance(trial, list) or len(trial) != 2:
                raise RegistryIntegrityError(
                    f"policy.required_trials[{index}] must contain task_id and seed"
                )
            required_trials.append(
                (
                    _strict_string(trial[0], f"policy.required_trials[{index}].task_id"),
                    _strict_integer(
                        trial[1], f"policy.required_trials[{index}].seed", minimum=0
                    ),
                )
            )
        return PromotionPolicy(
            minimum_lcb_lift=_strict_number(
                value["minimum_lcb_lift"], "policy.minimum_lcb_lift"
            ),
            valid_rate_tolerance=_strict_number(
                value["valid_rate_tolerance"], "policy.valid_rate_tolerance"
            ),
            minimum_cost_adjusted_lift=_strict_number(
                value["minimum_cost_adjusted_lift"], "policy.minimum_cost_adjusted_lift"
            ),
            maximum_regression=_strict_number(
                value["maximum_regression"], "policy.maximum_regression"
            ),
            minimum_transfer_coverage=_strict_integer(
                value["minimum_transfer_coverage"],
                "policy.minimum_transfer_coverage",
                minimum=1,
            ),
            require_matched_budget=_strict_boolean(
                value["require_matched_budget"], "policy.require_matched_budget"
            ),
            require_matched_conditions=_strict_boolean(
                value["require_matched_conditions"], "policy.require_matched_conditions"
            ),
            baseline_id=_strict_string(
                value["baseline_id"], "policy.baseline_id", allow_empty=True
            ),
            baseline_content_hash=_strict_string(
                value["baseline_content_hash"],
                "policy.baseline_content_hash",
                allow_empty=True,
            ),
            required_trials=tuple(required_trials),
            required_split=_strict_string(value["required_split"], "policy.required_split"),
            required_evaluator_id=_strict_string(
                value["required_evaluator_id"],
                "policy.required_evaluator_id",
                allow_empty=True,
            ),
            metric_direction=MetricDirection(
                _strict_string(value["metric_direction"], "policy.metric_direction")
            ),
            bootstrap_samples=_strict_integer(
                value["bootstrap_samples"], "policy.bootstrap_samples", minimum=1
            ),
            bootstrap_seed=_strict_integer(value["bootstrap_seed"], "policy.bootstrap_seed"),
            bootstrap_confidence=_strict_number(
                value["bootstrap_confidence"], "policy.bootstrap_confidence"
            ),
            cost_penalty=_strict_number(value["cost_penalty"], "policy.cost_penalty"),
        )
    except (TypeError, ValueError, RegistryIntegrityError) as error:
        raise RegistryIntegrityError(f"invalid promotion policy: {error}") from error


def _gate_result_from_primitive(value: Any) -> GateResult:
    if not isinstance(value, Mapping) or set(value) != {
        "decision",
        "operator_id",
        "report_id",
        "reasons",
        "policy_hash",
    }:
        raise RegistryIntegrityError("promotion receipt gate_result must be an object")
    raw_reasons = value.get("reasons")
    if not isinstance(raw_reasons, list) or any(not isinstance(item, str) for item in raw_reasons):
        raise RegistryIntegrityError("promotion receipt gate reasons must be an array of strings")
    try:
        return GateResult(
            decision=PromotionDecision(value["decision"]),
            operator_id=_strict_string(value["operator_id"], "gate.operator_id"),
            report_id=_strict_string(value["report_id"], "gate.report_id"),
            reasons=tuple(raw_reasons),
            policy_hash=_strict_string(value["policy_hash"], "gate.policy_hash"),
        )
    except (KeyError, TypeError, ValueError, RegistryIntegrityError) as error:
        raise RegistryIntegrityError(f"invalid promotion gate result: {error}") from error


def _atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace a JSON file without exposing a partial write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(pretty_json(value))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            # Some filesystems do not support fsync on directories.  The file
            # replacement itself is still atomic on supported local platforms.
            pass
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink():
        raise RegistryIntegrityError(f"registry files cannot be symbolic links: {path}")
    try:
        def reject_constant(value: str) -> None:
            raise ValueError(f"non-finite JSON constant: {value}")

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = item
            return result

        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                parse_constant=reject_constant,
                object_pairs_hook=reject_duplicates,
            )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RegistryIntegrityError(f"cannot read registry file {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise RegistryIntegrityError(f"registry file must contain an object: {path}")
    return value


class LocalRegistry:
    """A process-safe local registry with immutable semantic objects.

    Mutable indexes and installation pointers are atomically replaced while
    passports are stored under their semantic ``OperatorPassport.content_hash``.
    The model-facing foundry has no API for bypassing lifecycle transitions.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects_dir = self.root / "objects" / "sha256"
        self.generators_dir = self.root / "objects" / "generators" / "sha256"
        self.promotion_receipts_dir = self.root / "evidence" / "promotions" / "sha256"
        self.installations_dir = self.root / "installations"
        self.index_path = self.root / "index.json"
        self.lock_path = self.root / ".registry.lock"
        self._process_lock = _process_lock_for(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in (
            self.root / "objects",
            self.objects_dir,
            self.root / "objects" / "generators",
            self.generators_dir,
            self.root / "evidence",
            self.root / "evidence" / "promotions",
            self.promotion_receipts_dir,
            self.installations_dir,
        ):
            if directory.is_symlink():
                raise RegistryIntegrityError(f"registry directories cannot be symbolic links: {directory}")
            directory.mkdir(exist_ok=True)
            if not directory.resolve().is_relative_to(self.root):
                raise RegistryIntegrityError(f"registry directory escapes its root: {directory}")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._process_lock:
            if self.lock_path.is_symlink():
                raise RegistryIntegrityError("registry lock cannot be a symbolic link")
            with self.lock_path.open("a+b") as lock_stream:
                if _fcntl is not None:
                    _fcntl.flock(lock_stream.fileno(), _fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if _fcntl is not None:
                        _fcntl.flock(lock_stream.fileno(), _fcntl.LOCK_UN)

    @staticmethod
    def _empty_index() -> dict[str, Any]:
        return {"schema_version": REGISTRY_SCHEMA, "operators": {}}

    def _read_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return self._empty_index()
        value = dict(_read_json(self.index_path))
        if value.get("schema_version") != REGISTRY_SCHEMA or not isinstance(value.get("operators"), Mapping):
            raise RegistryIntegrityError("registry index schema is invalid")
        value["operators"] = dict(value["operators"])
        return value

    def object_path(self, content_hash: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
            raise ValueError("content hash must be a lowercase SHA-256 digest")
        return self.objects_dir / f"{content_hash}.json"

    def _store_object(self, passport: OperatorPassport) -> None:
        path = self.object_path(passport.content_hash)
        payload = {
            "schema_version": OBJECT_SCHEMA,
            "kind": "OperatorPassport",
            "content_hash": passport.content_hash,
            "passport": to_primitive(passport),
        }
        if path.exists():
            existing = _read_json(path)
            if canonical_json(existing) != canonical_json(payload):
                raise RegistryIntegrityError("content-address collision or modified operator object")
            return
        _atomic_write_json(path, payload)

    def generator_path(self, digest: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("generator digest must be a lowercase SHA-256 digest")
        return self.generators_dir / f"{digest}.json"

    def _store_generator(self, prompt_template: str, expected_digest: str) -> None:
        if not isinstance(prompt_template, str) or not prompt_template:
            raise ValueError("prompt-template generator must be non-empty text")
        actual_digest = sha256_text(prompt_template)
        if actual_digest != expected_digest:
            raise RegistryIntegrityError("generator content does not match passport generator_ref")
        path = self.generator_path(expected_digest)
        payload = {
            "schema_version": GENERATOR_SCHEMA,
            "generator_type": "prompt_template",
            "content_digest": expected_digest,
            "content": prompt_template,
        }
        if path.exists():
            if canonical_json(_read_json(path)) != canonical_json(payload):
                raise RegistryIntegrityError("stored generator content was modified")
            return
        _atomic_write_json(path, payload)

    @staticmethod
    def _generator_digest(passport: OperatorPassport) -> str:
        prefix = "sha256:"
        if passport.generator_type != "prompt_template" or not passport.generator_ref.startswith(prefix):
            raise RegistryIntegrityError(
                "only content-addressed prompt_template generators are locally loadable"
            )
        digest = passport.generator_ref.removeprefix(prefix)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RegistryIntegrityError("passport generator_ref has an invalid content address")
        return digest

    def register_executable(self, operator: Any) -> RegistryRecord:
        """Persist a callback-free L0 operator and its generator artifact."""

        passport = getattr(operator, "passport", None)
        prompt_template = getattr(operator, "prompt_template", None)
        callback = getattr(operator, "callback", None)
        if not isinstance(passport, OperatorPassport) or not isinstance(prompt_template, str):
            raise TypeError("operator must expose a passport and prompt_template")
        if callback is not None:
            raise ValueError("Python callbacks are not serializable registry generators")
        digest = self._generator_digest(passport)
        self._store_generator(prompt_template, digest)
        return self.register(passport)

    def _load_prompt_template(self, passport: OperatorPassport) -> str:
        digest = self._generator_digest(passport)
        payload = _read_json(self.generator_path(digest))
        content = payload.get("content")
        if (
            set(payload)
            != {"schema_version", "generator_type", "content_digest", "content"}
            or payload.get("schema_version") != GENERATOR_SCHEMA
            or payload.get("generator_type") != "prompt_template"
            or payload.get("content_digest") != digest
            or not isinstance(content, str)
            or sha256_text(content) != digest
        ):
            raise RegistryIntegrityError("generator artifact failed content verification")
        return content

    def load_operator(self, operator_id: str) -> Any:
        """Rehydrate a persisted L0 operator without an in-memory callback."""

        from evolsa.foundry import CallbackOperator

        passport = self.get(operator_id)
        return CallbackOperator(
            passport=passport,
            prompt_template=self._load_prompt_template(passport),
            callback=None,
        )

    def load_installed_operator(
        self,
        name: str,
        *,
        trust_kernel: Any,
        actor: str,
    ) -> Any:
        """Rehydrate the active stable generation by installation name."""

        from evolsa.foundry import CallbackOperator

        passport = self.installed(name, trust_kernel=trust_kernel, actor=actor)
        return CallbackOperator(
            passport=passport,
            prompt_template=self._load_prompt_template(passport),
            callback=None,
        )

    def promotion_receipt_path(self, receipt_id: str) -> Path:
        if not _DIGEST.fullmatch(receipt_id):
            raise ValueError("promotion receipt id must be a lowercase SHA-256 digest")
        return self.promotion_receipts_dir / f"{receipt_id}.json"

    @staticmethod
    def _promotion_receipt_body(
        *,
        operator_id: str,
        candidate_content_hash: str,
        stable_content_hash: str,
        lab_report: LabReport,
        promotion_policy: PromotionPolicy,
        gate_result: GateResult,
    ) -> dict[str, Any]:
        return {
            "operator_id": operator_id,
            "candidate_content_hash": candidate_content_hash,
            "stable_content_hash": stable_content_hash,
            "lab_report": to_primitive(lab_report),
            "promotion_policy": to_primitive(promotion_policy),
            "gate_result": to_primitive(gate_result),
        }

    def _store_promotion_receipt(
        self,
        *,
        operator_id: str,
        candidate_content_hash: str,
        stable_content_hash: str,
        lab_report: LabReport,
        promotion_policy: PromotionPolicy,
        gate_result: GateResult,
    ) -> str:
        body = self._promotion_receipt_body(
            operator_id=operator_id,
            candidate_content_hash=candidate_content_hash,
            stable_content_hash=stable_content_hash,
            lab_report=lab_report,
            promotion_policy=promotion_policy,
            gate_result=gate_result,
        )
        addressed = {"schema_version": PROMOTION_RECEIPT_SCHEMA, **body}
        receipt_id = sha256_json(addressed)
        payload = {**addressed, "receipt_id": receipt_id}
        path = self.promotion_receipt_path(receipt_id)
        if path.exists():
            if canonical_json(_read_json(path)) != canonical_json(payload):
                raise RegistryIntegrityError("promotion receipt content address was modified")
            return receipt_id
        _atomic_write_json(path, payload)
        return receipt_id

    def promotion_receipt(self, receipt_id: str) -> PromotionReceipt:
        """Load and semantically verify a content-addressed promotion receipt."""

        try:
            path = self.promotion_receipt_path(receipt_id)
        except ValueError as error:
            raise RegistryIntegrityError("registry contains an invalid promotion receipt id") from error
        payload = _read_json(path)
        expected_keys = {
            "schema_version",
            "receipt_id",
            "operator_id",
            "candidate_content_hash",
            "stable_content_hash",
            "lab_report",
            "promotion_policy",
            "gate_result",
        }
        if set(payload) != expected_keys or payload.get("schema_version") != PROMOTION_RECEIPT_SCHEMA:
            raise RegistryIntegrityError("promotion receipt schema is invalid")
        addressed = {key: value for key, value in payload.items() if key != "receipt_id"}
        declared_receipt_id = payload.get("receipt_id")
        if declared_receipt_id != receipt_id or sha256_json(addressed) != receipt_id:
            raise RegistryIntegrityError("promotion receipt content address does not match its content")
        operator_id = payload.get("operator_id")
        candidate_content_hash = payload.get("candidate_content_hash")
        stable_content_hash = payload.get("stable_content_hash")
        if (
            not isinstance(operator_id, str)
            or not operator_id
            or not isinstance(candidate_content_hash, str)
            or not _DIGEST.fullmatch(candidate_content_hash)
            or not isinstance(stable_content_hash, str)
            or not _DIGEST.fullmatch(stable_content_hash)
        ):
            raise RegistryIntegrityError("promotion receipt identities are malformed")
        report = _lab_report_from_primitive(payload.get("lab_report"))
        policy = _promotion_policy_from_primitive(payload.get("promotion_policy"))
        gate = _gate_result_from_primitive(payload.get("gate_result"))
        if canonical_json(payload["lab_report"]) != canonical_json(to_primitive(report)):
            raise RegistryIntegrityError("promotion receipt lab_report uses a non-canonical contract")
        if canonical_json(payload["promotion_policy"]) != canonical_json(to_primitive(policy)):
            raise RegistryIntegrityError("promotion receipt policy uses a non-canonical contract")
        if canonical_json(payload["gate_result"]) != canonical_json(to_primitive(gate)):
            raise RegistryIntegrityError("promotion receipt gate result uses a non-canonical contract")
        return PromotionReceipt(
            receipt_id=receipt_id,
            operator_id=operator_id,
            candidate_content_hash=candidate_content_hash,
            stable_content_hash=stable_content_hash,
            lab_report=report,
            promotion_policy=policy,
            gate_result=gate,
        )

    def get_by_hash(self, content_hash: str) -> OperatorPassport:
        path = self.object_path(content_hash)
        if not path.exists():
            raise KeyError(content_hash)
        payload = _read_json(path)
        if (
            set(payload) != {"schema_version", "kind", "content_hash", "passport"}
            or payload.get("schema_version") != OBJECT_SCHEMA
            or payload.get("kind") != "OperatorPassport"
        ):
            raise RegistryIntegrityError("operator object schema is invalid")
        passport_data = payload.get("passport")
        if not isinstance(passport_data, Mapping):
            raise RegistryIntegrityError("operator object is missing its passport")
        passport = _passport_from_primitive(passport_data)
        declared_hash = payload.get("content_hash")
        if passport.content_hash != content_hash or declared_hash != content_hash:
            raise RegistryIntegrityError("operator object content address does not match its content")
        return passport

    @staticmethod
    def _record_from_entry(operator_id: str, entry: Mapping[str, Any]) -> RegistryRecord:
        required = {
            "content_hash",
            "lifecycle",
            "evidence_ids",
            "gate_report_id",
            "promotion_receipt_id",
        }
        if set(entry) not in (required, required | {"history"}):
            raise RegistryIntegrityError(f"invalid registry index fields for {operator_id}")
        try:
            content_hash = _strict_string(entry["content_hash"], "index.content_hash")
            gate_report_id = _strict_string(
                entry["gate_report_id"], "index.gate_report_id", allow_empty=True
            )
            receipt_id = _strict_string(
                entry["promotion_receipt_id"],
                "index.promotion_receipt_id",
                allow_empty=True,
            )
            if not _DIGEST.fullmatch(content_hash):
                raise RegistryIntegrityError("registry content_hash is malformed")
            if gate_report_id and not _DIGEST.fullmatch(gate_report_id):
                raise RegistryIntegrityError("registry gate_report_id is malformed")
            if receipt_id and not _DIGEST.fullmatch(receipt_id):
                raise RegistryIntegrityError("registry promotion_receipt_id is malformed")
            return RegistryRecord(
                operator_id=_strict_string(operator_id, "index.operator_id"),
                content_hash=content_hash,
                lifecycle=OperatorLifecycle(entry["lifecycle"]),
                evidence_ids=_strict_string_array(entry["evidence_ids"], "index.evidence_ids"),
                gate_report_id=gate_report_id,
                promotion_receipt_id=receipt_id,
            )
        except (KeyError, TypeError, ValueError, RegistryIntegrityError) as error:
            raise RegistryIntegrityError(f"invalid registry index entry for {operator_id}: {error}") from error

    @staticmethod
    def _entry_for(
        passport: OperatorPassport,
        gate_report_id: str = "",
        promotion_receipt_id: str = "",
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "content_hash": passport.content_hash,
            "lifecycle": passport.lifecycle.value,
            "evidence_ids": list(passport.evidence_ids),
            "gate_report_id": gate_report_id,
            "promotion_receipt_id": promotion_receipt_id,
            "history": list(history or ()),
        }

    def register(self, passport: OperatorPassport) -> RegistryRecord:
        _validate_component(passport.name, "operator name")
        with self._locked():
            index = self._read_index()
            existing = index["operators"].get(passport.operator_id)
            if existing is not None:
                record = self._record_from_entry(passport.operator_id, existing)
                if record.content_hash != passport.content_hash:
                    raise RegistryConflictError(
                        f"{passport.operator_id} is already registered with different content"
                    )
                # Verify the complete record even on idempotent registration.
                self._verify_record_object(record)
                return record
            self._store_object(passport)
            index["operators"][passport.operator_id] = self._entry_for(passport)
            _atomic_write_json(self.index_path, index)
            return self._record_from_entry(passport.operator_id, index["operators"][passport.operator_id])

    def _verify_record_object(self, record: RegistryRecord) -> OperatorPassport:
        """Resolve and verify every index field against immutable content.

        Stable records additionally replay their promotion receipt.  Keeping
        this check in one place prevents read-only registry APIs from turning
        a forged index entry into apparently trusted metadata.
        """

        try:
            passport = self.get_by_hash(record.content_hash)
        except (KeyError, ValueError) as error:
            raise RegistryIntegrityError(
                "registry index contains an invalid content address"
            ) from error
        if (
            passport.operator_id != record.operator_id
            or passport.lifecycle is not record.lifecycle
            or tuple(passport.evidence_ids) != tuple(record.evidence_ids)
        ):
            raise RegistryIntegrityError(
                "registry index does not match its addressed operator object"
            )
        if record.lifecycle is OperatorLifecycle.STABLE:
            self._verify_stable_promotion(record, passport)
        elif record.gate_report_id or record.promotion_receipt_id:
            raise RegistryIntegrityError(
                "non-stable registry record contains promotion metadata"
            )
        return passport

    def get(self, operator_id: str) -> OperatorPassport:
        index = self._read_index()
        entry = index["operators"].get(operator_id)
        if not isinstance(entry, Mapping):
            raise KeyError(operator_id)
        record = self._record_from_entry(operator_id, entry)
        return self._verify_record_object(record)

    def list(self, lifecycle: OperatorLifecycle | str | None = None) -> tuple[RegistryRecord, ...]:
        selected = OperatorLifecycle(lifecycle) if lifecycle is not None else None
        index = self._read_index()
        records_list: list[RegistryRecord] = []
        for operator_id, entry in index["operators"].items():
            if not isinstance(operator_id, str) or not isinstance(entry, Mapping):
                raise RegistryIntegrityError("registry index contains an invalid operator entry")
            record = self._record_from_entry(operator_id, entry)
            self._verify_record_object(record)
            records_list.append(record)
        records = tuple(records_list)
        return tuple(
            sorted(
                (record for record in records if selected is None or record.lifecycle is selected),
                key=lambda record: record.operator_id,
            )
        )

    list_operators = list

    def history(self, operator_id: str) -> tuple[RegistryRecord, ...]:
        """Return the immutable lifecycle states ending with the current one."""

        index = self._read_index()
        entry = index["operators"].get(operator_id)
        if not isinstance(entry, Mapping):
            raise KeyError(operator_id)
        raw_history = entry.get("history", ())
        if not isinstance(raw_history, list):
            raise RegistryIntegrityError("operator lifecycle history is invalid")
        records: list[RegistryRecord] = []
        for raw_entry in raw_history:
            if not isinstance(raw_entry, Mapping):
                raise RegistryIntegrityError("operator lifecycle history entry is invalid")
            record = self._record_from_entry(operator_id, raw_entry)
            self._verify_record_object(record)
            records.append(record)
        current_record = self._record_from_entry(operator_id, entry)
        self._verify_record_object(current_record)
        records.append(current_record)
        return tuple(records)

    def _registered_object_at_hash(
        self,
        operator_id: str,
        content_hash: str,
    ) -> OperatorPassport:
        """Resolve an immutable object from any registered lifecycle point.

        Promotion receipts pin the exact baseline object used at evaluation
        time.  A later, legitimate lifecycle transition changes the current
        passport hash, so receipt replay must consult immutable history rather
        than silently retargeting the evidence to the current index entry.
        """

        index = self._read_index()
        entry = index["operators"].get(operator_id)
        if not isinstance(entry, Mapping):
            raise KeyError(operator_id)
        candidates = entry.get("history", ())
        if not isinstance(candidates, list):
            raise RegistryIntegrityError("operator lifecycle history is invalid")
        for raw_entry in (*candidates, entry):
            if not isinstance(raw_entry, Mapping):
                raise RegistryIntegrityError("operator lifecycle history entry is invalid")
            record = self._record_from_entry(operator_id, raw_entry)
            if record.content_hash != content_hash:
                continue
            try:
                passport = self.get_by_hash(content_hash)
            except (KeyError, ValueError) as error:
                raise RegistryIntegrityError(
                    "registered baseline content is unavailable"
                ) from error
            if (
                passport.operator_id != record.operator_id
                or passport.lifecycle is not record.lifecycle
                or tuple(passport.evidence_ids) != tuple(record.evidence_ids)
            ):
                raise RegistryIntegrityError(
                    "registered baseline history does not match immutable content"
                )
            return passport
        raise KeyError(content_hash)

    def _validate_promotion_evidence(
        self,
        candidate: OperatorPassport,
        gate_result: GateResult | None,
        lab_report: LabReport | None,
        promotion_policy: PromotionPolicy | None,
        *,
        allow_historical_baseline: bool = False,
    ) -> tuple[GateResult, LabReport, PromotionPolicy]:
        """Canonicalize evidence and independently reproduce the gate decision."""

        if not isinstance(gate_result, GateResult):
            raise LifecycleTransitionError("candidate -> stable requires a GateResult")
        if not isinstance(lab_report, LabReport):
            raise LifecycleTransitionError("candidate -> stable requires its complete LabReport")
        if not isinstance(promotion_policy, PromotionPolicy):
            raise LifecycleTransitionError("candidate -> stable requires its PromotionPolicy")
        try:
            canonical_report = _lab_report_from_primitive(to_primitive(lab_report))
            canonical_policy = _promotion_policy_from_primitive(to_primitive(promotion_policy))
            canonical_gate = _gate_result_from_primitive(to_primitive(gate_result))
        except RegistryIntegrityError as error:
            raise LifecycleTransitionError(f"promotion evidence is malformed: {error}") from error
        if canonical_report != lab_report or canonical_policy != promotion_policy or canonical_gate != gate_result:
            raise LifecycleTransitionError("promotion evidence is not canonical")
        if canonical_report.operator_id != candidate.operator_id:
            raise LifecycleTransitionError("lab report does not evaluate the candidate operator")
        if canonical_report.baseline_id != candidate.verification.baseline:
            raise LifecycleTransitionError("lab report baseline does not match passport verification")
        if (
            canonical_policy.baseline_id != candidate.verification.baseline
            or canonical_policy.baseline_id != canonical_report.baseline_id
            or not canonical_policy.baseline_content_hash
        ):
            raise LifecycleTransitionError(
                "promotion policy does not bind the passport baseline identity"
            )
        try:
            baseline = (
                self._registered_object_at_hash(
                    canonical_policy.baseline_id,
                    canonical_policy.baseline_content_hash,
                )
                if allow_historical_baseline
                else self.get(canonical_policy.baseline_id)
            )
        except (KeyError, RegistryIntegrityError) as error:
            raise LifecycleTransitionError(
                "promotion baseline is not a verified registry object"
            ) from error
        if (
            baseline.content_hash != canonical_policy.baseline_content_hash
            or baseline.operator_id == candidate.operator_id
            or baseline.lifecycle
            in {OperatorLifecycle.REJECTED, OperatorLifecycle.QUARANTINED}
        ):
            raise LifecycleTransitionError(
                "promotion policy baseline does not match an eligible registry object"
            )
        distinct_tasks = {outcome.task_id for outcome in canonical_report.outcomes if outcome.task_id}
        if len(distinct_tasks) < candidate.verification.minimum_tasks:
            raise LifecycleTransitionError(
                "lab report does not meet the passport minimum distinct task requirement"
            )
        policy_hash = sha256_json(canonical_policy)
        if canonical_gate.policy_hash != policy_hash:
            raise LifecycleTransitionError("gate result policy hash does not match PromotionPolicy")
        expected_gate = PromotionGate(canonical_policy).evaluate(
            canonical_report,
            operator=candidate,
        )
        if canonical_gate != expected_gate:
            raise LifecycleTransitionError("gate result does not match independent registry evaluation")
        if canonical_gate.decision is not PromotionDecision.PROMOTE:
            raise LifecycleTransitionError("only independently reproduced PROMOTE evidence can create stable")
        return canonical_gate, canonical_report, canonical_policy

    def _verify_stable_promotion(
        self,
        record: RegistryRecord,
        stable: OperatorPassport,
    ) -> PromotionReceipt:
        if stable.lifecycle is not OperatorLifecycle.STABLE:
            raise RegistryIntegrityError("promotion receipt can only authorize a stable passport")
        if not record.promotion_receipt_id or not _DIGEST.fullmatch(record.promotion_receipt_id):
            raise RegistryIntegrityError("stable registry record has no valid promotion receipt")
        receipt = self.promotion_receipt(record.promotion_receipt_id)
        if (
            receipt.operator_id != stable.operator_id
            or receipt.stable_content_hash != stable.content_hash
            or record.content_hash != receipt.stable_content_hash
            or record.gate_report_id != receipt.lab_report.report_id
        ):
            raise RegistryIntegrityError("promotion receipt does not match the stable registry record")
        try:
            candidate = self.get_by_hash(receipt.candidate_content_hash)
        except (KeyError, ValueError) as error:
            raise RegistryIntegrityError("promotion receipt candidate object is unavailable") from error
        if candidate.operator_id != stable.operator_id or candidate.lifecycle is not OperatorLifecycle.CANDIDATE:
            raise RegistryIntegrityError("promotion receipt does not reference the candidate lifecycle object")
        expected_stable = replace(candidate, lifecycle=OperatorLifecycle.STABLE, signature="")
        if stable != expected_stable:
            raise RegistryIntegrityError("stable object is not the receipt candidate's lifecycle successor")
        try:
            expected_gate, expected_report, expected_policy = self._validate_promotion_evidence(
                candidate,
                receipt.gate_result,
                receipt.lab_report,
                receipt.promotion_policy,
                allow_historical_baseline=True,
            )
        except LifecycleTransitionError as error:
            raise RegistryIntegrityError(f"promotion receipt failed semantic verification: {error}") from error
        if (
            expected_gate != receipt.gate_result
            or expected_report != receipt.lab_report
            or expected_policy != receipt.promotion_policy
        ):
            raise RegistryIntegrityError("promotion receipt evidence is not canonical")
        return receipt

    def transition(
        self,
        operator_id: str,
        target: OperatorLifecycle | str,
        *,
        gate_result: GateResult | None = None,
        lab_report: LabReport | None = None,
        promotion_policy: PromotionPolicy | None = None,
    ) -> RegistryRecord:
        target_lifecycle = OperatorLifecycle(target)
        with self._locked():
            index = self._read_index()
            entry = index["operators"].get(operator_id)
            if not isinstance(entry, Mapping):
                raise KeyError(operator_id)
            current_record = self._record_from_entry(operator_id, entry)
            current = self._verify_record_object(current_record)
            if current.lifecycle is target_lifecycle:
                if target_lifecycle is OperatorLifecycle.STABLE:
                    supplied_gate, supplied_report, supplied_policy = self._validate_promotion_evidence(
                        self.get_by_hash(
                            self._verify_stable_promotion(current_record, current).candidate_content_hash
                        ),
                        gate_result,
                        lab_report,
                        promotion_policy,
                        allow_historical_baseline=True,
                    )
                    stored_receipt = self._verify_stable_promotion(current_record, current)
                    if (
                        supplied_gate != stored_receipt.gate_result
                        or supplied_report != stored_receipt.lab_report
                        or supplied_policy != stored_receipt.promotion_policy
                    ):
                        raise LifecycleTransitionError("idempotent promotion evidence differs from receipt")
                return current_record
            if target_lifecycle not in _ALLOWED_TRANSITIONS[current.lifecycle]:
                raise LifecycleTransitionError(
                    f"lifecycle transition {current.lifecycle.value} -> {target_lifecycle.value} is not allowed"
                )
            gate_report_id = ""
            promotion_receipt_id = ""
            if target_lifecycle is OperatorLifecycle.STABLE:
                gate_result, lab_report, promotion_policy = self._validate_promotion_evidence(
                    current,
                    gate_result,
                    lab_report,
                    promotion_policy,
                )
                gate_report_id = lab_report.report_id
            elif any(item is not None for item in (gate_result, lab_report, promotion_policy)):
                raise LifecycleTransitionError("promotion evidence is only accepted for a stable transition")
            # Lifecycle participates in the semantic content hash, so a
            # signature over the previous state is necessarily stale.  Never
            # persist that stale attestation on the new object; a trusted
            # signer must explicitly sign the candidate/stable state it is
            # authorizing.
            updated = replace(current, lifecycle=target_lifecycle, signature="")
            self._store_object(updated)
            if target_lifecycle is OperatorLifecycle.STABLE:
                assert gate_result is not None and lab_report is not None and promotion_policy is not None
                promotion_receipt_id = self._store_promotion_receipt(
                    operator_id=operator_id,
                    candidate_content_hash=current.content_hash,
                    stable_content_hash=updated.content_hash,
                    lab_report=lab_report,
                    promotion_policy=promotion_policy,
                    gate_result=gate_result,
                )
            raw_history = entry.get("history", [])
            if not isinstance(raw_history, list):
                raise RegistryIntegrityError("operator lifecycle history is invalid")
            history = [*raw_history, dict(entry)]
            history[-1].pop("history", None)
            index["operators"][operator_id] = self._entry_for(
                updated,
                gate_report_id,
                promotion_receipt_id,
                history,
            )
            _atomic_write_json(self.index_path, index)
            record = self._record_from_entry(operator_id, index["operators"][operator_id])
            if target_lifecycle is OperatorLifecycle.STABLE:
                self._verify_stable_promotion(record, updated)
            return record

    def promote(
        self,
        operator_id: str,
        gate_result: GateResult,
        lab_report: LabReport,
        promotion_policy: PromotionPolicy,
    ) -> RegistryRecord:
        return self.transition(
            operator_id,
            OperatorLifecycle.STABLE,
            gate_result=gate_result,
            lab_report=lab_report,
            promotion_policy=promotion_policy,
        )

    def quarantine(self, operator_id: str) -> RegistryRecord:
        return self.transition(operator_id, OperatorLifecycle.QUARANTINED)

    def _installation_path(self, name: str) -> Path:
        return self.installations_dir / f"{_validate_component(name, 'operator name')}.json"

    def install(
        self,
        operator_id: str,
        *,
        signed_passport: OperatorPassport,
        trust_kernel: Any,
        actor: str,
    ) -> Installation:
        """Activate a promoted operator after receipt and release-signature checks.

        The registry owns promotion verification, while ``trust_kernel`` owns
        signer and principal authorization.  A detached signature is retained
        on the installation pointer because signatures are deliberately not
        part of the passport's semantic content hash.
        """

        with self._locked():
            index = self._read_index()
            entry = index["operators"].get(operator_id)
            if not isinstance(entry, Mapping):
                raise KeyError(operator_id)
            record = self._record_from_entry(operator_id, entry)
            passport = self.get_by_hash(record.content_hash)
            if passport.operator_id != operator_id or passport.lifecycle is not record.lifecycle:
                raise RegistryIntegrityError("registry index does not match its addressed operator object")
            if passport.lifecycle is not OperatorLifecycle.STABLE:
                raise LifecycleTransitionError("only stable operators can be installed")
            receipt = self._verify_stable_promotion(record, passport)
            # Do not publish a pointer that cannot be rehydrated after the
            # creating process exits. This also verifies generator integrity
            # before release authorization is consumed.
            self._load_prompt_template(passport)
            if not isinstance(signed_passport, OperatorPassport):
                raise LifecycleTransitionError("installation requires a signed stable OperatorPassport")
            if (
                signed_passport.operator_id != passport.operator_id
                or signed_passport.content_hash != passport.content_hash
                or signed_passport.lifecycle is not OperatorLifecycle.STABLE
                or replace(signed_passport, signature="") != passport
                or not _SIGNATURE.fullmatch(signed_passport.signature)
            ):
                raise LifecycleTransitionError("signed passport does not attest the exact stable object")
            from evolsa.kernel import TrustKernel

            if not isinstance(trust_kernel, TrustKernel):
                raise LifecycleTransitionError("installation requires a configured TrustKernel")
            authorize_release = getattr(trust_kernel, "authorize_release", None)
            try:
                release_decision = authorize_release(
                    signed_passport,
                    actor,
                    expected_hash=passport.content_hash,
                )
            except Exception as error:
                raise LifecycleTransitionError(f"stable passport release authorization failed: {error}") from error
            if (
                not getattr(release_decision, "release_authorized", False)
                or getattr(release_decision, "operator_id", "") != operator_id
                or getattr(release_decision, "content_hash", "") != passport.content_hash
            ):
                raise LifecycleTransitionError("trust kernel did not authorize the exact stable object")
            path = self._installation_path(passport.name)
            state: dict[str, Any]
            if path.exists():
                state = dict(_read_json(path))
                if state.get("schema_version") != INSTALL_SCHEMA or not isinstance(state.get("history"), list):
                    raise RegistryIntegrityError("installation state schema is invalid")
            else:
                state = {
                    "schema_version": INSTALL_SCHEMA,
                    "name": passport.name,
                    "active": None,
                    "history": [],
                }
            active = state.get("active")
            new_active = {
                "operator_id": operator_id,
                "content_hash": record.content_hash,
                "promotion_receipt_id": receipt.receipt_id,
                "passport_signature": signed_passport.signature,
                "release_signer": getattr(release_decision, "signer_id", ""),
                "release_actor": getattr(release_decision, "release_actor", ""),
            }
            if active == new_active:
                return Installation(passport.name, operator_id, record.content_hash)
            if active is not None:
                if not isinstance(active, Mapping):
                    raise RegistryIntegrityError("active installation pointer is invalid")
                state["history"].append(dict(active))
            state["active"] = new_active
            _atomic_write_json(path, state)
            return Installation(passport.name, operator_id, record.content_hash)

    install_stable = install

    def _passport_from_install_pointer(
        self,
        name: str,
        active: Mapping[str, Any],
        *,
        trust_kernel: Any,
        actor: str,
    ) -> OperatorPassport:
        expected_keys = {
            "operator_id",
            "content_hash",
            "promotion_receipt_id",
            "passport_signature",
            "release_signer",
            "release_actor",
        }
        if set(active) != expected_keys:
            raise RegistryIntegrityError("installation pointer schema is invalid")
        content_hash = active.get("content_hash")
        if not isinstance(content_hash, str) or not _DIGEST.fullmatch(content_hash):
            raise RegistryIntegrityError("installation pointer has an invalid content address")
        try:
            passport = self.get_by_hash(content_hash)
        except (KeyError, ValueError) as error:
            raise RegistryIntegrityError("installation pointer has an invalid content address") from error
        if passport.operator_id != active.get("operator_id") or passport.name != name:
            raise RegistryIntegrityError("installation pointer does not match addressed content")
        if passport.lifecycle is not OperatorLifecycle.STABLE:
            raise RegistryIntegrityError("installation pointer does not reference stable content")
        index = self._read_index()
        entry = index["operators"].get(passport.operator_id)
        if not isinstance(entry, Mapping):
            raise RegistryIntegrityError("installed operator is missing from the registry index")
        current = self._record_from_entry(passport.operator_id, entry)
        if current.lifecycle is not OperatorLifecycle.STABLE or current.content_hash != passport.content_hash:
            raise LifecycleTransitionError("installed operator is no longer in the stable lifecycle")
        receipt_id = active.get("promotion_receipt_id")
        if receipt_id != current.promotion_receipt_id:
            raise RegistryIntegrityError("installation pointer does not match the promotion receipt")
        self._verify_record_object(current)
        signature = active.get("passport_signature")
        if not isinstance(signature, str) or not _SIGNATURE.fullmatch(signature):
            raise RegistryIntegrityError("installation pointer has no valid stable-passport signature")
        release_signer = active.get("release_signer")
        release_actor = active.get("release_actor")
        signature_signer = signature.split(":", 2)[1]
        if (
            not isinstance(release_signer, str)
            or release_signer != signature_signer
            or not isinstance(release_actor, str)
            or not release_actor
        ):
            raise RegistryIntegrityError("installation release attestation is malformed")
        signed_passport = replace(passport, signature=signature)
        from evolsa.kernel import TrustKernel

        if not isinstance(trust_kernel, TrustKernel):
            raise RegistryIntegrityError(
                "installed operator verification requires a configured TrustKernel"
            )
        try:
            release_decision = trust_kernel.authorize_release(
                signed_passport,
                actor,
                expected_hash=passport.content_hash,
            )
        except Exception as error:
            raise RegistryIntegrityError(
                f"installed operator release authorization failed: {error}"
            ) from error
        if (
            not release_decision.release_authorized
            or release_decision.operator_id != passport.operator_id
            or release_decision.content_hash != passport.content_hash
            or release_decision.signer_id != release_signer
            or release_decision.release_actor != release_actor
            or actor != release_actor
        ):
            raise RegistryIntegrityError(
                "installation release attestation does not match trusted authorization"
            )
        return signed_passport

    def installed(
        self,
        name: str,
        *,
        trust_kernel: Any,
        actor: str,
    ) -> OperatorPassport:
        path = self._installation_path(name)
        if not path.exists():
            raise KeyError(name)
        state = _read_json(path)
        if (
            state.get("schema_version") != INSTALL_SCHEMA
            or state.get("name") != name
            or not isinstance(state.get("active"), Mapping)
            or not isinstance(state.get("history"), list)
        ):
            raise RegistryIntegrityError("installation state schema is invalid")
        return self._passport_from_install_pointer(
            name,
            state["active"],
            trust_kernel=trust_kernel,
            actor=actor,
        )

    resolve = installed

    def rollback(
        self,
        name: str,
        *,
        trust_kernel: Any,
        actor: str,
    ) -> Installation:
        """Atomically restore the previous installed stable object."""

        with self._locked():
            path = self._installation_path(name)
            if not path.exists():
                raise KeyError(name)
            state = dict(_read_json(path))
            history = state.get("history")
            if (
                state.get("schema_version") != INSTALL_SCHEMA
                or state.get("name") != name
                or not isinstance(history, list)
            ):
                raise RegistryIntegrityError("installation state schema is invalid")
            if not history:
                raise RegistryError(f"no previous installation is available for {name}")
            active = state.get("active")
            if not isinstance(active, Mapping):
                raise RegistryIntegrityError("active installation pointer is invalid")
            self._passport_from_install_pointer(
                name,
                active,
                trust_kernel=trust_kernel,
                actor=actor,
            )
            previous = history.pop()
            if not isinstance(previous, Mapping):
                raise RegistryIntegrityError("rollback pointer is invalid")
            passport = self._passport_from_install_pointer(
                name,
                previous,
                trust_kernel=trust_kernel,
                actor=actor,
            )
            state["active"] = dict(previous)
            state["history"] = history
            _atomic_write_json(path, state)
            return Installation(name, passport.operator_id, passport.content_hash)

    rollback_installation = rollback


OperatorRegistry = LocalRegistry


__all__ = [
    "Installation",
    "LifecycleTransitionError",
    "LocalRegistry",
    "OperatorRegistry",
    "PromotionReceipt",
    "RegistryConflictError",
    "RegistryError",
    "RegistryIntegrityError",
    "RegistryRecord",
]
