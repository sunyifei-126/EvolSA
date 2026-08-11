"""Versioned data contracts for bounded recursive self-improvement."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .hashing import sha256_json


_SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_OPERATOR_NAME = re.compile(r"^[a-z][a-z0-9-]{1,63}$")
_SHA256_ID = re.compile(r"^[a-f0-9]{64}$")
_CONTENT_ADDRESSED_GENERATOR_REF = re.compile(r"^sha256:[a-f0-9]{64}$")
_RELATIVE_GENERATOR_REF = re.compile(
    r"^(?!/)(?!.*[\\:])(?!\.\.(?:/|$))(?!.*?/\.\.(?:/|$))"
    r"(?!.*[\x00-\x1f\x7f])[^/]+(?:/[^/]+)*$"
)
_GENERATOR_TYPES = frozenset({"prompt_template", "tool_graph", "python", "adapter"})


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class ExecutionStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    INVALID = "invalid"
    BUDGET_EXHAUSTED = "budget_exhausted"


class OperatorLifecycle(str, Enum):
    PROPOSED = "proposed"
    SHADOW = "shadow"
    CANDIDATE = "candidate"
    STABLE = "stable"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    DEPRECATED = "deprecated"


class PromotionDecision(str, Enum):
    PROMOTE = "promote"
    SHADOW = "shadow"
    REVISE = "revise"
    REJECT = "reject"
    QUARANTINE = "quarantine"


class EffectScope(str, Enum):
    TASK_ARTIFACT = "task_artifact"
    OPERATOR = "operator"
    SCHEDULER = "scheduler"
    MEMORY_POLICY = "memory_policy"
    EVALUATOR = "evaluator"
    SANDBOX = "sandbox"
    BUDGET = "budget"
    PROMOTION_GATE = "promotion_gate"
    EVENT_LOG = "event_log"
    RELEASE_PERMISSION = "release_permission"


EVOLVABLE_SCOPES = frozenset(
    {
        EffectScope.TASK_ARTIFACT,
        EffectScope.OPERATOR,
        EffectScope.SCHEDULER,
        EffectScope.MEMORY_POLICY,
    }
)
PROTECTED_SCOPES = frozenset(set(EffectScope) - set(EVOLVABLE_SCOPES))
ORDERED_PROTECTED_SCOPES = tuple(sorted(PROTECTED_SCOPES, key=lambda scope: scope.value))


def _finite_nonnegative(value: float, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be finite and non-negative")


def _strict_string_tuple(
    value: Sequence[str],
    label: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    """Normalize an internal string sequence without coercing its members."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence of strings")
    normalized = tuple(value)
    if required and not normalized:
        raise ValueError(f"{label} must contain at least one value")
    if any(type(item) is not str or not item for item in normalized):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must contain unique values")
    return normalized


def _safe_generator_ref(value: str) -> bool:
    """Accept portable relative registry refs or an explicit SHA-256 ref."""

    return bool(
        _CONTENT_ADDRESSED_GENERATOR_REF.fullmatch(value)
        or _RELATIVE_GENERATOR_REF.fullmatch(value)
    )


@dataclass(frozen=True, slots=True)
class Budget:
    """Immutable resource ceiling.

    ``max_tokens == 0`` and ``max_cost == 0`` mean that the corresponding
    meter is not configured. Time and attempt ceilings are always enforced.
    This sentinel convention is shared by operators, search, and the trusted
    budget ledger.
    """

    max_tokens: int = 0
    max_exec_seconds: float = 60.0
    max_attempts: int = 1
    max_cost: float = 0.0

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or type(self.max_attempts) is not int:
            raise ValueError("budget token and attempt ceilings must be integers")
        if type(self.max_exec_seconds) not in (int, float) or type(self.max_cost) not in (
            int,
            float,
        ):
            raise ValueError("budget time and cost ceilings must be numeric")
        if self.max_tokens < 0 or self.max_attempts < 1:
            raise ValueError("budget requires max_tokens >= 0 and max_attempts >= 1")
        _finite_nonnegative(self.max_exec_seconds, "max_exec_seconds")
        _finite_nonnegative(self.max_cost, "max_cost")
        object.__setattr__(self, "max_exec_seconds", float(self.max_exec_seconds))
        object.__setattr__(self, "max_cost", float(self.max_cost))


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    tokens: int = 0
    exec_seconds: float = 0.0
    attempts: int = 0
    cost: float = 0.0

    def __post_init__(self) -> None:
        if self.tokens < 0 or self.attempts < 0:
            raise ValueError("usage counters cannot be negative")
        _finite_nonnegative(self.exec_seconds, "exec_seconds")
        _finite_nonnegative(self.cost, "cost")


@dataclass(frozen=True, slots=True)
class Artifact:
    content: str
    media_type: str = "application/json"
    artifact_id: str = ""
    parent_ids: tuple[str, ...] = ()
    created_by: str = "seed"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.media_type or not self.created_by:
            raise ValueError("artifact media_type and created_by are required")
        digest = sha256_json({"content": self.content, "media_type": self.media_type})
        if self.artifact_id and self.artifact_id != digest:
            raise ValueError("artifact_id does not match content")
        object.__setattr__(self, "artifact_id", digest)
        object.__setattr__(self, "parent_ids", tuple(self.parent_ids))


@dataclass(frozen=True, slots=True)
class TaskState:
    task_id: str
    split: str
    seed: int
    specification: str
    workspace: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id or not self.split or not self.specification:
            raise ValueError("task_id, split, and specification are required")


@dataclass(frozen=True, slots=True)
class Observation:
    status: ExecutionStatus
    output: str = ""
    logs: str = ""
    error_signature: str = ""
    runtime_seconds: float = 0.0
    token_usage: int = 0
    cost: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _finite_nonnegative(self.runtime_seconds, "runtime_seconds")
        _finite_nonnegative(self.cost, "cost")
        if self.token_usage < 0:
            raise ValueError("token_usage cannot be negative")


@dataclass(frozen=True, slots=True)
class ScoreCard:
    primary: float
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER
    metrics: Mapping[str, float] = field(default_factory=dict)
    guardrails: Mapping[str, float] = field(default_factory=dict)
    valid: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.primary):
            raise ValueError("primary score must be finite")
        for name, value in (*self.metrics.items(), *self.guardrails.items()):
            if not name or not math.isfinite(value):
                raise ValueError("metric names must be non-empty and values finite")

    @property
    def utility(self) -> float:
        return self.primary if self.direction is MetricDirection.HIGHER_IS_BETTER else -self.primary


@dataclass(frozen=True, slots=True)
class Applicability:
    applies: bool
    confidence: float
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("applicability confidence must be in [0, 1]")
        object.__setattr__(self, "reasons", tuple(self.reasons))


@dataclass(frozen=True, slots=True)
class OperatorVerification:
    baseline: str
    primary_metric: str = "paired_lift"
    guardrails: tuple[str, ...] = ("valid_rate", "cost", "regression")
    minimum_tasks: int = 4

    def __post_init__(self) -> None:
        if type(self.baseline) is not str or not self.baseline:
            raise ValueError("operator verification baseline must be a non-empty string")
        if type(self.primary_metric) is not str or not self.primary_metric:
            raise ValueError("operator verification primary_metric must be a non-empty string")
        if type(self.minimum_tasks) is not int or self.minimum_tasks < 1:
            raise ValueError(
                "operator verification requires a baseline, metric, and minimum_tasks >= 1"
            )
        object.__setattr__(
            self,
            "guardrails",
            _strict_string_tuple(self.guardrails, "operator verification guardrails"),
        )


@dataclass(frozen=True, slots=True)
class OperatorPassport:
    name: str
    version: str
    intent: str
    inputs: tuple[str, ...]
    preconditions: tuple[str, ...]
    expected_effects: tuple[EffectScope, ...]
    forbidden_effects: tuple[EffectScope, ...]
    context_policy: str
    generator_type: str
    generator_ref: str
    budget: Budget
    verification: OperatorVerification
    parents: tuple[str, ...] = ()
    known_failures: tuple[str, ...] = ()
    lifecycle: OperatorLifecycle = OperatorLifecycle.PROPOSED
    evidence_ids: tuple[str, ...] = ()
    signature: str = ""
    api_version: str = "evolsa/v1"

    def __post_init__(self) -> None:
        if type(self.api_version) is not str or self.api_version != "evolsa/v1":
            raise ValueError("unsupported operator passport api_version")
        if type(self.name) is not str or not _OPERATOR_NAME.fullmatch(self.name):
            raise ValueError("operator name must match ^[a-z][a-z0-9-]{1,63}$")
        if type(self.version) is not str or not _SEMVER.fullmatch(self.version):
            raise ValueError("operator version must be valid semantic versioning")
        if type(self.intent) is not str or len(self.intent) < 8:
            raise ValueError("operator intent must contain at least 8 characters")
        inputs = _strict_string_tuple(self.inputs, "operator inputs", required=True)
        preconditions = _strict_string_tuple(self.preconditions, "operator preconditions")
        if type(self.context_policy) is not str or not self.context_policy:
            raise ValueError("operator context_policy must be a non-empty string")
        if type(self.generator_type) is not str or self.generator_type not in _GENERATOR_TYPES:
            raise ValueError(f"unsupported operator generator_type: {self.generator_type!r}")
        if type(self.generator_ref) is not str or not _safe_generator_ref(self.generator_ref):
            raise ValueError(
                "generator_ref must be a portable relative reference without traversal "
                "or sha256:<64 lowercase hex>"
            )
        if type(self.budget) is not Budget:
            raise ValueError("operator budget must be a Budget")
        if type(self.verification) is not OperatorVerification:
            raise ValueError("operator verification must be an OperatorVerification")
        if isinstance(self.expected_effects, (str, bytes)) or not isinstance(
            self.expected_effects, Sequence
        ):
            raise ValueError("operator expected_effects must be a sequence")
        if isinstance(self.forbidden_effects, (str, bytes)) or not isinstance(
            self.forbidden_effects, Sequence
        ):
            raise ValueError("operator forbidden_effects must be a sequence")
        try:
            expected = tuple(EffectScope(item) for item in self.expected_effects)
            forbidden = tuple(EffectScope(item) for item in self.forbidden_effects)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid operator effect scope: {error}") from error
        if len(expected) != len(set(expected)) or len(forbidden) != len(set(forbidden)):
            raise ValueError("operator effect scopes must contain unique values")
        if set(expected) & set(forbidden):
            raise ValueError("an effect cannot be both expected and forbidden")
        if set(expected) & set(PROTECTED_SCOPES):
            raise ValueError("operators cannot declare protected effects")
        parents = _strict_string_tuple(self.parents, "operator parents")
        known_failures = _strict_string_tuple(self.known_failures, "operator known_failures")
        evidence_ids = _strict_string_tuple(self.evidence_ids, "operator evidence_ids")
        if any(not _SHA256_ID.fullmatch(evidence_id) for evidence_id in evidence_ids):
            raise ValueError("operator evidence_ids must be 64-character lowercase SHA-256 values")
        if type(self.signature) is not str:
            raise ValueError("operator signature must be a string")
        try:
            lifecycle = OperatorLifecycle(self.lifecycle)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid operator lifecycle: {self.lifecycle!r}") from error
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "preconditions", preconditions)
        object.__setattr__(self, "expected_effects", expected)
        object.__setattr__(self, "forbidden_effects", forbidden)
        object.__setattr__(self, "parents", parents)
        object.__setattr__(self, "known_failures", known_failures)
        object.__setattr__(self, "lifecycle", lifecycle)
        object.__setattr__(self, "evidence_ids", evidence_ids)

    @property
    def operator_id(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def content_hash(self) -> str:
        payload = {
            "api_version": self.api_version,
            "name": self.name,
            "version": self.version,
            "intent": self.intent,
            "inputs": self.inputs,
            "preconditions": self.preconditions,
            "expected_effects": self.expected_effects,
            "forbidden_effects": self.forbidden_effects,
            "context_policy": self.context_policy,
            "generator_type": self.generator_type,
            "generator_ref": self.generator_ref,
            "budget": self.budget,
            "verification": self.verification,
            "parents": self.parents,
            "known_failures": self.known_failures,
            "lifecycle": self.lifecycle,
            "evidence_ids": self.evidence_ids,
        }
        return sha256_json(payload)


@dataclass(frozen=True, slots=True)
class SearchNode:
    node_id: str
    artifact: Artifact
    operator_id: str
    parent_ids: tuple[str, ...]
    generation: int
    observation: Observation
    score: ScoreCard
    method_family: str
    plan: str = ""
    task_id: str = ""
    split: str = ""
    seed: int = -1
    evaluator_id: str = ""
    model_id: str = ""
    budget: Budget | None = None

    def __post_init__(self) -> None:
        if (
            not self.node_id
            or not self.operator_id
            or not self.method_family
            or self.generation < 0
            or self.seed < -1
        ):
            raise ValueError("invalid search node identity")
        object.__setattr__(self, "parent_ids", tuple(self.parent_ids))


@dataclass(frozen=True, slots=True)
class ExperienceCard:
    node_id: str
    step_id: int
    operator_id: str
    parent_node_ids: tuple[str, ...]
    generation_id: int
    score: float
    utility: float
    status: ExecutionStatus
    error_signature: str
    runtime_seconds: float
    token_count: int
    cost: float
    method_family: str
    family_count_before: int
    delta_vs_parent: float
    novelty_score: float
    rank: int
    current_best: bool
    selection_utility: float
    plan: str = ""
    analysis: str = ""
    method_overview: str = ""
    parent_comparison_experience: str = ""
    change_tags: tuple[str, ...] = ()
    valid: bool = False
    validity_reason: str = ""
    guardrails: Mapping[str, float] = field(default_factory=dict)
    artifact_id: str = ""
    task_id: str = ""
    split: str = ""
    seed: int = -1
    evaluator_id: str = ""
    model_id: str = ""
    budget: Budget | None = None
    card_id: str = ""

    def __post_init__(self) -> None:
        if (
            self.step_id < 0
            or self.generation_id < 0
            or self.rank < 1
            or self.family_count_before < 0
            or self.seed < -1
        ):
            raise ValueError("experience card counters are invalid")
        if type(self.valid) is not bool:
            raise ValueError("experience card valid must be a boolean")
        if self.token_count < 0:
            raise ValueError("token_count cannot be negative")
        for value, label in ((self.runtime_seconds, "runtime_seconds"), (self.cost, "cost")):
            _finite_nonnegative(value, label)
        object.__setattr__(self, "parent_node_ids", tuple(self.parent_node_ids))
        object.__setattr__(self, "change_tags", tuple(self.change_tags))
        payload = {
            "node_id": self.node_id,
            "step_id": self.step_id,
            "operator_id": self.operator_id,
            "parent_node_ids": self.parent_node_ids,
            "generation_id": self.generation_id,
            "score": self.score,
            "utility": self.utility,
            "status": self.status,
            "error_signature": self.error_signature,
            "runtime_seconds": self.runtime_seconds,
            "token_count": self.token_count,
            "cost": self.cost,
            "method_family": self.method_family,
            "family_count_before": self.family_count_before,
            "delta_vs_parent": self.delta_vs_parent,
            "novelty_score": self.novelty_score,
            "rank": self.rank,
            "current_best": self.current_best,
            "selection_utility": self.selection_utility,
            "plan": self.plan,
            "analysis": self.analysis,
            "method_overview": self.method_overview,
            "parent_comparison_experience": self.parent_comparison_experience,
            "change_tags": self.change_tags,
            "valid": self.valid,
            "validity_reason": self.validity_reason,
            "guardrails": self.guardrails,
            "artifact_id": self.artifact_id,
            "task_id": self.task_id,
            "split": self.split,
            "seed": self.seed,
            "evaluator_id": self.evaluator_id,
            "model_id": self.model_id,
            "budget": self.budget,
        }
        digest = sha256_json(payload)
        if self.card_id and self.card_id != digest:
            raise ValueError("card_id does not match experience card content")
        object.__setattr__(self, "card_id", digest)


@dataclass(frozen=True, slots=True)
class ExperienceBoard:
    num_nodes: int
    best_node_id: str
    best_score: float
    current_best_family: str
    method_family_stats: Mapping[str, Mapping[str, float | int | str]]
    family_best_nodes: Mapping[str, str]
    underexplored_families: tuple[str, ...]
    score_history: tuple[float, ...]
    recent_delta_trend: float
    repeated_errors: Mapping[str, int]
    status_count: Mapping[str, int]
    operator_counts: Mapping[str, int]
    parent_graph: Mapping[str, tuple[str, ...]]
    novelty_by_node: Mapping[str, float]
    rank_by_node: Mapping[str, int]
    runtime_stats: Mapping[str, float]
    parent_selection_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_nodes < 0:
            raise ValueError("num_nodes cannot be negative")
        object.__setattr__(self, "underexplored_families", tuple(self.underexplored_families))
        object.__setattr__(self, "score_history", tuple(self.score_history))


@dataclass(frozen=True, slots=True)
class PairedOutcome:
    task_id: str
    seed: int
    candidate_score: float
    baseline_score: float
    candidate_valid: bool
    baseline_valid: bool
    candidate_cost: float
    baseline_cost: float
    regression: float = 0.0
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER
    split: str = ""
    evaluator_id: str = ""
    candidate_parent_id: str = ""
    baseline_parent_id: str = ""
    candidate_model_id: str = ""
    baseline_model_id: str = ""
    candidate_sampling: Mapping[str, Any] | None = None
    baseline_sampling: Mapping[str, Any] | None = None
    candidate_budget: Budget | None = None
    baseline_budget: Budget | None = None

    def __post_init__(self) -> None:
        if not self.task_id or self.seed < 0:
            raise ValueError("paired outcome requires a task_id and non-negative seed")
        if type(self.candidate_valid) is not bool or type(self.baseline_valid) is not bool:
            raise ValueError("paired outcome validity fields must be booleans")
        for value, label in (
            (self.candidate_score, "candidate_score"),
            (self.baseline_score, "baseline_score"),
            (self.candidate_cost, "candidate_cost"),
            (self.baseline_cost, "baseline_cost"),
            (self.regression, "regression"),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.candidate_cost < 0 or self.baseline_cost < 0 or self.regression < 0:
            raise ValueError("paired outcome costs and regression must be non-negative")
        object.__setattr__(self, "direction", MetricDirection(self.direction))
        if self.candidate_sampling is not None:
            object.__setattr__(self, "candidate_sampling", dict(self.candidate_sampling))
        if self.baseline_sampling is not None:
            object.__setattr__(self, "baseline_sampling", dict(self.baseline_sampling))

    @property
    def lift(self) -> float:
        difference = self.candidate_score - self.baseline_score
        return difference if self.direction is MetricDirection.HIGHER_IS_BETTER else -difference

    @property
    def budget_evidence_matches(self) -> bool:
        return (
            self.candidate_budget is not None
            and self.baseline_budget is not None
            and self.candidate_budget == self.baseline_budget
        )

    @property
    def condition_evidence_matches(self) -> bool:
        return (
            bool(self.split)
            and bool(self.evaluator_id)
            and bool(self.candidate_parent_id)
            and self.candidate_parent_id == self.baseline_parent_id
            and bool(self.candidate_model_id)
            and self.candidate_model_id == self.baseline_model_id
            and self.candidate_sampling is not None
            and self.baseline_sampling is not None
            and dict(self.candidate_sampling) == dict(self.baseline_sampling)
            and self.budget_evidence_matches
        )


@dataclass(frozen=True, slots=True)
class LabReport:
    operator_id: str
    baseline_id: str
    outcomes: tuple[PairedOutcome, ...]
    mean_lift: float
    lcb95: float
    ucb95: float
    valid_rate: float
    baseline_valid_rate: float
    cost_adjusted_lift: float
    transfer_coverage: int
    worst_regression: float
    matched_budget: bool
    matched_conditions: bool = False
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 0
    bootstrap_confidence: float = 0.95
    cost_penalty: float = 1.0
    report_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        if type(self.matched_budget) is not bool or type(self.matched_conditions) is not bool:
            raise ValueError("lab matching fields must be booleans")
        if self.bootstrap_samples < 1 or not 0.0 < self.bootstrap_confidence < 1.0:
            raise ValueError("lab bootstrap configuration is invalid")
        if not math.isfinite(self.cost_penalty) or self.cost_penalty < 0:
            raise ValueError("lab cost penalty must be finite and non-negative")
        metrics = (
            self.mean_lift,
            self.lcb95,
            self.ucb95,
            self.valid_rate,
            self.baseline_valid_rate,
            self.cost_adjusted_lift,
            self.worst_regression,
        )
        if not all(math.isfinite(metric) for metric in metrics):
            raise ValueError("lab report metrics must be finite")
        digest = sha256_json({key: value for key, value in self.__dict__.items() if key != "report_id"}) if hasattr(self, "__dict__") else sha256_json({
            "operator_id": self.operator_id,
            "baseline_id": self.baseline_id,
            "outcomes": self.outcomes,
            "mean_lift": self.mean_lift,
            "lcb95": self.lcb95,
            "ucb95": self.ucb95,
            "valid_rate": self.valid_rate,
            "baseline_valid_rate": self.baseline_valid_rate,
            "cost_adjusted_lift": self.cost_adjusted_lift,
            "transfer_coverage": self.transfer_coverage,
            "worst_regression": self.worst_regression,
            "matched_budget": self.matched_budget,
            "matched_conditions": self.matched_conditions,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_confidence": self.bootstrap_confidence,
            "cost_penalty": self.cost_penalty,
        })
        if self.report_id and self.report_id != digest:
            raise ValueError("report_id does not match report content")
        object.__setattr__(self, "report_id", digest)


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    minimum_lcb_lift: float = 0.0
    valid_rate_tolerance: float = 0.0
    minimum_cost_adjusted_lift: float = 0.0
    maximum_regression: float = 0.0
    minimum_transfer_coverage: int = 4
    require_matched_budget: bool = True
    require_matched_conditions: bool = True
    baseline_id: str = ""
    baseline_content_hash: str = ""
    required_trials: tuple[tuple[str, int], ...] = ()
    required_split: str = "promotion"
    required_evaluator_id: str = ""
    metric_direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 0
    bootstrap_confidence: float = 0.95
    cost_penalty: float = 1.0

    def __post_init__(self) -> None:
        if type(self.minimum_transfer_coverage) is not int or self.minimum_transfer_coverage < 1:
            raise ValueError("minimum_transfer_coverage must be positive")
        if (
            isinstance(self.minimum_lcb_lift, bool)
            or not isinstance(self.minimum_lcb_lift, (int, float))
            or not math.isfinite(self.minimum_lcb_lift)
            or isinstance(self.minimum_cost_adjusted_lift, bool)
            or not isinstance(self.minimum_cost_adjusted_lift, (int, float))
            or not math.isfinite(self.minimum_cost_adjusted_lift)
        ):
            raise ValueError("promotion lift thresholds must be finite")
        _finite_nonnegative(self.valid_rate_tolerance, "valid_rate_tolerance")
        _finite_nonnegative(self.maximum_regression, "maximum_regression")
        if type(self.require_matched_budget) is not bool or type(self.require_matched_conditions) is not bool:
            raise ValueError("promotion evidence requirements must be booleans")
        if type(self.baseline_id) is not str or type(self.baseline_content_hash) is not str:
            raise ValueError("promotion baseline identity must be strings")
        if self.baseline_content_hash and not _SHA256_ID.fullmatch(self.baseline_content_hash):
            raise ValueError("promotion baseline_content_hash must be a lowercase SHA-256 digest")
        if type(self.required_split) is not str or not self.required_split:
            raise ValueError("promotion required_split must be non-empty")
        if type(self.required_evaluator_id) is not str:
            raise ValueError("promotion required_evaluator_id must be a string")
        if isinstance(self.required_trials, (str, bytes)) or not isinstance(
            self.required_trials, Sequence
        ):
            raise ValueError("promotion required_trials must be a sequence")
        trials: list[tuple[str, int]] = []
        for trial in self.required_trials:
            if (
                isinstance(trial, (str, bytes))
                or not isinstance(trial, Sequence)
                or len(trial) != 2
                or type(trial[0]) is not str
                or not trial[0]
                or type(trial[1]) is not int
                or trial[1] < 0
            ):
                raise ValueError("each required trial must contain a task_id and non-negative seed")
            trials.append((trial[0], trial[1]))
        if len(trials) != len(set(trials)):
            raise ValueError("promotion required_trials must be unique")
        if type(self.bootstrap_samples) is not int or self.bootstrap_samples < 1:
            raise ValueError("promotion bootstrap_samples must be positive")
        if type(self.bootstrap_seed) is not int:
            raise ValueError("promotion bootstrap_seed must be an integer")
        if (
            isinstance(self.bootstrap_confidence, bool)
            or not isinstance(self.bootstrap_confidence, (int, float))
            or not math.isfinite(self.bootstrap_confidence)
            or not 0.0 < self.bootstrap_confidence < 1.0
        ):
            raise ValueError("promotion bootstrap_confidence must be between zero and one")
        _finite_nonnegative(self.cost_penalty, "cost_penalty")
        object.__setattr__(self, "required_trials", tuple(trials))
        object.__setattr__(self, "metric_direction", MetricDirection(self.metric_direction))


@dataclass(frozen=True, slots=True)
class GateResult:
    decision: PromotionDecision
    operator_id: str
    report_id: str
    reasons: tuple[str, ...]
    policy_hash: str


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    mode: str
    task_pack: str
    seeds: tuple[int, ...]
    budget: Budget
    model_id: str
    operator_ids: tuple[str, ...]
    protected_scopes: tuple[EffectScope, ...] = ORDERED_PROTECTED_SCOPES
    schema_version: str = "evolsa.run/v1"

    def __post_init__(self) -> None:
        if not self.run_id or self.mode not in {"library", "search", "rsi"} or not self.seeds:
            raise ValueError("run manifest requires a run_id, supported mode, and seeds")
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "operator_ids", tuple(self.operator_ids))
        scopes = tuple(EffectScope(scope) for scope in self.protected_scopes)
        if len(scopes) != len(set(scopes)):
            raise ValueError("run manifest protected scopes must be unique")
        object.__setattr__(self, "protected_scopes", tuple(sorted(scopes, key=lambda scope: scope.value)))


def best_score(cards: Sequence[ExperienceCard]) -> float:
    if not cards:
        raise ValueError("at least one experience card is required")
    return max(card.utility for card in cards)
