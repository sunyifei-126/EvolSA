"""End-to-end, deterministic bounded-RSI demonstration and replay.

The demo is intentionally small enough to run without a network connection or
third-party dependency. It exercises the same separation used by larger EvolSA
deployments: task-artifact search is evolvable, while evaluation, budgets,
evidence storage, promotion, and release authorization remain protected.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from evolsa.core import (
    Artifact,
    Budget,
    EffectScope,
    ExecutionStatus,
    ExperienceCard,
    LabReport,
    OperatorLifecycle,
    PairedOutcome,
    PromotionPolicy,
    PROTECTED_SCOPES,
    RunManifest,
    SearchNode,
)
from evolsa.core.serde import pretty_json, to_primitive
from evolsa.foundry import OperatorCompiler, OperatorSynthesizer, PatternKind, TraceMiner
from evolsa.kernel import ContentAddressedStore, EventLog, TrustKernel, sign_passport
from evolsa.lab import PromotionGate, aggregate_lab_report
from evolsa.operators import ImproveOperator, bind_operation_family, seed_operators
from evolsa.registry import LocalRegistry
from evolsa.report import render_lineage_html, render_review_markdown, write_report
from evolsa.schemas import operator_passport_document, operator_passport_from_document
from evolsa.search import SearchEngine

from .incident_routing import (
    DeterministicModel,
    IncidentRoutingAdapter,
    infer_normalizer_from_feedback,
    parse_policy,
    policy_with_normalizer,
)


RUN_SCHEMA = "evolsa.offline-rsi-demo/v1"
INDEX_SCHEMA = "evolsa.evidence-index/v1"
RUN_ID = "offline-rsi-demo-v1"
_DEMO_SIGNER = "offline-demo-kernel"
_DEMO_ACTOR = "offline-demo-release"
_DEMO_SECRET = b"evolsa-offline-demo-key-v1"
_DEMO_SEEDS = (0, 1, 2, 3)
_DEMO_LAB_BUDGET = Budget(max_tokens=256, max_exec_seconds=1.0, max_attempts=1)
_DEMO_TRAJECTORY_BUDGET = Budget(
    max_tokens=256,
    max_exec_seconds=1.0,
    max_attempts=5,
    max_cost=1.0,
)
_DEMO_PROTECTED_SCOPES = tuple(sorted(PROTECTED_SCOPES, key=lambda scope: scope.value))
_EVENT_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CLAIM_BOUNDARY = (
    "This deterministic result demonstrates a persistent improvement to the improvement "
    "process in matched sealed-search trajectories under fixed evaluators and budgets; "
    "it is not evidence of unbounded or model-weight RSI."
)


class DemoError(RuntimeError):
    """The requested demo run cannot be created or inspected."""


class RunIntegrityError(DemoError):
    """Persisted run evidence does not match its sealed audit record."""


class _DemoOperation:
    """Deterministic implementation behind the four real seed passports."""

    def __init__(self, delegate: Any, seed_artifact: Artifact) -> None:
        self._delegate = delegate
        self._seed_artifact = seed_artifact
        self.passport = delegate.passport
        self.operation = delegate.operation
        self.parent_count = delegate.parent_count

    def applies(self, state: Any, parents: Sequence[SearchNode]) -> Any:
        return self._delegate.applies(state, parents)

    def build_context(
        self,
        state: Any,
        parents: Sequence[SearchNode],
        experience_view: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        context = dict(self._delegate.build_context(state, parents, experience_view))
        # Only mine-split evaluator feedback may influence generation-zero
        # mutation discovery.  Held-out labels never enter an operator prompt.
        if getattr(state, "split", "") == "mine":
            feedback: list[Mapping[str, str]] = []
            for parent in parents:
                misses = parent.observation.metadata.get("misses", ())
                if not isinstance(misses, Sequence):
                    continue
                feedback.extend(
                    dict(item)
                    for item in misses
                    if isinstance(item, Mapping)
                    and isinstance(item.get("input"), str)
                    and isinstance(item.get("expected"), str)
                )
            context["mining_feedback"] = feedback
        return context

    @staticmethod
    def _artifact_from_parent(parent: Mapping[str, Any]) -> Artifact:
        raw_artifact = parent.get("artifact", {})
        if not isinstance(raw_artifact, Mapping):
            raise ValueError("demo parent is missing its artifact")
        parent_id = str(raw_artifact.get("artifact_id", ""))
        return Artifact(
            content=str(raw_artifact.get("content", "")),
            media_type=str(raw_artifact.get("media_type", "application/json")),
            parent_ids=(parent_id,) if parent_id else (),
            created_by="demo-parent-loader",
        )

    def propose(self, context: Mapping[str, Any], model: Any, budget: Budget) -> Artifact:
        del model
        if budget.max_attempts < 1 or budget.max_exec_seconds <= 0:
            raise ValueError("demo operation budget is exhausted")
        operation = str(context.get("operation", ""))
        parents = context.get("parents", ())
        if not isinstance(parents, Sequence):
            raise ValueError("demo operator context has invalid parents")

        if operation == "draft":
            # This intentionally malformed branch gives Debug a real failure
            # to repair and keeps the negative branch in the evidence DAG.
            return Artifact(
                content="{malformed-policy",
                media_type="application/json",
                created_by=self.passport.operator_id,
                metadata={
                    "method_family": "experimental-draft",
                    "change_tags": ["malformed_draft"],
                    "plan": "probe a structurally invalid policy branch",
                },
            )

        if operation == "debug":
            parent_ids = tuple(
                str(parent.get("artifact", {}).get("artifact_id", ""))
                for parent in parents
                if isinstance(parent, Mapping) and isinstance(parent.get("artifact"), Mapping)
            )
            return Artifact(
                content=self._seed_artifact.content,
                media_type=self._seed_artifact.media_type,
                parent_ids=tuple(item for item in parent_ids if item),
                created_by=self.passport.operator_id,
                metadata={
                    "method_family": "structural-repair",
                    "change_tags": ["repair_invalid"],
                    "plan": "restore the validated policy schema",
                },
            )

        if operation not in {"improve", "crossover"} or not parents:
            raise ValueError(f"unsupported deterministic operation: {operation!r}")

        # Crossover still receives two distinct scheduler-selected parents. The
        # tiny task needs only one structurally valid policy as its substrate.
        parent_artifact: Artifact | None = None
        for raw_parent in parents:
            if not isinstance(raw_parent, Mapping):
                continue
            candidate = self._artifact_from_parent(raw_parent)
            try:
                parse_policy(candidate)
            except ValueError:
                continue
            parent_artifact = candidate
            break
        if parent_artifact is None:
            parent_artifact = self._seed_artifact

        if operation == "improve":
            normalizer = "strip_punctuation"
        else:
            feedback = context.get("mining_feedback", ())
            if not isinstance(feedback, Sequence):
                raise ValueError("crossover mining feedback must be a sequence")
            normalizer = infer_normalizer_from_feedback(
                parent_artifact,
                (item for item in feedback if isinstance(item, Mapping)),
            )
        artifact = policy_with_normalizer(
            parent_artifact,
            normalizer,
            created_by=self.passport.operator_id,
        )
        return replace(
            artifact,
            metadata={
                "method_family": "normalization",
                "change_tags": [normalizer],
                "plan": f"add the {normalizer} normalization rule",
            },
        )


class _DeterministicEvents:
    """Assign stable timestamps while retaining EventLog's hash chain."""

    def __init__(self, log: EventLog) -> None:
        self.log = log
        self.count = 0

    def append(self, event_type: str, payload: Mapping[str, Any]) -> Any:
        timestamp = (_EVENT_EPOCH + timedelta(seconds=self.count)).isoformat()
        self.count += 1
        return self.log.append(event_type, payload, timestamp=timestamp)


class _DiscardEvents:
    """Replay sink used to recompute search evidence without writing files."""

    def append(self, event_type: str, payload: Mapping[str, Any]) -> None:
        del event_type, payload


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, pretty_json(value) + "\n")


def _read_json(path: Path) -> Mapping[str, Any]:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RunIntegrityError(f"cannot read JSON evidence {path.name}: {error}") from error
    if not isinstance(value, Mapping):
        raise RunIntegrityError(f"JSON evidence must contain an object: {path.name}")
    return value


def _json_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise RunIntegrityError(f"{label} must be a JSON boolean")
    return value


def _json_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise RunIntegrityError(f"{label} must be a JSON integer")
    return value


def _json_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunIntegrityError(f"{label} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise RunIntegrityError(f"{label} must be finite")
    return result


def _json_string(value: Any, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        suffix = " non-empty" if nonempty else ""
        raise RunIntegrityError(f"{label} must be a{suffix} JSON string")
    return value


def _budget_from_document(value: Any, label: str = "budget") -> Budget:
    if not isinstance(value, Mapping) or set(value) != {
        "max_tokens",
        "max_exec_seconds",
        "max_attempts",
        "max_cost",
    }:
        raise RunIntegrityError(f"{label} must contain the four budget ceilings")
    try:
        return Budget(
            max_tokens=_json_int(value["max_tokens"], f"{label}.max_tokens"),
            max_exec_seconds=_json_number(
                value["max_exec_seconds"], f"{label}.max_exec_seconds"
            ),
            max_attempts=_json_int(value["max_attempts"], f"{label}.max_attempts"),
            max_cost=_json_number(value["max_cost"], f"{label}.max_cost"),
        )
    except ValueError as error:
        raise RunIntegrityError(f"invalid {label}: {error}") from error


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RunIntegrityError(f"{label} must be an array of strings")
    return tuple(value)


def _experience_card_from_document(value: Mapping[str, Any]) -> ExperienceCard:
    expected_fields = {
        "node_id",
        "step_id",
        "operator_id",
        "parent_node_ids",
        "generation_id",
        "score",
        "utility",
        "status",
        "error_signature",
        "runtime_seconds",
        "token_count",
        "cost",
        "method_family",
        "family_count_before",
        "delta_vs_parent",
        "novelty_score",
        "rank",
        "current_best",
        "selection_utility",
        "plan",
        "analysis",
        "method_overview",
        "parent_comparison_experience",
        "change_tags",
        "valid",
        "validity_reason",
        "guardrails",
        "artifact_id",
        "task_id",
        "split",
        "seed",
        "evaluator_id",
        "model_id",
        "budget",
        "card_id",
    }
    if set(value) != expected_fields:
        raise RunIntegrityError("experience card fields do not match the v1 evidence contract")
    guardrails = value["guardrails"]
    if not isinstance(guardrails, Mapping) or not all(
        isinstance(name, str) for name in guardrails
    ):
        raise RunIntegrityError("experience card guardrails must be an object")
    try:
        card = ExperienceCard(
            node_id=_json_string(value["node_id"], "card.node_id", nonempty=True),
            step_id=_json_int(value["step_id"], "card.step_id"),
            operator_id=_json_string(value["operator_id"], "card.operator_id", nonempty=True),
            parent_node_ids=_string_tuple(value["parent_node_ids"], "card.parent_node_ids"),
            generation_id=_json_int(value["generation_id"], "card.generation_id"),
            score=_json_number(value["score"], "card.score"),
            utility=_json_number(value["utility"], "card.utility"),
            status=ExecutionStatus(value["status"]),
            error_signature=_json_string(value["error_signature"], "card.error_signature"),
            runtime_seconds=_json_number(value["runtime_seconds"], "card.runtime_seconds"),
            token_count=_json_int(value["token_count"], "card.token_count"),
            cost=_json_number(value["cost"], "card.cost"),
            method_family=_json_string(value["method_family"], "card.method_family", nonempty=True),
            family_count_before=_json_int(
                value["family_count_before"], "card.family_count_before"
            ),
            delta_vs_parent=_json_number(value["delta_vs_parent"], "card.delta_vs_parent"),
            novelty_score=_json_number(value["novelty_score"], "card.novelty_score"),
            rank=_json_int(value["rank"], "card.rank"),
            current_best=_json_bool(value["current_best"], "card.current_best"),
            selection_utility=_json_number(
                value["selection_utility"], "card.selection_utility"
            ),
            plan=_json_string(value["plan"], "card.plan"),
            analysis=_json_string(value["analysis"], "card.analysis"),
            method_overview=_json_string(value["method_overview"], "card.method_overview"),
            parent_comparison_experience=_json_string(
                value["parent_comparison_experience"], "card.parent_comparison_experience"
            ),
            change_tags=_string_tuple(value["change_tags"], "card.change_tags"),
            valid=_json_bool(value["valid"], "card.valid"),
            validity_reason=_json_string(value["validity_reason"], "card.validity_reason"),
            guardrails={
                name: _json_number(metric, f"card.guardrails.{name}")
                for name, metric in guardrails.items()
            },
            artifact_id=_json_string(value["artifact_id"], "card.artifact_id", nonempty=True),
            task_id=_json_string(value["task_id"], "card.task_id", nonempty=True),
            split=_json_string(value["split"], "card.split", nonempty=True),
            seed=_json_int(value["seed"], "card.seed"),
            evaluator_id=_json_string(
                value["evaluator_id"], "card.evaluator_id", nonempty=True
            ),
            model_id=_json_string(value["model_id"], "card.model_id", nonempty=True),
            budget=_budget_from_document(value["budget"], "card.budget"),
            card_id=_json_string(value["card_id"], "card.card_id", nonempty=True),
        )
    except (TypeError, ValueError) as error:
        raise RunIntegrityError(f"invalid experience card evidence: {error}") from error
    if not _SHA256.fullmatch(card.node_id) or not _SHA256.fullmatch(card.artifact_id):
        raise RunIntegrityError("experience card contains a malformed content identifier")
    return card


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(128 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RunIntegrityError(f"cannot hash evidence file {path}: {error}") from error
    return digest.hexdigest()


def _build_artifact_index(
    artifacts: Sequence[Artifact],
    *,
    content_store: ContentAddressedStore | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the canonical occurrence-level provenance index.

    Content identity alone is insufficient: the same bytes can be produced by
    multiple operators and parents.  Every occurrence therefore remains an
    ordered, replayable receipt.
    """

    artifact_index: dict[str, dict[str, Any]] = {}
    for occurrence, artifact in enumerate(artifacts):
        digest = (
            content_store.put_text(artifact.content)
            if content_store is not None
            else hashlib.sha256(artifact.content.encode("utf-8")).hexdigest()
        )
        entry = artifact_index.setdefault(
            artifact.artifact_id,
            {
                "store_digest": digest,
                "media_type": artifact.media_type,
                "occurrences": [],
            },
        )
        if entry["store_digest"] != digest or entry["media_type"] != artifact.media_type:
            raise DemoError("artifact content address collision")
        entry["occurrences"].append(
            {
                "occurrence": occurrence,
                "created_by": artifact.created_by,
                "parent_ids": list(artifact.parent_ids),
                "metadata": to_primitive(artifact.metadata),
            }
        )
    return artifact_index


def _safe_evidence_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise RunIntegrityError(f"unsafe evidence path: {relative!r}")
    unresolved = root / candidate
    cursor = unresolved
    while cursor != root:
        if cursor.is_symlink():
            raise RunIntegrityError(f"evidence path contains a symbolic link: {relative!r}")
        cursor = cursor.parent
    resolved = unresolved.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError:
        raise RunIntegrityError(f"evidence path escapes run directory: {relative!r}") from None
    return resolved


def _promotion_policy() -> PromotionPolicy:
    baseline = next(
        operator.passport for operator in seed_operators() if operator.passport.name == "improve"
    )
    return PromotionPolicy(
        minimum_lcb_lift=0.05,
        valid_rate_tolerance=0.0,
        minimum_cost_adjusted_lift=0.05,
        maximum_regression=0.0,
        minimum_transfer_coverage=4,
        require_matched_budget=True,
        baseline_id=baseline.operator_id,
        baseline_content_hash=baseline.content_hash,
        required_trials=tuple(
            (f"incident-routing-promotion-{seed}", seed) for seed in _DEMO_SEEDS
        ),
        required_split="promotion",
        required_evaluator_id="incident-routing-lite",
        bootstrap_samples=2_000,
        bootstrap_seed=7,
        bootstrap_confidence=0.95,
        cost_penalty=1.0,
    )


def _select_transfer_pattern(patterns: Sequence[Any]) -> Any:
    candidates = tuple(
        pattern
        for pattern in patterns
        if pattern.kind is PatternKind.HIGH_YIELD
        and any(operator.startswith("crossover@") for operator in pattern.source_operators)
        and "normalization" in pattern.method_families
    )
    if not candidates:
        raise DemoError("trace miner did not discover a transferable normalization pattern")
    return max(
        candidates,
        key=lambda pattern: (pattern.mean_delta, pattern.support, pattern.pattern_id),
    )


def _run_manifest_from_document(value: Mapping[str, Any]) -> RunManifest:
    expected = {
        "run_id",
        "mode",
        "task_pack",
        "seeds",
        "budget",
        "model_id",
        "operator_ids",
        "protected_scopes",
        "schema_version",
    }
    if set(value) != expected:
        raise RunIntegrityError("run evidence fields do not match the v1 manifest contract")
    raw_seeds = value["seeds"]
    if not isinstance(raw_seeds, list):
        raise RunIntegrityError("run seeds must be an array")
    try:
        manifest = RunManifest(
            run_id=_json_string(value["run_id"], "run.run_id", nonempty=True),
            mode=_json_string(value["mode"], "run.mode", nonempty=True),
            task_pack=_json_string(value["task_pack"], "run.task_pack", nonempty=True),
            seeds=tuple(_json_int(seed, "run.seed") for seed in raw_seeds),
            budget=_budget_from_document(value["budget"], "run.budget"),
            model_id=_json_string(value["model_id"], "run.model_id", nonempty=True),
            operator_ids=_string_tuple(value["operator_ids"], "run.operator_ids"),
            protected_scopes=tuple(
                EffectScope(item)
                for item in _string_tuple(value["protected_scopes"], "run.protected_scopes")
            ),
            schema_version=_json_string(
                value["schema_version"], "run.schema_version", nonempty=True
            ),
        )
    except (TypeError, ValueError) as error:
        raise RunIntegrityError(f"invalid run manifest evidence: {error}") from error
    if to_primitive(manifest) != dict(value):
        raise RunIntegrityError("run manifest does not round-trip canonically")
    return manifest


def _cards_from_evidence(value: Mapping[str, Any]) -> tuple[ExperienceCard, ...]:
    if value.get("schema_version") != RUN_SCHEMA or set(value) != {"schema_version", "cards"}:
        raise RunIntegrityError("cards evidence has an unsupported envelope")
    raw_cards = value.get("cards")
    if not isinstance(raw_cards, list):
        raise RunIntegrityError("cards evidence must contain an array")
    cards = tuple(
        _experience_card_from_document(item)
        for item in raw_cards
        if isinstance(item, Mapping)
    )
    if len(cards) != len(raw_cards):
        raise RunIntegrityError("every card must be an object")
    return cards


def _lab_report_from_document(value: Mapping[str, Any]) -> LabReport:
    raw_outcomes = value.get("outcomes")
    if not isinstance(raw_outcomes, list):
        raise RunIntegrityError("lab report outcomes must be an array")
    try:
        def budget_from(raw: Any) -> Budget | None:
            if raw is None:
                return None
            return _budget_from_document(raw, "trial budget")

        outcomes = tuple(
            PairedOutcome(
                task_id=_json_string(item["task_id"], "outcome.task_id", nonempty=True),
                seed=_json_int(item["seed"], "outcome.seed"),
                candidate_score=_json_number(
                    item["candidate_score"], "outcome.candidate_score"
                ),
                baseline_score=_json_number(
                    item["baseline_score"], "outcome.baseline_score"
                ),
                candidate_valid=_json_bool(
                    item["candidate_valid"], "outcome.candidate_valid"
                ),
                baseline_valid=_json_bool(
                    item["baseline_valid"], "outcome.baseline_valid"
                ),
                candidate_cost=_json_number(item["candidate_cost"], "outcome.candidate_cost"),
                baseline_cost=_json_number(item["baseline_cost"], "outcome.baseline_cost"),
                regression=_json_number(item.get("regression", 0.0), "outcome.regression"),
                direction=item.get("direction", "higher_is_better"),
                split=_json_string(item.get("split", ""), "outcome.split"),
                evaluator_id=_json_string(
                    item.get("evaluator_id", ""), "outcome.evaluator_id"
                ),
                candidate_parent_id=_json_string(
                    item.get("candidate_parent_id", ""), "outcome.candidate_parent_id"
                ),
                baseline_parent_id=_json_string(
                    item.get("baseline_parent_id", ""), "outcome.baseline_parent_id"
                ),
                candidate_model_id=_json_string(
                    item.get("candidate_model_id", ""), "outcome.candidate_model_id"
                ),
                baseline_model_id=_json_string(
                    item.get("baseline_model_id", ""), "outcome.baseline_model_id"
                ),
                candidate_sampling=item.get("candidate_sampling"),
                baseline_sampling=item.get("baseline_sampling"),
                candidate_budget=budget_from(item.get("candidate_budget")),
                baseline_budget=budget_from(item.get("baseline_budget")),
            )
            for item in raw_outcomes
            if isinstance(item, Mapping)
        )
        if len(outcomes) != len(raw_outcomes):
            raise ValueError("every lab outcome must be an object")
        matched_budget = value["matched_budget"]
        matched_conditions = value.get("matched_conditions", False)
        if not isinstance(matched_budget, bool) or not isinstance(matched_conditions, bool):
            raise ValueError("lab matching fields must be booleans")
        return LabReport(
            operator_id=_json_string(value["operator_id"], "report.operator_id", nonempty=True),
            baseline_id=_json_string(value["baseline_id"], "report.baseline_id", nonempty=True),
            outcomes=outcomes,
            mean_lift=_json_number(value["mean_lift"], "report.mean_lift"),
            lcb95=_json_number(value["lcb95"], "report.lcb95"),
            ucb95=_json_number(value["ucb95"], "report.ucb95"),
            valid_rate=_json_number(value["valid_rate"], "report.valid_rate"),
            baseline_valid_rate=_json_number(
                value["baseline_valid_rate"], "report.baseline_valid_rate"
            ),
            cost_adjusted_lift=_json_number(
                value["cost_adjusted_lift"], "report.cost_adjusted_lift"
            ),
            transfer_coverage=_json_int(
                value["transfer_coverage"], "report.transfer_coverage"
            ),
            worst_regression=_json_number(
                value["worst_regression"], "report.worst_regression"
            ),
            matched_budget=matched_budget,
            matched_conditions=matched_conditions,
            bootstrap_samples=_json_int(
                value.get("bootstrap_samples", 10_000), "report.bootstrap_samples"
            ),
            bootstrap_seed=_json_int(
                value.get("bootstrap_seed", 0), "report.bootstrap_seed"
            ),
            bootstrap_confidence=_json_number(
                value.get("bootstrap_confidence", 0.95), "report.bootstrap_confidence"
            ),
            cost_penalty=_json_number(value.get("cost_penalty", 1.0), "report.cost_penalty"),
            report_id=_json_string(value["report_id"], "report.report_id", nonempty=True),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RunIntegrityError(f"invalid lab report evidence: {error}") from error


def _search_mining_split(
    adapter: IncidentRoutingAdapter,
    model: DeterministicModel,
    seeds: tuple[int, ...],
    event_writer: _DeterministicEvents,
) -> tuple[tuple[ExperienceCard, ...], tuple[Mapping[str, Any], ...], tuple[Artifact, ...]]:
    cards: list[ExperienceCard] = []
    boards: list[Mapping[str, Any]] = []
    artifacts: list[Artifact] = []
    global_step = 0
    for seed in seeds:
        state = adapter.prepare(seed, "mine")
        seed_artifact = adapter.seed_artifact(state)
        operators = tuple(_DemoOperation(operator, seed_artifact) for operator in seed_operators())
        engine = SearchEngine(
            adapter,
            model,
            operators,
            Budget(max_tokens=2048, max_exec_seconds=1.0, max_attempts=5),
            seed=seed,
        )
        result = engine.run(
            state,
            ("draft", "improve", "debug", "crossover"),
            include_seed=True,
            seed_artifact=seed_artifact,
        )
        if len(result.nodes) != 5 or result.best_node is None or result.best_node.score.primary < 1.0:
            raise DemoError("the deterministic mining search did not reach its expected evidence state")
        for card in result.cards:
            cards.append(replace(card, step_id=global_step, card_id=""))
            global_step += 1
        artifacts.extend(node.artifact for node in result.nodes)
        boards.append({"seed": seed, "board": to_primitive(result.board), "usage": to_primitive(result.usage)})
        event_writer.append(
            "search_task_completed",
            {
                "task_id": state.task_id,
                "seed": seed,
                "node_ids": [node.node_id for node in result.nodes],
                "best_node_id": result.best_node.node_id,
                "best_score": result.best_node.score.primary,
                "operations": ["draft", "improve", "debug", "crossover"],
            },
        )
    return tuple(cards), tuple(boards), tuple(artifacts)


def _generation_zero_improve() -> ImproveOperator:
    """Return the canonical generation-zero Improve runtime.

    Promotion and sealed controls must execute the implementation identified
    by the canonical passport. A demo-specific wrapper sharing that passport
    would make the content hash claim a different generator than the one that
    actually ran.
    """

    return ImproveOperator()


def _model_generate_calls(model: DeterministicModel) -> int:
    calls = getattr(model, "generate_calls", None)
    if type(calls) is not int or calls < 0:
        raise DemoError("the deterministic model backend must expose a non-negative generate_calls counter")
    return calls


def _evaluated_seed_parent(
    *,
    adapter: IncidentRoutingAdapter,
    model: DeterministicModel,
    state: Any,
    budget: Budget,
) -> SearchNode:
    artifact = adapter.seed_artifact(state)
    observation = adapter.execute(state, artifact, budget)
    score = adapter.score(state, observation)
    node_id = hashlib.sha256(
        pretty_json(
            {
                "task": state,
                "artifact_id": artifact.artifact_id,
                "operator_id": "seed",
                "observation": observation,
                "score": score,
                "budget": budget,
                "evaluator_id": adapter.name,
                "model_id": model.model_id,
            }
        ).encode()
    ).hexdigest()
    return SearchNode(
        node_id=node_id,
        artifact=artifact,
        operator_id="seed",
        parent_ids=(),
        generation=0,
        observation=observation,
        score=score,
        method_family="phrase-rules",
        plan="matched evaluated seed parent",
        task_id=state.task_id,
        split=state.split,
        seed=state.seed,
        evaluator_id=adapter.name,
        model_id=model.model_id,
        budget=budget,
    )


def _evaluate_operator_from_parent(
    *,
    adapter: IncidentRoutingAdapter,
    model: DeterministicModel,
    operator: Any,
    state: Any,
    parent: SearchNode,
    budget: Budget,
) -> SearchNode:
    applicability = operator.applies(state, (parent,))
    if not applicability.applies:
        raise DemoError(
            f"{operator.passport.operator_id} rejected the matched parent: "
            f"{applicability.reasons}"
        )
    context = operator.build_context(state, (parent,), {})
    artifact = operator.propose(context, model, budget)
    observation = adapter.execute(state, artifact, budget)
    score = adapter.score(state, observation)
    node_id = hashlib.sha256(
        pretty_json(
            {
                "task": state,
                "artifact_id": artifact.artifact_id,
                "operator_id": operator.passport.operator_id,
                "operator_content_hash": operator.passport.content_hash,
                "parent_ids": (parent.node_id,),
                "observation": observation,
                "score": score,
                "budget": budget,
                "evaluator_id": adapter.name,
                "model_id": model.model_id,
            }
        ).encode()
    ).hexdigest()
    return SearchNode(
        node_id=node_id,
        artifact=artifact,
        operator_id=operator.passport.operator_id,
        parent_ids=(parent.node_id,),
        generation=1,
        observation=observation,
        score=score,
        method_family=str(artifact.metadata.get("method_family", "normalization")),
        plan=str(artifact.metadata.get("plan", operator.passport.intent)),
        task_id=state.task_id,
        split=state.split,
        seed=state.seed,
        evaluator_id=adapter.name,
        model_id=model.model_id,
        budget=budget,
    )


@dataclass(frozen=True, slots=True)
class _PairedEvaluation:
    state: Any
    parent: SearchNode
    baseline: SearchNode
    candidate: SearchNode
    baseline_content_hash: str
    candidate_content_hash: str
    baseline_model_generate_calls: int
    candidate_model_generate_calls: int

    @property
    def artifacts(self) -> tuple[Artifact, ...]:
        return (self.parent.artifact, self.baseline.artifact, self.candidate.artifact)


def _evaluate_paired_operators(
    *,
    adapter: IncidentRoutingAdapter,
    model: DeterministicModel,
    operator: Any,
    split: str,
    seed: int,
    budget: Budget,
) -> _PairedEvaluation:
    state = adapter.prepare(seed, split)
    parent = _evaluated_seed_parent(
        adapter=adapter,
        model=model,
        state=state,
        budget=budget,
    )
    baseline_operator = _generation_zero_improve()
    baseline_calls_before = _model_generate_calls(model)
    baseline = _evaluate_operator_from_parent(
        adapter=adapter,
        model=model,
        operator=baseline_operator,
        state=state,
        parent=parent,
        budget=budget,
    )
    baseline_model_generate_calls = _model_generate_calls(model) - baseline_calls_before
    candidate_calls_before = _model_generate_calls(model)
    candidate = _evaluate_operator_from_parent(
        adapter=adapter,
        model=model,
        operator=operator,
        state=state,
        parent=parent,
        budget=budget,
    )
    candidate_model_generate_calls = _model_generate_calls(model) - candidate_calls_before
    if (
        baseline.parent_ids != candidate.parent_ids
        or baseline.model_id != candidate.model_id
        or baseline.evaluator_id != candidate.evaluator_id
        or baseline.budget != candidate.budget
    ):
        raise DemoError("paired baseline and candidate execution conditions diverged")
    if baseline.artifact.created_by != baseline_operator.passport.operator_id:
        raise DemoError("the promotion baseline was not produced by the Improve operator")
    if baseline_model_generate_calls != 1 or candidate_model_generate_calls != 1:
        raise DemoError("paired baseline and candidate must each execute exactly one model generation")
    return _PairedEvaluation(
        state=state,
        parent=parent,
        baseline=baseline,
        candidate=candidate,
        baseline_content_hash=baseline_operator.passport.content_hash,
        candidate_content_hash=operator.passport.content_hash,
        baseline_model_generate_calls=baseline_model_generate_calls,
        candidate_model_generate_calls=candidate_model_generate_calls,
    )


def _paired_execution_receipt(pair: _PairedEvaluation) -> Mapping[str, Any]:
    return {
        "task_id": pair.state.task_id,
        "seed": pair.state.seed,
        "split": pair.state.split,
        "evaluator_id": pair.baseline.evaluator_id,
        "model_id": pair.baseline.model_id,
        "sampling": {},
        "budget": to_primitive(pair.baseline.budget),
        "model_generate_calls": {
            "baseline": pair.baseline_model_generate_calls,
            "candidate": pair.candidate_model_generate_calls,
        },
        "parent": {
            "node_id": pair.parent.node_id,
            "artifact_id": pair.parent.artifact.artifact_id,
            "operator_id": pair.parent.operator_id,
            "score": pair.parent.score.primary,
            "valid": pair.parent.score.valid,
        },
        "baseline": {
            "node_id": pair.baseline.node_id,
            "operator_id": pair.baseline.operator_id,
            "operator_content_hash": pair.baseline_content_hash,
            "parent_node_ids": list(pair.baseline.parent_ids),
            "artifact_id": pair.baseline.artifact.artifact_id,
            "artifact_parent_ids": list(pair.baseline.artifact.parent_ids),
            "artifact_created_by": pair.baseline.artifact.created_by,
            "score": pair.baseline.score.primary,
            "valid": pair.baseline.score.valid,
            "cost": pair.baseline.observation.cost,
        },
        "candidate": {
            "node_id": pair.candidate.node_id,
            "operator_id": pair.candidate.operator_id,
            "operator_content_hash": pair.candidate_content_hash,
            "parent_node_ids": list(pair.candidate.parent_ids),
            "artifact_id": pair.candidate.artifact.artifact_id,
            "artifact_parent_ids": list(pair.candidate.artifact.parent_ids),
            "artifact_created_by": pair.candidate.artifact.created_by,
            "score": pair.candidate.score.primary,
            "valid": pair.candidate.score.valid,
            "cost": pair.candidate.observation.cost,
        },
    }


def _validation_evidence(
    adapter: IncidentRoutingAdapter,
    model: DeterministicModel,
    operator: Any,
    seeds: tuple[int, ...],
    budget: Budget,
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Artifact, ...]]:
    outcomes: list[Mapping[str, Any]] = []
    artifacts: list[Artifact] = []
    for seed in seeds:
        pair = _evaluate_paired_operators(
            adapter=adapter,
            model=model,
            operator=operator,
            split="validation",
            seed=seed,
            budget=budget,
        )
        outcomes.append(
            {
                "task_id": pair.state.task_id,
                "seed": seed,
                "baseline_score": pair.baseline.score.primary,
                "candidate_score": pair.candidate.score.primary,
                "baseline_valid": pair.baseline.score.valid,
                "candidate_valid": pair.candidate.score.valid,
                "split": pair.state.split,
                "evaluator_id": adapter.name,
                "baseline_operator_id": pair.baseline.operator_id,
                "baseline_content_hash": pair.baseline_content_hash,
                "baseline_artifact_id": pair.baseline.artifact.artifact_id,
                "candidate_operator_id": pair.candidate.operator_id,
                "candidate_content_hash": pair.candidate_content_hash,
                "candidate_artifact_id": pair.candidate.artifact.artifact_id,
                "parent_node_id": pair.parent.node_id,
                "model_id": model.model_id,
                "sampling": {},
                "budget": to_primitive(budget),
                "operations": 1,
                "model_generate_calls": {
                    "baseline": pair.baseline_model_generate_calls,
                    "candidate": pair.candidate_model_generate_calls,
                },
            }
        )
        artifacts.extend(pair.artifacts)
    return tuple(outcomes), tuple(artifacts)


def _promotion_evidence(
    adapter: IncidentRoutingAdapter,
    model: DeterministicModel,
    operator: Any,
    seeds: tuple[int, ...],
    budget: Budget,
) -> tuple[tuple[PairedOutcome, ...], tuple[Mapping[str, Any], ...], tuple[Artifact, ...]]:
    outcomes: list[PairedOutcome] = []
    receipts: list[Mapping[str, Any]] = []
    artifacts: list[Artifact] = []
    for seed in seeds:
        pair = _evaluate_paired_operators(
            adapter=adapter,
            model=model,
            operator=operator,
            split="promotion",
            seed=seed,
            budget=budget,
        )
        outcomes.append(
            PairedOutcome(
                task_id=pair.state.task_id,
                seed=seed,
                candidate_score=pair.candidate.score.primary,
                baseline_score=pair.baseline.score.primary,
                candidate_valid=pair.candidate.score.valid,
                baseline_valid=pair.baseline.score.valid,
                candidate_cost=pair.candidate.observation.cost,
                baseline_cost=pair.baseline.observation.cost,
                regression=max(
                    0.0,
                    pair.baseline.score.primary - pair.candidate.score.primary,
                ),
                direction=pair.candidate.score.direction,
                split=pair.state.split,
                evaluator_id=adapter.name,
                candidate_parent_id=pair.parent.node_id,
                baseline_parent_id=pair.parent.node_id,
                candidate_model_id=model.model_id,
                baseline_model_id=model.model_id,
                candidate_sampling={},
                baseline_sampling={},
                candidate_budget=budget,
                baseline_budget=budget,
            )
        )
        receipts.append(_paired_execution_receipt(pair))
        artifacts.extend(pair.artifacts)
    return tuple(outcomes), tuple(receipts), tuple(artifacts)


def _trajectory_document(
    result: Any,
    *,
    operation_plan: Sequence[tuple[str, str]],
    run_budget: Budget,
    model_generate_calls: int,
) -> Mapping[str, Any]:
    cumulative_tokens = 0
    cumulative_seconds = 0.0
    cumulative_cost = 0.0
    best_utility = float("-inf")
    trajectory: list[Mapping[str, Any]] = []
    for step, node in enumerate(result.nodes):
        cumulative_tokens += node.observation.token_usage
        cumulative_seconds += node.observation.runtime_seconds
        cumulative_cost += node.observation.cost
        best_utility = max(best_utility, node.score.utility)
        operator_content_hash = "" if step == 0 else operation_plan[step - 1][1]
        trajectory.append(
            {
                "step": step,
                "plan_index": None if step == 0 else step - 1,
                "node_id": node.node_id,
                "operator_id": node.operator_id,
                "operator_content_hash": operator_content_hash,
                "parent_node_ids": list(node.parent_ids),
                "artifact_id": node.artifact.artifact_id,
                "artifact_parent_ids": list(node.artifact.parent_ids),
                "artifact_created_by": node.artifact.created_by,
                "generation": node.generation,
                "score": node.score.primary,
                "utility": node.score.utility,
                "best_utility": best_utility,
                "valid": node.score.valid,
                "status": node.observation.status.value,
                "budget": to_primitive(node.budget),
                "observation_usage": {
                    "tokens": node.observation.token_usage,
                    "exec_seconds": node.observation.runtime_seconds,
                    "cost": node.observation.cost,
                },
                "cumulative_usage": {
                    "tokens": cumulative_tokens,
                    "exec_seconds": cumulative_seconds,
                    "attempts": step + 1,
                    "cost": cumulative_cost,
                },
            }
        )
    if len(trajectory) != 1 + len(operation_plan):
        raise DemoError("sealed search did not execute its fixed operation plan")
    cumulative_usage = trajectory[-1]["cumulative_usage"]
    if cumulative_usage != to_primitive(result.usage):
        raise DemoError("sealed trajectory usage does not match SearchEngine accounting")
    operation_best = [float(item["best_utility"]) for item in trajectory[1:]]
    return {
        "operation_plan": [
            {
                "operator_id": operator_id,
                "operator_content_hash": content_hash,
                "operation_family": "improve",
            }
            for operator_id, content_hash in operation_plan
        ],
        "run_budget": to_primitive(run_budget),
        "trajectory": trajectory,
        "usage": to_primitive(result.usage),
        "model_usage": {"generate_calls": model_generate_calls},
        "final_best_utility": trajectory[-1]["best_utility"],
        "best_utility_auc_per_attempt": fmean(operation_best),
    }


def _sealed_trajectory_evidence(
    adapter: IncidentRoutingAdapter,
    model: DeterministicModel,
    operator: Any,
    seeds: tuple[int, ...],
    run_budget: Budget,
    *,
    persistent_installation: bool,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any], tuple[Artifact, ...]]:
    outcomes: list[Mapping[str, Any]] = []
    artifacts: list[Artifact] = []
    baseline_auc: dict[str, float] = {}
    candidate_auc: dict[str, float] = {}
    for seed in seeds:
        state = adapter.prepare(seed, "sealed")
        seed_artifact = adapter.seed_artifact(state)
        baseline_operator = _generation_zero_improve()
        candidate_operator = bind_operation_family(operator, "improve")
        baseline_plan = (baseline_operator,) * 4
        candidate_plan = (candidate_operator, *([baseline_operator] * 3))
        baseline_calls_before = _model_generate_calls(model)
        baseline_result = SearchEngine(
            adapter,
            model,
            (baseline_operator,),
            run_budget,
            seed=seed,
        ).run(
            state,
            baseline_plan,
            include_seed=True,
            seed_artifact=seed_artifact,
        )
        baseline_model_generate_calls = _model_generate_calls(model) - baseline_calls_before
        candidate_calls_before = _model_generate_calls(model)
        candidate_result = SearchEngine(
            adapter,
            model,
            (baseline_operator, candidate_operator),
            run_budget,
            seed=seed,
        ).run(
            state,
            candidate_plan,
            include_seed=True,
            seed_artifact=seed_artifact,
        )
        candidate_model_generate_calls = _model_generate_calls(model) - candidate_calls_before
        if (
            baseline_result.nodes[0].node_id != candidate_result.nodes[0].node_id
            or baseline_result.nodes[0].artifact.artifact_id
            != candidate_result.nodes[0].artifact.artifact_id
            or baseline_result.usage != candidate_result.usage
            or any(
                left.budget != right.budget
                for left, right in zip(
                    baseline_result.nodes,
                    candidate_result.nodes,
                    strict=True,
                )
            )
        ):
            raise DemoError("sealed G0/G1 search conditions are not exactly matched")
        if (
            baseline_model_generate_calls != len(baseline_plan)
            or candidate_model_generate_calls != len(candidate_plan)
            or baseline_model_generate_calls != candidate_model_generate_calls
        ):
            raise DemoError("sealed G0/G1 model-generation usage is not exactly matched")
        baseline_hash = baseline_operator.passport.content_hash
        candidate_hash = operator.passport.content_hash
        generation_0 = _trajectory_document(
            baseline_result,
            operation_plan=((baseline_operator.passport.operator_id, baseline_hash),) * 4,
            run_budget=run_budget,
            model_generate_calls=baseline_model_generate_calls,
        )
        generation_1 = _trajectory_document(
            candidate_result,
            operation_plan=(
                (operator.passport.operator_id, candidate_hash),
                *((baseline_operator.passport.operator_id, baseline_hash),) * 3,
            ),
            run_budget=run_budget,
            model_generate_calls=candidate_model_generate_calls,
        )
        if generation_0["operation_plan"][1:] != generation_1["operation_plan"][1:]:
            raise DemoError("sealed controlled plans differ outside replacement index zero")
        baseline_score = float(generation_0["final_best_utility"])
        candidate_score = float(generation_1["final_best_utility"])
        baseline_auc[str(seed)] = float(generation_0["best_utility_auc_per_attempt"])
        candidate_auc[str(seed)] = float(generation_1["best_utility_auc_per_attempt"])
        outcomes.append(
            {
                "task_id": state.task_id,
                "seed": seed,
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "lift": candidate_score - baseline_score,
                "baseline_valid": bool(
                    baseline_result.best_node and baseline_result.best_node.score.valid
                ),
                "candidate_valid": bool(
                    candidate_result.best_node and candidate_result.best_node.score.valid
                ),
                "split": state.split,
                "evaluator_id": adapter.name,
                "model_id": model.model_id,
                "sampling": {},
                "operator_id": operator.passport.operator_id,
                "operator_content_hash": candidate_hash,
                "budget": to_primitive(run_budget),
                "matched_conditions": {
                    "initial_node_id": baseline_result.nodes[0].node_id,
                    "initial_artifact_id": seed_artifact.artifact_id,
                    "planned_operations": 4,
                    "actual_attempts": baseline_result.usage.attempts,
                    "model_generate_calls_per_trajectory": baseline_model_generate_calls,
                    "only_changed_plan_index": 0,
                },
                "controlled_change": {
                    "generation_0_operator_id": baseline_operator.passport.operator_id,
                    "generation_0_content_hash": baseline_hash,
                    "generation_1_operator_id": operator.passport.operator_id,
                    "generation_1_content_hash": candidate_hash,
                    "operation_family": "improve",
                },
                "generation_0": generation_0,
                "generation_1": generation_1,
            }
        )
        artifacts.extend(node.artifact for node in baseline_result.nodes)
        artifacts.extend(node.artifact for node in candidate_result.nodes)
    generation_0_auc = fmean(baseline_auc.values())
    generation_1_auc = fmean(candidate_auc.values())
    rsi_evidence = {
        "metric": "cumulative_best_utility_auc_per_matched_operation",
        "planned_operations_per_trajectory": 4,
        "generation_0_mean_auc_per_attempt": generation_0_auc,
        "generation_1_mean_auc_per_attempt": generation_1_auc,
        "improvement_acceleration": generation_1_auc - generation_0_auc,
        "generation_0_auc_by_seed": baseline_auc,
        "generation_1_auc_by_seed": candidate_auc,
        "matched_task_seed_budget_trajectories": True,
        "matched_model_generate_calls": True,
        "controlled_operator_replacement_index": 0,
        "persistent_operator_installation": persistent_installation,
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    return tuple(outcomes), rsi_evidence, tuple(artifacts)


def run_offline_rsi_demo(output: str | Path) -> Mapping[str, Any]:
    """Run the complete offline RSI loop and return a compact summary."""

    root = Path(output).expanduser().resolve(strict=False)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise DemoError(f"output directory already exists and is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    event_log = EventLog(root / "events.jsonl")
    events = _DeterministicEvents(event_log)
    content_store = ContentAddressedStore(root / "artifacts")
    adapter = IncidentRoutingAdapter()
    model = DeterministicModel()
    seeds = _DEMO_SEEDS
    lab_budget = _DEMO_LAB_BUDGET
    split_fingerprints = {
        split: tuple(
            sorted(
                {
                    digest
                    for seed in seeds
                    for digest in adapter.case_fingerprints(seed, split)
                }
            )
        )
        for split in ("mine", "validation", "promotion", "sealed")
    }
    for index, left in enumerate(split_fingerprints):
        for right in tuple(split_fingerprints)[index + 1 :]:
            if set(split_fingerprints[left]) & set(split_fingerprints[right]):
                raise DemoError(f"task split inputs overlap: {left} and {right}")
    events.append(
        "run_started",
        {
            "run_id": RUN_ID,
            "mode": "rsi",
            "task_pack": adapter.name,
            "split_fingerprint_counts": {
                split: len(digests) for split, digests in split_fingerprints.items()
            },
            "protected_scopes": [scope.value for scope in _DEMO_PROTECTED_SCOPES],
        },
    )

    cards, boards, search_artifacts = _search_mining_split(adapter, model, seeds, events)
    miner = TraceMiner(minimum_support=4, minimum_delta=0.05)
    patterns = miner.mine(cards)
    selected_pattern = _select_transfer_pattern(patterns)
    events.append(
        "trace_pattern_mined",
        {
            "pattern_id": selected_pattern.pattern_id,
            "change_tags": selected_pattern.change_tags,
            "support": selected_pattern.support,
            "mean_delta": selected_pattern.mean_delta,
            "evidence_ids": selected_pattern.evidence_ids,
        },
    )

    synthesizer = OperatorSynthesizer(
        baseline_id="improve@1.0.0",
        budget=lab_budget,
        minimum_tasks=4,
    )
    draft = synthesizer.synthesize(selected_pattern, name="normalize-hyphenated-incidents")

    compiled = OperatorCompiler().compile(draft)
    proposed_passport = sign_passport(compiled.passport, _DEMO_SIGNER, _DEMO_SECRET)
    compiled.passport = proposed_passport
    trust_kernel = TrustKernel(
        {_DEMO_SIGNER: _DEMO_SECRET},
        release_signers={_DEMO_SIGNER},
        release_principals={_DEMO_ACTOR},
    )
    trust_kernel.authorize_execution(
        proposed_passport,
        (EffectScope.TASK_ARTIFACT,),
        expected_hash=proposed_passport.content_hash,
    )
    events.append(
        "operator_compiled",
        {
            "operator_id": proposed_passport.operator_id,
            "content_hash": proposed_passport.content_hash,
            "pattern_id": selected_pattern.pattern_id,
            "evidence_ids": proposed_passport.evidence_ids,
            "authorized_effects": [EffectScope.TASK_ARTIFACT.value],
        },
    )

    # Persist and immediately rehydrate the generated prompt artifact. Every
    # held-out stage below uses the registry-loaded executable, not the
    # in-memory compiler object.
    registry = LocalRegistry(root / "registry")
    for seed_operator in seed_operators():
        registry.register(seed_operator.passport)
    registry.register_executable(compiled)
    candidate_operator = registry.load_operator(proposed_passport.operator_id)

    validation_outcomes, validation_artifacts = _validation_evidence(
        adapter,
        model,
        candidate_operator,
        seeds,
        lab_budget,
    )
    if any(
        not bool(item["candidate_valid"]) or float(item["candidate_score"]) < 1.0
        for item in validation_outcomes
    ):
        raise DemoError("candidate failed the disjoint validation split")
    events.append(
        "validation_split_completed",
        {
            "operator_id": proposed_passport.operator_id,
            "task_ids": [item["task_id"] for item in validation_outcomes],
            "candidate_mean": fmean(float(item["candidate_score"]) for item in validation_outcomes),
        },
    )

    outcomes, promotion_receipts, promotion_artifacts = _promotion_evidence(
        adapter,
        model,
        candidate_operator,
        seeds,
        lab_budget,
    )

    report = aggregate_lab_report(
        proposed_passport.operator_id,
        "improve@1.0.0",
        outcomes,
        candidate_budgets=(lab_budget,) * len(outcomes),
        baseline_budgets=(lab_budget,) * len(outcomes),
        bootstrap_samples=2_000,
        bootstrap_seed=7,
    )
    events.append(
        "matched_budget_lab_completed",
        {
            "operator_id": report.operator_id,
            "baseline_id": report.baseline_id,
            "report_id": report.report_id,
            "mean_lift": report.mean_lift,
            "lcb95": report.lcb95,
            "ucb95": report.ucb95,
            "matched_budget": report.matched_budget,
            "matched_conditions": report.matched_conditions,
            "task_ids": [outcome.task_id for outcome in outcomes],
        },
    )

    policy = _promotion_policy()
    gate = PromotionGate(policy).evaluate(report, operator=proposed_passport)
    if gate.decision.value != "promote":
        raise DemoError(f"the deterministic candidate failed its evidence gate: {gate.reasons}")
    events.append(
        "promotion_gate_decided",
        {
            "operator_id": gate.operator_id,
            "report_id": gate.report_id,
            "decision": gate.decision.value,
            "policy_hash": gate.policy_hash,
            "reasons": gate.reasons,
        },
    )

    registry.transition(proposed_passport.operator_id, OperatorLifecycle.SHADOW)
    registry.transition(proposed_passport.operator_id, OperatorLifecycle.CANDIDATE)
    registry.promote(proposed_passport.operator_id, gate, report, policy)
    stable_passport = sign_passport(
        registry.get(proposed_passport.operator_id),
        _DEMO_SIGNER,
        _DEMO_SECRET,
    )
    release_decision = trust_kernel.authorize_release(
        stable_passport,
        _DEMO_ACTOR,
        expected_hash=stable_passport.content_hash,
    )
    # Activation occurs only after the external trust kernel authorizes this
    # exact stable hash and release principal.
    installation = registry.install(
        proposed_passport.operator_id,
        signed_passport=stable_passport,
        trust_kernel=trust_kernel,
        actor=_DEMO_ACTOR,
    )
    # Re-open the registry to prove installation is durable rather than an
    # in-memory pointer to the compiler object used earlier in this process.
    installed_operator = LocalRegistry(root / "registry").load_installed_operator(
        proposed_passport.name,
        trust_kernel=trust_kernel,
        actor=_DEMO_ACTOR,
    )
    events.append(
        "operator_installed",
        {
            "operator_id": installation.operator_id,
            "content_hash": installation.content_hash,
            "lifecycle": stable_passport.lifecycle.value,
            "release_actor": release_decision.release_actor,
            "registry_root": "registry",
        },
    )

    persistent_installation = (
        installed_operator.passport.operator_id == installation.operator_id
        and installed_operator.passport.generator_ref.startswith("sha256:")
    )
    sealed_outcomes, rsi_evidence, sealed_artifacts = _sealed_trajectory_evidence(
        adapter,
        model,
        installed_operator,
        seeds,
        _DEMO_TRAJECTORY_BUDGET,
        persistent_installation=persistent_installation,
    )
    baseline_mean = fmean(float(item["baseline_score"]) for item in sealed_outcomes)
    candidate_mean = fmean(float(item["candidate_score"]) for item in sealed_outcomes)
    events.append(
        "sealed_split_evaluated",
        {
            "operator_id": stable_passport.operator_id,
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "mean_lift": candidate_mean - baseline_mean,
            "task_ids": [str(item["task_id"]) for item in sealed_outcomes],
            "rsi_evidence": rsi_evidence,
        },
    )

    all_artifacts = (
        *search_artifacts,
        *validation_artifacts,
        *promotion_artifacts,
        *sealed_artifacts,
    )
    artifact_index = _build_artifact_index(all_artifacts, content_store=content_store)

    run_manifest = RunManifest(
        run_id=RUN_ID,
        mode="rsi",
        task_pack=adapter.name,
        seeds=seeds,
        budget=lab_budget,
        model_id=model.model_id,
        operator_ids=tuple(operator.passport.operator_id for operator in seed_operators())
        + (stable_passport.operator_id,),
    )
    _write_json(root / "run.json", run_manifest)
    _write_json(root / "cards.json", {"schema_version": RUN_SCHEMA, "cards": cards})
    _write_json(root / "boards.json", {"schema_version": RUN_SCHEMA, "boards": boards})
    _write_json(
        root / "split-fingerprints.json",
        {"schema_version": RUN_SCHEMA, "splits": split_fingerprints},
    )
    _write_json(root / "patterns.json", {"schema_version": RUN_SCHEMA, "patterns": patterns})
    _write_json(
        root / "operator-passport.json",
        {
            "schema_version": "evolsa.operator-passport-evidence/v1",
            "content_hash": stable_passport.content_hash,
            "passport": operator_passport_document(stable_passport),
        },
    )
    _write_json(root / "lab-report.json", report)
    _write_json(root / "gate-result.json", gate)
    _write_json(
        root / "validation-results.json",
        {"schema_version": RUN_SCHEMA, "outcomes": validation_outcomes},
    )
    _write_json(
        root / "sealed-results.json",
        {
            "schema_version": RUN_SCHEMA,
            "outcomes": sealed_outcomes,
            "baseline_mean": baseline_mean,
            "candidate_mean": candidate_mean,
            "mean_lift": candidate_mean - baseline_mean,
            "rsi_evidence": rsi_evidence,
        },
    )
    _write_json(root / "artifact-index.json", {"schema_version": RUN_SCHEMA, "artifacts": artifact_index})
    _write_json(
        root / "promotion-trials.json",
        {
            "schema_version": RUN_SCHEMA,
            "outcomes": report.outcomes,
            "receipts": promotion_receipts,
            "budget": lab_budget,
        },
    )
    write_report(
        root / "review.md",
        render_review_markdown(run_id=RUN_ID, cards=cards, reports=(report,), gates=(gate,)),
    )
    write_report(
        root / "index.html",
        render_lineage_html(
            run_id=RUN_ID,
            cards=cards,
            passports=(*seed_operators_passports(), stable_passport),
            reports=(report,),
            gates=(gate,),
            metadata={
                "task_pack": adapter.name,
                "offline": True,
                "network_requests": 0,
                "sealed_mean_lift": candidate_mean - baseline_mean,
                "rsi_evidence": rsi_evidence,
            },
        ),
    )

    fixed_evidence = (
        "run.json",
        "cards.json",
        "boards.json",
        "split-fingerprints.json",
        "patterns.json",
        "operator-passport.json",
        "lab-report.json",
        "gate-result.json",
        "validation-results.json",
        "sealed-results.json",
        "artifact-index.json",
        "promotion-trials.json",
        "review.md",
        "index.html",
    )
    registry_evidence = tuple(
        path.relative_to(root).as_posix()
        for path in sorted((root / "registry").rglob("*.json"))
        if path.is_file() and not path.is_symlink()
    )
    evidence_files = {
        relative: _sha256_file(root / relative)
        for relative in (*fixed_evidence, *registry_evidence)
    }
    content_digests = content_store.verify_all()
    expected_event_count = len(event_log) + 1
    events.append(
        "run_sealed",
        {
            "run_id": RUN_ID,
            "evidence_files": evidence_files,
            "content_digests": content_digests,
            "event_count": expected_event_count,
            "operator_id": stable_passport.operator_id,
            "report_id": report.report_id,
            "decision": gate.decision.value,
        },
    )
    records = event_log.verify()
    if len(records) != expected_event_count:
        raise DemoError("event log length changed while sealing the run")
    index = {
        "schema_version": INDEX_SCHEMA,
        "run_id": RUN_ID,
        "evidence_files": evidence_files,
        "content_digests": content_digests,
        "event_count": len(records),
        "event_head": records[-1].event_hash,
        "operator_id": stable_passport.operator_id,
        "report_id": report.report_id,
        "decision": gate.decision.value,
    }
    _write_json(root / "manifest.json", index)

    summary = replay_run(root)
    return {**summary, "output": str(root)}


def seed_operators_passports() -> tuple[Any, ...]:
    """Return generation-zero passports without sharing mutable wrappers."""

    return tuple(operator.passport for operator in seed_operators())


def replay_run(path: str | Path) -> Mapping[str, Any]:
    """Verify all sealed evidence and return a replay receipt."""

    root = Path(path).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise RunIntegrityError(f"run directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RunIntegrityError("run manifest is missing or is a symbolic link")
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != INDEX_SCHEMA or manifest.get("run_id") != RUN_ID:
        raise RunIntegrityError("unsupported or malformed run manifest")
    raw_files = manifest.get("evidence_files")
    if not isinstance(raw_files, Mapping) or not raw_files:
        raise RunIntegrityError("run manifest has no sealed evidence files")
    if not all(
        isinstance(name, str)
        and isinstance(digest, str)
        and _SHA256.fullmatch(digest)
        for name, digest in raw_files.items()
    ):
        raise RunIntegrityError("run manifest contains malformed evidence entries")
    evidence_files = dict(raw_files)
    required_evidence = {
        "run.json",
        "cards.json",
        "boards.json",
        "split-fingerprints.json",
        "patterns.json",
        "operator-passport.json",
        "lab-report.json",
        "gate-result.json",
        "validation-results.json",
        "sealed-results.json",
        "artifact-index.json",
        "promotion-trials.json",
        "review.md",
        "index.html",
    }
    registry_root = root / "registry"
    registry_lock = registry_root / ".registry.lock"
    if (
        not registry_root.is_dir()
        or registry_root.is_symlink()
        or not registry_lock.is_file()
        or registry_lock.is_symlink()
    ):
        raise RunIntegrityError("sealed registry is missing or redirected")
    registry_entries = tuple(registry_root.rglob("*"))
    if any(path.is_symlink() for path in registry_entries):
        raise RunIntegrityError("sealed registry contains a symbolic link")
    unexpected_registry_files = tuple(
        path
        for path in registry_entries
        if path.is_file() and path != registry_lock and path.suffix != ".json"
    )
    if unexpected_registry_files:
        raise RunIntegrityError("sealed registry contains an unexpected file type")
    registry_evidence = {
        path.relative_to(root).as_posix()
        for path in registry_entries
        if path.is_file() and path.suffix == ".json"
    }
    if set(evidence_files) != required_evidence | registry_evidence:
        raise RunIntegrityError("sealed evidence file set is incomplete or contains extras")
    for relative, expected in evidence_files.items():
        evidence_path = _safe_evidence_path(root, relative)
        if not evidence_path.is_file() or _sha256_file(evidence_path) != expected:
            raise RunIntegrityError(f"evidence digest mismatch: {relative}")

    event_path = root / "events.jsonl"
    head_path = root / "events.jsonl.head"
    if not event_path.is_file() or not head_path.is_file() or event_path.is_symlink() or head_path.is_symlink():
        raise RunIntegrityError("event log or its durable head is missing")
    try:
        records = EventLog(event_path).verify()
    except Exception as error:
        raise RunIntegrityError(f"event-log verification failed: {error}") from error
    if not records or records[-1].event_type != "run_sealed":
        raise RunIntegrityError("event log has no terminal run_sealed record")
    terminal = records[-1].payload
    terminal_files = terminal.get("evidence_files")
    if not isinstance(terminal_files, Mapping) or dict(terminal_files) != evidence_files:
        raise RunIntegrityError("manifest evidence index does not match the sealed event")
    if (
        _json_int(manifest.get("event_count"), "manifest.event_count") != len(records)
        or _json_int(terminal.get("event_count"), "terminal.event_count") != len(records)
    ):
        raise RunIntegrityError("event count does not match sealed evidence")
    if manifest.get("event_head") != records[-1].event_hash:
        raise RunIntegrityError("event head does not match the manifest")
    expected_event_types = (
        "run_started",
        "search_task_completed",
        "search_task_completed",
        "search_task_completed",
        "search_task_completed",
        "trace_pattern_mined",
        "operator_compiled",
        "validation_split_completed",
        "matched_budget_lab_completed",
        "promotion_gate_decided",
        "operator_installed",
        "sealed_split_evaluated",
        "run_sealed",
    )
    if tuple(record.event_type for record in records) != expected_event_types:
        raise RunIntegrityError("event log does not match the v1 bounded-RSI lifecycle")

    artifact_root = root / "artifacts"
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise RunIntegrityError("content-addressed artifact store is missing or redirected")
    try:
        verified_digests = ContentAddressedStore(artifact_root).verify_all()
    except Exception as error:
        raise RunIntegrityError(f"content store verification failed: {error}") from error
    raw_declared_digests = manifest.get("content_digests")
    raw_terminal_digests = terminal.get("content_digests")
    if not isinstance(raw_declared_digests, (list, tuple)) or not isinstance(
        raw_terminal_digests, (list, tuple)
    ):
        raise RunIntegrityError("content digest indexes must be arrays")
    declared_digests = tuple(raw_declared_digests)
    terminal_digests = tuple(raw_terminal_digests)
    if not all(isinstance(item, str) and _SHA256.fullmatch(item) for item in declared_digests):
        raise RunIntegrityError("content digest index contains malformed hashes")
    if verified_digests != declared_digests or verified_digests != terminal_digests:
        raise RunIntegrityError("content-addressed artifact set does not match the sealed run")

    run_document = _read_json(root / "run.json")
    typed_run = _run_manifest_from_document(run_document)
    adapter = IncidentRoutingAdapter()
    model = DeterministicModel()
    seeds = _DEMO_SEEDS
    if (
        typed_run.run_id != RUN_ID
        or typed_run.mode != "rsi"
        or typed_run.task_pack != adapter.name
        or typed_run.seeds != seeds
        or typed_run.budget != _DEMO_LAB_BUDGET
        or typed_run.model_id != model.model_id
        or typed_run.protected_scopes != _DEMO_PROTECTED_SCOPES
    ):
        raise RunIntegrityError("run manifest does not describe the v1 offline RSI protocol")
    lab_budget = _DEMO_LAB_BUDGET

    cards_document = _read_json(root / "cards.json")
    cards = _cards_from_evidence(cards_document)
    if len(cards) != 20 or tuple(card.step_id for card in cards) != tuple(range(20)):
        raise RunIntegrityError("mining evidence must contain 20 ordered experience cards")
    if len({card.card_id for card in cards}) != len(cards) or len({card.node_id for card in cards}) != len(cards):
        raise RunIntegrityError("mining evidence contains duplicate card or node identities")
    node_task = {card.node_id: card.task_id for card in cards}
    for card in cards:
        if (
            card.seed not in seeds
            or card.split != "mine"
            or card.task_id != f"incident-routing-mine-{card.seed}"
            or card.evaluator_id != adapter.name
            or card.model_id != model.model_id
        ):
            raise RunIntegrityError("experience card task provenance is inconsistent")
        if any(parent not in node_task or node_task[parent] != card.task_id for parent in card.parent_node_ids):
            raise RunIntegrityError("experience card parent lineage crosses or omits a task")
    operator_counts = {
        operator: sum(card.operator_id == operator for card in cards)
        for operator in ("seed", "draft@1.0.0", "improve@1.0.0", "debug@1.0.0", "crossover@1.0.0")
    }
    if set(operator_counts.values()) != {4}:
        raise RunIntegrityError("mining evidence does not cover every generation-zero operator")

    replayed_cards, replayed_boards, replayed_search_artifacts = _search_mining_split(
        adapter,
        model,
        seeds,
        _DiscardEvents(),  # type: ignore[arg-type]
    )
    if tuple(to_primitive(card) for card in replayed_cards) != tuple(
        to_primitive(card) for card in cards
    ):
        raise RunIntegrityError("experience cards do not reproduce from the task adapter")
    boards_document = _read_json(root / "boards.json")
    if boards_document != {
        "schema_version": RUN_SCHEMA,
        "boards": to_primitive(replayed_boards),
    }:
        raise RunIntegrityError("experience boards do not reproduce from search evidence")

    patterns_document = _read_json(root / "patterns.json")
    raw_patterns = patterns_document.get("patterns")
    expected_patterns = TraceMiner(minimum_support=4, minimum_delta=0.05).mine(cards)
    if (
        patterns_document.get("schema_version") != RUN_SCHEMA
        or set(patterns_document) != {"schema_version", "patterns"}
        or not isinstance(raw_patterns, list)
        or raw_patterns != to_primitive(expected_patterns)
    ):
        raise RunIntegrityError("trace patterns do not reproduce from experience cards")
    try:
        selected_pattern = _select_transfer_pattern(expected_patterns)
    except DemoError as error:
        raise RunIntegrityError(str(error)) from error

    fingerprints_document = _read_json(root / "split-fingerprints.json")
    expected_fingerprints = {
        split: sorted(
            {
                digest
                for seed in seeds
                for digest in adapter.case_fingerprints(seed, split)
            }
        )
        for split in ("mine", "validation", "promotion", "sealed")
    }
    if fingerprints_document != {
        "schema_version": RUN_SCHEMA,
        "splits": expected_fingerprints,
    }:
        raise RunIntegrityError("task split fingerprints do not reproduce")
    for index, left in enumerate(expected_fingerprints):
        for right in tuple(expected_fingerprints)[index + 1 :]:
            if set(expected_fingerprints[left]) & set(expected_fingerprints[right]):
                raise RunIntegrityError(f"task inputs overlap across {left} and {right}")

    report_document = _read_json(root / "lab-report.json")
    typed_report = _lab_report_from_document(report_document)
    gate = _read_json(root / "gate-result.json")
    validation = _read_json(root / "validation-results.json")
    sealed = _read_json(root / "sealed-results.json")
    trials = _read_json(root / "promotion-trials.json")
    passport_evidence = _read_json(root / "operator-passport.json")
    artifact_index = _read_json(root / "artifact-index.json")
    operator_id = _json_string(manifest.get("operator_id"), "manifest.operator_id", nonempty=True)
    report_id = _json_string(manifest.get("report_id"), "manifest.report_id", nonempty=True)
    if not _SHA256.fullmatch(report_id):
        raise RunIntegrityError("manifest report_id is not a SHA-256 evidence identifier")
    passport = passport_evidence.get("passport")
    if (
        passport_evidence.get("schema_version") != "evolsa.operator-passport-evidence/v1"
        or set(passport_evidence) != {"schema_version", "content_hash", "passport"}
        or not isinstance(passport, Mapping)
    ):
        raise RunIntegrityError("operator passport evidence is malformed")
    try:
        typed_passport = operator_passport_from_document(passport)
    except (TypeError, ValueError) as error:
        raise RunIntegrityError(f"invalid operator passport evidence: {error}") from error
    if (
        passport_evidence.get("content_hash") != typed_passport.content_hash
        or typed_passport.evidence_ids != selected_pattern.evidence_ids
        or typed_passport.parents != selected_pattern.source_operators
    ):
        raise RunIntegrityError("operator passport does not bind the mined evidence lineage")
    expected_operator_ids = tuple(
        operator.passport.operator_id for operator in seed_operators()
    ) + (typed_passport.operator_id,)
    if typed_run.operator_ids != expected_operator_ids:
        raise RunIntegrityError("run manifest operator set does not match the replayed lineage")

    registry = LocalRegistry(root / "registry")
    trust_kernel = TrustKernel(
        {_DEMO_SIGNER: _DEMO_SECRET},
        release_signers={_DEMO_SIGNER},
        release_principals={_DEMO_ACTOR},
    )
    try:
        registered = registry.installed(
            typed_passport.name,
            trust_kernel=trust_kernel,
            actor=_DEMO_ACTOR,
        )
        installed_operator = registry.load_installed_operator(
            typed_passport.name,
            trust_kernel=trust_kernel,
            actor=_DEMO_ACTOR,
        )
    except Exception as error:
        raise RunIntegrityError(f"installed operator registry verification failed: {error}") from error
    if (
        operator_passport_document(registered) != dict(passport)
        or installed_operator.passport.content_hash != typed_passport.content_hash
    ):
        raise RunIntegrityError("installed registry object does not match operator evidence")
    try:
        trust_kernel.authorize_release(
            registered,
            _DEMO_ACTOR,
            expected_hash=typed_passport.content_hash,
        )
    except Exception as error:
        raise RunIntegrityError(f"operator release authorization failed replay: {error}") from error

    expected_gate = PromotionGate(_promotion_policy()).evaluate(
        typed_report,
        operator=typed_passport,
    )
    if to_primitive(expected_gate) != dict(gate):
        raise RunIntegrityError("promotion result does not reproduce from sealed lab evidence")
    if (
        typed_report.report_id != report_id
        or gate.get("report_id") != report_id
        or terminal.get("report_id") != report_id
        or gate.get("operator_id") != operator_id
        or terminal.get("operator_id") != operator_id
        or typed_passport.operator_id != operator_id
        or typed_passport.lifecycle is not OperatorLifecycle.STABLE
        or gate.get("decision") != "promote"
        or manifest.get("decision") != "promote"
        or terminal.get("decision") != "promote"
    ):
        raise RunIntegrityError("operator, report, lifecycle, or promotion identities disagree")

    proposed_passport = replace(
        typed_passport,
        lifecycle=OperatorLifecycle.PROPOSED,
        signature="",
    )
    proposed_content_hash = proposed_passport.content_hash
    proposed_operator = replace(installed_operator, passport=proposed_passport)
    expected_validation, replayed_validation_artifacts = _validation_evidence(
        adapter,
        model,
        proposed_operator,
        seeds,
        lab_budget,
    )
    if validation != {
        "schema_version": RUN_SCHEMA,
        "outcomes": to_primitive(expected_validation),
    }:
        raise RunIntegrityError("validation outcomes do not reproduce from the installed operator")
    (
        expected_promotion,
        expected_promotion_receipts,
        replayed_promotion_artifacts,
    ) = _promotion_evidence(
        adapter, model, proposed_operator, seeds, lab_budget
    )
    if typed_report.outcomes != expected_promotion:
        raise RunIntegrityError("paired promotion outcomes do not reproduce from the installed operator")
    if trials != {
        "schema_version": RUN_SCHEMA,
        "outcomes": to_primitive(typed_report.outcomes),
        "receipts": to_primitive(expected_promotion_receipts),
        "budget": to_primitive(lab_budget),
    }:
        raise RunIntegrityError("promotion trial receipts do not match the lab report")
    expected_sealed, expected_rsi, replayed_sealed_artifacts = _sealed_trajectory_evidence(
        adapter,
        model,
        installed_operator,
        seeds,
        _DEMO_TRAJECTORY_BUDGET,
        persistent_installation=True,
    )
    sealed_outcomes = sealed.get("outcomes")
    if not isinstance(sealed_outcomes, list) or sealed_outcomes != to_primitive(expected_sealed):
        raise RunIntegrityError("sealed outcomes do not reproduce from the installed operator")

    candidate_score = fmean(float(item["candidate_score"]) for item in expected_sealed)
    baseline_score = fmean(float(item["baseline_score"]) for item in expected_sealed)
    if (
        set(sealed) != {
            "schema_version",
            "outcomes",
            "baseline_mean",
            "candidate_mean",
            "mean_lift",
            "rsi_evidence",
        }
        or sealed.get("schema_version") != RUN_SCHEMA
        or not math.isclose(
            _json_number(sealed.get("candidate_mean"), "sealed.candidate_mean"),
            candidate_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _json_number(sealed.get("baseline_mean"), "sealed.baseline_mean"),
            baseline_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _json_number(sealed.get("mean_lift"), "sealed.mean_lift"),
            candidate_score - baseline_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or sealed.get("rsi_evidence") != expected_rsi
    ):
        raise RunIntegrityError("sealed aggregates or RSI acceleration evidence do not reproduce")
    if candidate_score <= baseline_score:
        raise RunIntegrityError("sealed split does not reproduce a positive improvement")

    expected_review = render_review_markdown(
        run_id=RUN_ID,
        cards=cards,
        reports=(typed_report,),
        gates=(expected_gate,),
    )
    expected_html = render_lineage_html(
        run_id=RUN_ID,
        cards=cards,
        passports=(*seed_operators_passports(), typed_passport),
        reports=(typed_report,),
        gates=(expected_gate,),
        metadata={
            "task_pack": adapter.name,
            "offline": True,
            "network_requests": 0,
            "sealed_mean_lift": candidate_score - baseline_score,
            "rsi_evidence": expected_rsi,
        },
    )
    try:
        rendered_review = (root / "review.md").read_text(encoding="utf-8")
        rendered_html = (root / "index.html").read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RunIntegrityError(f"cannot read rendered run report: {error}") from error
    if rendered_review != expected_review or rendered_html != expected_html:
        raise RunIntegrityError("rendered reports do not reproduce from verified evidence")

    expected_artifact_index = {
        "schema_version": RUN_SCHEMA,
        "artifacts": _build_artifact_index(
            (
                *replayed_search_artifacts,
                *replayed_validation_artifacts,
                *replayed_promotion_artifacts,
                *replayed_sealed_artifacts,
            )
        ),
    }
    if artifact_index != expected_artifact_index:
        raise RunIntegrityError("artifact occurrence provenance does not reproduce")

    raw_artifacts = artifact_index.get("artifacts")
    if (
        artifact_index.get("schema_version") != RUN_SCHEMA
        or set(artifact_index) != {"schema_version", "artifacts"}
        or not isinstance(raw_artifacts, Mapping)
        or not raw_artifacts
    ):
        raise RunIntegrityError("artifact index is malformed")
    content_store = ContentAddressedStore(artifact_root)
    indexed_digests: set[str] = set()
    occurrence_ids: list[int] = []
    for artifact_id, item in raw_artifacts.items():
        if not isinstance(artifact_id, str) or not _SHA256.fullmatch(artifact_id):
            raise RunIntegrityError("artifact index contains a malformed artifact identifier")
        if not isinstance(item, Mapping) or set(item) != {
            "store_digest",
            "media_type",
            "occurrences",
        }:
            raise RunIntegrityError("artifact index entry is malformed")
        digest = _json_string(item["store_digest"], "artifact.store_digest", nonempty=True)
        media_type = _json_string(item["media_type"], "artifact.media_type", nonempty=True)
        if not _SHA256.fullmatch(digest):
            raise RunIntegrityError("artifact store digest is malformed")
        indexed_digests.add(digest)
        content = content_store.get_text(digest)
        if Artifact(content=content, media_type=media_type).artifact_id != artifact_id:
            raise RunIntegrityError("artifact index does not bind media type and content")
        occurrences = item["occurrences"]
        if not isinstance(occurrences, list) or not occurrences:
            raise RunIntegrityError("artifact entry has no provenance occurrences")
        for occurrence in occurrences:
            if not isinstance(occurrence, Mapping) or set(occurrence) != {
                "occurrence",
                "created_by",
                "parent_ids",
                "metadata",
            }:
                raise RunIntegrityError("artifact occurrence receipt is malformed")
            occurrence_ids.append(_json_int(occurrence["occurrence"], "artifact.occurrence"))
            _json_string(occurrence["created_by"], "artifact.created_by", nonempty=True)
            _string_tuple(occurrence["parent_ids"], "artifact.parent_ids")
            if not isinstance(occurrence["metadata"], Mapping):
                raise RunIntegrityError("artifact metadata must be an object")
    if indexed_digests != set(verified_digests):
        raise RunIntegrityError("artifact index does not bind every content-addressed blob")
    expected_occurrences = sum(
        len(items)
        for items in (
            replayed_search_artifacts,
            replayed_validation_artifacts,
            replayed_promotion_artifacts,
            replayed_sealed_artifacts,
        )
    )
    if (
        sorted(occurrence_ids) != list(range(len(occurrence_ids)))
        or len(occurrence_ids) != expected_occurrences
    ):
        raise RunIntegrityError("artifact occurrence lineage is incomplete or duplicated")
    if any(card.artifact_id not in raw_artifacts for card in cards):
        raise RunIntegrityError("experience cards reference artifacts absent from the index")

    expected_search_digests = {
        hashlib.sha256(artifact.content.encode("utf-8")).hexdigest()
        for artifact in replayed_search_artifacts
    }
    if not expected_search_digests <= indexed_digests:
        raise RunIntegrityError("mining artifacts are absent from the content store")

    search_payloads: list[Mapping[str, Any]] = []
    for seed in seeds:
        seed_cards = tuple(card for card in replayed_cards if card.seed == seed)
        board_entry = next(item for item in replayed_boards if item["seed"] == seed)
        board = board_entry["board"]
        search_payloads.append(
            {
                "task_id": f"incident-routing-mine-{seed}",
                "seed": seed,
                "node_ids": [card.node_id for card in seed_cards],
                "best_node_id": board["best_node_id"],
                "best_score": board["best_score"],
                "operations": ["draft", "improve", "debug", "crossover"],
            }
        )
    expected_event_payloads: tuple[Mapping[str, Any], ...] = (
        {
            "run_id": RUN_ID,
            "mode": "rsi",
            "task_pack": adapter.name,
            "split_fingerprint_counts": {
                split: len(digests) for split, digests in expected_fingerprints.items()
            },
            "protected_scopes": [scope.value for scope in _DEMO_PROTECTED_SCOPES],
        },
        *search_payloads,
        {
            "pattern_id": selected_pattern.pattern_id,
            "change_tags": selected_pattern.change_tags,
            "support": selected_pattern.support,
            "mean_delta": selected_pattern.mean_delta,
            "evidence_ids": selected_pattern.evidence_ids,
        },
        {
            "operator_id": operator_id,
            "content_hash": proposed_content_hash,
            "pattern_id": selected_pattern.pattern_id,
            "evidence_ids": typed_passport.evidence_ids,
            "authorized_effects": [EffectScope.TASK_ARTIFACT.value],
        },
        {
            "operator_id": operator_id,
            "task_ids": [item["task_id"] for item in expected_validation],
            "candidate_mean": fmean(
                float(item["candidate_score"]) for item in expected_validation
            ),
        },
        {
            "operator_id": typed_report.operator_id,
            "baseline_id": typed_report.baseline_id,
            "report_id": typed_report.report_id,
            "mean_lift": typed_report.mean_lift,
            "lcb95": typed_report.lcb95,
            "ucb95": typed_report.ucb95,
            "matched_budget": typed_report.matched_budget,
            "matched_conditions": typed_report.matched_conditions,
            "task_ids": [outcome.task_id for outcome in typed_report.outcomes],
        },
        {
            "operator_id": expected_gate.operator_id,
            "report_id": expected_gate.report_id,
            "decision": expected_gate.decision.value,
            "policy_hash": expected_gate.policy_hash,
            "reasons": expected_gate.reasons,
        },
        {
            "operator_id": operator_id,
            "content_hash": typed_passport.content_hash,
            "lifecycle": OperatorLifecycle.STABLE.value,
            "release_actor": _DEMO_ACTOR,
            "registry_root": "registry",
        },
        {
            "operator_id": operator_id,
            "baseline_mean": baseline_score,
            "candidate_mean": candidate_score,
            "mean_lift": candidate_score - baseline_score,
            "task_ids": [item["task_id"] for item in expected_sealed],
            "rsi_evidence": expected_rsi,
        },
        {
            "run_id": RUN_ID,
            "evidence_files": evidence_files,
            "content_digests": verified_digests,
            "event_count": len(records),
            "operator_id": operator_id,
            "report_id": report_id,
            "decision": "promote",
        },
    )
    for index, (record, expected_payload) in enumerate(
        zip(records, expected_event_payloads, strict=True)
    ):
        expected_timestamp = (_EVENT_EPOCH + timedelta(seconds=index)).isoformat()
        if (
            record.timestamp != expected_timestamp
            or to_primitive(record.payload) != to_primitive(expected_payload)
        ):
            raise RunIntegrityError(
                f"event payload or timestamp does not reproduce at sequence {index}"
            )

    return {
        "run_id": RUN_ID,
        "output": str(root),
        "decision": "promote",
        "operator_id": operator_id,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "report_id": report_id,
        "event_count": len(records),
        "artifact_count": len(verified_digests),
        "replay_verified": True,
        "integrity_checked": True,
        "semantics_replayed": True,
        "authenticated": False,
    }


def inspect_run(path: str | Path) -> Mapping[str, Any]:
    """Return the compact, verified summary used by the CLI inspector."""

    return replay_run(path)


__all__ = [
    "DemoError",
    "INDEX_SCHEMA",
    "RUN_ID",
    "RUN_SCHEMA",
    "RunIntegrityError",
    "inspect_run",
    "replay_run",
    "run_offline_rsi_demo",
]
