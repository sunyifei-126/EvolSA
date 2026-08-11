"""Deterministic evaluator-backed task pack for the offline RSI demo."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

from evolsa.core import (
    Artifact,
    Budget,
    ExecutionStatus,
    MetricDirection,
    Observation,
    ScoreCard,
    TaskState,
)
from evolsa.core.serde import canonical_json
from evolsa.core.hashing import sha256_json


_PLAIN_CASES = {
    "mine": (
        ("Finance reported a possible phishing message", "security"),
        ("A regional outage is affecting the dashboard", "reliability"),
        ("Please process a refund for order 104", "billing"),
        ("The customer is locked out of the portal", "access"),
    ),
    "validation": (
        ("A credential leak was reported in staging", "security"),
        ("The status page shows service unavailable", "reliability"),
        ("The buyer was charged twice during checkout", "billing"),
        ("The operator cannot login to the console", "access"),
    ),
    "promotion": (
        ("A malware alert came from the attachment scanner", "security"),
        ("The api unreachable alarm fired in Europe", "reliability"),
        ("A duplicate charge appears on invoice 72", "billing"),
        ("A password reset is required before sign in", "access"),
    ),
    "sealed": (
        ("The message contains a phishing redirect", "security"),
        ("The worker queue stalled after deployment", "reliability"),
        ("The payment reversed overnight", "billing"),
        ("The browser session expired unexpectedly", "access"),
    ),
}
_HYPHEN_CASES = {
    "mine": (
        ("credential-leak detected in build logs", "security"),
        ("cannot-login after the SSO migration", "access"),
        ("charged-twice for the same invoice", "billing"),
        ("service-unavailable in the west region", "reliability"),
    ),
    "validation": (
        ("account-takeover warning from monitoring", "security"),
        ("password-reset loop blocks sign in", "access"),
        ("duplicate-charge appears on the statement", "billing"),
        ("api-unreachable from two regions", "reliability"),
    ),
    "promotion": (
        ("malware-alert triggered by an attachment", "security"),
        ("two-factor failure prevents access", "access"),
        ("payment-reversed but balance is unchanged", "billing"),
        ("database-offline during deployment", "reliability"),
    ),
    "sealed": (
        ("secret-exposure found in a public artifact", "security"),
        ("session-expired immediately after login", "access"),
        ("duplicate-charge on card after retry", "billing"),
        ("queue-stalled across all workers", "reliability"),
    ),
}

_SPLIT_RULES = {
    "security": ("phishing", "credential leak", "account takeover", "malware alert", "secret exposure"),
    "reliability": ("outage", "service unavailable", "api unreachable", "database offline", "queue stalled"),
    "billing": ("refund", "charged twice", "duplicate charge", "payment reversed", "card charged"),
    "access": ("locked out", "cannot login", "password reset", "two factor", "session expired"),
}
_ALLOWED_NORMALIZERS = frozenset({"lower", "collapse_whitespace", "strip_punctuation", "hyphen_to_space"})


def _cases(split: str, seed: int) -> tuple[tuple[str, str], ...]:
    if split not in _HYPHEN_CASES:
        raise ValueError(f"unknown task split: {split}")
    special = _HYPHEN_CASES[split][seed % len(_HYPHEN_CASES[split])]
    plain_cases = _PLAIN_CASES[split]
    rotation = seed % len(plain_cases)
    base = plain_cases[rotation:] + plain_cases[:rotation]
    return (special,) + base


def _normalize(text: str, normalizers: Iterable[str]) -> str:
    result = text
    for normalizer in normalizers:
        if normalizer == "lower":
            result = result.lower()
        elif normalizer == "collapse_whitespace":
            result = " ".join(result.split())
        elif normalizer == "strip_punctuation":
            result = re.sub(r"[^\w\s-]", "", result)
        elif normalizer == "hyphen_to_space":
            result = result.replace("-", " ")
        else:
            raise ValueError(f"unsupported normalizer: {normalizer}")
    return result


def parse_policy(artifact: Artifact) -> dict[str, object]:
    try:
        policy = json.loads(artifact.content)
    except json.JSONDecodeError as error:
        raise ValueError(f"artifact is not valid JSON: {error.msg}") from error
    if not isinstance(policy, dict) or set(policy) != {"normalizers", "rules", "fallback"}:
        raise ValueError("policy must contain exactly normalizers, rules, and fallback")
    normalizers = policy["normalizers"]
    rules = policy["rules"]
    if not isinstance(normalizers, list) or not all(isinstance(item, str) for item in normalizers):
        raise ValueError("normalizers must be a list of strings")
    if len(normalizers) != len(set(normalizers)) or not set(normalizers) <= _ALLOWED_NORMALIZERS:
        raise ValueError("normalizers contain duplicates or unsupported values")
    if not isinstance(rules, dict) or set(rules) != set(_SPLIT_RULES):
        raise ValueError("rules must define the four supported labels")
    for label, phrases in rules.items():
        if not isinstance(label, str) or not isinstance(phrases, list) or not phrases:
            raise ValueError("every label requires a non-empty phrase list")
        if not all(isinstance(phrase, str) and phrase for phrase in phrases):
            raise ValueError("rule phrases must be non-empty strings")
    if policy["fallback"] != "other":
        raise ValueError("fallback must be other")
    return policy


def policy_with_normalizer(parent: Artifact, normalizer: str, *, created_by: str) -> Artifact:
    policy = parse_policy(parent)
    normalizers = list(policy["normalizers"])
    if normalizer not in _ALLOWED_NORMALIZERS:
        raise ValueError(f"unsupported normalizer: {normalizer}")
    if normalizer not in normalizers:
        normalizers.append(normalizer)
    policy["normalizers"] = normalizers
    return Artifact(
        content=canonical_json(policy),
        parent_ids=(parent.artifact_id,),
        created_by=created_by,
        metadata={"method_family": "normalization", "change_tags": [normalizer]},
    )


def infer_normalizer_from_feedback(
    parent: Artifact,
    feedback: Iterable[Mapping[str, str]],
) -> str:
    """Select a declarative normalizer from structured mining feedback.

    This is a tiny deterministic stand-in for an LLM proposal policy: it tests
    every task-declared mutation against observed *mine-split* misses and
    returns the best repair. Promotion and sealed feedback are never supplied.
    """

    policy = parse_policy(parent)
    existing = tuple(str(item) for item in policy["normalizers"])
    misses = tuple(feedback)
    scored: list[tuple[int, str]] = []
    for candidate in sorted(_ALLOWED_NORMALIZERS - set(existing)):
        repaired = 0
        for miss in misses:
            text = miss.get("input")
            expected = miss.get("expected")
            if not isinstance(text, str) or expected not in policy["rules"]:
                continue
            normalized = _normalize(text, (*existing, candidate))
            phrases = policy["rules"][expected]
            if any(str(phrase) in normalized for phrase in phrases):
                repaired += 1
        scored.append((repaired, candidate))
    if not scored or max(score for score, _ in scored) <= 0:
        raise ValueError("no declared normalizer repairs the observed mining feedback")
    return max(scored, key=lambda item: (item[0], item[1]))[1]


@dataclass(slots=True)
class IncidentRoutingAdapter:
    """Tiny prompt/workflow task with a hidden deterministic evaluator."""

    name: str = "incident-routing-lite"

    def case_fingerprints(self, seed: int, split: str) -> tuple[str, ...]:
        """Expose only deterministic input digests for split-overlap audits."""

        return tuple(sha256_json({"input": text}) for text, _ in _cases(split, seed))

    def prepare(self, seed: int, split: str = "mine") -> TaskState:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if split not in _HYPHEN_CASES:
            raise ValueError(f"unsupported split: {split}")
        return TaskState(
            task_id=f"incident-routing-{split}-{seed}",
            split=split,
            seed=seed,
            specification=(
                "Route incident text to security, reliability, billing, access, or other. "
                "The artifact is a JSON policy with an ordered normalization pipeline and phrase rules."
            ),
            metadata={"labels": sorted(_SPLIT_RULES), "pack": self.name},
        )

    def seed_artifact(self, state: TaskState) -> Artifact:
        del state
        policy = {
            "normalizers": ["lower", "collapse_whitespace"],
            "rules": {label: list(phrases) for label, phrases in _SPLIT_RULES.items()},
            "fallback": "other",
        }
        return Artifact(
            content=canonical_json(policy),
            created_by="seed",
            metadata={"method_family": "phrase-rules", "change_tags": []},
        )

    def execute(self, state: TaskState, artifact: Artifact, budget: Budget) -> Observation:
        if budget.max_attempts < 1 or budget.max_exec_seconds <= 0:
            return Observation(status=ExecutionStatus.BUDGET_EXHAUSTED, error_signature="budget")
        try:
            policy = parse_policy(artifact)
            predictions: list[str] = []
            expected: list[str] = []
            misses: list[Mapping[str, str]] = []
            for text, label in _cases(state.split, state.seed):
                normalized = _normalize(text, policy["normalizers"])
                prediction = str(policy["fallback"])
                for candidate_label, phrases in policy["rules"].items():
                    if any(str(phrase) in normalized for phrase in phrases):
                        prediction = candidate_label
                        break
                predictions.append(prediction)
                expected.append(label)
                if prediction != label:
                    misses.append({"input": text, "predicted": prediction, "expected": label})
            correct = sum(prediction == label for prediction, label in zip(predictions, expected, strict=True))
            confusion = Counter(f"{label}->{prediction}" for prediction, label in zip(predictions, expected, strict=True))
            return Observation(
                status=ExecutionStatus.SUCCESS,
                output=canonical_json({"predictions": predictions}),
                logs=f"evaluated {len(expected)} held-out examples",
                runtime_seconds=0.001 * len(expected),
                metadata={
                    "correct": correct,
                    "total": len(expected),
                    "misses": misses,
                    "confusion": dict(sorted(confusion.items())),
                    "artifact_id": artifact.artifact_id,
                },
            )
        except (TypeError, ValueError) as error:
            return Observation(
                status=ExecutionStatus.INVALID,
                logs=str(error),
                error_signature="invalid-policy",
                metadata={"artifact_id": artifact.artifact_id},
            )

    def score(self, state: TaskState, observation: Observation) -> ScoreCard:
        del state
        if observation.status is not ExecutionStatus.SUCCESS:
            return ScoreCard(
                primary=0.0,
                direction=MetricDirection.HIGHER_IS_BETTER,
                metrics={"accuracy": 0.0},
                guardrails={"valid_rate": 0.0, "protected_effects": 0.0},
                valid=False,
                reason=observation.error_signature or observation.status.value,
            )
        total = int(observation.metadata["total"])
        correct = int(observation.metadata["correct"])
        accuracy = correct / total
        return ScoreCard(
            primary=accuracy,
            direction=MetricDirection.HIGHER_IS_BETTER,
            metrics={"accuracy": accuracy, "coverage": total},
            guardrails={"valid_rate": 1.0, "protected_effects": 0.0},
        )


class DeterministicModel:
    """Offline ModelBackend used only to make the reference demo reproducible."""

    model_id = "evolsa/deterministic-demo-v1"

    def __init__(self) -> None:
        self.generate_calls = 0

    def generate(self, messages: Iterable[Mapping[str, str]], budget: Budget) -> str:
        self.generate_calls += 1
        if budget.max_attempts < 1 or budget.max_exec_seconds <= 0:
            raise ValueError("generation budget is exhausted")
        materialized = tuple(messages)
        instruction = materialized[0].get("content", "") if materialized else ""
        selected = next(
            (normalizer for normalizer in sorted(_ALLOWED_NORMALIZERS) if normalizer in instruction),
            "collapse_whitespace",
        )
        # A foundry-compiled prompt carries its mined change tag, while the
        # user message carries the audited parent.  The backend interprets the
        # declarative tag generically; no demo-only Python callback is needed.
        for message in reversed(materialized):
            try:
                context = json.loads(message.get("content", ""))
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(context, Mapping):
                continue
            # CallbackOperator supplies its context directly. SeedOperator
            # supplies an evidence envelope whose ``context`` member contains
            # the same parent surface. Supporting both keeps the deterministic
            # backend model-independent while ensuring the canonical seed
            # operator implementation is actually executed in paired labs.
            operator_context = context.get("context")
            if isinstance(operator_context, Mapping):
                context = operator_context
            parents = context.get("parents")
            if not isinstance(parents, list) or not parents or not isinstance(parents[0], Mapping):
                continue
            raw_artifact = parents[0].get("artifact")
            if not isinstance(raw_artifact, Mapping):
                continue
            parent = Artifact(
                content=str(raw_artifact.get("content", "")),
                media_type=str(raw_artifact.get("media_type", "application/json")),
                created_by="deterministic-model-parent",
            )
            operator = context.get("operator")
            created_by = (
                str(operator.get("id"))
                if isinstance(operator, Mapping) and operator.get("id")
                else str(context.get("operator_id") or self.model_id)
            )
            improved = policy_with_normalizer(parent, selected, created_by=created_by)
            return canonical_json({"content": improved.content, "media_type": improved.media_type})
        return canonical_json({"normalizer": selected})
