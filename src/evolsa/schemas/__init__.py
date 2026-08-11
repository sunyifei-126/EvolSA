"""Versioned JSON Schemas shipped with every EvolSA distribution.

The loading helpers use :mod:`importlib.resources`, so they work from an
installed wheel as well as a source checkout.  Callers should load the schema
through this module instead of assuming that the repository-level ``schemas``
directory is present.
"""

from __future__ import annotations

import json
import math
from importlib.resources import files
from typing import Any, Mapping

from evolsa.core import (
    Budget,
    EffectScope,
    OperatorLifecycle,
    OperatorPassport,
    OperatorVerification,
)
from evolsa.core.serde import to_primitive


OPERATOR_PASSPORT_SCHEMA = "operator-passport.schema.json"


def operator_passport_schema_bytes() -> bytes:
    """Return the canonical operator-passport schema as UTF-8 bytes."""

    return files(__package__).joinpath(OPERATOR_PASSPORT_SCHEMA).read_bytes()


def load_operator_passport_schema() -> dict[str, Any]:
    """Load and parse the canonical operator-passport JSON Schema.

    A fresh dictionary is returned on every call, allowing consumers to
    annotate a schema locally without mutating shared module state.
    """

    return json.loads(operator_passport_schema_bytes())


def operator_passport_document(passport: OperatorPassport) -> dict[str, Any]:
    """Serialize a passport using the public JSON Schema representation.

    The Python dataclass keeps ``generator_type`` and ``generator_ref`` flat
    for ergonomic access.  The wire contract intentionally groups them under
    ``generator``.  Keeping this conversion explicit prevents generic
    dataclass serialization from drifting away from the published schema.
    """

    if not isinstance(passport, OperatorPassport):
        raise TypeError("passport must be an OperatorPassport")
    document = to_primitive(passport)
    generator_type = document.pop("generator_type")
    generator_ref = document.pop("generator_ref")
    document["generator"] = {"type": generator_type, "ref": generator_ref}
    return document


def operator_passport_from_document(document: Mapping[str, Any]) -> OperatorPassport:
    """Parse the public JSON representation into its typed contract.

    This parser is deliberately strict about unknown top-level fields and the
    nested generator shape.  JSON Schema validation can still be applied by a
    caller that wants richer diagnostics, but unsafe ambiguity is rejected
    without requiring a third-party validator.
    """

    if type(document) is not dict:
        raise TypeError("operator passport document must be a JSON object")
    if any(type(key) is not str for key in document):
        raise ValueError("operator passport object keys must be strings")
    schema = load_operator_passport_schema()
    properties = set(schema["properties"])
    keys = set(document)
    missing = set(schema["required"]) - keys
    unknown = keys - properties
    if missing:
        raise ValueError(f"operator passport is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"operator passport has unknown fields: {sorted(unknown)}")
    if (
        type(document.get("api_version")) is not str
        or document["api_version"] != "evolsa/v1"
    ):
        raise ValueError("unsupported operator passport api_version")
    generator = document.get("generator")
    budget = document.get("budget")
    verification = document.get("verification")
    if type(generator) is not dict or set(generator) != {"type", "ref"}:
        raise ValueError("operator passport generator must contain exactly type and ref")
    if type(budget) is not dict or type(verification) is not dict:
        raise ValueError("operator passport budget and verification must be objects")
    if any(
        type(key) is not str
        for nested in (generator, budget, verification)
        for key in nested
    ):
        raise ValueError("operator passport nested object keys must be strings")

    budget_required = {"max_tokens", "max_exec_seconds", "max_attempts"}
    budget_allowed = budget_required | {"max_cost"}
    budget_missing = budget_required - set(budget)
    budget_unknown = set(budget) - budget_allowed
    if budget_missing or budget_unknown:
        raise ValueError(
            "operator passport budget has invalid fields: "
            f"missing={sorted(budget_missing)}, unknown={sorted(budget_unknown)}"
        )
    verification_fields = {"baseline", "primary_metric", "guardrails", "minimum_tasks"}
    if set(verification) != verification_fields:
        raise ValueError(
            "operator passport verification must contain exactly "
            "baseline, primary_metric, guardrails, and minimum_tasks"
        )

    def string(container: Mapping[str, Any], name: str, *, allow_empty: bool = False) -> str:
        value = container.get(name)
        if type(value) is not str or (not allow_empty and not value):
            qualifier = "a string" if allow_empty else "a non-empty string"
            raise ValueError(f"operator passport {name} must be {qualifier}")
        return value

    def strings(name: str) -> tuple[str, ...]:
        value = document.get(name, ())
        if type(value) is not list or not all(type(item) is str and item for item in value):
            raise ValueError(f"operator passport {name} must be an array of strings")
        if len(value) != len(set(value)):
            raise ValueError(f"operator passport {name} must contain unique values")
        return tuple(value)

    def integer(container: Mapping[str, Any], name: str, *, minimum: int) -> int:
        value = container.get(name)
        if type(value) is not int or value < minimum:
            raise ValueError(f"operator passport {name} must be an integer >= {minimum}")
        return value

    def number(
        container: Mapping[str, Any],
        name: str,
        *,
        default: float | None = None,
    ) -> float | int:
        value = container.get(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"operator passport {name} must be a finite number >= 0")
        return value

    name = string(document, "name")
    version = string(document, "version")
    intent = string(document, "intent")
    context_policy = string(document, "context_policy")
    generator_type = string(generator, "type")
    generator_ref = string(generator, "ref")
    baseline = string(verification, "baseline")
    primary_metric = string(verification, "primary_metric")
    lifecycle = string(document, "lifecycle")
    signature = string(document, "signature", allow_empty=True) if "signature" in document else ""
    inputs = strings("inputs")
    if not inputs:
        raise ValueError("operator passport inputs must contain at least one value")
    guardrails_value = verification["guardrails"]
    if (
        type(guardrails_value) is not list
        or not all(type(item) is str and item for item in guardrails_value)
        or len(guardrails_value) != len(set(guardrails_value))
    ):
        raise ValueError("operator passport verification guardrails must be unique strings")

    try:
        return OperatorPassport(
            name=name,
            version=version,
            intent=intent,
            inputs=inputs,
            preconditions=strings("preconditions"),
            expected_effects=tuple(EffectScope(item) for item in strings("expected_effects")),
            forbidden_effects=tuple(EffectScope(item) for item in strings("forbidden_effects")),
            context_policy=context_policy,
            generator_type=generator_type,
            generator_ref=generator_ref,
            budget=Budget(
                max_tokens=integer(budget, "max_tokens", minimum=0),
                max_exec_seconds=number(budget, "max_exec_seconds"),
                max_attempts=integer(budget, "max_attempts", minimum=1),
                max_cost=number(budget, "max_cost", default=0.0),
            ),
            verification=OperatorVerification(
                baseline=baseline,
                primary_metric=primary_metric,
                guardrails=tuple(guardrails_value),
                minimum_tasks=integer(verification, "minimum_tasks", minimum=1),
            ),
            parents=strings("parents") if "parents" in document else (),
            known_failures=strings("known_failures") if "known_failures" in document else (),
            lifecycle=OperatorLifecycle(lifecycle),
            evidence_ids=strings("evidence_ids") if "evidence_ids" in document else (),
            signature=signature,
            api_version=document["api_version"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid operator passport: {error}") from error


__all__ = [
    "OPERATOR_PASSPORT_SCHEMA",
    "load_operator_passport_schema",
    "operator_passport_document",
    "operator_passport_from_document",
    "operator_passport_schema_bytes",
]
