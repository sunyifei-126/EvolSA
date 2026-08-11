"""Deterministic serialization helpers used by evidence and content hashes."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


def to_primitive(value: Any) -> Any:
    """Convert EvolSA values into JSON-compatible deterministic primitives."""

    if is_dataclass(value):
        return {field.name: to_primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [to_primitive(item) for item in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported value for deterministic serialization: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return stable UTF-8 JSON suitable for hashing and audit records."""

    return json.dumps(
        to_primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def pretty_json(value: Any) -> str:
    """Return human-readable JSON without changing its semantics."""

    return json.dumps(to_primitive(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
