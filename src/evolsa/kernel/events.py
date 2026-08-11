"""Durable append-only, hash-chained audit events."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from evolsa.core.hashing import sha256_json
from evolsa.core.serde import canonical_json, to_primitive

try:  # pragma: no cover - the fallback is for non-POSIX consumers.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


GENESIS_HASH = "0" * 64
EVENT_SCHEMA = "evolsa.event/v1"
_PROCESS_LOCKS: dict[Path, RLock] = {}
_PROCESS_LOCKS_GUARD = RLock()


class EventLogError(RuntimeError):
    """Base class for event-log failures."""


class EventLogIntegrityError(EventLogError):
    """Raised when persisted audit history is malformed or has changed."""


@dataclass(frozen=True, slots=True)
class EventRecord:
    sequence: int
    timestamp: str
    event_type: str
    payload: Mapping[str, Any]
    previous_hash: str
    event_hash: str
    schema_version: str = EVENT_SCHEMA

    @property
    def event_id(self) -> str:
        return self.event_hash

    def hash_payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
        }

    def to_dict(self) -> Mapping[str, Any]:
        return {**self.hash_payload(), "event_hash": self.event_hash}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EventLogIntegrityError(f"duplicate JSON key in event record: {key}")
        result[key] = value
    return result


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


class EventLog:
    """A newline-delimited audit log with a SHA-256 chain and durable head.

    The sidecar head detects suffix truncation in addition to normal chain
    edits. This is tamper-evident local storage, not a signature or remote
    transparency service; protect the log and head with OS access controls.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path).expanduser().resolve(strict=False)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._head_path = self._path.with_name(self._path.name + ".head")
        self._lock_path = self._path.with_name(self._path.name + ".lock")
        with _PROCESS_LOCKS_GUARD:
            self._thread_lock = _PROCESS_LOCKS.setdefault(self._lock_path, RLock())

        with self._exclusive_lock():
            log_exists = self._path.exists()
            head_exists = self._head_path.exists()
            if not log_exists:
                self._path.touch(mode=0o600)
            if not head_exists:
                if log_exists and self._path.stat().st_size:
                    raise EventLogIntegrityError("non-empty event log is missing its durable head")
                self._write_head(-1, GENESIS_HASH)
            self._verify_unlocked()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def head_path(self) -> Path:
        return self._head_path

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        timestamp: str | None = None,
    ) -> EventRecord:
        if not isinstance(event_type, str) or not event_type.strip() or "\x00" in event_type:
            raise ValueError("event_type must be a non-empty string without NUL")
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        primitive_payload = to_primitive(payload)
        if not isinstance(primitive_payload, dict):  # defensive: Mapping should serialize to dict.
            raise TypeError("payload must serialize to a JSON object")
        event_timestamp = timestamp or datetime.now(timezone.utc).isoformat(timespec="microseconds")
        self._validate_timestamp(event_timestamp)

        with self._exclusive_lock():
            records = self._verify_unlocked()
            sequence = len(records)
            previous_hash = records[-1].event_hash if records else GENESIS_HASH
            hash_payload = {
                "schema_version": EVENT_SCHEMA,
                "sequence": sequence,
                "timestamp": event_timestamp,
                "event_type": event_type,
                "payload": primitive_payload,
                "previous_hash": previous_hash,
            }
            event_hash = sha256_json(hash_payload)
            record = EventRecord(
                event_hash=event_hash,
                **{**hash_payload, "payload": _freeze_json(primitive_payload)},
            )
            encoded = (canonical_json(record.to_dict()) + "\n").encode("utf-8")
            descriptor = os.open(self._path, os.O_WRONLY | os.O_APPEND)
            try:
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise EventLogError("event append made no progress")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._write_head(sequence, event_hash)
            return record

    def verify(self) -> tuple[EventRecord, ...]:
        """Re-read and verify every record, including the durable head."""

        with self._exclusive_lock():
            return self._verify_unlocked()

    def records(self) -> tuple[EventRecord, ...]:
        return self.verify()

    def __iter__(self) -> Iterator[EventRecord]:
        return iter(self.verify())

    def __len__(self) -> int:
        return len(self.verify())

    @property
    def last_hash(self) -> str:
        records = self.verify()
        return records[-1].event_hash if records else GENESIS_HASH

    def _verify_unlocked(self) -> tuple[EventRecord, ...]:
        records: list[EventRecord] = []
        previous_hash = GENESIS_HASH
        try:
            with self._path.open("r", encoding="utf-8", newline="") as stream:
                for sequence, line in enumerate(stream):
                    if not line.endswith("\n"):
                        raise EventLogIntegrityError("event log ends with a partial record")
                    try:
                        raw = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
                    except (json.JSONDecodeError, UnicodeDecodeError) as error:
                        raise EventLogIntegrityError(f"invalid event JSON at sequence {sequence}") from error
                    record = self._parse_record(raw, sequence, previous_hash)
                    records.append(record)
                    previous_hash = record.event_hash
        except FileNotFoundError as error:
            raise EventLogIntegrityError("event log disappeared") from error

        head_sequence, head_hash = self._read_head()
        expected_sequence = len(records) - 1
        expected_hash = records[-1].event_hash if records else GENESIS_HASH
        if (head_sequence, head_hash) != (expected_sequence, expected_hash):
            raise EventLogIntegrityError("event log does not match its durable head")
        return tuple(records)

    def _parse_record(self, raw: Any, sequence: int, previous_hash: str) -> EventRecord:
        required = {
            "schema_version",
            "sequence",
            "timestamp",
            "event_type",
            "payload",
            "previous_hash",
            "event_hash",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise EventLogIntegrityError(f"invalid event fields at sequence {sequence}")
        if raw["schema_version"] != EVENT_SCHEMA or raw["sequence"] != sequence:
            raise EventLogIntegrityError(f"invalid event identity at sequence {sequence}")
        if raw["previous_hash"] != previous_hash:
            raise EventLogIntegrityError(f"broken hash chain at sequence {sequence}")
        if not isinstance(raw["event_type"], str) or not raw["event_type"]:
            raise EventLogIntegrityError(f"invalid event type at sequence {sequence}")
        if not isinstance(raw["payload"], dict):
            raise EventLogIntegrityError(f"invalid event payload at sequence {sequence}")
        self._validate_timestamp(raw["timestamp"], integrity=True)
        supplied_hash = raw["event_hash"]
        hash_payload = {key: raw[key] for key in required if key != "event_hash"}
        actual_hash = sha256_json(hash_payload)
        if not isinstance(supplied_hash, str) or supplied_hash != actual_hash:
            raise EventLogIntegrityError(f"event hash mismatch at sequence {sequence}")
        return EventRecord(**{**raw, "payload": _freeze_json(raw["payload"])})

    @staticmethod
    def _validate_timestamp(timestamp: Any, *, integrity: bool = False) -> None:
        try:
            if not isinstance(timestamp, str):
                raise ValueError
            parsed = datetime.fromisoformat(timestamp)
            if parsed.tzinfo is None:
                raise ValueError
        except ValueError as error:
            exception = EventLogIntegrityError if integrity else ValueError
            raise exception("event timestamp must be timezone-aware ISO-8601") from error

    def _read_head(self) -> tuple[int, str]:
        try:
            raw = json.loads(self._head_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
        except (OSError, json.JSONDecodeError) as error:
            raise EventLogIntegrityError("event-log head is unreadable") from error
        if not isinstance(raw, dict) or set(raw) != {"sequence", "event_hash"}:
            raise EventLogIntegrityError("event-log head is malformed")
        sequence = raw["sequence"]
        event_hash = raw["event_hash"]
        if not isinstance(sequence, int) or sequence < -1:
            raise EventLogIntegrityError("event-log head sequence is invalid")
        if (
            not isinstance(event_hash, str)
            or len(event_hash) != 64
            or any(character not in "0123456789abcdef" for character in event_hash)
        ):
            raise EventLogIntegrityError("event-log head hash is invalid")
        return sequence, event_hash

    def _write_head(self, sequence: int, event_hash: str) -> None:
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._head_path.parent,
                prefix=".event-head-",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(canonical_json({"sequence": sequence, "event_hash": event_hash}) + "\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, self._head_path)
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        with self._thread_lock:
            descriptor = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
