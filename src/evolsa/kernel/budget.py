"""Thread-safe, fail-closed accounting for bounded EvolSA runs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from threading import RLock
from uuid import uuid4

from evolsa.core import Budget, BudgetUsage


class BudgetError(RuntimeError):
    """Base class for budget ledger failures."""


class BudgetExceededError(BudgetError):
    """Raised before an operation would exceed a configured hard limit."""


class UnknownReservationError(BudgetError):
    """Raised when a reservation is absent or has already been finalized."""


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    """An opaque claim against a ledger's remaining budget."""

    reservation_id: str
    requested: BudgetUsage


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """A consistent point-in-time view of a budget ledger."""

    limit: Budget
    used: BudgetUsage
    reserved: BudgetUsage

    @property
    def committed(self) -> BudgetUsage:
        """Return usage plus outstanding reservations."""

        return _add(self.used, self.reserved)

    @property
    def remaining(self) -> BudgetUsage:
        committed = self.committed
        return BudgetUsage(
            tokens=(
                0
                if self.limit.max_tokens == 0
                else self.limit.max_tokens - committed.tokens
            ),
            exec_seconds=max(0.0, self.limit.max_exec_seconds - committed.exec_seconds),
            attempts=self.limit.max_attempts - committed.attempts,
            cost=(
                0.0
                if self.limit.max_cost == 0.0
                else max(0.0, self.limit.max_cost - committed.cost)
            ),
        )


def _add(left: BudgetUsage, right: BudgetUsage) -> BudgetUsage:
    return BudgetUsage(
        tokens=left.tokens + right.tokens,
        exec_seconds=left.exec_seconds + right.exec_seconds,
        attempts=left.attempts + right.attempts,
        cost=left.cost + right.cost,
    )


def _subtract(left: BudgetUsage, right: BudgetUsage) -> BudgetUsage:
    # Values originate from prior additions under the same lock. Clamp tiny
    # floating-point cancellation artifacts while retaining integer checks.
    tokens = left.tokens - right.tokens
    attempts = left.attempts - right.attempts
    exec_seconds = left.exec_seconds - right.exec_seconds
    cost = left.cost - right.cost
    if tokens < 0 or attempts < 0 or exec_seconds < -1e-12 or cost < -1e-12:
        raise BudgetError("ledger invariant violated while subtracting usage")
    return BudgetUsage(
        tokens=tokens,
        exec_seconds=max(0.0, exec_seconds),
        attempts=attempts,
        cost=max(0.0, cost),
    )


def _exceeded_resources(usage: BudgetUsage, limit: Budget) -> tuple[str, ...]:
    exceeded: list[str] = []
    if limit.max_tokens and usage.tokens > limit.max_tokens:
        exceeded.append("tokens")
    if usage.exec_seconds > limit.max_exec_seconds and not math.isclose(
        usage.exec_seconds, limit.max_exec_seconds, rel_tol=1e-12, abs_tol=1e-12
    ):
        exceeded.append("exec_seconds")
    if usage.attempts > limit.max_attempts:
        exceeded.append("attempts")
    if limit.max_cost and usage.cost > limit.max_cost and not math.isclose(
        usage.cost, limit.max_cost, rel_tol=1e-12, abs_tol=1e-12
    ):
        exceeded.append("cost")
    return tuple(exceeded)


class BudgetLedger:
    """Atomically reserve and settle resources against immutable hard limits.

    Capacity is checked against both settled usage and every outstanding
    reservation.  A failed reservation or settlement leaves all counters
    unchanged, making the ledger safe for concurrent schedulers.
    """

    def __init__(self, limit: Budget) -> None:
        if not isinstance(limit, Budget):
            raise TypeError("limit must be a Budget")
        self._limit = limit
        self._used = BudgetUsage()
        self._reserved = BudgetUsage()
        self._reservations: dict[str, BudgetUsage] = {}
        self._lock = RLock()

    @property
    def limit(self) -> Budget:
        return self._limit

    @property
    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(self._limit, self._used, self._reserved)

    @property
    def used(self) -> BudgetUsage:
        return self.snapshot.used

    @property
    def reserved(self) -> BudgetUsage:
        return self.snapshot.reserved

    @property
    def remaining(self) -> BudgetUsage:
        return self.snapshot.remaining

    def reserve(self, requested: BudgetUsage) -> BudgetReservation:
        """Reserve capacity, raising without mutation when it is unavailable."""

        if not isinstance(requested, BudgetUsage):
            raise TypeError("requested must be BudgetUsage")
        with self._lock:
            candidate = _add(_add(self._used, self._reserved), requested)
            self._require_within_limit(candidate)
            reservation_id = uuid4().hex
            self._reservations[reservation_id] = requested
            self._reserved = _add(self._reserved, requested)
            return BudgetReservation(reservation_id, requested)

    def settle(
        self,
        reservation: BudgetReservation | str,
        actual: BudgetUsage | None = None,
    ) -> BudgetSnapshot:
        """Atomically replace a reservation with actual measured usage.

        Actual usage may exceed the original reservation only when the global
        hard limits still have capacity.  Otherwise the reservation remains
        live so the caller can retry with corrected accounting or cancel it.
        """

        reservation_id = self._reservation_id(reservation)
        with self._lock:
            requested = self._reservations.get(reservation_id)
            if requested is None:
                raise UnknownReservationError(f"unknown reservation: {reservation_id}")
            measured = requested if actual is None else actual
            if not isinstance(measured, BudgetUsage):
                raise TypeError("actual must be BudgetUsage")

            other_reserved = _subtract(self._reserved, requested)
            candidate = _add(_add(self._used, other_reserved), measured)
            self._require_within_limit(candidate)

            self._reserved = other_reserved
            self._used = _add(self._used, measured)
            del self._reservations[reservation_id]
            return BudgetSnapshot(self._limit, self._used, self._reserved)

    def cancel(self, reservation: BudgetReservation | str) -> BudgetSnapshot:
        """Release an outstanding reservation without charging it."""

        reservation_id = self._reservation_id(reservation)
        with self._lock:
            requested = self._reservations.pop(reservation_id, None)
            if requested is None:
                raise UnknownReservationError(f"unknown reservation: {reservation_id}")
            self._reserved = _subtract(self._reserved, requested)
            return BudgetSnapshot(self._limit, self._used, self._reserved)

    def charge(self, actual: BudgetUsage) -> BudgetSnapshot:
        """Reserve and immediately settle a measured usage atomically."""

        if not isinstance(actual, BudgetUsage):
            raise TypeError("actual must be BudgetUsage")
        with self._lock:
            candidate = _add(_add(self._used, self._reserved), actual)
            self._require_within_limit(candidate)
            self._used = _add(self._used, actual)
            return BudgetSnapshot(self._limit, self._used, self._reserved)

    def has_reservation(self, reservation: BudgetReservation | str) -> bool:
        reservation_id = self._reservation_id(reservation)
        with self._lock:
            return reservation_id in self._reservations

    @staticmethod
    def _reservation_id(reservation: BudgetReservation | str) -> str:
        if isinstance(reservation, BudgetReservation):
            return reservation.reservation_id
        if isinstance(reservation, str) and reservation:
            return reservation
        raise TypeError("reservation must be a BudgetReservation or non-empty id")

    def _require_within_limit(self, usage: BudgetUsage) -> None:
        exceeded = _exceeded_resources(usage, self._limit)
        if exceeded:
            raise BudgetExceededError("budget exceeded: " + ", ".join(exceeded))
