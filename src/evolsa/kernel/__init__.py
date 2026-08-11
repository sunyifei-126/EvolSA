"""Immutable trusted-kernel primitives for bounded self-improvement."""

from .budget import (
    BudgetError,
    BudgetExceededError,
    BudgetLedger,
    BudgetReservation,
    BudgetSnapshot,
    UnknownReservationError,
)
from .events import EVENT_SCHEMA, GENESIS_HASH, EventLog, EventLogError, EventLogIntegrityError, EventRecord
from .sandbox import SandboxLimits, SandboxViolation, SubprocessSandbox
from .store import (
    ContentAddressedStore,
    ContentHashMismatchError,
    ContentStoreError,
    UnsafeContentPathError,
)
from .trust import (
    EffectAuthorizationError,
    PassportIntegrityError,
    ReleaseAuthorizationError,
    TrustDecision,
    TrustKernel,
    TrustKernelError,
    sign_passport,
)

__all__ = [name for name in globals() if not name.startswith("_")]
