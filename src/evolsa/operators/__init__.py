"""Model-independent seed operators for bounded artifact evolution."""

from .runtime import (
    OPERATION_FAMILIES,
    OperationFamilyBinding,
    bind_operation_family,
    operation_family_for,
)
from .seed import (
    CrossoverOperator,
    DebugOperator,
    DraftOperator,
    ImproveOperator,
    SeedOperator,
    seed_operators,
)

# Short names are convenient when declaring an operator instruction set.
Draft = DraftOperator
Improve = ImproveOperator
Debug = DebugOperator
Crossover = CrossoverOperator

__all__ = [
    "OPERATION_FAMILIES",
    "OperationFamilyBinding",
    "Crossover",
    "CrossoverOperator",
    "Debug",
    "DebugOperator",
    "Draft",
    "DraftOperator",
    "Improve",
    "ImproveOperator",
    "SeedOperator",
    "bind_operation_family",
    "operation_family_for",
    "seed_operators",
]
