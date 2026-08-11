"""Matched-budget operator experiments, statistics, and promotion gates."""

from .gate import PromotionGate
from .paired import (
    LabReportBuilder,
    PairedTrial,
    aggregate_lab_report,
    budgets_match,
    build_lab_report,
    validate_matched_budget,
)
from .statistics import BootstrapInterval, bootstrap_paired_ci, paired_bootstrap, paired_bootstrap_ci

__all__ = [name for name in globals() if not name.startswith("_")]
