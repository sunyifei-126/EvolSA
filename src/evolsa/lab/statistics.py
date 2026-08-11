"""Small deterministic statistical routines for paired operator trials."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Iterator, Sequence

from evolsa.core import PairedOutcome


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A percentile bootstrap interval over paired score differences."""

    mean: float
    lower: float
    upper: float
    confidence: float
    samples: int
    seed: int

    @property
    def lcb95(self) -> float:
        return self.lower

    @property
    def ucb95(self) -> float:
        return self.upper

    def __iter__(self) -> Iterator[float]:
        """Allow ``lower, upper = interval`` without losing metadata."""

        yield self.lower
        yield self.upper


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("a percentile requires at least one value")
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


def _paired_lifts(
    outcomes_or_candidate: Iterable[PairedOutcome] | Iterable[float],
    baseline_scores: Iterable[float] | None,
) -> tuple[float, ...]:
    values = tuple(outcomes_or_candidate)
    if baseline_scores is None:
        lifts = tuple(item.lift if isinstance(item, PairedOutcome) else float(item) for item in values)
    else:
        candidates = tuple(float(item) for item in values)
        baselines = tuple(float(item) for item in baseline_scores)
        if len(candidates) != len(baselines):
            raise ValueError("candidate and baseline score sequences must have equal length")
        lifts = tuple(candidate - baseline for candidate, baseline in zip(candidates, baselines, strict=True))
    if not lifts:
        raise ValueError("paired bootstrap requires at least one outcome")
    if not all(math.isfinite(value) for value in lifts):
        raise ValueError("paired bootstrap values must be finite")
    return lifts


def paired_bootstrap_ci(
    outcomes_or_candidate: Iterable[PairedOutcome] | Iterable[float],
    baseline_scores: Iterable[float] | None = None,
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 0,
) -> BootstrapInterval:
    """Return a deterministic percentile CI for the mean paired lift.

    Resampling is over *pairs*, never independently over candidate and baseline
    observations.  Passing the same data, sample count, and seed therefore
    produces byte-for-byte identical report metrics.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    if samples < 1:
        raise ValueError("samples must be positive")
    lifts = _paired_lifts(outcomes_or_candidate, baseline_scores)
    mean = fmean(lifts)
    if len(lifts) == 1:
        return BootstrapInterval(mean, mean, mean, confidence, samples, seed)

    random_source = random.Random(seed)
    size = len(lifts)
    boot_means = [
        fmean(lifts[random_source.randrange(size)] for _ in range(size))
        for _ in range(samples)
    ]
    boot_means.sort()
    tail = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        mean=mean,
        lower=_percentile(boot_means, tail),
        upper=_percentile(boot_means, 1.0 - tail),
        confidence=confidence,
        samples=samples,
        seed=seed,
    )


bootstrap_paired_ci = paired_bootstrap_ci
paired_bootstrap = paired_bootstrap_ci


__all__ = [
    "BootstrapInterval",
    "bootstrap_paired_ci",
    "paired_bootstrap",
    "paired_bootstrap_ci",
]
