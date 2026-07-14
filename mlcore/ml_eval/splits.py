"""Chronological train/val/test holdout splits for ML evaluation.

The split is strictly time-ordered to prevent data leakage:
    train: full years (typically 2024 + 2025)
    val:   first half of the current season (used for hyperparameter tuning)
    test:  second half of the current season (HOLDOUT — never touched until final report)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class SampleBucket:
    """A single split bucket (train, val, or test)."""

    name: str
    samples: List[Any] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.samples)

    def date_range(self) -> Tuple[Optional[str], Optional[str]]:
        """Return (min_date, max_date) ISO strings; None if empty or no dates."""
        if not self.samples:
            return None, None
        dates = [str(getattr(s, "date_day", "") or "") for s in self.samples]
        dates = sorted([d for d in dates if d.strip()])
        return (dates[0], dates[-1]) if dates else (None, None)


@dataclass(frozen=True)
class HoldoutSplit:
    """Container for train/val/test buckets plus metadata."""

    train: SampleBucket
    val: SampleBucket
    test: SampleBucket
    cutoff_val_start: str
    cutoff_test_start: str

    def summary(self) -> dict:
        """Return a dict summary suitable for logging or report serialization."""
        return {
            "train": {
                "n": self.train.n,
                "date_range": self.train.date_range(),
            },
            "val": {
                "n": self.val.n,
                "date_range": self.val.date_range(),
                "cutoff_start": self.cutoff_val_start,
            },
            "test": {
                "n": self.test.n,
                "date_range": self.test.date_range(),
                "cutoff_start": self.cutoff_test_start,
            },
        }


def _sample_date(sample: Any) -> str:
    return str(getattr(sample, "date_day", "") or "").strip()


def _sample_year(sample: Any) -> Optional[int]:
    d = _sample_date(sample)
    if not d:
        return None
    try:
        return int(d[:4])
    except (ValueError, TypeError):
        return None


def build_split(
    samples: Sequence[Any],
    *,
    train_years: Sequence[int] = (2024, 2025),
    val_year: int = 2026,
    val_cutoff: str = "2026-06-30",
) -> HoldoutSplit:
    """Build a chronological train/val/test split.

    Parameters
    ----------
    samples : sequence of sample-like objects with `date_day` attribute (ISO string)
    train_years : years whose samples go entirely into TRAIN
    val_year : year split into val (first half) and test (second half) by val_cutoff
    val_cutoff : ISO date string. Samples in val_year with date <= cutoff go to VAL,
                 samples with date > cutoff go to TEST.

    Returns
    -------
    HoldoutSplit with train, val, test buckets and metadata.
    """
    train_set = set(int(y) for y in train_years)
    cutoff = str(val_cutoff).strip()
    if not cutoff:
        raise ValueError("val_cutoff must be a non-empty ISO date string")
    try:
        pd.Timestamp(cutoff)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"val_cutoff is not a valid ISO date: {cutoff}") from exc

    train: List[Any] = []
    val: List[Any] = []
    test: List[Any] = []

    val_year_int = int(val_year)

    for s in samples:
        d = _sample_date(s)
        y = _sample_year(s)
        if y is None or not d:
            continue
        if y in train_set:
            train.append(s)
        elif y == val_year_int:
            if d <= cutoff:
                val.append(s)
            else:
                test.append(s)
        elif y < min(train_set, default=val_year_int):
            train.append(s)
        # samples with y > val_year_int are ignored (future / unsupported)

    # Compute test cutoff start = cutoff + 1 day
    try:
        test_start = (pd.Timestamp(cutoff) + pd.Timedelta(days=1)).date().isoformat()
    except (ValueError, TypeError):
        test_start = cutoff

    return HoldoutSplit(
        train=SampleBucket(name="train", samples=train),
        val=SampleBucket(name="val", samples=val),
        test=SampleBucket(name="test", samples=test),
        cutoff_val_start=f"{val_year_int}-01-01",
        cutoff_test_start=test_start,
    )
