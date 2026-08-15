"""Metrics with bootstrap confidence intervals for rigorous ML evaluation.

All metrics return point estimate + 95% CI by default, computed via bootstrap
resampling (default n=1000). This is the standard for any new layer claiming
to improve the pipeline.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from mlcore.utils import brier, logloss


@dataclass(frozen=True)
class MetricsCI:
    """A metric with bootstrap confidence interval."""

    name: str
    point: float
    ci_low: float
    ci_high: float
    n: int
    n_bootstrap: int

    def as_dict(self) -> dict:
        return {
            "metric": self.name,
            "point": float(self.point),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "n": int(self.n),
            "n_bootstrap": int(self.n_bootstrap),
        }


def ece_score(y: Sequence[int], p: Sequence[float], *, bins: int = 10) -> float:
    """Expected Calibration Error (ECE) — lower is better."""
    if not y or not p or len(y) != len(p):
        return float("nan")
    n = len(y)
    b = max(2, int(bins))
    acc = 0.0
    for i in range(b):
        lo = i / b
        hi = (i + 1) / b
        idx = [j for j, pp in enumerate(p) if (pp >= lo and (pp < hi if i < b - 1 else pp <= hi))]
        if not idx:
            continue
        py = sum(float(y[j]) for j in idx) / float(len(idx))
        pp = sum(float(p[j]) for j in idx) / float(len(idx))
        acc += (float(len(idx)) / float(n)) * abs(py - pp)
    return float(acc)


def bootstrap_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    metric_fn,
    *,
    n_bootstrap: int = 1000,
    ci_pct: float = 95.0,
    seed: int = 1337,
) -> Tuple[float, float, float]:
    """Bootstrap a metric over (values_a, values_b) pairs.

    metric_fn must accept (a_sample, b_sample) and return float.

    Returns
    -------
    (point_estimate, ci_low, ci_high)
    """
    if not values_a or len(values_a) != len(values_b):
        return float("nan"), float("nan"), float("nan")
    n = len(values_a)
    point = float(metric_fn(list(values_a), list(values_b)))
    if n_bootstrap <= 0:
        return point, point, point

    rng = random.Random(int(seed))
    samples: List[float] = []
    for _ in range(int(n_bootstrap)):
        idx = [rng.randrange(0, n) for _ in range(n)]
        a_resample = [values_a[i] for i in idx]
        b_resample = [values_b[i] for i in idx]
        try:
            samples.append(float(metric_fn(a_resample, b_resample)))
        except (ValueError, ZeroDivisionError):
            continue
    if not samples:
        return point, float("nan"), float("nan")
    samples.sort()
    alpha = (100.0 - float(ci_pct)) / 2.0 / 100.0
    lo_idx = max(0, int(alpha * len(samples)))
    hi_idx = min(len(samples) - 1, int((1.0 - alpha) * len(samples)))
    return point, float(samples[lo_idx]), float(samples[hi_idx])


def metrics_with_ci(
    y: Sequence[int],
    p: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    ci_pct: float = 95.0,
    ece_bins: int = 10,
    seed: int = 1337,
) -> dict:
    """Compute logloss, ECE, brier with bootstrap CIs.

    Returns a dict of MetricsCI objects (one per metric) plus metadata.
    """
    n = len(y) if y else 0
    if n == 0 or n != len(p):
        return {"error": "invalid_inputs", "n": n}

    ll_point, ll_lo, ll_hi = bootstrap_ci(
        y, p, lambda a, b: float(logloss(a, b)),
        n_bootstrap=n_bootstrap, ci_pct=ci_pct, seed=seed,
    )
    br_point, br_lo, br_hi = bootstrap_ci(
        y, p, lambda a, b: float(brier(a, b)),
        n_bootstrap=n_bootstrap, ci_pct=ci_pct, seed=seed + 1,
    )
    ece_point, ece_lo, ece_hi = bootstrap_ci(
        y, p, lambda a, b: float(ece_score(a, b, bins=ece_bins)),
        n_bootstrap=n_bootstrap, ci_pct=ci_pct, seed=seed + 2,
    )

    return {
        "n": n,
        "n_bootstrap": int(n_bootstrap),
        "ci_pct": float(ci_pct),
        "logloss": MetricsCI("logloss", ll_point, ll_lo, ll_hi, n, n_bootstrap),
        "brier": MetricsCI("brier", br_point, br_lo, br_hi, n, n_bootstrap),
        "ece": MetricsCI("ece", ece_point, ece_lo, ece_hi, n, n_bootstrap),
    }


def paired_bootstrap_delta(
    y: Sequence[int],
    p_baseline: Sequence[float],
    p_candidate: Sequence[float],
    metric_fn,
    *,
    n_bootstrap: int = 1000,
    ci_pct: float = 95.0,
    seed: int = 1337,
) -> Tuple[float, float, float, float]:
    """Paired bootstrap for delta = metric(baseline) - metric(candidate).

    Positive delta = candidate improves over baseline (assuming metric_fn is lower-is-better).

    Returns
    -------
    (delta_point, ci_low, ci_high, p_value_two_sided)
    """
    n = len(y)
    if n == 0 or n != len(p_baseline) or n != len(p_candidate):
        return float("nan"), float("nan"), float("nan"), 1.0

    point_baseline = float(metric_fn(y, p_baseline))
    point_candidate = float(metric_fn(y, p_candidate))
    delta_point = point_baseline - point_candidate

    rng = random.Random(int(seed))
    deltas: List[float] = []
    for _ in range(int(n_bootstrap)):
        idx = [rng.randrange(0, n) for _ in range(n)]
        y_r = [y[i] for i in idx]
        p_b_r = [p_baseline[i] for i in idx]
        p_c_r = [p_candidate[i] for i in idx]
        try:
            d = float(metric_fn(y_r, p_b_r)) - float(metric_fn(y_r, p_c_r))
            deltas.append(d)
        except (ValueError, ZeroDivisionError):
            continue
    if not deltas:
        return delta_point, float("nan"), float("nan"), 1.0
    deltas.sort()
    alpha = (100.0 - float(ci_pct)) / 2.0 / 100.0
    lo = float(deltas[max(0, int(alpha * len(deltas)))])
    hi = float(deltas[min(len(deltas) - 1, int((1.0 - alpha) * len(deltas)))])
    # Two-sided p-value: fraction of bootstrap deltas with opposite sign or zero
    if delta_point >= 0:
        adverse = sum(1 for d in deltas if d <= 0)
    else:
        adverse = sum(1 for d in deltas if d >= 0)
    p_value = 2.0 * float(adverse) / float(len(deltas))
    p_value = min(1.0, max(0.0, p_value))
    return float(delta_point), lo, hi, float(p_value)
