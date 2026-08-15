"""Acceptance gate for ML layer changes.

A new layer is ACCEPTED if all of the following hold:
  - delta_logloss > min_delta_logloss (default 0.002)
  - p_value < alpha (default 0.05) for delta_logloss
  - ECE does not get worse (delta_ece >= -ece_tolerance, default 0.005)

Otherwise REJECTED with reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from mlcore.ml_eval.metrics import ece_score, paired_bootstrap_delta
from mlcore.utils import logloss


@dataclass(frozen=True)
class GateDecision:
    """Outcome of running a layer through the acceptance gate."""

    accept: bool
    reason: str
    delta_logloss: float
    delta_logloss_ci: tuple
    p_value: float
    delta_ece: float
    n: int
    n_bootstrap: int
    thresholds: dict

    def as_dict(self) -> dict:
        return {
            "accept": bool(self.accept),
            "reason": str(self.reason),
            "delta_logloss": float(self.delta_logloss),
            "delta_logloss_ci_low": float(self.delta_logloss_ci[0]),
            "delta_logloss_ci_high": float(self.delta_logloss_ci[1]),
            "p_value": float(self.p_value),
            "delta_ece": float(self.delta_ece),
            "n": int(self.n),
            "n_bootstrap": int(self.n_bootstrap),
            "thresholds": dict(self.thresholds),
        }

    def summary_line(self) -> str:
        sign = "+" if self.delta_logloss >= 0 else ""
        verdict = "ACCEPT" if self.accept else "REJECT"
        return (
            f"[{verdict}] delta_ll={sign}{self.delta_logloss*100:.3f}pp "
            f"CI[{self.delta_logloss_ci[0]*100:+.3f}, {self.delta_logloss_ci[1]*100:+.3f}]pp "
            f"p={self.p_value:.4f} delta_ece={self.delta_ece*100:+.3f}pp "
            f"reason={self.reason}"
        )


def acceptance_gate(
    y: Sequence[int],
    p_baseline: Sequence[float],
    p_candidate: Sequence[float],
    *,
    min_delta_logloss: float = 0.002,
    alpha: float = 0.05,
    ece_tolerance: float = 0.005,
    n_bootstrap: int = 1000,
    ci_pct: float = 95.0,
    ece_bins: int = 10,
    seed: int = 1337,
) -> GateDecision:
    """Run the acceptance gate on a candidate layer.

    Parameters
    ----------
    y : ground truth labels (0/1)
    p_baseline : probabilities WITHOUT the new layer
    p_candidate : probabilities WITH the new layer
    min_delta_logloss : minimum required improvement in logloss (default 0.002 = 0.2pp)
    alpha : significance level for the paired bootstrap p-value (default 0.05)
    ece_tolerance : max allowed worsening of ECE (default 0.005 = 0.5pp).
                    If ECE worsens by more than this, the layer is rejected even if
                    logloss improves.

    Returns
    -------
    GateDecision with accept flag and detailed reasoning.
    """
    n = len(y) if y else 0
    thresholds = {
        "min_delta_logloss": float(min_delta_logloss),
        "alpha": float(alpha),
        "ece_tolerance": float(ece_tolerance),
        "n_bootstrap": int(n_bootstrap),
    }

    if n == 0 or n != len(p_baseline) or n != len(p_candidate):
        return GateDecision(
            accept=False,
            reason="invalid_inputs",
            delta_logloss=float("nan"),
            delta_logloss_ci=(float("nan"), float("nan")),
            p_value=1.0,
            delta_ece=float("nan"),
            n=n,
            n_bootstrap=int(n_bootstrap),
            thresholds=thresholds,
        )

    # Paired bootstrap for delta_logloss + p-value + CI
    delta_ll, ci_lo, ci_hi, p_value = paired_bootstrap_delta(
        y, p_baseline, p_candidate,
        lambda yy, pp: float(logloss(yy, pp)),
        n_bootstrap=int(n_bootstrap),
        ci_pct=float(ci_pct),
        seed=int(seed),
    )

    # ECE delta (positive = improvement)
    ece_baseline = ece_score(y, p_baseline, bins=int(ece_bins))
    ece_candidate = ece_score(y, p_candidate, bins=int(ece_bins))
    delta_ece = float(ece_baseline - ece_candidate)

    # Apply gate criteria
    if delta_ll < float(min_delta_logloss):
        return GateDecision(
            accept=False,
            reason=f"insufficient_logloss_gain (delta={delta_ll*100:.3f}pp < min={min_delta_logloss*100:.3f}pp)",
            delta_logloss=delta_ll, delta_logloss_ci=(ci_lo, ci_hi),
            p_value=p_value, delta_ece=delta_ece, n=n,
            n_bootstrap=int(n_bootstrap), thresholds=thresholds,
        )

    if p_value >= float(alpha):
        return GateDecision(
            accept=False,
            reason=f"not_significant (p={p_value:.4f} >= alpha={alpha:.4f})",
            delta_logloss=delta_ll, delta_logloss_ci=(ci_lo, ci_hi),
            p_value=p_value, delta_ece=delta_ece, n=n,
            n_bootstrap=int(n_bootstrap), thresholds=thresholds,
        )

    if delta_ece < -float(ece_tolerance):
        return GateDecision(
            accept=False,
            reason=f"ece_worsened (delta_ece={delta_ece*100:.3f}pp < -tolerance={-ece_tolerance*100:.3f}pp)",
            delta_logloss=delta_ll, delta_logloss_ci=(ci_lo, ci_hi),
            p_value=p_value, delta_ece=delta_ece, n=n,
            n_bootstrap=int(n_bootstrap), thresholds=thresholds,
        )

    return GateDecision(
        accept=True,
        reason="passed_all_checks",
        delta_logloss=delta_ll, delta_logloss_ci=(ci_lo, ci_hi),
        p_value=p_value, delta_ece=delta_ece, n=n,
        n_bootstrap=int(n_bootstrap), thresholds=thresholds,
    )
