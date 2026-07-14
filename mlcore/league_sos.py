"""Strength-of-schedule (SOS) adjusted league strength via Bradley-Terry.

The raw cross-league win rate is confounded by *who* each league plays (a tier-2
league that only meets weak teams looks strong). Bradley-Terry — the batch-MLE
counterpart of Elo — controls for opponent strength: it fits one strength per
league from the interleague results graph.

These league strengths become the model's data-derived `league_base_points`
(stored in the artifact), replacing the hand-set static priors. They are fit
together with the model so the learned per-league offset stays consistent (no
double counting) and they auto-update as more interleague games accrue.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


def fit_league_bt(
    pairs: Iterable[Tuple[str, str, float]],
    *,
    l2: float = 1e-3,
    iters: int = 4000,
    lr: float = 0.5,
) -> Dict[str, float]:
    """Bradley-Terry league strengths (centered) from interleague results.

    pairs: iterable of (league_a, league_b, a_win) with a_win in {0.0, 1.0}.
    Returns {league: theta} in natural-logit units, mean-centered. Empty if no data.
    """
    rows: List[Tuple[str, str, float]] = [
        (str(a), str(b), float(y)) for (a, b, y) in pairs
        if str(a) and str(b) and str(a) != str(b)
    ]
    if not rows:
        return {}
    leagues = sorted({a for a, _, _ in rows} | {b for _, b, _ in rows})
    idx = {lg: i for i, lg in enumerate(leagues)}
    A = np.zeros((len(rows), len(leagues)), dtype=float)
    y = np.zeros(len(rows), dtype=float)
    for k, (a, b, win) in enumerate(rows):
        A[k, idx[a]] += 1.0
        A[k, idx[b]] -= 1.0
        y[k] = win
    theta = np.zeros(len(leagues), dtype=float)
    n = max(1, len(rows))
    for _ in range(int(iters)):
        p = 1.0 / (1.0 + np.exp(-(A @ theta)))
        theta -= lr * (A.T @ (p - y) / n + l2 * theta)
    theta -= float(theta.mean())
    return {lg: float(theta[idx[lg]]) for lg in leagues}


def sos_league_base_points(
    pairs: Iterable[Tuple[str, str, float]],
    *,
    scale: float = 800.0,
    min_games: int = 20,
    l2: float = 1e-3,
) -> Dict[str, float]:
    """Data-derived league base points = scale * BT theta.

    Only leagues with >= min_games interleague games are returned (others keep
    the global static prior via the model fallback). `scale` maps logit units to
    the points scale; 800 was the validated value (interleague OOS logloss).
    """
    pair_list = [
        (str(a), str(b), float(y)) for (a, b, y) in pairs
        if str(a) and str(b) and str(a) != str(b)
    ]
    counts: Dict[str, int] = {}
    for a, b, _ in pair_list:
        counts[a] = counts.get(a, 0) + 1
        counts[b] = counts.get(b, 0) + 1
    theta = fit_league_bt(pair_list, l2=l2)
    return {
        lg: float(scale) * t
        for lg, t in theta.items()
        if counts.get(lg, 0) >= int(min_games)
    }
