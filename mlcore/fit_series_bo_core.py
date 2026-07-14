"""Walk-forward OOS fitting for series-BO gamma parameters.

The existing _fit_series_gamma (fit_series_core.py) evaluates gamma in-sample
over all available rows. This module wraps the same building blocks with a
proper temporal train/test split so the reported improvement is genuinely OOS
and cannot overfit to the evaluated data.

Fitting strategy
----------------
1. Build series rows from map-level Sample objects, preserving date_day.
2. Sort by date_day (earliest-first); split: first (1 - oos_frac) → train,
   last oos_frac → holdout test.
3. Grid-search gamma on train rows; pick the gamma that minimises train logloss.
4. Evaluate the winning gamma on the held-out test rows.
5. Gate: if OOS improvement vs gamma=1.0 < min_oos_gain → shadow_only.
6. Return gate decision, OOS metrics, and full grid for diagnostics.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .bo import prob_win_series
from .fit_form_core import _apply_form_blend_on_sample
from .fit_samples_core import Sample, _fuse_probs_by_precision
from .utils import brier, clip_prob, logit, logloss, sigmoid


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_GAMMAS = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00)


def _series_prob(p_map: float, bo: int, gamma: float) -> float:
    p0 = float(clip_prob(float(prob_win_series(float(p_map), int(bo)))))
    g = max(0.25, float(gamma))
    z = float(logit(p0))
    return float(clip_prob(float(sigmoid(z * g))))


def _eval_gamma(rows: List[Dict[str, Any]], *, bo: int, gamma: float) -> Dict[str, Any]:
    rb = [r for r in rows if int(r.get("bo", 0)) == int(bo)]
    if not rb:
        return {"error": "no_rows", "n": 0}
    y = [int(r["y"]) for r in rb]
    p = [_series_prob(float(r["p_map"]), int(bo), float(gamma)) for r in rb]
    try:
        ll = float(logloss(y, p))
        br = float(brier(y, p))
    except Exception:
        return {"error": "metric_error", "n": len(rb)}
    return {"gamma": float(gamma), "logloss": float(ll), "brier": float(br), "n": len(rb)}


def _build_gamma_rows(
    samples: List[Sample],
    *,
    team_scale: float,
    players_scale: float,
    form_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build one row per completed BO3/BO5 series, preserving date_day for sorting."""
    groups: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        ta = str(s.blue_team or "").strip()
        tb = str(s.red_team or "").strip()
        if not ta or not tb:
            continue
        sd = str(s.date_day or "").strip()
        lg = str(s.league or "").strip()
        sp = str(s.split or "").strip()
        ph = str(s.phase or "").strip().lower()
        u1, u2 = sorted([ta, tb])
        key = f"{lg}|{sp}|{ph}|{sd}|{u1}|{u2}"
        groups[key].append(s)

    out: List[Dict[str, Any]] = []
    for _key, maps in groups.items():
        maps = sorted(maps, key=lambda x: int(getattr(x, "game_no", 1) or 1))
        bo = int(max(int(getattr(m, "game_no", 1) or 1) for m in maps))
        if bo not in (3, 5):
            continue
        wins: Dict[str, int] = defaultdict(int)
        for m in maps:
            if int(getattr(m, "y", 0) or 0) == 1:
                wins[str(getattr(m, "blue_team", ""))] += 1
            else:
                wins[str(getattr(m, "red_team", ""))] += 1
        if not wins:
            continue
        m1 = maps[0]
        first_blue = str(getattr(m1, "blue_team", "") or "")
        winner = max(wins.items(), key=lambda kv: kv[1])[0]
        y_series = 1 if winner == first_blue else 0

        p_pre = _fuse_probs_by_precision(
            p_team=float(getattr(m1, "p_team", 0.5)),
            p_players=float(getattr(m1, "p_players", 0.5)),
            n_team=float(getattr(m1, "n_team", 0.0)),
            n_players=float(getattr(m1, "n_players", 0.0)),
            team_scale=float(team_scale),
            players_scale=float(players_scale),
            coverage=float(getattr(m1, "coverage_ratio", 0.0) or 0.0),
            transfer_signal=float(getattr(m1, "transfer_signal", 0.0) or 0.0),
            lineup_continuity=float(getattr(m1, "lineup_continuity", 1.0) or 1.0),
            roster_changed=bool(getattr(m1, "roster_changed", False)),
            season_phase=("playoffs" if str(getattr(m1, "phase", "")).strip().lower() == "playoffs" else "auto"),
            players_freshness=float(getattr(m1, "freshness_avg", 0.0) or 0.0),
        )
        p_map = _apply_form_blend_on_sample(
            p_model=float(p_pre),
            sample=m1,
            enabled=bool((form_cfg or {}).get("enabled", False)),
            max_weight=float((form_cfg or {}).get("max_weight", 0.75) or 0.75),
            knee_games=float((form_cfg or {}).get("knee_games", 6.0) or 6.0),
            beta_prior=float((form_cfg or {}).get("beta_prior", 2.0) or 2.0),
            signal_power=float((form_cfg or {}).get("signal_power", 1.0) or 1.0),
            elite_same_league_parity_strength=float(
                (form_cfg or {}).get("elite_same_league_parity_strength", 0.35) or 0.35
            ),
        )
        out.append(
            {
                "bo": int(bo),
                "y": int(y_series),
                "p_map": float(clip_prob(float(p_map))),
                "date_day": str(getattr(m1, "date_day", "") or ""),
                "league": str(getattr(m1, "league", "") or ""),
            }
        )
    return out


def _split_rows_temporal(
    rows: List[Dict[str, Any]],
    oos_frac: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split rows chronologically into (train, test) by date_day."""
    sorted_rows = sorted(rows, key=lambda r: str(r.get("date_day") or ""))
    n = len(sorted_rows)
    split = max(1, round(n * (1.0 - float(oos_frac))))
    return sorted_rows[:split], sorted_rows[split:]


def _fit_gamma_on_rows(
    train: List[Dict[str, Any]],
    *,
    bo: int,
    gammas: Tuple[float, ...],
) -> Optional[float]:
    """Return the gamma that minimises logloss on the given rows, or None."""
    best_ll = float("inf")
    best_g: Optional[float] = None
    for g in gammas:
        m = _eval_gamma(train, bo=int(bo), gamma=float(g))
        if m.get("error") or not math.isfinite(float(m.get("logloss", float("inf")))):
            continue
        if float(m["logloss"]) < best_ll:
            best_ll = float(m["logloss"])
            best_g = float(g)
    return best_g


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_series_gamma_oos(
    samples: List[Sample],
    *,
    team_scale: float,
    players_scale: float,
    form_cfg: Dict[str, Any],
    gammas: Tuple[float, ...] = _DEFAULT_GAMMAS,
    oos_frac: float = 0.25,
    min_train_n: int = 40,
    min_test_n: int = 20,
    min_oos_gain: float = 0.0002,
) -> Dict[str, Any]:
    """Fit series gamma with a temporal train/test split and gate decision.

    Parameters
    ----------
    samples :
        All Sample objects (map-level). Series are reconstructed from these.
    team_scale, players_scale, form_cfg :
        Same parameters passed to _fuse_probs_by_precision and form blend.
    gammas :
        Grid of gamma values to search. 1.0 is the baseline (no amplification).
    oos_frac :
        Fraction of rows (by date) held out as the OOS test set.
    min_train_n :
        Minimum series in the training set to attempt fitting.
    min_test_n :
        Minimum series in the test set to evaluate OOS improvement.
    min_oos_gain :
        Minimum logloss improvement over gamma=1.0 on OOS set to gate promote.

    Returns
    -------
    Dict with keys:
    - gate: {decision: "promote"|"shadow_only", reasons: [...]}
    - bo3 / bo5: {gamma, train_logloss, oos_logloss, oos_baseline_logloss,
                  oos_gain, n_train, n_test}
    - grid: full train-side grid per BO
    """
    all_rows = _build_gamma_rows(
        samples, team_scale=float(team_scale), players_scale=float(players_scale),
        form_cfg=dict(form_cfg or {}),
    )
    if not all_rows:
        return {
            "gate": {"decision": "shadow_only", "reasons": ["no_series_rows"]},
            "n_rows_total": 0,
        }

    result: Dict[str, Any] = {
        "gate": {"decision": "promote", "reasons": []},
        "n_rows_total": len(all_rows),
        "oos_frac": float(oos_frac),
        "gammas_searched": list(gammas),
    }
    gate_reasons: List[str] = []
    any_bo_fitted = False

    for bo in (3, 5):
        bo_rows = [r for r in all_rows if int(r.get("bo", 0)) == int(bo)]
        if not bo_rows:
            continue
        train, test = _split_rows_temporal(bo_rows, float(oos_frac))
        n_train = len(train)
        n_test = len(test)

        bo_key = f"bo{bo}"
        if n_train < int(min_train_n):
            result[bo_key] = {
                "gate_reason": "low_train_n",
                "n_train": n_train,
                "n_test": n_test,
                "n_total": len(bo_rows),
            }
            gate_reasons.append(f"bo{bo}_low_train_n:{n_train}")
            continue

        best_gamma = _fit_gamma_on_rows(train, bo=int(bo), gammas=tuple(gammas))
        if best_gamma is None:
            result[bo_key] = {"gate_reason": "fit_failed", "n_train": n_train, "n_test": n_test}
            gate_reasons.append(f"bo{bo}_fit_failed")
            continue

        train_metrics = _eval_gamma(train, bo=int(bo), gamma=float(best_gamma))

        # Build grid (train only) for diagnostics
        grid = []
        for g in gammas:
            gm = _eval_gamma(train, bo=int(bo), gamma=float(g))
            if not gm.get("error"):
                grid.append({"gamma": float(g), "train_logloss": float(gm.get("logloss", 9.0))})

        if n_test < int(min_test_n):
            result[bo_key] = {
                "gamma": float(best_gamma),
                "gate_reason": "low_test_n_for_oos",
                "train_logloss": float(train_metrics.get("logloss", float("nan"))),
                "n_train": n_train,
                "n_test": n_test,
                "grid": grid,
            }
            gate_reasons.append(f"bo{bo}_low_test_n:{n_test}")
            any_bo_fitted = True
            continue

        oos = _eval_gamma(test, bo=int(bo), gamma=float(best_gamma))
        oos_baseline = _eval_gamma(test, bo=int(bo), gamma=1.0)

        if oos.get("error") or oos_baseline.get("error"):
            result[bo_key] = {"gate_reason": "oos_eval_error", "n_train": n_train, "n_test": n_test}
            gate_reasons.append(f"bo{bo}_oos_eval_error")
            continue

        oos_ll = float(oos.get("logloss", float("nan")))
        oos_base_ll = float(oos_baseline.get("logloss", float("nan")))
        oos_gain = (oos_base_ll - oos_ll) if (math.isfinite(oos_ll) and math.isfinite(oos_base_ll)) else float("nan")

        row: Dict[str, Any] = {
            "gamma": float(best_gamma),
            "train_logloss": float(train_metrics.get("logloss", float("nan"))),
            "oos_logloss": float(oos_ll),
            "oos_baseline_logloss": float(oos_base_ll),
            "oos_gain": (float(oos_gain) if math.isfinite(oos_gain) else None),
            "oos_brier": float(oos.get("brier", float("nan"))) if math.isfinite(float(oos.get("brier", float("nan")))) else None,
            "n_train": n_train,
            "n_test": n_test,
            "n_total": len(bo_rows),
            "grid": grid,
        }
        result[bo_key] = row
        any_bo_fitted = True

        if not math.isfinite(oos_gain) or oos_gain < float(min_oos_gain):
            gate_reasons.append(
                f"bo{bo}_oos_gain_small:{oos_gain:.4f}" if math.isfinite(oos_gain) else f"bo{bo}_oos_gain_not_finite"
            )
        if math.isfinite(oos_gain) and oos_gain < -0.0005:
            gate_reasons.append(f"bo{bo}_oos_regression:{oos_gain:.4f}")

    if not any_bo_fitted:
        gate_reasons.append("no_bo_fitted")

    if gate_reasons:
        result["gate"] = {"decision": "shadow_only", "reasons": gate_reasons}
    else:
        result["gate"] = {"decision": "promote", "reasons": ["pass"]}

    return result


__all__ = ["fit_series_gamma_oos"]
