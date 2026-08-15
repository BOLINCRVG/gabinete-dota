"""Pure computation for ML micro-segment calibration residuals."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from mlcore.utils import clip_prob, safe_float_or_none as _fnum
from app_core.ml_league_runtime_core import tier_from_league as _tier_from_league
from app_core.ml_probability_adjustments_core import seg_key as _seg_key


def _estimate_shrink_k(
    agg_specific: Dict[str, Dict[str, float]],
    *,
    k_min: float = 25.0,
    k_max: float = 300.0,
    min_segments: int = 8,
    min_n_per_segment: float = 15.0,
) -> float:
    """Empirical Bayes k = sigma_within^2 / sigma_true^2 (James-Stein).

    Estimates the optimal shrinkage constant from observed segment-level biases.
    Falls back to k_max (max shrinkage) when there are too few segments or
    when segment biases are indistinguishable from sampling noise.
    """
    mus: List[float] = []
    ns: List[float] = []
    for d in agg_specific.values():
        n = float(d.get("n", 0.0))
        if n < min_n_per_segment:
            continue
        mus.append(float(d.get("sum_err", 0.0)) / n)
        ns.append(n)
    if len(mus) < min_segments:
        return float(k_max)
    mu_arr = np.asarray(mus, dtype=float)
    n_arr = np.asarray(ns, dtype=float)
    sigma_within_sq = float(np.mean(0.25 / n_arr))
    sigma_total_sq = float(np.var(mu_arr, ddof=1)) if len(mus) > 1 else 0.0
    sigma_true_sq = max(0.0, sigma_total_sq - sigma_within_sq)
    if sigma_true_sq < 1e-9:
        return float(k_max)
    return float(max(k_min, min(k_max, sigma_within_sq / sigma_true_sq)))


def _extract_fitted_k(
    fitted_microseg_table: Optional[Dict[str, Dict[str, float]]],
) -> Optional[float]:
    """Return the shrink_k stored in any segment of the fitted table, or None."""
    if not isinstance(fitted_microseg_table, dict) or not fitted_microseg_table:
        return None
    for v in fitted_microseg_table.values():
        if not isinstance(v, dict):
            continue
        k = _fnum(v.get("shrink_k"))
        if k is not None and k > 0.0:
            return float(k)
    return None


def build_ml_microseg_table(
    team_games: Optional[pd.DataFrame],
    art: Any,
    *,
    lookback_games: int = 1200,
    shrink_k: float = 90.0,
    fitted_microseg_table: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Build micro-segment calibration residuals with hierarchical fallback buckets.

    Segment hierarchy (for Blue-side probability):
    1) league + split + side + bo + phase
    2) league + * + side + bo + phase
    3) league + * + side + * + phase
    4) league + * + * + * + phase
    5) league + * + * + * + *
    6) * + * + * + * + *

    When fitted_microseg_table is provided its stored shrink_k is blended with
    the runtime EB estimate via geometric mean, anchoring the live shrinkage to
    the fit-time evidence and reducing variance from short live windows.
    """
    out: Dict[str, Dict[str, float]] = {}
    try:
        if team_games is None or not isinstance(team_games, pd.DataFrame) or team_games.empty:
            return out
        if not {"side", "team", "opponent", "win"}.issubset(set(team_games.columns)):
            return out

        df = team_games.copy()
        df["side"] = df["side"].astype(str).str.lower()
        df = df[df["side"] == "blue"].copy()
        if df.empty:
            return out
        if "date" in df.columns:
            df = df.sort_values("date")
        df = df.tail(int(max(200, int(lookback_games))))
        if "game" in df.columns:
            _g = pd.to_numeric(df["game"], errors="coerce")
            df["_game_num"] = _g.where(_g >= 1, 1).fillna(1).astype(int)
        else:
            df["_game_num"] = 1
        if "date" in df.columns:
            _d = pd.to_datetime(df["date"], errors="coerce")
            df["_date_day"] = _d.dt.date.astype(str)
        else:
            df["_date_day"] = ""

        _grp_cols = ["_date_day", "league", "split", "playoffs", "team", "opponent"]
        for _gc in _grp_cols:
            if _gc not in df.columns:
                df[_gc] = ""
        df["_series_max_game"] = df.groupby(_grp_cols, dropna=False)["_game_num"].transform("max")
        df["_bo_bucket"] = np.where(
            df["_series_max_game"] >= 5, "5",
            np.where(df["_series_max_game"] >= 3, "3", "1"),
        )
        _po = pd.to_numeric(df.get("playoffs"), errors="coerce").fillna(0).astype(int)
        df["_phase"] = np.where(_po > 0, "playoffs", "regular")

        model = getattr(art, "model", None)
        cal = getattr(art, "calibrator", None)
        predict_blue = getattr(model, "p_blue", None) if model is not None else None
        predict_cal = getattr(cal, "predict_proba", None) if cal is not None else None
        if not callable(predict_blue):
            return out

        agg: Dict[str, Dict[str, float]] = {}
        agg_specific: Dict[str, Dict[str, float]] = {}  # level-0 keys only, for k estimation

        for r in df.itertuples(index=False):
            b = str(getattr(r, "team", "") or "").strip()
            rr = str(getattr(r, "opponent", "") or "").strip()
            lg = str(getattr(r, "league", "") or "").strip()
            sp = str(getattr(r, "split", "") or "").strip()
            sd = "blue"
            bo_bucket = str(getattr(r, "_bo_bucket", "") or "").strip()
            phase = str(getattr(r, "_phase", "") or "").strip().lower()
            if not b or not rr:
                continue
            y = _fnum(getattr(r, "win", 0)) or 0.0

            tier = _tier_from_league(lg) if lg else "OTHER"
            try:
                p_raw = float(
                    predict_blue(
                        b, rr,
                        blue_league=(lg or None),
                        red_league=(lg or None),
                        blue_tier=tier,
                        red_tier=tier,
                    )
                )
                p_raw = float(clip_prob(p_raw))
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue
            try:
                p_cal = float(clip_prob(float(predict_cal(p_raw)))) if callable(predict_cal) else float(p_raw)
            except (AttributeError, TypeError, ValueError, OverflowError):
                p_cal = float(p_raw)

            err = float(y - p_cal)
            k0 = _seg_key(lg, sp, sd, bo_bucket, phase)
            keys = [
                k0,
                _seg_key(lg, "*", sd, bo_bucket, phase),
                _seg_key(lg, "*", sd, "*", phase),
                _seg_key(lg, "*", "*", "*", phase),
                _seg_key(lg, "*", "*", "*", "*"),
                _seg_key("*", "*", "*", "*", "*"),
            ]
            for k in keys:
                d = agg.setdefault(k, {"n": 0.0, "sum_err": 0.0})
                d["n"] += 1.0
                d["sum_err"] += float(err)
            ds = agg_specific.setdefault(k0, {"n": 0.0, "sum_err": 0.0})
            ds["n"] += 1.0
            ds["sum_err"] += float(err)

        sk_runtime = float(max(1.0, _estimate_shrink_k(agg_specific)))
        k_fitted = _extract_fitted_k(fitted_microseg_table)
        if k_fitted is not None and k_fitted > 0.0:
            # Blend runtime EB estimate with fit-time estimate via geometric mean.
            # This anchors live shrinkage to the signal quality measured at fit time
            # and reduces variance when the live window is short.
            sk = float(max(1.0, (sk_runtime * k_fitted) ** 0.5))
        else:
            sk = sk_runtime
        for k, d in agg.items():
            n = float(d.get("n", 0.0))
            if n <= 0:
                continue
            raw_bias = float(d.get("sum_err", 0.0)) / n
            shrink = n / (n + sk)
            bias = float(raw_bias * shrink)
            out[k] = {"n": n, "raw_bias": raw_bias, "bias": bias, "shrink_k": float(sk)}
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return {}
    return out


__all__ = ["build_ml_microseg_table"]
