from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from .utils import brier, clip_prob, logloss
from .fit_samples_core import (
    Sample,
    _ece_metric,
    _fuse_probs_by_precision,
)


def _apply_form_blend_on_sample(
    *,
    p_model: float,
    sample: Sample,
    enabled: bool,
    max_weight: float,
    knee_games: float,
    beta_prior: float,
    signal_power: float,
    elite_same_league_parity_strength: float = 0.35,
) -> float:
    pm = float(clip_prob(float(p_model)))
    if not bool(enabled):
        return pm
    n_a = int(max(0, int(sample.form_n_a)))
    n_b = int(max(0, int(sample.form_n_b)))
    if n_a <= 0 or n_b <= 0:
        return pm

    wins_a = int(max(0, int(sample.form_wins_a)))
    wins_b = int(max(0, int(sample.form_wins_b)))
    bp = max(0.5, float(beta_prior))
    p_a = float(clip_prob((float(wins_a) + bp) / (float(n_a) + 2.0 * bp)))
    p_b = float(clip_prob((float(wins_b) + bp) / (float(n_b) + 2.0 * bp)))
    logit_diff = math.log(p_a / (1.0 - p_a)) - math.log(p_b / (1.0 - p_b))

    if bool(sample.same_league):
        p_weak = min(float(p_a), float(p_b))
        if p_weak >= 0.60:
            parity_signal = max(0.0, min(1.0, (p_weak - 0.60) / 0.25))
            parity_strength = max(0.0, min(0.80, float(elite_same_league_parity_strength)))
            logit_diff = float(logit_diff) * float(1.0 - (parity_strength * parity_signal))

    p_form = float(1.0 / (1.0 + math.exp(-logit_diff)))
    n_min = float(min(n_a, n_b))
    knee = max(1.0, float(knee_games))
    sample_factor = n_min / (n_min + knee)
    sig = abs(p_form - 0.5) / 0.5
    sig = max(0.0, min(1.0, sig))
    sp = max(0.1, float(signal_power))
    signal_factor = sig ** sp
    w_cap = max(0.0, min(0.95, float(max_weight)))
    w_form = max(0.0, min(0.95, float(w_cap * sample_factor * signal_factor)))
    return float(clip_prob((1.0 - w_form) * pm + w_form * p_form))


def _evaluate_walkforward(
    samples: List[Sample],
    *,
    team_scale: float,
    players_scale: float,
    min_train_games: int,
    league_filter: str | None = None,
    macro_filter: str | None = None,
    form_cfg: Dict[str, Any] | None = None,
    return_rows: bool = False,
) -> Dict[str, Any]:
    if league_filter:
        lg = str(league_filter)
        samples = [s for s in samples if str(s.league) == lg]
    if macro_filter:
        mk = str(macro_filter)
        samples = [s for s in samples if str(s.macro) == mk]
    if not samples:
        return {"folds": [], "logloss": None, "brier": None, "n_total": 0}

    months = sorted({s.month for s in samples})
    folds: List[Dict[str, Any]] = []
    all_y: List[int] = []
    all_p: List[float] = []
    pred_rows: List[Dict[str, Any]] = []

    for i in range(1, len(months)):
        m = months[i]
        train_n = sum(1 for s in samples if s.month < m)
        if train_n < int(min_train_games):
            continue
        test = [s for s in samples if s.month == m]
        if not test:
            continue
        y = [int(s.y) for s in test]
        p = []
        for s in test:
            p_pre = _fuse_probs_by_precision(
                p_team=float(s.p_team),
                p_players=float(s.p_players),
                n_team=float(s.n_team),
                n_players=float(s.n_players),
                team_scale=float(team_scale),
                players_scale=float(players_scale),
                coverage=float(s.coverage_ratio),
                transfer_signal=float(s.transfer_signal),
                lineup_continuity=float(s.lineup_continuity),
                roster_changed=bool(s.roster_changed),
                season_phase=("playoffs" if str(s.phase).strip().lower() == "playoffs" else "auto"),
                players_freshness=float(s.freshness_avg),
            )
            p_post = _apply_form_blend_on_sample(
                p_model=float(p_pre),
                sample=s,
                enabled=bool((form_cfg or {}).get("enabled", False)),
                max_weight=float((form_cfg or {}).get("max_weight", 0.75) or 0.75),
                knee_games=float((form_cfg or {}).get("knee_games", 6.0) or 6.0),
                beta_prior=float((form_cfg or {}).get("beta_prior", 2.0) or 2.0),
                signal_power=float((form_cfg or {}).get("signal_power", 1.0) or 1.0),
                elite_same_league_parity_strength=float(
                    (form_cfg or {}).get("elite_same_league_parity_strength", 0.35) or 0.35
                ),
            )
            p.append(float(p_post))
            if bool(return_rows):
                pred_rows.append(
                    {
                        "month": str(m),
                        "league": str(s.league),
                        "split": str(s.split),
                        "side": str(s.side),
                        "bo": str(s.bo_bucket),
                        "phase": str(s.phase),
                        "y": int(s.y),
                        "p": float(p_post),
                    }
                )

        ll = float(logloss(y, p))
        br = float(brier(y, p))
        folds.append({"month": m, "n": len(y), "logloss": ll, "brier": br})
        all_y.extend(y)
        all_p.extend(p)

    if not all_y:
        rep0 = {"folds": folds, "logloss": None, "brier": None, "ece": None, "n_total": 0}
        if bool(return_rows):
            rep0["pred_rows"] = pred_rows
        return rep0
    rep = {
        "folds": folds,
        "logloss": float(logloss(all_y, all_p)),
        "brier": float(brier(all_y, all_p)),
        "ece": _ece_metric(all_y, all_p, bins=10),
        "n_total": len(all_y),
    }
    if bool(return_rows):
        rep["pred_rows"] = pred_rows
    return rep


def _meta_recent_evidence_from_sample(sample: Sample) -> float:
    n_a = int(max(0, int(sample.form_n_a)))
    n_b = int(max(0, int(sample.form_n_b)))
    if n_a <= 0 or n_b <= 0:
        return 0.0
    bp = 2.0
    p_a = float(clip_prob((float(int(sample.form_wins_a)) + bp) / (float(n_a) + 2.0 * bp)))
    p_b = float(clip_prob((float(int(sample.form_wins_b)) + bp) / (float(n_b) + 2.0 * bp)))
    sig = abs(p_a - p_b) / 0.5
    sig = max(0.0, min(1.0, sig))
    n_min = float(min(n_a, n_b))
    sample_factor = n_min / (n_min + 6.0)
    return float(max(0.0, min(1.0, sample_factor * sig)))


def _meta_probs_for_sample(
    sample: Sample,
    *,
    team_scale: float,
    players_scale: float,
    form_cfg: Dict[str, Any],
) -> Tuple[float, float]:
    p_anchor = float(clip_prob(float(sample.p_team)))
    p_pre = _fuse_probs_by_precision(
        p_team=float(sample.p_team),
        p_players=float(sample.p_players),
        n_team=float(sample.n_team),
        n_players=float(sample.n_players),
        team_scale=float(team_scale),
        players_scale=float(players_scale),
        coverage=float(sample.coverage_ratio),
        transfer_signal=float(sample.transfer_signal),
        lineup_continuity=float(sample.lineup_continuity),
        roster_changed=bool(sample.roster_changed),
        season_phase=("playoffs" if str(sample.phase).strip().lower() == "playoffs" else "auto"),
        players_freshness=float(sample.freshness_avg),
    )
    p_adj = _apply_form_blend_on_sample(
        p_model=float(p_pre),
        sample=sample,
        enabled=bool((form_cfg or {}).get("enabled", False)),
        max_weight=float((form_cfg or {}).get("max_weight", 0.75) or 0.75),
        knee_games=float((form_cfg or {}).get("knee_games", 6.0) or 6.0),
        beta_prior=float((form_cfg or {}).get("beta_prior", 2.0) or 2.0),
        signal_power=float((form_cfg or {}).get("signal_power", 1.0) or 1.0),
        elite_same_league_parity_strength=float(
            (form_cfg or {}).get("elite_same_league_parity_strength", 0.35) or 0.35
        ),
    )
    return float(clip_prob(p_anchor)), float(clip_prob(p_adj))


def _np_sigmoid(z: np.ndarray) -> np.ndarray:
    zz = np.clip(np.asarray(z, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-zz))


def _fit_logreg_numpy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 0.1,
    lr: float = 0.08,
    epochs: int = 260,
) -> Dict[str, Any]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) < 20:
        return {"error": "invalid_train_data"}
    n, d = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    Xn = (X - mu) / sd

    w = np.zeros(d, dtype=float)
    b = 0.0
    lam = float(max(0.0, l2))
    eta = float(max(1e-4, lr))
    for t in range(int(max(20, epochs))):
        p = _np_sigmoid(Xn @ w + b)
        err = p - y
        grad_w = (Xn.T @ err) / float(n)
        if lam > 0.0:
            grad_w = grad_w + (lam / float(n)) * w
        grad_b = float(err.mean())
        step = eta / math.sqrt(1.0 + 0.02 * float(t))
        w -= step * grad_w
        b -= step * grad_b

    p_hat = _np_sigmoid(Xn @ w + b)
    ll = float(logloss([int(v) for v in y.tolist()], [float(v) for v in p_hat.tolist()]))
    br = float(brier([int(v) for v in y.tolist()], [float(v) for v in p_hat.tolist()]))
    return {
        "intercept": float(b),
        "coef": [float(v) for v in w.tolist()],
        "x_mean": [float(v) for v in mu.tolist()],
        "x_std": [float(v) for v in sd.tolist()],
        "train_logloss": float(ll),
        "train_brier": float(br),
        "n_train": int(n),
    }


def _predict_logreg_numpy(model: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        return np.asarray([], dtype=float)
    try:
        b = float(model.get("intercept", 0.0) or 0.0)
        w = np.asarray(model.get("coef") or [], dtype=float)
        mu = np.asarray(model.get("x_mean") or [], dtype=float)
        sd = np.asarray(model.get("x_std") or [], dtype=float)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    if w.ndim != 1 or len(w) != X.shape[1]:
        return np.asarray([], dtype=float)
    if mu.shape != w.shape:
        mu = np.zeros_like(w)
    if sd.shape != w.shape:
        sd = np.ones_like(w)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    Xn = (X - mu) / sd
    return _np_sigmoid(Xn @ w + b)


def _fit_form_grid(
    samples: List[Sample],
    *,
    team_scale: float,
    players_scale: float,
    min_train_games: int,
    league_filter: str | None = None,
    macro_filter: str | None = None,
) -> Dict[str, Any]:
    baseline = _evaluate_walkforward(
        samples,
        team_scale=float(team_scale),
        players_scale=float(players_scale),
        min_train_games=int(min_train_games),
        league_filter=league_filter,
        macro_filter=macro_filter,
        form_cfg={"enabled": False},
    )

    max_weight_grid = [0.40, 0.55, 0.70, 0.85]
    knee_games_grid = [3.0, 6.0, 10.0, 16.0]
    beta_prior_grid = [1.0, 2.0, 3.5]
    signal_power_grid = [0.7, 1.0, 1.4, 2.0]
    parity_grid = [0.0, 0.20, 0.35]

    best: Dict[str, Any] = {
        "enabled": False,
        "max_weight": 0.75,
        "knee_games": 6.0,
        "beta_prior": 2.0,
        "signal_power": 1.0,
        "elite_same_league_parity_strength": 0.35,
        "logloss": (
            float(baseline.get("logloss")) if baseline.get("logloss") is not None else float("inf")
        ),
        "brier": (
            float(baseline.get("brier")) if baseline.get("brier") is not None else float("inf")
        ),
        "n_total": int(baseline.get("n_total") or 0),
        "folds": len(baseline.get("folds") or []),
    }
    grid_rows: List[Dict[str, Any]] = []

    for mw in max_weight_grid:
        for kg in knee_games_grid:
            for bp in beta_prior_grid:
                for sp in signal_power_grid:
                    for pg in parity_grid:
                        cfg = {
                            "enabled": True,
                            "max_weight": float(mw),
                            "knee_games": float(kg),
                            "beta_prior": float(bp),
                            "signal_power": float(sp),
                            "elite_same_league_parity_strength": float(pg),
                        }
                        rep = _evaluate_walkforward(
                            samples,
                            team_scale=float(team_scale),
                            players_scale=float(players_scale),
                            min_train_games=int(min_train_games),
                            league_filter=league_filter,
                            macro_filter=macro_filter,
                            form_cfg=cfg,
                        )
                        ll = rep.get("logloss")
                        br = rep.get("brier")
                        if ll is None:
                            continue
                        row = {
                            **cfg,
                            "logloss": float(ll),
                            "brier": float(br) if br is not None else None,
                            "n_total": int(rep.get("n_total") or 0),
                            "folds": len(rep.get("folds") or []),
                        }
                        grid_rows.append(row)
                        if float(ll) < float(best["logloss"]):
                            best = dict(row)

    improvement_ll = (
        float(baseline.get("logloss")) - float(best["logloss"])
        if baseline.get("logloss") is not None and math.isfinite(float(best["logloss"]))
        else None
    )
    improvement_br = (
        float(baseline.get("brier")) - float(best["brier"])
        if baseline.get("brier") is not None and best.get("brier") is not None
        and math.isfinite(float(best["brier"]))
        else None
    )
    best["improvement_logloss_vs_no_form"] = improvement_ll
    best["improvement_brier_vs_no_form"] = improvement_br

    # Winner's-curse guard: 576 OOS evaluations will almost always find a
    # combination that beats the baseline by noise alone.  Require a minimum
    # meaningful improvement before enabling form.
    #
    # Thresholds chosen empirically:
    #   0.0008 logloss — corresponds to ~0.1-0.15 pp calibration gain per game.
    #   -0.0003 brier  — disallow form if it clearly hurts calibration even
    #                    when logloss improves (discrimination vs calibration
    #                    divergence signals over-fitting to the selection set).
    #   150 OOS games  — too few OOS points makes any metric estimate unreliable.
    _MIN_LL_GAIN = 0.0008
    _MAX_BR_REGRESSION = -0.0003
    _MIN_OOS_GAMES = 150

    n_oos = int(best.get("n_total") or 0)
    ll_ok = improvement_ll is not None and math.isfinite(improvement_ll) and improvement_ll >= _MIN_LL_GAIN
    br_ok = improvement_br is None or improvement_br >= _MAX_BR_REGRESSION
    n_ok = n_oos >= _MIN_OOS_GAMES

    if not (ll_ok and br_ok and n_ok):
        disable_reasons = []
        if not ll_ok:
            disable_reasons.append(f"logloss_gain_small({improvement_ll:.6f}<{_MIN_LL_GAIN})" if improvement_ll is not None else "logloss_gain_unknown")
        if not br_ok:
            disable_reasons.append(f"brier_regression({improvement_br:.6f}<{_MAX_BR_REGRESSION})")
        if not n_ok:
            disable_reasons.append(f"insufficient_oos_games({n_oos}<{_MIN_OOS_GAMES})")
        best["enabled"] = False
        best["reason_disabled"] = "|".join(disable_reasons)

    return {
        "baseline_no_form": {
            "enabled": False,
            "logloss": baseline.get("logloss"),
            "brier": baseline.get("brier"),
            "n_total": int(baseline.get("n_total") or 0),
            "folds": len(baseline.get("folds") or []),
        },
        "best_form": best,
        "grid_form": sorted(
            grid_rows,
            key=lambda r: (
                float(r["logloss"]),
                float(r["brier"]) if r.get("brier") is not None else 999.0,
            ),
        ),
    }


def _form_settings_from_best(best_form: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ml_form_blend_enabled": bool(best_form.get("enabled", False)),
        "ml_form_blend_max_weight": float(best_form.get("max_weight", 0.75) or 0.75),
        "ml_form_blend_knee_games": float(best_form.get("knee_games", 6.0) or 6.0),
        "ml_form_blend_beta": float(best_form.get("beta_prior", 2.0) or 2.0),
        "ml_form_blend_signal_power": float(best_form.get("signal_power", 1.0) or 1.0),
        "ml_form_blend_parity_strength": float(best_form.get("elite_same_league_parity_strength", 0.35) or 0.35),
        "ml_form_use_fitted": True,
    }
