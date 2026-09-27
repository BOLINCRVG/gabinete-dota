from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .artifact import MLArtifact
from .calibration import fit_platt_from_raw_probs
from .fusion_core import fuse_with_coherence_guard
from .league_context import (
    is_explicit_interregional_league_label as shared_is_explicit_interregional_league_label,
    league_macro_key,
)
from .players_layer import (
    PlayersArtifact,
    compute_delta_and_confidence,
    norm_pos,
)
from .roster_signal_core import lineup_continuity_from_details, transfer_signal_from_details
from .stats_teams import build_team_games, load_team_rows
from .tiers import tier_for_league, tier_priority
from .utils import brier, clip_prob, coerce_loud, logit, logloss, safe_float as _f, sigmoid
from .fit_io_core import _read_json_file_safe, _safe_int
def _is_valid_areal_prob_tuning(cfg: Any) -> bool:
    if not isinstance(cfg, dict):
        return False
    if not str(cfg.get("version") or "").startswith("areal_prob_tuning_v"):
        return False
    g = cfg.get("global") if isinstance(cfg.get("global"), dict) else {}
    if not g:
        return False
    for key, default in {
        "bayes_info_scale": 1.0,
        "bayes_mu_q_strength": 1.0,
        "quality_k_boost": 0.22,
        "qwl_surplus_mix": 0.70,
        "qwl_throw_weight": 1.20,
        "qwl_comeback_weight": 0.65,
        "qwl_phase_dynamic_strength": 0.35,
        "qwl_opponent_context_mix": 0.30,
    }.items():
        if not math.isfinite(_f(g.get(key), float(default))):
            return False
    return True


def _default_ml_marginal_band_profiles() -> Dict[str, Dict[str, float]]:
    return {
        "top": {
            "strength_scale": 0.35,
            "prior_scale": 0.70,
            "max_shrink_scale": 0.55,
            "cap_shift": +0.06,
            "soft_shift": +10.0,
            "hard_shift": +8.0,
            "gate_score": 80.0,
            "gate_n_eff": 30.0,
        },
        "core": {
            "strength_scale": 0.75,
            "prior_scale": 0.95,
            "max_shrink_scale": 0.85,
            "cap_shift": +0.02,
            "soft_shift": +3.0,
            "hard_shift": +2.0,
            "gate_score": 84.0,
            "gate_n_eff": 42.0,
        },
        "edge": {
            "strength_scale": 1.15,
            "prior_scale": 1.20,
            "max_shrink_scale": 1.05,
            "cap_shift": -0.04,
            "soft_shift": -6.0,
            "hard_shift": -4.0,
            "gate_score": 88.0,
            "gate_n_eff": 55.0,
        },
    }


def _sanitize_ml_marginal_band_coeffs(raw: Any) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    base = _default_ml_marginal_band_profiles()
    if not isinstance(raw, dict):
        return out
    for b in ("top", "core", "edge"):
        rv = raw.get(b)
        if not isinstance(rv, dict):
            continue
        row: Dict[str, float] = {}
        for k in base[b].keys():
            if k not in rv:
                continue
            fv = _f(rv.get(k), float("nan"))
            if math.isfinite(fv):
                row[str(k)] = float(fv)
        if row:
            out[b] = row
    return out


def _is_valid_ml_marginal_band_coeffs(cfg: Any) -> bool:
    return bool(_sanitize_ml_marginal_band_coeffs(cfg))


def _load_areal_prob_tuning_snapshot(base_dir: Path) -> Dict[str, Any]:
    """Load promoted A-real probabilistic tuning to preserve it in params_fitted.

    Priority:
      1) phase2b report tuning_cfg when decision is promote.
      2) existing params_fitted.json areal_prob_tuning.
    """
    root = Path(base_dir).expanduser()
    rep = _read_json_file_safe(root / "phase2b_areal_tuning_report.json")
    if isinstance(rep, dict) and str(rep.get("decision") or "").strip().lower() == "promote":
        tcfg = rep.get("tuning_cfg")
        if _is_valid_areal_prob_tuning(tcfg):
            out = dict(tcfg)
            out["_source"] = "phase2b_areal_tuning_report.json"
            return out

    cur = _read_json_file_safe(root / "params_fitted.json")
    cur_t = cur.get("areal_prob_tuning") if isinstance(cur, dict) else None
    if _is_valid_areal_prob_tuning(cur_t):
        out = dict(cur_t)
        out["_source"] = "params_fitted.json"
        return out
    return {}


def _load_ml_marginal_band_coeffs_snapshot(base_dir: Path) -> Dict[str, Any]:
    """Load promoted marginal band tuning to preserve it in params_fitted.

    Priority:
      1) phase4b report band_coeffs (or tuning_cfg.band_coeffs) when decision is promote.
      2) existing params_fitted.json ml_marginal_trust_band_coeffs.
    """
    root = Path(base_dir).expanduser()
    rep = _read_json_file_safe(root / "phase4b_marginal_tuning_report.json")
    if isinstance(rep, dict) and str(rep.get("decision") or "").strip().lower() == "promote":
        tcfg = rep.get("tuning_cfg") if isinstance(rep.get("tuning_cfg"), dict) else {}
        bcfg = tcfg.get("band_coeffs") if isinstance(tcfg.get("band_coeffs"), dict) else rep.get("band_coeffs")
        clean = _sanitize_ml_marginal_band_coeffs(bcfg)
        if clean:
            out = dict(clean)
            out["_source"] = "phase4b_marginal_tuning_report.json"
            return out

    cur = _read_json_file_safe(root / "params_fitted.json")
    cur_t = cur.get("ml_marginal_trust_band_coeffs") if isinstance(cur, dict) else None
    clean = _sanitize_ml_marginal_band_coeffs(cur_t)
    if clean:
        out = dict(clean)
        out["_source"] = "params_fitted.json"
        return out
    return {}


def _month_key(ts: pd.Timestamp) -> str:
    return f"{int(ts.year):04d}-{int(ts.month):02d}"


def _league_macro_key(league: str) -> str:
    return str(league_macro_key(league, unknown="OTHER"))


def _is_interregional_league(league: Any) -> bool:
    return bool(shared_is_explicit_interregional_league_label(league))


def _ece_metric(y: List[int], p: List[float], bins: int = 10) -> Optional[float]:
    if not y or not p or len(y) != len(p):
        return None
    yy = np.asarray([int(v) for v in y], dtype=float)
    pp = np.asarray([float(v) for v in p], dtype=float)
    if yy.size <= 0 or pp.size <= 0 or yy.size != pp.size:
        return None
    pp = np.clip(pp, 1e-12, 1.0 - 1e-12)
    n = int(yy.size)
    idx = np.minimum((pp * int(max(2, bins))).astype(int), int(max(2, bins)) - 1)
    ece = 0.0
    for b in range(int(max(2, bins))):
        m = idx == b
        if not np.any(m):
            continue
        conf = float(np.mean(pp[m]))
        acc = float(np.mean(yy[m]))
        w = float(np.sum(m)) / float(max(1, n))
        ece += float(abs(acc - conf) * w)
    return float(ece)


def _load_team_games_multi(csv_paths: List[str]) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for p in (csv_paths or []):
        try:
            tr = load_team_rows(str(p))
            if tr is None or tr.empty:
                continue
            tg = build_team_games(tr)
            if tg is None or tg.empty:
                continue
            parts.append(tg)
        except Exception:
            continue
    if not parts:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for _df in parts:
        if not isinstance(_df, pd.DataFrame) or _df.empty:
            continue
        _x = _df.dropna(axis=1, how="all")
        if _x.empty or _x.shape[1] == 0:
            continue
        frames.append(_x)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out.get("date"), errors="coerce")
        out = out.dropna(subset=["date"]).copy()
    if "gameid" in out.columns and "team" in out.columns:
        out = out.sort_values(["date", "gameid"]).drop_duplicates(subset=["gameid", "team"], keep="last").reset_index(drop=True)
    return out


def _ms_key(league: str, split: str, side: str, bo: str, phase: str) -> str:
    return (
        f"{str(league or '').strip()}|"
        f"{str(split or '').strip()}|"
        f"{str(side or '').strip().lower()}|"
        f"{str(bo or '').strip()}|"
        f"{str(phase or '').strip().lower()}"
    )


def _estimate_shrink_k_empirical_bayes(
    agg_specific: Dict[str, Dict[str, float]],
    *,
    k_min: float = 25.0,
    k_max: float = 300.0,
    min_segments: int = 8,
    min_n_per_segment: float = 15.0,
) -> float:
    """Estimate optimal Bayesian shrinkage k via empirical Bayes (James-Stein).

    k = sigma_within^2 / sigma_true^2

    sigma_within^2: expected variance of the sample mean due to sampling noise
                    (approx 0.25/n for binary outcomes)
    sigma_true^2:   estimated true between-segment variance (total observed
                    variance minus sampling noise)

    A large k means segment biases are mostly sampling noise → shrink hard.
    A small k means genuine systematic biases exist → trust the data sooner.
    Falls back to k_max when data is insufficient or signal is indistinguishable
    from noise.
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

    # Expected sampling variance of the segment means (binary outcome ≈ 0.25/n)
    sigma_within_sq = float(np.mean(0.25 / n_arr))

    # Observed total variance of segment-level biases (ddof=1 for unbiasedness)
    sigma_total_sq = float(np.var(mu_arr, ddof=1)) if len(mus) > 1 else 0.0

    # True between-segment variance: subtract out sampling noise
    sigma_true_sq = max(0.0, sigma_total_sq - sigma_within_sq)

    if sigma_true_sq < 1e-9:
        # No detectable real signal: shrink maximally
        return float(k_max)

    k_est = sigma_within_sq / sigma_true_sq
    return float(max(k_min, min(k_max, k_est)))


def _build_microseg_table_from_pred_rows(
    rows: List[Dict[str, Any]],
    *,
    shrink_k: float = 90.0,
) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, float]] = {}
    agg_specific: Dict[str, Dict[str, float]] = {}  # level-0 only, for k estimation
    for r in (rows or []):
        if not isinstance(r, dict):
            continue
        lg = str(r.get("league") or "").strip()
        sp = str(r.get("split") or "").strip()
        sd = str(r.get("side") or "").strip().lower() or "blue"
        bo = str(r.get("bo") or "").strip() or "1"
        ph = str(r.get("phase") or "").strip().lower() or "regular"
        y = _f(r.get("y"), float("nan"))
        p = _f(r.get("p"), float("nan"))
        if not (math.isfinite(y) and math.isfinite(p)):
            continue
        err = float(y - p)
        keys = [
            _ms_key(lg, sp, sd, bo, ph),
            _ms_key(lg, "*", sd, bo, ph),
            _ms_key(lg, "*", sd, "*", ph),
            _ms_key(lg, "*", "*", "*", ph),
            _ms_key(lg, "*", "*", "*", "*"),
            _ms_key("*", "*", "*", "*", "*"),
        ]
        for i, k in enumerate(keys):
            d = agg.setdefault(k, {"n": 0.0, "sum_err": 0.0})
            d["n"] += 1.0
            d["sum_err"] += float(err)
            if i == 0:
                ds = agg_specific.setdefault(k, {"n": 0.0, "sum_err": 0.0})
                ds["n"] += 1.0
                ds["sum_err"] += float(err)

    # Estimate k from the most specific level using empirical Bayes.
    # Falls back to the caller-supplied default if there's insufficient data.
    sk = float(max(1.0, _estimate_shrink_k_empirical_bayes(agg_specific)))
    if not math.isfinite(sk) or sk <= 0.0:
        sk = float(max(1.0, float(shrink_k)))

    out: Dict[str, Dict[str, float]] = {}
    for k, d in agg.items():
        n = float(d.get("n", 0.0))
        if n <= 0:
            continue
        raw_bias = float(d.get("sum_err", 0.0)) / n
        shrink = n / (n + sk)
        out[k] = {"n": n, "raw_bias": raw_bias, "bias": float(raw_bias * shrink), "shrink_k": float(sk)}
    return out


def _fit_fused_post_calibration_from_pred_rows(
    pred_rows: List[Dict[str, Any]],
    *,
    min_train_n: int = 1200,
) -> Dict[str, Any]:
    out_base: Dict[str, Any] = {
        "enabled": False,
        "version": "fused_post_calibration_v1",
        "source": "walkforward_oos_platt",
        "gate": {"decision": "shadow_only", "reasons": ["not_evaluated"]},
    }
    if not pred_rows:
        out_base["reason"] = "no_pred_rows"
        return out_base

    rows: List[Dict[str, Any]] = []
    for r in (pred_rows or []):
        if not isinstance(r, dict):
            continue
        y = _safe_int(r.get("y"), -1)
        p = _f(r.get("p"), float("nan"))
        if y not in (0, 1):
            continue
        if (not math.isfinite(p)) or p <= 0.0 or p >= 1.0:
            continue
        rows.append(
            {
                "month": str(r.get("month") or ""),
                "league": str(r.get("league") or ""),
                "y": int(y),
                "p": float(clip_prob(float(p))),
            }
        )
    if len(rows) < 200:
        out_base["reason"] = "insufficient_rows"
        out_base["n_total"] = len(rows)
        return out_base

    df = pd.DataFrame(rows)
    if df.empty:
        out_base["reason"] = "empty_df"
        return out_base
    months = sorted({str(x) for x in df["month"].astype(str).tolist() if str(x).strip()})
    if len(months) < 2:
        out_base["reason"] = "insufficient_months"
        out_base["n_total"] = len(df)
        return out_base

    p_oof = coerce_loud(df["p"], nome="p (OOF p/ calibracao)", default=0.5).astype(float).clip(lower=1e-9, upper=1.0 - 1e-9).copy()
    fold_rows: List[Dict[str, Any]] = []
    for m in months:
        tr = df[df["month"].astype(str) < str(m)]
        te = df[df["month"].astype(str) == str(m)]
        if te.empty:
            continue
        if len(tr) < int(max(80, int(min_train_n))):
            continue
        y_tr = [int(v) for v in tr["y"].tolist()]
        p_tr = [float(v) for v in tr["p"].tolist()]
        try:
            cal = fit_platt_from_raw_probs(y_tr, p_tr, l2=1e-3, max_iter=220)
        except Exception:
            continue
        idx = te.index
        p_raw_te = coerce_loud(df.loc[idx, "p"], nome="p (fold de teste)", default=0.5).astype(float).clip(lower=1e-9, upper=1.0 - 1e-9)
        p_cal_te = p_raw_te.map(lambda x: float(clip_prob(float(cal.predict_proba(float(x))))))
        p_oof.loc[idx] = p_cal_te
        y_te = [int(v) for v in te["y"].tolist()]
        p_raw_list = [float(v) for v in p_raw_te.tolist()]
        p_cal_list = [float(v) for v in p_cal_te.tolist()]
        fold_rows.append(
            {
                "month": str(m),
                "n": len(te),
                "raw_logloss": float(logloss(y_te, p_raw_list)),
                "cal_logloss": float(logloss(y_te, p_cal_list)),
                "raw_brier": float(brier(y_te, p_raw_list)),
                "cal_brier": float(brier(y_te, p_cal_list)),
                "raw_ece": _ece_metric(y_te, p_raw_list, bins=10),
                "cal_ece": _ece_metric(y_te, p_cal_list, bins=10),
            }
        )

    y_all = [int(v) for v in df["y"].tolist()]
    p_raw_all = [float(v) for v in df["p"].tolist()]
    p_cal_all = [float(v) for v in p_oof.tolist()]
    if not y_all:
        out_base["reason"] = "no_valid_points"
        return out_base

    raw_ll = float(logloss(y_all, p_raw_all))
    raw_br = float(brier(y_all, p_raw_all))
    raw_ece = _ece_metric(y_all, p_raw_all, bins=10)
    cal_ll = float(logloss(y_all, p_cal_all))
    cal_br = float(brier(y_all, p_cal_all))
    cal_ece = _ece_metric(y_all, p_cal_all, bins=10)

    gain_ll = float(raw_ll - cal_ll)
    gain_br = float(raw_br - cal_br)
    gain_ece = float((raw_ece - cal_ece) if (raw_ece is not None and cal_ece is not None) else float("nan"))

    # Train runtime model on all OOS points.
    try:
        cal_global = fit_platt_from_raw_probs(y_all, p_raw_all, l2=1e-3, max_iter=260)
        model_global = {"a": float(getattr(cal_global, "a", 1.0)), "b": float(getattr(cal_global, "b", 0.0))}
    except Exception:
        model_global = {"a": 1.0, "b": 0.0}

    # Per-league minimum: scales with min_train_n, capped at 500, floored at 150.
    _per_league_min_n = int(max(150, min(500, int(min_train_n * 1.5))))

    by_league: Dict[str, Dict[str, Any]] = {}
    for lg, gl in df.groupby("league"):
        if len(gl) < _per_league_min_n:
            continue
        y_l = [int(v) for v in gl["y"].tolist()]
        p_l = [float(v) for v in gl["p"].tolist()]
        try:
            cal_l = fit_platt_from_raw_probs(y_l, p_l, l2=1e-3, max_iter=240)
        except Exception:
            continue
        p_l_cal = [float(clip_prob(float(cal_l.predict_proba(float(x))))) for x in p_l]
        ll_l_raw = float(logloss(y_l, p_l))
        ll_l_cal = float(logloss(y_l, p_l_cal))
        br_l_raw = float(brier(y_l, p_l))
        br_l_cal = float(brier(y_l, p_l_cal))
        ece_l_raw = _ece_metric(y_l, p_l, bins=10)
        ece_l_cal = _ece_metric(y_l, p_l_cal, bins=10)
        by_league[str(lg)] = {
            "model": {"a": float(getattr(cal_l, "a", 1.0)), "b": float(getattr(cal_l, "b", 0.0))},
            "n_total": len(gl),
            "raw_logloss": float(ll_l_raw),
            "cal_logloss": float(ll_l_cal),
            "raw_brier": float(br_l_raw),
            "cal_brier": float(br_l_cal),
            "raw_ece": ece_l_raw,
            "cal_ece": ece_l_cal,
            "gain_logloss": float(ll_l_raw - ll_l_cal),
            "gain_brier": float(br_l_raw - br_l_cal),
            "gain_ece": (float(ece_l_raw - ece_l_cal) if (ece_l_raw is not None and ece_l_cal is not None) else None),
        }

    # Global N gate: adaptive — scales with min_train_n so small-league fits
    # aren't permanently blocked by a one-size-fits-all threshold of 1200.
    # Derivation: we need at least min_train_n samples for the first fold plus
    # a meaningful validation window, so 1.5× is a conservative lower bound.
    # Hard floor at 200 (below that, Platt estimates are unreliable regardless).
    _min_n_gate = int(max(200, int(min_train_n * 1.5)))
    n_total = len(df)

    # When N is smaller we have noisier metric estimates, so tighten the ECE
    # threshold proportionally to avoid promoting marginal calibrations.
    # Interpolates linearly: at _min_n_gate → 0.0030, at 2×_min_n_gate → 0.0020.
    if n_total >= _min_n_gate * 2:
        _ece_threshold = 0.0020
    else:
        _ratio = max(0.0, min(1.0, (n_total - _min_n_gate) / max(1, _min_n_gate)))
        _ece_threshold = float(0.0030 - _ratio * 0.0010)

    gate_decision = "promote"
    gate_reasons: List[str] = []
    if n_total < _min_n_gate:
        gate_decision = "shadow_only"
        gate_reasons.append("low_sample")
    if gain_ll < -0.0005:
        gate_decision = "shadow_only"
        gate_reasons.append("logloss_regression")
    if gain_br < -0.0005:
        gate_decision = "shadow_only"
        gate_reasons.append("brier_regression")
    if (not math.isfinite(gain_ece)) or gain_ece < _ece_threshold:
        gate_decision = "shadow_only"
        gate_reasons.append("ece_gain_small")
    if not gate_reasons:
        gate_reasons = ["pass"]

    promoted_by_league: Dict[str, Dict[str, float]] = {}
    if gate_decision == "promote":
        for lg, row in by_league.items():
            gll = float(row.get("gain_logloss", 0.0) or 0.0)
            gbe = float(row.get("gain_brier", 0.0) or 0.0)
            gce = float(row.get("gain_ece", 0.0) or 0.0)
            if gll < -0.0010 or gbe < -0.0010 or gce < 0.0010:
                continue
            md = row.get("model") if isinstance(row.get("model"), dict) else {}
            if md:
                promoted_by_league[str(lg)] = {"a": float(md.get("a", 1.0)), "b": float(md.get("b", 0.0))}

    return {
        "enabled": True,
        "version": "fused_post_calibration_v1",
        "source": "walkforward_oos_platt",
        "gate": {
            "decision": str(gate_decision),
            "reasons": list(gate_reasons),
            "thresholds": {
                "min_n_total": int(_min_n_gate),
                "min_gain_ece": float(_ece_threshold),
                "max_logloss_regression": 0.0005,
                "max_brier_regression": 0.0005,
            },
        },
        "raw": {
            "n_total": len(df),
            "logloss": float(raw_ll),
            "brier": float(raw_br),
            "ece": raw_ece,
        },
        "oof_platt": {
            "n_total": len(df),
            "logloss": float(cal_ll),
            "brier": float(cal_br),
            "ece": cal_ece,
            "folds": len(fold_rows),
            "fold_rows": fold_rows,
        },
        "improvements": {
            "gain_logloss": float(gain_ll),
            "gain_brier": float(gain_br),
            "gain_ece": (float(gain_ece) if math.isfinite(gain_ece) else None),
        },
        "model_global": dict(model_global),
        "model_by_league": {str(k): dict((v or {}).get("model") or {}) for k, v in by_league.items()},
        "model_by_league_diagnostics": by_league,
        "promoted_model_global": (dict(model_global) if gate_decision == "promote" else {}),
        "promoted_model_by_league": promoted_by_league,
        "runtime": {
            "enabled": True,
            # Allow larger Platt corrections when N is large (reliable estimates).
            # Interpolates: N≤600→0.80, N=1200→1.00, N≥2400→1.40.
            "max_abs_logit_shift": float(
                0.80 + 0.60 * max(0.0, min(1.0, (n_total - 600) / 1800.0))
            ),
            "use_league_override": True,
        },
    }


def _fuse_probs_by_precision(
    *,
    p_team: float,
    p_players: float,
    n_team: float,
    n_players: float,
    team_scale: float,
    players_scale: float,
    coverage: float | None = None,
    transfer_signal: float = 0.0,
    lineup_continuity: float | None = None,
    roster_changed: bool = False,
    season_phase: str = "auto",
    players_freshness: float = 1.0,
) -> float:
    nt = max(0.0, float(n_team))
    np_ = max(0.0, float(n_players))
    pf = float(players_freshness) if math.isfinite(float(players_freshness)) else 1.0
    np_eff = float(np_ * max(0.0, min(1.0, pf)))
    fused = fuse_with_coherence_guard(
        p_team=float(p_team),
        p_players=float(p_players),
        n_team=float(nt),
        n_players=float(np_eff),
        team_scale=float(max(0.0, float(team_scale))),
        players_scale=float(max(0.0, float(players_scale))),
        coverage=coverage,
        transfer_signal=float(transfer_signal),
        lineup_continuity=lineup_continuity,
        roster_changed=bool(roster_changed),
        season_phase=str(season_phase or "auto"),
    )
    p_fused = fused.get("p_fused") if isinstance(fused, dict) else None
    try:
        return float(clip_prob(float(p_fused)))
    except (TypeError, ValueError):
        return float(clip_prob(float(p_team)))


def _sym_team_probs(art: MLArtifact, team_a: str, team_b: str, league: str) -> Tuple[float, float]:
    tier = str(tier_for_league(league))
    p_raw_f = float(
        art.model.p_blue(
            str(team_a),
            str(team_b),
            blue_league=str(league),
            red_league=str(league),
            blue_tier=tier,
            red_tier=tier,
        )
    )
    p_raw_r = float(
        art.model.p_blue(
            str(team_b),
            str(team_a),
            blue_league=str(league),
            red_league=str(league),
            blue_tier=tier,
            red_tier=tier,
        )
    )
    p_raw = 0.5 * (p_raw_f + (1.0 - p_raw_r))
    p_cal_f = float(art.calibrator.predict_proba(p_raw_f))
    p_cal_r = float(art.calibrator.predict_proba(p_raw_r))
    p_cal = 0.5 * (p_cal_f + (1.0 - p_cal_r))
    return float(clip_prob(p_raw)), float(clip_prob(p_cal))


def _build_lineup_cache(player_rows: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if player_rows is None or player_rows.empty:
        return out
    d = player_rows.copy()
    d["gameid"] = d["gameid"].astype(str)
    d["side"] = d["side"].astype(str).str.lower()
    d["pos_can"] = d["position"].apply(norm_pos) if "pos_can" not in d.columns else d["pos_can"]
    for gid, g in d.groupby("gameid", sort=False):
        blue: Dict[str, str] = {}
        red: Dict[str, str] = {}
        blue_team = ""
        red_team = ""
        for _, r in g.iterrows():
            side = str(r.get("side") or "").strip().lower()
            pos = str(r.get("pos_can") or "").strip()
            pk = str(r.get("player_key") or "").strip()
            tm = str(r.get("teamname") or "").strip()
            if side == "blue":
                if tm and not blue_team:
                    blue_team = tm
                if pos and pk and pos not in blue:
                    blue[pos] = pk
            elif side == "red":
                if tm and not red_team:
                    red_team = tm
                if pos and pk and pos not in red:
                    red[pos] = pk
        out[str(gid)] = {
            "blue": blue,
            "red": red,
            "blue_team": blue_team,
            "red_team": red_team,
        }
    return out


@dataclass(frozen=True)
class Sample:
    month: str
    gameid: str
    league: str
    patch: str
    split: str
    macro: str
    side: str
    bo_bucket: str
    phase: str
    y: int
    p_team: float
    p_players: float
    n_team: float
    n_players: float
    form_n_a: int
    form_n_b: int
    form_wins_a: int
    form_wins_b: int
    coverage_ratio: float = 0.0
    freshness_avg: float = 0.0
    transfer_signal: float = 0.0
    lineup_continuity: float = 1.0
    roster_changed: bool = False
    same_league: bool = True
    blue_team: str = ""
    red_team: str = ""
    game_no: int = 1
    date_day: str = ""
    lane_top: float = 0.0
    lane_jungle: float = 0.0
    lane_mid: float = 0.0
    lane_bot: float = 0.0
    lane_support: float = 0.0


def _make_samples(
    *,
    art: MLArtifact,
    p_art: PlayersArtifact,
    games: pd.DataFrame,
    lineup_cache: Dict[str, Dict[str, Any]],
    w_lineup: float,
) -> List[Sample]:
    samples: List[Sample] = []
    games_count: Dict[str, int] = defaultdict(int)
    wins_count: Dict[str, int] = defaultdict(int)
    series_max_by_gid: Dict[str, int] = {}
    try:
        gtmp = games.copy()
        gtmp["date_day"] = pd.to_datetime(gtmp.get("date"), errors="coerce").dt.date.astype(str)
        gtmp["blue"] = gtmp.get("blue", "").astype(str)
        gtmp["red"] = gtmp.get("red", "").astype(str)
        gtmp["pair_key"] = gtmp.apply(
            lambda rr: "||".join(sorted([str(rr.get("blue") or ""), str(rr.get("red") or "")])),
            axis=1,
        )
        gtmp["gno"] = pd.to_numeric(gtmp.get("game"), errors="coerce").fillna(1).astype(int).clip(lower=1)
        grp = (
            gtmp.groupby(["date_day", "league", "split", "playoffs", "pair_key"], dropna=False)["gno"]
            .max()
            .reset_index()
            .rename(columns={"gno": "series_max"})
        )
        gtmp = gtmp.merge(
            grp,
            on=["date_day", "league", "split", "playoffs", "pair_key"],
            how="left",
        )
        series_max_by_gid = {
            str(gid): int(mx) if pd.notna(mx) else 1
            for gid, mx in zip(gtmp.get("gameid", []), gtmp.get("series_max", []))
        }
    except Exception:
        series_max_by_gid = {}

    for r in games.itertuples(index=False):
        gid = str(r.gameid)
        blue = str(r.blue)
        red = str(r.red)
        league = str(r.league or "")
        split = str(r.split or "")
        y = int(r.blue_win)
        dt = pd.to_datetime(r.date, errors="coerce")
        if not pd.notna(dt):
            continue
        playoffs_raw = getattr(r, "playoffs", 0)
        playoffs_i = _safe_int(playoffs_raw, 0)
        phase = "playoffs" if playoffs_i > 0 else "regular"

        form_n_a = int(games_count.get(blue, 0))
        form_n_b = int(games_count.get(red, 0))
        form_w_a = int(wins_count.get(blue, 0))
        form_w_b = int(wins_count.get(red, 0))

        p_raw, p_team = _sym_team_probs(art, blue, red, league)
        _gno_raw = r.game
        _gno_f = _f(_gno_raw, float("nan"))
        _gno = int(_gno_f) if math.isfinite(_gno_f) else 1
        _mx = int(series_max_by_gid.get(gid, max(1, _gno)))
        bo_bucket = "5" if _mx >= 5 else ("3" if _mx >= 3 else "1")

        n_team = float(min(int(games_count.get(blue, 0)), int(games_count.get(red, 0))))
        p_players = float(p_team)
        n_players = 0.0
        coverage_ratio = 0.0
        freshness_avg = 0.0
        transfer_sig = 0.0
        lineup_cont = 1.0
        roster_changed = False
        lane_map: Dict[str, float] = {"Top": 0.0, "Jungle": 0.0, "Mid": 0.0, "Bot": 0.0, "Support": 0.0}

        lu = lineup_cache.get(gid) or {}
        blue_lu = lu.get("blue") or {}
        red_lu = lu.get("red") or {}
        if blue_lu and red_lu:
            info = compute_delta_and_confidence(
                p_art,
                blue_lineup=blue_lu,
                red_lineup=red_lu,
                as_of=dt.date(),
                match_league=league,
                require_both_sides=False,
            )
            delta = float(info.get("delta", 0.0) or 0.0)
            for d in (info.get("details") or []):
                if not isinstance(d, dict):
                    continue
                _pos = str(d.get("pos") or "").strip()
                if _pos not in lane_map:
                    continue
                _dv = _f(d.get("lane_delta_raw", d.get("delta", 0.0)), float("nan"))
                if math.isfinite(_dv):
                    lane_map[_pos] += float(_dv)
            cov = float(info.get("coverage", 0) or 0.0) / float(info.get("coverage_total", 5) or 5.0)
            avg_gp_min = float(info.get("avg_gp_min", 0.0) or 0.0)
            freshness_avg = float(max(0.0, min(1.0, float(info.get("freshness_avg", 0.0) or 0.0))))
            coverage_ratio = float(max(0.0, min(1.0, cov)))
            details = list(info.get("details") or [])
            transfer_sig = float(
                transfer_signal_from_details(
                    details,
                    tier_priority_fn=tier_priority,
                )
            )
            lineup_cont = float(
                lineup_continuity_from_details(
                    details,
                    freshness_avg=float(freshness_avg),
                    cov_ratio=float(coverage_ratio),
                )
            )
            roster_gate = 0.55
            knee = 18.0
            if float(n_team) < float(knee):
                roster_gate += 0.10 * max(0.0, 1.0 - (float(n_team) / float(knee)))
            if float(freshness_avg) < 0.60:
                roster_gate += 0.05 * max(0.0, 1.0 - (float(freshness_avg) / 0.60))
            roster_gate = float(max(0.30, min(0.90, roster_gate)))
            roster_changed = bool(float(lineup_cont) < float(roster_gate))
            n_players = float(max(0.0, cov * avg_gp_min))

            z = float(logit(p_raw) + float(w_lineup) * delta)
            p_raw_players = float(sigmoid(z))
            p_players = float(clip_prob(float(art.calibrator.predict_proba(p_raw_players))))

        samples.append(
            Sample(
                month=_month_key(dt),
                gameid=str(gid),
                league=str(league),
                patch=str(getattr(r, "patch", "") or ""),
                split=str(split),
                macro=str(_league_macro_key(league)),
                side="blue",
                bo_bucket=str(bo_bucket),
                phase=str(phase),
                y=int(y),
                p_team=float(clip_prob(p_team)),
                p_players=float(clip_prob(p_players)),
                n_team=float(max(0.0, n_team)),
                n_players=float(max(0.0, n_players)),
                form_n_a=int(max(0, form_n_a)),
                form_n_b=int(max(0, form_n_b)),
                form_wins_a=int(max(0, form_w_a)),
                form_wins_b=int(max(0, form_w_b)),
                coverage_ratio=float(coverage_ratio),
                freshness_avg=float(freshness_avg),
                transfer_signal=float(transfer_sig),
                lineup_continuity=float(lineup_cont),
                roster_changed=bool(roster_changed),
                same_league=True,
                blue_team=str(blue),
                red_team=str(red),
                game_no=int(max(1, _gno)),
                date_day=str(dt.date().isoformat()),
                lane_top=float(lane_map.get("Top", 0.0) or 0.0),
                lane_jungle=float(lane_map.get("Jungle", 0.0) or 0.0),
                lane_mid=float(lane_map.get("Mid", 0.0) or 0.0),
                lane_bot=float(lane_map.get("Bot", 0.0) or 0.0),
                lane_support=float(lane_map.get("Support", 0.0) or 0.0),
            )
        )

        games_count[blue] += 1
        games_count[red] += 1
        if int(y) == 1:
            wins_count[blue] += 1
        else:
            wins_count[red] += 1

    return samples
