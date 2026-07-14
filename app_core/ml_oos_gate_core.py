"""OOS gate evaluation and walkforward probability correction for ML confidence baseline."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mlcore.utils import safe_float_or_none as _safe_float


def evaluate_confidence_baseline_oos_gate(
    *,
    candidate: Dict[str, Any],
    previous: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cand = candidate if isinstance(candidate, dict) else {}
    prev = previous if isinstance(previous, dict) else {}
    th = cand.get("thresholds") if isinstance(cand.get("thresholds"), dict) else {}

    enabled = bool(int(float(th.get("oos_gate_enabled", 1) or 1)))
    if not enabled:
        return {"enabled": False, "pass": True, "reason": "disabled", "checks": []}

    cand_oos = cand.get("oos_eval") if isinstance(cand.get("oos_eval"), dict) else {}
    prev_oos = prev.get("oos_eval") if isinstance(prev.get("oos_eval"), dict) else {}
    if not prev_oos:
        return {
            "enabled": True,
            "pass": True,
            "reason": "no_previous_oos_eval",
            "checks": [],
        }

    ll_tol = float(max(0.0, min(0.02, _safe_float(th.get("oos_logloss_tol")) or 0.0015)))
    br_tol = float(max(0.0, min(0.02, _safe_float(th.get("oos_brier_tol")) or 0.0010)))
    ci_level = float(max(0.80, min(0.999, _safe_float(th.get("oos_ci_level")) or 0.95)))
    n_boot = int(max(200, min(5000, int((_safe_float(th.get("oos_bootstrap_samples")) or 1000)))))
    min_overlap = int(max(4, min(36, int((_safe_float(th.get("oos_min_overlap_folds")) or 6)))))

    checks: List[Dict[str, Any]] = []

    def _add_scalar_check(name: str, key: str, tol: float, lower_is_better: bool) -> None:
        c = _safe_float(cand_oos.get(key))
        p = _safe_float(prev_oos.get(key))
        if c is None or p is None:
            checks.append(
                {
                    "name": str(name),
                    "key": str(key),
                    "skipped": True,
                    "reason": "missing_metric",
                }
            )
            return
        diff = float(c - p)
        ok = bool(diff <= tol) if lower_is_better else bool(diff >= -tol)
        checks.append(
            {
                "name": str(name),
                "key": str(key),
                "candidate": float(c),
                "previous": float(p),
                "diff": float(diff),
                "tol": float(tol),
                "lower_is_better": bool(lower_is_better),
                "pass": bool(ok),
                "skipped": False,
            }
        )

    _add_scalar_check("wf_logloss_mean", "wf_logloss_mean", ll_tol, True)
    _add_scalar_check("wf_brier_mean", "wf_brier_mean", br_tol, True)
    _add_scalar_check("lv_fused_logloss", "lv_fused_logloss", ll_tol, True)
    _add_scalar_check("lv_gain_vs_team_logloss", "lv_gain_vs_team_logloss", ll_tol, False)

    def _add_fold_bootstrap_check(metric_key: str, tol: float) -> None:
        c_rows = cand_oos.get("wf_fold_metrics") if isinstance(cand_oos.get("wf_fold_metrics"), list) else []
        p_rows = prev_oos.get("wf_fold_metrics") if isinstance(prev_oos.get("wf_fold_metrics"), list) else []
        c_map: Dict[str, float] = {}
        p_map: Dict[str, float] = {}
        for _r in c_rows:
            if not isinstance(_r, dict):
                continue
            _m = str(_r.get("month") or "").strip()
            _v = _safe_float(_r.get(metric_key))
            if _m and (_v is not None):
                c_map[_m] = float(_v)
        for _r in p_rows:
            if not isinstance(_r, dict):
                continue
            _m = str(_r.get("month") or "").strip()
            _v = _safe_float(_r.get(metric_key))
            if _m and (_v is not None):
                p_map[_m] = float(_v)
        overlap = sorted(set(c_map.keys()).intersection(set(p_map.keys())))
        if len(overlap) < int(min_overlap):
            checks.append(
                {
                    "name": f"{metric_key}_paired_bootstrap",
                    "key": f"wf_fold_metrics.{metric_key}",
                    "skipped": True,
                    "reason": f"overlap_lt_min({len(overlap)}<{int(min_overlap)})",
                    "overlap": len(overlap),
                }
            )
            return
        diffs = np.array([float(c_map[m] - p_map[m]) for m in overlap], dtype=float)
        if diffs.size <= 0:
            checks.append(
                {
                    "name": f"{metric_key}_paired_bootstrap",
                    "key": f"wf_fold_metrics.{metric_key}",
                    "skipped": True,
                    "reason": "empty_diffs",
                }
            )
            return
        rng = np.random.default_rng(20260305)
        boot_idx = rng.integers(0, diffs.size, size=(int(n_boot), int(diffs.size)))
        boot_means = diffs[boot_idx].mean(axis=1)
        alpha = float(max(1e-6, min(0.499999, (1.0 - float(ci_level)) * 0.5)))
        ci_lo = float(np.quantile(boot_means, alpha))
        ci_hi = float(np.quantile(boot_means, 1.0 - alpha))
        mean_diff = float(np.mean(diffs))
        ok = bool(ci_hi <= float(tol))
        checks.append(
            {
                "name": f"{metric_key}_paired_bootstrap",
                "key": f"wf_fold_metrics.{metric_key}",
                "candidate_mean_diff": float(mean_diff),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "tol": float(tol),
                "pass": bool(ok),
                "skipped": False,
                "overlap": len(overlap),
                "n_boot": int(n_boot),
                "ci_level": float(ci_level),
            }
        )

    _add_fold_bootstrap_check("logloss", ll_tol)
    _add_fold_bootstrap_check("brier", br_tol)

    active = [c for c in checks if isinstance(c, dict) and (not bool(c.get("skipped", False)))]
    if not active:
        return {
            "enabled": True,
            "pass": True,
            "reason": "no_active_checks",
            "checks": checks,
            "params": {
                "oos_logloss_tol": float(ll_tol),
                "oos_brier_tol": float(br_tol),
                "oos_ci_level": float(ci_level),
                "oos_bootstrap_samples": int(n_boot),
                "oos_min_overlap_folds": int(min_overlap),
            },
        }
    gate_ok = bool(all(bool(c.get("pass", False)) for c in active))
    return {
        "enabled": True,
        "pass": bool(gate_ok),
        "reason": ("pass" if gate_ok else "degradation_detected"),
        "checks": checks,
        "params": {
            "oos_logloss_tol": float(ll_tol),
            "oos_brier_tol": float(br_tol),
            "oos_ci_level": float(ci_level),
            "oos_bootstrap_samples": int(n_boot),
            "oos_min_overlap_folds": int(min_overlap),
        },
    }


def wf_correct_p_map(
    report: Optional[Dict[str, Any]],
    p_in: float,
    league_hint: str = "",
    strength: float = 0.60,
) -> Tuple[float, Dict[str, Any]]:
    info: Dict[str, Any] = {"applied": False}
    p0 = _safe_float(p_in)
    if p0 is None:
        return p_in, info
    if (not math.isfinite(p0)) or p0 <= 0.0 or p0 >= 1.0:
        return p_in, info
    if not isinstance(report, dict):
        return p0, info

    s = float(max(0.0, min(1.0, strength)))
    if s <= 0.0:
        return p0, info

    shrink = 1.0
    overall = report.get("overall") if isinstance(report.get("overall"), dict) else {}
    rg = _safe_float(overall.get("roi_proxy_mean"))
    if rg is not None:
        if rg <= 0.0:
            shrink *= (1.0 - 0.10 * s)
        elif rg > 0.03:
            shrink *= (1.0 + 0.05 * s)

    league_key = str(league_hint or "").strip()
    by_league = report.get("by_league") if isinstance(report.get("by_league"), dict) else {}
    if league_key and isinstance(by_league.get(league_key), dict):
        row = by_league.get(league_key) or {}
        n_lg = int(_safe_float(row.get("n_total")) or 0)
        if n_lg >= 40:
            drift = _safe_float(row.get("drift_mean"))
            if drift is not None:
                if drift > 0.02:
                    shrink *= (1.0 - 0.15 * s)
                elif drift > 0.01:
                    shrink *= (1.0 - 0.08 * s)
                elif drift < -0.02:
                    shrink *= (1.0 + 0.06 * s)
            roi_l = _safe_float(row.get("roi_proxy_mean"))
            if roi_l is not None:
                if roi_l < 0.0:
                    shrink *= (1.0 - 0.08 * s)
                elif roi_l > 0.03:
                    shrink *= (1.0 + 0.04 * s)
            info["league_n"] = n_lg

    shrink = float(max(0.80, min(1.10, shrink)))
    p_out = float(max(0.01, min(0.99, 0.5 + (p0 - 0.5) * shrink)))
    info.update({"applied": True, "p_in": p0, "p_out": p_out, "shrink": shrink, "league": league_key})
    return p_out, info


def wf_gate_multiplier(report: Optional[Dict[str, Any]], league_hint: str = "") -> float:
    if not isinstance(report, dict):
        return 1.0
    factor = 1.0
    overall = report.get("overall") if isinstance(report.get("overall"), dict) else {}
    roi_f = _safe_float(overall.get("roi_proxy_mean"))
    if roi_f is not None:
        if roi_f <= 0.0:
            factor *= 1.30
        elif roi_f < 0.01:
            factor *= 1.15
        elif roi_f > 0.03:
            factor *= 0.90

    league_key = str(league_hint or "").strip()
    by_league = report.get("by_league") if isinstance(report.get("by_league"), dict) else {}
    if league_key and league_key in by_league and isinstance(by_league.get(league_key), dict):
        row = by_league.get(league_key) or {}
        n_lg = int(_safe_float(row.get("n_total")) or 0)
        if n_lg >= 80:
            drift = _safe_float(row.get("drift_mean"))
            if drift is not None:
                if drift > 0.02:
                    factor *= 1.10
                elif drift < -0.02:
                    factor *= 0.95
    return float(max(0.70, min(1.80, factor)))


__all__ = ["evaluate_confidence_baseline_oos_gate", "wf_correct_p_map", "wf_gate_multiplier"]
