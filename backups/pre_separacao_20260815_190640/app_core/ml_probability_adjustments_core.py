"""Pure probability adjustment functions: uncertainty shrink, marginal-trust shrink, microseg."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from mlcore.utils import clip_prob, safe_float_or_none as _safe_float


def _safe_logit(p: float, clip_prob_fn: Callable[[float], float]) -> float:
    x = float(clip_prob_fn(float(p)))
    return float(math.log(x / (1.0 - x)))


def _safe_sigmoid(z: float) -> float:
    v = float(z)
    if v >= 0.0:
        ez = math.exp(-v)
        return float(1.0 / (1.0 + ez))
    ez = math.exp(v)
    return float(ez / (1.0 + ez))


def safe_logit(p: float) -> float:
    x = float(clip_prob(float(p)))
    return float(math.log(x / (1.0 - x)))


def safe_sigmoid(z: float) -> float:
    return _safe_sigmoid(z)


def effective_games_from_ml_info(ml_info: Optional[Dict[str, Any]]) -> Dict[str, float]:
    out = {
        "n_blue": float("nan"),
        "n_red": float("nan"),
        "n_blue_eff": float("nan"),
        "n_red_eff": float("nan"),
        "n_eff": float("nan"),
    }
    if not isinstance(ml_info, dict):
        return out
    b = ml_info.get("blue") if isinstance(ml_info.get("blue"), dict) else {}
    r = ml_info.get("red") if isinstance(ml_info.get("red"), dict) else {}
    n_b = _safe_float(b.get("games_played")) or 0.0
    n_r = _safe_float(r.get("games_played")) or 0.0
    sf_b = _safe_float(b.get("shrink_factor"))
    sf_r = _safe_float(r.get("shrink_factor"))
    sf_b = float(sf_b if sf_b is not None else (1.0 if n_b > 0 else 0.0))
    sf_r = float(sf_r if sf_r is not None else (1.0 if n_r > 0 else 0.0))
    sf_b = float(max(0.0, min(1.0, sf_b)))
    sf_r = float(max(0.0, min(1.0, sf_r)))
    n_b_eff = float(max(0.0, n_b * sf_b))
    n_r_eff = float(max(0.0, n_r * sf_r))
    if n_b_eff > 0.0 and n_r_eff > 0.0:
        n_eff = float((2.0 * n_b_eff * n_r_eff) / (n_b_eff + n_r_eff))
    else:
        n_eff = float(max(n_b_eff, n_r_eff, 0.0))
    out.update(
        {
            "n_blue": float(n_b),
            "n_red": float(n_r),
            "n_blue_eff": float(n_b_eff),
            "n_red_eff": float(n_r_eff),
            "n_eff": float(n_eff),
        }
    )
    return out


def apply_ml_uncertainty_adjust(
    p_in: float,
    ml_info: Optional[Dict[str, Any]],
    *,
    clip_prob_fn: Callable[[float], float],
    enabled: bool = True,
    strength: float = 0.55,
    prior_games: float = 22.0,
    max_shrink: float = 0.65,
) -> Tuple[float, Dict[str, Any]]:
    info: Dict[str, Any] = {"enabled": bool(enabled), "applied": False}
    try:
        p0 = float(p_in)
    except (TypeError, ValueError):
        return p_in, info
    if (not math.isfinite(p0)) or p0 <= 0.0 or p0 >= 1.0 or (not bool(enabled)):
        info.update({"p_in": p0 if math.isfinite(p0) else None, "p_out": p0 if math.isfinite(p0) else None})
        return p0, info

    n_pack = effective_games_from_ml_info(ml_info)
    n_eff = float(n_pack.get("n_eff", float("nan")) or float("nan"))
    if not math.isfinite(n_eff) or n_eff < 0.0:
        n_eff = 0.0

    pg = float(max(1.0, float(prior_games)))
    s = float(max(0.0, min(1.0, float(strength))))
    cap = float(max(0.0, min(0.95, float(max_shrink))))

    raw_shrink = float(pg / (n_eff + pg))
    shrink = float(min(cap, s * raw_shrink))
    w_keep = float(max(0.0, min(1.0, 1.0 - shrink)))
    z0 = float(_safe_logit(p0, clip_prob_fn))
    z1 = float(w_keep * z0)
    p1 = float(clip_prob_fn(_safe_sigmoid(z1)))

    info.update(
        {
            "enabled": True,
            "applied": bool(abs(p1 - p0) > 1e-12),
            "space": "logit_shrink",
            "p_in": float(p0),
            "p_out": float(p1),
            "shrink": float(shrink),
            "raw_shrink": float(raw_shrink),
            "w_keep": float(w_keep),
            "logit_in": float(z0),
            "logit_out": float(z1),
            "strength": float(s),
            "prior_games": float(pg),
            "max_shrink": float(cap),
            "n_eff": float(n_eff),
            "n_blue": float(n_pack.get("n_blue", float("nan"))) if math.isfinite(float(n_pack.get("n_blue", float("nan")))) else None,
            "n_red": float(n_pack.get("n_red", float("nan"))) if math.isfinite(float(n_pack.get("n_red", float("nan")))) else None,
            "n_blue_eff": float(n_pack.get("n_blue_eff", float("nan"))) if math.isfinite(float(n_pack.get("n_blue_eff", float("nan")))) else None,
            "n_red_eff": float(n_pack.get("n_red_eff", float("nan"))) if math.isfinite(float(n_pack.get("n_red_eff", float("nan")))) else None,
        }
    )
    return p1, info


def apply_ml_marginal_trust_adjust(
    p_in: float,
    ml_info: Optional[Dict[str, Any]],
    *,
    clip_prob_fn: Callable[[float], float],
    norm_league_key_fn: Callable[[str], str],
    tier_from_league_fn: Callable[[str], str],
    ml_league_conf_band_fn: Callable[[str, str], str],
    ml_marginal_band_profile_fn: Callable[..., Dict[str, Any]],
    league_is_excluded_fn: Callable[[str, Any], bool],
    is_marginal_league_for_ml_fn: Callable[[str], bool],
    sanitize_ml_marginal_band_coeffs_fn: Callable[[Any], Dict[str, Any]],
    league: str = "",
    enabled: bool = True,
    strength: float = 0.70,
    prior_games: float = 28.0,
    max_shrink: float = 0.72,
    cap_prob: float = 0.84,
    confidence_score: Optional[float] = None,
    score_soft: float = 68.0,
    score_hard: float = 46.0,
    exclude_leagues: Any = "",
    band_tuning: bool = True,
    band_coeffs: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, Any]]:
    info: Dict[str, Any] = {"enabled": bool(enabled), "applied": False}
    try:
        p0 = float(p_in)
    except (TypeError, ValueError):
        return p_in, info
    if (not math.isfinite(p0)) or p0 <= 0.0 or p0 >= 1.0:
        info.update({"p_in": p0 if math.isfinite(p0) else None, "p_out": p0 if math.isfinite(p0) else None})
        return p0, info

    lg = str(league or "").strip()
    lk = norm_league_key_fn(lg)
    tier = tier_from_league_fn(lg)
    band = ml_league_conf_band_fn(lg, tier)
    prof = ml_marginal_band_profile_fn(band, band_coeffs=band_coeffs if isinstance(band_coeffs, dict) else None)
    is_excluded = league_is_excluded_fn(lg, exclude_leagues)
    if (not bool(enabled)) or bool(is_excluded):
        info.update(
            {
                "enabled": bool(enabled),
                "applied": False,
                "reason": "disabled" if not bool(enabled) else "excluded_league",
                "league": str(lk or lg),
                "tier": str(tier or ""),
                "band": str(band),
                "is_marginal": bool(is_marginal_league_for_ml_fn(lg) or (tier in {"T2", "MINOR"})),
                "excluded": bool(is_excluded),
                "p_in": float(p0),
                "p_out": float(p0),
            }
        )
        return p0, info

    n_pack = effective_games_from_ml_info(ml_info)
    n_eff = float(n_pack.get("n_eff", float("nan")) or float("nan"))
    if not math.isfinite(n_eff) or n_eff < 0.0:
        n_eff = 0.0

    cs = float("nan")
    try:
        if confidence_score is not None:
            cs = float(confidence_score)
    except (TypeError, ValueError):
        cs = float("nan")
    if not math.isfinite(cs):
        cs = 50.0
    cs = float(max(0.0, min(100.0, cs)))

    pg = float(max(1.0, float(prior_games)))
    s = float(max(0.0, min(1.0, float(strength))))
    cap_shrink = float(max(0.0, min(0.95, float(max_shrink))))
    cap_hi_in = float(max(0.55, min(0.95, float(cap_prob))))
    sc_soft = float(max(1.0, min(100.0, float(score_soft))))
    sc_hard = float(max(0.0, min(99.0, float(score_hard))))
    gate_score = float(prof.get("gate_score", 85.0))
    gate_n_eff = float(prof.get("gate_n_eff", 40.0))
    if bool(band_tuning):
        s = float(max(0.0, min(1.0, s * float(prof.get("strength_scale", 1.0)))))
        pg = float(max(1.0, pg * float(prof.get("prior_scale", 1.0))))
        cap_shrink = float(max(0.0, min(0.95, cap_shrink * float(prof.get("max_shrink_scale", 1.0)))))
        cap_hi_in = float(max(0.55, min(0.95, cap_hi_in + float(prof.get("cap_shift", 0.0)))))
        sc_soft = float(max(1.0, min(100.0, sc_soft + float(prof.get("soft_shift", 0.0)))))
        sc_hard = float(max(0.0, min(99.0, sc_hard + float(prof.get("hard_shift", 0.0)))))
    if sc_soft <= (sc_hard + 1.0):
        sc_soft = float(min(100.0, sc_hard + 1.0))

    if (band in {"top", "core"}) and (cs >= gate_score) and (n_eff >= gate_n_eff):
        info.update(
            {
                "enabled": True,
                "applied": False,
                "reason": "gate_strong_band",
                "league": str(lk or lg),
                "tier": str(tier or ""),
                "band": str(band),
                "is_marginal": bool(is_marginal_league_for_ml_fn(lg) or (tier in {"T2", "MINOR"})),
                "excluded": bool(is_excluded),
                "confidence_score": float(cs),
                "n_eff": float(n_eff),
                "gate_score": float(gate_score),
                "gate_n_eff": float(gate_n_eff),
                "p_in": float(p0),
                "p_out": float(p0),
            }
        )
        return p0, info

    conf_weak = float((sc_soft - cs) / max(1e-9, (sc_soft - sc_hard)))
    conf_weak = float(max(0.0, min(1.0, conf_weak)))
    evidence_weak = float(pg / (n_eff + pg))
    evidence_weak = float(max(0.0, min(1.0, evidence_weak)))
    weak = float(max(0.0, min(1.0, 0.60 * conf_weak + 0.40 * evidence_weak)))

    shrink = float(min(cap_shrink, s * weak))
    w_keep = float(max(0.0, min(1.0, 1.0 - shrink)))
    z0 = float(_safe_logit(p0, clip_prob_fn))
    z1 = float(w_keep * z0)
    p1 = float(clip_prob_fn(_safe_sigmoid(z1)))

    cap_hi_dyn = float(cap_hi_in - (0.12 * weak))
    cap_hi = float(max(0.55, min(cap_hi_in, cap_hi_dyn)))
    cap_lo = float(1.0 - cap_hi)
    p2 = float(max(cap_lo, min(cap_hi, p1)))
    cap_applied = bool(abs(p2 - p1) > 1e-12)

    info.update(
        {
            "enabled": True,
            "applied": bool(abs(p2 - p0) > 1e-12),
            "reason": "applied",
            "league": str(lk or lg),
            "tier": str(tier or ""),
            "band": str(band),
            "is_marginal": bool(is_marginal_league_for_ml_fn(lg) or (tier in {"T2", "MINOR"})),
            "excluded": bool(is_excluded),
            "p_in": float(p0),
            "p_out": float(p2),
            "p_after_shrink": float(p1),
            "cap_applied": bool(cap_applied),
            "cap_hi": float(cap_hi),
            "cap_lo": float(cap_lo),
            "strength": float(s),
            "prior_games": float(pg),
            "max_shrink": float(cap_shrink),
            "cap_prob": float(cap_hi_in),
            "score_soft": float(sc_soft),
            "score_hard": float(sc_hard),
            "band_tuning": bool(band_tuning),
            "band_profile": dict(prof),
            "band_coeff_overrides": (
                sanitize_ml_marginal_band_coeffs_fn(band_coeffs).get(str(band), {})
                if isinstance(band_coeffs, dict)
                else {}
            ),
            "gate_score": float(gate_score),
            "gate_n_eff": float(gate_n_eff),
            "confidence_score": float(cs),
            "n_eff": float(n_eff),
            "weak_conf": float(conf_weak),
            "weak_n": float(evidence_weak),
            "weak_blend": float(weak),
            "shrink": float(shrink),
            "w_keep": float(w_keep),
            "logit_in": float(z0),
            "logit_out": float(z1),
        }
    )
    return p2, info


def seg_key(league: str, split: str, side: str, bo: str, phase: str) -> str:
    return (
        f"{str(league or '').strip()}|"
        f"{str(split or '').strip()}|"
        f"{str(side or '').strip().lower()}|"
        f"{str(bo or '').strip()}|"
        f"{str(phase or '').strip().lower()}"
    )


def apply_ml_microseg_calibration(
    p_map_cal: float,
    *,
    clip_prob_fn: Callable[[float], float],
    league: str,
    split: str,
    side: str,
    bo: str,
    phase: str,
    table: Dict[str, Dict[str, float]],
    min_n: int,
    strength: float,
    cap_pp: float,
    hierarchical: bool = True,
    hierarchy_decay: float = 0.80,
) -> Tuple[float, Dict[str, Any]]:
    p0 = float(clip_prob_fn(float(p_map_cal)))
    z0 = float(_safe_logit(p0, clip_prob_fn))
    keys = [
        seg_key(league, split, side, bo, phase),
        seg_key(league, "*", side, bo, phase),
        seg_key(league, "*", side, "*", phase),
        seg_key(league, "*", "*", "*", phase),
        seg_key(league, "*", "*", "*", "*"),
        seg_key("*", "*", "*", "*", "*"),
    ]
    info: Dict[str, Any] = {
        "applied": False,
        "key": "",
        "n": 0.0,
        "bias": 0.0,
        "strength": float(strength),
        "bo": str(bo or ""),
        "phase": str(phase or ""),
        "mode": "hierarchical" if bool(hierarchical) else "first_valid",
        "space": "logit_delta",
        "keys_used": [],
    }

    if not bool(hierarchical):
        _max_n_seen = 0.0
        _table_empty = not isinstance(table, dict) or len(table) == 0
        _keys_found = 0
        for k in keys:
            d = table.get(k) if isinstance(table, dict) else None
            if not isinstance(d, dict):
                continue
            _keys_found += 1
            n = float(d.get("n", 0.0))
            _max_n_seen = max(_max_n_seen, n)
            if n < float(max(1, int(min_n))):
                continue
            b = float(d.get("bias", 0.0))
            cap = abs(float(cap_pp)) / 100.0
            if cap > 0:
                b = max(-cap, min(cap, b))
            p_bias = float(clip_prob_fn(p0 + b))
            delta_logit = float(_safe_logit(p_bias, clip_prob_fn) - z0)
            delta_logit_adj = float(float(strength) * delta_logit)
            p1 = float(clip_prob_fn(_safe_sigmoid(z0 + delta_logit_adj)))
            info.update(
                {
                    "applied": True,
                    "key": k,
                    "n": n,
                    "bias": b,
                    "bias_logit": delta_logit,
                    "bias_logit_adj": delta_logit_adj,
                    "p_before": p0,
                    "p_after": p1,
                    "keys_used": [k],
                }
            )
            return p1, info
        _fallback_reason = (
            "empty_table" if _table_empty
            else ("no_matching_keys" if _keys_found == 0 else "all_below_min_n")
        )
        info.update({
            "p_before": p0,
            "p_after": p0,
            "fallback_reason": _fallback_reason,
            "keys_tried": len(keys),
            "keys_found_in_table": _keys_found,
            "max_n_seen": _max_n_seen,
        })
        return p0, info

    dec = float(max(0.30, min(1.00, float(hierarchy_decay))))
    n_gate = float(max(1, int(min_n)))
    cands: List[Tuple[str, float, float, float, int]] = []
    _h_table_empty = not isinstance(table, dict) or len(table) == 0
    _h_keys_found = 0
    _h_max_n_seen = 0.0
    for lvl, k in enumerate(keys):
        d = table.get(k) if isinstance(table, dict) else None
        if not isinstance(d, dict):
            continue
        _h_keys_found += 1
        n = float(d.get("n", 0.0))
        _h_max_n_seen = max(_h_max_n_seen, n)
        if n <= 0.0:
            continue
        b = float(d.get("bias", 0.0))
        w_n = float(n / (n + n_gate))
        w_lvl = float(dec ** int(lvl))
        w = float(max(0.0, w_n * w_lvl))
        if w <= 0.0:
            continue
        cands.append((k, n, b, w, int(lvl)))

    if cands:
        w_sum = float(sum(x[3] for x in cands))
        if w_sum > 0.0:
            b_mix = float(sum(x[2] * x[3] for x in cands) / w_sum)
            cap = abs(float(cap_pp)) / 100.0
            if cap > 0:
                b_mix = max(-cap, min(cap, b_mix))
            p_bias = float(clip_prob_fn(p0 + b_mix))
            delta_logit = float(_safe_logit(p_bias, clip_prob_fn) - z0)
            delta_logit_adj = float(float(strength) * delta_logit)
            p1 = float(clip_prob_fn(_safe_sigmoid(z0 + delta_logit_adj)))
            best = max(cands, key=lambda x: x[3])
            info.update(
                {
                    "applied": True,
                    "key": str(best[0]),
                    "n": float(best[1]),
                    "bias": float(b_mix),
                    "bias_logit": float(delta_logit),
                    "bias_logit_adj": float(delta_logit_adj),
                    "p_before": p0,
                    "p_after": p1,
                    "hier_decay": float(dec),
                    "n_gate": float(n_gate),
                    "keys_used": [str(x[0]) for x in cands],
                    "n_eff_sum": float(sum(float(x[1]) * float(x[3]) for x in cands)),
                }
            )
            return p1, info

    _h_fallback_reason = (
        "empty_table" if _h_table_empty
        else ("no_matching_keys" if _h_keys_found == 0 else "all_zero_n")
    )
    info.update({
        "p_before": p0,
        "p_after": p0,
        "fallback_reason": _h_fallback_reason,
        "keys_tried": len(keys),
        "keys_found_in_table": _h_keys_found,
        "max_n_seen": _h_max_n_seen,
        "hier_decay": float(dec),
        "n_gate": float(n_gate),
    })
    return p0, info


def apply_series_gamma(p_series: float, gamma: float) -> float:
    try:
        p_in = float(p_series)
    except (TypeError, ValueError):
        return float("nan")
    try:
        p0 = float(clip_prob(p_in))
        g = max(0.25, float(gamma))
        z = math.log(p0 / (1.0 - p0))
        return float(clip_prob(1.0 / (1.0 + math.exp(-(z * g)))))
    except (OverflowError, TypeError, ValueError):
        return float(clip_prob(p_in))


__all__ = [
    "apply_ml_marginal_trust_adjust",
    "apply_ml_microseg_calibration",
    "apply_ml_uncertainty_adjust",
    "apply_series_gamma",
    "effective_games_from_ml_info",
    "safe_logit",
    "safe_sigmoid",
    "seg_key",
]
