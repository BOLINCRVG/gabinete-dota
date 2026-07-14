from __future__ import annotations

import math
from typing import Any, Dict

from app_core.prob_math_core import clip_prob as _clip_prob
from app_core.prob_math_core import safe_float as _f
from app_core.prob_math_core import safe_logit as _logit
from app_core.prob_math_core import safe_sigmoid as _sigmoid


def _abs_diff_pp(layer: Dict[str, Any] | None) -> float:
    if not isinstance(layer, dict):
        return float("nan")
    for k in ("diff_pp", "diff_pp_map"):
        v = _f(layer.get(k))
        if math.isfinite(v):
            return abs(v)
    return float("nan")


def _layer_applied(layer: Dict[str, Any] | None) -> bool:
    return bool(isinstance(layer, dict) and layer.get("applied", False))


def _layer_reason(layer: Dict[str, Any] | None) -> str:
    if not isinstance(layer, dict):
        return ""
    return str(layer.get("reason", "") or "").strip().lower()


def _layer_method(layer: Dict[str, Any] | None) -> str:
    if not isinstance(layer, dict):
        return ""
    return str(layer.get("method", "") or "").strip()


def apply_layer_uncertainty_live(
    *,
    p_map_in: Any,
    p_series_in: Any,
    bo: Any = 1,
    interregional: bool = False,
    playoffs: bool = False,
    patch_status: Any = None,
    n_team_eff: Any = None,
    n_players_eff: Any = None,
    state_areal_live: Dict[str, Any] | None = None,
    state_alite_live: Dict[str, Any] | None = None,
    style_b_live: Dict[str, Any] | None = None,
    rank_sos_live: Dict[str, Any] | None = None,
    roster_continuity_live: Dict[str, Any] | None = None,
    hier_calibration_live: Dict[str, Any] | None = None,
    series_bo_dplite_live: Dict[str, Any] | None = None,
    meta_c_live: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "",
        "method": "layer_uncertainty_v2",
        "p_before": None,
        "p_after": None,
        "diff_pp": None,
        "p_series_before": None,
        "p_series_after": None,
        "diff_pp_series": None,
        "bo": None,
        "interregional": bool(interregional),
        "playoffs": bool(playoffs),
        "patch_status": str(patch_status or ""),
        "evidence_score": None,
        "uncertainty_score": None,
        "confidence_score": None,
        "shrink_map": None,
        "shrink_series": None,
        "cap_pp_map": None,
        "cap_pp_series": None,
        "components": {},
        "layer_scores": {},
        "dominant_risk_layers": [],
        "actionability_zone": None,
        "actionability_note": "",
        "applied_layers": 0,
        "missing_layers": 0,
    }
    p0 = _f(p_map_in)
    ps0 = _f(p_series_in)
    if not math.isfinite(p0):
        out["reason"] = "invalid_p_map"
        return out
    out["p_before"] = float(p0)
    if math.isfinite(ps0):
        out["p_series_before"] = float(ps0)

    try:
        bo_i = int(float(bo))
    except Exception:
        bo_i = 1
    bo_i = 5 if bo_i >= 5 else (3 if bo_i >= 3 else 1)
    out["bo"] = int(bo_i)

    nte = max(0.0, _f(n_team_eff, 0.0))
    npe = max(0.0, _f(n_players_eff, 0.0))
    evidence = max(0.0, min(1.0, (0.75 * nte + 0.25 * npe) / 65.0))
    hier_inter = {}
    if isinstance(hier_calibration_live, dict):
        hier_inter = hier_calibration_live.get("interleague") if isinstance(hier_calibration_live.get("interleague"), dict) else {}
    inter_ev = max(0.0, min(1.0, _f((hier_inter or {}).get("inter_ev"), 0.0))) if isinstance(hier_inter, dict) else 0.0
    inter_strength_gap = max(0.0, min(1.0, _f((hier_inter or {}).get("strength_gap"), 0.0))) if isinstance(hier_inter, dict) else 0.0
    inter_scope_kind = str((hier_inter or {}).get("pair_scope_kind") or "") if isinstance(hier_inter, dict) else ""
    inter_band_a = str((hier_inter or {}).get("strength_band_a") or "") if isinstance(hier_inter, dict) else ""
    inter_band_b = str((hier_inter or {}).get("strength_band_b") or "") if isinstance(hier_inter, dict) else ""
    sa_state = {}
    try:
        sa_state = (
            (((state_areal_live or {}).get("qwl_trace") or {}).get("state") or {})
            if isinstance(((state_areal_live or {}).get("qwl_trace") or {}), dict)
            else {}
        )
    except Exception:
        sa_state = {}
    fallback_cross_strength = bool((sa_state or {}).get("cross_strength_interregional", False))
    fallback_strength_gap = max(0.0, min(1.0, _f((sa_state or {}).get("strength_gap"), 0.0)))
    strong_cross_tier_interregional = bool(
        interregional
        and inter_scope_kind in {"blend", "league_pair", "tier_pair"}
        and inter_ev >= 0.70
        and inter_strength_gap >= 0.12
        and inter_band_a in {"elite", "major"}
        and inter_band_b in {"upper", "regional", "development", "local"}
    )
    if (not strong_cross_tier_interregional) and bool(interregional) and fallback_cross_strength:
        strong_cross_tier_interregional = bool(fallback_strength_gap >= 0.16)
    very_strong_cross_tier_interregional = bool(
        strong_cross_tier_interregional
        and inter_ev >= 0.85
        and inter_strength_gap >= 0.16
    )
    if (not very_strong_cross_tier_interregional) and bool(interregional) and fallback_cross_strength:
        very_strong_cross_tier_interregional = bool(fallback_strength_gap >= 0.16)
    if strong_cross_tier_interregional:
        evidence = max(evidence, min(0.92, 0.38 + (0.42 * inter_ev) + (0.55 * inter_strength_gap)))
    if very_strong_cross_tier_interregional:
        evidence = max(evidence, min(0.96, 0.55 + (0.28 * inter_ev) + (0.35 * inter_strength_gap)))

    layer_map = {
        "Areal": state_areal_live,
        "Alite": state_alite_live,
        "B": style_b_live,
        "RankSOS": rank_sos_live,
        "Roster": roster_continuity_live,
        "HierCal": hier_calibration_live,
        "SeriesBO": series_bo_dplite_live,
        "C": meta_c_live,
    }
    components: Dict[str, Any] = {}
    layer_scores: Dict[str, Any] = {}
    total_move = 0.0
    big_moves = 0
    missing_ct = 0
    applied_ct = 0
    # relative impact weights by layer type (data-risk + model-risk prior).
    w_by_layer = {
        "Areal": 1.25,
        "Alite": 1.10,
        "B": 0.95,
        "RankSOS": 0.85,
        "Roster": 0.95,
        "HierCal": 0.90,
        "SeriesBO": 0.80,
        "C": 1.05,
    }
    weighted_unc_num = 0.0
    weighted_unc_den = 0.0
    for name, layer in layer_map.items():
        dpp = _abs_diff_pp(layer)
        applied = _layer_applied(layer)
        l_reason = _layer_reason(layer)
        l_method = _layer_method(layer)
        if applied:
            applied_ct += 1
        else:
            missing_ct += 1
        if math.isfinite(dpp):
            total_move += min(20.0, dpp)
            if dpp >= 6.0:
                big_moves += 1
        components[name] = {
            "applied": bool(applied),
            "diff_pp_abs": (float(dpp) if math.isfinite(dpp) else None),
            "reason": str(layer.get("reason", "") or "") if isinstance(layer, dict) else "",
            "method": str(layer.get("method", "") or "") if isinstance(layer, dict) else "",
        }
        l_unc = 0.45
        if not applied:
            if l_reason.startswith("disabled"):
                l_unc = 0.60
            elif "error" in l_reason:
                l_unc = 0.82
            elif "missing" in l_reason or "invalid" in l_reason:
                l_unc = 0.76
            else:
                l_unc = 0.66
        else:
            if math.isfinite(dpp):
                if dpp <= 0.4:
                    l_unc = 0.22
                elif dpp <= 1.2:
                    l_unc = 0.30
                elif dpp <= 2.5:
                    l_unc = 0.40
                elif dpp <= 4.0:
                    l_unc = 0.52
                elif dpp <= 7.0:
                    l_unc = 0.64
                else:
                    l_unc = 0.74
            if "tiny_move" in l_reason:
                l_unc = min(l_unc, 0.35)
            if "warn" in l_reason:
                l_unc = max(l_unc, 0.56)
        if ("shadow" in l_method.lower()) and applied:
            l_unc = max(0.28, l_unc)
        l_unc = float(max(0.0, min(1.0, l_unc)))
        lw = float(w_by_layer.get(name, 1.0))
        weighted_unc_num += lw * l_unc
        weighted_unc_den += lw
        layer_scores[name] = {
            "weight": lw,
            "uncertainty": l_unc,
            "confidence": float(max(0.0, min(1.0, 1.0 - l_unc))),
        }

    avg_move = (total_move / max(1, applied_ct)) if applied_ct > 0 else 0.0
    patch_mult = 1.0
    ps = str(patch_status or "").upper()
    if "REFIT" in ps:
        patch_mult = 1.20
    elif "WARN" in ps:
        patch_mult = 1.08
    bo_mult = 1.12 if bo_i == 1 else (0.96 if bo_i == 3 else 0.90)
    inter_mult = 1.10 if bool(interregional) else 1.0
    if strong_cross_tier_interregional:
        inter_mult = 0.86
    if very_strong_cross_tier_interregional:
        inter_mult = 0.70
    # Raw uncertainty grows with stacked layer movement, missing layers and low evidence.
    raw_unc = (
        0.18
        + 0.30 * (1.0 - evidence)
        + 0.16 * min(1.0, avg_move / 8.0)
        + 0.08 * min(1.0, big_moves / 3.0)
        + 0.06 * min(1.0, missing_ct / 4.0)
    )
    unc = float(max(0.0, min(1.0, raw_unc * patch_mult * bo_mult * inter_mult)))
    conf = float(max(0.0, min(1.0, 1.0 - unc)))
    out["evidence_score"] = float(evidence)
    unc_layers = float(weighted_unc_num / max(1e-9, weighted_unc_den))
    # final uncertainty blends movement/evidence and layer-structural risk.
    unc = float(max(0.0, min(1.0, 0.68 * unc + 0.32 * unc_layers)))
    conf = float(max(0.0, min(1.0, 1.0 - unc)))
    out["uncertainty_score"] = float(unc)
    out["confidence_score"] = float(conf)
    out["components"] = components
    out["layer_scores"] = layer_scores
    out["applied_layers"] = int(applied_ct)
    out["missing_layers"] = int(missing_ct)
    if isinstance(hier_inter, dict) and hier_inter:
        out["interleague_context"] = {
            "strong_cross_tier": bool(strong_cross_tier_interregional),
            "very_strong_cross_tier": bool(very_strong_cross_tier_interregional),
            "inter_ev": float(inter_ev),
            "strength_gap": float(inter_strength_gap),
            "pair_scope_kind": str(inter_scope_kind),
            "strength_band_a": str(inter_band_a),
            "strength_band_b": str(inter_band_b),
        }
    _risk_rank = sorted(
        [
            {
                "layer": k,
                "uncertainty": float((v or {}).get("uncertainty", 1.0)),
                "applied": bool((components.get(k) or {}).get("applied", False)),
                "diff_pp_abs": (components.get(k) or {}).get("diff_pp_abs"),
            }
            for k, v in layer_scores.items()
        ],
        key=lambda x: float(x.get("uncertainty", 1.0)),
        reverse=True,
    )
    out["dominant_risk_layers"] = _risk_rank[:3]
    if unc >= 0.72:
        out["actionability_zone"] = "low"
        out["actionability_note"] = "Alta incerteza operacional: evite forçar preço; priorize confirmação adicional."
    elif unc >= 0.50:
        out["actionability_zone"] = "medium"
        out["actionability_note"] = "Incerteza moderada: stake conservadora e confirmação por mercado."
    else:
        out["actionability_zone"] = "high"
        out["actionability_note"] = "Boa acionabilidade: pipeline estável para execução padrão."

    # Convert uncertainty to conservative logit shrink toward 50%.
    shrink_map = 0.0 + 0.42 * unc * (0.70 + 0.30 * (1.0 - evidence))
    if bo_i == 1:
        shrink_map *= 1.08
    if bool(playoffs) and bo_i >= 5:
        shrink_map *= 0.90
    shrink_map = float(max(0.0, min(0.55, shrink_map)))
    out["shrink_map"] = float(shrink_map)

    z0 = _logit(p0)
    z1 = z0 * (1.0 - shrink_map)
    p1 = float(_sigmoid(z1))
    diff_pp = float(100.0 * (p1 - p0))
    cap_pp_map = 2.6 if bo_i == 1 else (2.0 if bo_i == 3 else 1.6)
    if very_strong_cross_tier_interregional and bo_i >= 5:
        cap_pp_map = min(cap_pp_map, 1.0)
    out["cap_pp_map"] = float(cap_pp_map)
    if abs(diff_pp) > cap_pp_map:
        p1 = float(_clip_prob(p0 + math.copysign(cap_pp_map / 100.0, diff_pp)))
        diff_pp = float(100.0 * (p1 - p0))

    p1s = ps0 if math.isfinite(ps0) else float("nan")
    diff_pp_series = float("nan")
    shrink_series = 0.0
    cap_pp_series = 0.0
    if math.isfinite(ps0):
        shrink_series = shrink_map * (0.85 if bo_i >= 5 else 0.95)
        cap_pp_series = 2.0 if bo_i >= 5 else (2.3 if bo_i == 3 else 2.6)
        zs0 = _logit(ps0)
        zs1 = zs0 * (1.0 - shrink_series)
        p1s = float(_sigmoid(zs1))
        diff_pp_series = float(100.0 * (p1s - ps0))
        if abs(diff_pp_series) > cap_pp_series:
            p1s = float(_clip_prob(ps0 + math.copysign(cap_pp_series / 100.0, diff_pp_series)))
            diff_pp_series = float(100.0 * (p1s - ps0))
    out["shrink_series"] = float(shrink_series)
    out["cap_pp_series"] = float(cap_pp_series) if cap_pp_series else None

    if abs(diff_pp) < 0.02 and (not math.isfinite(diff_pp_series) or abs(diff_pp_series) < 0.02):
        out["reason"] = "tiny_move"
        out["p_after"] = float(p0)
        out["p_series_after"] = (float(ps0) if math.isfinite(ps0) else None)
        out["diff_pp"] = 0.0
        out["diff_pp_series"] = (0.0 if math.isfinite(ps0) else None)
        return out

    out["applied"] = True
    out["reason"] = "ok"
    out["p_after"] = float(p1)
    out["diff_pp"] = float(diff_pp)
    out["p_series_after"] = (float(p1s) if math.isfinite(p1s) else None)
    out["diff_pp_series"] = (float(diff_pp_series) if math.isfinite(diff_pp_series) else None)
    return out
