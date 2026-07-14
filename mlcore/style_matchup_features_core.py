"""Style-matchup feature building for ML map probability (B layer).

Shared utilities, constants, gate, and feature builder used by both
B-lite (heuristic) and B-real (trained logistic) models.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from mlcore.utils import clip_prob as _clip_prob
from mlcore.utils import isfinite_num as _isfinite
from mlcore.utils import safe_float as _f
from mlcore.utils import logit as _safe_logit
from mlcore.league_context import (
    is_explicit_interregional_league_label as shared_is_explicit_interregional_league_label,
    league_macro_key,
)
from mlcore.tiers import league_strength_score, tier_for_league


def _combine_mean(a_for: Any, b_against: Any) -> float:
    af = _f(a_for)
    ba = _f(b_against)
    if not math.isfinite(af) or not math.isfinite(ba):
        return float("nan")
    return float(0.5 * (af + ba))


def _safe_ratio(num: Any, den: Any, default: float = float("nan")) -> float:
    n = _f(num, float("nan"))
    d = _f(den, float("nan"))
    if not math.isfinite(n) or not math.isfinite(d) or d <= 0.0:
        return float(default)
    return float(n / d)


def _is_explicit_interregional_league(league: Any) -> bool:
    return bool(shared_is_explicit_interregional_league_label(league))


def _macro_key(league: str) -> str:
    return str(league_macro_key(league, unknown="OTHER"))


def _patch_bucket_from_status(patch_status: str) -> str:
    ps = str(patch_status or "").strip().lower()
    if ps in {"refit", "refit_needed", "changed"}:
        return "refit"
    return "stable"


def _tier_rank(league: Any) -> int:
    tier = str(tier_for_league(str(league or "")) or "OTHER").upper()
    return {
        "T1": 0,
        "T2": 1,
        "ERL": 2,
        "MINOR": 3,
        "OTHER": 4,
    }.get(tier, 4)


def _phase_bucket_from_expected(*, exp_time_min: float, exp_total_kills: float) -> str:
    """Coarse expected game-phase bucket from pregame expectations.

    Uses only pregame profile-derived expectations (no target-game leakage).
    Buckets:
      - early: faster expected game / high kill pace
      - late: slower expected game / low kill pace
      - mid: otherwise
    """
    t = _f(exp_time_min, float("nan"))
    k = _f(exp_total_kills, float("nan"))
    if not math.isfinite(t) or t <= 0.0:
        return "mid"
    kpm = (k / t) if math.isfinite(k) and k > 0.0 else float("nan")
    if t <= 30.5 and (not math.isfinite(kpm) or kpm >= 0.90):
        return "early"
    if t >= 33.5 or (math.isfinite(kpm) and kpm <= 0.72):
        return "late"
    return "mid"


def should_apply_style_matchup_b_model_gate(
    *,
    bo: Any = None,
    n_team_eff: Any = None,
    n_players_eff: Any = None,
    pA: Any = None,
    pB: Any = None,
    interregional: bool = False,
    patch_status: str = "",
    roster_changed: bool = False,
) -> Dict[str, Any]:
    """Context gate for B-model/B-lite application.

    Goal: avoid noisy style over-adjustment on weak BO1 spots while preserving
    value in series and well-sampled intra-league matchups.
    """
    try:
        bo_i = int(float(bo)) if bo is not None else 1
    except Exception:
        bo_i = 1
    bo_i = 5 if bo_i >= 5 else (3 if bo_i >= 3 else 1)
    try:
        nt = float(n_team_eff) if n_team_eff is not None else 0.0
    except Exception:
        nt = 0.0
    try:
        np_ = float(n_players_eff) if n_players_eff is not None else 0.0
    except Exception:
        np_ = 0.0
    try:
        nA = float(getattr(pA, "n_games", 0) or 0)
    except Exception:
        nA = 0.0
    try:
        nB = float(getattr(pB, "n_games", 0) or 0)
    except Exception:
        nB = 0.0
    n_min_prof = float(min(max(0.0, nA), max(0.0, nB)))
    inter_b = bool(interregional)
    patch_s = str(patch_status or "").strip().lower()
    roster_b = bool(roster_changed)

    if bo_i >= 5:
        return {"allow": True, "reason": "bo5", "bo": bo_i}
    if bo_i >= 3 and n_min_prof >= 4.0:
        return {"allow": True, "reason": "series_profile_ok", "bo": bo_i}
    if n_min_prof < 3.0:
        return {"allow": False, "reason": "low_profile_sample", "bo": bo_i}
    if inter_b and bo_i == 1:
        if nt < 18.0 and np_ < 5.0:
            return {"allow": False, "reason": "bo1_interregional_low_evidence", "bo": bo_i}
        if roster_b and patch_s in {"changed", "refit", "refit_needed"}:
            return {"allow": False, "reason": "bo1_interregional_patch_roster", "bo": bo_i}
    if bo_i == 1 and nt < 12.0 and n_min_prof < 5.0:
        return {"allow": False, "reason": "bo1_low_evidence", "bo": bo_i}
    return {"allow": True, "reason": "default_allow", "bo": bo_i}


STYLE_MATCHUP_B_MODEL_FEATURES: List[str] = [
    "z_in",
    "log1p_n_min",
    "log1p_n_a",
    "log1p_n_b",
    "style_score",
    "raw_delta_logit",
    "w_sample",
    "w_context",
    "sample_imbalance",
    "kills_edge",
    "towers_edge",
    "dragons_edge",
    "barons_edge",
    "inhib_edge",
    "pace_edge",
    "volatility_edge",
    "interregional",
    "unknown_macro",
    "patch_refit",
    "roster_changed",
    "bo_is_1",
    "bo_is_3",
    "bo_is_5",
    "is_playoffs",
    "same_league",
    "bo5_playoffs",
    "bo1_interregional",
    "recent_style_score",
    "recent_minus_base_style",
    "recent_log1p_n_min",
    "recent_available",
    "recent_weight",
    "recent_disagreement_abs",
]

STYLE_MATCHUP_QUALITY_TERM_KEYS: List[str] = [
    "quality_objective_control_edge",
    "quality_closeout_edge",
    "quality_resilience_edge",
    "quality_net_control_edge",
    "quality_conversion_edge",
    "quality_containment_edge",
]

STYLE_MATCHUP_TERM_WEIGHTS: Dict[str, float] = {
    "kills_edge": 0.17,
    "towers_edge": 0.15,
    "dragons_edge": 0.13,
    "barons_edge": 0.10,
    "inhib_edge": 0.05,
    "pace_edge": 0.03,
    "volatility_edge": 0.02,
    "quality_objective_control_edge": 0.15,
    "quality_closeout_edge": 0.03,
    "quality_resilience_edge": 0.02,
    "quality_net_control_edge": 0.07,
    "quality_conversion_edge": 0.05,
    "quality_containment_edge": 0.03,
}


def _score_terms_to_style_score(score_terms: Dict[str, float]) -> Tuple[float, float]:
    w_sum = 0.0
    s_sum = 0.0
    for k, v in score_terms.items():
        w = float(STYLE_MATCHUP_TERM_WEIGHTS.get(k, 0.0))
        if w <= 0.0:
            continue
        s_sum += float(v) * w
        w_sum += w
    if w_sum <= 0.0:
        return float("nan"), 0.0
    return float(s_sum / w_sum), float(w_sum)


def _norm_key_upper(x: Any) -> str:
    return str(x or "").strip().upper()


def _extract_quality_scale_value(v: Any) -> Optional[float]:
    if isinstance(v, dict):
        for k in ("scale", "value", "quality_scale"):
            if k in v:
                sv = _f(v.get(k), float("nan"))
                if math.isfinite(sv):
                    return float(sv)
        return None
    sv = _f(v, float("nan"))
    if math.isfinite(sv):
        return float(sv)
    return None


def _resolve_quality_term_scale(
    *,
    model_cfg: Dict[str, Any],
    league_a: str,
    league_b: str,
    patch_status: str,
    feat: Dict[str, Any],
) -> Dict[str, Any]:
    out = {
        "scale": 1.0,
        "global_scale": 1.0,
        "selected_key": "global",
        "selected_raw": 1.0,
        "source": "default",
        "min_scale": 0.70,
        "max_scale": 1.35,
    }
    if not isinstance(model_cfg, dict):
        return out

    qcfg = model_cfg.get("quality_term_scale")
    qcfg = qcfg if isinstance(qcfg, dict) else {}
    map_node = qcfg.get("map") if isinstance(qcfg.get("map"), dict) else {}
    if not map_node and isinstance(model_cfg.get("quality_scale_map"), dict):
        map_node = dict(model_cfg.get("quality_scale_map") or {})
    source = "quality_term_scale.map" if map_node else "global_only"

    global_scale = _f(qcfg.get("global"), float("nan"))
    if not math.isfinite(global_scale):
        global_scale = _f(model_cfg.get("quality_scale_global"), 1.0)
    min_scale = _f(qcfg.get("min"), 0.70)
    max_scale = _f(qcfg.get("max"), 1.35)
    if not math.isfinite(min_scale):
        min_scale = 0.70
    if not math.isfinite(max_scale):
        max_scale = 1.35
    if max_scale < min_scale:
        min_scale, max_scale = max_scale, min_scale
    min_scale = float(max(0.50, min(min_scale, 1.25)))
    max_scale = float(max(1.00, min(max_scale, 1.80)))
    global_scale = float(max(min_scale, min(max_scale, float(global_scale))))

    la = _norm_key_upper(league_a)
    lb = _norm_key_upper(league_b)
    macro_a = _macro_key(league_a)
    macro_b = _macro_key(league_b)
    patch_bucket = _patch_bucket_from_status(patch_status)
    macros_feat = ((feat.get("style") or {}).get("macros") if isinstance((feat.get("style") or {}), dict) else {}) or {}
    phase_bucket = str(macros_feat.get("phase_bucket") or "").strip().lower()
    if phase_bucket not in {"early", "mid", "late"}:
        phase_bucket = "mid"

    keys_try: List[str] = []
    if la and lb:
        if la == lb:
            keys_try.extend(
                [
                    f"league:{la}|patch:{patch_bucket}|phase:{phase_bucket}",
                    f"league:{la}|patch:{patch_bucket}",
                    f"league:{la}|phase:{phase_bucket}",
                    f"league:{la}",
                ]
            )
        else:
            lp = "|".join(sorted([la, lb]))
            keys_try.extend(
                [
                    f"league_pair:{lp}|patch:{patch_bucket}|phase:{phase_bucket}",
                    f"league_pair:{lp}|patch:{patch_bucket}",
                    f"league_pair:{lp}|phase:{phase_bucket}",
                    f"league_pair:{lp}",
                ]
            )
    if macro_a and macro_b and macro_a == macro_b:
        keys_try.extend(
            [
                f"macro:{macro_a}|patch:{patch_bucket}|phase:{phase_bucket}",
                f"macro:{macro_a}|patch:{patch_bucket}",
                f"macro:{macro_a}|phase:{phase_bucket}",
                f"macro:{macro_a}",
            ]
        )
    keys_try.extend(
        [
            f"global|patch:{patch_bucket}|phase:{phase_bucket}",
            f"global|patch:{patch_bucket}",
            f"global|phase:{phase_bucket}",
            f"patch:{patch_bucket}|phase:{phase_bucket}",
            f"patch:{patch_bucket}",
            f"phase:{phase_bucket}",
            "global",
        ]
    )

    selected_key = "global"
    selected_raw = float(global_scale)
    candidates: List[Tuple[str, float, float]] = []
    ctx_models = model_cfg.get("context_models") if isinstance(model_cfg.get("context_models"), dict) else {}
    for i, k in enumerate(keys_try):
        if not isinstance(map_node, dict):
            break
        if k not in map_node:
            continue
        sv = _extract_quality_scale_value(map_node.get(k))
        if sv is None:
            continue
        if selected_key == "global":
            selected_key = str(k)
            selected_raw = float(sv)
        # Hierarchical weighting:
        # - earlier keys are more specific (higher base)
        # - context sample size (if available) raises reliability
        # - very specific patch+phase contexts are slightly regularized
        rank_w = float(max(0.20, min(1.0, 0.90 ** i)))
        n_ctx = 0.0
        if isinstance(ctx_models, dict) and isinstance(ctx_models.get(k), dict):
            n_ctx = max(0.0, _f((ctx_models.get(k) or {}).get("n_total"), 0.0))
        rel_w = float(0.40 + 0.60 * (n_ctx / (n_ctx + 220.0))) if n_ctx > 0.0 else 0.55
        if "|patch:" in str(k) and "|phase:" in str(k):
            rel_w *= 0.92
        elif "|patch:" in str(k):
            rel_w *= 0.96
        if str(k).startswith("global|"):
            rel_w *= 0.95
        if (macro_a and macro_b and macro_a != macro_b) and str(k).startswith("league:"):
            # avoid letting single-league calibration dominate cross-macro matches
            rel_w *= 0.90
        w = float(max(0.05, min(1.0, rank_w * rel_w)))
        candidates.append((str(k), float(sv), float(w)))

    blended_raw = float(selected_raw)
    if candidates:
        den = float(sum(w for _, _, w in candidates))
        if den > 1e-9:
            blended_raw = float(sum(v * w for _, v, w in candidates) / den)
        source = "quality_term_scale.map_blend_v2"

    # Conservative blend with global to avoid context overfit.
    # If we have a blended hierarchical estimate, trust it a bit more than single-key pick.
    alpha = 0.80 if candidates else 0.75
    scale = float(global_scale + (alpha * (float(blended_raw) - float(global_scale))))
    scale = float(max(min_scale, min(max_scale, scale)))
    out.update(
        {
            "scale": float(scale),
            "global_scale": float(global_scale),
            "selected_key": str(selected_key),
            "selected_raw": float(selected_raw),
            "blended_raw": float(blended_raw),
            "candidates_used": len(candidates),
            "source": str(source),
            "min_scale": float(min_scale),
            "max_scale": float(max_scale),
        }
    )
    return out


def _apply_quality_scale_to_features(feat: Dict[str, Any], scale: float) -> Dict[str, Any]:
    s = _f(scale, float("nan"))
    if not math.isfinite(s) or abs(float(s) - 1.0) < 1e-9:
        return feat
    style = (feat.get("style") if isinstance(feat.get("style"), dict) else {}) or {}
    terms = (style.get("terms") if isinstance(style.get("terms"), dict) else {}) or {}
    if not terms:
        return feat

    terms_new = dict(terms)
    changed = False
    for k in STYLE_MATCHUP_QUALITY_TERM_KEYS:
        if k not in terms_new:
            continue
        vv = _f(terms_new.get(k), float("nan"))
        if not math.isfinite(vv):
            continue
        terms_new[k] = float(vv * float(s))
        changed = True
    if not changed:
        return feat

    features = dict((feat.get("features") if isinstance(feat.get("features"), dict) else {}) or {})
    for k in STYLE_MATCHUP_QUALITY_TERM_KEYS:
        if k not in features:
            continue
        vv = _f(features.get(k), float("nan"))
        if not math.isfinite(vv):
            continue
        features[k] = float(vv * float(s))

    style_score, w_sum = _score_terms_to_style_score({k: float(v) for k, v in terms_new.items() if _isfinite(v)})
    if w_sum > 0.0 and math.isfinite(style_score):
        style_score = float(max(-2.5, min(2.5, style_score)))
        n_min = float(max(0.0, math.expm1(max(0.0, _f(features.get("log1p_n_min"), 0.0)))))
        macros = (style.get("macros") if isinstance(style.get("macros"), dict) else {}) or {}
        interregional = bool(macros.get("interregional", _f(features.get("interregional"), 0.0) >= 0.5))
        delta_cap = 0.75 if n_min >= 10.0 else 0.55
        if interregional:
            delta_cap *= 0.80
        raw_delta_logit = float(max(-delta_cap, min(delta_cap, 0.38 * style_score)))
        features["style_score"] = float(style_score)
        features["raw_delta_logit"] = float(raw_delta_logit)
        style["score"] = float(style_score)
        style["raw_delta_logit"] = float(raw_delta_logit)
        style["delta_cap"] = float(delta_cap)
    style["terms"] = terms_new

    out_feat = dict(feat)
    out_feat["style"] = style
    out_feat["features"] = features
    out_feat["vector"] = [float(features.get(k, 0.0)) for k in STYLE_MATCHUP_B_MODEL_FEATURES]
    return out_feat


def build_style_matchup_b_model_features(
    *,
    p_map_in: Any,
    pA: Any,
    pB: Any,
    league_a: str = "",
    league_b: str = "",
    patch_status: str = "",
    roster_changed: bool = False,
    bo: Any = None,
    is_playoffs: bool = False,
    pA_recent: Any = None,
    pB_recent: Any = None,
) -> Dict[str, Any]:
    if not _isfinite(p_map_in):
        return {"ok": False, "reason": "invalid_p_in", "features": {}, "vector": []}
    if pA is None or pB is None:
        return {"ok": False, "reason": "missing_profiles", "features": {}, "vector": []}

    nA = max(0.0, _f(getattr(pA, "n_games", 0), 0.0))
    nB = max(0.0, _f(getattr(pB, "n_games", 0), 0.0))
    if min(nA, nB) < 3.0:
        return {"ok": False, "reason": "low_sample_profiles", "features": {}, "vector": []}

    p0 = _clip_prob(_f(p_map_in))
    z0 = _safe_logit(p0)

    exp_kills_A = _combine_mean(getattr(pA, "kills_for", None), getattr(pB, "kills_against", None))
    exp_kills_B = _combine_mean(getattr(pB, "kills_for", None), getattr(pA, "kills_against", None))
    exp_towers_A = _combine_mean(getattr(pA, "towers_for", None), getattr(pB, "towers_against", None))
    exp_towers_B = _combine_mean(getattr(pB, "towers_for", None), getattr(pA, "towers_against", None))
    exp_dragons_A = _combine_mean(getattr(pA, "dragons_for", None), getattr(pB, "dragons_against", None))
    exp_dragons_B = _combine_mean(getattr(pB, "dragons_for", None), getattr(pA, "dragons_against", None))
    exp_barons_A = _combine_mean(getattr(pA, "barons_for", None), getattr(pB, "barons_against", None))
    exp_barons_B = _combine_mean(getattr(pB, "barons_for", None), getattr(pA, "barons_against", None))
    exp_inhib_A = _combine_mean(getattr(pA, "inhib_for", None), getattr(pB, "inhib_against", None))
    exp_inhib_B = _combine_mean(getattr(pB, "inhib_for", None), getattr(pA, "inhib_against", None))

    paceA = _f(getattr(pA, "time_min", None))
    paceB = _f(getattr(pB, "time_min", None))
    pace_delta = (paceB - paceA) if (math.isfinite(paceA) and math.isfinite(paceB)) else 0.0
    volA = _f(getattr(pA, "kills_for_sd", None), 0.0) + _f(getattr(pA, "kills_against_sd", None), 0.0)
    volB = _f(getattr(pB, "kills_for_sd", None), 0.0) + _f(getattr(pB, "kills_against_sd", None), 0.0)
    vol_delta = float(volA - volB)

    # Causal quality-of-result proxies built only from pregame team profiles.
    # They improve separation between "wins by control/closeout" vs noisy records.
    kfA = _f(getattr(pA, "kills_for", None), float("nan"))
    kaA = _f(getattr(pA, "kills_against", None), float("nan"))
    tfA = _f(getattr(pA, "towers_for", None), float("nan"))
    taA = _f(getattr(pA, "towers_against", None), float("nan"))
    dfA = _f(getattr(pA, "dragons_for", None), float("nan"))
    daA = _f(getattr(pA, "dragons_against", None), float("nan"))
    bfA = _f(getattr(pA, "barons_for", None), float("nan"))
    baA = _f(getattr(pA, "barons_against", None), float("nan"))
    infA = _f(getattr(pA, "inhib_for", None), float("nan"))
    inaA = _f(getattr(pA, "inhib_against", None), float("nan"))

    kfB = _f(getattr(pB, "kills_for", None), float("nan"))
    kaB = _f(getattr(pB, "kills_against", None), float("nan"))
    tfB = _f(getattr(pB, "towers_for", None), float("nan"))
    taB = _f(getattr(pB, "towers_against", None), float("nan"))
    dfB = _f(getattr(pB, "dragons_for", None), float("nan"))
    daB = _f(getattr(pB, "dragons_against", None), float("nan"))
    bfB = _f(getattr(pB, "barons_for", None), float("nan"))
    baB = _f(getattr(pB, "barons_against", None), float("nan"))
    infB = _f(getattr(pB, "inhib_for", None), float("nan"))
    inaB = _f(getattr(pB, "inhib_against", None), float("nan"))

    obj_for_A = (tfA + 1.1 * dfA + 1.7 * bfA + 1.3 * infA) if all(
        math.isfinite(v) for v in [tfA, dfA, bfA, infA]
    ) else float("nan")
    obj_agn_A = (taA + 1.1 * daA + 1.7 * baA + 1.3 * inaA) if all(
        math.isfinite(v) for v in [taA, daA, baA, inaA]
    ) else float("nan")
    obj_for_B = (tfB + 1.1 * dfB + 1.7 * bfB + 1.3 * infB) if all(
        math.isfinite(v) for v in [tfB, dfB, bfB, infB]
    ) else float("nan")
    obj_agn_B = (taB + 1.1 * daB + 1.7 * baB + 1.3 * inaB) if all(
        math.isfinite(v) for v in [taB, daB, baB, inaB]
    ) else float("nan")

    q_obj_ctrl_A = _safe_ratio(obj_for_A + 1.0, (obj_for_A + obj_agn_A + 2.0))
    q_obj_ctrl_B = _safe_ratio(obj_for_B + 1.0, (obj_for_B + obj_agn_B + 2.0))
    q_closeout_A = _safe_ratio(obj_for_A, max(1e-6, paceA)) if math.isfinite(obj_for_A) and math.isfinite(paceA) else float("nan")
    q_closeout_B = _safe_ratio(obj_for_B, max(1e-6, paceB)) if math.isfinite(obj_for_B) and math.isfinite(paceB) else float("nan")
    q_resilience_A = _safe_ratio(obj_for_A + 1.0, (max(0.0, kaA) + 1.0)) if math.isfinite(obj_for_A) and math.isfinite(kaA) else float("nan")
    q_resilience_B = _safe_ratio(obj_for_B + 1.0, (max(0.0, kaB) + 1.0)) if math.isfinite(obj_for_B) and math.isfinite(kaB) else float("nan")
    q_net_ctrl_A = _safe_ratio((obj_for_A - obj_agn_A), (obj_for_A + obj_agn_A + 2.0)) if math.isfinite(obj_for_A) and math.isfinite(obj_agn_A) else float("nan")
    q_net_ctrl_B = _safe_ratio((obj_for_B - obj_agn_B), (obj_for_B + obj_agn_B + 2.0)) if math.isfinite(obj_for_B) and math.isfinite(obj_agn_B) else float("nan")
    q_convert_A = _safe_ratio(obj_for_A, (max(0.0, kfA) + 2.0)) if math.isfinite(obj_for_A) and math.isfinite(kfA) else float("nan")
    q_convert_B = _safe_ratio(obj_for_B, (max(0.0, kfB) + 2.0)) if math.isfinite(obj_for_B) and math.isfinite(kfB) else float("nan")
    q_contain_A = _safe_ratio(obj_agn_A, (max(0.0, kaA) + 2.0)) if math.isfinite(obj_agn_A) and math.isfinite(kaA) else float("nan")
    q_contain_B = _safe_ratio(obj_agn_B, (max(0.0, kaB) + 2.0)) if math.isfinite(obj_agn_B) and math.isfinite(kaB) else float("nan")

    score_terms: Dict[str, float] = {}
    if math.isfinite(exp_kills_A) and math.isfinite(exp_kills_B):
        score_terms["kills_edge"] = float((exp_kills_A - exp_kills_B) / 8.0)
    if math.isfinite(exp_towers_A) and math.isfinite(exp_towers_B):
        score_terms["towers_edge"] = float((exp_towers_A - exp_towers_B) / 3.0)
    if math.isfinite(exp_dragons_A) and math.isfinite(exp_dragons_B):
        score_terms["dragons_edge"] = float((exp_dragons_A - exp_dragons_B) / 1.8)
    if math.isfinite(exp_barons_A) and math.isfinite(exp_barons_B):
        score_terms["barons_edge"] = float((exp_barons_A - exp_barons_B) / 0.9)
    if math.isfinite(exp_inhib_A) and math.isfinite(exp_inhib_B):
        score_terms["inhib_edge"] = float((exp_inhib_A - exp_inhib_B) / 1.2)
    if math.isfinite(pace_delta):
        score_terms["pace_edge"] = float(pace_delta / 8.0)
    if math.isfinite(vol_delta):
        score_terms["volatility_edge"] = float((-vol_delta) / 10.0)
    if math.isfinite(q_obj_ctrl_A) and math.isfinite(q_obj_ctrl_B):
        score_terms["quality_objective_control_edge"] = float((q_obj_ctrl_A - q_obj_ctrl_B) / 0.20)
    if math.isfinite(q_closeout_A) and math.isfinite(q_closeout_B):
        score_terms["quality_closeout_edge"] = float((q_closeout_A - q_closeout_B) / 0.55)
    if math.isfinite(q_resilience_A) and math.isfinite(q_resilience_B):
        score_terms["quality_resilience_edge"] = float((q_resilience_A - q_resilience_B) / 0.90)
    if math.isfinite(q_net_ctrl_A) and math.isfinite(q_net_ctrl_B):
        score_terms["quality_net_control_edge"] = float((q_net_ctrl_A - q_net_ctrl_B) / 0.22)
    if math.isfinite(q_convert_A) and math.isfinite(q_convert_B):
        score_terms["quality_conversion_edge"] = float((q_convert_A - q_convert_B) / 0.45)
    if math.isfinite(q_contain_A) and math.isfinite(q_contain_B):
        score_terms["quality_containment_edge"] = float((q_contain_B - q_contain_A) / 0.28)
    if not score_terms:
        return {"ok": False, "reason": "no_valid_style_features", "features": {}, "vector": []}

    w_sum = 0.0
    s_sum = 0.0
    for k, v in score_terms.items():
        w = float(STYLE_MATCHUP_TERM_WEIGHTS.get(k, 0.0))
        if w <= 0.0:
            continue
        s_sum += float(v) * w
        w_sum += w
    if w_sum <= 0.0:
        return {"ok": False, "reason": "invalid_style_weights", "features": {}, "vector": []}

    style_score = float(s_sum / w_sum)
    n_min = float(min(nA, nB))
    w_sample = float(max(0.15, min(1.0, n_min / (n_min + 8.0))))
    sample_imbalance = float(max(-2.5, min(2.5, math.log1p(max(nA, 0.0)) - math.log1p(max(nB, 0.0)))))
    macro_a = _macro_key(league_a)
    macro_b = _macro_key(league_b)
    interregional = bool(
        (macro_a and macro_b and macro_a != macro_b)
        or _is_explicit_interregional_league(league_a)
        or _is_explicit_interregional_league(league_b)
    )
    unknown_macro = bool(macro_a == "OTHER" or macro_b == "OTHER")
    patch_s = str(patch_status or "").strip().lower()
    try:
        bo_i = int(float(bo)) if bo is not None else 1
    except Exception:
        bo_i = 1
    if bo_i not in {1, 2, 3, 5}:
        bo_i = 1
    same_league = bool(
        str(league_a or "").strip().upper()
        and str(league_a or "").strip().upper() == str(league_b or "").strip().upper()
        and (not _is_explicit_interregional_league(league_a))
        and (not _is_explicit_interregional_league(league_b))
    )
    tier_gap = abs(_tier_rank(league_a) - _tier_rank(league_b)) if interregional else 0
    cross_tier_interregional = bool(interregional and tier_gap >= 1)
    strength_gap = abs(
        float(max(0.0, min(1.0, league_strength_score(league_a))))
        - float(max(0.0, min(1.0, league_strength_score(league_b))))
    ) if interregional else 0.0
    cross_strength_interregional = bool(interregional and strength_gap >= 0.16)

    w_ctx = 1.0
    interregional_relief = 0.0
    if interregional:
        sample_support = float(max(0.0, min(1.0, n_min / (n_min + 10.0)))) if n_min > 0.0 else 0.0
        interregional_relief = float(max(0.0, min(0.16, (0.10 * sample_support) + (0.03 if bo_i >= 3 else 0.0))))
        if patch_s in {"refit", "refit_needed", "changed"}:
            interregional_relief *= 0.80
        if roster_changed:
            interregional_relief *= 0.90
        w_ctx *= float(0.70 + interregional_relief)
        if cross_tier_interregional:
            w_ctx *= (0.78 if tier_gap == 1 else 0.66)
        if cross_strength_interregional:
            w_ctx *= (0.76 if strength_gap < 0.22 else 0.62)
    elif unknown_macro:
        w_ctx *= 0.82
    if patch_s in {"refit", "refit_needed", "changed"}:
        w_ctx *= 0.86
    if roster_changed:
        w_ctx *= 0.95

    style_score = float(max(-2.5, min(2.5, style_score)))
    raw_delta_logit = float(0.38 * style_score)
    delta_cap = 0.55
    if n_min >= 10.0:
        delta_cap = 0.75
    if interregional:
        delta_cap *= float(0.80 + (0.12 * interregional_relief))
    if cross_tier_interregional:
        delta_cap *= (0.82 if tier_gap == 1 else 0.70)
    if cross_strength_interregional:
        delta_cap *= (0.80 if strength_gap < 0.22 else 0.68)
    raw_delta_logit = float(max(-delta_cap, min(delta_cap, raw_delta_logit)))
    w_final = float(max(0.10, min(0.55, w_sample * w_ctx)))

    # Rolling recent style overlay (optional, no-leakage if caller builds profiles from pregame history).
    recent_style_score = 0.0
    recent_minus_base_style = 0.0
    recent_log1p_n_min = 0.0
    recent_available = 0.0
    recent_weight = 0.0
    recent_disagreement_abs = 0.0
    if pA_recent is not None and pB_recent is not None:
        try:
            nA_r = max(0.0, _f(getattr(pA_recent, "n_games", 0), 0.0))
            nB_r = max(0.0, _f(getattr(pB_recent, "n_games", 0), 0.0))
            n_min_r = float(min(nA_r, nB_r))
            if n_min_r >= 2.0:
                r_terms: Dict[str, float] = {}
                rkA = _combine_mean(getattr(pA_recent, "kills_for", None), getattr(pB_recent, "kills_against", None))
                rkB = _combine_mean(getattr(pB_recent, "kills_for", None), getattr(pA_recent, "kills_against", None))
                rtA = _combine_mean(getattr(pA_recent, "towers_for", None), getattr(pB_recent, "towers_against", None))
                rtB = _combine_mean(getattr(pB_recent, "towers_for", None), getattr(pA_recent, "towers_against", None))
                rdA = _combine_mean(getattr(pA_recent, "dragons_for", None), getattr(pB_recent, "dragons_against", None))
                rdB = _combine_mean(getattr(pB_recent, "dragons_for", None), getattr(pA_recent, "dragons_against", None))
                rbA = _combine_mean(getattr(pA_recent, "barons_for", None), getattr(pB_recent, "barons_against", None))
                rbB = _combine_mean(getattr(pB_recent, "barons_for", None), getattr(pA_recent, "barons_against", None))
                riA = _combine_mean(getattr(pA_recent, "inhib_for", None), getattr(pB_recent, "inhib_against", None))
                riB = _combine_mean(getattr(pB_recent, "inhib_for", None), getattr(pA_recent, "inhib_against", None))
                rpA = _f(getattr(pA_recent, "time_min", None))
                rpB = _f(getattr(pB_recent, "time_min", None))
                rvA = _f(getattr(pA_recent, "kills_for_sd", None), 0.0) + _f(getattr(pA_recent, "kills_against_sd", None), 0.0)
                rvB = _f(getattr(pB_recent, "kills_for_sd", None), 0.0) + _f(getattr(pB_recent, "kills_against_sd", None), 0.0)

                if math.isfinite(rkA) and math.isfinite(rkB):
                    r_terms["kills_edge"] = float((rkA - rkB) / 8.0)
                if math.isfinite(rtA) and math.isfinite(rtB):
                    r_terms["towers_edge"] = float((rtA - rtB) / 3.0)
                if math.isfinite(rdA) and math.isfinite(rdB):
                    r_terms["dragons_edge"] = float((rdA - rdB) / 1.8)
                if math.isfinite(rbA) and math.isfinite(rbB):
                    r_terms["barons_edge"] = float((rbA - rbB) / 0.9)
                if math.isfinite(riA) and math.isfinite(riB):
                    r_terms["inhib_edge"] = float((riA - riB) / 1.2)
                if math.isfinite(rpA) and math.isfinite(rpB):
                    r_terms["pace_edge"] = float((rpB - rpA) / 8.0)
                if math.isfinite(rvA) and math.isfinite(rvB):
                    r_terms["volatility_edge"] = float((-(rvA - rvB)) / 10.0)

                rkfA = _f(getattr(pA_recent, "kills_for", None), float("nan"))
                rkaA = _f(getattr(pA_recent, "kills_against", None), float("nan"))
                rtfA = _f(getattr(pA_recent, "towers_for", None), float("nan"))
                rtaA = _f(getattr(pA_recent, "towers_against", None), float("nan"))
                rdfA = _f(getattr(pA_recent, "dragons_for", None), float("nan"))
                rdaA = _f(getattr(pA_recent, "dragons_against", None), float("nan"))
                rbfA = _f(getattr(pA_recent, "barons_for", None), float("nan"))
                rbaA = _f(getattr(pA_recent, "barons_against", None), float("nan"))
                rifA = _f(getattr(pA_recent, "inhib_for", None), float("nan"))
                riaA = _f(getattr(pA_recent, "inhib_against", None), float("nan"))
                rkfB = _f(getattr(pB_recent, "kills_for", None), float("nan"))
                rkaB = _f(getattr(pB_recent, "kills_against", None), float("nan"))
                rtfB = _f(getattr(pB_recent, "towers_for", None), float("nan"))
                rtaB = _f(getattr(pB_recent, "towers_against", None), float("nan"))
                rdfB = _f(getattr(pB_recent, "dragons_for", None), float("nan"))
                rdaB = _f(getattr(pB_recent, "dragons_against", None), float("nan"))
                rbfB = _f(getattr(pB_recent, "barons_for", None), float("nan"))
                rbaB = _f(getattr(pB_recent, "barons_against", None), float("nan"))
                rifB = _f(getattr(pB_recent, "inhib_for", None), float("nan"))
                riaB = _f(getattr(pB_recent, "inhib_against", None), float("nan"))
                robj_for_A = (rtfA + 1.1 * rdfA + 1.7 * rbfA + 1.3 * rifA) if all(
                    math.isfinite(v) for v in [rtfA, rdfA, rbfA, rifA]
                ) else float("nan")
                robj_agn_A = (rtaA + 1.1 * rdaA + 1.7 * rbaA + 1.3 * riaA) if all(
                    math.isfinite(v) for v in [rtaA, rdaA, rbaA, riaA]
                ) else float("nan")
                robj_for_B = (rtfB + 1.1 * rdfB + 1.7 * rbfB + 1.3 * rifB) if all(
                    math.isfinite(v) for v in [rtfB, rdfB, rbfB, rifB]
                ) else float("nan")
                robj_agn_B = (rtaB + 1.1 * rdaB + 1.7 * rbaB + 1.3 * riaB) if all(
                    math.isfinite(v) for v in [rtaB, rdaB, rbaB, riaB]
                ) else float("nan")
                rq_obj_ctrl_A = _safe_ratio(robj_for_A + 1.0, (robj_for_A + robj_agn_A + 2.0))
                rq_obj_ctrl_B = _safe_ratio(robj_for_B + 1.0, (robj_for_B + robj_agn_B + 2.0))
                rq_closeout_A = _safe_ratio(robj_for_A, max(1e-6, rpA)) if math.isfinite(robj_for_A) and math.isfinite(rpA) else float("nan")
                rq_closeout_B = _safe_ratio(robj_for_B, max(1e-6, rpB)) if math.isfinite(robj_for_B) and math.isfinite(rpB) else float("nan")
                rq_resilience_A = _safe_ratio(robj_for_A + 1.0, (max(0.0, rkaA) + 1.0)) if math.isfinite(robj_for_A) and math.isfinite(rkaA) else float("nan")
                rq_resilience_B = _safe_ratio(robj_for_B + 1.0, (max(0.0, rkaB) + 1.0)) if math.isfinite(robj_for_B) and math.isfinite(rkaB) else float("nan")
                rq_net_ctrl_A = _safe_ratio((robj_for_A - robj_agn_A), (robj_for_A + robj_agn_A + 2.0)) if math.isfinite(robj_for_A) and math.isfinite(robj_agn_A) else float("nan")
                rq_net_ctrl_B = _safe_ratio((robj_for_B - robj_agn_B), (robj_for_B + robj_agn_B + 2.0)) if math.isfinite(robj_for_B) and math.isfinite(robj_agn_B) else float("nan")
                rq_convert_A = _safe_ratio(robj_for_A, (max(0.0, rkfA) + 2.0)) if math.isfinite(robj_for_A) and math.isfinite(rkfA) else float("nan")
                rq_convert_B = _safe_ratio(robj_for_B, (max(0.0, rkfB) + 2.0)) if math.isfinite(robj_for_B) and math.isfinite(rkfB) else float("nan")
                rq_contain_A = _safe_ratio(robj_agn_A, (max(0.0, rkaA) + 2.0)) if math.isfinite(robj_agn_A) and math.isfinite(rkaA) else float("nan")
                rq_contain_B = _safe_ratio(robj_agn_B, (max(0.0, rkaB) + 2.0)) if math.isfinite(robj_agn_B) and math.isfinite(rkaB) else float("nan")
                if math.isfinite(rq_obj_ctrl_A) and math.isfinite(rq_obj_ctrl_B):
                    r_terms["quality_objective_control_edge"] = float((rq_obj_ctrl_A - rq_obj_ctrl_B) / 0.20)
                if math.isfinite(rq_closeout_A) and math.isfinite(rq_closeout_B):
                    r_terms["quality_closeout_edge"] = float((rq_closeout_A - rq_closeout_B) / 0.55)
                if math.isfinite(rq_resilience_A) and math.isfinite(rq_resilience_B):
                    r_terms["quality_resilience_edge"] = float((rq_resilience_A - rq_resilience_B) / 0.90)
                if math.isfinite(rq_net_ctrl_A) and math.isfinite(rq_net_ctrl_B):
                    r_terms["quality_net_control_edge"] = float((rq_net_ctrl_A - rq_net_ctrl_B) / 0.22)
                if math.isfinite(rq_convert_A) and math.isfinite(rq_convert_B):
                    r_terms["quality_conversion_edge"] = float((rq_convert_A - rq_convert_B) / 0.45)
                if math.isfinite(rq_contain_A) and math.isfinite(rq_contain_B):
                    r_terms["quality_containment_edge"] = float((rq_contain_B - rq_contain_A) / 0.28)
                if r_terms:
                    rs_sum = 0.0
                    rw_sum = 0.0
                    for k, v in r_terms.items():
                        w = float(STYLE_MATCHUP_TERM_WEIGHTS.get(k, 0.0))
                        if w <= 0.0:
                            continue
                        rs_sum += float(v) * w
                        rw_sum += w
                    if rw_sum > 0.0:
                        recent_style_score = float(max(-2.5, min(2.5, rs_sum / rw_sum)))
                        recent_log1p_n_min = float(math.log1p(n_min_r))
                        # Trust recent style only when recent sample is usable.
                        _w_recent = float(max(0.0, min(1.0, n_min_r / (n_min_r + 4.0))))
                        recent_available = 1.0
                        recent_weight = float(_w_recent)
                        recent_minus_base_style = float(max(-1.5, min(1.5, (recent_style_score - style_score) * _w_recent)))
                        recent_disagreement_abs = float(max(0.0, min(2.0, abs(recent_style_score - style_score) * _w_recent)))
        except Exception:
            recent_style_score = 0.0
            recent_minus_base_style = 0.0
            recent_log1p_n_min = 0.0
            recent_available = 0.0
            recent_weight = 0.0
            recent_disagreement_abs = 0.0

    features: Dict[str, float] = {
        "z_in": float(z0),
        "log1p_n_min": float(math.log1p(n_min)),
        "log1p_n_a": float(math.log1p(nA)),
        "log1p_n_b": float(math.log1p(nB)),
        "style_score": float(style_score),
        "raw_delta_logit": float(raw_delta_logit),
        "w_sample": float(w_sample),
        "w_context": float(w_ctx),
        "sample_imbalance": float(sample_imbalance),
        "kills_edge": float(score_terms.get("kills_edge", 0.0)),
        "towers_edge": float(score_terms.get("towers_edge", 0.0)),
        "dragons_edge": float(score_terms.get("dragons_edge", 0.0)),
        "barons_edge": float(score_terms.get("barons_edge", 0.0)),
        "inhib_edge": float(score_terms.get("inhib_edge", 0.0)),
        "pace_edge": float(score_terms.get("pace_edge", 0.0)),
        "volatility_edge": float(score_terms.get("volatility_edge", 0.0)),
        "interregional": float(1.0 if interregional else 0.0),
        "unknown_macro": float(1.0 if unknown_macro else 0.0),
        "patch_refit": float(1.0 if patch_s in {"refit", "refit_needed", "changed"} else 0.0),
        "roster_changed": float(1.0 if bool(roster_changed) else 0.0),
        "bo_is_1": float(1.0 if bo_i <= 1 else 0.0),
        "bo_is_3": float(1.0 if bo_i == 3 else 0.0),
        "bo_is_5": float(1.0 if bo_i >= 5 else 0.0),
        "is_playoffs": float(1.0 if bool(is_playoffs) else 0.0),
        "same_league": float(1.0 if same_league else 0.0),
        "bo5_playoffs": float(1.0 if (bo_i >= 5 and bool(is_playoffs)) else 0.0),
        "bo1_interregional": float(1.0 if (bo_i <= 1 and interregional) else 0.0),
        "recent_style_score": float(recent_style_score),
        "recent_minus_base_style": float(recent_minus_base_style),
        "recent_log1p_n_min": float(recent_log1p_n_min),
        "recent_available": float(recent_available),
        "recent_weight": float(recent_weight),
        "recent_disagreement_abs": float(recent_disagreement_abs),
    }
    vec = [float(features.get(k, 0.0)) for k in STYLE_MATCHUP_B_MODEL_FEATURES]
    exp_time = float(0.5 * (paceA + paceB)) if (math.isfinite(paceA) and math.isfinite(paceB)) else float("nan")
    exp_total_kills = (
        float(exp_kills_A + exp_kills_B)
        if (math.isfinite(exp_kills_A) and math.isfinite(exp_kills_B))
        else float("nan")
    )
    phase_bucket = _phase_bucket_from_expected(exp_time_min=exp_time, exp_total_kills=exp_total_kills)
    exp_kpm = (exp_total_kills / exp_time) if (math.isfinite(exp_total_kills) and math.isfinite(exp_time) and exp_time > 0.0) else float("nan")
    return {
        "ok": True,
        "reason": "",
        "features": features,
        "vector": vec,
        "weights": {"sample": float(w_sample), "context": float(w_ctx), "final": float(w_final)},
        "style": {
            "score": float(style_score),
            "raw_delta_logit": float(raw_delta_logit),
            "delta_cap": float(delta_cap),
            "terms": {k: float(v) for k, v in score_terms.items()},
            "expected": {
                "kills_A": (float(exp_kills_A) if math.isfinite(exp_kills_A) else None),
                "kills_B": (float(exp_kills_B) if math.isfinite(exp_kills_B) else None),
                "towers_A": (float(exp_towers_A) if math.isfinite(exp_towers_A) else None),
                "towers_B": (float(exp_towers_B) if math.isfinite(exp_towers_B) else None),
                "dragons_A": (float(exp_dragons_A) if math.isfinite(exp_dragons_A) else None),
                "dragons_B": (float(exp_dragons_B) if math.isfinite(exp_dragons_B) else None),
                "barons_A": (float(exp_barons_A) if math.isfinite(exp_barons_A) else None),
                "barons_B": (float(exp_barons_B) if math.isfinite(exp_barons_B) else None),
                "inhib_A": (float(exp_inhib_A) if math.isfinite(exp_inhib_A) else None),
                "inhib_B": (float(exp_inhib_B) if math.isfinite(exp_inhib_B) else None),
            },
            "macros": {
                "macro_a": str(macro_a),
                "macro_b": str(macro_b),
                "interregional": bool(interregional),
                "unknown_macro": bool(unknown_macro),
                "same_league": bool(same_league),
                "cross_tier_interregional": bool(cross_tier_interregional),
                "cross_strength_interregional": bool(cross_strength_interregional),
                "tier_gap": int(tier_gap),
                "strength_gap": float(strength_gap),
                "bo": int(bo_i),
                "is_playoffs": bool(is_playoffs),
                "sample_imbalance": float(sample_imbalance),
                "interregional_relief": float(interregional_relief),
                "phase_bucket": str(phase_bucket),
                "exp_time_min": (float(exp_time) if math.isfinite(exp_time) else None),
                "exp_total_kills": (float(exp_total_kills) if math.isfinite(exp_total_kills) else None),
                "exp_kills_per_min": (float(exp_kpm) if math.isfinite(exp_kpm) else None),
            },
            "recent": {
                "style_score": float(recent_style_score),
                "minus_base_style": float(recent_minus_base_style),
                "log1p_n_min": float(recent_log1p_n_min),
                "available": bool(recent_available > 0.0),
                "weight": float(recent_weight),
                "disagreement_abs": float(recent_disagreement_abs),
            },
        },
    }
