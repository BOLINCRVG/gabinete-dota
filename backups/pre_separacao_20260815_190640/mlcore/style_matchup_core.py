"""Matchup/style layer for ML map probability (B-lite + B-real).

B-lite: heuristic conservative style adjustment.
B-real: trained logistic correction on top of the same style features.

Feature building lives in style_matchup_features_core.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from mlcore.utils import clip_prob as _clip_prob
from mlcore.utils import isfinite_num as _isfinite
from mlcore.utils import safe_float as _f
from mlcore.utils import logit as _safe_logit
from mlcore.utils import sigmoid as _safe_sigmoid

from mlcore.style_matchup_features_core import (
    STYLE_MATCHUP_B_MODEL_FEATURES,
    _apply_quality_scale_to_features,
    _is_explicit_interregional_league,
    _macro_key,
    _patch_bucket_from_status,
    _resolve_quality_term_scale,
    build_style_matchup_b_model_features,
    should_apply_style_matchup_b_model_gate,
)

__all__ = [
    "STYLE_MATCHUP_B_MODEL_FEATURES",
    "apply_style_matchup_b_lite",
    "apply_style_matchup_b_live",
    "apply_style_matchup_b_model",
    "build_style_matchup_b_model_features",
    "should_apply_style_matchup_b_model_gate",
]


def apply_style_matchup_b_model(
    *,
    model_cfg: Dict[str, Any] | None,
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
    out: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "",
        "method": "style_matchup_b_model_v7",
        "p_before": (_f(p_map_in) if _isfinite(p_map_in) else None),
        "p_after": None,
        "diff_pp": None,
    }
    if not isinstance(model_cfg, dict):
        out["reason"] = "invalid_model_cfg"
        return out
    if not bool(model_cfg.get("enabled", True)):
        out["reason"] = "model_disabled"
        return out

    feat = build_style_matchup_b_model_features(
        p_map_in=p_map_in,
        pA=pA,
        pB=pB,
        league_a=league_a,
        league_b=league_b,
        patch_status=patch_status,
        roster_changed=roster_changed,
        bo=bo,
        is_playoffs=is_playoffs,
        pA_recent=pA_recent,
        pB_recent=pB_recent,
    )
    if not bool(feat.get("ok")):
        out["reason"] = str(feat.get("reason") or "invalid_features")
        return out

    quality_scale_info = _resolve_quality_term_scale(
        model_cfg=model_cfg,
        league_a=league_a,
        league_b=league_b,
        patch_status=patch_status,
        feat=feat if isinstance(feat, dict) else {},
    )
    feat = _apply_quality_scale_to_features(feat, float(quality_scale_info.get("scale", 1.0)))

    def _norm_key(x: Any) -> str:
        return str(x or "").strip().upper()

    def _predict_logit(cfg_any: Dict[str, Any], x_raw: List[float]) -> Optional[float]:
        if not isinstance(cfg_any, dict):
            return None
        names_any = list(cfg_any.get("features") or [])
        coef_any = list(cfg_any.get("coef") or [])
        if names_any != STYLE_MATCHUP_B_MODEL_FEATURES or len(coef_any) != len(STYLE_MATCHUP_B_MODEL_FEATURES):
            return None
        try:
            intercept_any = float(cfg_any.get("intercept", 0.0) or 0.0)
        except Exception:
            return None
        mu_any = cfg_any.get("x_mean")
        sd_any = cfg_any.get("x_std")
        if isinstance(mu_any, list) and isinstance(sd_any, list) and len(mu_any) == len(x_raw) and len(sd_any) == len(x_raw):
            x_use_any: List[float] = []
            for xv, m, s in zip(x_raw, mu_any, sd_any):
                try:
                    mm = float(m)
                    ss = float(s)
                except Exception:
                    mm, ss = 0.0, 1.0
                if (not math.isfinite(ss)) or abs(ss) < 1e-12:
                    ss = 1.0
                x_use_any.append(float((float(xv) - mm) / ss))
        else:
            x_use_any = list(x_raw)
        z_any = float(intercept_any)
        for c, xv in zip(coef_any, x_use_any):
            try:
                z_any += float(c) * float(xv)
            except Exception:
                return None
        return float(z_any)

    def _select_submodel(cfg_root: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        # Hierarquia: league/macro com patch+phase > patch > phase > global
        la = _norm_key(league_a)
        lb = _norm_key(league_b)
        macro_a = _macro_key(league_a)
        macro_b = _macro_key(league_b)
        interregional = bool(
            (macro_a and macro_b and macro_a != macro_b)
            or _is_explicit_interregional_league(league_a)
            or _is_explicit_interregional_league(league_b)
        )
        patch_bucket = _patch_bucket_from_status(patch_status)
        macros_feat = ((feat.get("style") or {}).get("macros") if isinstance((feat.get("style") or {}), dict) else {}) or {}
        phase_bucket = str(macros_feat.get("phase_bucket") or "").strip().lower()
        if phase_bucket not in {"early", "mid", "late"}:
            phase_bucket = "mid"
        contexts = cfg_root.get("context_models") if isinstance(cfg_root.get("context_models"), dict) else {}
        if not isinstance(contexts, dict) or not contexts:
            return cfg_root, "global"

        keys_try: List[str] = []
        if la and lb and la == lb and not interregional:
            keys_try.extend(
                [
                    f"league:{la}|patch:{patch_bucket}|phase:{phase_bucket}",
                    f"league:{la}|patch:{patch_bucket}",
                    f"league:{la}|phase:{phase_bucket}",
                    f"league:{la}",
                ]
            )
        if macro_a and macro_b and macro_a == macro_b and not interregional:
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
                "global",
            ]
        )

        for k in keys_try:
            cand = contexts.get(k)
            if isinstance(cand, dict) and cand:
                _c = dict(cand)
                # inherit schema/defaults from root
                for field in ("features", "x_mean", "x_std"):
                    if field not in _c and field in cfg_root:
                        _c[field] = cfg_root.get(field)
                if "version" not in _c:
                        _c["version"] = cfg_root.get("version", "style_matchup_b_model_v7")
                return _c, str(k)
        return cfg_root, "global"

    def _metric_triplet(cfg_any: Dict[str, Any]) -> Tuple[float, float, float]:
        n_tot = max(0.0, _f(cfg_any.get("n_total"), 0.0))
        imp_l = _f(cfg_any.get("improvement_vs_lite"), float("nan"))
        imp_a = _f(cfg_any.get("improvement_vs_anchor"), float("nan"))
        return float(n_tot), float(imp_l), float(imp_a)

    model_use, model_ctx_key = _select_submodel(model_cfg)
    model_ver = str(model_use.get("version", model_cfg.get("version", "style_matchup_b_model_v7")))

    x = [float(v) for v in (feat.get("vector") or [])]
    z_ctx = _predict_logit(model_use, x)
    if z_ctx is None:
        out["reason"] = "feature_schema_mismatch"
        return out

    # Runtime reliability guard: if B-real is statistically underperforming on OOS
    # for the selected scope, fallback to B-lite at caller level.
    guard_cfg = model_cfg.get("runtime_guard") if isinstance(model_cfg.get("runtime_guard"), dict) else {}
    guard_enabled = bool(guard_cfg.get("enabled", True)) if isinstance(guard_cfg, dict) and bool(guard_cfg) else False
    min_n = float(max(40.0, _f(guard_cfg.get("min_n_total"), 220.0)))
    min_imp_l = float(_f(guard_cfg.get("min_improvement_vs_lite"), -0.003))
    min_imp_a = float(_f(guard_cfg.get("min_improvement_vs_anchor"), -0.0015))
    allow_context_override = bool(guard_cfg.get("allow_context_override", True))
    n_global, imp_l_global, imp_a_global = _metric_triplet(model_cfg if isinstance(model_cfg, dict) else {})
    n_use, imp_l_use, imp_a_use = _metric_triplet(model_use if isinstance(model_use, dict) else {})

    def _is_bad(imp_l: float, imp_a: float, n_tot: float) -> bool:
        if not (math.isfinite(imp_l) and math.isfinite(imp_a)):
            return False
        if n_tot < min_n:
            return False
        return bool(float(imp_l) < float(min_imp_l) and float(imp_a) < float(min_imp_a))

    global_bad = _is_bad(imp_l_global, imp_a_global, n_global)
    use_bad = _is_bad(imp_l_use, imp_a_use, n_use)
    guard_block = False
    if guard_enabled:
        if str(model_ctx_key) == "global":
            guard_block = bool(global_bad)
        else:
            # Context model must be at least non-bad; if it is weak and global is also bad,
            # block B-real and let caller fallback to B-lite.
            if use_bad:
                guard_block = True
            elif (not allow_context_override) and global_bad:
                guard_block = True
            elif n_use < min_n and global_bad:
                guard_block = True
    guard_info = {
        "enabled": bool(guard_enabled),
        "blocked": bool(guard_block),
        "context_key": str(model_ctx_key),
        "selected_version": str(model_ver),
        "thresholds": {
            "min_n_total": float(min_n),
            "min_improvement_vs_lite": float(min_imp_l),
            "min_improvement_vs_anchor": float(min_imp_a),
            "allow_context_override": bool(allow_context_override),
        },
        "selected_model": {
            "n_total": int(n_use),
            "improvement_vs_lite": (float(imp_l_use) if math.isfinite(imp_l_use) else None),
            "improvement_vs_anchor": (float(imp_a_use) if math.isfinite(imp_a_use) else None),
            "underperforming": bool(use_bad),
        },
        "global_model": {
            "n_total": int(n_global),
            "improvement_vs_lite": (float(imp_l_global) if math.isfinite(imp_l_global) else None),
            "improvement_vs_anchor": (float(imp_a_global) if math.isfinite(imp_a_global) else None),
            "underperforming": bool(global_bad),
        },
    }
    if bool(guard_block):
        out.update(
            {
                "applied": False,
                "reason": "model_reliability_guard",
                "method": str(model_ver),
                "model_meta": {
                    "version": str(model_ver),
                    "context_key": str(model_ctx_key),
                    "trained_on_n": int(n_use),
                    "oos_logloss": model_use.get("oos_logloss", model_cfg.get("oos_logloss")),
                    "improvement_vs_lite": (float(imp_l_use) if math.isfinite(imp_l_use) else None),
                    "improvement_vs_anchor": (float(imp_a_use) if math.isfinite(imp_a_use) else None),
                    "guard": guard_info,
                },
            }
        )
        return out

    z = float(z_ctx)
    context_blend: Dict[str, Any] = {"used": False}
    # B-real v6: hierarchical runtime regularization for contextual submodels.
    # If a context submodel is selected, blend it with the global model according to sample/quality.
    if str(model_ctx_key) != "global":
        z_global = _predict_logit(model_cfg, x)
        if z_global is not None:
            n_ctx = max(0.0, _f(model_use.get("n_total"), 0.0))
            imp_lite = _f(model_use.get("improvement_vs_lite"), 0.0)
            imp_anchor = _f(model_use.get("improvement_vs_anchor"), 0.0)
            # Hard reliability gate for contextual submodels:
            # low-N contexts or contexts with clearly negative OOS improvements should mostly fallback to global.
            bad_imp_count = ((1 if (math.isfinite(imp_lite) and float(imp_lite) < -0.002) else 0)
                + (1 if (math.isfinite(imp_anchor) and float(imp_anchor) < -0.002) else 0))
            ctx_min_n = max(40.0, _f(model_cfg.get("ctx_min_n"), 120.0))
            w_n = float(max(0.15, min(1.0, n_ctx / (n_ctx + 260.0))))
            # Conservative quality term: contexts with poor OOS improvement are shrunk harder.
            q = 0.0
            if math.isfinite(imp_lite):
                q += max(0.0, min(0.08, float(imp_lite))) * 6.0
            if math.isfinite(imp_anchor):
                q += max(0.0, min(0.08, float(imp_anchor))) * 4.0
            w_q = float(max(0.25, min(1.0, 0.35 + q)))
            if "|patch:" in str(model_ctx_key):
                w_n *= 0.88
            if str(model_ctx_key).startswith("league:"):
                w_n *= 1.05
            elif str(model_ctx_key).startswith("macro:"):
                w_n *= 0.95
            w_ctx_blend = float(max(0.15, min(1.0, w_n * w_q)))
            if n_ctx < ctx_min_n:
                w_ctx_blend = float(min(w_ctx_blend, max(0.05, n_ctx / max(1e-6, ctx_min_n))))
            if bad_imp_count >= 2:
                w_ctx_blend = float(min(w_ctx_blend, 0.12))
            elif bad_imp_count == 1:
                w_ctx_blend = float(min(w_ctx_blend, 0.35))
            z = float(z_global + (w_ctx_blend * (z_ctx - z_global)))
            context_blend = {
                "used": True,
                "context_key": str(model_ctx_key),
                "w_n": float(w_n),
                "w_q": float(w_q),
                "w_ctx_blend": float(w_ctx_blend),
                "n_ctx": int(n_ctx),
                "ctx_min_n": float(ctx_min_n),
                "bad_imp_count": int(bad_imp_count),
                "z_global": float(z_global),
                "z_context": float(z_ctx),
                "z_blended": float(z),
                "imp_lite": (float(imp_lite) if math.isfinite(imp_lite) else None),
                "imp_anchor": (float(imp_anchor) if math.isfinite(imp_anchor) else None),
            }
    p_before = _clip_prob(_f(p_map_in))
    z_before = _safe_logit(p_before)

    # B-real v7: contextual output shrink (post-model, in logit space).
    # Keeps strong context signals, but reduces overreaction in weak/volatile contexts.
    macros_feat = ((feat.get("style") or {}).get("macros") if isinstance((feat.get("style") or {}), dict) else {}) or {}
    rec_feat = ((feat.get("style") or {}).get("recent") if isinstance((feat.get("style") or {}), dict) else {}) or {}
    interregional_flag = bool(macros_feat.get("interregional", False))
    same_league_flag = bool(macros_feat.get("same_league", False))
    bo_i_feat = int(_f(macros_feat.get("bo"), 1.0))
    recent_avail = bool(rec_feat.get("available", False))
    recent_w = float(max(0.0, min(1.0, _f(rec_feat.get("weight"), 0.0))))
    recent_dis = float(max(0.0, min(3.0, _f(rec_feat.get("disagreement_abs"), 0.0))))
    out_shrink = 1.0
    if str(model_ctx_key) != "global":
        cb = context_blend if isinstance(context_blend, dict) else {}
        w_ctx_blend = float(max(0.0, min(1.0, _f(cb.get("w_ctx_blend"), 0.0))))
        n_ctx = float(max(0.0, _f(cb.get("n_ctx"), 0.0)))
        w_n_out = float(max(0.35, min(1.0, n_ctx / (n_ctx + 220.0)))) if n_ctx > 0 else 0.35
        w_q_out = float(max(0.45, min(1.0, 0.55 + 0.45 * w_ctx_blend)))
        out_shrink = w_n_out * w_q_out
        if "|patch:" in str(model_ctx_key):
            out_shrink *= 0.90
        if interregional_flag:
            inter_out_relief = float(
                max(
                    0.0,
                    min(
                        0.14,
                        (0.07 * (n_ctx / (n_ctx + 260.0) if n_ctx > 0.0 else 0.0))
                        + (0.03 if bo_i_feat >= 3 else 0.0)
                        + (0.04 * recent_w if recent_avail else 0.0),
                    ),
                )
            )
            if recent_dis > 1.0:
                inter_out_relief *= max(0.75, 1.0 - 0.10 * (recent_dis - 1.0))
            out_shrink *= float(min(0.98, 0.86 + inter_out_relief))
        if bo_i_feat == 1:
            out_shrink *= 0.92
        elif bo_i_feat == 5:
            out_shrink *= 1.04
        if recent_avail:
            out_shrink *= (0.92 + 0.16 * recent_w)
            if recent_dis > 1.0:
                out_shrink *= max(0.82, 1.0 - 0.08 * (recent_dis - 1.0))
        if same_league_flag and bo_i_feat in {3, 5}:
            out_shrink *= 1.03
        out_shrink = float(max(0.35, min(1.0, out_shrink)))
    z_final = float(z_before + out_shrink * (z - z_before))
    p_out = _clip_prob(_safe_sigmoid(z_final))
    out.update(
        {
            "applied": True,
            "reason": "style_matchup_b_model",
            "p_after": float(p_out),
            "diff_pp": float(100.0 * (float(p_out) - float(p_before))),
            "weights": dict(feat.get("weights") or {}),
            "style": dict(feat.get("style") or {}),
            "features": dict(feat.get("features") or {}),
            "targets": {"z_model": float(z), "z_before": float(z_before), "z_final": float(z_final)},
            "model_meta": {
                "version": str(model_ver),
                "trained_on_n": int(model_use.get("n_total", model_cfg.get("n_total", 0)) or 0),
                "oos_logloss": model_use.get("oos_logloss", model_cfg.get("oos_logloss")),
                "improvement_vs_lite": (
                    float(_f(model_use.get("improvement_vs_lite"), float("nan")))
                    if math.isfinite(_f(model_use.get("improvement_vs_lite"), float("nan")))
                    else None
                ),
                "improvement_vs_anchor": (
                    float(_f(model_use.get("improvement_vs_anchor"), float("nan")))
                    if math.isfinite(_f(model_use.get("improvement_vs_anchor"), float("nan")))
                    else None
                ),
                "context_key": str(model_ctx_key),
                "quality_term_scale": dict(quality_scale_info),
                "guard": guard_info,
                "context_blend": context_blend,
                "output_shrink": {
                    "used": bool(str(model_ctx_key) != "global"),
                    "value": float(out_shrink),
                    "bo": int(bo_i_feat),
                    "interregional": bool(interregional_flag),
                    "same_league": bool(same_league_flag),
                    "recent_available": bool(recent_avail),
                    "recent_weight": float(recent_w),
                    "recent_disagreement_abs": float(recent_dis),
                },
            },
        }
    )
    out["method"] = str((out.get("model_meta") or {}).get("version") or "style_matchup_b_model_v7")
    return out


def apply_style_matchup_b_lite(
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
    """Return style/matchup adjusted probability + trace.

    `pA` and `pB` are expected to look like `mlcore.lines_engine.TeamProfile`.
    """
    out: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "",
        "p_before": (_f(p_map_in) if _isfinite(p_map_in) else None),
        "p_after": None,
        "diff_pp": None,
    }
    feat = build_style_matchup_b_model_features(
        p_map_in=p_map_in,
        pA=pA,
        pB=pB,
        league_a=league_a,
        league_b=league_b,
        patch_status=patch_status,
        roster_changed=roster_changed,
        bo=bo,
        is_playoffs=is_playoffs,
        pA_recent=pA_recent,
        pB_recent=pB_recent,
    )
    if not bool(feat.get("ok")):
        out["reason"] = str(feat.get("reason") or "invalid_features")
        return out

    p0 = _clip_prob(_f(p_map_in))
    z0 = _safe_logit(p0)
    style = dict(feat.get("style") or {})
    weights = dict(feat.get("weights") or {})
    style_score = float(style.get("score", 0.0) or 0.0)
    raw_delta_logit = float(style.get("raw_delta_logit", 0.0) or 0.0)
    delta_cap = float(style.get("delta_cap", 0.55) or 0.55)
    w_final = float(weights.get("final", 0.10) or 0.10)
    if not math.isfinite(w_final):
        w_final = 0.10
    macros = (style.get("macros") if isinstance(style.get("macros"), dict) else {}) or {}
    n_games_a = int(max(0.0, _f(getattr(pA, "n_games", 0), 0.0)))
    n_games_b = int(max(0.0, _f(getattr(pB, "n_games", 0), 0.0)))
    if bool(macros.get("interregional", False)):
        sample_min = min(n_games_a, n_games_b)
        inter_damp = 1.0
        if sample_min < 10:
            inter_damp *= 0.55
        elif sample_min < 18:
            inter_damp *= 0.68
        elif sample_min < 26:
            inter_damp *= 0.80
        else:
            inter_damp *= 0.88
        if str(macros.get("macro_a", "") or "") and str(macros.get("macro_b", "") or "") and str(macros.get("macro_a")) != str(macros.get("macro_b")):
            inter_damp *= 0.90
        w_final = float(max(0.05, min(0.85, w_final * inter_damp)))

    z1 = float(z0 + (w_final * raw_delta_logit))
    p1 = float(_clip_prob(_safe_sigmoid(z1)))
    out.update(
        {
            "applied": True,
            "reason": "style_matchup_b_lite",
            "p_after": float(p1),
            "diff_pp": float(100.0 * (p1 - p0)),
            "weights": {
                "sample": float(weights.get("sample", 0.0) or 0.0),
                "context": float(weights.get("context", 0.0) or 0.0),
                "final": float(w_final),
            },
            "inputs": {
                "n_games_a": int(n_games_a),
                "n_games_b": int(n_games_b),
                "league_a": str(league_a or ""),
                "league_b": str(league_b or ""),
                "macro_a": str(macros.get("macro_a", "") or ""),
                "macro_b": str(macros.get("macro_b", "") or ""),
                "interregional": bool(macros.get("interregional", False)),
                "unknown_macro": bool(macros.get("unknown_macro", False)),
                "patch_status": str(patch_status or ""),
                "roster_changed": bool(roster_changed),
                "bo": int(float(bo)) if bo is not None and _isfinite(bo) else 1,
                "is_playoffs": bool(is_playoffs),
            },
            "style": {
                "score": float(style_score),
                "raw_delta_logit": float(raw_delta_logit),
                "delta_cap": float(delta_cap),
                "terms": {k: float(v) for k, v in dict(style.get("terms") or {}).items()},
                "expected": dict(style.get("expected") or {}),
            },
        }
    )
    return out


def apply_style_matchup_b_live(
    *,
    model_cfg: Dict[str, Any] | None,
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
    n_team_eff: Any = None,
    n_players_eff: Any = None,
    use_lite_when_gate_blocked: bool = True,
) -> Dict[str, Any]:
    """Runtime orchestrator for B layer.

    Order:
    1) Evaluate B-model context gate.
    2) Try B-model when allowed.
    3) Fallback to B-lite when model does not apply or gate blocks (configurable).
    """

    gate = should_apply_style_matchup_b_model_gate(
        bo=bo,
        n_team_eff=n_team_eff,
        n_players_eff=n_players_eff,
        pA=pA,
        pB=pB,
        interregional=(str(league_a or "").strip().upper() != str(league_b or "").strip().upper()),
        patch_status=str(patch_status or ""),
        roster_changed=bool(roster_changed),
    )

    common_kwargs = {
        "p_map_in": p_map_in,
        "pA": pA,
        "pB": pB,
        "league_a": str(league_a or ""),
        "league_b": str(league_b or ""),
        "patch_status": str(patch_status or ""),
        "roster_changed": bool(roster_changed),
        "bo": bo,
        "is_playoffs": bool(is_playoffs),
        "pA_recent": pA_recent,
        "pB_recent": pB_recent,
    }

    def _model_attempt_payload(model_out: Dict[str, Any] | None) -> Dict[str, Any]:
        md = model_out if isinstance(model_out, dict) else {}
        mm = md.get("model_meta") if isinstance(md.get("model_meta"), dict) else {}
        guard = mm.get("guard") if isinstance(mm.get("guard"), dict) else {}
        return {
            "applied": bool(md.get("applied", False)),
            "reason": str(md.get("reason", "") or ""),
            "method": str(md.get("method", "") or ""),
            "context_key": str(mm.get("context_key", "") or ""),
            "guard_blocked": bool(guard.get("blocked", False)),
        }

    if bool((gate or {}).get("allow", False)):
        model_out = apply_style_matchup_b_model(
            model_cfg=(model_cfg if isinstance(model_cfg, dict) else {}),
            **common_kwargs,
        )
        if isinstance(model_out, dict) and bool(model_out.get("applied", False)):
            model_out.setdefault("enabled", True)
            model_out["fallback_from_model"] = False
            model_out["fallback_from_gate"] = False
            model_out["fallback_reason"] = ""
            model_out["b_pipeline_path"] = "model_applied"
            model_out["b_model_gate"] = dict(gate or {})
            model_out["b_model_attempt"] = _model_attempt_payload(model_out)
            return model_out

        lite_out = apply_style_matchup_b_lite(**common_kwargs)
        if isinstance(lite_out, dict):
            lite_out.setdefault("enabled", True)
            lite_out["fallback_from_model"] = True
            lite_out["fallback_from_gate"] = False
            lite_out["fallback_reason"] = str((model_out or {}).get("reason", "") or "")
            lite_out["b_pipeline_path"] = "lite_fallback_model_not_applied"
            lite_out["b_model_gate"] = dict(gate or {})
            lite_out["b_model_attempt"] = _model_attempt_payload(model_out if isinstance(model_out, dict) else {})
            return lite_out

        return {
            "enabled": True,
            "applied": False,
            "reason": "style_lite_fallback_failed",
            "method": "style_matchup_b_model_gate",
            "fallback_from_model": True,
            "fallback_from_gate": False,
            "fallback_reason": str((model_out or {}).get("reason", "") or ""),
            "b_pipeline_path": "model_failed_and_lite_failed",
            "b_model_gate": dict(gate or {}),
            "b_model_attempt": _model_attempt_payload(model_out if isinstance(model_out, dict) else {}),
            "p_before": (_f(p_map_in) if _isfinite(p_map_in) else None),
            "p_after": None,
            "diff_pp": None,
        }

    if bool(use_lite_when_gate_blocked):
        lite_out = apply_style_matchup_b_lite(**common_kwargs)
        if isinstance(lite_out, dict):
            lite_out.setdefault("enabled", True)
            lite_out["fallback_from_model"] = False
            lite_out["fallback_from_gate"] = True
            lite_out["fallback_reason"] = str((gate or {}).get("reason", "") or "")
            lite_out["b_pipeline_path"] = "lite_fallback_gate_blocked"
            lite_out["b_model_gate"] = dict(gate or {})
            lite_out["b_model_attempt"] = {
                "applied": False,
                "reason": "not_attempted_gate_blocked",
                "method": "style_matchup_b_model",
                "context_key": "",
                "guard_blocked": False,
            }
            return lite_out

    return {
        "enabled": True,
        "applied": False,
        "reason": "b_gate_blocked_no_b_apply",
        "method": "style_matchup_b_model_gate",
        "fallback_from_model": False,
        "fallback_from_gate": False,
        "fallback_reason": str((gate or {}).get("reason", "") or ""),
        "b_pipeline_path": "gate_blocked_no_fallback",
        "b_model_gate": dict(gate or {}),
        "b_model_attempt": {
            "applied": False,
            "reason": "not_attempted_gate_blocked",
            "method": "style_matchup_b_model",
            "context_key": "",
            "guard_blocked": False,
        },
        "p_before": (_f(p_map_in) if _isfinite(p_map_in) else None),
        "p_after": None,
        "diff_pp": None,
    }
