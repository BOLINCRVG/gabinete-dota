"""Meta-blend C layer — model application.

Feature building lives in meta_blend_features_core.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from mlcore.utils import clip_prob as _clip_prob
from mlcore.utils import safe_float as _f
from mlcore.utils import logit as _safe_logit
from mlcore.utils import sigmoid as _safe_sigmoid

from mlcore.meta_blend_features_core import (
    META_BLEND_C_MODEL_FEATURES,
    _interregional_strength_profile,
    _isfinite,
    _macro_key,
    _norm_key,
    _patch_bucket_from_status,
    _phase_bucket,
    build_meta_blend_c_model_features,
    should_apply_meta_blend_c_model_gate,
)

__all__ = [
    "META_BLEND_C_MODEL_FEATURES",
    "apply_meta_blend_c_lite",
    "apply_meta_blend_c_model",
    "build_meta_blend_c_model_features",
    "should_apply_meta_blend_c_model_gate",
]


def apply_meta_blend_c_model(
    *,
    model_cfg: Dict[str, Any] | None,
    p_anchor: Any,
    p_adjusted: Any,
    n_team_eff: Any = None,
    n_players_eff: Any = None,
    recent_evidence: Any = None,
    rd_conf: Any = None,
    hybrid_agree: Any = None,
    bo: Any = None,
    is_playoffs: Any = None,
    patch_status: str = "",
    interregional: bool = False,
    unknown_macro: bool = False,
    roster_changed: bool = False,
    consistency_guard_triggered: bool = False,
    consistency_guard_drift_pp: Any = None,
    rank_sos_applied: bool = False,
    rank_sos_weight: Any = None,
    rank_sos_diff_pp: Any = None,
    rank_sos_p: Any = None,
    roster_cont_applied: bool = False,
    roster_cont_weight: Any = None,
    roster_cont_diff_pp: Any = None,
    roster_cont_target_p: Any = None,
    league_a: str = "",
    league_b: str = "",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "",
        "method": "meta_blend_c_model_v3",
        "p_anchor": (_f(p_anchor) if _isfinite(p_anchor) else None),
        "p_before": (_f(p_adjusted) if _isfinite(p_adjusted) else None),
        "p_after": None,
        "diff_pp": None,
    }
    if not isinstance(model_cfg, dict):
        out["reason"] = "invalid_model_cfg"
        return out
    if not bool(model_cfg.get("enabled", True)):
        out["reason"] = "model_disabled"
        return out

    feat = build_meta_blend_c_model_features(
        p_anchor=p_anchor,
        p_adjusted=p_adjusted,
        n_team_eff=n_team_eff,
        n_players_eff=n_players_eff,
        recent_evidence=recent_evidence,
        rd_conf=rd_conf,
        hybrid_agree=hybrid_agree,
        bo=bo,
        is_playoffs=is_playoffs,
        patch_status=patch_status,
        interregional=interregional,
        unknown_macro=unknown_macro,
        roster_changed=roster_changed,
        consistency_guard_triggered=consistency_guard_triggered,
        consistency_guard_drift_pp=consistency_guard_drift_pp,
        rank_sos_applied=rank_sos_applied,
        rank_sos_weight=rank_sos_weight,
        rank_sos_diff_pp=rank_sos_diff_pp,
        rank_sos_p=rank_sos_p,
        roster_cont_applied=roster_cont_applied,
        roster_cont_weight=roster_cont_weight,
        roster_cont_diff_pp=roster_cont_diff_pp,
        roster_cont_target_p=roster_cont_target_p,
    )
    if not bool(feat.get("ok")):
        out["reason"] = str(feat.get("reason") or "invalid_features")
        return out

    feat_map = dict(feat.get("features") or {})
    strength_info = _interregional_strength_profile(league_a, league_b) if bool(interregional) else {}
    cross_strength = bool((strength_info or {}).get("cross_strength", False))
    very_strong_cross_strength = bool((strength_info or {}).get("very_strong_cross_strength", False))

    def _predict_logit(cfg_any: Dict[str, Any]) -> float | None:
        if not isinstance(cfg_any, dict):
            return None
        names_any = list(cfg_any.get("features") or [])
        coef_any = list(cfg_any.get("coef") or [])
        if not names_any:
            names_any = list(META_BLEND_C_MODEL_FEATURES)
        if any(str(k) not in feat_map for k in names_any):
            return None
        if len(coef_any) != len(names_any):
            return None
        try:
            intercept_any = float(cfg_any.get("intercept", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        x_raw = [float(feat_map.get(str(k), 0.0)) for k in names_any]
        mu_any = cfg_any.get("x_mean")
        sd_any = cfg_any.get("x_std")
        if isinstance(mu_any, list) and isinstance(sd_any, list) and len(mu_any) == len(x_raw) and len(sd_any) == len(x_raw):
            x_use: List[float] = []
            for xv, m, s in zip(x_raw, mu_any, sd_any):
                try:
                    mm = float(m)
                    ss = float(s)
                except (TypeError, ValueError):
                    mm, ss = 0.0, 1.0
                if (not math.isfinite(ss)) or abs(ss) < 1e-12:
                    ss = 1.0
                x_use.append(float((float(xv) - mm) / ss))
        else:
            x_use = list(x_raw)
        z_any = float(intercept_any)
        for c, xv in zip(coef_any, x_use):
            try:
                z_any += float(c) * float(xv)
            except (TypeError, ValueError):
                return None
        return float(z_any)

    def _select_submodel(cfg_root: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        contexts = cfg_root.get("context_models") if isinstance(cfg_root.get("context_models"), dict) else {}
        if not isinstance(contexts, dict) or not contexts:
            return cfg_root, "global"
        la = _norm_key(league_a)
        lb = _norm_key(league_b)
        ma = _macro_key(la)
        mb = _macro_key(lb)
        inter = bool(ma and mb and ma != mb)
        pb = _patch_bucket_from_status(patch_status)
        try:
            bo_i = int(float(bo)) if bo is not None else 1
        except (TypeError, ValueError):
            bo_i = 1
        bo_i = 5 if bo_i >= 5 else (3 if bo_i >= 3 else 1)
        ph = _phase_bucket(bo=bo_i, is_playoffs=bool(is_playoffs))
        keys_try: List[str] = []
        if la and lb and la == lb and not inter:
            keys_try.extend(
                [
                    f"league:{la}|patch:{pb}|phase:{ph}",
                    f"league:{la}|patch:{pb}",
                    f"league:{la}|phase:{ph}",
                    f"league:{la}",
                ]
            )
        if ma and mb and ma == mb and not inter:
            keys_try.extend(
                [
                    f"macro:{ma}|patch:{pb}|phase:{ph}",
                    f"macro:{ma}|patch:{pb}",
                    f"macro:{ma}|phase:{ph}",
                    f"macro:{ma}",
                ]
            )
        keys_try.extend([f"global|patch:{pb}|phase:{ph}", f"global|patch:{pb}", f"global|phase:{ph}", "global"])
        for k in keys_try:
            cand = contexts.get(k)
            if isinstance(cand, dict) and cand:
                c = dict(cand)
                for fld in ("features", "x_mean", "x_std"):
                    if fld not in c and fld in cfg_root:
                        c[fld] = cfg_root.get(fld)
                if "version" not in c:
                    c["version"] = cfg_root.get("version", "meta_blend_c_model_v3")
                return c, str(k)
        return cfg_root, "global"

    model_use, model_ctx_key = _select_submodel(model_cfg)
    model_ver = str(model_use.get("version", model_cfg.get("version", "meta_blend_c_model_v3")))
    z_ctx = _predict_logit(model_use)
    if z_ctx is None:
        out["reason"] = "feature_schema_mismatch"
        return out

    def _triplet(cfg_any: Dict[str, Any]) -> tuple[float, float, float]:
        n_tot = max(0.0, _f(cfg_any.get("n_total"), 0.0))
        imp_adj = _f(cfg_any.get("improvement_vs_adjusted"), float("nan"))
        imp_anchor = _f(cfg_any.get("improvement_vs_anchor"), float("nan"))
        return float(n_tot), float(imp_adj), float(imp_anchor)

    guard_cfg = model_cfg.get("runtime_guard") if isinstance(model_cfg.get("runtime_guard"), dict) else {}
    guard_enabled = bool(guard_cfg.get("enabled", True)) if isinstance(guard_cfg, dict) and bool(guard_cfg) else False
    min_n = float(max(40.0, _f(guard_cfg.get("min_n_total"), 220.0)))
    min_imp_adj = float(_f(guard_cfg.get("min_improvement_vs_adjusted"), -0.003))
    min_imp_anchor = float(_f(guard_cfg.get("min_improvement_vs_anchor"), -0.0015))
    allow_context_override = bool(guard_cfg.get("allow_context_override", True))
    n_g, imp_adj_g, imp_anchor_g = _triplet(model_cfg if isinstance(model_cfg, dict) else {})
    n_u, imp_adj_u, imp_anchor_u = _triplet(model_use if isinstance(model_use, dict) else {})

    def _is_bad(imp_adj: float, imp_anchor: float, n_tot: float) -> bool:
        if not (math.isfinite(imp_adj) and math.isfinite(imp_anchor)):
            return False
        if n_tot < min_n:
            return False
        return bool(float(imp_adj) < float(min_imp_adj) and float(imp_anchor) < float(min_imp_anchor))

    global_bad = _is_bad(imp_adj_g, imp_anchor_g, n_g)
    use_bad = _is_bad(imp_adj_u, imp_anchor_u, n_u)
    guard_block = False
    if guard_enabled:
        if str(model_ctx_key) == "global":
            guard_block = bool(global_bad)
        else:
            if use_bad:
                guard_block = True
            elif (not allow_context_override) and global_bad:
                guard_block = True
            elif n_u < min_n and global_bad:
                guard_block = True
    guard_info = {
        "enabled": bool(guard_enabled),
        "blocked": bool(guard_block),
        "context_key": str(model_ctx_key),
        "selected_version": str(model_ver),
        "thresholds": {
            "min_n_total": float(min_n),
            "min_improvement_vs_adjusted": float(min_imp_adj),
            "min_improvement_vs_anchor": float(min_imp_anchor),
            "allow_context_override": bool(allow_context_override),
        },
        "selected_model": {
            "n_total": int(n_u),
            "improvement_vs_adjusted": (float(imp_adj_u) if math.isfinite(imp_adj_u) else None),
            "improvement_vs_anchor": (float(imp_anchor_u) if math.isfinite(imp_anchor_u) else None),
            "underperforming": bool(use_bad),
        },
        "global_model": {
            "n_total": int(n_g),
            "improvement_vs_adjusted": (float(imp_adj_g) if math.isfinite(imp_adj_g) else None),
            "improvement_vs_anchor": (float(imp_anchor_g) if math.isfinite(imp_anchor_g) else None),
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
                    "trained_on_n": int(n_u),
                    "oos_logloss": model_use.get("oos_logloss", model_cfg.get("oos_logloss")),
                    "improvement_vs_adjusted": (float(imp_adj_u) if math.isfinite(imp_adj_u) else None),
                    "improvement_vs_anchor": (float(imp_anchor_u) if math.isfinite(imp_anchor_u) else None),
                    "guard": guard_info,
                },
            }
        )
        return out

    def _resolve_runtime_stability_cfg(
        cfg_any: Dict[str, Any],
        *,
        model_context_key: str,
        patch_bucket: str,
    ) -> Dict[str, Any]:
        rs = cfg_any.get("runtime_stability") if isinstance(cfg_any.get("runtime_stability"), dict) else {}
        if not isinstance(rs, dict):
            rs = {}

        def _num_from(src: Dict[str, Any], name: str, default: float, lo: float, hi: float) -> float:
            v = _f(src.get(name), default)
            if not math.isfinite(v):
                v = float(default)
            return float(max(float(lo), min(float(hi), float(v))))

        def _build_cfg(src: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "enabled": bool(src.get("enabled", True)),
                "cap_pp_lo": _num_from(src, "cap_pp_lo", 2.2, 0.5, 10.0),
                "cap_pp_hi_bo1": _num_from(src, "cap_pp_hi_bo1", 5.2, 1.0, 18.0),
                "cap_pp_hi_bo3": _num_from(src, "cap_pp_hi_bo3", 6.3, 1.0, 18.0),
                "cap_pp_hi_bo5": _num_from(src, "cap_pp_hi_bo5", 7.2, 1.0, 18.0),
                "small_delta_logit": _num_from(src, "small_delta_logit", 0.08, 0.002, 0.60),
                "tiny_delta_logit": _num_from(src, "tiny_delta_logit", 0.03, 0.001, 0.35),
                "cap_pp_small_delta": _num_from(src, "cap_pp_small_delta", 3.2, 0.4, 10.0),
                "cap_pp_tiny_delta": _num_from(src, "cap_pp_tiny_delta", 2.2, 0.3, 10.0),
                "mult_interregional": _num_from(src, "mult_interregional", 0.88, 0.40, 1.20),
                "mult_patch_refit": _num_from(src, "mult_patch_refit", 0.86, 0.40, 1.20),
                "mult_patch_warn": _num_from(src, "mult_patch_warn", 0.94, 0.40, 1.20),
                "mult_roster_changed": _num_from(src, "mult_roster_changed", 0.93, 0.40, 1.20),
                "weak_imp_threshold": _num_from(src, "weak_imp_threshold", -0.001, -0.20, 0.10),
                "mult_weak_improvement": _num_from(src, "mult_weak_improvement", 0.92, 0.40, 1.20),
            }

        def _merge_cfg(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
            if not isinstance(src, dict) or not isinstance(dst, dict):
                return

            def _maybe_num(name: str, default: float, lo: float, hi: float) -> None:
                if name in src:
                    dst[name] = _num_from(src, name, default, lo, hi)

            if "enabled" in src:
                dst["enabled"] = bool(src.get("enabled", dst.get("enabled", True)))
            _maybe_num("cap_pp_lo", float(dst.get("cap_pp_lo", 2.2)), 0.5, 10.0)
            _maybe_num("cap_pp_hi_bo1", float(dst.get("cap_pp_hi_bo1", 5.2)), 1.0, 18.0)
            _maybe_num("cap_pp_hi_bo3", float(dst.get("cap_pp_hi_bo3", 6.3)), 1.0, 18.0)
            _maybe_num("cap_pp_hi_bo5", float(dst.get("cap_pp_hi_bo5", 7.2)), 1.0, 18.0)
            _maybe_num("small_delta_logit", float(dst.get("small_delta_logit", 0.08)), 0.002, 0.60)
            _maybe_num("tiny_delta_logit", float(dst.get("tiny_delta_logit", 0.03)), 0.001, 0.35)
            _maybe_num("cap_pp_small_delta", float(dst.get("cap_pp_small_delta", 3.2)), 0.4, 10.0)
            _maybe_num("cap_pp_tiny_delta", float(dst.get("cap_pp_tiny_delta", 2.2)), 0.3, 10.0)
            _maybe_num("mult_interregional", float(dst.get("mult_interregional", 0.88)), 0.40, 1.20)
            _maybe_num("mult_patch_refit", float(dst.get("mult_patch_refit", 0.86)), 0.40, 1.20)
            _maybe_num("mult_patch_warn", float(dst.get("mult_patch_warn", 0.94)), 0.40, 1.20)
            _maybe_num("mult_roster_changed", float(dst.get("mult_roster_changed", 0.93)), 0.40, 1.20)
            _maybe_num("weak_imp_threshold", float(dst.get("weak_imp_threshold", -0.001)), -0.20, 0.10)
            _maybe_num("mult_weak_improvement", float(dst.get("mult_weak_improvement", 0.92)), 0.40, 1.20)

        cfg_out = _build_cfg(rs)
        map_any = rs.get("map") if isinstance(rs.get("map"), dict) else {}
        map_cfg = map_any if isinstance(map_any, dict) else {}

        ctx = str(model_context_key or "global")
        patch = str(patch_bucket or "stable")
        ctx_no_phase = str(ctx.split("|phase:")[0]) if "|phase:" in ctx else str(ctx)
        ctx_no_patch = str(ctx_no_phase.split("|patch:")[0]) if "|patch:" in ctx_no_phase else str(ctx_no_phase)
        ctx_patch = str(ctx_no_phase)
        if "|patch:" not in ctx_patch and ctx_no_patch and ctx_no_patch != "global":
            ctx_patch = f"{ctx_no_patch}|patch:{patch}"

        keys_try: List[str] = []
        for k in ["global", f"global|patch:{patch}", ctx_no_patch, ctx_patch, ctx_no_phase, ctx]:
            kk = str(k or "").strip()
            if kk and kk not in keys_try:
                keys_try.append(kk)

        applied_keys: List[str] = []
        for k in keys_try:
            node = map_cfg.get(k) if isinstance(map_cfg.get(k), dict) else {}
            if isinstance(node, dict) and node:
                _merge_cfg(cfg_out, node)
                applied_keys.append(str(k))

        cfg_out["selected_key"] = (str(applied_keys[-1]) if applied_keys else "")
        cfg_out["applied_keys"] = list(applied_keys)
        # Safety consistency: tiny cap cannot exceed small cap.
        cfg_out["cap_pp_small_delta"] = float(max(cfg_out["cap_pp_tiny_delta"], cfg_out["cap_pp_small_delta"]))
        return cfg_out

    z = float(z_ctx)
    context_blend: Dict[str, Any] = {"used": False}
    if str(model_ctx_key) != "global":
        z_global = _predict_logit(model_cfg)
        if z_global is not None:
            n_ctx = max(0.0, _f(model_use.get("n_total"), 0.0))
            imp_adj = _f(model_use.get("improvement_vs_adjusted"), 0.0)
            imp_anchor = _f(model_use.get("improvement_vs_anchor"), 0.0)
            bad_imp_count = ((1 if (math.isfinite(imp_adj) and float(imp_adj) < -0.002) else 0)
                + (1 if (math.isfinite(imp_anchor) and float(imp_anchor) < -0.002) else 0))
            ctx_min_n = max(40.0, _f(model_cfg.get("ctx_min_n"), 120.0))
            w_n = float(max(0.15, min(1.0, n_ctx / (n_ctx + 260.0))))
            q = 0.0
            if math.isfinite(imp_adj):
                q += max(0.0, min(0.08, float(imp_adj))) * 6.0
            if math.isfinite(imp_anchor):
                q += max(0.0, min(0.08, float(imp_anchor))) * 4.0
            w_q = float(max(0.25, min(1.0, 0.35 + q)))
            if "|patch:" in str(model_ctx_key):
                w_n *= 0.90
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
                "improvement_vs_adjusted": (float(imp_adj) if math.isfinite(imp_adj) else None),
                "improvement_vs_anchor": (float(imp_anchor) if math.isfinite(imp_anchor) else None),
            }

    p_before = float(_clip_prob(_f(p_adjusted)))
    z_before = float(_safe_logit(p_before))
    out_shrink = 1.0
    if str(model_ctx_key) != "global":
        recent_w = max(0.0, min(1.0, _f(recent_evidence, 0.0)))
        out_shrink = float((context_blend.get("w_ctx_blend") if isinstance(context_blend, dict) else 0.5) or 0.5)
        out_shrink *= (0.86 + 0.14 * recent_w)
        if bool(interregional):
            out_shrink *= 0.88
        if bool(roster_changed):
            out_shrink *= 0.94
        if _patch_bucket_from_status(patch_status) == "refit":
            out_shrink *= 0.90
        try:
            bo_i = int(float(bo)) if bo is not None else 1
        except (TypeError, ValueError):
            bo_i = 1
        if bo_i <= 1:
            out_shrink *= 0.92
        elif bo_i >= 5:
            out_shrink *= 1.03
        out_shrink = float(max(0.35, min(1.0, out_shrink)))

    z_final = float(z_before + out_shrink * (z - z_before))
    p_out = float(_clip_prob(_safe_sigmoid(z_final)))
    p_before = float(_clip_prob(_f(p_adjusted)))
    diff_pp_raw = float(100.0 * (p_out - p_before))

    # Runtime stability guardrail (v2.2):
    # cap C-layer movement in pp using contextual support and anchor-adjusted delta.
    rs_cfg = _resolve_runtime_stability_cfg(
        (model_cfg if isinstance(model_cfg, dict) else {}),
        model_context_key=str(model_ctx_key),
        patch_bucket=_patch_bucket_from_status(patch_status),
    )
    rs_cap_applied = False
    rs_modifiers: List[str] = []
    rs_support = 0.5
    rs_cap_pp = float("nan")
    if bool(rs_cfg.get("enabled", True)):
        try:
            bo_i = int(float(bo)) if bo is not None else 1
        except (TypeError, ValueError):
            bo_i = 1
        bo_i = 5 if bo_i >= 5 else (3 if bo_i >= 3 else 1)

        ev_recent = max(0.0, min(1.0, _f(recent_evidence, _f(feat_map.get("recent_evidence"), 0.0))))
        rd_c = max(0.10, min(1.0, _f(rd_conf, _f(feat_map.get("rd_conf"), 0.5))))
        ev_team = max(0.0, _f(n_team_eff, float("nan")))
        if not math.isfinite(ev_team):
            ev_team = max(0.0, math.expm1(max(0.0, _f(feat_map.get("log1p_n_team"), 0.0))))
        ev_players = max(0.0, _f(n_players_eff, float("nan")))
        if not math.isfinite(ev_players):
            ev_players = max(0.0, math.expm1(max(0.0, _f(feat_map.get("log1p_n_players"), 0.0))))
        ev_team_w = float(max(0.0, min(1.0, ev_team / (ev_team + 20.0)))) if ev_team > 0.0 else 0.0
        ev_players_w = float(max(0.0, min(1.0, ev_players / (ev_players + 10.0)))) if ev_players > 0.0 else 0.0
        ctx_w = float(
            max(
                0.0,
                min(
                    1.0,
                    _f(
                        (context_blend.get("w_ctx_blend") if isinstance(context_blend, dict) else None),
                        0.62 if str(model_ctx_key) == "global" else 0.50,
                    ),
                ),
            )
        )
        imp_quality = 0.55
        if math.isfinite(imp_adj_u) and math.isfinite(imp_anchor_u):
            imp_mean = 0.5 * (float(imp_adj_u) + float(imp_anchor_u))
            imp_quality = float(max(0.20, min(1.0, 0.55 + 6.0 * max(-0.06, min(0.06, imp_mean)))))
        rs_support = float(
            max(
                0.0,
                min(
                    1.0,
                    (0.33 * ev_recent)
                    + (0.20 * rd_c)
                    + (0.22 * ev_team_w)
                    + (0.10 * ev_players_w)
                    + (0.10 * ctx_w)
                    + (0.05 * imp_quality),
                ),
            )
        )
        cap_lo = float(rs_cfg.get("cap_pp_lo", 2.2))
        if bo_i <= 1:
            cap_hi = float(rs_cfg.get("cap_pp_hi_bo1", 5.2))
        elif bo_i >= 5:
            cap_hi = float(rs_cfg.get("cap_pp_hi_bo5", 7.2))
        else:
            cap_hi = float(rs_cfg.get("cap_pp_hi_bo3", 6.3))
        cap_pp = float(cap_lo + ((cap_hi - cap_lo) * rs_support))

        abs_dz = abs(_f(feat_map.get("abs_z_delta"), float("nan")))
        if not math.isfinite(abs_dz):
            abs_dz = abs(float(_safe_logit(_clip_prob(_f(p_adjusted))) - _safe_logit(_clip_prob(_f(p_anchor)))))
        if abs_dz <= float(rs_cfg.get("small_delta_logit", 0.08)):
            cap_pp = min(cap_pp, float(rs_cfg.get("cap_pp_small_delta", 3.2)))
            rs_modifiers.append("small_anchor_adjusted_delta")
        if abs_dz <= float(rs_cfg.get("tiny_delta_logit", 0.03)):
            cap_pp = min(cap_pp, float(rs_cfg.get("cap_pp_tiny_delta", 2.2)))
            rs_modifiers.append("tiny_anchor_adjusted_delta")

        pbucket = _patch_bucket_from_status(patch_status)
        if bool(interregional):
            cap_pp *= float(rs_cfg.get("mult_interregional", 0.88))
            rs_modifiers.append("interregional")
            if cross_strength:
                cap_pp *= 0.82 if not very_strong_cross_strength else 0.72
                rs_modifiers.append(
                    "cross_strength_interregional" if not very_strong_cross_strength else "very_strong_cross_strength_interregional"
                )
        if bool(roster_changed):
            cap_pp *= float(rs_cfg.get("mult_roster_changed", 0.93))
            rs_modifiers.append("roster_changed")
        if pbucket == "refit":
            cap_pp *= float(rs_cfg.get("mult_patch_refit", 0.86))
            rs_modifiers.append("patch_refit")
        elif pbucket == "warn":
            cap_pp *= float(rs_cfg.get("mult_patch_warn", 0.94))
            rs_modifiers.append("patch_warn")
        if math.isfinite(imp_adj_u) and math.isfinite(imp_anchor_u):
            weak_thr = float(rs_cfg.get("weak_imp_threshold", -0.001))
            if float(imp_adj_u) < weak_thr and float(imp_anchor_u) < weak_thr:
                cap_pp *= float(rs_cfg.get("mult_weak_improvement", 0.92))
                rs_modifiers.append("weak_oos_improvement")
        cap_pp = float(max(0.30, min(cap_hi, cap_pp)))
        rs_cap_pp = float(cap_pp)
        if abs(diff_pp_raw) > cap_pp:
            p_out = float(_clip_prob(p_before + math.copysign(cap_pp / 100.0, diff_pp_raw)))
            rs_cap_applied = True
            rs_modifiers.append("cap_applied")
        if cross_strength:
            stronger_side = str((strength_info or {}).get("stronger_side") or "")
            before_stronger = float(p_before if stronger_side == "a" else (1.0 - p_before))
            after_stronger = float(p_out if stronger_side == "a" else (1.0 - p_out))
            adverse_pp = float(max(0.0, (before_stronger - after_stronger) * 100.0))
            adverse_cap_pp = 2.4 if not very_strong_cross_strength else 1.4
            if adverse_pp > adverse_cap_pp:
                clipped_stronger = float(before_stronger - (adverse_cap_pp / 100.0))
                clipped_stronger = max(1e-6, min(1.0 - 1e-6, clipped_stronger))
                p_out = float(clipped_stronger if stronger_side == "a" else (1.0 - clipped_stronger))
                rs_cap_applied = True
                rs_modifiers.append("cross_strength_directional_cap")

    diff_pp_final = float(100.0 * (p_out - p_before))
    z_final_out = float(_safe_logit(p_out))
    out.update(
        {
            "applied": True,
            "reason": "meta_blend_c_model",
            "p_after": float(p_out),
            "diff_pp": float(diff_pp_final),
            "features": (feat.get("features") or {}),
            "targets": {
                "z_model": float(z),
                "z_before": float(z_before),
                "z_final_pre_cap": float(z_final),
                "z_final": float(z_final_out),
            },
            "model_meta": {
                "version": str(model_ver),
                "trained_on_n": int(model_use.get("n_total", model_cfg.get("n_total", 0)) or 0),
                "oos_logloss": model_use.get("oos_logloss", model_cfg.get("oos_logloss")),
                "improvement_vs_adjusted": (
                    float(_f(model_use.get("improvement_vs_adjusted"), float("nan")))
                    if math.isfinite(_f(model_use.get("improvement_vs_adjusted"), float("nan")))
                    else None
                ),
                "improvement_vs_anchor": (
                    float(_f(model_use.get("improvement_vs_anchor"), float("nan")))
                    if math.isfinite(_f(model_use.get("improvement_vs_anchor"), float("nan")))
                    else None
                ),
                "context_key": str(model_ctx_key),
                "guard": guard_info,
                "context_blend": context_blend,
                "output_shrink": {"used": bool(str(model_ctx_key) != "global"), "value": float(out_shrink)},
                "runtime_stability": {
                    "enabled": bool(rs_cfg.get("enabled", True)),
                    "selected_key": str(rs_cfg.get("selected_key", "") or ""),
                    "applied_keys": list(rs_cfg.get("applied_keys", [])) if isinstance(rs_cfg.get("applied_keys"), list) else [],
                    "support_score": (float(rs_support) if math.isfinite(float(rs_support)) else None),
                    "cap_pp": (float(rs_cap_pp) if math.isfinite(float(rs_cap_pp)) else None),
                    "raw_diff_pp": float(diff_pp_raw),
                    "final_diff_pp": float(diff_pp_final),
                    "capped": bool(rs_cap_applied),
                    "modifiers": list(rs_modifiers),
                    "config": dict(rs_cfg),
                    "interregional_strength": dict(strength_info) if isinstance(strength_info, dict) else {},
                },
            },
        }
    )
    out["method"] = str((out.get("model_meta") or {}).get("version") or "meta_blend_c_model_v3")
    return out


def apply_meta_blend_c_lite(
    *,
    p_anchor: Any,
    p_adjusted: Any,
    n_team_eff: Any = None,
    n_players_eff: Any = None,
    recent_evidence: Any = None,
    rd_conf: Any = None,
    patch_status: str = "",
    interregional: bool = False,
    unknown_macro: bool = False,
    roster_changed: bool = False,
    consistency_guard_triggered: bool = False,
    consistency_guard_drift_pp: Any = None,
    hybrid_agree: Any = None,
    rank_sos_applied: bool = False,
    rank_sos_weight: Any = None,
    rank_sos_diff_pp: Any = None,
    rank_sos_p: Any = None,
    roster_cont_applied: bool = False,
    roster_cont_weight: Any = None,
    roster_cont_diff_pp: Any = None,
    roster_cont_target_p: Any = None,
    league_a: str = "",
    league_b: str = "",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "",
        "p_anchor": (_f(p_anchor) if _isfinite(p_anchor) else None),
        "p_before": (_f(p_adjusted) if _isfinite(p_adjusted) else None),
        "p_after": None,
        "diff_pp": None,
    }
    if not (_isfinite(p_anchor) and _isfinite(p_adjusted)):
        out["reason"] = "invalid_inputs"
        return out

    pa = _clip_prob(_f(p_anchor))
    pb = _clip_prob(_f(p_adjusted))
    za = _safe_logit(pa)
    zb = _safe_logit(pb)
    dz = float(zb - za)
    if abs(dz) < 1e-12:
        out["reason"] = "no_delta"
        out["applied"] = True
        out["p_after"] = float(pb)
        out["diff_pp"] = 0.0
        out["weights"] = {"meta_gate": 1.0}
        return out

    ev_team = max(0.0, _f(n_team_eff, 0.0))
    ev_players = max(0.0, _f(n_players_eff, 0.0))
    ev_recent = max(0.0, min(1.0, _f(recent_evidence, 0.0)))
    rd_c = _f(rd_conf, 0.5)
    if not math.isfinite(rd_c):
        rd_c = 0.5
    rd_c = max(0.10, min(1.0, rd_c))
    h_agree = _f(hybrid_agree, 0.5)
    if not math.isfinite(h_agree):
        h_agree = 0.5
    h_agree = max(0.0, min(1.0, h_agree))
    rs_w = max(0.0, min(1.0, _f(rank_sos_weight, 0.0)))
    rc_w = max(0.0, min(1.0, _f(roster_cont_weight, 0.0)))
    rs_diff = abs(_f(rank_sos_diff_pp, 0.0))
    if not math.isfinite(rs_diff):
        rs_diff = 0.0
    rc_diff = abs(_f(roster_cont_diff_pp, 0.0))
    if not math.isfinite(rc_diff):
        rc_diff = 0.0
    strength_info = _interregional_strength_profile(league_a, league_b) if bool(interregional) else {}
    cross_strength = bool((strength_info or {}).get("cross_strength", False))
    very_strong_cross_strength = bool((strength_info or {}).get("very_strong_cross_strength", False))

    # Contextual gate approximating a learned meta-model.
    w = 0.22
    w += 0.28 * (ev_team / (ev_team + 10.0)) if ev_team > 0 else 0.0
    w += 0.14 * (ev_players / (ev_players + 8.0)) if ev_players > 0 else 0.0
    w += 0.18 * ev_recent
    w += 0.12 * rd_c
    # If hybrid agrees with model strongly, reduce trust in extra adjustments; if disagrees, allow a bit more.
    w += 0.06 * (1.0 - h_agree)
    # Runtime edges from Rank-SOS / Roster-continuity.
    if bool(rank_sos_applied):
        w += 0.03 * rs_w
        if rs_diff > 8.0:
            w *= 0.96
    if bool(roster_cont_applied):
        w += 0.03 * rc_w
        if rc_diff > 8.0:
            w *= 0.96

    if interregional:
        w *= 0.78
        if cross_strength:
            w *= 0.82 if not very_strong_cross_strength else 0.72
    elif unknown_macro:
        w *= 0.86
    if str(patch_status or "").strip().lower() in {"refit", "refit_needed", "changed"}:
        w *= 0.90
    if roster_changed:
        w *= 1.06

    drift_pp = abs(_f(consistency_guard_drift_pp, 0.0))
    if bool(consistency_guard_triggered) and math.isfinite(drift_pp):
        w *= max(0.70, 1.0 - min(0.18, drift_pp / 120.0))

    # Large deltas need more evidence to pass through.
    delta_mag = abs(dz)
    if delta_mag > 0.35:
        support = 0.35 + 0.35 * ev_recent + 0.20 * rd_c + 0.10 * (ev_team / (ev_team + 12.0) if ev_team > 0 else 0.0)
        w *= max(0.55, min(1.0, support))

    w = float(max(0.18, min(0.92, w)))
    dz_cap = 1.05
    if interregional:
        dz_cap *= 0.80
        if cross_strength:
            dz_cap *= 0.86 if not very_strong_cross_strength else 0.74
    elif unknown_macro:
        dz_cap *= 0.90
    dz_gated = float(max(-dz_cap, min(dz_cap, dz * w)))
    z_out = float(za + dz_gated)
    p_out = float(_clip_prob(_safe_sigmoid(z_out)))
    if cross_strength:
        stronger_side = str((strength_info or {}).get("stronger_side") or "")
        before_stronger = float(pb if stronger_side == "a" else (1.0 - pb))
        after_stronger = float(p_out if stronger_side == "a" else (1.0 - p_out))
        adverse_pp = float(max(0.0, (before_stronger - after_stronger) * 100.0))
        adverse_cap_pp = 2.4 if not very_strong_cross_strength else 1.4
        if adverse_pp > adverse_cap_pp:
            clipped_stronger = float(before_stronger - (adverse_cap_pp / 100.0))
            clipped_stronger = max(1e-6, min(1.0 - 1e-6, clipped_stronger))
            p_out = float(clipped_stronger if stronger_side == "a" else (1.0 - clipped_stronger))

    out.update(
        {
            "applied": True,
            "reason": "meta_blend_c_lite",
            "p_after": float(p_out),
            "diff_pp": float(100.0 * (p_out - pb)),
            "weights": {"meta_gate": float(w)},
            "inputs": {
                "n_team_eff": float(ev_team),
                "n_players_eff": float(ev_players),
                "recent_evidence": float(ev_recent),
                "rd_conf": float(rd_c),
                "hybrid_agree": float(h_agree),
                "rank_sos_applied": bool(rank_sos_applied),
                "rank_sos_weight": float(rs_w),
                "rank_sos_diff_pp": float(rs_diff),
                "rank_sos_p": (_f(rank_sos_p, 0.5) if _isfinite(rank_sos_p) else None),
                "roster_cont_applied": bool(roster_cont_applied),
                "roster_cont_weight": float(rc_w),
                "roster_cont_diff_pp": float(rc_diff),
                "roster_cont_target_p": (_f(roster_cont_target_p, 0.5) if _isfinite(roster_cont_target_p) else None),
                "patch_status": str(patch_status or ""),
                "interregional": bool(interregional),
                "interregional_strength": dict(strength_info) if isinstance(strength_info, dict) else {},
                "unknown_macro": bool(unknown_macro),
                "roster_changed": bool(roster_changed),
                "consistency_guard_triggered": bool(consistency_guard_triggered),
                "consistency_guard_drift_pp": (float(drift_pp) if math.isfinite(drift_pp) else None),
            },
            "targets": {
                "z_anchor": float(za),
                "z_before": float(zb),
                "z_delta": float(dz),
                "z_delta_gated": float(dz_gated),
                "z_cap": float(dz_cap),
                "z_out": float(z_out),
            },
        }
    )
    return out
