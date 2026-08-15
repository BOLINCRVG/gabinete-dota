from __future__ import annotations

import math
from typing import Any, Callable, Dict, Tuple

from mlcore.utils import clip_prob, safe_float as _safe_float


_LANE_KEYS = ("Top", "Jungle", "Mid", "Bot", "Support")


def _league_macro_key(league_hint: str, league_macro_key_fn: Callable[[str], str] | None) -> str:
    lg = str(league_hint or "").strip()
    if not lg or league_macro_key_fn is None:
        return ""
    try:
        return str(league_macro_key_fn(lg) or "").strip()
    except Exception:
        return ""


def build_fit_runtime_state(
    fit_cfg: Dict[str, Any] | None,
    *,
    sanitize_marginal_band_coeffs: Callable[[Any], Dict[str, Dict[str, float]]] | None = None,
    current_marginal_band_coeffs: Any = None,
) -> Dict[str, Any]:
    cfg = fit_cfg if isinstance(fit_cfg, dict) else {}
    sanitizer = sanitize_marginal_band_coeffs or (lambda raw: dict(raw) if isinstance(raw, dict) else {})

    fit_team_scale = max(0.0, _safe_float((((cfg.get("best") or {}).get("team_scale")) or 1.0), 1.0))
    fit_players_scale = max(0.0, _safe_float((((cfg.get("best") or {}).get("players_scale")) or 1.0), 1.0))
    fit_form_cfg = dict((cfg.get("best_form") or {}).get("settings") or {})
    lane_weights_cfg = dict(cfg.get("players_lane_weights") or {})
    players_league_weights_cfg = dict(cfg.get("players_league_weights") or {})

    players_context_cfg_raw = cfg.get("players_context_fit") or {}
    players_context_cfg = dict(players_context_cfg_raw) if isinstance(players_context_cfg_raw, dict) else {}

    fused_post_cal_cfg_raw = cfg.get("fused_post_calibration") or {}
    fused_post_cal_cfg = dict(fused_post_cal_cfg_raw) if isinstance(fused_post_cal_cfg_raw, dict) else {}

    series_gamma_cfg = {
        "global": dict(cfg.get("series_gamma_global") or {}),
        "by_league": dict(cfg.get("series_gamma_by_league") or {}),
        "by_macro": dict(cfg.get("series_gamma_by_macro") or {}),
    }

    series_bo_dplite_cfg_raw = cfg.get("series_bo_dplite_tuning") or {}
    series_bo_dplite_cfg = dict(series_bo_dplite_cfg_raw) if isinstance(series_bo_dplite_cfg_raw, dict) else {}

    fit_mtbc_cfg = sanitizer(cfg.get("ml_marginal_trust_band_coeffs") or {})
    current_mtbc = sanitizer(current_marginal_band_coeffs or {})

    state = {
        "_fit_team_scale": float(fit_team_scale),
        "_fit_players_scale": float(fit_players_scale),
        "_fit_team_scale_used": float(fit_team_scale),
        "_fit_players_scale_used": float(fit_players_scale),
        "_fit_scale_league_used": "",
        "_fit_form_cfg": dict(fit_form_cfg),
        "_fit_form_cfg_used": dict(fit_form_cfg),
        "_fit_form_cfg_league_used": "",
        "_fit_lane_weights_cfg": dict(lane_weights_cfg) if isinstance(lane_weights_cfg, dict) else {},
        "_fit_players_league_weights_cfg": (
            dict(players_league_weights_cfg) if isinstance(players_league_weights_cfg, dict) else {}
        ),
        "_fit_players_context_cfg": dict(players_context_cfg),
        "_fit_fused_post_cal_cfg": dict(fused_post_cal_cfg),
        "_fit_series_gamma_cfg": dict(series_gamma_cfg),
        "_fit_series_bo_dplite_cfg": dict(series_bo_dplite_cfg),
        "_fit_marginal_trust_band_coeffs_cfg": dict(fit_mtbc_cfg),
    }
    if not current_mtbc and fit_mtbc_cfg:
        state["ml_marginal_trust_band_coeffs"] = dict(fit_mtbc_cfg)
    return state


def fit_scales_for_league(
    fit_cfg: Dict[str, Any] | None,
    fit_team_scale: float,
    fit_players_scale: float,
    league_hint: str,
) -> Tuple[float, float, str]:
    lg = str(league_hint or "").strip()
    cfg = fit_cfg if isinstance(fit_cfg, dict) else {}
    if not cfg:
        return float(fit_team_scale), float(fit_players_scale), ""
    by_lg = cfg.get("best_by_league") or {}
    if isinstance(by_lg, dict) and lg:
        row = by_lg.get(lg)
        if isinstance(row, dict):
            ts = max(0.0, _safe_float(row.get("team_scale", fit_team_scale), fit_team_scale))
            ps = max(0.0, _safe_float(row.get("players_scale", fit_players_scale), fit_players_scale))
            return ts, ps, lg
    return float(fit_team_scale), float(fit_players_scale), ""


def fit_form_for_league(
    fit_cfg: Dict[str, Any] | None,
    fit_form_cfg: Dict[str, Any] | None,
    league_hint: str,
    *,
    league_macro_key_fn: Callable[[str], str] | None = None,
) -> Tuple[Dict[str, Any], str]:
    lg = str(league_hint or "").strip()
    base = dict(fit_form_cfg) if isinstance(fit_form_cfg, dict) else {}
    cfg = fit_cfg if isinstance(fit_cfg, dict) else {}
    if not cfg:
        return base, ""
    by_lg = cfg.get("best_form_by_league") or {}
    if isinstance(by_lg, dict) and lg:
        row = by_lg.get(lg)
        if isinstance(row, dict):
            settings = row.get("settings")
            if isinstance(settings, dict) and settings:
                return dict(settings), lg
    by_mk = cfg.get("best_form_by_macro") or {}
    if isinstance(by_mk, dict):
        mk = _league_macro_key(lg, league_macro_key_fn)
        row = by_mk.get(mk)
        if isinstance(row, dict):
            settings = row.get("settings")
            if isinstance(settings, dict) and settings:
                return dict(settings), f"macro:{mk}"
    return base, ""


def fit_players_lane_for_league(
    lane_weights_cfg: Dict[str, Any] | None,
    league_hint: str,
    *,
    league_macro_key_fn: Callable[[str], str] | None = None,
) -> Tuple[Dict[str, float], str]:
    lg = str(league_hint or "").strip()
    cfg = lane_weights_cfg if isinstance(lane_weights_cfg, dict) else {}
    if not cfg:
        return {}, ""
    global_cfg = cfg.get("global") if isinstance(cfg.get("global"), dict) else {}
    by_lg = cfg.get("by_league") if isinstance(cfg.get("by_league"), dict) else {}
    by_mk = cfg.get("by_macro") if isinstance(cfg.get("by_macro"), dict) else {}

    row: Dict[str, Any] = {}
    src = ""
    if lg and isinstance(by_lg.get(lg), dict):
        row = dict(by_lg.get(lg) or {})
        src = f"league:{lg}"
    elif lg:
        mk = _league_macro_key(lg, league_macro_key_fn)
        if mk and isinstance(by_mk.get(mk), dict):
            row = dict(by_mk.get(mk) or {})
            src = f"macro:{mk}"
    if not row and isinstance(global_cfg, dict):
        row = dict(global_cfg)
        src = "global"

    out: Dict[str, float] = {}
    for key in _LANE_KEYS:
        out[key] = _safe_float(row.get(key, 1.0), 1.0)
    return out, src


def fit_players_league_weight_for_league(
    players_league_weights_cfg: Dict[str, Any] | None,
    league_hint: str,
    *,
    league_macro_key_fn: Callable[[str], str] | None = None,
) -> Tuple[float, str]:
    lg = str(league_hint or "").strip()
    cfg = players_league_weights_cfg if isinstance(players_league_weights_cfg, dict) else {}
    if not cfg:
        return 1.0, ""

    by_lg = cfg.get("by_league") if isinstance(cfg.get("by_league"), dict) else {}
    by_mk = cfg.get("by_macro") if isinstance(cfg.get("by_macro"), dict) else {}
    global_weight = cfg.get("global")

    if lg and lg in by_lg:
        return _safe_float(by_lg.get(lg), 1.0), f"league:{lg}"
    mk = _league_macro_key(lg, league_macro_key_fn)
    if mk and mk in by_mk:
        return _safe_float(by_mk.get(mk), 1.0), f"macro:{mk}"
    return _safe_float(global_weight, 1.0), "global"


_PLAYERS_CTX_REGRESSION_REASONS = frozenset({
    "logloss_regression", "brier_regression", "ece_regression",
})
_PLAYERS_CTX_DATA_REASONS = frozenset({"low_sample", "not_evaluated"})


def _players_context_shadow_fallback(
    cfg: Dict[str, Any],
    gate: Dict[str, Any],
    decision: str,
) -> Tuple[Dict[str, Any], str]:
    """Conservative settings for shadow_only gates where signal may be real.

    Never applies when there was an active regression (signal was harmful).
    Uses fitted settings scaled down so the model can contribute cautiously.
    """
    if decision != "shadow_only":
        return {}, ""
    reasons = [str(r) for r in (gate.get("reasons") or []) if r]
    if any(r in _PLAYERS_CTX_REGRESSION_REASONS for r in reasons):
        return {}, ""
    fitted = cfg.get("settings") if isinstance(cfg.get("settings"), dict) else {}
    if not fitted:
        return {}, ""

    strength = float(fitted.get("ml_players_context_strength", 0.35) or 0.35)
    cap = float(fitted.get("ml_players_context_cap", 0.22) or 0.22)
    min_games = int(fitted.get("ml_players_context_min_games", 6) or 6)
    min_rel = float(fitted.get("ml_players_context_min_reliability", 0.28) or 0.28)

    # Data-quantity failure only → moderate reduction; gain-too-small → heavy reduction.
    if all((not r) or (r in _PLAYERS_CTX_DATA_REASONS) for r in reasons):
        mult = 0.60
        src_tag = "shadow_low_sample"
    else:
        mult = 0.35
        src_tag = "shadow_weak_signal"

    conservative = dict(fitted)
    conservative["ml_players_context_strength"] = float(strength * mult)
    conservative["ml_players_context_cap"] = float(cap * mult)
    conservative["ml_players_context_min_games"] = int(min(min_games + 2, 12))
    conservative["ml_players_context_min_reliability"] = float(min(min_rel + 0.06, 0.50))
    return conservative, f"conservative_default:{src_tag}"


def fit_players_context_for_league(
    players_context_cfg: Dict[str, Any] | None,
    league_hint: str,
    *,
    league_macro_key_fn: Callable[[str], str] | None = None,
) -> Tuple[Dict[str, Any], str]:
    lg = str(league_hint or "").strip()
    cfg = players_context_cfg if isinstance(players_context_cfg, dict) else {}
    if not cfg:
        return {}, ""

    gate = cfg.get("gate") if isinstance(cfg.get("gate"), dict) else {}
    decision = str(gate.get("decision") or "").strip().lower()
    if decision != "promote":
        fallback_cfg, fallback_src = _players_context_shadow_fallback(cfg, gate, decision)
        if fallback_cfg:
            return fallback_cfg, fallback_src
        return {}, f"gate:{decision or 'unknown'}"

    by_lg = cfg.get("promoted_settings_by_league") if isinstance(cfg.get("promoted_settings_by_league"), dict) else {}
    by_mk = cfg.get("promoted_settings_by_macro") if isinstance(cfg.get("promoted_settings_by_macro"), dict) else {}
    global_cfg = cfg.get("promoted_settings") if isinstance(cfg.get("promoted_settings"), dict) else {}

    if lg and isinstance(by_lg.get(lg), dict):
        return dict(by_lg.get(lg) or {}), f"league:{lg}"
    mk = _league_macro_key(lg, league_macro_key_fn)
    if mk and isinstance(by_mk.get(mk), dict):
        return dict(by_mk.get(mk) or {}), f"macro:{mk}"
    if global_cfg:
        return dict(global_cfg), "global"
    return {}, ""


def fit_fused_post_calibration_for_league(
    fused_post_cal_cfg: Dict[str, Any] | None,
    league_hint: str,
) -> Tuple[Dict[str, float], str]:
    lg = str(league_hint or "").strip()
    cfg = fused_post_cal_cfg if isinstance(fused_post_cal_cfg, dict) else {}
    if not cfg:
        return {}, ""

    gate = cfg.get("gate") if isinstance(cfg.get("gate"), dict) else {}
    gate_decision = str(gate.get("decision") or "").strip().lower()
    if gate_decision != "promote":
        return {}, f"gate:{gate_decision or 'unknown'}"

    runtime = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}
    use_league_override = bool(runtime.get("use_league_override", True))

    by_lg = cfg.get("promoted_model_by_league") if isinstance(cfg.get("promoted_model_by_league"), dict) else {}
    if use_league_override and lg and isinstance(by_lg.get(lg), dict):
        row = dict(by_lg.get(lg) or {})
        a = _safe_float(row.get("a", 1.0), 1.0)
        b = _safe_float(row.get("b", 0.0), 0.0)
        if math.isfinite(a) and math.isfinite(b):
            return {"a": a, "b": b}, f"league:{lg}"

    global_cfg = cfg.get("promoted_model_global") if isinstance(cfg.get("promoted_model_global"), dict) else {}
    if global_cfg:
        a = _safe_float(global_cfg.get("a", 1.0), 1.0)
        b = _safe_float(global_cfg.get("b", 0.0), 0.0)
        if math.isfinite(a) and math.isfinite(b):
            return {"a": a, "b": b}, "global"
    return {}, "promoted_model_missing"


def apply_fused_post_calibration(
    p_map_cal: float,
    fused_post_cal_cfg: Dict[str, Any] | None,
    *,
    league_hint: str,
    use_fitted: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    p_in = float(clip_prob(float(p_map_cal)))
    info: Dict[str, Any] = {
        "enabled": bool(use_fitted),
        "applied": False,
        "source": "",
        "reason": "",
        "league_hint": str(league_hint or ""),
        "p_in": p_in,
        "p_out": p_in,
    }
    if not bool(use_fitted):
        info["reason"] = "disabled_by_flag"
        return p_in, info
    if not math.isfinite(p_in):
        info["reason"] = "invalid_p_in"
        return float("nan"), info

    cfg = fused_post_cal_cfg if isinstance(fused_post_cal_cfg, dict) else {}
    if not cfg:
        info["reason"] = "missing_fit_cfg"
        return p_in, info
    info["version"] = str(cfg.get("version") or "")

    model, src = fit_fused_post_calibration_for_league(cfg, str(league_hint or ""))
    info["source"] = str(src or "")
    gate = cfg.get("gate") if isinstance(cfg.get("gate"), dict) else {}
    gate_decision = str(gate.get("decision") or "").strip().lower()
    info["gate_decision"] = gate_decision
    if not model:
        info["reason"] = ("gate_not_promoted" if gate_decision and gate_decision != "promote" else "model_not_available")
        return p_in, info

    runtime = cfg.get("runtime") if isinstance(cfg.get("runtime"), dict) else {}
    max_abs_logit_shift = abs(_safe_float(runtime.get("max_abs_logit_shift", 1.00), 1.00))
    max_abs_logit_shift = max(0.05, min(3.0, max_abs_logit_shift))

    try:
        a = _safe_float(model.get("a", 1.0), 1.0)
        b = _safe_float(model.get("b", 0.0), 0.0)
        z_in = float(math.log(p_in / max(1e-12, 1.0 - p_in)))
        z_raw = float(a * z_in + b)
        z_shift_raw = float(z_raw - z_in)
        z_shift = float(max(-max_abs_logit_shift, min(max_abs_logit_shift, z_shift_raw)))
        z_out = float(z_in + z_shift)
        p_out = float(clip_prob(1.0 / (1.0 + math.exp(-z_out))))
    except Exception:
        info["reason"] = "calibration_runtime_error"
        return p_in, info

    info.update(
        {
            "applied": True,
            "reason": "ok",
            "a": float(a),
            "b": float(b),
            "max_abs_logit_shift": float(max_abs_logit_shift),
            "logit_shift_raw": float(z_shift_raw),
            "logit_shift_applied": float(z_shift),
            "p_out": float(p_out),
            "delta_pp": float((p_out - p_in) * 100.0),
        }
    )
    return float(p_out), info
