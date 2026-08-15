"""Shadow ML layer orchestration extracted from plays_app."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping

from mlcore.league_context import (
    is_explicit_interregional_league_label as shared_is_explicit_interregional_league_label,
    league_macro_key,
)


def _ff_enabled(
    fn: Callable[..., Any] | None,
    key: str,
    default: bool,
) -> bool:
    if not callable(fn):
        return bool(default)
    try:
        return bool(fn(str(key), default=bool(default)))
    except TypeError:
        try:
            return bool(fn(str(key), bool(default)))
        except Exception:
            return bool(default)
    except Exception:
        return bool(default)


def _is_explicit_interregional_league(league: Any) -> bool:
    return bool(shared_is_explicit_interregional_league_label(league))


def run_ml_shadow_layers(
    *,
    p_map_used: Any,
    p_series_used: Any,
    bo: int,
    ml_info: Dict[str, Any] | None,
    trace_uncertainty_info: Dict[str, Any] | None,
    trace_n_team_eff: Any,
    trace_n_players_eff: Any,
    team_games_min_used: Any,
    trace_patch_guard_info: Dict[str, Any] | None,
    trace_hybrid_info: Dict[str, Any] | None,
    trace_form_info: Dict[str, Any] | None,
    trace_fusion_info: Dict[str, Any] | None,
    guard_triggered: bool,
    guard_dpp: Any,
    teamA: str,
    teamB: str,
    session_state: MutableMapping[str, Any] | Mapping[str, Any] | None,
    team_state_artifact_path: Path | None,
    safe_logit_fn: Callable[[float], float],
    safe_sigmoid_fn: Callable[[float], float],
    clip_prob_fn: Callable[[float], float],
    prob_win_series_fn: Callable[[float, int], float],
    odd_from_prob_fn: Callable[[float], float],
    feature_flag_enabled_fn: Callable[..., Any] | None,
    compute_state_shadow_alite_fn: Callable[..., Dict[str, Any]],
    compute_state_shadow_areal_fn: Callable[..., Dict[str, Any]],
    load_team_state_artifact_cached_fn: Callable[[Any], Any],
    file_sig_fn: Callable[[str], Any],
) -> Dict[str, Any]:
    ssw = session_state if isinstance(session_state, MutableMapping) else None
    ml = ml_info if isinstance(ml_info, dict) else {}
    tr_unc = trace_uncertainty_info if isinstance(trace_uncertainty_info, dict) else {}
    tr_patch = trace_patch_guard_info if isinstance(trace_patch_guard_info, dict) else {}
    tr_hybrid = trace_hybrid_info if isinstance(trace_hybrid_info, dict) else {}
    tr_form = trace_form_info if isinstance(trace_form_info, dict) else {}
    tr_fusion = trace_fusion_info if isinstance(trace_fusion_info, dict) else {}
    bo_i = int(bo)

    # Legacy shadow vNext disabled after A-lite promotion (kept only for fallback/debug).
    try:
        if isinstance(ssw, MutableMapping):
            ssw["ff_ml_shadow_vnext_trace"] = False
    except Exception:
        pass
    ff_shadow_vnext = False
    trace_shadow_ml_vnext: Dict[str, Any] = {
        "enabled": bool(ff_shadow_vnext),
        "applied": False,
        "reason": "",
        "p_map_old": (float(p_map_used) if math.isfinite(float(p_map_used)) else None),
        "p_series_old": (float(p_series_used) if math.isfinite(float(p_series_used)) else None),
        "p_map_new": None,
        "p_series_new": None,
        "diff_pp_map": None,
        "diff_pp_series": None,
    }

    lg_a_shadow = ""
    lg_b_shadow = ""
    try:
        p_shadow_old = float(p_map_used) if math.isfinite(float(p_map_used)) else float("nan")
        ps_shadow_old = float(p_series_used) if math.isfinite(float(p_series_used)) else float("nan")
        if bool(ff_shadow_vnext) and math.isfinite(p_shadow_old):
            z_old = float(safe_logit_fn(p_shadow_old))
            unc_n_eff = float((tr_unc or {}).get("n_eff", float("nan")) or float("nan"))
            unc_prior = float((tr_unc or {}).get("prior_games", 22.0) or 22.0)
            w_unc = 1.0
            if math.isfinite(unc_n_eff) and unc_n_eff >= 0.0:
                den_unc = max(1e-9, float(unc_n_eff) + max(1.0, float(unc_prior)))
                w_unc = max(0.20, min(1.0, float(unc_n_eff) / den_unc))

            n_team_shadow = float(trace_n_team_eff) if math.isfinite(float(trace_n_team_eff)) else float("nan")
            if not math.isfinite(n_team_shadow):
                try:
                    n_team_shadow = (
                        float(team_games_min_used)
                        if math.isfinite(float(team_games_min_used))
                        else float("nan")
                    )
                except Exception:
                    n_team_shadow = float("nan")
            n_players_shadow = (
                float(trace_n_players_eff)
                if math.isfinite(float(trace_n_players_eff))
                else float("nan")
            )
            ev_team = max(0.0, n_team_shadow) if math.isfinite(n_team_shadow) else 0.0
            ev_players = max(0.0, n_players_shadow) if math.isfinite(n_players_shadow) else 0.0
            ev_total = float(ev_team + 0.75 * ev_players)
            k_ev = 12.0
            w_evidence = max(0.35, min(1.0, ev_total / (ev_total + k_ev))) if ev_total > 0 else 0.35

            patch_status = str((tr_patch or {}).get("status", "") or "").strip().lower()
            patch_shrink = float((tr_patch or {}).get("shrink", 0.0) or 0.0)
            w_patch = 1.0
            if patch_status in {"refit_needed", "refit", "changed"}:
                w_patch = max(0.55, min(1.0, 1.0 - 0.70 * max(0.0, min(1.0, patch_shrink))))
            elif bool((tr_patch or {}).get("applied", False)):
                w_patch = max(0.60, min(1.0, 1.0 - 0.55 * max(0.0, min(1.0, patch_shrink))))

            agree_h = float((tr_hybrid or {}).get("agree_to_model", float("nan")) or float("nan"))
            w_hybrid_agree = 1.0
            if math.isfinite(agree_h):
                w_hybrid_agree = max(0.80, min(1.0, 0.80 + 0.20 * max(0.0, min(1.0, agree_h))))

            cons_drift_pp = float(guard_dpp) if math.isfinite(float(guard_dpp)) else float("nan")
            w_consistency = 1.0
            if bool(guard_triggered) and math.isfinite(cons_drift_pp):
                w_consistency = max(0.55, min(1.0, 1.0 - min(0.45, abs(cons_drift_pp) / 40.0)))
            p_form_shadow = float((tr_form or {}).get("p_form", float("nan")) or float("nan"))
            p_form_model = float((tr_form or {}).get("p_model", float("nan")) or float("nan"))
            n_form_a = float((tr_form or {}).get("n_a", float("nan")) or float("nan"))
            n_form_b = float((tr_form or {}).get("n_b", float("nan")) or float("nan"))
            n_form_min = min(n_form_a, n_form_b) if math.isfinite(n_form_a) and math.isfinite(n_form_b) else float("nan")
            w_form_used = float((tr_form or {}).get("w_form", 0.0) or 0.0)
            form_applied = bool((tr_form or {}).get("applied", False))
            roster_changed_shadow = bool((tr_fusion or {}).get("roster_changed", False))

            def _shadow_macro_key(lg: str) -> str:
                return str(league_macro_key(lg, unknown="OTHER"))

            lg_a_shadow = str((((ml or {}).get("blue") or {}).get("league")) or "").strip()
            lg_b_shadow = str((((ml or {}).get("red") or {}).get("league")) or "").strip()
            macro_a_shadow = _shadow_macro_key(lg_a_shadow)
            macro_b_shadow = _shadow_macro_key(lg_b_shadow)
            interregional_shadow = bool(
                (macro_a_shadow and macro_b_shadow and (macro_a_shadow != macro_b_shadow))
                or _is_explicit_interregional_league(lg_a_shadow)
                or _is_explicit_interregional_league(lg_b_shadow)
            )
            unknown_macro_shadow = bool(macro_a_shadow == "OTHER" or macro_b_shadow == "OTHER")
            recent_evidence = 0.0
            if math.isfinite(n_form_min) and n_form_min > 0:
                recent_evidence = max(0.0, min(1.0, float(n_form_min) / (float(n_form_min) + 6.0)))
            team_recent_trust = 1.0
            if ev_team > 0.0:
                team_recent_trust = float(max(0.55, min(1.0, math.sqrt(float(ev_team) / (float(ev_team) + 40.0)))))
            w_recent_dir = 0.0
            recent_delta_logit = 0.0
            recent_applied = False
            cn_extreme_favorite_damp = False
            if form_applied and math.isfinite(p_form_shadow):
                try:
                    z_form = float(safe_logit_fn(p_form_shadow))
                    z_form_base = float(safe_logit_fn(p_form_model)) if math.isfinite(p_form_model) else z_old
                    recent_delta_logit = float(z_form - z_form_base)
                    w_recent_dir = float(max(0.0, min(0.65, (0.05 + 0.60 * max(0.0, min(1.0, w_form_used))) * recent_evidence)))
                    w_recent_dir = float(w_recent_dir * team_recent_trust)
                    if roster_changed_shadow:
                        w_recent_dir = float(min(0.75, w_recent_dir * 1.15))
                    if patch_status in {"refit_needed", "refit", "changed"}:
                        w_recent_dir = float(min(0.80, w_recent_dir * 1.05))
                    if interregional_shadow:
                        w_recent_dir = float(min(0.28, w_recent_dir * 0.55))
                    elif unknown_macro_shadow:
                        w_recent_dir = float(min(0.20, w_recent_dir * 0.80))
                    elif macro_a_shadow != "OTHER" and macro_b_shadow != "OTHER":
                        w_recent_dir = float(min(0.42, w_recent_dir))
                    cn_extreme_favorite_damp = bool(
                        (macro_a_shadow == "CN")
                        and (macro_b_shadow == "CN")
                        and math.isfinite(p_shadow_old)
                        and (float(p_shadow_old) >= 0.80)
                        and (float(recent_delta_logit) < 0.0)
                    )
                    if cn_extreme_favorite_damp:
                        fav_excess = max(0.0, min(1.0, (float(p_shadow_old) - 0.80) / 0.15))
                        cn_damp = float(max(0.55, 0.85 - 0.20 * fav_excess))
                        w_recent_dir = float(min(0.24, w_recent_dir * cn_damp))
                    delta_cap = float(0.55 + 0.55 * recent_evidence)
                    if math.isfinite(n_form_min):
                        if float(n_form_min) < 6.0:
                            delta_cap *= 0.55
                        elif float(n_form_min) < 10.0:
                            delta_cap *= 0.75
                    if interregional_shadow:
                        delta_cap *= 0.60
                    elif unknown_macro_shadow:
                        delta_cap *= 0.80
                    elif cn_extreme_favorite_damp:
                        delta_cap *= 0.72
                    if roster_changed_shadow:
                        delta_cap *= 1.10
                    delta_cap = float(max(0.20, min(1.25, delta_cap)))
                    recent_delta_logit = float(max(-delta_cap, min(delta_cap, recent_delta_logit)))
                    recent_applied = abs(float(recent_delta_logit)) > 1e-9 and w_recent_dir > 0.0
                except Exception:
                    recent_delta_logit = 0.0
                    w_recent_dir = 0.0
                    recent_applied = False

            w_floor = 0.35
            if recent_evidence > 0.0:
                w_floor = float(min(0.78, w_floor + 0.20 * recent_evidence))
            if ev_team > 0.0:
                w_floor = float(min(0.82, w_floor + 0.18 * max(0.0, min(1.0, ev_team / (ev_team + 10.0)))))
            if unknown_macro_shadow and not interregional_shadow and ev_team > 0.0:
                w_floor = float(min(0.86, w_floor + 0.04 * max(0.0, min(1.0, ev_team / (ev_team + 25.0)))))
            w_final = float(max(w_floor, min(1.0, w_unc * w_evidence * w_patch * w_hybrid_agree * w_consistency)))
            z_new = float(z_old * w_final)
            if recent_applied:
                z_new = float(z_new + (w_recent_dir * recent_delta_logit))
                z_cap = 3.25
                z_new = float(max(-z_cap, min(z_cap, z_new)))
            p_shadow_new = float(clip_prob_fn(safe_sigmoid_fn(z_new)))
            try:
                ps_shadow_new = float(prob_win_series_fn(float(p_shadow_new), int(bo_i)))
            except Exception:
                ps_shadow_new = float("nan")

            trace_shadow_ml_vnext = {
                "enabled": True,
                "applied": True,
                "method": "logit_shrink_plus_recent_shadow_v2",
                "reason": "uncertainty+evidence+patch+hybrid_agree+consistency+recent_form_direction",
                "bo": int(bo_i),
                "league_hint": str((tr_patch or {}).get("league_hint", "") or ""),
                "weights": {
                    "uncertainty": float(w_unc),
                    "evidence": float(w_evidence),
                    "patch": float(w_patch),
                    "hybrid_agree": float(w_hybrid_agree),
                    "consistency": float(w_consistency),
                    "floor": float(w_floor),
                    "final": float(w_final),
                    "recent_direction": float(w_recent_dir),
                },
                "inputs": {
                    "n_eff_uncertainty": (float(unc_n_eff) if math.isfinite(unc_n_eff) else None),
                    "prior_games_uncertainty": float(unc_prior),
                    "n_team_eff": (float(n_team_shadow) if math.isfinite(n_team_shadow) else None),
                    "n_players_eff": (float(n_players_shadow) if math.isfinite(n_players_shadow) else None),
                    "league_a": str(lg_a_shadow),
                    "league_b": str(lg_b_shadow),
                    "macro_a": str(macro_a_shadow),
                    "macro_b": str(macro_b_shadow),
                    "interregional": bool(interregional_shadow),
                    "unknown_macro": bool(unknown_macro_shadow),
                    "cn_extreme_favorite_damp": bool(cn_extreme_favorite_damp),
                    "season_form_applied": bool(form_applied),
                    "season_form_w": (float(w_form_used) if math.isfinite(w_form_used) else None),
                    "season_form_n_min": (float(n_form_min) if math.isfinite(n_form_min) else None),
                    "season_form_p_form": (float(p_form_shadow) if math.isfinite(p_form_shadow) else None),
                    "season_form_p_model": (float(p_form_model) if math.isfinite(p_form_model) else None),
                    "recent_evidence": float(recent_evidence),
                    "team_recent_trust": float(team_recent_trust),
                    "recent_delta_logit": float(recent_delta_logit),
                    "roster_changed": bool(roster_changed_shadow),
                    "patch_status": str(patch_status),
                    "patch_shrink": (float(patch_shrink) if math.isfinite(patch_shrink) else None),
                    "hybrid_agree_to_model": (float(agree_h) if math.isfinite(agree_h) else None),
                    "consistency_guard_triggered": bool(guard_triggered),
                    "consistency_guard_drift_pp": (float(cons_drift_pp) if math.isfinite(cons_drift_pp) else None),
                },
                "p_map_old": float(p_shadow_old),
                "p_map_new": float(p_shadow_new),
                "p_series_old": (float(ps_shadow_old) if math.isfinite(ps_shadow_old) else None),
                "p_series_new": (float(ps_shadow_new) if math.isfinite(ps_shadow_new) else None),
                "diff_pp_map": float(100.0 * (float(p_shadow_new) - float(p_shadow_old))),
                "diff_pp_series": (
                    float(100.0 * (float(ps_shadow_new) - float(ps_shadow_old)))
                    if math.isfinite(ps_shadow_old) and math.isfinite(ps_shadow_new)
                    else None
                ),
                "odds": {
                    "teamA_map_old": float(odd_from_prob_fn(float(p_shadow_old))),
                    "teamA_map_new": float(odd_from_prob_fn(float(p_shadow_new))),
                    "teamA_series_old": (
                        float(odd_from_prob_fn(float(ps_shadow_old)))
                        if math.isfinite(ps_shadow_old)
                        else None
                    ),
                    "teamA_series_new": (
                        float(odd_from_prob_fn(float(ps_shadow_new)))
                        if math.isfinite(ps_shadow_new)
                        else None
                    ),
                },
            }
        elif not bool(ff_shadow_vnext):
            trace_shadow_ml_vnext["reason"] = "disabled_by_flag"
        else:
            trace_shadow_ml_vnext["reason"] = "invalid_live_probability"
    except Exception as e_ml_shadow:
        trace_shadow_ml_vnext = {
            "enabled": bool(ff_shadow_vnext),
            "applied": False,
            "reason": f"error:{type(e_ml_shadow).__name__}",
            "error": str(e_ml_shadow),
            "p_map_old": (float(p_map_used) if math.isfinite(float(p_map_used)) else None),
            "p_series_old": (float(p_series_used) if math.isfinite(float(p_series_used)) else None),
            "p_map_new": None,
            "p_series_new": None,
            "diff_pp_map": None,
            "diff_pp_series": None,
        }

    ff_shadow_state_alite = _ff_enabled(feature_flag_enabled_fn, "ff_ml_shadow_state_alite_trace", True)
    trace_shadow_state_alite: Dict[str, Any] = {
        "enabled": bool(ff_shadow_state_alite),
        "applied": False,
        "reason": "disabled_by_flag" if not bool(ff_shadow_state_alite) else "",
        "p_map_old": (float(p_map_used) if math.isfinite(float(p_map_used)) else None),
        "p_series_old": (float(p_series_used) if math.isfinite(float(p_series_used)) else None),
        "p_map_new": None,
        "p_series_new": None,
        "diff_pp_map": None,
        "diff_pp_series": None,
    }
    try:
        if bool(ff_shadow_state_alite):
            trace_shadow_state_alite = compute_state_shadow_alite_fn(
                p_map_old=(float(p_map_used) if math.isfinite(float(p_map_used)) else None),
                p_series_old=(float(p_series_used) if math.isfinite(float(p_series_used)) else None),
                bo=int(bo_i),
                ml_info=(ml if isinstance(ml, dict) else None),
                trace_form_info=(tr_form if isinstance(tr_form, dict) else None),
                trace_fusion_info=(tr_fusion if isinstance(tr_fusion, dict) else None),
                trace_patch_guard_info=(tr_patch if isinstance(tr_patch, dict) else None),
                trace_uncertainty_info=(tr_unc if isinstance(tr_unc, dict) else None),
                n_team_eff=(float(trace_n_team_eff) if math.isfinite(float(trace_n_team_eff)) else None),
                n_players_eff=(float(trace_n_players_eff) if math.isfinite(float(trace_n_players_eff)) else None),
                consistency_guard_triggered=bool(guard_triggered),
                consistency_guard_drift_pp=(float(guard_dpp) if math.isfinite(float(guard_dpp)) else None),
            )
            if isinstance(trace_shadow_state_alite, dict):
                trace_shadow_state_alite.setdefault("enabled", True)
        else:
            trace_shadow_state_alite["reason"] = "disabled_by_flag"
    except Exception as e_state_shadow:
        trace_shadow_state_alite = {
            "enabled": bool(ff_shadow_state_alite),
            "applied": False,
            "reason": f"error:{type(e_state_shadow).__name__}",
            "error": str(e_state_shadow),
            "p_map_old": (float(p_map_used) if math.isfinite(float(p_map_used)) else None),
            "p_series_old": (float(p_series_used) if math.isfinite(float(p_series_used)) else None),
            "p_map_new": None,
            "p_series_new": None,
            "diff_pp_map": None,
            "diff_pp_series": None,
        }

    ff_shadow_state_areal = _ff_enabled(feature_flag_enabled_fn, "ff_ml_shadow_state_areal_trace", True)
    trace_shadow_state_areal: Dict[str, Any] = {
        "enabled": bool(ff_shadow_state_areal),
        "applied": False,
        "reason": "disabled_by_flag" if not bool(ff_shadow_state_areal) else "",
        "p_map_old": (float(p_map_used) if math.isfinite(float(p_map_used)) else None),
        "p_series_old": (float(p_series_used) if math.isfinite(float(p_series_used)) else None),
        "p_map_new": None,
        "p_series_new": None,
        "diff_pp_map": None,
        "diff_pp_series": None,
    }
    try:
        if bool(ff_shadow_state_areal):
            state_art_live = (
                load_team_state_artifact_cached_fn(file_sig_fn(str(team_state_artifact_path)))
                if isinstance(team_state_artifact_path, Path) and team_state_artifact_path.exists()
                else {}
            )
            mlb = (ml.get("blue") or {}) if isinstance(ml, dict) and isinstance(ml.get("blue"), dict) else {}
            mlr = (ml.get("red") or {}) if isinstance(ml, dict) and isinstance(ml.get("red"), dict) else {}
            tm_b_areal = str(mlb.get("team") or teamA or "").strip()
            tm_r_areal = str(mlr.get("team") or teamB or "").strip()
            lg_b_areal = str(mlb.get("league") or "").strip()
            lg_r_areal = str(mlr.get("league") or "").strip()
            trace_shadow_state_areal = compute_state_shadow_areal_fn(
                p_map_old=(float(p_map_used) if math.isfinite(float(p_map_used)) else None),
                p_series_old=(float(p_series_used) if math.isfinite(float(p_series_used)) else None),
                bo=int(bo_i),
                team_blue=tm_b_areal,
                team_red=tm_r_areal,
                league_blue=lg_b_areal,
                league_red=lg_r_areal,
                state_artifact=(state_art_live if isinstance(state_art_live, dict) else None),
                trace_fusion_info=(tr_fusion if isinstance(tr_fusion, dict) else None),
                trace_patch_guard_info=(tr_patch if isinstance(tr_patch, dict) else None),
                trace_uncertainty_info=(tr_unc if isinstance(tr_unc, dict) else None),
                n_team_eff=(float(trace_n_team_eff) if math.isfinite(float(trace_n_team_eff)) else None),
                n_players_eff=(float(trace_n_players_eff) if math.isfinite(float(trace_n_players_eff)) else None),
                consistency_guard_triggered=bool(guard_triggered),
                consistency_guard_drift_pp=(float(guard_dpp) if math.isfinite(float(guard_dpp)) else None),
            )
            if isinstance(trace_shadow_state_areal, dict):
                trace_shadow_state_areal.setdefault("enabled", True)
        else:
            trace_shadow_state_areal["reason"] = "disabled_by_flag"
    except Exception as e_state_areal_shadow:
        trace_shadow_state_areal = {
            "enabled": bool(ff_shadow_state_areal),
            "applied": False,
            "reason": f"error:{type(e_state_areal_shadow).__name__}",
            "error": str(e_state_areal_shadow),
            "p_map_old": (float(p_map_used) if math.isfinite(float(p_map_used)) else None),
            "p_series_old": (float(p_series_used) if math.isfinite(float(p_series_used)) else None),
            "p_map_new": None,
            "p_series_new": None,
            "diff_pp_map": None,
            "diff_pp_series": None,
        }

    return {
        "_ff_ml_shadow_vnext": bool(ff_shadow_vnext),
        "trace_shadow_ml_vnext": trace_shadow_ml_vnext,
        "_ff_ml_shadow_state_alite": bool(ff_shadow_state_alite),
        "trace_shadow_state_alite": trace_shadow_state_alite,
        "_ff_ml_shadow_state_areal": bool(ff_shadow_state_areal),
        "trace_shadow_state_areal": trace_shadow_state_areal,
        "_lg_a_shadow": str(lg_a_shadow or ""),
        "_lg_b_shadow": str(lg_b_shadow or ""),
    }
