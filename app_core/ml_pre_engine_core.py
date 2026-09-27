"""Pre-live uncertainty and engine/lineup orchestration."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, MutableMapping

from app_core.pipeline_trace_core import append_pipeline_step
from app_core.players_context_core import compute_players_context_quality_signal
from app_core.prob_math_core import safe_float as _f
from mlcore.league_context import (
    is_explicit_interregional_league_label as shared_is_explicit_interregional_league_label,
    league_macro_key,
)


def _fmt_num(v: Any, digits: int, fn: Callable[[Any, int], str] | None = None) -> str:
    if callable(fn):
        try:
            return str(fn(v, int(digits)))
        except Exception:
            pass
    x = _f(v, float("nan"))
    if not math.isfinite(x):
        return "-"
    return f"{x:.{int(digits)}f}"


def _fmt_pct(v: Any, fn: Callable[[Any], str] | None = None) -> str:
    if callable(fn):
        try:
            return str(fn(v))
        except Exception:
            pass
    x = _f(v, float("nan"))
    if not math.isfinite(x):
        return "-"
    return f"{100.0 * x:.1f}%"


def _safe_caption(caption_fn: Callable[[str], None] | None, text: str) -> None:
    if not callable(caption_fn):
        return
    try:
        caption_fn(str(text))
    except Exception:
        pass


def _safe_warning(warning_fn: Callable[[str], None] | None, text: str) -> None:
    if not callable(warning_fn):
        return
    try:
        warning_fn(str(text))
    except Exception:
        pass


def _is_explicit_interregional_league_label(v: Any) -> bool:
    return bool(shared_is_explicit_interregional_league_label(v))


def _is_interregional_from_ml(ml_info: Dict[str, Any]) -> bool:
    blue = str(((ml_info or {}).get("blue") or {}).get("league") or "").strip().upper()
    red = str(((ml_info or {}).get("red") or {}).get("league") or "").strip().upper()
    if bool(blue and red and blue != red):
        return True
    return bool(_is_explicit_interregional_league_label(blue) or _is_explicit_interregional_league_label(red))


def _players_reliability_profile(
    *,
    cov_ratio: Any,
    freshness_avg: Any,
    lineup_continuity: Any,
    avg_gp_min: Any,
    prior_games: Any,
    interregional: bool,
    min_rel: Any = 0.35,
    max_rel: Any = 1.00,
    inter_mult: Any = 0.92,
) -> Dict[str, float]:
    cov = max(0.0, min(1.0, _f(cov_ratio, 0.0)))
    fresh = max(0.0, min(1.0, _f(freshness_avg, 0.0)))
    cont = max(0.0, min(1.0, _f(lineup_continuity, 0.0)))
    gp = max(0.0, _f(avg_gp_min, 0.0))
    pg = max(1.0, _f(prior_games, 22.0))
    evid = gp / (gp + pg)

    # Weighted reliability score in [0,1].
    base = 0.12 + (0.40 * cov) + (0.20 * fresh) + (0.28 * cont)
    base = max(0.0, min(1.0, base))
    rel = base * (0.70 + 0.30 * evid)
    if bool(interregional):
        rel *= max(0.70, min(1.0, _f(inter_mult, 0.92)))
    rel = max(max(0.0, _f(min_rel, 0.35)), min(max(0.0, _f(max_rel, 1.0)), rel))
    rel = min(max(0.0, rel), 1.0)

    return {
        "reliability": float(rel),
        "delta_mult": float(rel),
        "n_players_mult": float(rel),
        "coverage": float(cov),
        "freshness": float(fresh),
        "continuity": float(cont),
        "evidence": float(evid),
        "interregional": float(1.0 if interregional else 0.0),
    }


def _safe_prob_win_series(prob_win_series_fn: Callable[[float, int], float], p_map: float, bo: int) -> float:
    try:
        return float(prob_win_series_fn(float(p_map), int(bo)))
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _compute_players_context_info(
    *,
    team_games: Any,
    team_a: str,
    team_b: str,
    filters: Any,
    as_of_date: Any,
    session_state: Mapping[str, Any] | None,
    ml_info: Dict[str, Any] | None,
    league_hint_a_fallback: str = "",
    league_hint_b_fallback: str = "",
) -> Dict[str, Any]:
    ss = session_state if isinstance(session_state, Mapping) else {}
    ml = ml_info if isinstance(ml_info, dict) else {}
    ctx_info = compute_players_context_quality_signal(
        team_games=team_games,
        team_a=str(team_a),
        team_b=str(team_b),
        filters=filters,
        as_of_date=as_of_date,
        enabled=bool(ss.get("ml_players_context_enabled", True)),
        lookback_games=int(ss.get("ml_players_context_lookback_games", 24) or 24),
        half_life_days=float(ss.get("ml_players_context_half_life_days", 45.0) or 45.0),
        min_games=int(ss.get("ml_players_context_min_games", 6) or 6),
        strength=float(ss.get("ml_players_context_strength", 0.35) or 0.35),
        cap=float(ss.get("ml_players_context_cap", 0.22) or 0.22),
        min_reliability=float(ss.get("ml_players_context_min_reliability", 0.28) or 0.28),
        interregional=_is_interregional_from_ml(ml),
        interregional_mult=float(ss.get("ml_players_context_inter_mult", 0.88) or 0.88),
        league_hint_a=str((ml.get("blue") or {}).get("league") or league_hint_a_fallback or ""),
        league_hint_b=str((ml.get("red") or {}).get("league") or league_hint_b_fallback or ""),
    )
    return dict(ctx_info) if isinstance(ctx_info, dict) else {}


def run_ml_pre_engine_layers(
    *,
    p_map_used: Any,
    p_series_used: Any,
    bo: int,
    ml_info: Dict[str, Any] | None,
    lgA: str,
    lineup_mode: str,
    pl_enabled: bool,
    pl_players_path: Any,
    lineupA_sel: Dict[str, Any] | None,
    lineupB_sel: Dict[str, Any] | None,
    as_of_date: Any,
    ml_engine: str,
    use_lineup: bool,
    artifact_path: Any,
    pl_w: Any,
    playoffs_opt: Any,
    teamA: str,
    teamB: str,
    team_aliases_a: List[str] | None = None,
    team_aliases_b: List[str] | None = None,
    teamA_key: str,
    teamB_key: str,
    team_games: Any,
    csv_path: Any,
    filters: Any,
    diff_used: Any,
    delta_players: Any,
    delta_players_eff: Any,
    trace_p_map_team: Any,
    trace_p_map_players: Any,
    trace_p_map_fused: Any,
    trace_n_team_eff: Any,
    trace_n_players_eff: Any,
    trace_fusion_info: Dict[str, Any] | None,
    trace_uncertainty_info: Dict[str, Any] | None,
    league_transfer_features: Dict[str, Any] | None = None,
    p_pipe_prev: Any,
    session_state: MutableMapping[str, Any] | Mapping[str, Any] | None,
    trace_pipeline_steps: List[Dict[str, Any]] | None,
    league_is_excluded_fn: Callable[[str, Any], bool],
    apply_ml_uncertainty_adjust_fn: Callable[..., Any],
    prob_win_series_fn: Callable[[float, int], float],
    load_players_artifact_cached_fn: Callable[[Any], Any],
    file_sig_fn: Callable[[str], Any],
    fit_players_lane_for_league_fn: Callable[[str], Any],
    fit_players_league_weight_for_league_fn: Callable[[str], Any],
    compute_delta_and_confidence_fn: Callable[..., Dict[str, Any]],
    delta_players_effect_fn: Callable[..., float],
    transfer_signal_from_details_fn: Callable[[Any], float],
    lineup_continuity_from_details_fn: Callable[..., float],
    calibrate_p_from_artifact_fn: Callable[[Any, float], float],
    fuse_probs_by_precision_fn: Callable[..., Dict[str, Any]],
    score_team_fn: Callable[..., Any],
    p_from_diff_points_fn: Callable[..., float],
    elo_season_players_p_map_fn: Callable[..., Dict[str, Any]],
    format_num_fn: Callable[[Any, int], str] | None = None,
    format_pct_fn: Callable[[Any], str] | None = None,
    caption_fn: Callable[[str], None] | None = None,
    warning_fn: Callable[[str], None] | None = None,
    match_league: str = "",
    apply_confidence_governor_fn: Callable[..., Dict[str, Any]] | None = None,
    # Post-fusion calibration callbacks (optional; enabled via ml_postfusion_cal_enabled flag).
    # Pass these to enable Microseg + Platt re-calibration on p_fused AFTER team+players fusion.
    # Ablation showed +0.97pp logloss + ECE -48% (tools/run_ablation_cal_position.py).
    apply_microseg_fn: Callable[..., Any] | None = None,
    apply_fused_postcal_fn: Callable[..., Any] | None = None,
    postfusion_microseg_table: Mapping[str, Any] | None = None,
    # Calibrador com grupo (05/09/2026, divida D8). Unico estagio pos-fusao
    # que o app roda; ms2 e Platt da fusao seguem desligados por medicao.
    postfusion_grupo_coef: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    ss = session_state if isinstance(session_state, Mapping) else {}
    ssw = session_state if isinstance(session_state, MutableMapping) else None
    steps = trace_pipeline_steps if isinstance(trace_pipeline_steps, list) else []
    ml = ml_info if isinstance(ml_info, dict) else {}
    blue_league_out = str((ml.get("blue") or {}).get("league") or "").strip()
    red_league_out = str((ml.get("red") or {}).get("league") or "").strip()

    p_map = _f(p_map_used, float("nan"))
    # Marca se o calibrador COM GRUPO ja foi aplicado num dos ramos de
    # escalacao. Ver a aplicacao de reserva no fim (08/09/2026).
    _pfc_grupo_aplicado = False
    p_series = _f(p_series_used, float("nan"))
    bo_i = int(bo)
    diff_cur = _f(diff_used, float("nan"))
    delta_cur = _f(delta_players, 0.0)
    delta_eff_cur = _f(delta_players_eff, 0.0)
    p_team = _f(trace_p_map_team, p_map)
    p_players = _f(trace_p_map_players, float("nan"))
    p_fused = _f(trace_p_map_fused, p_map)
    n_team_eff = _f(trace_n_team_eff, float("nan"))
    n_players_eff = _f(trace_n_players_eff, float("nan"))
    fusion_info = dict(trace_fusion_info) if isinstance(trace_fusion_info, dict) else {}
    uncertainty_info = dict(trace_uncertainty_info) if isinstance(trace_uncertainty_info, dict) else {}
    p_prev = _f(p_pipe_prev, p_map)
    gov_enabled = bool(ss.get("ml_confidence_governor_enabled", True))
    gov_budget_pp = float(
        ss.get(
            "_ml_confidence_governor_budget_runtime_pp",
            ss.get("ml_confidence_governor_max_total_shrink_pp", 8.0),
        )
        or 8.0
    )

    lineup_mode_out = str(lineup_mode or "")
    match_league_out = str(match_league or "")
    if blue_league_out and red_league_out and blue_league_out != red_league_out:
        match_league_out = ""
    if not match_league_out:
        if blue_league_out and blue_league_out == red_league_out:
            match_league_out = str(blue_league_out)
        elif blue_league_out and not red_league_out:
            match_league_out = str(blue_league_out)
        elif red_league_out and not blue_league_out:
            match_league_out = str(red_league_out)
    elo_info: Dict[str, Any] | None = None
    players_context_info_out: Dict[str, Any] = {}

    if callable(apply_confidence_governor_fn):
        try:
            apply_confidence_governor_fn(
                stage="pipeline_start",
                p_before=p_map,
                p_candidate=p_map,
                session_state=ssw,
                enabled=bool(gov_enabled),
                max_total_shrink_pp=float(gov_budget_pp),
                reset=True,
            )
        except Exception:
            pass

    # Uncertainty layer.
    try:
        p_before_unc = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        p_after_unc = float(p_before_unc)
        gov_unc: Dict[str, Any] = {}
        if math.isfinite(float(p_map)):
            unc_league = str(lgA or ((ml.get("blue") or {}).get("league") or "")).strip()
            unc_excl_raw = ss.get("ml_uncertainty_exclude_leagues", "")
            unc_skip_by_league = bool(league_is_excluded_fn(unc_league, unc_excl_raw))
            unc_strength = float(ss.get("ml_uncertainty_strength", 0.55) or 0.55)
            unc_prior_games = float(ss.get("ml_uncertainty_prior_games", 22.0) or 22.0)
            unc_max_shrink = float(ss.get("ml_uncertainty_max_shrink", 0.65) or 0.65)
            unc_interregional = bool(_is_interregional_from_ml(ml))
            unc_damp = {
                "interregional": bool(unc_interregional),
                "strength_mult": 1.0,
                "prior_mult": 1.0,
                "cap_mult": 1.0,
            }
            if bool(unc_interregional):
                strength_mult = float(ss.get("ml_interregional_uncertainty_strength_mult", 0.30) or 0.30)
                prior_mult = float(ss.get("ml_interregional_uncertainty_prior_mult", 0.60) or 0.60)
                cap_mult = float(ss.get("ml_interregional_uncertainty_cap_mult", 0.40) or 0.40)
                unc_strength *= max(0.0, min(1.0, strength_mult))
                unc_prior_games *= max(0.10, prior_mult)
                unc_max_shrink *= max(0.0, min(1.0, cap_mult))
                unc_damp["strength_mult"] = float(max(0.0, min(1.0, strength_mult)))
                unc_damp["prior_mult"] = float(max(0.10, prior_mult))
                unc_damp["cap_mult"] = float(max(0.0, min(1.0, cap_mult)))
            p_unc, unc_i = apply_ml_uncertainty_adjust_fn(
                float(p_map),
                (ml if isinstance(ml, dict) else None),
                enabled=(bool(ss.get("ml_uncertainty_enabled", True)) and (not bool(unc_skip_by_league))),
                strength=float(unc_strength),
                prior_games=float(unc_prior_games),
                max_shrink=float(unc_max_shrink),
            )
            if math.isfinite(_f(p_unc)):
                if callable(apply_confidence_governor_fn):
                    try:
                        gov_unc = apply_confidence_governor_fn(
                            stage="uncertainty_layer",
                            p_before=p_before_unc,
                            p_candidate=float(p_unc),
                            session_state=ssw,
                            enabled=bool(gov_enabled),
                            max_total_shrink_pp=float(gov_budget_pp),
                            reset=False,
                        )
                    except Exception:
                        gov_unc = {}
                p_unc_used = (
                    float(gov_unc.get("p_out"))
                    if isinstance(gov_unc, dict) and math.isfinite(_f(gov_unc.get("p_out")))
                    else float(p_unc)
                )
                p_map = float(p_unc)
                p_map = float(p_unc_used)
                p_after_unc = float(p_map)
                p_series = _safe_prob_win_series(prob_win_series_fn, float(p_map), int(bo_i))
            if isinstance(unc_i, dict):
                unc_i["league"] = str(unc_league or "")
                unc_i["exclude_leagues"] = str(unc_excl_raw or "")
                unc_i["skipped_by_league"] = bool(unc_skip_by_league)
                unc_i["interregional_damp"] = dict(unc_damp)
                if isinstance(gov_unc, dict) and gov_unc:
                    unc_i["governor"] = {
                        "enabled": bool(gov_unc.get("enabled", False)),
                        "limited": bool(gov_unc.get("limited", False)),
                        "requested_shrink_pp": gov_unc.get("requested_shrink_pp"),
                        "applied_shrink_pp": gov_unc.get("applied_shrink_pp"),
                        "used_shrink_pp": gov_unc.get("used_shrink_pp"),
                        "remaining_shrink_pp": gov_unc.get("remaining_shrink_pp"),
                    }
                uncertainty_info = dict(unc_i)
                if isinstance(ssw, MutableMapping):
                    ssw["_ml_uncertainty_info"] = dict(unc_i)
                if bool(unc_i.get("applied", False)):
                    _safe_caption(
                        caption_fn,
                        (
                            f"Incerteza ML: n_eff={_fmt_num(float(unc_i.get('n_eff', 0.0) or 0.0),1,format_num_fn)} | "
                            f"shrink={_fmt_num(100.0 * float(unc_i.get('shrink', 0.0) or 0.0),1,format_num_fn)}% | "
                            f"p_final={_fmt_num(float(unc_i.get('p_out', p_map) or p_map),4,format_num_fn)}"
                        ),
                    )
                elif bool(unc_skip_by_league):
                    _safe_caption(
                        caption_fn,
                        f"Incerteza ML: desativada para liga {unc_league or '-'!s} (regra por liga).",
                    )
        ap_unc = bool(uncertainty_info.get("applied", False)) or (
            math.isfinite(float(p_before_unc))
            and math.isfinite(float(p_after_unc))
            and abs(float(p_after_unc) - float(p_before_unc)) > 1e-12
        )
        append_pipeline_step(
            steps=steps,
            stage="uncertainty_layer",
            before=p_before_unc,
            after=p_after_unc,
            applied=bool(ap_unc),
            enabled=bool(uncertainty_info.get("enabled", False)),
            skipped_by_league=bool(uncertainty_info.get("skipped_by_league", False)),
            league=str(uncertainty_info.get("league", "") or ""),
            governor_limited=bool(((uncertainty_info or {}).get("governor") or {}).get("limited", False)),
            governor_applied_shrink_pp=(
                float(((uncertainty_info or {}).get("governor") or {}).get("applied_shrink_pp"))
                if ((uncertainty_info or {}).get("governor") or {}).get("applied_shrink_pp") is not None
                else None
            ),
            n_eff=(
                float(uncertainty_info.get("n_eff"))
                if uncertainty_info.get("n_eff") is not None
                else None
            ),
            shrink=(
                float(uncertainty_info.get("shrink"))
                if uncertainty_info.get("shrink") is not None
                else None
            ),
        )
        p_prev = float(p_after_unc) if math.isfinite(float(p_after_unc)) else p_prev
    except Exception as exc:
        append_pipeline_step(
            steps=steps,
            stage="uncertainty_layer",
            before=p_prev,
            after=p_map,
            applied=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    delta_mode = str(ss.get("lineup_delta_mode", "saturado") or "saturado")
    delta_cap = float(ss.get("lineup_delta_cap", 1.6) or 1.6)
    delta_slope = float(ss.get("lineup_delta_slope", 2.4) or 2.4)

    # Players layer.
    if lineup_mode_out == "ML players layer" and bool(pl_enabled):
        try:
            p_art = load_players_artifact_cached_fn(file_sig_fn(str(pl_players_path)))
            lane_w = {
                "Top": float(ss.get("ml_players_lane_w_top", 1.0) or 1.0),
                "Jungle": float(ss.get("ml_players_lane_w_jungle", 1.0) or 1.0),
                "Mid": float(ss.get("ml_players_lane_w_mid", 1.0) or 1.0),
                "Bot": float(ss.get("ml_players_lane_w_adc", 1.0) or 1.0),
                "Support": float(ss.get("ml_players_lane_w_support", 1.0) or 1.0),
                "ADC": float(ss.get("ml_players_lane_w_adc", 1.0) or 1.0),
            }
            fit_lane_w, fit_lane_src = fit_players_lane_for_league_fn(match_league_out)
            if isinstance(fit_lane_w, dict) and fit_lane_w:
                for lk in ["Top", "Jungle", "Mid", "Bot", "Support"]:
                    try:
                        lane_w[lk] = float(lane_w.get(lk, 1.0) or 1.0) * float(fit_lane_w.get(lk, 1.0) or 1.0)
                    except (TypeError, ValueError):
                        pass
                try:
                    lane_w["ADC"] = float(lane_w.get("Bot", lane_w.get("ADC", 1.0)) or 1.0)
                except (TypeError, ValueError):
                    pass
            league_w = {
                "LPL": float(ss.get("ml_players_lg_w_lpl", 1.0) or 1.0),
                "CN": float(ss.get("ml_players_lg_w_lpl", 1.0) or 1.0),
                "LCK": float(ss.get("ml_players_lg_w_lck", 1.0) or 1.0),
                "KR": float(ss.get("ml_players_lg_w_lck", 1.0) or 1.0),
                "EMEA": float(ss.get("ml_players_lg_w_emea", 1.0) or 1.0),
                "NA": float(ss.get("ml_players_lg_w_na", 1.0) or 1.0),
                "BR": float(ss.get("ml_players_lg_w_br", 1.0) or 1.0),
                "APAC": float(ss.get("ml_players_lg_w_apac", 1.0) or 1.0),
                "OTHER": float(ss.get("ml_players_lg_w_other", 1.0) or 1.0),
                "DEFAULT": 1.0,
            }
            fit_lg_w, fit_lg_src = fit_players_league_weight_for_league_fn(match_league_out)
            if math.isfinite(_f(fit_lg_w)) and float(fit_lg_w) > 0:
                lg_u = str(match_league_out or "").strip().upper()
                if lg_u:
                    league_w[lg_u] = float(league_w.get(lg_u, 1.0) or 1.0) * float(fit_lg_w)
                mk = str(league_macro_key(lg_u, unknown=""))
                if mk:
                    league_w[mk] = float(league_w.get(mk, 1.0) or 1.0) * float(fit_lg_w)
            info = compute_delta_and_confidence_fn(
                p_art,
                blue_lineup=lineupA_sel,
                red_lineup=lineupB_sel,
                as_of=as_of_date,
                match_league=match_league_out,
                blue_league=blue_league_out,
                red_league=red_league_out,
                require_both_sides=False,
                lane_weights=lane_w,
                league_weights=league_w,
                transfer_payload=(league_transfer_features if isinstance(league_transfer_features, dict) else None),
                team_a=str(teamA),
                team_b=str(teamB),
                team_aliases_a=list(team_aliases_a or []),
                team_aliases_b=list(team_aliases_b or []),
            )
            delta_cur = float(info.get("delta", 0.0) or 0.0)
            delta_eff_cur = float(
                delta_players_effect_fn(delta_cur, mode=delta_mode, cap=delta_cap, slope=delta_slope)
            )
            cov = float(info.get("coverage", 0.0) or 0.0)
            cov_ratio = float(cov / max(float(info.get("coverage_total", 5) or 5), 1.0))
            freshness_avg = float(info.get("freshness_avg", 0.0) or 0.0)
            transfer_sig = float(transfer_signal_from_details_fn(info.get("details") or []))
            lineup_cont = float(
                lineup_continuity_from_details_fn(
                    info.get("details") or [],
                    freshness_avg=float(freshness_avg),
                    cov_ratio=float(cov_ratio),
                )
            )
            avg_gp_min_info = float(info.get("avg_gp_min", 0.0) or 0.0)
            base_thr = float(ss.get("fusion_roster_cont_thresh_base", 0.55) or 0.55)
            early_boost = float(ss.get("fusion_roster_cont_thresh_early_boost", 0.10) or 0.10)
            fresh_cut = float(ss.get("fusion_roster_low_freshness_cut", 0.60) or 0.60)
            fresh_boost = float(ss.get("fusion_roster_low_freshness_boost", 0.05) or 0.05)
            n_team_hint = float(
                min(
                    float(((ml.get("blue") or {}).get("games_played", 0) or 0)),
                    float(((ml.get("red") or {}).get("games_played", 0) or 0)),
                )
            )
            knee = max(1.0, float(ss.get("fusion_season_knee_games", 18.0) or 18.0))
            thr = float(base_thr)
            if math.isfinite(n_team_hint) and n_team_hint < knee:
                thr += float(early_boost) * max(0.0, 1.0 - (n_team_hint / knee))
            if float(fresh_cut) > 0 and math.isfinite(float(freshness_avg)) and float(freshness_avg) < float(fresh_cut):
                thr += float(fresh_boost) * max(0.0, 1.0 - (float(freshness_avg) / float(fresh_cut)))
            thr = float(max(0.30, min(0.90, thr)))
            roster_changed = bool(lineup_cont < thr)
            rel_info = _players_reliability_profile(
                cov_ratio=float(cov_ratio),
                freshness_avg=float(freshness_avg),
                lineup_continuity=float(lineup_cont),
                avg_gp_min=float(avg_gp_min_info),
                prior_games=float(ss.get("ml_uncertainty_prior_games", 22.0) or 22.0),
                interregional=_is_interregional_from_ml(ml),
                min_rel=float(ss.get("ml_players_reliability_min", 0.35) or 0.35),
                max_rel=float(ss.get("ml_players_reliability_max", 1.00) or 1.00),
                inter_mult=float(ss.get("ml_players_reliability_inter_mult", 0.92) or 0.92),
            )
            rel = float(rel_info.get("reliability", 1.0) or 1.0)
            players_context_info_out = _compute_players_context_info(
                team_games=team_games,
                team_a=str(teamA),
                team_b=str(teamB),
                filters=filters,
                as_of_date=as_of_date,
                session_state=ss,
                ml_info=ml,
                league_hint_a_fallback=str(blue_league_out or match_league_out or ""),
                league_hint_b_fallback=str(red_league_out or match_league_out or ""),
            )
            ctx_delta = float(_f((players_context_info_out or {}).get("delta"), 0.0))
            transfer_adj = float(_f(info.get("delta_transfer_adj"), 0.0))
            player_portability_summary = dict(info.get("player_portability_summary") or {}) if isinstance(info.get("player_portability_summary"), dict) else {}
            delta_eff_used = float(float(delta_eff_cur) * rel + ctx_delta + transfer_adj)
            _safe_caption(
                caption_fn,
                (
                    f"Players layer: delta_raw={_fmt_num(delta_cur,3,format_num_fn)} | "
                    f"delta_eff={_fmt_num(delta_eff_cur,3,format_num_fn)} ({delta_mode}) | "
                    f"delta_eff_used={_fmt_num(delta_eff_used,3,format_num_fn)} | "
                    f"ctx_delta={_fmt_num(ctx_delta,3,format_num_fn)} "
                    f"| transfer_adj={_fmt_num(transfer_adj,3,format_num_fn)} "
                    f"(ctx_edge={_fmt_num((players_context_info_out or {}).get('edge_score'),3,format_num_fn)}, "
                    f"ctx_rel={_fmt_pct((players_context_info_out or {}).get('edge_reliability'),format_pct_fn)}) | "
                    f"coverage={_fmt_pct(cov_ratio,format_pct_fn)} | freshness={_fmt_pct(freshness_avg,format_pct_fn)} | "
                    f"transfer_sig={_fmt_num(transfer_sig,2,format_num_fn)} | continuity={_fmt_pct(lineup_cont,format_pct_fn)} | "
                    f"reliability={_fmt_pct(rel,format_pct_fn)} | "
                    f"roster_gate={_fmt_pct(thr,format_pct_fn)} | "
                    f"w_lineup={_fmt_num(pl_w,3,format_num_fn)} | "
                    f"league_w={_fmt_num(float(info.get('league_weight', 1.0) or 1.0),2,format_num_fn)} | "
                    f"lane_fit={fit_lane_src or 'off'!s} | "
                    f"league_fit={_fmt_num(float(fit_lg_w),2,format_num_fn)} ({fit_lg_src or 'off'!s})"
                ),
            )
            if ml_engine == "mlcore_v2" and bool(use_lineup) and (ml is not None):
                base_p_raw = float(ml.get("p_raw", float("nan")))
                base_p_raw = min(max(base_p_raw, 1e-6), 1 - 1e-6)
                logit_base = math.log(base_p_raw / (1.0 - base_p_raw))
                logit_used = float(logit_base + float(pl_w) * float(delta_eff_used))
                p_raw_used = 1.0 / (1.0 + math.exp(-logit_used))
                p_map_players_local = float(calibrate_p_from_artifact_fn(artifact_path, float(p_raw_used)))
                p_map_team_local = float(p_map)
                n_team_eff_local = float(
                    min(
                        float(((ml.get("blue") or {}).get("games_played", 0) or 0)),
                        float(((ml.get("red") or {}).get("games_played", 0) or 0)),
                    )
                )
                n_players_eff_local = float(
                    max(0.0, float(avg_gp_min_info) * float(cov_ratio) * float(freshness_avg) * float(rel_info.get("n_players_mult", 1.0) or 1.0))
                )
                fus = fuse_probs_by_precision_fn(
                    p_team=p_map_team_local,
                    p_players=p_map_players_local,
                    n_team=n_team_eff_local,
                    n_players=n_players_eff_local,
                    team_scale=float(ss.get("_fit_team_scale_used", ss.get("_fit_team_scale", 1.0)) or 1.0),
                    players_scale=float(ss.get("_fit_players_scale_used", ss.get("_fit_players_scale", 1.0)) or 1.0),
                    coverage=float(cov_ratio),
                    transfer_signal=float(transfer_sig),
                    lineup_continuity=float(lineup_cont),
                    roster_changed=bool(roster_changed),
                    season_phase=("playoffs" if str(playoffs_opt).lower() in {"playoffs", "true", "1"} else "auto"),
                )
                p_map = float(fus.get("p_fused", p_map_team_local))
                fusion_info = dict(fus)
                p_team = float(p_map_team_local)
                p_players = float(p_map_players_local)
                p_fused = float(p_map)
                n_team_eff = float(n_team_eff_local)
                n_players_eff = float(n_players_eff_local)
                # Post-fusion calibration (feature-flagged). Ablation: +0.97pp logloss + ECE -48%.
                if bool(ss.get("ml_postfusion_cal_enabled", False)) and math.isfinite(float(p_map)):
                    try:
                        from app_core.postfusion_calibration_core import apply_postfusion_calibration as _apply_pfc
                        _split_hint_pf = str(getattr(filters, "split", "") or "")
                        _phase_hint_pf = "playoffs" if str(playoffs_opt).lower() in {"playoffs", "true", "1"} else "regular"
                        _p_postfus, _pfc_info = _apply_pfc(
                            float(p_map),
                            enabled=True,
                            league_hint=str(match_league or ""),
                            bo=int(bo_i),
                            split_hint=_split_hint_pf,
                            phase_hint=_phase_hint_pf,
                            side="blue",
                            microseg_table=postfusion_microseg_table,
                            microseg_min_n=int(ss.get("ml_microseg_min_n", 30)),
                            microseg_strength=float(ss.get("ml_microseg_strength", 0.60) or 0.60),
                            microseg_cap_pp=float(ss.get("ml_microseg_cap_pp", 8.0) or 8.0),
                            apply_microseg_fn=apply_microseg_fn,
                            apply_fused_postcal_fn=apply_fused_postcal_fn,
                            grupo_coef=postfusion_grupo_coef,
                            use_fitted_postcal=bool(ss.get("ml_fused_post_cal_use_fitted", True)),
                        )
                        if math.isfinite(_p_postfus):
                            p_map = float(_p_postfus)
                            p_fused = float(p_map)
                        fusion_info["postfusion_calibration"] = _pfc_info
                        if ((_pfc_info or {}).get("postcal") or {}).get("modo") == "grupo":
                            _pfc_grupo_aplicado = True
                    except Exception as _exc_pfc:
                        fusion_info["postfusion_calibration"] = {"enabled": True, "error": f"{type(_exc_pfc).__name__}: {_exc_pfc}"}
                fusion_info["freshness_avg"] = float(freshness_avg)
                fusion_info["lineup_continuity"] = float(lineup_cont)
                fusion_info["transfer_signal"] = float(transfer_sig)
                fusion_info["roster_gate"] = float(thr)
                fusion_info["roster_changed"] = bool(roster_changed)
                fusion_info["players_reliability"] = float(rel)
                fusion_info["players_reliability_components"] = {
                    "coverage": float(rel_info.get("coverage", 0.0)),
                    "freshness": float(rel_info.get("freshness", 0.0)),
                    "continuity": float(rel_info.get("continuity", 0.0)),
                    "evidence": float(rel_info.get("evidence", 0.0)),
                    "interregional": bool(rel_info.get("interregional", 0.0) >= 0.5),
                }
                fusion_info["players_context_quality"] = dict(players_context_info_out)
                fusion_info["player_portability_summary"] = dict(player_portability_summary)
                fusion_info["delta_transfer_adj"] = float(transfer_adj)
                fusion_info["lineup_transfer_power_a"] = _f(player_portability_summary.get("lineup_transfer_power_a"), float("nan"))
                fusion_info["lineup_transfer_power_b"] = _f(player_portability_summary.get("lineup_transfer_power_b"), float("nan"))
                delta_eff_cur = float(delta_eff_used)
                p_series = _safe_prob_win_series(prob_win_series_fn, float(p_map), int(bo_i))
                if math.isfinite(_f(ml.get("scale_points", float("nan")))):
                    scale_pts = float(ml.get("scale_points"))
                    diff_cur = float((scale_pts / math.log(10.0)) * logit_used)
                _safe_caption(
                    caption_fn,
                    (
                        f"p_map team={_fmt_num(p_map_team_local,4,format_num_fn)} | "
                        f"players={_fmt_num(p_map_players_local,4,format_num_fn)} | "
                        f"fused={_fmt_num(p_map,4,format_num_fn)} "
                        f"(w_team={_fmt_num(fus.get('w_team', float('nan')),2,format_num_fn)}, "
                        f"w_players={_fmt_num(fus.get('w_players', float('nan')),2,format_num_fn)}, "
                        f"n_team={_fmt_num(n_team_eff_local,1,format_num_fn)}, "
                        f"n_players={_fmt_num(n_players_eff_local,1,format_num_fn)}) | "
                        f"p_series={_fmt_num(p_series,4,format_num_fn)}"
                    ),
                )
                cg = fus.get("coherence_guard") if isinstance(fus, dict) else None
                if isinstance(cg, dict) and bool(cg.get("applied")):
                    _safe_caption(
                        caption_fn,
                        (
                            f"Coerencia Team/Players aplicada: d={_fmt_num(float(cg.get('dpp', float('nan'))),1,format_num_fn)} pp | "
                            f"coverage={_fmt_pct(float(cg.get('coverage', float('nan'))),format_pct_fn)} | "
                            f"pull={_fmt_num(100.0 * float(cg.get('pull', 0.0)),1,format_num_fn)}%."
                        ),
                    )
        except Exception as exc:
            _safe_warning(warning_fn, f"Falha ao aplicar ML players layer: {type(exc).__name__}: {exc}")
            lineup_mode_out = "CSV byTournament (manual)"

    # Manual players layer.
    if lineup_mode_out == "CSV byTournament (manual)":
        sA, fA, tA, gpA = score_team_fn(teamA_key, lineupA_sel)
        sB, fB, tB, gpB = score_team_fn(teamB_key, lineupB_sel)
        delta_cur = float(sA - sB)
        delta_eff_cur = float(delta_players_effect_fn(delta_cur, mode=delta_mode, cap=delta_cap, slope=delta_slope))
        cov_manual = 0.0
        try:
            cov_manual = min(float(fA / max(tA, 1)), float(fB / max(tB, 1)))
        except (TypeError, ValueError, ZeroDivisionError):
            cov_manual = 0.0
        avg_gp_min = (
            min(float(gpA) / max(float(fA), 1.0), float(gpB) / max(float(fB), 1.0))
            if (fA > 0 and fB > 0)
            else 0.0
        )
        fresh_gp_ref = float(ss.get("ml_players_manual_freshness_gp_ref", 18.0) or 18.0)
        freshness_proxy = float(max(0.0, min(1.0, float(avg_gp_min) / max(1.0, float(fresh_gp_ref)))))
        continuity_proxy = float(max(0.0, min(1.0, float(cov_manual))))
        rel_info = _players_reliability_profile(
            cov_ratio=float(cov_manual),
            freshness_avg=float(freshness_proxy),
            lineup_continuity=float(continuity_proxy),
            avg_gp_min=float(avg_gp_min),
            prior_games=float(ss.get("ml_uncertainty_prior_games", 22.0) or 22.0),
            interregional=_is_interregional_from_ml(ml),
            min_rel=float(ss.get("ml_players_reliability_min", 0.35) or 0.35),
            max_rel=float(ss.get("ml_players_reliability_max", 1.00) or 1.00),
            inter_mult=float(ss.get("ml_players_reliability_inter_mult", 0.92) or 0.92),
        )
        rel = float(rel_info.get("reliability", 1.0) or 1.0)
        players_context_info_out = _compute_players_context_info(
            team_games=team_games,
            team_a=str(teamA),
            team_b=str(teamB),
            filters=filters,
            as_of_date=as_of_date,
            session_state=ss,
            ml_info=ml,
            league_hint_a_fallback=str(blue_league_out or lgA or ""),
            league_hint_b_fallback=str(red_league_out or ""),
        )
        ctx_delta = float(_f((players_context_info_out or {}).get("delta"), 0.0))
        transfer_adj = 0.0
        player_portability_summary = {}
        delta_eff_used = float(float(delta_eff_cur) * rel + ctx_delta + transfer_adj)
        _safe_caption(
            caption_fn,
            (
                f"Score (players, manual): {teamA}={_fmt_num(sA,2,format_num_fn)} (ok {fA}/{tA}) | "
                f"{teamB}={_fmt_num(sB,2,format_num_fn)} (ok {fB}/{tB}) | "
                f"delta_raw={_fmt_num(delta_cur,2,format_num_fn)} | "
                f"delta_eff={_fmt_num(delta_eff_cur,2,format_num_fn)} ({delta_mode}) | "
                f"delta_eff_used={_fmt_num(delta_eff_used,2,format_num_fn)} | "
                f"ctx_delta={_fmt_num(ctx_delta,2,format_num_fn)} "
                f"(ctx_edge={_fmt_num((players_context_info_out or {}).get('edge_score'),2,format_num_fn)}, "
                f"ctx_rel={_fmt_pct((players_context_info_out or {}).get('edge_reliability'),format_pct_fn)}) | "
                f"coverage={_fmt_pct(cov_manual,format_pct_fn)} | "
                f"freshness={_fmt_pct(freshness_proxy,format_pct_fn)} | "
                f"continuity={_fmt_pct(continuity_proxy,format_pct_fn)} | "
                f"reliability={_fmt_pct(rel,format_pct_fn)}"
            ),
        )
        if (
            bool(use_lineup)
            and (ml is not None)
            and math.isfinite(_f(ml.get("diff_points", float("nan"))))
            and math.isfinite(_f(ml.get("scale_points", float("nan"))))
        ):
            base_diff = float(ml.get("diff_points"))
            scale_pts = float(ml.get("scale_points"))
            lam = float(ss.get("lineup_lambda_points", 90) or 90)
            diff_cur = float(base_diff + lam * float(delta_eff_used))
            p_raw_used = float(p_from_diff_points_fn(diff_cur, scale_points=scale_pts))
            p_map_players_local = float(calibrate_p_from_artifact_fn(artifact_path, p_raw_used))
            p_map_team_local = float(p_map)
            n_team_eff_local = float(
                min(
                    float(((ml.get("blue") or {}).get("games_played", 0) or 0)),
                    float(((ml.get("red") or {}).get("games_played", 0) or 0)),
                )
            )
            n_players_eff_local = float(
                max(0.0, float(avg_gp_min) * float(cov_manual) * float(rel_info.get("n_players_mult", 1.0) or 1.0))
            )
            thr_manual = float(max(0.30, min(0.90, float(ss.get("fusion_roster_cont_thresh_base", 0.55) or 0.55))))
            roster_changed_manual = bool(float(cov_manual) < float(thr_manual))
            fus = fuse_probs_by_precision_fn(
                p_team=p_map_team_local,
                p_players=p_map_players_local,
                n_team=n_team_eff_local,
                n_players=n_players_eff_local,
                team_scale=float(ss.get("_fit_team_scale_used", ss.get("_fit_team_scale", 1.0)) or 1.0),
                players_scale=float(ss.get("_fit_players_scale_used", ss.get("_fit_players_scale", 1.0)) or 1.0),
                coverage=float(cov_manual),
                transfer_signal=0.0,
                lineup_continuity=float(continuity_proxy),
                roster_changed=bool(roster_changed_manual),
                season_phase=("playoffs" if str(playoffs_opt).lower() in {"playoffs", "true", "1"} else "auto"),
            )
            p_map = float(fus.get("p_fused", p_map_team_local))
            fusion_info = dict(fus)
            p_team = float(p_map_team_local)
            p_players = float(p_map_players_local)
            p_fused = float(p_map)
            n_team_eff = float(n_team_eff_local)
            n_players_eff = float(n_players_eff_local)
            # Post-fusion calibration (feature-flagged). Ablation: +0.97pp logloss + ECE -48%.
            if bool(ss.get("ml_postfusion_cal_enabled", False)) and math.isfinite(float(p_map)):
                try:
                    from app_core.postfusion_calibration_core import apply_postfusion_calibration as _apply_pfc
                    _split_hint_pf = str(getattr(filters, "split", "") or "")
                    _phase_hint_pf = "playoffs" if str(playoffs_opt).lower() in {"playoffs", "true", "1"} else "regular"
                    _p_postfus, _pfc_info = _apply_pfc(
                        float(p_map),
                        enabled=True,
                        league_hint=str(match_league or ""),
                        bo=int(bo_i),
                        split_hint=_split_hint_pf,
                        phase_hint=_phase_hint_pf,
                        side="blue",
                        microseg_table=postfusion_microseg_table,
                        microseg_min_n=int(ss.get("ml_microseg_min_n", 30)),
                        microseg_strength=float(ss.get("ml_microseg_strength", 0.60) or 0.60),
                        microseg_cap_pp=float(ss.get("ml_microseg_cap_pp", 8.0) or 8.0),
                        apply_microseg_fn=apply_microseg_fn,
                        apply_fused_postcal_fn=apply_fused_postcal_fn,
                        grupo_coef=postfusion_grupo_coef,
                        use_fitted_postcal=bool(ss.get("ml_fused_post_cal_use_fitted", True)),
                    )
                    if math.isfinite(_p_postfus):
                        p_map = float(_p_postfus)
                        p_fused = float(p_map)
                    fusion_info["postfusion_calibration"] = _pfc_info
                    if ((_pfc_info or {}).get("postcal") or {}).get("modo") == "grupo":
                        _pfc_grupo_aplicado = True
                except Exception as _exc_pfc:
                    fusion_info["postfusion_calibration"] = {"enabled": True, "error": f"{type(_exc_pfc).__name__}: {_exc_pfc}"}
            fusion_info["freshness_avg"] = float(freshness_proxy)
            fusion_info["lineup_continuity"] = float(continuity_proxy)
            fusion_info["transfer_signal"] = 0.0
            fusion_info["roster_gate"] = float(thr_manual)
            fusion_info["roster_changed"] = bool(roster_changed_manual)
            fusion_info["players_reliability"] = float(rel)
            fusion_info["players_reliability_components"] = {
                "coverage": float(rel_info.get("coverage", 0.0)),
                "freshness": float(rel_info.get("freshness", 0.0)),
                "continuity": float(rel_info.get("continuity", 0.0)),
                "evidence": float(rel_info.get("evidence", 0.0)),
                "interregional": bool(rel_info.get("interregional", 0.0) >= 0.5),
            }
            fusion_info["players_context_quality"] = dict(players_context_info_out)
            fusion_info["player_portability_summary"] = dict(player_portability_summary)
            fusion_info["delta_transfer_adj"] = float(transfer_adj)
            delta_eff_cur = float(delta_eff_used)
            p_series = _safe_prob_win_series(prob_win_series_fn, float(p_map), int(bo_i))
            _safe_caption(
                caption_fn,
                (
                    f"Diff base={_fmt_num(base_diff,1,format_num_fn)} | "
                    f"diff players={_fmt_num(diff_cur,1,format_num_fn)} | "
                    f"p_map team={_fmt_num(p_map_team_local,4,format_num_fn)} | "
                    f"players={_fmt_num(p_map_players_local,4,format_num_fn)} | "
                    f"fused={_fmt_num(p_map,4,format_num_fn)} "
                    f"(cov={_fmt_pct(cov_manual,format_pct_fn)}, "
                    f"rel={_fmt_pct(rel,format_pct_fn)}, "
                    f"n_team={_fmt_num(n_team_eff_local,1,format_num_fn)}, "
                    f"n_players={_fmt_num(n_players_eff_local,1,format_num_fn)})"
                ),
            )
            cg = fus.get("coherence_guard") if isinstance(fus, dict) else None
            if isinstance(cg, dict) and bool(cg.get("applied")):
                _safe_caption(
                    caption_fn,
                    (
                        f"Coerencia Team/Players aplicada: d={_fmt_num(float(cg.get('dpp', float('nan'))),1,format_num_fn)} pp | "
                        f"coverage={_fmt_pct(float(cg.get('coverage', float('nan'))),format_pct_fn)} | "
                        f"pull={_fmt_num(100.0 * float(cg.get('pull', 0.0)),1,format_num_fn)}%."
                    ),
                )

    # CALIBRADOR COM GRUPO — APLICACAO DE RESERVA (08/09/2026).
    #
    # Os dois pontos acima vivem dentro de ramos `use_lineup`, e `use_lineup`
    # vem DESLIGADO por padrao (session_state['use_lineup_adjust'], default
    # False). Medido em 08/09 abrindo o app: com e sem o bloco em
    # params_fitted.json, o numero servido era IDENTICO ate o ultimo digito --
    # o estagio nunca era alcancado. O ganho de +0,02 que o gate media nao
    # estava sendo entregue na tela.
    #
    # POR QUE VALE PARA TODO JOGO, e nao so com escalacao: medido no teste
    # cego, o calibrador ajuda NOS DOIS ramos --
    #   com fusao (onde foi ajustado): +0,02073  IC [+0,01614, +0,02510]
    #   sem fusao (o app por padrao):  +0,01889  IC [+0,01465, +0,02296]
    # A literatura previa que um calibrador unico entre ramos de distribuicao
    # diferente "sub-corrigiria uns e sobre-corrigiria outros"
    # (https://arxiv.org/pdf/2605.13030, conferido em 08/09/2026 — e sobre
    # calibracao de features em modelos fundidos, entao e analogia, nao prova
    # do nosso caso). Aqui isso NAO aconteceu, e quem decide e a medicao.
    #
    # Sem o bloco em params_fitted.json isto e inerte: `grupo_coef=None` faz
    # apply_postfusion_calibration devolver p igual. Apagar a chave desliga.
    if (postfusion_grupo_coef and not _pfc_grupo_aplicado
            and bool(ss.get("ml_postfusion_cal_enabled", False))
            and math.isfinite(_f(p_map, float("nan")))):
        try:
            from app_core.postfusion_calibration_core import (
                apply_postfusion_calibration as _apply_pfc_reserva,
            )
            _p_res, _info_res = _apply_pfc_reserva(
                float(p_map),
                enabled=True,
                league_hint=str(match_league or ""),
                bo=int(bo_i),
                split_hint=str(getattr(filters, "split", "") or ""),
                phase_hint=("playoffs" if str(playoffs_opt).lower()
                            in {"playoffs", "true", "1"} else "regular"),
                side="blue",
                microseg_table=None,
                apply_microseg_fn=None,
                apply_fused_postcal_fn=None,
                grupo_coef=postfusion_grupo_coef,
            )
            if math.isfinite(float(_p_res)):
                p_map = float(_p_res)
                _pfc_grupo_aplicado = True
            if isinstance(trace_fusion_info, dict):
                trace_fusion_info["postfusion_calibration_reserva"] = dict(_info_res)
        except Exception as _exc_res:
            if isinstance(trace_fusion_info, dict):
                trace_fusion_info["postfusion_calibration_reserva"] = {
                    "erro": f"{type(_exc_res).__name__}: {_exc_res}"}


    # ML engine switch: elo season + players.
    elo_info = None
    if ml_engine == "elo_season_players" or bool(ss.get("ml_engine_compare", False)):
        try:
            elo_info = elo_season_players_p_map_fn(
                team_games=team_games,
                csv_path=str(csv_path),
                teamA=str(teamA),
                teamB=str(teamB),
                filters=filters,
                team_half_life_days=float(ss.get("elo_team_hl", 45) or 45),
                player_half_life_days=float(ss.get("elo_player_hl", 90) or 90),
                k_team=float(ss.get("elo_k_team", 26) or 26),
                k_player=float(ss.get("elo_k_player", 24) or 24),
                shrink_m=float(ss.get("elo_shrink_m", 18) or 18),
                w_players=float(ss.get("elo_w_players", 0.85) or 0.85),
            )
        except Exception as exc:
            elo_info = {"__error__": f"{type(exc).__name__}: {exc}"}

    if ml_engine == "elo_season_players":
        if elo_info is None or (isinstance(elo_info, dict) and elo_info.get("__error__")):
            _safe_warning(warning_fn, "Elo season + players: nao consegui calcular (ver Detalhes do ML).")
        else:
            p_map = float(elo_info.get("p_cal", float("nan")))
            diff_cur = float(elo_info.get("diff_points", float("nan")))
            p_series = _safe_prob_win_series(prob_win_series_fn, float(p_map), int(bo_i))
            delta_cur = 0.0

    try:
        p_after = float(p_map) if math.isfinite(float(p_map)) else float("nan")
        ap = (
            math.isfinite(float(p_prev))
            and math.isfinite(float(p_after))
            and abs(float(p_after) - float(p_prev)) > 1e-12
        )
        append_pipeline_step(
            steps=steps,
            stage="engine_lineup",
            before=p_prev,
            after=p_after,
            applied=bool(ap),
            engine=str(ml_engine),
            lineup_mode=str(lineup_mode_out),
        )
        p_prev = float(p_after)
    except Exception as exc:
        append_pipeline_step(
            steps=steps,
            stage="engine_lineup",
            before=p_prev,
            after=p_map,
            applied=False,
            engine=str(ml_engine),
            lineup_mode=str(lineup_mode_out),
            error=f"{type(exc).__name__}: {exc}",
        )

    return {
        "p_map_used": float(p_map) if math.isfinite(float(p_map)) else float("nan"),
        "p_series_used": float(p_series) if math.isfinite(float(p_series)) else float("nan"),
        "diff_used": float(diff_cur) if math.isfinite(float(diff_cur)) else float("nan"),
        "delta_players": float(delta_cur) if math.isfinite(float(delta_cur)) else 0.0,
        "delta_players_eff": float(delta_eff_cur) if math.isfinite(float(delta_eff_cur)) else 0.0,
        "trace_p_map_team": float(p_team) if math.isfinite(float(p_team)) else float("nan"),
        "trace_p_map_players": float(p_players) if math.isfinite(float(p_players)) else float("nan"),
        "trace_p_map_fused": float(p_fused) if math.isfinite(float(p_fused)) else float("nan"),
        "trace_n_team_eff": float(n_team_eff) if math.isfinite(float(n_team_eff)) else float("nan"),
        "trace_n_players_eff": float(n_players_eff) if math.isfinite(float(n_players_eff)) else float("nan"),
        "trace_fusion_info": fusion_info if isinstance(fusion_info, dict) else {},
        "trace_players_context_info": players_context_info_out if isinstance(players_context_info_out, dict) else {},
        "trace_uncertainty_info": uncertainty_info if isinstance(uncertainty_info, dict) else {},
        "_delta_mode": str(delta_mode),
        "_delta_cap": float(delta_cap),
        "_delta_slope": float(delta_slope),
        "lineup_mode": str(lineup_mode_out),
        "match_league": str(match_league_out or ""),
        "_elo_info": elo_info,
        "_p_pipe_prev": float(p_prev) if math.isfinite(float(p_prev)) else float("nan"),
        "trace_pipeline_steps": steps,
    }
