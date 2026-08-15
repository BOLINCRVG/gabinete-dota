from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple

from app_core.stats_config import flow_line_keys


_LINE_KEYS = (
    *flow_line_keys(),
    "teamA_kills_line_text",
    "teamB_kills_line_text",
    "hc_kills_text",
    "hc_towers_text",
    "hc_dragons_text",
    "combo_kills_text",
    "combo_time_text",
)


def compute_sim_seed_series(
    session_state: MutableMapping[str, Any],
    *,
    stable_seed_fn: Callable[..., int],
    season_label: Any,
    team_a: Any,
    team_b: Any,
    bo: Any,
    year_opt: Any,
    split_opt: Any,
    playoffs_opt: Any,
    league_mode: Any,
    map_mode_ui: Any,
    weight_mode: Any,
    half_life_days: Any,
    n_sims: Any,
    totals_mode: Any,
    ml_mode: Any,
) -> int:
    seed_mode = str(session_state.get("sim_seed_mode", "Deterministico (confronto+config)") or "Deterministico (confronto+config)")
    if seed_mode == "Fixo manual":
        seed = int(session_state.get("sim_seed_manual", 1337) or 1337)
    else:
        seed = int(
            stable_seed_fn(
                season_label,
                team_a,
                team_b,
                int(bo),
                str(year_opt),
                str(split_opt),
                str(playoffs_opt),
                str(league_mode),
                str(map_mode_ui),
                str(weight_mode),
                int(half_life_days) if half_life_days is not None else "uniform",
                int(n_sims),
                str(totals_mode),
                str(ml_mode),
            )
        )
    session_state["_sim_seed_series"] = int(seed)
    return int(seed)


def resolve_active_line_texts(
    session_state: Mapping[str, Any],
    *,
    flow_stage: str,
    saved_lines: Mapping[str, Any] | None,
) -> Dict[str, str]:
    saved = dict(saved_lines or {}) if (flow_stage == "results" and isinstance(saved_lines, Mapping)) else {}
    out: Dict[str, str] = {}
    for key in _LINE_KEYS:
        if flow_stage == "results":
            out[key] = str(saved.get(key, "") or "")
        else:
            out[key] = str(session_state.get(key, "") or "")
    return out


def build_analysis_signature(
    session_state: Mapping[str, Any],
    *,
    stable_seed_fn: Callable[..., int],
    sanitize_marginal_band_coeffs_fn: Callable[[Any], Dict[str, Dict[str, float]]],
    season_label: Any,
    team_a: Any,
    team_b: Any,
    bo: Any,
    year_opt: Any,
    split_opt: Any,
    playoffs_opt: Any,
    league_mode: Any,
    fixed_league: Any,
    window_opt: Any,
    weight_mode: Any,
    half_life_days: Any,
    n_sims: Any,
    totals_mode: Any,
    ml_mode: Any,
    map_mode_ui: Any,
    line_texts: Mapping[str, Any],
) -> int:
    return int(
        stable_seed_fn(
            season_label,
            str(team_a),
            str(team_b),
            int(bo),
            str(year_opt),
            str(split_opt),
            str(playoffs_opt),
            str(league_mode),
            str(fixed_league),
            str(window_opt),
            str(weight_mode),
            int(half_life_days) if half_life_days is not None else "uniform",
            int(n_sims),
            str(totals_mode),
            str(ml_mode),
            str(map_mode_ui),
            *[str(line_texts.get(key, "")) for key in flow_line_keys()],
            str(line_texts.get("teamA_kills_line_text", "")),
            str(line_texts.get("teamB_kills_line_text", "")),
            str(line_texts.get("hc_kills_text", "")),
            str(line_texts.get("hc_towers_text", "")),
            str(line_texts.get("hc_dragons_text", "")),
            str(line_texts.get("combo_kills_text", "")),
            str(line_texts.get("combo_time_text", "")),
            bool(session_state.get("resumo_combo_use_mismatch", False)),
            float(session_state.get("fusion_team_weight_early", 0.58) or 0.58),
            float(session_state.get("fusion_team_weight_mid", 0.75) or 0.75),
            float(session_state.get("fusion_team_weight_playoffs", 0.82) or 0.82),
            float(session_state.get("fusion_early_season_team_weight", 0.55) or 0.55),
            float(session_state.get("fusion_season_knee_games", 18.0) or 18.0),
            float(session_state.get("fusion_coverage_power", 1.5) or 1.5),
            float(session_state.get("fusion_transfer_boost", 0.35) or 0.35),
            float(session_state.get("fusion_divergence_pp_cap", 22.0) or 22.0),
            float(session_state.get("fusion_divergence_low_coverage", 0.30) or 0.30),
            float(session_state.get("fusion_divergence_shrink", 0.35) or 0.35),
            bool(session_state.get("ml_microseg_enabled", True)),
            int(session_state.get("ml_microseg_min_n", 30) or 30),
            float(session_state.get("ml_microseg_strength", 0.60) or 0.60),
            float(session_state.get("ml_microseg_shrink_k", 90.0) or 90.0),
            int(session_state.get("ml_microseg_lookback_games", 1200) or 1200),
            float(session_state.get("ml_microseg_cap_pp", 8.0) or 8.0),
            bool(session_state.get("ml_microseg_hierarchical", True)),
            float(session_state.get("ml_microseg_hier_decay", 0.80) or 0.80),
            bool(session_state.get("ml_fused_post_cal_use_fitted", True)),
            bool(session_state.get("ml_uncertainty_enabled", True)),
            float(session_state.get("ml_uncertainty_strength", 0.55) or 0.55),
            float(session_state.get("ml_uncertainty_prior_games", 22.0) or 22.0),
            float(session_state.get("ml_uncertainty_max_shrink", 0.65) or 0.65),
            str(session_state.get("ml_uncertainty_exclude_leagues", "") or ""),
            float(session_state.get("ml_interregional_uncertainty_strength_mult", 0.30) or 0.30),
            float(session_state.get("ml_interregional_uncertainty_prior_mult", 0.60) or 0.60),
            float(session_state.get("ml_interregional_uncertainty_cap_mult", 0.40) or 0.40),
            float(session_state.get("ml_interregional_conf_guard_players_eff_weight", 0.12) or 0.12),
            float(session_state.get("ml_interregional_conf_guard_players_eff_cap", 18.0) or 18.0),
            float(session_state.get("ml_interregional_conf_guard_max_shrink_mult", 0.35) or 0.35),
            bool(session_state.get("ml_marginal_trust_enabled", True)),
            float(session_state.get("ml_marginal_trust_strength", 0.70) or 0.70),
            float(session_state.get("ml_marginal_trust_prior_games", 28.0) or 28.0),
            float(session_state.get("ml_marginal_trust_max_shrink", 0.72) or 0.72),
            float(session_state.get("ml_marginal_trust_cap_prob", 0.84) or 0.84),
            float(session_state.get("ml_marginal_trust_score_soft", 68.0) or 68.0),
            float(session_state.get("ml_marginal_trust_score_hard", 46.0) or 46.0),
            str(session_state.get("ml_marginal_trust_exclude_leagues", "") or ""),
            bool(session_state.get("ml_marginal_trust_band_tuning", True)),
            json.dumps(
                sanitize_marginal_band_coeffs_fn(session_state.get("ml_marginal_trust_band_coeffs", {})),
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
    )


def build_consistency_analysis_signature(
    session_state: Mapping[str, Any],
    *,
    stable_seed_fn: Callable[..., int],
    sanitize_marginal_band_coeffs_fn: Callable[[Any], Dict[str, Dict[str, float]]],
    season_label: Any,
    team_a: Any,
    team_b: Any,
    bo: Any,
    year_opt: Any,
    split_opt: Any,
    playoffs_opt: Any,
    league_mode: Any,
    fixed_league: Any,
    window_opt: Any,
    weight_mode: Any,
    half_life_days: Any,
    n_sims: Any,
    totals_mode: Any,
    map_mode_ui: Any,
    line_texts: Mapping[str, Any],
) -> int:
    return int(
        stable_seed_fn(
            season_label,
            str(team_a),
            str(team_b),
            int(bo),
            str(year_opt),
            str(split_opt),
            str(playoffs_opt),
            str(league_mode),
            str(fixed_league),
            str(window_opt),
            str(weight_mode),
            int(half_life_days) if half_life_days is not None else "uniform",
            int(n_sims),
            str(totals_mode),
            str(map_mode_ui),
            *[str(line_texts.get(key, "")) for key in flow_line_keys()],
            str(line_texts.get("teamA_kills_line_text", "")),
            str(line_texts.get("teamB_kills_line_text", "")),
            str(line_texts.get("hc_kills_text", "")),
            str(line_texts.get("hc_towers_text", "")),
            str(line_texts.get("hc_dragons_text", "")),
            str(line_texts.get("combo_kills_text", "")),
            str(line_texts.get("combo_time_text", "")),
            bool(session_state.get("resumo_combo_use_mismatch", False)),
            bool(session_state.get("ml_microseg_enabled", True)),
            int(session_state.get("ml_microseg_min_n", 30) or 30),
            float(session_state.get("ml_microseg_strength", 0.60) or 0.60),
            float(session_state.get("ml_microseg_shrink_k", 90.0) or 90.0),
            int(session_state.get("ml_microseg_lookback_games", 1200) or 1200),
            float(session_state.get("ml_microseg_cap_pp", 8.0) or 8.0),
            bool(session_state.get("ml_microseg_hierarchical", True)),
            float(session_state.get("ml_microseg_hier_decay", 0.80) or 0.80),
            bool(session_state.get("ml_fused_post_cal_use_fitted", True)),
            bool(session_state.get("ml_uncertainty_enabled", True)),
            float(session_state.get("ml_uncertainty_strength", 0.55) or 0.55),
            float(session_state.get("ml_uncertainty_prior_games", 22.0) or 22.0),
            float(session_state.get("ml_uncertainty_max_shrink", 0.65) or 0.65),
            str(session_state.get("ml_uncertainty_exclude_leagues", "") or ""),
            float(session_state.get("ml_interregional_uncertainty_strength_mult", 0.30) or 0.30),
            float(session_state.get("ml_interregional_uncertainty_prior_mult", 0.60) or 0.60),
            float(session_state.get("ml_interregional_uncertainty_cap_mult", 0.40) or 0.40),
            float(session_state.get("ml_interregional_conf_guard_players_eff_weight", 0.12) or 0.12),
            float(session_state.get("ml_interregional_conf_guard_players_eff_cap", 18.0) or 18.0),
            float(session_state.get("ml_interregional_conf_guard_max_shrink_mult", 0.35) or 0.35),
            bool(session_state.get("ml_marginal_trust_enabled", True)),
            float(session_state.get("ml_marginal_trust_strength", 0.70) or 0.70),
            float(session_state.get("ml_marginal_trust_prior_games", 28.0) or 28.0),
            float(session_state.get("ml_marginal_trust_max_shrink", 0.72) or 0.72),
            float(session_state.get("ml_marginal_trust_cap_prob", 0.84) or 0.84),
            float(session_state.get("ml_marginal_trust_score_soft", 68.0) or 68.0),
            float(session_state.get("ml_marginal_trust_score_hard", 46.0) or 46.0),
            str(session_state.get("ml_marginal_trust_exclude_leagues", "") or ""),
            bool(session_state.get("ml_marginal_trust_band_tuning", True)),
            json.dumps(
                sanitize_marginal_band_coeffs_fn(session_state.get("ml_marginal_trust_band_coeffs", {})),
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
    )



def stable_seed_from_parts(*parts: Any) -> int:
    txt = "||".join([str(x) for x in parts])
    h = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def _file_sig_local(path: str) -> Tuple[str, int, int]:
    p = Path(path)
    try:
        st_ = p.stat()
        return (str(p.resolve()), int(st_.st_size), int(st_.st_mtime))
    except OSError:
        return (str(p), 0, 0)


def ml_consistency_signature(
    *,
    season_label: str,
    teamA: str,
    teamB: str,
    bo: int,
    ml_engine: str,
    calc_model_mode: str,
    year_opt: Any,
    split_opt: Any,
    playoffs_opt: Any,
    league_mode: str,
    fixed_league: Any,
    as_of_date: Any,
    artifact_path: str,
    use_lineup_adjust: bool,
    lineup_mode: str,
    lineupA: Optional[dict] = None,
    lineupB: Optional[dict] = None,
) -> int:
    art_sig = _file_sig_local(str(artifact_path)) if str(artifact_path or "").strip() else ("", 0, 0)
    return stable_seed_from_parts(
        season_label, teamA, teamB, int(bo), ml_engine, calc_model_mode,
        year_opt, split_opt, playoffs_opt, league_mode, fixed_league,
        str(as_of_date), art_sig, bool(use_lineup_adjust), str(lineup_mode or ""),
        json.dumps(lineupA or {}, sort_keys=True, ensure_ascii=True),
        json.dumps(lineupB or {}, sort_keys=True, ensure_ascii=True),
    )
