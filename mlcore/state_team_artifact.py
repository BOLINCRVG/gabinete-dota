from __future__ import annotations

import copy
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from .team_artifact_io import load_team_state_artifact, save_team_state_artifact  # noqa: F401 - re-exported for callers
from .io_oracle import load_oracle_elixir_games
from .players_layer import load_match_player_rows, lineup_for_game
from .utils import safe_float as _f, clip as _clip
from .team_artifact_normalize import _norm_team_key, _patch_key, _norm_upper_key
from .team_artifact_validate import _coerce_float_map, _param_map_sig
from .team_artifact_game_quality import (
    _game_quality_blue,
    _qwl_adjusted_signal,
)
from .team_artifact_utils import (
    _file_sig_info,
    _league_patch_vol_multiplier,
    _safe_iso_date,
    _same_source_path_order,
    _source_sig_changed,
)



# Helpers de matematica de estado movidos para state_team_math (16/08/2026).
# Re-exportados: testes e chamadores importam varios deles DAQUI ha tempo.

# SUPERFICIE PUBLICA do modulo: estes nomes sao importados DAQUI por testes e
# por outros modulos ha tempo. O fatiamento de 16/08/2026 moveu a implementacao,
# mas o endereco tem que continuar valendo — senao a quebra aparece como
# ImportError longe daqui.
from .team_artifact_utils import (  # noqa: F401 - superficie publica
    _days_between_iso,
    _file_sig_info,
    _safe_iso_date,
    _same_source_path_order,
)

from .state_team_math import (  # noqa: F401 - re-exportado
    TeamState,
    _apply_league_switch_recenter,
    _apply_patch_switch_recenter,
    _apply_roster_recenter,
    _effective_k,
    _get_or_create_team,
    _glicko_lite_p_blue,
    _inflate_sigma_for_roster,
    _inflate_sigma_for_time_and_regime,
    _league_prior_mu,
    _lineup_continuity,
    _params_match_seed,
    _recenter_mu,
    _resolve_effective_params,
    _resolve_scoped_param,
    _team_state_from_dict,
    _update_bayes_joint_after_game,
    _update_sigma_after_game,
    _update_team_qwl_stats,
)


def build_team_state_artifact(
    csv_paths: Iterable[str],
    *,
    keep_playoffs: bool | None = None,
    include_players_for_roster: bool = True,
    base_k: float = 0.22,
    base_sigma: float = 1.00,
    min_sigma: float = 0.20,
    max_sigma: float = 2.25,
    sigma_decay: float = 0.18,
    inactivity_sigma_per_day: float = 0.010,
    patch_sigma_bump: float = 0.22,
    roster_sigma_bump: float = 0.35,
    patch_mu_recenter: float = 0.11,
    roster_mu_recenter: float = 0.28,
    league_switch_mu_recenter: float = 0.12,
    roster_cont_hard_reset: float = 0.25,
    roster_hard_reset_extra_sigma: float = 0.18,
    quality_k_boost: float = 0.22,
    qwl_expected_base: float = 0.12,
    qwl_expected_slope: float = 0.60,
    qwl_surplus_mix: float = 0.70,
    qwl_tempo_weight: float = 0.08,
    qwl_conversion_weight: float = 0.12,
    qwl_throw_weight: float = 1.20,
    qwl_comeback_weight: float = 0.65,
    qwl_phase_dynamic_strength: float = 0.35,
    qwl_opponent_context_mix: float = 0.30,
    mu_update_mode: str = "bayes_joint_v1",
    sigma_update_mode: str = "bayes_logit_v1",
    bayes_info_scale: float = 1.0,
    bayes_mu_q_strength: float = 1.0,
    bayes_info_scale_by_league: Optional[Dict[str, float]] = None,
    bayes_info_scale_by_patch: Optional[Dict[str, float]] = None,
    quality_k_boost_by_league: Optional[Dict[str, float]] = None,
    quality_k_boost_by_patch: Optional[Dict[str, float]] = None,
    qwl_surplus_mix_by_league: Optional[Dict[str, float]] = None,
    qwl_surplus_mix_by_patch: Optional[Dict[str, float]] = None,
    qwl_throw_weight_by_league: Optional[Dict[str, float]] = None,
    qwl_throw_weight_by_patch: Optional[Dict[str, float]] = None,
    qwl_comeback_weight_by_league: Optional[Dict[str, float]] = None,
    qwl_comeback_weight_by_patch: Optional[Dict[str, float]] = None,
    qwl_phase_dynamic_strength_by_league: Optional[Dict[str, float]] = None,
    qwl_phase_dynamic_strength_by_patch: Optional[Dict[str, float]] = None,
    qwl_opponent_context_mix_by_league: Optional[Dict[str, float]] = None,
    qwl_opponent_context_mix_by_patch: Optional[Dict[str, float]] = None,
    seed_artifact: Optional[Dict[str, Any]] = None,
    incremental: bool = True,
    store_processed_gameids: bool = True,
    incremental_backfill_policy: str = "rebuild",
    online_journal_tail_keep: int = 120,
    eval_track_metrics: bool = False,
    max_games_eval: Optional[int] = None,
) -> Dict[str, Any]:
    csv_list = [str(Path(str(p)).expanduser()) for p in (csv_paths or []) if str(p or "").strip()]
    if not csv_list:
        return {"error": "no_csv_paths"}

    source_files_cur: List[Dict[str, Any]] = [_file_sig_info(p) for p in csv_list]
    csv_list_scan: List[str] = list(csv_list)
    source_scan_mode = "all_csvs"
    source_changed_count = 0
    source_changed_paths: List[str] = []

    states: Dict[str, TeamState] = {}
    processed_gameids: Set[str] = set()
    update_mode = "full_rebuild"
    seed_generated_at = None
    seed_used = False
    seed_processed_count = 0
    seed_teams_count = 0
    backfill_detected = False
    backfill_unknown_games = 0
    backfill_oldest_date: Optional[str] = None
    games_skipped_seen = 0
    online_journal_tail: List[Dict[str, Any]] = []
    last_processed_game: Optional[Dict[str, Any]] = None

    # Seed/incremental eligibility is evaluated before CSV loading so we can avoid
    # re-reading unchanged historical files on the common daily append path.
    _seed_paths_match = False
    if bool(incremental) and _params_match_seed(
        seed_artifact,
        base_k=float(base_k),
        base_sigma=float(base_sigma),
        min_sigma=float(min_sigma),
        max_sigma=float(max_sigma),
        sigma_decay=float(sigma_decay),
        inactivity_sigma_per_day=float(inactivity_sigma_per_day),
        patch_sigma_bump=float(patch_sigma_bump),
        roster_sigma_bump=float(roster_sigma_bump),
        patch_mu_recenter=float(patch_mu_recenter),
        roster_mu_recenter=float(roster_mu_recenter),
        league_switch_mu_recenter=float(league_switch_mu_recenter),
        roster_cont_hard_reset=float(roster_cont_hard_reset),
        roster_hard_reset_extra_sigma=float(roster_hard_reset_extra_sigma),
        quality_k_boost=float(quality_k_boost),
        qwl_expected_base=float(qwl_expected_base),
        qwl_expected_slope=float(qwl_expected_slope),
        qwl_surplus_mix=float(qwl_surplus_mix),
        qwl_tempo_weight=float(qwl_tempo_weight),
        qwl_conversion_weight=float(qwl_conversion_weight),
        qwl_throw_weight=float(qwl_throw_weight),
        qwl_comeback_weight=float(qwl_comeback_weight),
        qwl_phase_dynamic_strength=float(qwl_phase_dynamic_strength),
        qwl_opponent_context_mix=float(qwl_opponent_context_mix),
        mu_update_mode=str(mu_update_mode),
        sigma_update_mode=str(sigma_update_mode),
        bayes_info_scale=float(bayes_info_scale),
        bayes_mu_q_strength=float(bayes_mu_q_strength),
        bayes_info_scale_by_league=(_coerce_float_map(bayes_info_scale_by_league)),
        bayes_info_scale_by_patch=(_coerce_float_map(bayes_info_scale_by_patch)),
        quality_k_boost_by_league=(_coerce_float_map(quality_k_boost_by_league)),
        quality_k_boost_by_patch=(_coerce_float_map(quality_k_boost_by_patch)),
        qwl_surplus_mix_by_league=(_coerce_float_map(qwl_surplus_mix_by_league)),
        qwl_surplus_mix_by_patch=(_coerce_float_map(qwl_surplus_mix_by_patch)),
        qwl_throw_weight_by_league=(_coerce_float_map(qwl_throw_weight_by_league)),
        qwl_throw_weight_by_patch=(_coerce_float_map(qwl_throw_weight_by_patch)),
        qwl_comeback_weight_by_league=(_coerce_float_map(qwl_comeback_weight_by_league)),
        qwl_comeback_weight_by_patch=(_coerce_float_map(qwl_comeback_weight_by_patch)),
        qwl_phase_dynamic_strength_by_league=(_coerce_float_map(qwl_phase_dynamic_strength_by_league)),
        qwl_phase_dynamic_strength_by_patch=(_coerce_float_map(qwl_phase_dynamic_strength_by_patch)),
        qwl_opponent_context_mix_by_league=(_coerce_float_map(qwl_opponent_context_mix_by_league)),
        qwl_opponent_context_mix_by_patch=(_coerce_float_map(qwl_opponent_context_mix_by_patch)),
        include_players_for_roster=bool(include_players_for_roster),
    ):
        seed_processed = seed_artifact.get("processed_gameids")
        seed_generated_at = str(seed_artifact.get("generated_at") or "")
        if isinstance(seed_processed, list) and seed_processed:
            processed_gameids = {str(x) for x in seed_processed if str(x).strip()}
            seed_teams = seed_artifact.get("teams") if isinstance(seed_artifact.get("teams"), dict) else {}
            for _k, _v in (seed_teams or {}).items():
                _st = _team_state_from_dict(_v if isinstance(_v, dict) else {})
                if _st is not None:
                    states[_st.key] = _st
            seed_processed_count = len(processed_gameids)
            seed_teams_count = len(states)
            if states:
                update_mode = "incremental_from_seed"
                seed_used = True
            else:
                update_mode = "full_rebuild_seed_empty_states"
        else:
            update_mode = "full_rebuild_seed_no_index"
    elif bool(incremental) and isinstance(seed_artifact, dict):
        update_mode = "full_rebuild_param_mismatch"

    if bool(seed_used):
        _seed_meta = seed_artifact.get("meta") if isinstance(seed_artifact, dict) and isinstance(seed_artifact.get("meta"), dict) else {}
        _seed_sources_raw = (_seed_meta or {}).get("source_files")
        _seed_sources = [dict(x) for x in _seed_sources_raw if isinstance(x, dict)] if isinstance(_seed_sources_raw, list) else []
        _seed_paths_match = _same_source_path_order(_seed_sources, source_files_cur)
        _seed_sources_have_real_stats = bool(_seed_sources) and all(bool(x.get("exists")) for x in _seed_sources)
        _cur_sources_have_real_stats = bool(source_files_cur) and all(bool(x.get("exists")) for x in source_files_cur)
        if _seed_paths_match and _seed_sources_have_real_stats and _cur_sources_have_real_stats:
            _changed_idx = [
                i for i, (old_s, cur_s) in enumerate(zip(_seed_sources, source_files_cur))
                if _source_sig_changed(old_s, cur_s)
            ]
            source_changed_count = len(_changed_idx)
            source_changed_paths = [str(source_files_cur[i].get("path") or "") for i in _changed_idx]
            if len(_changed_idx) == 0:
                # True online fast-path: same params + same source signatures = reuse seed.
                rep = copy.deepcopy(seed_artifact)
                if not isinstance(rep, dict):
                    rep = {}
                rep["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
                rep["version"] = "team_state_artifact_v9"
                rep_meta = rep.get("meta") if isinstance(rep.get("meta"), dict) else {}
                if not isinstance(rep_meta, dict):
                    rep_meta = {}
                rep_meta.update(
                    {
                        "csv_count": len(csv_list),
                        "candidate_new_games": 0,
                        "processing_games": 0,
                        "games_used": 0,
                        "games_skipped_seen": 0,
                        "seed_used": True,
                        "incremental_fastpath": True,
                        "seed_processed_count": int(seed_processed_count),
                        "seed_teams_count": int(seed_teams_count),
                        "update_mode": "incremental_no_source_change_seed_reused",
                        "seed_generated_at": (str(seed_generated_at) if seed_generated_at else None),
                        "backfill_detected": False,
                        "backfill_unknown_games": 0,
                        "backfill_oldest_date": None,
                        "source_scan_mode": "none_seed_reuse",
                        "source_changed_count": 0,
                        "source_changed_paths": [],
                        "source_files": source_files_cur,
                        "source_files_scanned": [],
                        "last_processed_game": ((_seed_meta or {}).get("last_processed_game") if isinstance(_seed_meta, dict) else None),
                        "online_journal_tail_count": len((_seed_meta or {}).get("online_journal_tail") or []) if isinstance(_seed_meta, dict) else 0,
                    }
                )
                rep["meta"] = rep_meta
                return rep
            if len(_changed_idx) == 1 and int(_changed_idx[0]) == int(len(csv_list) - 1):
                # Common case: only the daily CSV changed; scan only that file.
                csv_list_scan = [str(csv_list[-1])]
                source_scan_mode = "last_csv_only_incremental"
            elif len(_changed_idx) > 0:
                # Historical file changed -> fail closed (rebuild) to keep chronology exact.
                states = {}
                processed_gameids = set()
                seed_used = False
                source_scan_mode = "all_csvs_history_source_changed"
                update_mode = "full_rebuild_history_source_changed"

    game_frames: List[pd.DataFrame] = []
    for p in csv_list_scan:
        try:
            g = load_oracle_elixir_games(p, keep_playoffs=keep_playoffs)
            if g is not None and not g.empty:
                game_frames.append(g)
        except Exception:
            continue
    if not game_frames:
        return {"error": "no_games_loaded", "csv_paths": csv_list}

    # Guard pandas concat dtype inference against frames that are effectively
    # empty payloads (all rows NaN and/or all columns NaN), which can trigger
    # FutureWarning and unstable dtype resolution across versions.
    frames_clean: List[pd.DataFrame] = []
    for g in game_frames:
        if g is None or g.empty:
            continue
        g2 = g.dropna(axis=1, how="all")
        if g2.empty:
            continue
        if g2.dropna(how="all").empty:
            continue
        frames_clean.append(g2)
    if not frames_clean:
        return {"error": "no_games_loaded_after_clean", "csv_paths": csv_list}

    games = pd.concat(frames_clean, ignore_index=True, sort=False)
    games["gameid"] = games["gameid"].astype(str)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.dropna(subset=["gameid", "date", "blue", "red"]).copy()
    games = (
        games.sort_values(["date", "gameid"])
        .drop_duplicates(subset=["gameid"], keep="last")
        .reset_index(drop=True)
    )
    if isinstance(max_games_eval, int) and int(max_games_eval) > 0 and len(games) > int(max_games_eval):
        games = games.tail(int(max_games_eval)).reset_index(drop=True)
    if games.empty:
        return {"error": "no_valid_games_after_dedupe", "csv_paths": csv_list}

    map_bayes_lg = _coerce_float_map(bayes_info_scale_by_league)
    map_bayes_patch = _coerce_float_map(bayes_info_scale_by_patch)
    map_quality_lg = _coerce_float_map(quality_k_boost_by_league)
    map_quality_patch = _coerce_float_map(quality_k_boost_by_patch)
    map_qwl_mix_lg = _coerce_float_map(qwl_surplus_mix_by_league)
    map_qwl_mix_patch = _coerce_float_map(qwl_surplus_mix_by_patch)
    map_qwl_throw_lg = _coerce_float_map(qwl_throw_weight_by_league)
    map_qwl_throw_patch = _coerce_float_map(qwl_throw_weight_by_patch)
    map_qwl_comeback_lg = _coerce_float_map(qwl_comeback_weight_by_league)
    map_qwl_comeback_patch = _coerce_float_map(qwl_comeback_weight_by_patch)
    map_qwl_phase_dyn_lg = _coerce_float_map(qwl_phase_dynamic_strength_by_league)
    map_qwl_phase_dyn_patch = _coerce_float_map(qwl_phase_dynamic_strength_by_patch)
    map_qwl_opp_ctx_lg = _coerce_float_map(qwl_opponent_context_mix_by_league)
    map_qwl_opp_ctx_patch = _coerce_float_map(qwl_opponent_context_mix_by_patch)

    # Specs dos params efetivos por jogo: nome -> (global, by_league, by_patch,
    # lo, hi). lo/hi None = sem clip. Resolvido por _resolve_effective_params
    # (mantém os MESMOS limites históricos por escopo).
    _scoped_param_specs: Dict[str, Tuple[float, Dict[str, float], Dict[str, float], Optional[float], Optional[float]]] = {
        "bayes_info_scale": (float(bayes_info_scale), map_bayes_lg, map_bayes_patch, None, None),
        "quality_k_boost": (float(quality_k_boost), map_quality_lg, map_quality_patch, None, None),
        "qwl_surplus_mix": (float(qwl_surplus_mix), map_qwl_mix_lg, map_qwl_mix_patch, 0.0, 1.0),
        "qwl_throw_weight": (float(qwl_throw_weight), map_qwl_throw_lg, map_qwl_throw_patch, 0.60, 1.80),
        "qwl_comeback_weight": (float(qwl_comeback_weight), map_qwl_comeback_lg, map_qwl_comeback_patch, 0.30, 1.20),
        "qwl_phase_dynamic_strength": (float(qwl_phase_dynamic_strength), map_qwl_phase_dyn_lg, map_qwl_phase_dyn_patch, 0.0, 0.80),
        "qwl_opponent_context_mix": (float(qwl_opponent_context_mix), map_qwl_opp_ctx_lg, map_qwl_opp_ctx_patch, 0.0, 1.0),
    }

    # Incremental is only safe for append-only updates. If a new unseen game is older than the
    # seed max_date, rebuild from scratch to preserve chronological state updates.
    if bool(seed_used) and bool(processed_gameids):
        _seed_meta = seed_artifact.get("meta") if isinstance(seed_artifact, dict) and isinstance(seed_artifact.get("meta"), dict) else {}
        _seed_max_date = str((_seed_meta or {}).get("max_date") or "").strip()
        _seed_max_dt = pd.to_datetime(_seed_max_date, errors="coerce") if _seed_max_date else pd.NaT
        if pd.isna(_seed_max_dt):
            states = {}
            processed_gameids = set()
            seed_used = False
            update_mode = "full_rebuild_seed_no_max_date"
        else:
            _unknown = games.loc[~games["gameid"].isin(processed_gameids), ["date"]].copy()
            if not _unknown.empty:
                _older = _unknown[_unknown["date"] < _seed_max_dt]
                if not _older.empty:
                    backfill_detected = True
                    backfill_unknown_games = len(_older)
                    backfill_oldest_date = _safe_iso_date(_older["date"].min())
                    if str(incremental_backfill_policy or "rebuild").strip().lower() == "rebuild":
                        states = {}
                        processed_gameids = set()
                        seed_used = False
                        update_mode = "full_rebuild_backfill_detected"
                    else:
                        update_mode = "incremental_backfill_unsafe"

    # Fast-path: when incremental seed is valid, process only unseen games.
    _seen_mask = games["gameid"].isin(processed_gameids) if processed_gameids else pd.Series(False, index=games.index)
    candidate_new_games = int((~_seen_mask).sum()) if len(games.index) else 0
    incremental_fastpath = bool(seed_used) and bool(processed_gameids)
    if incremental_fastpath:
        games_iter = games.loc[~_seen_mask].copy().reset_index(drop=True)
        games_skipped_seen = int(_seen_mask.sum())
    else:
        games_iter = games

    lineup_by_game: Dict[str, Tuple[Dict[str, str], Dict[str, str]]] = {}
    lineup_source_csvs_scanned: List[str] = []
    lineup_source_scan_mode = "none"
    if include_players_for_roster:
        try:
            gids = set(games_iter["gameid"].astype(str).tolist())
            if gids:
                # A-real v5 online fast-path: when only the daily CSV changed and the
                # state seed is valid, roster lineups can be sourced from the scanned
                # CSV subset only (usually just the new daily file), avoiding a full
                # multi-CSV player-row load on every append.
                lineup_csv_paths = list(csv_list)
                if bool(incremental_fastpath) and str(source_scan_mode) == "last_csv_only_incremental":
                    lineup_csv_paths = list(csv_list_scan)
                lineup_source_csvs_scanned = [str(x) for x in lineup_csv_paths]
                if not lineup_source_csvs_scanned:
                    lineup_source_scan_mode = "none"
                elif bool(incremental_fastpath) and str(source_scan_mode) == "last_csv_only_incremental":
                    lineup_source_scan_mode = "last_csv_only_incremental"
                else:
                    lineup_source_scan_mode = "all_csvs"
                pr = load_match_player_rows(lineup_csv_paths, keep_playoffs=keep_playoffs, gameids=gids)
                if pr is not None and not pr.empty:
                    for gid in games_iter["gameid"].astype(str).tolist():
                        lineup_by_game[str(gid)] = lineup_for_game(pr, str(gid))
        except Exception:
            lineup_by_game = {}
    games_used = 0
    patch_switches = 0
    roster_events = 0
    roster_events_with_change = 0
    patch_mu_recenter_events = 0
    roster_mu_recenter_events = 0
    league_switch_recenter_events = 0
    roster_hard_reset_events = 0
    quality_signal_used_games = 0
    quality_adjust_used_games = 0
    quality_adjust_surplus_games = 0
    quality_adjust_sum = 0.0
    quality_expected_sum = 0.0
    quality_surplus_sum = 0.0
    quality_tempo_used_games = 0
    quality_conversion_used_games = 0
    quality_throw_used_games = 0
    quality_comeback_used_games = 0
    quality_tempo_sum = 0.0
    quality_conversion_sum = 0.0
    quality_throw_sum = 0.0
    quality_comeback_sum = 0.0
    quality_early_sum = 0.0
    quality_objective_sum = 0.0
    quality_kill_sum = 0.0
    quality_data_reliability_sum = 0.0
    quality_data_partial_games = 0
    quality_data_unknown_games = 0
    quality_context_multiplier_sum = 0.0
    quality_context_boost_games = 0
    quality_context_cut_games = 0
    bayes_joint_updates = 0
    legacy_k_updates = 0
    bayes_joint_innov_abs_sum = 0.0
    bayes_joint_k_equiv_sum = 0.0
    games_skipped_empty_team = 0
    games_skipped_invalid_team_key = 0
    ll_sum = 0.0
    brier_sum = 0.0
    ll_n = 0
    by_league_eval: Dict[str, Dict[str, float]] = {}
    by_patch_eval: Dict[str, Dict[str, float]] = {}

    _journal_keep = int(max(0, int(online_journal_tail_keep)))
    for r in games_iter.itertuples(index=False):
        gid = str(r.gameid)
        if gid in processed_gameids:
            games_skipped_seen += 1
            continue
        blue_name = str(r.blue or "").strip()
        red_name = str(r.red or "").strip()
        if not blue_name or not red_name:
            games_skipped_empty_team += 1
            continue
        if not _norm_team_key(blue_name) or not _norm_team_key(red_name):
            games_skipped_invalid_team_key += 1
            continue
        try:
            y = 1.0 if int(r.blue_win) == 1 else 0.0
        except (TypeError, ValueError):
            continue
        dt_iso = _safe_iso_date(r.date)
        if not dt_iso:
            continue
        patch_k = _patch_key(getattr(r, "patch", ""))
        league = str(r.league or "").strip()
        split = str(r.split or "").strip()
        playoffs = bool(r.playoffs) if getattr(r, "playoffs", None) is not None else None
        _eff = _resolve_effective_params(_scoped_param_specs, str(league), str(patch_k))
        bayes_info_scale_eff = _eff["bayes_info_scale"]
        quality_k_boost_eff = _eff["quality_k_boost"]
        qwl_surplus_mix_eff = _eff["qwl_surplus_mix"]
        qwl_throw_weight_eff = _eff["qwl_throw_weight"]
        qwl_comeback_weight_eff = _eff["qwl_comeback_weight"]
        qwl_phase_dynamic_strength_eff = _eff["qwl_phase_dynamic_strength"]
        qwl_opponent_context_mix_eff = _eff["qwl_opponent_context_mix"]

        sb = _get_or_create_team(states, blue_name)
        sr = _get_or_create_team(states, red_name)
        league_prior_mu = _league_prior_mu(states, league)
        patch_sigma_mult = _league_patch_vol_multiplier(league)

        reg_b = _inflate_sigma_for_time_and_regime(
            sb,
            cur_date_iso=dt_iso,
            cur_patch_key=patch_k,
            inactivity_sigma_per_day=float(inactivity_sigma_per_day),
            patch_sigma_bump=float(patch_sigma_bump),
            patch_sigma_mult=float(patch_sigma_mult),
            max_sigma=float(max_sigma),
        )
        reg_r = _inflate_sigma_for_time_and_regime(
            sr,
            cur_date_iso=dt_iso,
            cur_patch_key=patch_k,
            inactivity_sigma_per_day=float(inactivity_sigma_per_day),
            patch_sigma_bump=float(patch_sigma_bump),
            patch_sigma_mult=float(patch_sigma_mult),
            max_sigma=float(max_sigma),
        )
        for _st, _reg in ((sb, reg_b), (sr, reg_r)):
            _ps, _pr = _apply_patch_switch_recenter(
                _st,
                patch_switched=bool((_reg or {}).get("patch_switched", False)),
                patch_mu_recenter=float(patch_mu_recenter),
                patch_sigma_mult=float(patch_sigma_mult),
                league_prior_mu=float(league_prior_mu),
            )
            patch_switches += _ps
            patch_mu_recenter_events += _pr

        for _st in (sb, sr):
            league_switch_recenter_events += _apply_league_switch_recenter(
                _st,
                league=league,
                league_switch_mu_recenter=float(league_switch_mu_recenter),
                league_prior_mu=float(league_prior_mu),
            )

        blue_lineup, red_lineup = lineup_by_game.get(gid, ({}, {}))
        ctb = _inflate_sigma_for_roster(sb, cur_lineup=blue_lineup, roster_sigma_bump=float(roster_sigma_bump), max_sigma=float(max_sigma))
        ctr = _inflate_sigma_for_roster(sr, cur_lineup=red_lineup, roster_sigma_bump=float(roster_sigma_bump), max_sigma=float(max_sigma))
        for _st, cc in ((sb, ctb), (sr, ctr)):
            _re, _rewc, _rhard, _rmu = _apply_roster_recenter(
                _st, cc,
                roster_mu_recenter=float(roster_mu_recenter),
                roster_cont_hard_reset=float(roster_cont_hard_reset),
                roster_hard_reset_extra_sigma=float(roster_hard_reset_extra_sigma),
                max_sigma=float(max_sigma),
                league_prior_mu=float(league_prior_mu),
            )
            roster_events += _re
            roster_events_with_change += _rewc
            roster_hard_reset_events += _rhard
            roster_mu_recenter_events += _rmu

        # Glicko-lite: opponent uncertainty modulates sensitivity.
        g_b, g_r, denom, p_blue = _glicko_lite_p_blue(mu_b=float(sb.mu), mu_r=float(sr.mu), sigma_b=float(sb.sigma), sigma_r=float(sr.sigma))
        err = float(y - p_blue)

        # Stronger leagues move state a bit more; high uncertainty also allows larger moves.
        k_eff = _effective_k(base_k=float(base_k), league=league, sigma_b=float(sb.sigma), sigma_r=float(sr.sigma), base_sigma=float(base_sigma))
        surprise = float(abs(err))
        q_blue, q_tempo_term, q_conversion_term, q_terms = _game_quality_blue(
            r,
            tempo_weight=float(qwl_tempo_weight),
            conversion_weight=float(qwl_conversion_weight),
            throw_weight=float(qwl_throw_weight_eff),
            comeback_weight=float(qwl_comeback_weight_eff),
            phase_dynamic_strength=float(qwl_phase_dynamic_strength_eff),
        )
        q_early_term = float(_f((q_terms or {}).get("q_early_term"), 0.0))
        q_objective_term = float(_f((q_terms or {}).get("q_objective_term"), 0.0))
        q_kill_term = float(_f((q_terms or {}).get("q_kill_term"), 0.0))
        q_throw_term = float(_f((q_terms or {}).get("q_throw_term"), 0.0))
        q_comeback_term = float(_f((q_terms or {}).get("q_comeback_term"), 0.0))
        q_lead_ref = float(_f((q_terms or {}).get("q_lead_ref"), 0.0))
        q_lead_source = str((q_terms or {}).get("q_lead_source") or "none")
        q_data_reliability = float(_clip(_f((q_terms or {}).get("q_data_reliability"), 1.0), 0.70, 1.00))
        q_data_blue_state = str((q_terms or {}).get("q_data_completeness_blue") or "unknown")
        q_data_red_state = str((q_terms or {}).get("q_data_completeness_red") or "unknown")
        q_objective_dragon_ref = float(_f((q_terms or {}).get("q_objective_dragon_ref"), 0.0))
        q_objective_dragon_source = str((q_terms or {}).get("q_objective_dragon_source") or "none")
        q_winner = float(q_blue if y >= 0.5 else -q_blue)
        q_winner_pos = float(max(0.0, min(1.0, q_winner)))
        p_winner = float(p_blue if y >= 0.5 else (1.0 - p_blue))
        q_adjusted, q_expected_winner, q_surplus, q_context_mult = _qwl_adjusted_signal(
            q_winner_pos,
            p_winner,
            expected_base=float(qwl_expected_base),
            expected_slope=float(qwl_expected_slope),
            surplus_mix=float(qwl_surplus_mix_eff),
            opponent_context_mix=float(qwl_opponent_context_mix_eff),
        )
        # Persist per-team signed QWL summaries for runtime use.
        qwl_signed_blue = float((1.0 if y >= 0.5 else -1.0) * float(q_adjusted))
        qwl_signed_red = float(-qwl_signed_blue)
        qwl_surplus_signed_blue = float((1.0 if y >= 0.5 else -1.0) * float(q_surplus))
        qwl_surplus_signed_red = float(-qwl_surplus_signed_blue)
        _update_team_qwl_stats(
            sb,
            qwl_signed=float(qwl_signed_blue),
            qwl_surplus_signed=float(qwl_surplus_signed_blue),
            qwl_tempo_signed=float(q_tempo_term),
            qwl_conversion_signed=float(q_conversion_term),
            qwl_throw_signed=float(q_throw_term),
            qwl_comeback_signed=float(q_comeback_term),
            qwl_data_reliability=float(q_data_reliability),
        )
        _update_team_qwl_stats(
            sr,
            qwl_signed=float(qwl_signed_red),
            qwl_surplus_signed=float(qwl_surplus_signed_red),
            qwl_tempo_signed=float(-q_tempo_term),
            qwl_conversion_signed=float(-q_conversion_term),
            qwl_throw_signed=float(-q_throw_term),
            qwl_comeback_signed=float(-q_comeback_term),
            qwl_data_reliability=float(q_data_reliability),
        )
        if q_winner_pos > 0.0:
            quality_signal_used_games += 1
        if q_adjusted > 0.0:
            quality_adjust_used_games += 1
        if q_surplus > 0.0:
            quality_adjust_surplus_games += 1
        quality_adjust_sum += float(q_adjusted)
        quality_expected_sum += float(q_expected_winner)
        quality_surplus_sum += float(q_surplus)
        quality_context_multiplier_sum += float(q_context_mult)
        if float(q_context_mult) > 1.000001:
            quality_context_boost_games += 1
        elif float(q_context_mult) < 0.999999:
            quality_context_cut_games += 1
        if abs(float(q_tempo_term)) > 1e-12:
            quality_tempo_used_games += 1
        if abs(float(q_conversion_term)) > 1e-12:
            quality_conversion_used_games += 1
        if abs(float(q_throw_term)) > 1e-12:
            quality_throw_used_games += 1
        if abs(float(q_comeback_term)) > 1e-12:
            quality_comeback_used_games += 1
        quality_tempo_sum += float(q_tempo_term)
        quality_conversion_sum += float(q_conversion_term)
        quality_throw_sum += float(q_throw_term)
        quality_comeback_sum += float(q_comeback_term)
        quality_early_sum += float(q_early_term)
        quality_objective_sum += float(q_objective_term)
        quality_kill_sum += float(q_kill_term)
        quality_data_reliability_sum += float(q_data_reliability)
        if q_data_blue_state == "partial" or q_data_red_state == "partial":
            quality_data_partial_games += 1
        if q_data_blue_state == "unknown" or q_data_red_state == "unknown":
            quality_data_unknown_games += 1
        # Quality of result: decisive wins/losses should update state a bit more than coin-flip wins.
        qk = float(max(0.0, min(0.55, float(quality_k_boost_eff))))
        mu_mode = str(mu_update_mode or "legacy_k_v1").strip().lower()

        # Fisher information proxy for Bernoulli-logit game observation.
        info_base = float(max(0.0, p_blue * (1.0 - p_blue)))
        info_b = float((g_r ** 2) * info_base)
        info_r = float((g_b ** 2) * info_base)
        delta_b = 0.0
        delta_r = 0.0

        if mu_mode == "bayes_joint_v1":
            _sigma_mode = str(sigma_update_mode or "legacy_decay_v1").strip().lower()
            _joint_updates_sigma = bool(_sigma_mode == "bayes_logit_v1")
            upd = _update_bayes_joint_after_game(
                sb,
                sr,
                y_blue=float(y),
                p_blue=float(p_blue),
                g_blue=float(g_b),
                g_red=float(g_r),
                denom=float(denom),
                min_sigma=float(min_sigma),
                max_sigma=float(max_sigma),
                bayes_info_scale=float(bayes_info_scale_eff),
                surprise=float(surprise),
                q_adjusted=float(q_adjusted),
                quality_k_boost_eff=float(qk * max(0.0, float(bayes_mu_q_strength))),
                update_sigma=bool(_joint_updates_sigma),
            )
            delta_b = float(upd.get("delta_blue") or 0.0)
            # delta_red in updater is already signed update for red team.
            delta_r = float(-(upd.get("delta_red") or 0.0))
            k_eff = float(max(0.0, upd.get("k_equiv") or 0.0))
            bayes_joint_updates += 1
            bayes_joint_innov_abs_sum += float(abs(upd.get("innov") or 0.0))
            bayes_joint_k_equiv_sum += float(k_eff)
            if not bool(_joint_updates_sigma):
                _update_sigma_after_game(
                    sb,
                    base_sigma=float(base_sigma),
                    sigma_decay=float(sigma_decay),
                    min_sigma=float(min_sigma),
                    max_sigma=float(max_sigma),
                    surprise=surprise,
                    sigma_update_mode=str(sigma_update_mode),
                    info_term=float(info_b),
                    bayes_info_scale=float(bayes_info_scale_eff),
                )
                _update_sigma_after_game(
                    sr,
                    base_sigma=float(base_sigma),
                    sigma_decay=float(sigma_decay),
                    min_sigma=float(min_sigma),
                    max_sigma=float(max_sigma),
                    surprise=surprise,
                    sigma_update_mode=str(sigma_update_mode),
                    info_term=float(info_r),
                    bayes_info_scale=float(bayes_info_scale_eff),
                )
        else:
            k_eff = float(max(0.04, min(0.62, k_eff * (1.0 + 0.35 * surprise))))
            k_eff = float(max(0.04, min(0.68, k_eff * (1.0 + (qk * q_adjusted)))))
            delta_b = float(k_eff * g_r * err)
            delta_r = float(k_eff * g_b * err)
            sb.mu = float(sb.mu + delta_b)
            sr.mu = float(sr.mu - delta_r)
            _update_sigma_after_game(
                sb,
                base_sigma=float(base_sigma),
                sigma_decay=float(sigma_decay),
                min_sigma=float(min_sigma),
                max_sigma=float(max_sigma),
                surprise=surprise,
                sigma_update_mode=str(sigma_update_mode),
                info_term=float(info_b),
                bayes_info_scale=float(bayes_info_scale_eff),
            )
            _update_sigma_after_game(
                sr,
                base_sigma=float(base_sigma),
                sigma_decay=float(sigma_decay),
                min_sigma=float(min_sigma),
                max_sigma=float(max_sigma),
                surprise=surprise,
                sigma_update_mode=str(sigma_update_mode),
                info_term=float(info_r),
                bayes_info_scale=float(bayes_info_scale_eff),
            )
            legacy_k_updates += 1

        if bool(eval_track_metrics):
            p_eval = float(max(1e-6, min(1.0 - 1e-6, p_blue)))
            y_eval = float(y)
            ll = float(-(y_eval * math.log(p_eval) + (1.0 - y_eval) * math.log(1.0 - p_eval)))
            br = float((p_eval - y_eval) ** 2)
            ll_sum += ll
            brier_sum += br
            ll_n += 1
            _lgk = _norm_upper_key(league) or "UNK"
            _ptk = _norm_upper_key(patch_k) or "UNK"
            d_l = by_league_eval.setdefault(_lgk, {"n": 0.0, "ll": 0.0, "brier": 0.0})
            d_l["n"] += 1.0
            d_l["ll"] += ll
            d_l["brier"] += br
            d_p = by_patch_eval.setdefault(_ptk, {"n": 0.0, "ll": 0.0, "brier": 0.0})
            d_p["n"] += 1.0
            d_p["ll"] += ll
            d_p["brier"] += br

        sb.gp = int(sb.gp) + 1
        sr.gp = int(sr.gp) + 1
        if y >= 0.5:
            sb.wins += 1
            sr.losses += 1
        else:
            sb.losses += 1
            sr.wins += 1
        for st, lu, tname in ((sb, blue_lineup, blue_name), (sr, red_lineup, red_name)):
            st.team = str(tname or st.team)
            st.last_date = dt_iso
            st.last_patch = str(patch_k or st.last_patch or "")
            st.last_league = str(league or st.last_league or "")
            st.last_split = str(split or st.last_split or "")
            st.last_playoffs = playoffs
            if isinstance(lu, dict) and lu:
                st.last_lineup = {str(k): str(v) for k, v in lu.items() if str(v or "").strip()}

        games_used += 1
        processed_gameids.add(gid)
        last_processed_game = {
            "gameid": str(gid),
            "date": str(dt_iso),
            "league": str(league or ""),
            "patch": str(patch_k or ""),
            "blue": str(blue_name or ""),
            "red": str(red_name or ""),
            "y_blue": (1 if y >= 0.5 else 0),
        }
        if _journal_keep > 0:
            online_journal_tail.append(
                {
                    "gameid": str(gid),
                    "date": str(dt_iso),
                    "league": str(league or ""),
                    "patch": str(patch_k or ""),
                    "blue": str(blue_name or ""),
                    "red": str(red_name or ""),
                    "p_blue_pre": float(p_blue),
                    "y_blue": (1 if y >= 0.5 else 0),
                    "k_eff": float(k_eff),
                    "surprise": float(surprise),
                    "q_blue": float(q_blue),
                    "q_tempo_term": float(q_tempo_term),
                    "q_conversion_term": float(q_conversion_term),
                    "q_early_term": float(q_early_term),
                    "q_objective_term": float(q_objective_term),
                    "q_kill_term": float(q_kill_term),
                    "q_throw_term": float(q_throw_term),
                    "q_comeback_term": float(q_comeback_term),
                    "q_lead_ref": float(q_lead_ref),
                    "q_lead_source": str(q_lead_source),
                    "q_data_reliability": float(q_data_reliability),
                    "q_data_completeness_blue": str(q_data_blue_state),
                    "q_data_completeness_red": str(q_data_red_state),
                    "q_objective_dragon_ref": float(q_objective_dragon_ref),
                    "q_objective_dragon_source": str(q_objective_dragon_source),
                    "q_winner": float(q_winner),
                    "q_winner_pos": float(q_winner_pos),
                    "q_adjusted": float(q_adjusted),
                    "q_expected_winner": float(q_expected_winner),
                    "q_surplus": float(q_surplus),
                    "p_winner_pre": float(p_winner),
                    "qwl_surplus_mix_eff": float(qwl_surplus_mix_eff),
                    "qwl_tempo_weight_eff": float(qwl_tempo_weight),
                    "qwl_conversion_weight_eff": float(qwl_conversion_weight),
                    "qwl_throw_weight_eff": float(qwl_throw_weight_eff),
                    "qwl_comeback_weight_eff": float(qwl_comeback_weight_eff),
                    "qwl_phase_dynamic_strength_eff": float(qwl_phase_dynamic_strength_eff),
                    "qwl_opponent_context_mix_eff": float(qwl_opponent_context_mix_eff),
                    "q_context_multiplier": float(q_context_mult),
                    "quality_k_boost_eff": float(quality_k_boost_eff),
                    "bayes_info_scale_eff": float(bayes_info_scale_eff),
                    "mu_update_mode_eff": str(mu_mode),
                    "bayes_mu_q_strength_eff": float(max(0.0, float(bayes_mu_q_strength))),
                    "delta_blue": float(delta_b),
                    "delta_red": float(delta_r),
                    "blue_mu_post": float(sb.mu),
                    "red_mu_post": float(sr.mu),
                    "blue_sigma_post": float(sb.sigma),
                    "red_sigma_post": float(sr.sigma),
                }
            )
            if len(online_journal_tail) > _journal_keep:
                del online_journal_tail[: len(online_journal_tail) - _journal_keep]

    teams_out = {k: st.to_dict() for k, st in states.items()}
    by_league: Dict[str, Dict[str, Any]] = {}
    for st in states.values():
        lg = str(st.last_league or "")
        if not lg:
            continue
        d = by_league.setdefault(lg, {"teams": 0, "games": 0})
        d["teams"] += 1
        d["games"] += int(st.gp or 0)

    eval_metrics: Dict[str, Any] = {}
    if bool(eval_track_metrics) and ll_n > 0:
        eval_metrics = {
            "n": int(ll_n),
            "logloss": float(ll_sum / ll_n),
            "brier": float(brier_sum / ll_n),
            "by_league": {
                k: {
                    "n": int(v["n"]),
                    "logloss": float(v["ll"] / v["n"]) if v["n"] > 0 else None,
                    "brier": float(v["brier"] / v["n"]) if v["n"] > 0 else None,
                }
                for k, v in by_league_eval.items()
            },
            "by_patch": {
                k: {
                    "n": int(v["n"]),
                    "logloss": float(v["ll"] / v["n"]) if v["n"] > 0 else None,
                    "brier": float(v["brier"] / v["n"]) if v["n"] > 0 else None,
                }
                for k, v in by_patch_eval.items()
            },
        }

    return {
        "version": "team_state_artifact_v9",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "params": {
            "base_k": float(base_k),
            "base_sigma": float(base_sigma),
            "min_sigma": float(min_sigma),
            "max_sigma": float(max_sigma),
            "sigma_decay": float(sigma_decay),
            "inactivity_sigma_per_day": float(inactivity_sigma_per_day),
            "patch_sigma_bump": float(patch_sigma_bump),
            "roster_sigma_bump": float(roster_sigma_bump),
            "patch_mu_recenter": float(patch_mu_recenter),
            "roster_mu_recenter": float(roster_mu_recenter),
            "league_switch_mu_recenter": float(league_switch_mu_recenter),
            "roster_cont_hard_reset": float(roster_cont_hard_reset),
            "roster_hard_reset_extra_sigma": float(roster_hard_reset_extra_sigma),
            "quality_k_boost": float(quality_k_boost),
            "qwl_expected_base": float(qwl_expected_base),
            "qwl_expected_slope": float(qwl_expected_slope),
            "qwl_surplus_mix": float(qwl_surplus_mix),
            "qwl_tempo_weight": float(qwl_tempo_weight),
            "qwl_conversion_weight": float(qwl_conversion_weight),
            "qwl_throw_weight": float(qwl_throw_weight),
            "qwl_comeback_weight": float(qwl_comeback_weight),
            "qwl_phase_dynamic_strength": float(qwl_phase_dynamic_strength),
            "qwl_opponent_context_mix": float(qwl_opponent_context_mix),
            "mu_update_mode": str(mu_update_mode or "legacy_k_v1"),
            "sigma_update_mode": str(sigma_update_mode or "legacy_decay_v1"),
            "bayes_info_scale": float(bayes_info_scale),
            "bayes_mu_q_strength": float(max(0.0, float(bayes_mu_q_strength))),
            "bayes_info_scale_by_league": [list(x) for x in _param_map_sig(map_bayes_lg)],
            "bayes_info_scale_by_patch": [list(x) for x in _param_map_sig(map_bayes_patch)],
            "quality_k_boost_by_league": [list(x) for x in _param_map_sig(map_quality_lg)],
            "quality_k_boost_by_patch": [list(x) for x in _param_map_sig(map_quality_patch)],
            "qwl_surplus_mix_by_league": [list(x) for x in _param_map_sig(map_qwl_mix_lg)],
            "qwl_surplus_mix_by_patch": [list(x) for x in _param_map_sig(map_qwl_mix_patch)],
            "qwl_throw_weight_by_league": [list(x) for x in _param_map_sig(map_qwl_throw_lg)],
            "qwl_throw_weight_by_patch": [list(x) for x in _param_map_sig(map_qwl_throw_patch)],
            "qwl_comeback_weight_by_league": [list(x) for x in _param_map_sig(map_qwl_comeback_lg)],
            "qwl_comeback_weight_by_patch": [list(x) for x in _param_map_sig(map_qwl_comeback_patch)],
            "qwl_phase_dynamic_strength_by_league": [list(x) for x in _param_map_sig(map_qwl_phase_dyn_lg)],
            "qwl_phase_dynamic_strength_by_patch": [list(x) for x in _param_map_sig(map_qwl_phase_dyn_patch)],
            "qwl_opponent_context_mix_by_league": [list(x) for x in _param_map_sig(map_qwl_opp_ctx_lg)],
            "qwl_opponent_context_mix_by_patch": [list(x) for x in _param_map_sig(map_qwl_opp_ctx_patch)],
            "include_players_for_roster": bool(include_players_for_roster),
        },
        "meta": {
            "csv_count": len(csv_list),
            "csv_scanned_count": len(csv_list_scan),
            "games_total_loaded": len(games),
            "candidate_new_games": int(candidate_new_games),
            "processing_games": len(games_iter),
            "games_used": int(games_used),
            "games_skipped_seen": int(games_skipped_seen),
            "games_skipped_empty_team": int(games_skipped_empty_team),
            "games_skipped_invalid_team_key": int(games_skipped_invalid_team_key),
            "teams": len(states),
            "seed_used": bool(seed_used),
            "incremental_fastpath": bool(incremental_fastpath),
            "seed_processed_count": int(seed_processed_count),
            "seed_teams_count": int(seed_teams_count),
            "patch_switch_events": int(patch_switches),
            "roster_events": int(roster_events),
            "roster_events_with_change": int(roster_events_with_change),
            "patch_mu_recenter_events": int(patch_mu_recenter_events),
            "roster_mu_recenter_events": int(roster_mu_recenter_events),
            "league_switch_recenter_events": int(league_switch_recenter_events),
            "roster_hard_reset_events": int(roster_hard_reset_events),
            "quality_signal_used_games": int(quality_signal_used_games),
            "quality_adjust_used_games": int(quality_adjust_used_games),
            "quality_adjust_surplus_games": int(quality_adjust_surplus_games),
            "quality_adjust_avg": (float(quality_adjust_sum / max(1, games_used)) if games_used > 0 else None),
            "quality_expected_avg": (float(quality_expected_sum / max(1, games_used)) if games_used > 0 else None),
            "quality_surplus_avg": (float(quality_surplus_sum / max(1, games_used)) if games_used > 0 else None),
            "quality_tempo_used_games": int(quality_tempo_used_games),
            "quality_conversion_used_games": int(quality_conversion_used_games),
            "quality_throw_used_games": int(quality_throw_used_games),
            "quality_comeback_used_games": int(quality_comeback_used_games),
            "quality_tempo_avg": (float(quality_tempo_sum / max(1, games_used)) if games_used > 0 else None),
            "quality_conversion_avg": (float(quality_conversion_sum / max(1, games_used)) if games_used > 0 else None),
            "quality_throw_avg": (float(quality_throw_sum / max(1, games_used)) if games_used > 0 else None),
            "quality_comeback_avg": (float(quality_comeback_sum / max(1, games_used)) if games_used > 0 else None),
            "quality_early_avg": (float(quality_early_sum / max(1, games_used)) if games_used > 0 else None),
            "quality_objective_avg": (float(quality_objective_sum / max(1, games_used)) if games_used > 0 else None),
            "quality_kill_avg": (float(quality_kill_sum / max(1, games_used)) if games_used > 0 else None),
            "quality_data_reliability_avg": (float(quality_data_reliability_sum / max(1, games_used)) if games_used > 0 else None),
            "quality_data_partial_games": int(quality_data_partial_games),
            "quality_data_unknown_games": int(quality_data_unknown_games),
            "quality_context_multiplier_avg": (float(quality_context_multiplier_sum / max(1, games_used)) if games_used > 0 else None),
            "quality_context_boost_games": int(quality_context_boost_games),
            "quality_context_cut_games": int(quality_context_cut_games),
            "bayes_joint_updates": int(bayes_joint_updates),
            "legacy_k_updates": int(legacy_k_updates),
            "bayes_joint_innov_abs_avg": (float(bayes_joint_innov_abs_sum / max(1, bayes_joint_updates)) if bayes_joint_updates > 0 else None),
            "bayes_joint_k_equiv_avg": (float(bayes_joint_k_equiv_sum / max(1, bayes_joint_updates)) if bayes_joint_updates > 0 else None),
            "update_mode": str(update_mode),
            "seed_generated_at": (str(seed_generated_at) if seed_generated_at else None),
            "backfill_detected": bool(backfill_detected),
            "backfill_unknown_games": int(backfill_unknown_games),
            "backfill_oldest_date": (str(backfill_oldest_date) if backfill_oldest_date else None),
            "source_scan_mode": str(source_scan_mode),
            "source_changed_count": int(source_changed_count),
            "source_changed_paths": list(source_changed_paths),
            "source_files": source_files_cur,
            "source_files_scanned": [str(p) for p in csv_list_scan],
            "lineup_source_csvs_scanned": list(lineup_source_csvs_scanned),
            "lineup_source_scan_mode": str(lineup_source_scan_mode),
            "lineup_csv_scanned_count": len(lineup_source_csvs_scanned),
            "last_processed_game": (dict(last_processed_game) if isinstance(last_processed_game, dict) else None),
            "online_journal_tail_count": len(online_journal_tail),
            "min_date": (_safe_iso_date(games["date"].min()) if not games.empty else None),
            "max_date": (_safe_iso_date(games["date"].max()) if not games.empty else None),
            "max_games_eval": (int(max_games_eval) if isinstance(max_games_eval, int) and int(max_games_eval) > 0 else None),
            "eval_track_metrics": bool(eval_track_metrics),
            "eval_metrics": eval_metrics,
        },
        "online_journal_tail": list(online_journal_tail),
        "by_league": by_league,
        "processed_gameids": (sorted(processed_gameids) if bool(store_processed_gameids) else []),
        "teams": teams_out,
    }





