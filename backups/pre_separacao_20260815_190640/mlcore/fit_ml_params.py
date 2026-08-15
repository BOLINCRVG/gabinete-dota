from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .artifact import MLArtifact
from .calibration_schedule import build_patch_snapshot
from .io_oracle import load_oracle_elixir_games
from .players_layer import PlayersArtifact, load_match_player_rows

# ── sub-modules (implementations moved here) ──────────────────────────────────
from .fit_io_core import (
    _build_fit_run_signature,
    _path_signature,  # noqa: F401 - re-exported for tools/ + tests
    _read_json_file_safe,
    _utc_now_iso,
    _write_json_file_safe,
)
from .fit_stages_core import (
    FIT_STAGE_ORDER,
    _fit_stage_index,
    _load_stage_checkpoint,
    _normalize_fit_stage,
    _prime_seed_stage_checkpoints_from_params,
    _print_fit_stage,
    _save_stage_checkpoint,
    _seed_stage_payload_from_params,  # noqa: F401 - re-exported for tests
    _write_fit_progress,
)
from .fit_samples_core import (
    Sample,  # noqa: F401 - re-exported for tools/ + tests
    _build_lineup_cache,
    _build_microseg_table_from_pred_rows,
    _fit_fused_post_calibration_from_pred_rows,
    _fuse_probs_by_precision,  # noqa: F401 - re-exported for tools/ + calibration_diagnostics
    _is_valid_areal_prob_tuning,
    _is_valid_ml_marginal_band_coeffs,
    _league_macro_key,
    _load_areal_prob_tuning_snapshot,
    _load_ml_marginal_band_coeffs_snapshot,
    _load_team_games_multi,
    _make_samples,
    _sanitize_ml_marginal_band_coeffs,
    _sym_team_probs,  # noqa: F401 - re-exported for tools/
)
from .fit_form_core import (
    _evaluate_walkforward,
    _fit_form_grid,
    _form_settings_from_best,
    _meta_recent_evidence_from_sample,  # noqa: F401 - re-exported for backtest tools
)
from .fit_form_meta_core import _fit_meta_blend_c_model
from .fit_form_style_core import _fit_style_matchup_b_model
from .fit_players_core import (
    _blind_backtest_by_league,
    _fit_players_context_tuning,
    _resolve_players_context_budget_profile,
)
from .fit_players_weights_core import _fit_players_lane_weights, _fit_players_league_weights
from .fit_series_core import (
    _fit_series_bo_dplite_tuning,
    _fit_series_gamma,
)
def fit_params(
    *,
    artifact_path: str,
    players_artifact_path: str,
    csv_paths: List[str],
    min_train_games: int,
    speed_profile: str = "full",
    progress_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    resume: bool = False,
    run_label: Optional[str] = None,
    start_stage: Optional[str] = None,
    base_params_path: Optional[str] = None,
) -> Dict[str, Any]:
    _speed = str(speed_profile or "full").strip().lower()
    if _speed not in {"full", "fast"}:
        _speed = "full"
    _players_context_budget = _resolve_players_context_budget_profile(_speed, run_label)
    progress_target = Path(str(progress_path)).expanduser().resolve() if str(progress_path or "").strip() else None
    checkpoint_root = Path(str(checkpoint_dir)).expanduser().resolve() if str(checkpoint_dir or "").strip() else None
    start_stage_key = _normalize_fit_stage(start_stage)
    base_params_target = Path(str(base_params_path)).expanduser().resolve() if str(base_params_path or "").strip() else None
    base_params_seed = _read_json_file_safe(base_params_target) if base_params_target is not None else {}
    run_signature = _build_fit_run_signature(
        artifact_path=str(artifact_path),
        players_artifact_path=str(players_artifact_path),
        csv_paths=[str(x) for x in (csv_paths or [])],
        min_train_games=int(min_train_games),
        speed_profile=str(_speed),
        run_label=str(run_label or ""),
        start_stage=str(start_stage_key),
        base_params_path=(str(base_params_target) if base_params_target is not None else ""),
    )
    stage_order = list(FIT_STAGE_ORDER)
    completed_stages: List[str] = []
    fit_started_at_iso = _utc_now_iso()
    fit_started_perf = time.perf_counter()
    stage_started_perf: Dict[str, float] = {}
    if _fit_stage_index(start_stage_key) > _fit_stage_index("load_inputs") and checkpoint_root is None:
        return {
            "error": "start_stage_requires_checkpoint_dir",
            "start_stage": str(start_stage_key),
        }
    if _fit_stage_index(start_stage_key) > _fit_stage_index("player_weights") and not base_params_seed:
        return {
            "error": "start_stage_requires_base_params",
            "start_stage": str(start_stage_key),
            "base_params_path": (str(base_params_target) if base_params_target is not None else ""),
        }
    if base_params_seed and _fit_stage_index(start_stage_key) > _fit_stage_index("player_weights"):
        seed_rep = _prime_seed_stage_checkpoints_from_params(
            checkpoint_root,
            run_signature=run_signature,
            base_params=base_params_seed,
            start_stage=str(start_stage_key),
            base_params_path=base_params_target,
        )
        if list(seed_rep.get("missing") or []):
            return {
                "error": "start_stage_missing_seed_payload",
                "start_stage": str(start_stage_key),
                "missing_stages": list(seed_rep.get("missing") or []),
                "base_params_path": (str(base_params_target) if base_params_target is not None else ""),
            }

    def _stage_start(stage: str, detail: Dict[str, Any] | None = None) -> None:
        stage_started_perf[str(stage)] = time.perf_counter()
        _print_fit_stage(stage, "running", detail)
        _write_fit_progress(
            progress_target,
            stage=stage,
            status="running",
            run_signature=run_signature,
            total_stages=len(stage_order),
            completed_stages=completed_stages,
            detail=detail,
            started_at=fit_started_at_iso,
            elapsed_s=float(max(0.0, time.perf_counter() - fit_started_perf)),
        )

    def _stage_done(stage: str, detail: Dict[str, Any] | None = None) -> None:
        if str(stage) not in completed_stages:
            completed_stages.append(str(stage))
        _print_fit_stage(stage, "completed", detail)
        _write_fit_progress(
            progress_target,
            stage=stage,
            status="completed",
            run_signature=run_signature,
            total_stages=len(stage_order),
            completed_stages=completed_stages,
            detail={**(detail or {}), "stage_elapsed_s": float(max(0.0, time.perf_counter() - stage_started_perf.get(str(stage), fit_started_perf)))},
            started_at=fit_started_at_iso,
            elapsed_s=float(max(0.0, time.perf_counter() - fit_started_perf)),
        )

    def _stage_pulse(stage: str, detail: Dict[str, Any] | None = None) -> None:
        _print_fit_stage(stage, "running", detail)
        _write_fit_progress(
            progress_target,
            stage=stage,
            status="running",
            run_signature=run_signature,
            total_stages=len(stage_order),
            completed_stages=completed_stages,
            detail={**(detail or {}), "stage_elapsed_s": float(max(0.0, time.perf_counter() - stage_started_perf.get(str(stage), fit_started_perf)))},
            started_at=fit_started_at_iso,
            elapsed_s=float(max(0.0, time.perf_counter() - fit_started_perf)),
        )

    def _stage_from_checkpoint(stage: str) -> Dict[str, Any]:
        if not bool(resume):
            return {}
        obj = _load_stage_checkpoint(checkpoint_root, stage, run_signature=run_signature)
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        if not payload:
            return {}
        detail = dict(obj.get("summary") or {})
        detail["from_checkpoint"] = True
        _stage_done(stage, detail)
        return dict(payload)

    def _checkpoint_payload(stage: str) -> Dict[str, Any]:
        if not bool(resume):
            return {}
        obj = _load_stage_checkpoint(checkpoint_root, stage, run_signature=run_signature)
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        return dict(payload or {})

    def _checkpoint_save(stage: str, payload: Dict[str, Any], detail: Dict[str, Any] | None = None) -> None:
        _save_stage_checkpoint(
            checkpoint_root,
            stage,
            run_signature=run_signature,
            payload=payload,
            summary=detail or {},
        )

    def _stage_save(stage: str, payload: Dict[str, Any], detail: Dict[str, Any] | None = None) -> None:
        _save_stage_checkpoint(
            checkpoint_root,
            stage,
            run_signature=run_signature,
            payload=payload,
            summary=detail or {},
        )
        _stage_done(stage, detail)

    _stage_start(
        "load_inputs",
        {
            "speed_profile": str(_speed),
            "players_context_budget": str(_players_context_budget),
            "resume": bool(resume),
            "start_stage": str(start_stage_key),
            "seeded_base_params": bool(base_params_seed),
        },
    )
    art = MLArtifact.load_json(str(artifact_path))
    p_art = PlayersArtifact.load_json(str(players_artifact_path))
    w_lineup = float((((art.meta or {}).get("players_layer") or {}).get("w_lineup") or 0.85))

    g_parts = []
    for p in csv_paths:
        d = load_oracle_elixir_games(p, keep_playoffs=None)
        if d is not None and not d.empty:
            g_parts.append(d)
    if not g_parts:
        return {"error": "no_games"}
    # Guard against pandas FutureWarning on concat with empty/all-NA frames.
    g_frames: List[pd.DataFrame] = []
    for _df in g_parts:
        try:
            if _df is None or _df.empty:
                continue
            _x = _df.dropna(axis=1, how="all")
            if _x.empty or _x.shape[1] == 0:
                continue
            g_frames.append(_x)
        except Exception:
            continue
    if not g_frames:
        return {"error": "no_games"}

    games = pd.concat(g_frames, ignore_index=True, sort=False)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.dropna(subset=["date"]).copy()
    games = (
        games.sort_values(["date", "gameid"])
        .drop_duplicates(subset=["gameid"], keep="last")
        .reset_index(drop=True)
    )
    patch_snapshot = build_patch_snapshot(games)

    gameids = set(games["gameid"].astype(str).tolist())
    pr = load_match_player_rows(csv_paths, keep_playoffs=None, gameids=gameids)
    lineup_cache = _build_lineup_cache(pr)
    samples = _make_samples(art=art, p_art=p_art, games=games, lineup_cache=lineup_cache, w_lineup=w_lineup)
    if not samples:
        return {"error": "no_samples"}
    if _speed == "fast":
        # Keep recent samples to reduce runtime while preserving current-meta signal.
        samples = samples[-2500:] if len(samples) > 2500 else samples
        min_train_games = int(max(120, min(int(min_train_games), 700)))
    _stage_done(
        "load_inputs",
        {
            "games_total": len(games),
            "samples_total": len(samples),
            "player_rows_total": len(pr),
            "start_stage": str(start_stage_key),
        },
    )
    _pw_cached = _stage_from_checkpoint("player_weights")
    if _pw_cached:
        players_lane_weights = dict(_pw_cached.get("players_lane_weights") or {})
        players_league_weights = dict(_pw_cached.get("players_league_weights") or {})
    else:
        _stage_start("player_weights")
        players_lane_weights = _fit_players_lane_weights(samples, min_n=max(120, int(min_train_games * 0.15)))
        players_league_weights = _fit_players_league_weights(samples, min_n=max(140, int(min_train_games * 0.20)))
        _stage_save(
            "player_weights",
            {
                "players_lane_weights": players_lane_weights,
                "players_league_weights": players_league_weights,
            },
            {
                "lane_global_keys": len((players_lane_weights.get("global") if isinstance(players_lane_weights, dict) else {}) or {}),
                "league_weight_keys": len((players_league_weights.get("by_league") if isinstance(players_league_weights, dict) else {}) or {}),
            },
        )

    if _speed == "fast":
        grid_team = [0.5, 1.0, 2.0]
        grid_players = [0.5, 1.0, 2.0]
    else:
        grid_team = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        grid_players = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    _gs_cached = _stage_from_checkpoint("global_scales")
    if _gs_cached:
        baseline = dict(_gs_cached.get("baseline") or {})
        best = dict(_gs_cached.get("best") or {})
        grid_rows = list(_gs_cached.get("grid_rows") or [])
        best_report = dict(_gs_cached.get("best_report") or {})
    else:
        _stage_start("global_scales")
        baseline = _evaluate_walkforward(
            samples,
            team_scale=1.0,
            players_scale=1.0,
            min_train_games=min_train_games,
            form_cfg={"enabled": False},
        )
        best = {"team_scale": 1.0, "players_scale": 1.0, "logloss": float("inf"), "brier": float("inf")}
        grid_rows: List[Dict[str, Any]] = []
        combo_total = int(len(grid_team) * len(grid_players))
        combo_idx = 0
        for ts in grid_team:
            for ps in grid_players:
                combo_idx += 1
                _stage_pulse(
                    "global_scales",
                    {
                        "combo": int(combo_idx),
                        "combo_total": int(combo_total),
                        "team_scale": float(ts),
                        "players_scale": float(ps),
                    },
                )
                rep = _evaluate_walkforward(
                    samples,
                    team_scale=float(ts),
                    players_scale=float(ps),
                    min_train_games=min_train_games,
                    form_cfg={"enabled": False},
                )
                ll = rep.get("logloss")
                br = rep.get("brier")
                if ll is None:
                    continue
                row = {
                    "team_scale": float(ts),
                    "players_scale": float(ps),
                    "n_total": int(rep.get("n_total") or 0),
                    "logloss": float(ll),
                    "brier": float(br) if br is not None else None,
                    "folds": len(rep.get("folds") or []),
                }
                grid_rows.append(row)
                if float(ll) < float(best["logloss"]):
                    best = dict(row)
        if not grid_rows:
            return {"error": "no_valid_folds"}
        best_report = _evaluate_walkforward(
            samples,
            team_scale=float(best["team_scale"]),
            players_scale=float(best["players_scale"]),
            min_train_games=min_train_games,
            form_cfg={"enabled": False},
        )
        _stage_save(
            "global_scales",
            {
                "baseline": baseline,
                "best": best,
                "grid_rows": grid_rows,
                "best_report": best_report,
            },
            {
                "grid_rows": len(grid_rows),
                "best_team_scale": float(best.get("team_scale", 1.0) or 1.0),
                "best_players_scale": float(best.get("players_scale", 1.0) or 1.0),
                "best_logloss": float(best.get("logloss", float("inf"))),
            },
        )

    _fc_cached = _stage_from_checkpoint("form_combined")
    if _fc_cached:
        form_fit = dict(_fc_cached.get("form_fit") or {})
        best_form = dict(_fc_cached.get("best_form") or {})
        best_form_settings = dict(_fc_cached.get("best_form_settings") or {})
        combined_report = dict(_fc_cached.get("combined_report") or {})
        fused_post_calibration = dict(_fc_cached.get("fused_post_calibration") or {})
        microseg_table = dict(_fc_cached.get("microseg_table") or {})
        _microseg_rows = list(combined_report.get("pred_rows") or []) if isinstance(combined_report, dict) else []
    else:
        _stage_start("form_combined")
        form_fit = _fit_form_grid(
            samples,
            team_scale=float(best["team_scale"]),
            players_scale=float(best["players_scale"]),
            min_train_games=min_train_games,
        )
        best_form = dict(form_fit.get("best_form") or {})
        best_form_settings = _form_settings_from_best(best_form) if best_form else {}
        combined_report = _evaluate_walkforward(
            samples,
            team_scale=float(best["team_scale"]),
            players_scale=float(best["players_scale"]),
            min_train_games=min_train_games,
            form_cfg={
                "enabled": bool(best_form.get("enabled", False)),
                "max_weight": float(best_form.get("max_weight", 0.75) or 0.75),
                "knee_games": float(best_form.get("knee_games", 6.0) or 6.0),
                "beta_prior": float(best_form.get("beta_prior", 2.0) or 2.0),
                "signal_power": float(best_form.get("signal_power", 1.0) or 1.0),
                "elite_same_league_parity_strength": float(
                    best_form.get("elite_same_league_parity_strength", 0.35) or 0.35
                ),
            },
            return_rows=True,
        )
        _microseg_rows = list(combined_report.get("pred_rows") or []) if isinstance(combined_report, dict) else []
        fused_post_calibration = _fit_fused_post_calibration_from_pred_rows(
            _microseg_rows,
            min_train_n=int(max(160, int(min_train_games))),
        )
        microseg_table = _build_microseg_table_from_pred_rows(_microseg_rows, shrink_k=90.0)
        _stage_save(
            "form_combined",
            {
                "form_fit": form_fit,
                "best_form": best_form,
                "best_form_settings": best_form_settings,
                "combined_report": combined_report,
                "fused_post_calibration": fused_post_calibration,
                "microseg_table": microseg_table,
            },
            {
                "form_enabled": bool(best_form.get("enabled", False)),
                "pred_rows_n": len(_microseg_rows),
                "combined_logloss": float(combined_report.get("logloss")) if combined_report.get("logloss") is not None else None,
            },
        )

    leagues = sorted({str(s.league) for s in samples if str(s.league).strip()})
    macros = sorted({str(s.macro) for s in samples if str(s.macro).strip()})
    _sf_cached = _stage_from_checkpoint("segmented_form")
    if _sf_cached:
        best_by_league = dict(_sf_cached.get("best_by_league") or {})
        best_form_by_league = dict(_sf_cached.get("best_form_by_league") or {})
        best_form_by_macro = dict(_sf_cached.get("best_form_by_macro") or {})
    else:
        _stage_start("segmented_form")
        best_by_league: Dict[str, Any] = {}
        best_form_by_league: Dict[str, Any] = {}
        lg_total = len([] if _speed == "fast" else leagues)
        lg_idx = 0
        for lg in ([] if _speed == "fast" else leagues):
            lg_idx += 1
            _stage_pulse(
                "segmented_form",
                {
                    "scope": "league",
                    "index": int(lg_idx),
                    "total": int(lg_total),
                    "league": str(lg),
                },
            )
            _n_lg_all = int(sum(1 for s in samples if str(s.league) == str(lg)))
            _min_train_lg = int(max(80, min(int(min_train_games), round(0.35 * max(1, _n_lg_all)))))
            base_lg = _evaluate_walkforward(
                samples,
                team_scale=1.0,
                players_scale=1.0,
                min_train_games=_min_train_lg,
                league_filter=lg,
                form_cfg={"enabled": False},
            )
            if int(base_lg.get("n_total") or 0) < 120:
                continue
            best_lg: Dict[str, Any] = {"team_scale": 1.0, "players_scale": 1.0, "logloss": float("inf"), "brier": float("inf")}
            for ts in grid_team:
                for ps in grid_players:
                    rep_lg = _evaluate_walkforward(
                        samples,
                        team_scale=float(ts),
                        players_scale=float(ps),
                        min_train_games=_min_train_lg,
                        league_filter=lg,
                        form_cfg={"enabled": False},
                    )
                    ll_lg = rep_lg.get("logloss")
                    br_lg = rep_lg.get("brier")
                    if ll_lg is None:
                        continue
                    if float(ll_lg) < float(best_lg["logloss"]):
                        best_lg = {
                            "team_scale": float(ts),
                            "players_scale": float(ps),
                            "logloss": float(ll_lg),
                            "brier": float(br_lg) if br_lg is not None else None,
                            "n_total": int(rep_lg.get("n_total") or 0),
                            "folds": len(rep_lg.get("folds") or []),
                        }
            if math.isfinite(float(best_lg.get("logloss", float("inf")))):
                bl = float(base_lg.get("logloss")) if base_lg.get("logloss") is not None else float("nan")
                best_lg["baseline_logloss"] = bl if math.isfinite(bl) else None
                if best_lg.get("baseline_logloss") is not None:
                    best_lg["improvement_logloss_vs_baseline"] = float(best_lg["baseline_logloss"]) - float(best_lg["logloss"])
                best_by_league[str(lg)] = best_lg
                try:
                    _form_lg = _fit_form_grid(
                        samples,
                        team_scale=float(best_lg.get("team_scale", best["team_scale"]) or best["team_scale"]),
                        players_scale=float(best_lg.get("players_scale", best["players_scale"]) or best["players_scale"]),
                        min_train_games=_min_train_lg,
                        league_filter=str(lg),
                    )
                    _best_form_lg = dict(_form_lg.get("best_form") or {})
                    if _best_form_lg:
                        best_form_by_league[str(lg)] = {
                            **_best_form_lg,
                            "settings": _form_settings_from_best(_best_form_lg),
                            "min_train_games_used": int(_min_train_lg),
                        }
                except Exception:
                    pass

        best_form_by_macro: Dict[str, Any] = {}
        mk_total = len([] if _speed == "fast" else macros)
        mk_idx = 0
        for mk in ([] if _speed == "fast" else macros):
            mk_idx += 1
            _stage_pulse(
                "segmented_form",
                {
                    "scope": "macro",
                    "index": int(mk_idx),
                    "total": int(mk_total),
                    "macro": str(mk),
                },
            )
            _n_mk_all = int(sum(1 for s in samples if str(s.macro) == str(mk)))
            if _n_mk_all < 120:
                continue
            _min_train_mk = int(max(80, min(int(min_train_games), round(0.30 * max(1, _n_mk_all)))))
            try:
                _form_mk = _fit_form_grid(
                    samples,
                    team_scale=float(best["team_scale"]),
                    players_scale=float(best["players_scale"]),
                    min_train_games=_min_train_mk,
                    macro_filter=str(mk),
                )
                _best_form_mk = dict(_form_mk.get("best_form") or {})
                if _best_form_mk:
                    best_form_by_macro[str(mk)] = {
                        **_best_form_mk,
                        "settings": _form_settings_from_best(_best_form_mk),
                        "min_train_games_used": int(_min_train_mk),
                    }
            except Exception:
                pass
        _stage_save(
            "segmented_form",
            {
                "best_by_league": best_by_league,
                "best_form_by_league": best_form_by_league,
                "best_form_by_macro": best_form_by_macro,
            },
            {
                "leagues_with_fit": len(best_by_league),
                "league_forms": len(best_form_by_league),
                "macro_forms": len(best_form_by_macro),
            },
        )

    form_cfg_global = {
        "enabled": bool(best_form.get("enabled", False)),
        "max_weight": float(best_form.get("max_weight", 0.75) or 0.75),
        "knee_games": float(best_form.get("knee_games", 6.0) or 6.0),
        "beta_prior": float(best_form.get("beta_prior", 2.0) or 2.0),
        "signal_power": float(best_form.get("signal_power", 1.0) or 1.0),
        "elite_same_league_parity_strength": float(best_form.get("elite_same_league_parity_strength", 0.35) or 0.35),
    }
    _sm_cached = _stage_from_checkpoint("series_models")
    if _sm_cached:
        series_gamma_global = dict(_sm_cached.get("series_gamma_global") or {})
        series_gamma_by_league = dict(_sm_cached.get("series_gamma_by_league") or {})
        series_gamma_by_macro = dict(_sm_cached.get("series_gamma_by_macro") or {})
        series_bo_dplite_tuning = dict(_sm_cached.get("series_bo_dplite_tuning") or {})
        blind_backtest = dict(_sm_cached.get("blind_backtest") or {})
    else:
        _stage_start("series_models")
        series_gamma_global = _fit_series_gamma(
            samples,
            team_scale=float(best["team_scale"]),
            players_scale=float(best["players_scale"]),
            form_cfg=form_cfg_global,
        )
        series_gamma_by_league = {}
        sg_lg_items = list([] if _speed == "fast" else best_by_league.items())
        sg_lg_total = len(sg_lg_items)
        sg_lg_idx = 0
        for lg, row in sg_lg_items:
            sg_lg_idx += 1
            _stage_pulse(
                "series_models",
                {
                    "scope": "series_gamma_league",
                    "index": int(sg_lg_idx),
                    "total": int(sg_lg_total),
                    "league": str(lg),
                },
            )
            try:
                lg_form_row = best_form_by_league.get(str(lg)) or {}
                lg_form_set = (lg_form_row.get("settings") if isinstance(lg_form_row, dict) else {}) or {}
                lg_form_cfg = {
                    "enabled": bool(lg_form_set.get("ml_form_blend_enabled", form_cfg_global["enabled"])),
                    "max_weight": float(lg_form_set.get("ml_form_blend_max_weight", form_cfg_global["max_weight"]) or form_cfg_global["max_weight"]),
                    "knee_games": float(lg_form_set.get("ml_form_blend_knee_games", form_cfg_global["knee_games"]) or form_cfg_global["knee_games"]),
                    "beta_prior": float(lg_form_set.get("ml_form_blend_beta", form_cfg_global["beta_prior"]) or form_cfg_global["beta_prior"]),
                    "signal_power": float(lg_form_set.get("ml_form_blend_signal_power", form_cfg_global["signal_power"]) or form_cfg_global["signal_power"]),
                    "elite_same_league_parity_strength": float(
                        lg_form_set.get("ml_form_blend_parity_strength", form_cfg_global["elite_same_league_parity_strength"])
                        or form_cfg_global["elite_same_league_parity_strength"]
                    ),
                }
                rep = _fit_series_gamma(
                    samples,
                    team_scale=float(row.get("team_scale", best["team_scale"]) or best["team_scale"]),
                    players_scale=float(row.get("players_scale", best["players_scale"]) or best["players_scale"]),
                    form_cfg=lg_form_cfg,
                    league_filter=str(lg),
                )
                if isinstance(rep, dict) and not rep.get("error"):
                    series_gamma_by_league[str(lg)] = rep
            except Exception:
                pass
        series_gamma_by_macro = {}
        sg_mk_items = list([] if _speed == "fast" else macros)
        sg_mk_total = len(sg_mk_items)
        sg_mk_idx = 0
        for mk in sg_mk_items:
            sg_mk_idx += 1
            _stage_pulse(
                "series_models",
                {
                    "scope": "series_gamma_macro",
                    "index": int(sg_mk_idx),
                    "total": int(sg_mk_total),
                    "macro": str(mk),
                },
            )
            try:
                mk_form_row = best_form_by_macro.get(str(mk)) or {}
                mk_form_set = (mk_form_row.get("settings") if isinstance(mk_form_row, dict) else {}) or {}
                mk_form_cfg = {
                    "enabled": bool(mk_form_set.get("ml_form_blend_enabled", form_cfg_global["enabled"])),
                    "max_weight": float(mk_form_set.get("ml_form_blend_max_weight", form_cfg_global["max_weight"]) or form_cfg_global["max_weight"]),
                    "knee_games": float(mk_form_set.get("ml_form_blend_knee_games", form_cfg_global["knee_games"]) or form_cfg_global["knee_games"]),
                    "beta_prior": float(mk_form_set.get("ml_form_blend_beta", form_cfg_global["beta_prior"]) or form_cfg_global["beta_prior"]),
                    "signal_power": float(mk_form_set.get("ml_form_blend_signal_power", form_cfg_global["signal_power"]) or form_cfg_global["signal_power"]),
                    "elite_same_league_parity_strength": float(
                        mk_form_set.get("ml_form_blend_parity_strength", form_cfg_global["elite_same_league_parity_strength"])
                        or form_cfg_global["elite_same_league_parity_strength"]
                    ),
                }
                rep = _fit_series_gamma(
                    samples,
                    team_scale=float(best["team_scale"]),
                    players_scale=float(best["players_scale"]),
                    form_cfg=mk_form_cfg,
                    macro_filter=str(mk),
                )
                if isinstance(rep, dict) and not rep.get("error"):
                    series_gamma_by_macro[str(mk)] = rep
            except Exception:
                pass
        series_bo_dplite_tuning = _fit_series_bo_dplite_tuning(
            samples,
            team_scale=float(best["team_scale"]),
            players_scale=float(best["players_scale"]),
            form_cfg=form_cfg_global,
            speed_profile=str(_speed),
        )
        blind_backtest = _blind_backtest_by_league(
            samples,
            team_scale=float(best["team_scale"]),
            players_scale=float(best["players_scale"]),
            form_cfg=form_cfg_global,
            min_holdout_n=max(30, int(min_train_games * 0.05)),
        )
        _stage_save(
            "series_models",
            {
                "series_gamma_global": series_gamma_global,
                "series_gamma_by_league": series_gamma_by_league,
                "series_gamma_by_macro": series_gamma_by_macro,
                "series_bo_dplite_tuning": series_bo_dplite_tuning,
                "blind_backtest": blind_backtest,
            },
            {
                "series_league_models": len(series_gamma_by_league),
                "series_macro_models": len(series_gamma_by_macro),
                "blind_backtest_rows": len((blind_backtest.get("rows") if isinstance(blind_backtest, dict) else []) or []),
            },
        )

    _cm_cached = _stage_from_checkpoint("context_models")
    if _cm_cached:
        meta_blend_c_model = dict(_cm_cached.get("meta_blend_c_model") or {})
        style_matchup_b_model = dict(_cm_cached.get("style_matchup_b_model") or {})
        players_context_fit = dict(_cm_cached.get("players_context_fit") or {})
    else:
        _stage_start("context_models")
        _meta_ck = _checkpoint_payload("context_models.meta_blend_c_model")
        if _meta_ck:
            meta_blend_c_model = dict(_meta_ck.get("meta_blend_c_model") or {})
            _stage_pulse("context_models", {"substage": "meta_blend_c_model", "from_checkpoint": True})
        else:
            _stage_pulse("context_models", {"substage": "meta_blend_c_model"})
            meta_blend_c_model = _fit_meta_blend_c_model(
                samples,
                team_scale=float(best["team_scale"]),
                players_scale=float(best["players_scale"]),
                min_train_games=int(min_train_games),
                form_cfg=form_cfg_global,
            )
            _checkpoint_save(
                "context_models.meta_blend_c_model",
                {"meta_blend_c_model": meta_blend_c_model},
                {
                    "substage": "meta_blend_c_model",
                    "rows_total": int((meta_blend_c_model.get("meta") or {}).get("rows_total") or 0)
                    if isinstance(meta_blend_c_model, dict)
                    else 0,
                },
            )
        _style_ck = _checkpoint_payload("context_models.style_matchup_b_model")
        if _style_ck:
            style_matchup_b_model = dict(_style_ck.get("style_matchup_b_model") or {})
            _stage_pulse("context_models", {"substage": "style_matchup_b_model", "from_checkpoint": True})
        else:
            _stage_pulse("context_models", {"substage": "style_matchup_b_model"})
            style_matchup_b_model = _fit_style_matchup_b_model(
                art=art,
                games=games,
                csv_paths=csv_paths,
                min_train_games=int(max(120, min_train_games)),
                speed_profile=str(_speed),
            )
            _checkpoint_save(
                "context_models.style_matchup_b_model",
                {"style_matchup_b_model": style_matchup_b_model},
                {
                    "substage": "style_matchup_b_model",
                    "rows_total": int((style_matchup_b_model.get("meta") or {}).get("rows_total") or 0)
                    if isinstance(style_matchup_b_model, dict)
                    else 0,
                },
            )
        team_games_multi = _load_team_games_multi(csv_paths)
        _players_ck = _checkpoint_payload("context_models.players_context_fit")
        _players_partial_ck = _checkpoint_payload("context_models.players_context_fit.partial")
        _stage_pulse("context_models", {"substage": "players_context_fit", "resuming_partial": bool(_players_partial_ck)})

        def _players_context_progress(payload: Optional[Dict[str, Any]] = None) -> None:
            detail = {"substage": "players_context_fit"}
            if isinstance(payload, dict):
                detail.update(payload)
            _stage_pulse("context_models", detail)

        if _players_ck:
            players_context_fit = dict(_players_ck.get("players_context_fit") or {})
            _stage_pulse("context_models", {"substage": "players_context_fit", "from_checkpoint": True})
        else:
            def _players_context_checkpoint(payload: Optional[Dict[str, Any]] = None) -> None:
                _checkpoint_save(
                    "context_models.players_context_fit.partial",
                    dict(payload or {}),
                    {
                        "substage": "players_context_fit_partial",
                        "phase": str((payload or {}).get("phase") or ""),
                    },
                )

            players_context_fit = _fit_players_context_tuning(
                samples,
                team_games=team_games_multi,
                team_scale=float(best["team_scale"]),
                players_scale=float(best["players_scale"]),
                min_train_games=int(min_train_games),
                form_cfg=form_cfg_global,
                speed_profile=str(_speed),
                budget_profile=str(_players_context_budget),
                progress_cb=_players_context_progress,
                checkpoint_state=_players_partial_ck,
                checkpoint_save_cb=_players_context_checkpoint,
            )
            _checkpoint_save(
                "context_models.players_context_fit",
                {"players_context_fit": players_context_fit},
                {
                    "substage": "players_context_fit",
                    "decision": str(((players_context_fit.get("gate") or {}).get("decision")) if isinstance(players_context_fit, dict) else ""),
                },
            )
        _stage_save(
            "context_models",
            {
                "meta_blend_c_model": meta_blend_c_model,
                "style_matchup_b_model": style_matchup_b_model,
                "players_context_fit": players_context_fit,
            },
            {
                "meta_model_rows": int((meta_blend_c_model.get("meta") or {}).get("rows_total") or 0) if isinstance(meta_blend_c_model, dict) else 0,
                "style_model_rows": int((style_matchup_b_model.get("meta") or {}).get("rows_total") or 0) if isinstance(style_matchup_b_model, dict) else 0,
                "players_context_decision": str(((players_context_fit.get("gate") or {}).get("decision")) if isinstance(players_context_fit, dict) else ""),
                "players_context_budget": str(((players_context_fit.get("meta") or {}).get("budget_profile")) if isinstance(players_context_fit, dict) else _players_context_budget),
            },
        )
    base_dir = Path(str(artifact_path)).expanduser().resolve().parent
    _ss_cached = _stage_from_checkpoint("snapshots")
    if _ss_cached:
        areal_prob_tuning = dict(_ss_cached.get("areal_prob_tuning") or {})
        areal_prob_tuning_source = str(_ss_cached.get("areal_prob_tuning_source") or "")
        ml_marginal_trust_band_coeffs = dict(_ss_cached.get("ml_marginal_trust_band_coeffs") or {})
        ml_marginal_trust_band_coeffs_source = str(_ss_cached.get("ml_marginal_trust_band_coeffs_source") or "")
    else:
        _stage_start("snapshots")
        areal_prob_tuning = _load_areal_prob_tuning_snapshot(base_dir)
        areal_prob_tuning_source = str(areal_prob_tuning.pop("_source", "")) if isinstance(areal_prob_tuning, dict) else ""
        ml_marginal_trust_band_coeffs = _load_ml_marginal_band_coeffs_snapshot(base_dir)
        ml_marginal_trust_band_coeffs_source = (
            str(ml_marginal_trust_band_coeffs.pop("_source", "")) if isinstance(ml_marginal_trust_band_coeffs, dict) else ""
        )
        _stage_save(
            "snapshots",
            {
                "areal_prob_tuning": areal_prob_tuning,
                "areal_prob_tuning_source": areal_prob_tuning_source,
                "ml_marginal_trust_band_coeffs": ml_marginal_trust_band_coeffs,
                "ml_marginal_trust_band_coeffs_source": ml_marginal_trust_band_coeffs_source,
            },
            {
                "has_areal_prob_tuning": bool(_is_valid_areal_prob_tuning(areal_prob_tuning)),
                "has_marginal_coeffs": bool(_is_valid_ml_marginal_band_coeffs(ml_marginal_trust_band_coeffs)),
            },
        )

    _stage_start("finalize")
    out: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": str(artifact_path),
        "players_artifact": str(players_artifact_path),
        "csvs": [str(x) for x in csv_paths],
        "samples_total": len(samples),
        "speed_profile": str(_speed),
        "players_context_budget": str(_players_context_budget),
        "min_train_games": int(min_train_games),
        "w_lineup_used": float(w_lineup),
        "fit_plan": {
            "run_label": str(run_label or ""),
            "start_stage": str(start_stage_key),
            "seeded_base_params": bool(base_params_seed),
            "base_params_path": (str(base_params_target) if base_params_target is not None else ""),
            "stage_order": list(stage_order),
        },
        "baseline": {
            "team_scale": 1.0,
            "players_scale": 1.0,
            "logloss": baseline.get("logloss"),
            "brier": baseline.get("brier"),
            "n_total": baseline.get("n_total"),
            "folds": len(baseline.get("folds") or []),
        },
        "best": {
            "team_scale": float(best["team_scale"]),
            "players_scale": float(best["players_scale"]),
            "logloss": float(best["logloss"]),
            "brier": float(best["brier"]) if best.get("brier") is not None else None,
            "n_total": int(best.get("n_total") or 0),
            "folds": int(best.get("folds") or 0),
            "improvement_logloss_vs_baseline": (
                float(baseline.get("logloss")) - float(best["logloss"])
                if baseline.get("logloss") is not None
                else None
            ),
        },
        "best_folds": best_report.get("folds") or [],
        "best_form_baseline_no_form": form_fit.get("baseline_no_form") or {},
        "best_form": {**best_form, "settings": best_form_settings},
        "best_combined": {
            "team_scale": float(best["team_scale"]),
            "players_scale": float(best["players_scale"]),
            "form_enabled": bool(best_form.get("enabled", False)),
            "form_settings": dict(best_form_settings),
            "logloss": combined_report.get("logloss"),
            "brier": combined_report.get("brier"),
            "n_total": combined_report.get("n_total"),
            "folds": len(combined_report.get("folds") or []),
            "improvement_logloss_vs_scale_only": (
                float(best.get("logloss")) - float(combined_report.get("logloss"))
                if (best.get("logloss") is not None and combined_report.get("logloss") is not None)
                else None
            ),
        },
        "microseg_calibration": {
            "schema": "league|split|side|bo|phase",
            "source": "walkforward_oos",
            "shrink_k": 90.0,
            "pred_rows_n": len(_microseg_rows),
            "table": microseg_table,
        },
        "fused_post_calibration": fused_post_calibration,
        "best_by_league": best_by_league,
        "best_form_by_league": best_form_by_league,
        "best_form_by_macro": best_form_by_macro,
        "league_macro_map": {str(lg): str(_league_macro_key(lg)) for lg in leagues},
        "players_lane_weights": players_lane_weights,
        "players_league_weights": players_league_weights,
        "patch_snapshot": patch_snapshot,
        "series_gamma_global": series_gamma_global,
        "series_gamma_by_league": series_gamma_by_league,
        "series_gamma_by_macro": series_gamma_by_macro,
        "series_bo_dplite_tuning": series_bo_dplite_tuning,
        "blind_backtest_by_league": blind_backtest,
        "style_matchup_b_model": style_matchup_b_model,
        "meta_blend_c_model": meta_blend_c_model,
        "players_context_fit": players_context_fit,
        "grid": sorted(
            grid_rows,
            key=lambda r: (
                float(r["logloss"]),
                float(r["brier"]) if r.get("brier") is not None else 999.0,
            ),
        ),
        "grid_form_top20": (form_fit.get("grid_form") or [])[:20],
    }
    if _is_valid_areal_prob_tuning(areal_prob_tuning):
        out["areal_prob_tuning"] = dict(areal_prob_tuning)
        out["areal_prob_tuning_source"] = str(areal_prob_tuning_source or "snapshot")
    if _is_valid_ml_marginal_band_coeffs(ml_marginal_trust_band_coeffs):
        out["ml_marginal_trust_band_coeffs"] = _sanitize_ml_marginal_band_coeffs(ml_marginal_trust_band_coeffs)
        out["ml_marginal_trust_band_coeffs_source"] = str(ml_marginal_trust_band_coeffs_source or "snapshot")
    _stage_done(
        "finalize",
        {
            "out_keys": len(out),
            "best_logloss": float((out.get("best") or {}).get("logloss") or 0.0),
            "samples_total": int(out.get("samples_total") or 0),
        },
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit ML params (team_scale, players_scale, season_form_blend) via walk-forward."
    )
    ap.add_argument("--artifact", default="ml_artifact.json")
    ap.add_argument("--players-artifact", default="players_artifact.json")
    ap.add_argument("--csv", action="append", required=True, help="OracleElixir CSV (can pass multiple)")
    ap.add_argument("--out", default="params_fitted.json")
    ap.add_argument("--min-train-games", type=int, default=300)
    ap.add_argument("--speed-profile", choices=["full", "fast"], default="full")
    ap.add_argument("--progress-out", default="ops/monitoring/fit_ml_params_progress.json")
    ap.add_argument("--checkpoint-dir", default="ops/fit_checkpoints/fit_ml_params")
    ap.add_argument("--run-label", default="")
    ap.add_argument("--start-stage", choices=FIT_STAGE_ORDER, default="load_inputs")
    ap.add_argument("--base-params", default="")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    artifact_path = str(Path(args.artifact).expanduser())
    players_artifact_path = str(Path(args.players_artifact).expanduser())
    csv_paths = [str(Path(x).expanduser()) for x in (args.csv or [])]
    rep = fit_params(
        artifact_path=artifact_path,
        players_artifact_path=players_artifact_path,
        csv_paths=csv_paths,
        min_train_games=int(args.min_train_games),
        speed_profile=str(args.speed_profile),
        progress_path=str(args.progress_out or ""),
        checkpoint_dir=str(args.checkpoint_dir or ""),
        resume=bool(args.resume),
        run_label=str(args.run_label or ""),
        start_stage=str(args.start_stage or ""),
        base_params_path=str(args.base_params or ""),
    )
    out = Path(str(args.out)).expanduser()
    _write_json_file_safe(out, rep)
    print("[fit_ml_params] saved:", str(out))
    if "best_combined" in rep:
        print(json.dumps(rep["best_combined"], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
