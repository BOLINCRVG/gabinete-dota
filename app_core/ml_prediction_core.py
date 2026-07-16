"""Pure ML prediction computation extracted from plays_app.py.

Two functions cover the full prediction pipeline:

1. compute_ml_core_prediction — cache-able block:
   league resolution → p_map_v2 (forward + symmetry reverse) →
   microseg calibration → fused post-calibration.

2. compute_ml_post_layers — always-run block (not cached):
   WF correction → p_series → fit scales + players context.

All I/O and Streamlit session state remain in plays_app.py; these
functions receive data and callbacks, return result dicts.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _f(v: Any, default: float = float("nan")) -> float:
    try:
        r = float(v)
        return r if math.isfinite(r) else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Core prediction (cache-able)
# ---------------------------------------------------------------------------

def compute_ml_core_prediction(
    *,
    # Matchup
    team_a: str,
    team_b: str,
    artifact_path: str,
    as_of_date: Any,
    achievement_path: Optional[str],
    # Runtime config
    ml_runtime_cfg: Dict[str, Any],
    # Filter context (for microseg segment hints)
    filter_year: Optional[int],
    filter_league: Optional[str],
    filter_split: Optional[str],
    filter_playoffs: Any,
    bo: int,
    league_mode: str,
    # Sigs for cached microseg building
    csv_sig: Any,
    art_sig: Any,
    fit_params_path: Optional[str],
    # Callbacks — injected by plays_app to keep caching/IO there
    resolve_leagues_fn: Callable,
    # signature: (team_a, team_b, year, league_mode, fixed_league) -> dict
    p_map_v2_fn: Callable,
    # signature: (artifact_path, team_blue, team_red, as_of, *, ach_override_path,
    #             league_hint_blue, league_hint_red, tier_hint_blue, tier_hint_red)
    #           -> Optional[dict]
    build_microseg_table_fn: Callable,
    # signature: (csv_sig, art_sig, lookback_games, shrink_k) -> dict
    apply_microseg_fn: Callable,
    # signature: (p_map_cal, *, league, split, side, bo, phase, table,
    #             min_n, strength, cap_pp, hierarchical, hierarchy_decay)
    #           -> (float, dict)
    apply_fused_postcal_fn: Callable,
    # signature: (p_map_cal, *, league_hint, use_fitted) -> (float, dict)
    load_fitted_params_fn: Optional[Callable] = None,
    # signature: (file_sig_tuple) -> dict
    file_sig_fn: Optional[Callable] = None,
    # signature: (path_str) -> (str, int, int)
) -> Dict[str, Any]:
    """Compute p_map + calibration for a matchup, returning a cache-able result dict.

    Returns
    -------
    Dict with keys:
      ml_info, p_map_raw, p_map_cal,
      league_a, league_b, tier_a, tier_b,
      microseg_info, fused_post_cal_info
    """
    p_map_raw = float("nan")
    p_map_cal = float("nan")
    ml_info: Optional[Dict[str, Any]] = None
    league_a = league_b = tier_a = tier_b = ""
    microseg_info: Optional[Dict[str, Any]] = None
    fused_post_cal_info: Optional[Dict[str, Any]] = None

    # 1. League resolution
    fixed_league = str(filter_league or "") if league_mode == "fixed" else ""
    try:
        _lg = resolve_leagues_fn(
            team_a, team_b, filter_year, str(league_mode), fixed_league
        )
        league_a = str(_lg.get("league_a") or "")
        league_b = str(_lg.get("league_b") or "")
        tier_a = str(_lg.get("tier_a") or "")
        tier_b = str(_lg.get("tier_b") or "")
    except (OSError, TypeError, ValueError):
        pass

    # 2. Forward + reverse core ML. Computing both directions lets us symmetrize
    #    the FINAL output (p(A,B) == 1 − p(B,A)). The reverse was already computed
    #    before, but symmetry was applied too early (only to the core), and then
    #    microseg/fused (order-dependent: league_a, side="blue") re-broke it. Now
    #    we run the full post pipeline on BOTH directions and average at the end.
    def _core_ml(ta: str, tb: str, la: str, lb: str, tta: str, ttb: str) -> Optional[Dict[str, Any]]:
        return p_map_v2_fn(
            artifact_path, ta, tb, as_of_date,
            ach_override_path=achievement_path,
            league_hint_blue=la or None, league_hint_red=lb or None,
            tier_hint_blue=tta or None, tier_hint_red=ttb or None,
        )

    ml_error = ""
    try:
        ml_info = _core_ml(team_a, team_b, league_a, league_b, tier_a, tier_b)
    except Exception as exc:
        # Nunca engolir mudo: o motivo viaja no resultado e aparece no aviso
        # da UI (na nuvem o traceback é redigido; sem isso o NaN é indecifrável).
        ml_error = f"{type(exc).__name__}: {exc}"
        ml_info = None
    if ml_info is None and not ml_error:
        ml_error = "motor v2 retornou vazio (times fora do artifact?)"
    ml_info_rev: Optional[Dict[str, Any]] = None
    if ml_info is not None:
        try:
            ml_info_rev = _core_ml(team_b, team_a, league_b, league_a, tier_b, tier_a)
        except Exception:
            ml_info_rev = None

    if ml_info is not None:
        p_cal_fwd = _f(ml_info.get("p_cal"))
        p_raw_fwd = _f(ml_info.get("p_raw"))
        p_cal_rev = _f(ml_info_rev.get("p_cal")) if isinstance(ml_info_rev, dict) else float("nan")
        p_raw_rev = _f(ml_info_rev.get("p_raw")) if isinstance(ml_info_rev, dict) else float("nan")

        # RAW: symmetrize directly (no post-layers apply to the raw probability).
        p_map_raw = p_raw_fwd
        if math.isfinite(p_raw_fwd) and math.isfinite(p_raw_rev):
            p_map_raw = 0.5 * (p_raw_fwd + (1.0 - p_raw_rev))

        # Build the microseg table ONCE (shared by both directions).
        seg_tbl: Dict[str, Any] = {}
        seg_src = "live"
        microseg_on = math.isfinite(p_cal_fwd) and bool(ml_runtime_cfg.get("ml_microseg_enabled"))
        if microseg_on:
            try:
                if load_fitted_params_fn is not None and file_sig_fn is not None and fit_params_path:
                    try:
                        _fit_live = load_fitted_params_fn(file_sig_fn(fit_params_path))
                        _ms_pack = (_fit_live or {}).get("microseg_calibration") if isinstance(_fit_live, dict) else {}
                        _tbl_fit = (_ms_pack or {}).get("table") if isinstance(_ms_pack, dict) else {}
                        if isinstance(_tbl_fit, dict) and len(_tbl_fit) > 0:
                            seg_tbl = _tbl_fit
                            seg_src = "fitted"
                    except (OSError, TypeError, ValueError):
                        seg_tbl = {}
                if not seg_tbl:
                    seg_tbl = build_microseg_table_fn(
                        csv_sig=csv_sig,
                        art_sig=art_sig,
                        lookback_games=int(ml_runtime_cfg.get("ml_microseg_lookback_games", 1200)),
                        shrink_k=float(ml_runtime_cfg.get("ml_microseg_shrink_k", 90.0)),
                    )
            except (AttributeError, KeyError, OSError, TypeError, ValueError):
                seg_tbl = {}

        _split_hint = str(filter_split or "")
        _phase_hint = (
            "playoffs"
            if str(filter_playoffs or "").lower() in {"1", "true", "playoffs"}
            else "regular"
        )
        _bo_hint = str(int(bo)) if str(int(bo)) in {"1", "3", "5"} else "1"

        def _apply_post(p_cal: float, league_hint: str) -> tuple:
            """microseg + fused post-cal applied to ONE direction's calibrated p.

            Returns (p, microseg_info, fused_info).
            """
            p = float(p_cal)
            ms_info: Optional[Dict[str, Any]] = None
            fp_info: Optional[Dict[str, Any]] = None
            if microseg_on and math.isfinite(p):
                try:
                    p, _seg = apply_microseg_fn(
                        float(p),
                        league=str(league_hint or ""),
                        split=_split_hint,
                        side="blue",
                        bo=_bo_hint,
                        phase=_phase_hint,
                        table=seg_tbl,
                        min_n=int(ml_runtime_cfg.get("ml_microseg_min_n", 30)),
                        strength=float(ml_runtime_cfg.get("ml_microseg_strength", 0.5)),
                        cap_pp=float(ml_runtime_cfg.get("ml_microseg_cap_pp", 3.0)),
                        hierarchical=bool(ml_runtime_cfg.get("ml_microseg_hierarchical", True)),
                        hierarchy_decay=float(ml_runtime_cfg.get("ml_microseg_hier_decay", 0.80)),
                    )
                    if isinstance(_seg, dict):
                        _seg["source"] = str(seg_src)
                        ms_info = _seg
                except (AttributeError, KeyError, OSError, TypeError, ValueError):
                    pass
            if math.isfinite(p):
                try:
                    p, _fp = apply_fused_postcal_fn(
                        float(p),
                        league_hint=str(league_hint or ""),
                        use_fitted=bool(ml_runtime_cfg.get("ml_fused_post_cal_use_fitted", True)),
                    )
                    if isinstance(_fp, dict):
                        fp_info = _fp
                except (TypeError, ValueError):
                    pass
            return p, ms_info, fp_info

        # Full post pipeline on BOTH directions; microseg/fused info from forward
        # (for display). The reverse is only needed to symmetrize the output.
        full_fwd = p_cal_fwd
        if math.isfinite(p_cal_fwd):
            full_fwd, microseg_info, fused_post_cal_info = _apply_post(p_cal_fwd, league_a)
        full_rev = p_cal_rev
        if math.isfinite(p_cal_rev):
            full_rev, _, _ = _apply_post(p_cal_rev, league_b)

        # 6. Output-level symmetry: guarantees p(A,B) == 1 − p(B,A).
        if math.isfinite(full_fwd) and math.isfinite(full_rev):
            p_map_cal = 0.5 * (full_fwd + (1.0 - full_rev))
        else:
            p_map_cal = full_fwd

    return {
        "ml_info": ml_info,
        "ml_error": str(ml_error),
        "p_map_raw": float(p_map_raw),
        "p_map_cal": float(p_map_cal),
        "league_a": str(league_a),
        "league_b": str(league_b),
        "tier_a": str(tier_a),
        "tier_b": str(tier_b),
        "microseg_info": microseg_info,
        "fused_post_cal_info": fused_post_cal_info,
    }


# ---------------------------------------------------------------------------
# Post layers (not cached — runs every render)
# ---------------------------------------------------------------------------

def compute_ml_post_layers(
    *,
    ml_info: Optional[Dict[str, Any]],
    p_map_cal: float,
    bo: int,
    ml_runtime_cfg: Dict[str, Any],
    wf_report: Optional[Dict[str, Any]],
    # Callbacks
    wf_correct_p_map_fn: Callable,
    # signature: (report, p_in, *, league_hint, strength) -> (float, dict)
    fit_scales_for_league_fn: Callable,
    # signature: (league_hint) -> (team_scale, players_scale, league_used)
    fit_players_context_for_league_fn: Callable,
    # signature: (league_hint) -> (cfg_dict, source_str)
    prob_win_series_fn: Callable,
    # signature: (p_map, bo) -> float
) -> Dict[str, Any]:
    """Apply WF correction, compute p_series, and resolve fit scales.

    Returns
    -------
    Dict with keys:
      p_map_cal, p_series, wf_corr_info,
      fit_team_scale, fit_players_scale, fit_league_used,
      players_context_cfg, players_context_src
    """
    wf_corr_info: Optional[Dict[str, Any]] = None

    # 1. WF correction
    if math.isfinite(p_map_cal) and bool(ml_runtime_cfg.get("wf_apply_ml_correction")):
        try:
            _league_hint = ""
            if isinstance(ml_info, dict) and not ml_info.get("__error__"):
                _lb = str(((ml_info.get("blue") or {}).get("league")) or "").strip()
                _lr = str(((ml_info.get("red") or {}).get("league")) or "").strip()
                if _lb and _lb == _lr:
                    _league_hint = _lb
            p_corr, corr_info = wf_correct_p_map_fn(
                wf_report if isinstance(wf_report, dict) else None,
                float(p_map_cal),
                league_hint=_league_hint,
                strength=float(ml_runtime_cfg.get("wf_ml_corr_strength", 0.60)),
            )
            if math.isfinite(float(p_corr)):
                p_map_cal = float(p_corr)
            if isinstance(corr_info, dict):
                wf_corr_info = corr_info
        except (TypeError, ValueError):
            pass

    # 2. p_series
    p_series = float("nan")
    if math.isfinite(p_map_cal):
        try:
            p_series = float(prob_win_series_fn(float(p_map_cal), int(bo)))
        except (TypeError, ValueError, OverflowError):
            p_series = float("nan")

    # 3. Fit scales + players context
    fit_league_hint = ""
    if isinstance(ml_info, dict) and not ml_info.get("__error__"):
        _lb = str(((ml_info.get("blue") or {}).get("league")) or "").strip()
        _lr = str(((ml_info.get("red") or {}).get("league")) or "").strip()
        if _lb and _lb == _lr:
            fit_league_hint = _lb

    try:
        fit_team_scale, fit_players_scale, fit_league_used = fit_scales_for_league_fn(fit_league_hint)
    except Exception:
        fit_team_scale, fit_players_scale, fit_league_used = 1.0, 1.0, ""

    try:
        players_context_cfg, players_context_src = fit_players_context_for_league_fn(fit_league_hint)
    except Exception:
        players_context_cfg, players_context_src = {}, ""

    return {
        "p_map_cal": float(p_map_cal),
        "p_series": float(p_series),
        "wf_corr_info": wf_corr_info,
        "fit_team_scale": float(fit_team_scale),
        "fit_players_scale": float(fit_players_scale),
        "fit_league_used": str(fit_league_used),
        "players_context_cfg": players_context_cfg if isinstance(players_context_cfg, dict) else {},
        "players_context_src": str(players_context_src),
    }


__all__ = ["compute_ml_core_prediction", "compute_ml_post_layers"]
