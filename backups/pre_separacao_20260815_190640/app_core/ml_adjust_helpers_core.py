# -*- coding: utf-8 -*-
"""Helpers extraídos do plays_app (fatiamento F3, 2026-07-18).

Funções PURAS (não referenciam nada definido no plays_app; sem __file__).
O plays_app importa todas de volta, então o namespace que alimenta o
ctx das abas (globals()->build_render_ctx) segue idêntico.
"""
from __future__ import annotations

from app_core.fusion_core import (
    fuse_with_coherence_guard as core_fuse_with_coherence_guard,
)
from app_core.ml_artifact_pricing_core import (
    p_from_diff_points as core_p_from_diff_points,
    strength_breakdown as core_strength_breakdown,
)
from app_core.ml_league_runtime_core import (
    is_marginal_league_for_ml as core_is_marginal_league_for_ml,
    league_is_excluded as core_league_is_excluded,
    ml_league_conf_band as core_ml_league_conf_band,
    ml_marginal_band_profile as core_ml_marginal_band_profile,
    norm_league as core_norm_league,
    norm_league_key as core_norm_league_key,
    sanitize_ml_marginal_band_coeffs as core_sanitize_ml_marginal_band_coeffs,
    tier_from_league as core_tier_from_league,
)
from app_core.ml_oos_gate_core import (
    wf_correct_p_map as _wf_correct_p_map_core,
)
from app_core.ml_probability_adjustments_core import (
    apply_ml_uncertainty_adjust as _apply_ml_uncertainty_adjust_core,
    apply_ml_marginal_trust_adjust as _apply_ml_marginal_trust_adjust_core,
)
from datetime import date
from mlcore.achievements import team_boost_points
from mlcore.artifact import MLArtifact
from mlcore.tiers import tier_base_points, league_base_points
from mlcore.utils import clip_prob
from typing import Any, Dict, Optional
import math
import streamlit as st


def _p_from_diff_points(diff_points: float, scale_points: float = 400.0) -> float:
    return float(core_p_from_diff_points(diff_points, scale_points=scale_points, clip_prob_fn=clip_prob))

def _strength_breakdown(
    *,
    art: MLArtifact,
    team: str,
    as_of: date,
    team_meta: Dict[str, Dict[str, str]],
    ach_layer: Dict[str, Any],
    league_hint: Optional[str] = None,
    tier_hint: Optional[str] = None,
) -> Dict[str, Any]:
    return core_strength_breakdown(
        art=art,
        team=team,
        as_of=as_of,
        team_meta=team_meta,
        ach_layer=ach_layer,
        norm_league_fn=core_norm_league,
        tier_from_league_fn=core_tier_from_league,
        tier_base_points_fn=tier_base_points,
        league_base_points_fn=league_base_points,
        team_boost_points_fn=team_boost_points,
        league_hint=league_hint,
        tier_hint=tier_hint,
    )

def _apply_ml_uncertainty_adjust(
    p_in: float,
    ml_info,
    *,
    enabled: bool = True,
    strength: float = 0.55,
    prior_games: float = 22.0,
    max_shrink: float = 0.65,
):
    return _apply_ml_uncertainty_adjust_core(
        p_in,
        ml_info,
        clip_prob_fn=clip_prob,
        enabled=enabled,
        strength=strength,
        prior_games=prior_games,
        max_shrink=max_shrink,
    )

def _apply_ml_marginal_trust_adjust(
    p_in: float,
    ml_info,
    *,
    league: str = "",
    enabled: bool = True,
    strength: float = 0.70,
    prior_games: float = 28.0,
    max_shrink: float = 0.72,
    cap_prob: float = 0.84,
    confidence_score=None,
    score_soft: float = 68.0,
    score_hard: float = 46.0,
    exclude_leagues=None,
    band_tuning: bool = True,
    band_coeffs=None,
):
    return _apply_ml_marginal_trust_adjust_core(
        p_in,
        ml_info,
        clip_prob_fn=clip_prob,
        norm_league_key_fn=core_norm_league_key,
        tier_from_league_fn=core_tier_from_league,
        ml_league_conf_band_fn=core_ml_league_conf_band,
        ml_marginal_band_profile_fn=core_ml_marginal_band_profile,
        league_is_excluded_fn=core_league_is_excluded,
        is_marginal_league_for_ml_fn=core_is_marginal_league_for_ml,
        sanitize_ml_marginal_band_coeffs_fn=core_sanitize_ml_marginal_band_coeffs,
        league=league,
        enabled=enabled,
        strength=strength,
        prior_games=prior_games,
        max_shrink=max_shrink,
        cap_prob=cap_prob,
        confidence_score=confidence_score,
        score_soft=score_soft,
        score_hard=score_hard,
        exclude_leagues=exclude_leagues if exclude_leagues is not None else "",
        band_tuning=band_tuning,
        band_coeffs=band_coeffs,
    )

def _fuse_probs_by_precision(
    *,
    p_team: float,
    p_players: float,
    n_team: float,
    n_players: float,
    team_scale: float = 1.0,
    players_scale: float = 1.0,
    coverage: Optional[float] = None,
    transfer_signal: float = 0.0,
    lineup_continuity: Optional[float] = None,
    roster_changed: bool = False,
    season_phase: str = "auto",
) -> Dict[str, Any]:
    nt = float(n_team) if math.isfinite(float(n_team)) else 0.0
    np_ = float(n_players) if math.isfinite(float(n_players)) else 0.0
    ts = float(team_scale) if math.isfinite(float(team_scale)) else 1.0
    ps = float(players_scale) if math.isfinite(float(players_scale)) else 1.0
    return core_fuse_with_coherence_guard(
        p_team=float(p_team),
        p_players=float(p_players),
        n_team=float(max(0.0, nt)),
        n_players=float(max(0.0, np_)),
        team_scale=float(max(0.0, ts)),
        players_scale=float(max(0.0, ps)),
        coverage=coverage,
        transfer_signal=float(transfer_signal),
        lineup_continuity=lineup_continuity,
        roster_changed=bool(roster_changed),
        season_phase=str(season_phase or "auto"),
        team_weight_early=float(st.session_state.get("fusion_team_weight_early", 0.58) or 0.58),
        team_weight_mid=float(st.session_state.get("fusion_team_weight_mid", 0.75) or 0.75),
        team_weight_playoffs=float(st.session_state.get("fusion_team_weight_playoffs", 0.82) or 0.82),
        early_season_team_weight=float(st.session_state.get("fusion_early_season_team_weight", 0.55) or 0.55),
        season_knee_games=float(st.session_state.get("fusion_season_knee_games", 18.0) or 18.0),
        coverage_power=float(st.session_state.get("fusion_coverage_power", 1.5) or 1.5),
        transfer_boost=float(st.session_state.get("fusion_transfer_boost", 0.35) or 0.35),
        divergence_pp_cap=float(st.session_state.get("fusion_divergence_pp_cap", 22.0) or 22.0),
        divergence_low_coverage=float(st.session_state.get("fusion_divergence_low_coverage", 0.30) or 0.30),
        divergence_shrink=float(st.session_state.get("fusion_divergence_shrink", 0.35) or 0.35),
    )

def _wf_correct_p_map(
    report,
    p_in: float,
    league_hint: str = "",
    strength: float = 0.60,
):
    return _wf_correct_p_map_core(report, p_in, league_hint=league_hint, strength=strength)

