# -*- coding: utf-8 -*-
"""Generates and persists the team state artifact from CSV data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from mlcore.state_team_artifact import (
    build_team_state_artifact as _build_team_state_artifact,
    load_team_state_artifact as _load_team_state_artifact,
    save_team_state_artifact as _save_team_state_artifact,
)


def generate_and_persist_team_state_artifact(
    *,
    root: Path,
    csv_paths: List[str],
    keep_playoffs: Optional[bool] = None,
    ctx: Mapping[str, Any],
) -> Dict[str, Any]:
    _file_sig = ctx.get("_file_sig")
    _load_fitted_params_cached = ctx.get("_load_fitted_params_cached")

    seed_art: Dict[str, Any] = {}
    seed_path = root / "team_state_artifact.json"
    try:
        if seed_path.exists():
            seed_art = _load_team_state_artifact(seed_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        seed_art = {}

    fit_cfg: Dict[str, Any] = {}
    try:
        fit_p = root / "params_fitted.json"
        if fit_p.exists():
            fit_cfg = _load_fitted_params_cached(_file_sig(str(fit_p))) or {}
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        fit_cfg = {}

    areal_tune = (fit_cfg.get("areal_prob_tuning") or {}) if isinstance(fit_cfg, dict) else {}
    at_global = (areal_tune.get("global") or {}) if isinstance(areal_tune, dict) else {}
    at_by_league = (areal_tune.get("by_league") or {}) if isinstance(areal_tune, dict) else {}
    at_by_patch = (areal_tune.get("by_patch") or {}) if isinstance(areal_tune, dict) else {}

    rep = _build_team_state_artifact(
        csv_paths,
        keep_playoffs=keep_playoffs,
        include_players_for_roster=True,
        mu_update_mode=str(areal_tune.get("mu_update_mode") or "bayes_joint_v1"),
        sigma_update_mode=str(areal_tune.get("sigma_update_mode") or "bayes_logit_v1"),
        bayes_info_scale=float(at_global.get("bayes_info_scale") or 1.0),
        bayes_mu_q_strength=float(at_global.get("bayes_mu_q_strength") or 1.0),
        quality_k_boost=float(at_global.get("quality_k_boost") or 0.22),
        qwl_surplus_mix=float(at_global.get("qwl_surplus_mix") or 0.70),
        qwl_throw_weight=float(at_global.get("qwl_throw_weight") or 1.20),
        qwl_comeback_weight=float(at_global.get("qwl_comeback_weight") or 0.65),
        qwl_phase_dynamic_strength=float(at_global.get("qwl_phase_dynamic_strength") or 0.35),
        qwl_opponent_context_mix=float(at_global.get("qwl_opponent_context_mix") or 0.30),
        bayes_info_scale_by_league=((at_by_league.get("bayes_info_scale") or {}) if isinstance(at_by_league, dict) else {}),
        quality_k_boost_by_league=((at_by_league.get("quality_k_boost") or {}) if isinstance(at_by_league, dict) else {}),
        qwl_surplus_mix_by_league=((at_by_league.get("qwl_surplus_mix") or {}) if isinstance(at_by_league, dict) else {}),
        qwl_throw_weight_by_league=((at_by_league.get("qwl_throw_weight") or {}) if isinstance(at_by_league, dict) else {}),
        qwl_comeback_weight_by_league=((at_by_league.get("qwl_comeback_weight") or {}) if isinstance(at_by_league, dict) else {}),
        qwl_phase_dynamic_strength_by_league=((at_by_league.get("qwl_phase_dynamic_strength") or {}) if isinstance(at_by_league, dict) else {}),
        qwl_opponent_context_mix_by_league=((at_by_league.get("qwl_opponent_context_mix") or {}) if isinstance(at_by_league, dict) else {}),
        bayes_info_scale_by_patch=((at_by_patch.get("bayes_info_scale") or {}) if isinstance(at_by_patch, dict) else {}),
        quality_k_boost_by_patch=((at_by_patch.get("quality_k_boost") or {}) if isinstance(at_by_patch, dict) else {}),
        qwl_surplus_mix_by_patch=((at_by_patch.get("qwl_surplus_mix") or {}) if isinstance(at_by_patch, dict) else {}),
        qwl_throw_weight_by_patch=((at_by_patch.get("qwl_throw_weight") or {}) if isinstance(at_by_patch, dict) else {}),
        qwl_comeback_weight_by_patch=((at_by_patch.get("qwl_comeback_weight") or {}) if isinstance(at_by_patch, dict) else {}),
        qwl_phase_dynamic_strength_by_patch=((at_by_patch.get("qwl_phase_dynamic_strength") or {}) if isinstance(at_by_patch, dict) else {}),
        qwl_opponent_context_mix_by_patch=((at_by_patch.get("qwl_opponent_context_mix") or {}) if isinstance(at_by_patch, dict) else {}),
        seed_artifact=(seed_art if isinstance(seed_art, dict) and seed_art else None),
        incremental=True,
    )
    if isinstance(rep, dict) and not rep.get("error"):
        _save_team_state_artifact(root / "team_state_artifact.json", rep)
    return rep
