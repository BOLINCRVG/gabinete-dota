# -*- coding: utf-8 -*-
"""Helpers extraídos do plays_app (fatiamento F3, 2026-07-18).

Funções PURAS (não referenciam nada definido no plays_app; sem __file__).
O plays_app importa todas de volta, então o namespace que alimenta o
ctx das abas (globals()->build_render_ctx) segue idêntico.
"""
from __future__ import annotations

from app_core.bootstrap_maintenance_state_core import (
    read_json_file_safe as core_read_json_file_safe,
)
from app_core.maintenance_runtime_core import (
    read_artifact_summary_path as core_read_artifact_summary_path,
)
from app_core.ml_confidence_baseline_persist_core import generate_and_persist_ml_confidence_baseline as _generate_and_persist_ml_confidence_baseline_core
from app_core.ml_league_runtime_core import (
    sanitize_ml_marginal_band_coeffs as core_sanitize_ml_marginal_band_coeffs,
)
from app_core.params_tuning_core import (
    enrich_params_with_ml_marginal_band_coeffs as core_enrich_params_with_ml_marginal_band_coeffs,
)
from app_core.process_runner_core import run_cmd as _run_cmd_core
from mlcore.calibration_diagnostics import run_calibration_diagnostics
from mlcore.layer_validation import run_layer_validation
from mlcore.walkforward_report import run_walkforward
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import streamlit as st


def _read_artifact_summary(artifact_path: str) -> dict:
    return dict(core_read_artifact_summary_path(artifact_path))

def _run_cmd(
    cmd,
    *,
    timeout_s=None,
    heartbeat_cb=None,
    heartbeat_interval_s: float = 2.0,
):
    return _run_cmd_core(cmd, timeout_s=timeout_s, heartbeat_cb=heartbeat_cb, heartbeat_interval_s=heartbeat_interval_s)

def _read_json_file_safe(path_like: str | Path) -> Dict[str, Any]:
    return dict(core_read_json_file_safe(path_like))

def _enrich_params_with_ml_marginal_band_coeffs(params_obj: Any, *, root: Path) -> Tuple[Dict[str, Any], str]:
    return core_enrich_params_with_ml_marginal_band_coeffs(
        params_obj,
        root=root,
        sanitize_band_coeffs_fn=core_sanitize_ml_marginal_band_coeffs,
    )

@st.cache_data(show_spinner=False, ttl=1800)
def _run_walkforward_cached(
    csv_paths: tuple[str, ...],
    min_train_games: int,
    k: float,
    scale: float,
    artifact_path: str = "",
    players_artifact_path: str = "",
    team_scale: float = 1.0,
    players_scale: float = 1.0,
    bins: int = 10,
    min_league_games: int = 250,
    params_fitted_path: str = "",
) -> dict:
    clean = [str(Path(p).expanduser()) for p in (csv_paths or ()) if str(p).strip()]
    return run_walkforward(
        clean,
        min_train_games=int(min_train_games),
        k=float(k),
        scale=float(scale),
        artifact_path=str(Path(str(artifact_path)).expanduser()) if str(artifact_path or "").strip() else "",
        players_artifact_path=str(Path(str(players_artifact_path)).expanduser()) if str(players_artifact_path or "").strip() else "",
        team_scale=float(team_scale),
        players_scale=float(players_scale),
        bins=int(bins),
        min_league_games=int(min_league_games),
        params_fitted_path=str(Path(str(params_fitted_path)).expanduser()) if str(params_fitted_path or "").strip() else "",
    )

@st.cache_data(show_spinner=False, ttl=1800)
def _run_layer_validation_cached(
    csv_paths: tuple[str, ...],
    artifact_path: str,
    players_artifact_path: str,
    min_train_games: int,
    team_scale: float,
    players_scale: float,
    min_league_games: int,
) -> dict:
    clean = [str(Path(p).expanduser()) for p in (csv_paths or ()) if str(p).strip()]
    return run_layer_validation(
        artifact_path=str(Path(str(artifact_path)).expanduser()),
        players_artifact_path=str(Path(str(players_artifact_path)).expanduser()),
        csv_paths=clean,
        min_train_games=int(min_train_games),
        team_scale=float(team_scale),
        players_scale=float(players_scale),
        min_league_games=int(min_league_games),
    )

@st.cache_data(show_spinner=False, ttl=1800)
def _run_calibration_diag_cached(
    csv_paths: tuple[str, ...],
    artifact_path: str,
    players_artifact_path: str,
    team_scale: float,
    players_scale: float,
    bins: int,
    min_league_games: int,
) -> dict:
    clean = [str(Path(p).expanduser()) for p in (csv_paths or ()) if str(p).strip()]
    return run_calibration_diagnostics(
        artifact_path=str(Path(str(artifact_path)).expanduser()),
        players_artifact_path=str(Path(str(players_artifact_path)).expanduser()),
        csv_paths=clean,
        team_scale=float(team_scale),
        players_scale=float(players_scale),
        bins=int(bins),
        min_league_games=int(min_league_games),
    )

def _generate_and_persist_ml_confidence_baseline(
    *,
    root: Path,
    lv_report: Optional[Dict[str, Any]],
    wf_report: Optional[Dict[str, Any]],
    params_fit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _generate_and_persist_ml_confidence_baseline_core(root=root, lv_report=lv_report, wf_report=wf_report, params_fit=params_fit)

