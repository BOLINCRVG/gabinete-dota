# -*- coding: utf-8 -*-
"""Helpers extraídos do plays_app (fatiamento F3, 2026-07-18).

Funções PURAS (não referenciam nada definido no plays_app; sem __file__).
O plays_app importa todas de volta, então o namespace que alimenta o
ctx das abas (globals()->build_render_ctx) segue idêntico.
"""
from __future__ import annotations

from app_core.league_transfer_features_core import (
    load_league_transfer_features as core_load_league_transfer_features,
)
from app_core.lineup_state import (
    norm_pos as _norm_pos,
    parse_pct as _parse_pct,
)
from app_core.patch_guard_core import patch_changed_leagues_set as _patch_changed_leagues_set_core
from app_core.players_lineup_impact_core import (
    load_players_impact as core_load_players_impact,
)
from mlcore.achievements import load_achievements
from mlcore.artifact import MLArtifact
from mlcore.players_layer import PlayersArtifact
from mlcore.state_team_artifact import (
    load_team_state_artifact as core_load_team_state_artifact,
)
from pathlib import Path
from typing import Any, Dict, Tuple
import json
import pandas as pd
import streamlit as st


def _file_sig(path: str) -> Tuple[str, int, int]:
    p = Path(path)
    try:
        st_ = p.stat()
        return (str(p.resolve()), int(st_.st_size), int(st_.st_mtime))
    except OSError:
        return (str(p), 0, 0)

def _resolve_relpath(base_file: str, maybe_rel: str) -> str:
    p = Path(maybe_rel)
    if p.is_absolute():
        return str(p)
    return str((Path(base_file).parent / p).resolve())

def _artifact_sig(path: str) -> Tuple[str, int, int]:
    return _file_sig(path)

@st.cache_resource(show_spinner=False)
def _load_artifact_cached(art_sig: Tuple[str, int, int]) -> MLArtifact:
    art_path, _size, _mtime = art_sig
    return MLArtifact.load_json(art_path)

@st.cache_data(show_spinner=False)
def _load_achievements_cached(ach_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    """Carrega achievements.json com tolerância a erros.

    Nunca deve derrubar o app (ex.: JSON corrompido). Em caso de falha,
    retorna um dict com '__load_error__' para o chamador decidir o fallback.
    """
    ach_path, _size, _mtime = ach_sig
    try:
        return load_achievements(ach_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as e:
        return {
            "__load_error__": f"{type(e).__name__}: {e}",
            "__path__": str(ach_path),
        }

@st.cache_data(show_spinner=False)
def _load_players_artifact_cached(p_sig: Tuple[str, int, int]) -> PlayersArtifact:
    p_path, _size, _mtime = p_sig
    return PlayersArtifact.load_json(p_path)

@st.cache_data(show_spinner=False)
def _load_fitted_params_cached(cfg_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    cfg_path, _size, _mtime = cfg_sig
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            return d
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return {}

@st.cache_data(show_spinner=False)
def _load_ml_confidence_baseline_cached(cfg_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    cfg_path, _size, _mtime = cfg_sig
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            return d
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return {}

@st.cache_data(show_spinner=False)
def _load_team_state_artifact_cached(cfg_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    cfg_path, _size, _mtime = cfg_sig
    d = core_load_team_state_artifact(cfg_path)
    return d if isinstance(d, dict) else {}

@st.cache_data(show_spinner=False)
def _load_league_transfer_features_cached(cfg_sig: Tuple[str, int, int]) -> Dict[str, Any]:
    cfg_path, _size, _mtime = cfg_sig
    d = core_load_league_transfer_features(cfg_path)
    return d if isinstance(d, dict) else {}

def _patch_changed_leagues_set(changed_rows: Any) -> set:
    return _patch_changed_leagues_set_core(changed_rows)

@st.cache_data(show_spinner=False)
def _load_players_impact(players_sig: tuple) -> pd.DataFrame:
    csv_path = str(players_sig[0])
    return core_load_players_impact(
        csv_path=csv_path,
        norm_pos_fn=_norm_pos,
        parse_pct_fn=_parse_pct,
    )

