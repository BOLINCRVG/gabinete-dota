# -*- coding: utf-8 -*-
"""Helpers extraídos do plays_app (fatiamento F3, 2026-07-18).

Funções PURAS (não referenciam nada definido no plays_app; sem __file__).
O plays_app importa todas de volta, então o namespace que alimenta o
ctx das abas (globals()->build_render_ctx) segue idêntico.
"""
from __future__ import annotations

from app_core.display_tables_core import (
    df_contract_view as core_df_contract_view,
)
from app_core.lines_table_core import lines_table as _lines_table_core, lines_table_with_p_ml as _lines_table_with_p_ml_core
from app_core.lineup_state import (
    ensure_lineup_runtime_state as _ensure_lineup_runtime_state_impl,
    get_lineup_loaded_key as _get_lineup_loaded_key_impl,
    get_lineup_runtime_settings as _get_lineup_runtime_settings_impl,
    set_lineup_loaded_key as _set_lineup_loaded_key_impl,
    load_lineup_for_matchup as _load_lineup_for_matchup_impl,
    save_lineup_for_matchup as _save_lineup_for_matchup_impl,
    get_lineup_selected as _get_lineup_selected_impl,
)
from app_core.odds_cards_ui_core import render_odds_cards as _render_odds_cards_core
from app_core.trace_core import (
    coerce_feature_flag as core_coerce_feature_flag,
)
from mlcore.league_context import league_macro_key as shared_league_macro_key
from mlcore.lines_engine import (
    matchup_expected_totals,
)
from typing import Any, Dict, List
import pandas as pd
import streamlit as st


def _league_macro_key(league: Any) -> str:
    return str(shared_league_macro_key(league, unknown="OTHER"))

def _clear_streamlit_cache_data() -> None:
    cache_data = getattr(st, "cache_data", None)
    clear_fn = getattr(cache_data, "clear", None) if cache_data is not None else None
    if callable(clear_fn):
        clear_fn()

def _request_streamlit_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return
    experimental_rerun_fn = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun_fn):
        experimental_rerun_fn()

def _load_lineup_for_matchup(key: str) -> None:
    _load_lineup_for_matchup_impl(st.session_state, key)

def _save_lineup_for_matchup(key: str) -> None:
    _save_lineup_for_matchup_impl(st.session_state, key)

def _ensure_lineup_runtime_state() -> Dict[str, Any]:
    return _ensure_lineup_runtime_state_impl(st.session_state)

def _get_lineup_runtime_settings() -> Dict[str, Any]:
    return _get_lineup_runtime_settings_impl(st.session_state)

def _get_lineup_loaded_key() -> str:
    return _get_lineup_loaded_key_impl(st.session_state)

def _set_lineup_loaded_key(key: str) -> str:
    return _set_lineup_loaded_key_impl(st.session_state, key)

def _get_lineup_selected(side: str, *, mode: str = "manual") -> dict:
    return _get_lineup_selected_impl(st.session_state, side, mode=mode)

def _df_contract_view(
    df: pd.DataFrame | None,
    required_cols: list[str],
    *,
    optional_cols: list[str] | None = None,
    context: str = "",
    fill_value: Any = "-",
    warn_on_missing: bool = True,
) -> pd.DataFrame:
    return core_df_contract_view(
        df=df,
        required_cols=required_cols,
        optional_cols=optional_cols,
        context=context,
        fill_value=fill_value,
        warn_on_missing=warn_on_missing,
        warning_fn=st.warning,
    )

def _render_odds_cards(df_raw: pd.DataFrame, mode: str = 'ou', table_label: str = 'Ver tabela') -> None:
    return _render_odds_cards_core(df_raw, mode, table_label, st=st)

def _lines_table(metric: str, lines_list: List[float], total_obj) -> pd.DataFrame:
    return _lines_table_core(metric, lines_list, total_obj, st=st)

def _lines_table_with_p_ml(metric: str, lines_list: List[float], total_obj, p_ml_value: float) -> pd.DataFrame:
    return _lines_table_with_p_ml_core(metric, lines_list, total_obj, p_ml_value, st=st)

def _feature_flag_enabled(flag_key: str, default: bool = False) -> bool:
    """Safe feature-flag read from session_state (tolerates stale/invalid values)."""
    return bool(core_coerce_feature_flag(st.session_state.get(str(flag_key), default), default=default))

def _expected_totals_dc(_df: pd.DataFrame, _pA, _pB) -> Dict[str, Any]:
    """Totais V0 + blend Dixon-Coles nos mercados validados (kills/dragões/
    tempo; gate ACCEPT 2026-07-06, ver mlcore.lines_engine.apply_dc_blend)
    + calibração global do sd do TEMPO (gate ACCEPT 2026-07-17)."""
    _t = matchup_expected_totals(_pA, _pB)
    try:
        from mlcore.lines_engine import apply_dc_blend
        _t = apply_dc_blend(_df, str(getattr(_pA, "team", "")), str(getattr(_pB, "team", "")), _t)
    except Exception:
        pass
    try:
        from mlcore.lines_engine import apply_time_sd_calibration
        return apply_time_sd_calibration(_t)
    except Exception:
        return _t

