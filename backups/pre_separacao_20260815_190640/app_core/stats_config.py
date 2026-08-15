"""Market configuration for championship stats views.

Fonte ÚNICA dos mercados de stats (kills/torres/dragões/barões/inibidores/tempo).
Adicionar/remover um desses mercados = editar SÓ este dicionário; os mapas usados
em outras partes (SC_METRIC_COL, RESUMO_DEFAULT_LINES, _MARKET_KEYS, _METRIC_COL)
são DERIVADOS daqui pelos helpers abaixo, em vez de duplicados.
"""
from __future__ import annotations

from typing import Any

# Campos v3_* alimentam a Análise de Linha (V3): v3_label/v3_key (o V3 usa
# "inibs", não "inhibitors"), mv2_col (total combinado calculado no augment),
# for/against (colunas por lado; Tempo não tem) — ver v3_core_markets().
# line_key = chave do texto de linha no session_state/flow ("lines_inhib" ≠
# "lines_inhibitors": o sufixo histórico não segue o metric_key).
STATS_MARKET_CONFIG: dict[str, dict[str, Any]] = {
    "Kills": {
        "metric_key": "kills",
        "combined_col": "total_kills",
        "threshold_key": "sc_th_kills",
        "default_thresholds": "23.5,25.5,27.5,29.5",
        "default_line": 27.5,
        "pct_prefix": "% Kills > ",
        "v3_label": "Kills", "v3_key": "kills", "mv2_col": "mv2_total_kills",
        "for_col": "kills_for", "against_col": "kills_against", "is_time": False,
        "line_key": "lines_kills",
    },
    "Torres": {
        "metric_key": "towers",
        "combined_col": "total_towers",
        "threshold_key": "sc_th_towers",
        "default_thresholds": "10.5,11.5,12.5",
        "default_line": 11.5,
        "pct_prefix": "% Towers > ",
        "v3_label": "Torres", "v3_key": "towers", "mv2_col": "mv2_total_towers",
        "for_col": "towers_for", "against_col": "towers_against", "is_time": False,
        "line_key": "lines_towers",
    },
    "Dragões": {
        "metric_key": "dragons",
        "combined_col": "total_dragons",
        "threshold_key": "sc_th_dragons",
        "default_thresholds": "4.5,5.5",
        "default_line": 4.5,
        "pct_prefix": "% Dragons > ",
        "v3_label": "Dragões", "v3_key": "dragons", "mv2_col": "mv2_total_dragons",
        "for_col": "dragons_for", "against_col": "dragons_against", "is_time": False,
        "line_key": "lines_dragons",
    },
    "Nashors": {
        "metric_key": "barons",
        "combined_col": "total_nashors",
        "threshold_key": "sc_th_barons",
        "default_thresholds": "1.5",
        "default_line": 1.5,
        "pct_prefix": "% Nashors > ",
        "v3_label": "Barões", "v3_key": "barons", "mv2_col": "mv2_total_barons",
        "for_col": "barons_for", "against_col": "barons_against", "is_time": False,
        "line_key": "lines_barons",
    },
    "Inibidores": {
        "metric_key": "inhibitors",
        "combined_col": "total_inhibitors",
        "threshold_key": "sc_th_inhib",
        "default_thresholds": "1.5",
        "default_line": 1.5,
        "pct_prefix": "% Inhib > ",
        "v3_label": "Inibidores", "v3_key": "inibs", "mv2_col": "mv2_total_inhibitors",
        "for_col": "inhibitors_for", "against_col": "inhibitors_against", "is_time": False,
        "line_key": "lines_inhib",
    },
    "Tempo": {
        "metric_key": "time",
        "combined_col": "game_time_min",
        "threshold_key": "sc_th_time",
        "default_thresholds": "",
        "default_line": 32.5,
        "pct_prefix": "% Game Time > ",
        "v3_label": "Tempo", "v3_key": "time", "mv2_col": "game_time_min",
        "for_col": None, "against_col": None, "is_time": True,
        "line_key": "lines_time",
    },
}


# --- Helpers derivados (mapas que antes eram duplicados em outros módulos) ---

def combined_col_map() -> dict[str, str]:
    """{metric_key -> combined_col}. Fonte do SC_METRIC_COL do Resumo."""
    return {str(cfg["metric_key"]): str(cfg["combined_col"]) for cfg in STATS_MARKET_CONFIG.values()}


def default_lines_map() -> dict[str, list[float]]:
    """{metric_key -> [default_line]}. Fonte do RESUMO_DEFAULT_LINES."""
    return {str(cfg["metric_key"]): [float(cfg["default_line"])] for cfg in STATS_MARKET_CONFIG.values()}


def market_key_by_label() -> dict[str, str]:
    """{label -> metric_key}. Fonte do _MARKET_KEYS da Análise de Linha."""
    return {label: str(cfg["metric_key"]) for label, cfg in STATS_MARKET_CONFIG.items()}


def _line_key_of(cfg: dict[str, Any]) -> str:
    """line_key do mercado, com fallback "lines_<metric_key>" — mercados
    injetados por override (ex.: Megacreeps do Dota) podem não trazer o campo."""
    return str(cfg.get("line_key") or ("lines_" + str(cfg["metric_key"])))


def flow_line_keys() -> tuple[str, ...]:
    """Chaves de texto de linha dos mercados core, na ordem do registro
    ("lines_kills", ..., "lines_time"). Fonte dos blocos core de
    LINE_FIELDS_DEFAULTS, _FLOW_LINE_KEYS, _SCHEDULE_LINE_FIELDS e _LINE_KEYS
    (fase 3 do registro central; equivalência em test_flow_line_keys_registry)."""
    return tuple(_line_key_of(cfg) for cfg in STATS_MARKET_CONFIG.values())


def line_key_by_label() -> dict[str, str]:
    """{label -> line_key}. Fonte dos inputs de linha da Agenda."""
    return {label: _line_key_of(cfg) for label, cfg in STATS_MARKET_CONFIG.items()}


def v3_core_markets() -> list[dict[str, Any]]:
    """Os 6 mercados core no formato do _V3_MARKETS da Análise de Linha
    (passo 2 do registro central, 2026-07-06). Extras V3 (winner/handicaps/
    first_x/kills_team/ml_totals) continuam definidos lá — são de UI, não de
    mercado de stats. Equivalência travada em test_v3_markets_registry.py."""
    return [
        {
            "label": str(cfg["v3_label"]),
            "key": str(cfg["v3_key"]),
            "metric_col": str(cfg["mv2_col"]),
            "default_line": float(cfg["default_line"]),
            "status": "ready",
            "metric_label": str(cfg["v3_label"]),
            "for_col": cfg["for_col"],
            "against_col": cfg["against_col"],
            "metric_decimals": 1,
            "is_time_metric": bool(cfg["is_time"]),
        }
        for cfg in STATS_MARKET_CONFIG.values()
    ]
