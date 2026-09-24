# -*- coding: utf-8 -*-
# app_core/tab_dispatcher.py
# Centraliza definições e despacho de tabs do Streamlit.
# Fonte única de verdade para rótulos e ordem das tabs.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Constantes de rótulo — use estas em vez de strings literais espalhadas
# ---------------------------------------------------------------------------

TAB_PARAMS = "Parâmetros"
TAB_VISAO = "Visão Geral"
TAB_CONS = "Consistência das linhas"
TAB_RANKINGS = "Rankings"
TAB_JOGOS = "Jogos"
TAB_SERIES = "Series"
TAB_STATSCAMPS = "StatsCamps"
TAB_CAMPEOES = "Campeões"
TAB_PLAYERS = "Players"
TAB_IDENTIDADE = "Identidade"
TAB_LINE_ANALYSIS = "Análise de Linha"
TAB_OPORTUNIDADES = "Oportunidades"

# Ordem canônica das tabs secundárias
SECONDARY_TAB_ORDER: tuple[str, ...] = (
    TAB_RANKINGS,
    TAB_JOGOS,
    TAB_SERIES,
    TAB_STATSCAMPS,
    TAB_LINE_ANALYSIS,
    TAB_OPORTUNIDADES,
    TAB_CAMPEOES,
    TAB_PLAYERS,
    TAB_IDENTIDADE,
)

# Todas as tabs conhecidas (primárias + secundárias)
ALL_TABS: tuple[str, ...] = (
    TAB_PARAMS,
    TAB_VISAO,
    TAB_CONS,
    *SECONDARY_TAB_ORDER,
)


# ---------------------------------------------------------------------------
# TabConfig — metadados por tab
# ---------------------------------------------------------------------------

@dataclass
class TabConfig:
    label: str
    render_fn: Callable[[], None]


# ---------------------------------------------------------------------------
# Helpers de despacho
# ---------------------------------------------------------------------------



def dispatch_secondary_tabs(
    st: Any,
    tab_refs: dict[str, Any],
    renderers: dict[str, Callable[[], None]],
    order: tuple[str, ...] = SECONDARY_TAB_ORDER,
) -> None:
    """Percorre as tabs secundárias na ordem canônica e chama cada renderer."""
    for tab_name in order:
        tab_ref = tab_refs.get(tab_name)
        if tab_ref is None:
            continue
        render_fn = renderers.get(tab_name)
        if render_fn is None:
            continue
        with tab_ref:
            render_fn()
