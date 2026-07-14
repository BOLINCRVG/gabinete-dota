from __future__ import annotations

from typing import Dict, List

TAB_ALL: List[str] = [
    "Visão Geral",
    "Consistência das linhas",
    "Series",
    "Players",
    "Campeões",
    "StatsCamps",
    "Rankings",
    "Análise de Linha",
    "Parâmetros",
    "Identidade",
]

TAB_GROUPS: Dict[str, List[str]] = {
    # StatsCamps fica antes de Rankings em todos os modos do dia-a-dia.
    "Fluxo Rápido": [
        "Visão Geral",
        "Análise de Linha",
        "Series",
        "Players",
        "Campeões",
        "StatsCamps",
        "Rankings",
    ],
    "Mercados": [
        "Visão Geral",
        "Análise de Linha",
        "Series",
        "Players",
        "Campeões",
        "StatsCamps",
        "Rankings",
    ],
    "Análise": [
        "Visão Geral",
        "Análise de Linha",
        "Series",
        "Campeões",
        "Players",
        "StatsCamps",
        "Rankings",
    ],
    "Completo": TAB_ALL,
    # Dev: ferramentas internas (calibração de parâmetros, mapeamento de
    # identidade, painel de consistência legada).
    "Dev": [
        "Visão Geral",
        "Consistência das linhas",
        "Parâmetros",
        "Identidade",
    ],
}


def pick_tabs(mode: str) -> List[str]:
    return list(TAB_GROUPS.get(str(mode), TAB_ALL))
