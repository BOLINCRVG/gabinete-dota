from __future__ import annotations

from pathlib import Path
from typing import Dict, List

TAB_ALL: List[str] = [
    "Seus sinais",
    "Visão Geral",
    "Modelo × Mercado",
    "Consistência das linhas",
    "Series",
    "Players",
    "Campeões",
    "StatsCamps",
    "Rankings",
    "Análise de Linha",
    "Parâmetros",
    "Identidade",
    "Melhores apostas",
]

TAB_GROUPS: Dict[str, List[str]] = {
    # StatsCamps fica antes de Rankings em todos os modos do dia-a-dia.
    "Fluxo Rápido": [
        "Seus sinais",
        "Visão Geral",
        "Modelo × Mercado",
        "Análise de Linha",
        "Series",
        "Players",
        "Campeões",
        "StatsCamps",
        "Rankings",
        "Melhores apostas",
    ],
    "Mercados": [
        "Seus sinais",
        "Visão Geral",
        "Modelo × Mercado",
        "Análise de Linha",
        "Series",
        "Players",
        "Campeões",
        "StatsCamps",
        "Rankings",
        "Melhores apostas",
    ],
    "Análise": [
        "Seus sinais",
        "Visão Geral",
        "Modelo × Mercado",
        "Análise de Linha",
        "Series",
        "Campeões",
        "Players",
        "StatsCamps",
        "Rankings",
        "Melhores apostas",
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


# "Seus sinais" (28/10/2026) e so do LoL: os pesos dos sinais foram medidos la.
# Arquivo identico nos dois repositorios; no Dota a aba sai de todas as listas.
if (Path(__file__).resolve().parents[1] / "dota_paths.py").exists():
    TAB_ALL[:] = [a for a in TAB_ALL if a != "Seus sinais"]
    for _abas in TAB_GROUPS.values():
        if _abas is not TAB_ALL:
            _abas[:] = [a for a in _abas if a != "Seus sinais"]


def pick_tabs(mode: str) -> List[str]:
    return list(TAB_GROUPS.get(str(mode), TAB_ALL))
