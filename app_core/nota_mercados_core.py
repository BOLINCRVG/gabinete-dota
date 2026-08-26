"""EM QUE MERCADO O NUMERO DA LINHA PRESTA (medido 26/08/2026).

A Analise de Linha serve numero para cinco mercados, e ate 26/08 ninguem sabia
se algum deles prestava. Medido (replay prequencial, 2026 cego, 3.010 partidas),
o ganho sobre chutar a taxa-base do over:

    kills       +0.0139   <- o unico com sinal de verdade
    time        +0.0054
    barons      +0.0038   (roshans)
    towers      +0.0007   <- nao sabe nada
    inhibitors  -0.0015   <- PIOR que a taxa-base (barracas)

O ganho sobre a taxa-base e o que vale, nao o Brier: Brier premia LINHA TORTA
(se o over so bate 32% das vezes, prever 32% fixo ja pontua bem sabendo nada).

O MESMO modulo roda nos dois repos (como o nota_liga_core): le
evidence/dota_totals_scorecard.json no Dota e evidence/totals_scorecard.json no
LoL, que tem schema diferente mas a mesma informacao. Mantenha os dois iguais.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Cortes na MESMA barra que o projeto usa nos gates: +0.002 e o minimo para
# um ganho contar (ACCEPT), entao abaixo disso o mercado e "nao presta"; 0.005
# (2.5x a barra) e onde da pra confiar no numero.
_GANHO_BOM = 0.005
_GANHO_FRACO = 0.002

_ROTULO_DOTA = {"kills": "Kills", "towers": "Torres", "barons": "Roshans",
                "inhibitors": "Barracas", "time": "Tempo"}
_ROTULO_LOL = {"kills": "Kills", "towers": "Torres", "dragons": "Dragoes",
               "barons": "Baroes", "inhibitors": "Inibidores", "time": "Tempo"}


def load_mercados(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Devolve [{mercado, ganho_base}] + o mapa de rotulos do jogo certo."""
    ev = Path(root) / "evidence"
    try:
        d = json.loads((ev / "dota_totals_scorecard.json").read_text(encoding="utf-8"))
        return list(d.get("mercados") or []), _ROTULO_DOTA
    except (OSError, ValueError, TypeError):
        pass
    try:
        d = json.loads((ev / "totals_scorecard.json").read_text(encoding="utf-8"))
        ganhos = d.get("ganho_vs_base") or {}
        return ([{"mercado": k, "ganho_base": v} for k, v in ganhos.items()],
                _ROTULO_LOL)
    except (OSError, ValueError, TypeError):
        return [], _ROTULO_LOL


def render_confianca_mercados(*, st: Any, root: Path) -> None:
    """Uma linha no topo da aba: onde o numero presta e onde nao presta."""
    linhas, rotulos = load_mercados(root)
    if not linhas:
        return
    bons, fracos, ruins = [], [], []
    for m in linhas:
        try:
            ganho = float(m.get("ganho_base"))
        except (TypeError, ValueError):
            continue
        nome = rotulos.get(str(m.get("mercado")), str(m.get("mercado")))
        (bons if ganho >= _GANHO_BOM else
         fracos if ganho >= _GANHO_FRACO else ruins).append(nome)
    if not (bons or fracos or ruins):
        return
    partes = []
    if bons:
        partes.append("🟢 confiavel: " + ", ".join(bons))
    if fracos:
        partes.append("🟡 fraco: " + ", ".join(fracos))
    if ruins:
        partes.append("🔴 nao presta: " + ", ".join(ruins))
    st.caption(
        " · ".join(partes) +
        ". Medido em replay cego (2026): ganho do modelo sobre chutar a "
        "frequencia historica do over. 🔴 = o numero da tela nao carrega "
        "informacao sobre os times naquele mercado; use so a sua leitura."
    )
