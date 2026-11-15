# -*- coding: utf-8 -*-
"""Na aba "Melhores apostas": o VALOR NAS SUAS CASAS agora e o PLACAR do que ja foi indicado
(15/11/2026). Arquivo identico nos dois repositorios. A conta esta em placar_apostas_core."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping


def _hora_br(v: Any) -> str:
    from app_core.placar_apostas_core import _quando
    t = _quando(v)
    return "" if t is None else (t - timedelta(hours=3)).strftime("%d/%m %H:%M")


def apostas_de_agora(raiz: Path, casas: Any = None) -> List[Dict[str, Any]]:
    """Todas as apostas com valor nas casas dele, de todos os jogos coletados (regua: Pinnacle)."""
    from app_core import placar_apostas_core as pac
    from app_core import suas_casas_core as sc
    doc = sc.carregar(Path(raiz))
    try:
        pin = json.loads((Path(raiz) / "odds_pinnacle.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pin = {}
    fora: List[Dict[str, Any]] = []
    for jogo in doc.get("jogos") or []:
        fora += pac.candidatos(jogo, pin, sc.esporte(Path(raiz)), casas)
    return sorted(fora, key=lambda a: (not a["estrela"], -a["valor_pct"]))


def gravadas(raiz: Path) -> List[Dict[str, Any]]:
    from app_core import suas_casas_core as sc
    arq = Path(raiz) / f"apostas_{sc.esporte(Path(raiz))}.jsonl"
    fora = []
    try:
        for l in arq.read_text(encoding="utf-8").splitlines():
            try:
                fora.append(json.loads(l))
            except ValueError:
                continue
    except OSError:
        pass
    return fora


def tabela_de_agora(aps: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [{"": "★" if a["estrela"] else "", "Jogo": f"{a['time1']} x {a['time2']}", "Início": _hora_br(a["inicio_utc"]),
             "Aposta": a["nome"], "Casa": a["casa"], "Odd": a["odd"], "Justa (Pinnacle)": a["justa"],
             "Valor %": a["valor_pct"]} for a in aps]


def tabela_das_fechadas(aps: List[Mapping[str, Any]], n: int = 30) -> List[Dict[str, Any]]:
    rot = {"ganhou": "ganhou", "perdeu": "perdeu", "anulada": "anulada", "sem_resultado": "sem resultado"}
    fech = [a for a in aps if a.get("resultado")]
    fech.sort(key=lambda a: str(a.get("inicio_utc")), reverse=True)
    return [{"Jogo": f"{a['time1']} x {a['time2']}", "Início": _hora_br(a["inicio_utc"]), "Aposta": a["nome"],
             "Casa": a["casa"], "Odd": a["odd"], "Valor %": a["valor_pct"], "Resultado": rot.get(a["resultado"], a["resultado"]),
             "Lucro": a.get("lucro")} for a in fech[:n]]


def render_valor_nas_casas(st: Any, raiz: Path) -> None:
    from app_core import placar_apostas_core as pac
    st.markdown("### Valor nas suas casas agora")
    aps = apostas_de_agora(Path(raiz))
    if aps:
        st.dataframe(tabela_de_agora(aps), hide_index=True, use_container_width=True)
        st.caption("Stake e Polymarket pagando MAIS que o preço justo da Pinnacle (sem margem) no mesmo mercado e "
                   "na mesma linha, valor de 1% a 20%. ★ = justa abaixo de 1,72. A Polymarket é a odd de compra, "
                   "lida de hora em hora; confira na casa antes de apostar.")
    else:
        st.caption("Nenhuma aposta nas suas casas pagando mais que a justa da Pinnacle agora "
                   "(precisa do mesmo jogo na Pinnacle e na Stake/Polymarket).")
    gr = gravadas(Path(raiz))
    st.caption(pac.texto_do_resumo(pac.resumo(gr)))
    fech = tabela_das_fechadas(gr)
    if fech:
        with st.expander(f"Últimas apostas fechadas ({len(fech)})"):
            st.dataframe(fech, hide_index=True, use_container_width=True)
