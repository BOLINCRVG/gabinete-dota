# -*- coding: utf-8 -*-
"""ML + TOTAL: a odd justa da COMBINADA (29/10/2026, pedido dele).

Ele: "toda odd que eu coloco parece pessima... T1 e under 30,5, eu quero saber a odd
justa daquilo". Duas coisas pesam:
  1. combinar MULTIPLICA: P(A vence e Menos) = P(A vence) x P(Menos | A vence);
  2. o total MUDA conforme quem vence -- medido em 2026: quando o favorito vence o jogo
     e ~4 min mais curto; em kills a diferenca e de 1-2 kills.

MEDIDO (scratchpad combinada.py, cego 2026 com a chance honesta do vencedor):
P(total | quem vence) pelo metodo dele com as janelas dos jogos que o time VENCEU e que
o adversario PERDEU (encolhido para o marginal com poucos jogos), contra multiplicar as
pernas independentes -- placar conjunto: tempo +0,011, torres +0,009, dragoes +0,014
(IC95 todos acima de zero); kills -0,001 (empate). Por isso: condicional em tempo,
torres e dragoes; kills (e baroes/inibidores, nao medidos) multiplicam as pernas.

DOTA (15/11/2026; arquivo identico nos dois repositorios). Mesma medicao, cego 2026 (3,5 mil
jogos, linha central): a condicional NAO bate multiplicar as pernas em nenhum mercado -- kills
-0,0000, torres +0,0015, roshans -0,0032, barracas +0,0021, tempo +0,0022, todos com IC95
cruzando o zero. No Dota a combinada e sempre o produto das pernas (vencedor x total do metodo).
"""
from __future__ import annotations

import math
from html import escape
from typing import Any, Dict, List, Optional

import pandas as pd

def _e_dota() -> bool:
    from pathlib import Path
    return (Path(__file__).resolve().parents[1] / "dota_paths.py").exists()


CONDICIONAL = set() if _e_dota() else {"time", "towers", "dragons"}
JANELA = 10


def _lg(p: float) -> float:
    p = min(max(float(p), 0.02), 0.98)
    return math.log(p / (1 - p))


def _p_metodo(tots: List[float], camp: List[float], L: float, pesos) -> Optional[float]:
    """A chance do Mais pelo metodo dele com as janelas dadas (a mesma conta da medicao)."""
    if len(tots) < 6 or len(camp) < 4:
        return None
    t = pd.Series(tots, dtype=float)
    h = {n: float((t.tail(2 * n) > L).mean()) for n in (5, 10, 20)}
    m = float(t.tail(20).mean()) - L
    c = float((pd.Series(camp) > L).mean())
    b0, wm, wa, w5, w10, w20, wc = pesos
    z = b0 + wm * m + wa * m + w5 * _lg(h[5]) + w10 * _lg(h[10]) + w20 * _lg(h[20]) + wc * _lg(c)
    return 1 / (1 + math.exp(-z))


def combinadas(team_games: pd.DataFrame, time_a: str, time_b: str, linha: float, metrica: str,
               p_vence_a: Any, so_mapa1: bool = False) -> Optional[Dict[str, Any]]:
    """As 4 combinadas (A/B vence x Mais/Menos) com chance, fair e o historico por tras. Pura."""
    from app_core.sinais_linha_core import METRICAS, _camp, _jogos, _total, chance, sinais
    try:
        pa = float(p_vence_a)
    except (TypeError, ValueError):
        return None
    if not (0.0 < pa < 1.0) or metrica not in METRICAS:
        return None
    sn = sinais(team_games, time_a, time_b, linha, so_mapa1=so_mapa1, metrica=metrica)
    if not sn or chance(sn, "over") is None:
        return None
    L = float(linha)
    p_mais = chance(sn, "over")
    da, db = _jogos(team_games, time_a, so_mapa1), _jogos(team_games, time_b, so_mapa1)
    win = lambda d: pd.to_numeric(d.get("win"), errors="coerce").fillna(0).astype(int)
    ta, tb = _total(da, metrica), _total(db, metrica)
    va, vb = win(da), win(db)
    camp = list(_camp(da, metrica)) + list(_camp(db, metrica))

    def _ult(t, mask):
        return [float(x) for x in t[mask].dropna().tail(JANELA)]
    jan = {"a": (_ult(ta, va == 1) + _ult(tb, vb == 0)), "b": (_ult(ta, va == 0) + _ult(tb, vb == 1))}
    cond = metrica in CONDICIONAL
    p_mais_dado = {}
    for k in ("a", "b"):
        pv = _p_metodo(jan[k], camp, L, METRICAS[metrica]["pesos"]) if cond else None
        w = min(1.0, len(jan[k]) / (2 * JANELA))
        p_mais_dado[k] = p_mais if pv is None else w * pv + (1 - w) * p_mais
    # H2H: os jogos entre os dois
    h2h = da[da["opponent"].astype(str) == str(time_b)]
    th = _total(h2h, metrica).dropna()
    vh = win(h2h).reindex(th.index)
    out = {"linha": L, "metrica": metrica, "p_vence_a": pa, "p_mais": p_mais, "condicional": cond, "combos": []}
    for quem, pv, k in ((time_a, pa, "a"), (time_b, 1 - pa, "b")):
        janela = jan[k]
        for lado in ("over", "under"):
            pc = p_mais_dado[k] if lado == "over" else 1 - p_mais_dado[k]
            pm = p_mais if lado == "over" else 1 - p_mais
            bate = sum(1 for x in janela if (x > L) == (lado == "over"))
            vence_h2h = (vh == 1) if k == "a" else (vh == 0)
            h_bate = int((vence_h2h & ((th > L) if lado == "over" else (th < L))).sum())
            out["combos"].append({
                "time": quem, "lado": lado, "p_vence": pv, "p_total_dado": pc, "p_total": pm,
                "p": pv * pc, "hist": (bate, len(janela)), "h2h": (h_bate, int(len(th)))})
    return out


def _f(x: Any, casas: int = 1) -> str:
    try:
        return f"{float(x):.{casas}f}".replace(".", ",")
    except (TypeError, ValueError):
        return "-"


def html_combinadas(r: Optional[Dict[str, Any]], unidade: str) -> str:
    if not r:
        return '<div class="nota">Pouco jogo para a combinada (mínimo 5 de cada time).</div>'
    L = _f(r["linha"])
    corpo = ""
    for c in sorted(r["combos"], key=lambda c: -c["p"]):
        lado = "Mais" if c["lado"] == "over" else "Menos"
        muda = (c["p_total_dado"] - c["p_total"]) * 100
        b, n = c["hist"]
        hb, hn = c["h2h"]
        corpo += (
            f'<tr><td class="n">{escape(c["time"])} vence + {lado} {L}</td>'
            f'<td>{_f(100 * c["p_vence"], 0)}%</td>'
            f'<td>{_f(100 * c["p_total_dado"], 0)}%<span class="num">sem condição {_f(100 * c["p_total"], 0)}% '
            f'({"+" if muda >= 0 else ""}{_f(muda, 0)})</span></td>'
            f'<td><b>{_f(100 * c["p"], 1)}%</b></td><td><b>{_f(1 / c["p"], 2) if c["p"] > 0 else "-"}</b></td>'
            f'<td>{b}/{n}<span class="num">nos jogos em que {escape(c["time"])} venceu / o rival perdeu</span></td>'
            f'<td>{hb}/{hn}<span class="num">H2H: venceu e {lado.lower()}</span></td></tr>')
    nota = ("o total foi lido SÓ nos jogos em que o time venceu (e o rival perdeu) — medido: aqui o resultado muda o total."
            if r["condicional"] else
            "em kills, vencer quase não muda o total (medido): a combinada é a multiplicação das pernas.")
    return ("""<style>
      .cb { width:100%; border-collapse:collapse; background:var(--painel); border:1px solid var(--borda);
            border-top:3px solid var(--modelo); border-radius:12px; overflow:hidden; font-variant-numeric:tabular-nums; }
      .cb th { background:var(--cab); font-size:12px; padding:8px; }
      .cb td { border-top:1px solid var(--linha); padding:7px 8px; text-align:center; font-size:13px; }
      .cb td.n { text-align:left; font-weight:700; }
      .cb .num { display:block; color:var(--fraco); font-size:11px; }
    </style>"""
            + '<table class="cb"><tr><th style="text-align:left">Combinada</th><th>Vence</th>'
              f'<th>Total, dado que vence</th><th>Chance</th><th>Odd justa</th><th>Histórico</th><th>H2H</th></tr>'
            + corpo + f'</table><div class="nota">{"Tempo de jogo (min)" if unidade == "min" else "Total de " + escape(unidade)} · {nota} Chance = vence × total dado '
              'que vence. Combinada multiplica: duas pernas de 60% dão 36% (justa 2,78).</div>')


def render_combinadas(*, st: Any, tg: Any, time_a: str, time_b: str, linha: float, p_vence_a: Any,
                      chave: str) -> None:
    from app_core.modelo_mercado_html_core import pagina
    from app_core.sinais_linha_core import METRICAS
    nomes = {"Kills": "kills", "Tempo": "time", "Torres": "towers", "Dragões": "dragons"}
    padrao = {"kills": float(linha), "time": 31.5, "towers": 11.5, "dragons": 4.5}
    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        escolha = st.radio("Total de", list(nomes), horizontal=True, key=f"cb_met_{chave}")
    met = nomes[escolha]
    with c2:
        modo = st.radio("Jogos", ["Todos os mapas", "Só mapa 1"], horizontal=True, key=f"cb_modo_{chave}")
    with c3:
        L = st.number_input("Linha", min_value=0.5, max_value=90.0, step=1.0, value=padrao[met],
                            key=f"cb_linha_{met}_{chave}")
    r = combinadas(tg, time_a, time_b, float(L), met, p_vence_a, so_mapa1=(modo == "Só mapa 1"))
    if r is None and (p_vence_a is None):
        st.caption("Sem a chance de vitória do confronto (abra o jogo pela lista do topo).")
        return
    corpo = html_combinadas(r, METRICAS[met]["unidade"])
    try:
        import streamlit.components.v1 as _cmp
        _cmp.html(pagina(corpo), height=330, scrolling=False)
    except Exception:
        st.markdown(corpo, unsafe_allow_html=True)
