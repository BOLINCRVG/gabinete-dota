# -*- coding: utf-8 -*-
"""SUAS CASAS x PINNACLE no jogo aberto (28/10/2026; mesmo molde do CS2).

Ele: "a pinnacle vai abrir quase sempre com a odd mais regulada... se
conseguissemos outras casas sinto que seria possivel ver a qualidade do modelo".
O robo (tools/coletar_casas.py) grava odds_casas_<esporte>.json de 3 em 3 h
com TODAS as casas da OddsPapi; esta tela mostra so as DELE.

A PINNACLE E A REGUA, e a do MESMO retrato da OddsPapi (preco justo = sem a
margem, em proporcao). A do nosso robo e de outra hora: comparar momentos
diferentes inventa valor -- o CS ja pagou por isso.
  verde = a casa paga MAIS que o justo da Pinnacle (valor pela regua do mercado)
  ★     = o modelo TAMBEM ve valor nesse preco

Arquivo identico no LoL e no Dota (o esporte sai de dota_paths.py).
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from app_core.mercado_vencedor_core import _chave, _num

# (prefixo do slug na OddsPapi, nome na tela) -- a ordem e a da tabela
MARCAS = (("pinnacle", "Pinnacle"), ("bet365", "bet365"), ("superbet", "Superbet"),
          ("polymarket", "Polymarket"), ("estrelabet", "Estrela Bet"), ("blaze", "Blaze"),
          ("betano", "Betano"), ("stake", "Stake"))
SPORT_ID = {"lol": "18", "dota": "16"}


def esporte(raiz: Path) -> str:
    return "dota" if (Path(raiz) / "dota_paths.py").exists() else "lol"


def mercados(esp: str) -> Dict[str, int]:
    """{id do mercado na OddsPapi: periodo (0 = serie, n = mapa n)}."""
    s = SPORT_ID[esp]
    return {s + "1": 0, s + "47": 1, s + "49": 2, s + "51": 3}


def carregar(raiz: Path) -> Dict[str, Any]:
    p = Path(raiz) / f"odds_casas_{esporte(raiz)}.json"
    if not p.exists():
        return {"jogos": [], "motivo": "o robo ainda nao coletou as suas casas"}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"jogos": [], "motivo": "arquivo das suas casas ilegivel"}


def versao_da_marca(casas: Mapping[str, Any], marca: str) -> Optional[str]:
    """A versao brasileira primeiro (estrelabet.bet.br), depois a global."""
    for slug in (f"{marca}.bet.br", marca):
        if slug in casas:
            return slug
    return None


def achar(jogos: List[Mapping[str, Any]], time_a: str, time_b: str,
          resolver: Optional[Callable[[str], str]] = None):
    """(jogo da OddsPapi, invertido) deste confronto, ou (None, False). Se o par
    tiver mais de um jogo no retrato, o mais cedo."""
    def _nome(n: str) -> str:
        if resolver is not None:
            try:
                r = resolver(n)
                if r:
                    return _chave(r)
            except Exception:
                pass
        return _chave(n)
    alvo = {_chave(time_a), _chave(time_b)}
    achados = []
    for g in jogos:
        a, b = _nome(str(g.get("time1") or "")), _nome(str(g.get("time2") or ""))
        if {a, b} == alvo and a != b:
            achados.append((str(g.get("inicio_utc") or ""), g, a != _chave(time_a)))
    if not achados:
        return None, False
    _, g, inv = min(achados, key=lambda x: x[0])
    return g, inv


def linhas(jogo: Mapping[str, Any], invertido: bool, esp: str, *,
           p_serie: Any = None, p_mapa: Any = None) -> List[Dict[str, Any]]:
    """Uma linha por (mercado, casa dele), na orientacao do Time 1 da tela. Pura.
    p_serie / p_mapa: chance do Time 1 pelo modelo (None = sem opiniao)."""
    casas = jogo.get("casas") or {}
    fora = []
    for mid, per in mercados(esp).items():
        p_mod = _num(p_serie if per == 0 else p_mapa)
        p_pin = None
        if mid in (casas.get("pinnacle") or {}):
            o1, o2 = casas["pinnacle"][mid][::-1] if invertido else casas["pinnacle"][mid]
            p_pin = (1 / o1) / (1 / o1 + 1 / o2)
        for marca, nome in MARCAS:
            slug = versao_da_marca(casas, marca)
            if not slug or mid not in casas[slug]:
                continue
            o1, o2 = casas[slug][mid][::-1] if invertido else casas[slug][mid]
            lin = {"mercado": "Série" if per == 0 else f"Mapa {per}", "periodo": per,
                   "casa": nome, "slug": slug, "odd_a": o1, "odd_b": o2}
            for lado, odd, pp, pm in (("a", o1, p_pin, p_mod),
                                      ("b", o2, None if p_pin is None else 1 - p_pin,
                                       None if p_mod is None else 1 - p_mod)):
                lin[f"valor_pin_{lado}"] = None if pp is None else round((pp * odd - 1) * 100, 1)
                lin[f"valor_mod_{lado}"] = None if pm is None else round((pm * odd - 1) * 100, 1)
            fora.append(lin)
    return fora


CSS_CASAS = """
<style>
  .tc { width:100%; border-collapse:collapse; background:var(--painel); border:1px solid var(--borda);
        border-top:3px solid var(--pin); border-radius:12px; overflow:hidden; font-variant-numeric:tabular-nums; }
  .tc th { background:var(--cab); font-size:11.5px; font-weight:700; padding:7px 8px; text-align:center; opacity:0.85; }
  .tc td { border-top:1px solid var(--linha); padding:5px 8px; text-align:center; }
  .tc td.l { text-align:left; } .tc td.m { color:var(--fraco); font-size:12px; }
  .tc .p { font-weight:800; display:block; } .tc .q { color:var(--fraco); font-size:11px; display:block; }
  .tc td.verde .q { color:var(--boa); font-weight:700; }
  .tc tr.pin td.l { color:var(--pin); font-weight:700; }
</style>
"""


def html(time_a: str, time_b: str, lins: List[Mapping[str, Any]], lido_em: str = "") -> str:
    if not lins:
        return ""

    def cel(odd, vp, vm, pin):
        verde = not pin and vp is not None and vp > 0
        est = " ★" if verde and vm is not None and vm > 0 else ""
        if pin:
            txt = "a régua"
        elif vp is None:
            txt = "sem Pinnacle"
        else:
            txt = "vs Pinnacle " + f"{vp:+.1f}%".replace(".", ",")
        return ('<td class="{}"><span class="p">{:.2f}{}</span><span class="q">{}</span></td>'
                .format("verde" if verde else "", odd, est, txt))
    corpo = "".join(
        '<tr class="{}"><td class="m">{}</td><td class="l">{}</td>{}{}</tr>'.format(
            "pin" if l["casa"] == "Pinnacle" else "", escape(l["mercado"]), escape(l["casa"]),
            cel(l["odd_a"], l["valor_pin_a"], l["valor_mod_a"], l["casa"] == "Pinnacle"),
            cel(l["odd_b"], l["valor_pin_b"], l["valor_mod_b"], l["casa"] == "Pinnacle"))
        for l in sorted(lins, key=lambda l: (l["periodo"], l["casa"] != "Pinnacle", l["casa"])))
    hora = ""
    if lido_em:
        from app_core.modelo_mercado_html_core import _hora_br
        hora = f" Leitura de {escape(_hora_br(lido_em))} (Brasília)."
    return (CSS_CASAS + '<table class="tc"><tr><th>Mercado</th><th style="text-align:left">Casa</th>'
            '<th>{}</th><th>{}</th></tr>'.format(escape(time_a), escape(time_b)) + corpo + '</table>'
            '<div class="nota">Odds das suas casas pela OddsPapi, de 3 em 3 h.{} <b>Verde</b>: a casa paga MAIS '
            'que o preço justo da Pinnacle (sem margem) no mesmo instante. <b>★</b>: o modelo também vê '
            'valor nesse preço.</div>'.format(hora))


def render_suas_casas(st: Any, *, raiz: Path, confronto: Mapping[str, Any],
                      resolver: Optional[Callable[[str], str]]) -> None:
    time_a = str(confronto.get("teamA") or "")
    time_b = str(confronto.get("teamB") or "")
    if not (time_a and time_b):
        return
    doc = carregar(raiz)
    jogo, inv = achar(doc.get("jogos") or [], time_a, time_b, resolver)
    st.markdown("#### Suas casas x Pinnacle")
    if jogo is None:
        st.caption("Nenhuma das suas casas tem este jogo na leitura atual (OddsPapi)"
                   + (f": {doc['motivo']}." if doc.get("motivo") else "."))
        return
    try:
        bo = int(confronto.get("bo") or 1)
    except (TypeError, ValueError):
        bo = 1
    p_mapa = confronto.get("p_map")
    p_serie = p_mapa if bo == 1 else confronto.get("p_series")
    lins = linhas(jogo, inv, esporte(raiz), p_serie=p_serie, p_mapa=p_mapa)
    if not lins:
        st.caption("A OddsPapi tem este jogo, mas nenhuma das suas casas abriu vencedor.")
        return
    from app_core.modelo_mercado_html_core import CSS, pagina
    corpo = html(time_a, time_b, lins, str(doc.get("lido_em") or ""))
    try:
        import streamlit.components.v1 as _componentes
        _componentes.html(pagina(corpo), height=40 + 46 * len(lins) + 60, scrolling=False)
    except Exception:
        st.markdown(CSS + corpo, unsafe_allow_html=True)
