# -*- coding: utf-8 -*-
"""O desenho da aba "Modelo x Mercado", no tema da Linha justa (23/10/2026).

Ele, depois da primeira reorganizacao: "alem de confuso esta feio" e "quero
que siga a tematica do modelo + da aba". A Linha justa (tabela_casa_core) e o
padrao: cartoes escuros, borda de cima colorida por fonte, ODDS no formato da
casa (nao porcentagem), verde onde ha valor. Mesmo CSS, mesmas cores:
  modelo #2daac4 · Pinnacle #d9a441 · Polymarket #7c8cf8 · valor #3fb950.

Dois blocos, so HTML (puro, testavel):
  - html_destaques: as melhores apostas do dia, em cartoes;
  - html_vencedor: Modelo | Pinnacle | Polymarket, como as colunas da Linha justa.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from html import escape
from typing import Any, List, Mapping, Optional

CSS = """
<style>
  :root { --texto:#e6e6e6; --fraco:#9aa0a6; --painel:rgba(255,255,255,0.035);
          --borda:rgba(255,255,255,0.12); --cab:rgba(255,255,255,0.06); --linha:rgba(255,255,255,0.07);
          --modelo:#2daac4; --pin:#d9a441; --poly:#7c8cf8; --boa:#3fb950; --ruim:#c9736a;
          --verde-fundo:rgba(63,185,80,0.14); --verde-borda:rgba(63,185,80,0.55); }
  @media (prefers-color-scheme: light) {
    :root { --texto:#1f2328; --fraco:#6a737d; --painel:rgba(0,0,0,0.025); --borda:rgba(0,0,0,0.12);
            --cab:rgba(0,0,0,0.045); --linha:rgba(0,0,0,0.07); }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:transparent; color:var(--texto);
         font-family:"Source Sans Pro","Segoe UI",Arial,sans-serif; font-size:13.5px; }
  .status { color:var(--fraco); font-size:12px; padding:2px 2px 10px; display:flex; gap:14px; flex-wrap:wrap; }
  .status b { color:var(--texto); font-weight:700; }
  .status .velho { color:var(--pin); }
  .grade { display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr)); gap:12px; }
  .card { background:var(--painel); border:1px solid var(--borda); border-top:3px solid var(--modelo);
          border-radius:12px; padding:10px 12px; display:flex; flex-direction:column; gap:6px; }
  .card .jogo { font-weight:700; font-size:13px; }
  .card .meta { color:var(--fraco); font-size:11.5px; }
  .card .merc { color:var(--fraco); font-size:12px; margin-top:2px; }
  .card .aposta { font-size:15.5px; font-weight:800; }
  .card .odds { display:flex; align-items:center; gap:10px; margin-top:2px; }
  .rot { color:var(--fraco); font-size:11px; text-transform:uppercase; letter-spacing:0.06em; }
  .odd { font-weight:800; }
  .o-pin { color:var(--pin); } .o-mod { color:var(--modelo); } .o-poly { color:var(--poly); }
  .badge { margin-left:auto; font-weight:800; font-size:12px; color:var(--boa);
           background:var(--verde-fundo); border:1px solid var(--verde-borda); border-radius:8px; padding:2px 8px; }
  .colunas { display:flex; gap:14px; align-items:flex-start; flex-wrap:wrap; }
  .coluna { flex:1 1 260px; min-width:240px; }
  .nome-col { font-weight:800; letter-spacing:0.07em; font-size:12px; padding:0 2px 6px; text-transform:uppercase; }
  .c-mod .nome-col { color:var(--modelo); } .c-pin .nome-col { color:var(--pin); } .c-poly .nome-col { color:var(--poly); }
  .secao { background:var(--painel); border:1px solid var(--borda); border-top:3px solid var(--modelo);
           border-radius:12px; overflow:hidden; }
  .c-pin .secao { border-top-color:var(--pin); } .c-poly .secao { border-top-color:var(--poly); }
  .secao h3 { margin:0; padding:10px 12px; font-size:13.5px; font-weight:700; }
  .cab, .row { display:grid; grid-template-columns:1fr 1fr; }
  .cab { background:var(--cab); font-weight:700; font-size:11.5px; opacity:0.85; }
  .cab div { padding:7px 6px; text-align:center; }
  .row { border-top:1px solid var(--linha); }
  .row > div { min-height:45px; display:flex; align-items:center; justify-content:center; gap:7px; }
  .row > div + div { border-left:1px solid var(--linha); }
  .c-mod .odd { color:var(--modelo); } .c-pin .odd { color:var(--pin); } .c-poly .odd { color:var(--poly); }
  .verde { background:var(--verde-fundo); box-shadow:inset 0 0 0 1px var(--verde-borda); }
  .v { font-size:11px; font-weight:700; } .v.b { color:var(--boa); } .v.r { color:var(--ruim); }
  .nada { color:var(--fraco); font-size:11.5px; }
  .selo { margin-top:10px; font-size:12.5px; padding:7px 10px; border-radius:8px; border:1px solid var(--borda); }
  .selo.forte, .selo.casas { border-color:var(--verde-borda); background:var(--verde-fundo); }
  .selo.desc { border-color:rgba(217,164,65,0.55); color:var(--pin); }
  .nota { color:var(--fraco); font-size:11.5px; padding:8px 2px 0; line-height:1.45; }
  .faixa { display:flex; align-items:center; gap:22px; flex-wrap:wrap; background:var(--painel);
           border:1px solid var(--borda); border-left:3px solid var(--poly); border-radius:10px; padding:9px 12px; }
</style>
"""


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _odd(x: Any) -> str:
    v = _num(x)
    return "-" if v is None or v <= 1 else f"{v:.2f}"


def _hora_br(iso: Any) -> str:
    """'2026-10-23T23:00:00Z' -> '23/10 20:00' (Brasilia, UTC-3)."""
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00").replace(" ", "T"))
        if dt.utcoffset() is not None:
            dt = (dt - dt.utcoffset()).replace(tzinfo=None)
        dt = dt - timedelta(hours=3)
        return dt.strftime("%d/%m %H:%M")
    except (TypeError, ValueError):
        return ""


def html_status(*, jogos: int, com_valor: int, idade_pin: str, idade_poly: str,
                pin_velha: bool = False) -> str:
    return ('<div class="status"><span><b>{}</b> jogos na Pinnacle</span>'
            '<span><b>{}</b> apostas com valor</span>'
            '<span>Pinnacle <b class="{}">{}</b></span><span>Polymarket <b>{}</b></span></div>').format(
        jogos, com_valor, "velho" if pin_velha else "", escape(idade_pin), escape(idade_poly))


def html_destaques(linhas: List[Mapping[str, Any]], aposta_fn, *, limite: int = 8) -> str:
    """Cartoes das melhores apostas do dia (ja ordenadas)."""
    cards = []
    for x in linhas[:limite]:
        v = _num(x.get("vantagem_pct"))
        cards.append(
            '<div class="card"><div class="jogo">{} x {}</div>'
            '<div class="meta">{} · {}</div>'
            '<div class="merc">{}</div><div class="aposta">{}</div>'
            '<div class="odds"><span class="rot">Pinnacle</span><span class="odd o-pin">{}</span>'
            '<span class="rot">Modelo</span><span class="odd o-mod">{}</span>'
            '<span class="badge">{}</span></div></div>'.format(
                escape(str(x.get("casa") or "?")), escape(str(x.get("fora") or "?")),
                escape(str(x.get("liga") or "")), escape(_hora_br(x.get("inicio_utc"))),
                escape(str(x.get("nome_pt") or x.get("mercado") or "")), escape(aposta_fn(x)),
                _odd(x.get("odd_casa")), _odd(x.get("odd_modelo")),
                "" if v is None else f"+{v:.1f}%".replace(".", ",")))
    if not cards:
        return '<div class="nada">Nenhuma aposta com vantagem entre 1% e 20% nesta leitura.</div>'
    return '<div class="grade">' + "".join(cards) + "</div>"


def _cel(odd: Any, vant: Any = None, verde: bool = False) -> str:
    v = _num(vant)
    extra = "" if v is None else ' <span class="v {}">{:+.1f}%</span>'.format("b" if v > 0 else "r", v)
    return '<div class="{}"><span class="odd">{}</span>{}</div>'.format(
        "verde" if verde else "", _odd(odd), extra.replace(".", ","))


def html_vencedor(comp: Mapping[str, Any], *, time_a: str, time_b: str, bo: int,
                  rotulos: Mapping[str, str], fontes_faltando: List[str]) -> str:
    """Modelo | Pinnacle | Polymarket, no desenho da Linha justa. Odds, nao %."""
    la, lb = (comp.get("linhas") or [{}, {}])[:2]
    titulo = "Série - Vencedor (BO{})".format(bo) if bo > 1 else "Vencedor (BO1)"
    cab = '<div class="cab"><div>{}</div><div>{}</div></div>'.format(escape(time_a), escape(time_b))

    def coluna(classe, nome, celulas):
        return ('<div class="coluna {}"><div class="nome-col">{}</div><div class="secao"><h3>{}</h3>{}'
                '<div class="row">{}</div></div></div>').format(classe, nome, titulo, cab, celulas)

    def _inv(p):
        p = _num(p)
        return None if not p else 1 / p

    mod = _cel(_inv(la.get("p_modelo"))) + _cel(_inv(lb.get("p_modelo")))
    pin = (_cel(la.get("odd_pinnacle"), la.get("vant_pinnacle"), (_num(la.get("vant_pinnacle")) or 0) > 0)
           + _cel(lb.get("odd_pinnacle"), lb.get("vant_pinnacle"), (_num(lb.get("vant_pinnacle")) or 0) > 0))
    poly = (_cel(_inv(la.get("preco_compra_poly")), la.get("vant_polymarket"),
                 (_num(la.get("vant_polymarket")) or 0) > 0)
            + _cel(_inv(lb.get("preco_compra_poly")), lb.get("vant_polymarket"),
                   (_num(lb.get("vant_polymarket")) or 0) > 0))
    html = '<div class="colunas">' + coluna("c-mod", "Modelo", mod) + coluna("c-pin", "Pinnacle", pin) \
        + coluna("c-poly", "Polymarket", poly) + "</div>"
    classe = {"forte": "forte", "entre casas": "casas", "suspeito": "desc"}
    for l, nome in ((la, time_a), (lb, time_b)):
        s = l.get("selo") or ""
        if s and rotulos.get(s):
            html += '<div class="selo {}"><b>{}</b>: {}.</div>'.format(
                classe.get(s, ""), escape(nome), escape(rotulos[s]))
    for l, nome in ((la, time_a), (lb, time_b)):
        v = _num(l.get("vant_entre_casas"))
        if v is not None and v > 0 and l.get("selo") not in ("forte", "entre casas"):
            html += '<div class="nota">{}: a Polymarket esta {:.1f}% mais barata que a Pinnacle.</div>'.format(
                escape(nome), v).replace(".", ",", 1)
    if comp.get("casas_discordam"):
        html += '<div class="nota">As duas casas discordam entre si em {:.0f} pontos: mercado dividido.</div>'.format(
            100 * (_num(comp.get("diferenca_casas")) or 0))
    if fontes_faltando:
        html += '<div class="nota">{}.</div>'.format(escape(" · ".join(fontes_faltando)))
    html += ('<div class="nota">Odds: no modelo, a justa (1/chance); na Pinnacle, a que ela paga; '
             'na Polymarket, 1/preço de compra. Verde: o modelo vê valor nesse preço.</div>')
    return html


def html_faixa_polymarket(comp: Mapping[str, Any], *, time_a: str, time_b: str,
                          rotulos: Mapping[str, str], fontes_faltando: List[str]) -> str:
    """So o que a Polymarket ACRESCENTA a Linha justa (24/10/2026).

    O vencedor em Modelo | Pinnacle ja esta na Linha justa logo abaixo; repetir
    o bloco inteiro era a terceira vez que a tela dizia a mesma coisa. Aqui:
    a odd da Polymarket de cada time (com a vantagem do modelo) e o selo."""
    la, lb = (comp.get("linhas") or [{}, {}])[:2]

    def _inv(p):
        p = _num(p)
        return None if not p else 1 / p

    def _lado(l, nome):
        v = _num(l.get("vant_polymarket"))
        cor = "" if v is None else ' <span class="v {}">{:+.1f}%</span>'.format("b" if v > 0 else "r", v)
        return '<span>{} <span class="odd o-poly">{}</span>{}</span>'.format(
            escape(nome), _odd(_inv(l.get("preco_compra_poly"))), cor.replace(".", ","))

    tem_poly = la.get("preco_compra_poly") is not None
    html = '<div class="faixa"><span class="rot o-poly">Polymarket</span>'
    if tem_poly:
        html += _lado(la, time_a) + _lado(lb, time_b)
    else:
        html += '<span class="nada">{}</span>'.format(
            escape(" · ".join(fontes_faltando) or "sem este confronto"))
    html += "</div>"
    classe = {"forte": "forte", "entre casas": "casas", "suspeito": "desc"}
    for l, nome in ((la, time_a), (lb, time_b)):
        s = l.get("selo") or ""
        if s and rotulos.get(s):
            html += '<div class="selo {}"><b>{}</b>: {}.</div>'.format(
                classe.get(s, ""), escape(nome), escape(rotulos[s]))
    if comp.get("casas_discordam"):
        html += '<div class="nota">Pinnacle e Polymarket discordam em {:.0f} pontos: mercado dividido.</div>'.format(
            100 * (_num(comp.get("diferenca_casas")) or 0))
    return html


def pagina(corpo: str) -> str:
    return "<!doctype html><html><head><meta charset='utf-8'>" + CSS + "</head><body>" + corpo + "</body></html>"
