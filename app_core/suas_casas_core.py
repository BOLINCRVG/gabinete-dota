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
# as que ele USA hoje (28/10/2026: "so estou usando basicamente a stake e polymarket");
# a Pinnacle entra sempre, como regua
CASAS_PADRAO = ("Stake", "Polymarket")


# COTACAO FURADA (28/10/2026): a Polymarket deu Mais 1,85 x Menos 1,04 no total de
# kills (soma 150%) -- livro vazio de um lado, preco que ninguem paga. Casa de
# verdade fica em 100-108%; acima de 115% o par e descartado.
SOMA_MAXIMA = 1.15


def par_ok(a: Any, b: Any) -> bool:
    a, b = _num(a), _num(b)
    return bool(a and b and a > 1 and b > 1 and 1 / a + 1 / b <= SOMA_MAXIMA)


def _marcas(casas: Optional[List[str]]):
    """MARCAS filtradas pelas escolhidas (None = todas); a Pinnacle sempre."""
    if casas is None:
        return MARCAS
    return tuple((m, n) for m, n in MARCAS if n == "Pinnacle" or n in casas)


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
           p_serie: Any = None, p_mapa: Any = None, casas_dele: Optional[List[str]] = None) -> List[Dict[str, Any]]:
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
        for marca, nome in _marcas(casas_dele):
            slug = versao_da_marca(casas, marca)
            if not slug or mid not in casas[slug]:
                continue
            o1, o2 = casas[slug][mid][::-1] if invertido else casas[slug][mid]
            if not par_ok(o1, o2):
                continue
            lin = {"mercado": "Série" if per == 0 else f"Mapa {per}", "periodo": per,
                   "casa": nome, "slug": slug, "odd_a": o1, "odd_b": o2}
            for lado, odd, pp, pm in (("a", o1, p_pin, p_mod),
                                      ("b", o2, None if p_pin is None else 1 - p_pin,
                                       None if p_mod is None else 1 - p_mod)):
                lin[f"valor_pin_{lado}"] = None if pp is None else round((pp * odd - 1) * 100, 1)
                lin[f"valor_mod_{lado}"] = None if pm is None else round((pm * odd - 1) * 100, 1)
            fora.append(lin)
    return fora


def linhas_handicap_mapas(jogo: Mapping[str, Any], invertido: bool, esp: str, *, p_mapa: Any = None,
                          p_serie: Any = None, bo: Any = None,
                          casas_dele: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """+-1,5 (e +-2,5) DE MAPAS das casas dele (29/10/2026, ele: "pegar das casas o +1,5 e -1,5
    dos times"). Uma linha por (casa, handicap do Time 1): odd_a = Time 1 com h, odd_b = Time 2
    com -h. Valor vs a Pinnacle do mesmo retrato; o modelo pela forma do dia, coerente com o
    ML da serie (oportunidades_core.prob_handicap_de_mapas). Pura."""
    from app_core.oportunidades_core import prob_handicap_de_mapas
    pares: Dict[Any, Dict[str, Any]] = {}
    for l in linhas_de_mapa(jogo, invertido, esp, casas_dele=casas_dele):
        if l["unidade"] != "maps" or l["tipo"] != "spread" or abs(abs(l["pontos"]) - 1.5) > 1e-9 \
                and abs(abs(l["pontos"]) - 2.5) > 1e-9:
            continue
        h_a = l["pontos"] if l["lado"] == "home" else -l["pontos"]
        pares.setdefault((l["slug"], h_a), {"casa": l["casa_aposta"], "slug": l["slug"]})[
            "a" if l["lado"] == "home" else "b"] = l["odd"]
    pin = {h: v for (slug, h), v in pares.items() if slug == "pinnacle" and "a" in v and "b" in v}
    fora = []
    for (slug, h), v in pares.items():
        if "a" not in v or "b" not in v:
            continue
        p_pin = None
        if h in pin:
            ia, ib = 1 / pin[h]["a"], 1 / pin[h]["b"]
            p_pin = ia / (ia + ib)
        p_mod = None
        try:
            p_mod = prob_handicap_de_mapas({"lado": "home", "pontos": h}, p_map=p_mapa, casa_e_time_a=True,
                                           bo=bo, p_series=p_serie)
        except Exception:
            p_mod = None
        lin = {"mercado": f"Mapas {'+' if h > 0 else ''}{h:g}".replace(".", ","), "periodo": 0.5 + (h + 5) / 100,   # um "periodo" por handicap: o selo "melhor" compara so o mesmo h
               "casa": v["casa"], "slug": slug, "odd_a": v["a"], "odd_b": v["b"], "handicap_a": h}
        for lado, odd, pp, pm in (("a", v["a"], p_pin, p_mod),
                                  ("b", v["b"], None if p_pin is None else 1 - p_pin,
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
  .tc .top { font-size:10px; font-weight:800; color:var(--modelo); border:1px solid var(--modelo);
             border-radius:6px; padding:0 4px; vertical-align:middle; }
  .tc td.esq { text-align:left; } .tc .outras { color:var(--fraco); font-size:11.5px; }
  .tc tr.melhor-linha td:first-child { box-shadow:inset 3px 0 0 var(--boa); }
  .tc .v { font-weight:800; } .tc .v.b { color:var(--boa); } .tc .v.r { color:var(--ruim); }
  .tc .v.d { color:var(--pin); }
</style>
"""


def html(time_a: str, time_b: str, lins: List[Mapping[str, Any]], lido_em: str = "") -> str:
    if not lins:
        return ""

    # a MELHOR odd de cada lado, entre as casas dele (28/10/2026: "ver onde tem a melhor odd")
    melhor = {}
    for l in lins:
        if l["casa"] != "Pinnacle":
            for lado in ("a", "b"):
                k = (l["periodo"], lado)
                melhor[k] = max(melhor.get(k, 0.0), l[f"odd_{lado}"])

    def cel(odd, vp, vm, pin, top):
        verde = not pin and vp is not None and vp > 0
        est = " ★" if verde and vm is not None and vm > 0 else ""
        if pin:
            txt = "a régua"
        elif vp is None:
            txt = "sem Pinnacle"
        else:
            txt = "vs Pinnacle " + f"{vp:+.1f}%".replace(".", ",")
        return ('<td class="{}"><span class="p">{:.2f}{}{}</span><span class="q">{}</span></td>'
                .format("verde" if verde else "", odd, est,
                        ' <span class="top">melhor</span>' if top else "", txt))

    def _top(l, lado):
        return l["casa"] != "Pinnacle" and l[f"odd_{lado}"] >= melhor.get((l["periodo"], lado), 99)
    def _merc(l):
        if l.get("handicap_a") is None:
            return escape(l["mercado"])
        h = l["handicap_a"]
        f = lambda v: f"{'+' if v > 0 else ''}{v:g}".replace(".", ",")
        return f"Mapas<br><span style='font-size:11px'>{escape(time_a)} {f(h)} · {escape(time_b)} {f(-h)}</span>"
    corpo = "".join(
        '<tr class="{}"><td class="m">{}</td><td class="l">{}</td>{}{}</tr>'.format(
            "pin" if l["casa"] == "Pinnacle" else "", _merc(l), escape(l["casa"]),
            cel(l["odd_a"], l["valor_pin_a"], l["valor_mod_a"], l["casa"] == "Pinnacle", _top(l, "a")),
            cel(l["odd_b"], l["valor_pin_b"], l["valor_mod_b"], l["casa"] == "Pinnacle", _top(l, "b")))
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


# ---- LINHAS DE MAPA: a melhor odd e a melhor linha (28/10/2026) ---------------
# Ele: "seria legal ver onde tem a melhor odd e a melhor linha possivel... por
# exemplo para kills". O robo guarda, das casas dele, total / handicap / total por
# time de cada mapa. Aqui cada linha vira o MESMO formato das linhas da Pinnacle
# (app_core/oportunidades_core) e a chance sai das MESMAS contas da Linha justa --
# nao existe uma segunda conta (principio V).
NOME_UNIDADE = {"kills": "Kills", "turrets": "Torres", "towers": "Torres", "dragons": "Dragões",
                "barons": "Barões", "inhibitors": "Inibidores", "roshans": "Roshans",
                "barracks": "Barracks"}


def _nome_da_casa(slug: str) -> str:
    return next((n for m, n in MARCAS if str(slug).startswith(m)), str(slug))


def _favorito_do_mapa(jogo: Mapping[str, Any], esp: str, mapa: int) -> Optional[int]:
    """+1 se o time 1 da OddsPapi e favorito claro no mapa (>= 60%), -1 se e o time 2."""
    mid = next((m for m, per in mercados(esp).items() if per == mapa), None)
    casas = jogo.get("casas") or {}
    for slug in ["pinnacle"] + sorted(casas):
        par = (casas.get(slug) or {}).get(mid)
        if par:
            ia, ib = 1 / par[0], 1 / par[1]
            p1 = ia / (ia + ib)
            return 1 if p1 >= 0.6 else (-1 if p1 <= 0.4 else None)
    return None


def _handicap_trocado(lista: List[Any], fav: Optional[int]) -> bool:
    """A casa publicou o handicap com o sinal do OUTRO time? (28/10/2026: a bet365 veio na
    OddsPapi com o T1, favorito, +8,5 a 1,80 x 1,80, e a Stake -7,5.) Olha a linha mais
    equilibrada da casa: nela o favorito tem de DAR kills."""
    if fav is None or not lista:
        return False
    _, _, _, pontos, oa, ob, _ = min(lista, key=lambda x: abs(x[4] - x[5]))
    return abs(pontos) >= 1.5 and (pontos > 0) == (fav > 0)


def linhas_de_mapa(jogo: Mapping[str, Any], invertido: bool, esp: str = "lol",
                   casas_dele: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Pura: cada lado de cada linha das casas dele, no formato da Pinnacle, com o
    Time 1 da tela como "home". Pontos do handicap = os do lado apostado. So a versao
    da casa que ele usa (estrelabet.bet.br, senao a global) -- nada de bet365-nj, betano.de."""
    fora = []
    todas = jogo.get("linhas") or {}
    usar = [versao_da_marca(todas, m) for m, _ in _marcas(casas_dele)]
    for slug in [u for u in usar if u]:
        lista = todas[slug]
        nome = _nome_da_casa(slug)
        trocar = {}
        for unidade, mapa in {(x[0], x[2]) for x in lista if x[1] == "spread"}:
            trocar[(unidade, mapa)] = _handicap_trocado(
                [x for x in lista if x[1] == "spread" and x[0] == unidade and x[2] == mapa],
                _favorito_do_mapa(jogo, esp, mapa))
        for unidade, tipo, mapa, pontos, oa, ob, qual in lista:
            if not par_ok(oa, ob):
                continue
            if tipo == "spread" and trocar.get((unidade, mapa)):
                pontos = -pontos
            base = {"jogo": 1, "periodo": int(mapa), "unidade": unidade, "tipo": tipo,
                    "casa_aposta": nome, "slug": slug}
            um, dois = ("away", "home") if invertido else ("home", "away")
            if tipo == "spread":
                # o handicap do catalogo e o do time 1 da OddsPapi
                fora.append(dict(base, lado=um, pontos=float(pontos), odd=oa))
                fora.append(dict(base, lado=dois, pontos=-float(pontos), odd=ob))
            else:
                extra = {"time": um if int(qual) == 1 else dois} if tipo == "team_total" else {}
                fora.append(dict(base, lado="over", pontos=float(pontos), odd=oa, **extra))
                fora.append(dict(base, lado="under", pontos=float(pontos), odd=ob, **extra))
    return fora


def _chave_linha(l: Mapping[str, Any]):
    return (l["periodo"], l["unidade"], l["tipo"], l.get("time"), l["lado"], l["pontos"])


def _chave_grupo(l: Mapping[str, Any]):
    """O mesmo mercado e o mesmo lado, so mudando a linha (Mais de 27,5 / 28,5 ...)."""
    return (l["periodo"], l["unidade"], l["tipo"], l.get("time"), l["lado"])


def chance_do_modelo(l: Mapping[str, Any], *, time_a: str, time_b: str,
                     calcular_totais: Optional[Callable] = None, prob_over: Optional[Callable] = None,
                     medias_kills: Optional[Callable] = None, p_casa: Any = None,
                     _cache: Optional[dict] = None) -> Optional[float]:
    """A chance deste lado pela conta de PRODUCAO, ou None (o modelo nao precifica)."""
    from app_core.oportunidades_core import (avaliar_linha, forma_de_kills_por_time,
                                             metrica_da_linha, prob_de_kills_por_time)
    cache = _cache if _cache is not None else {}
    if forma_de_kills_por_time(l):
        if medias_kills is None:
            return None
        if "medias" not in cache:
            try:
                cache["medias"] = medias_kills(time_a, time_b)
            except Exception:
                cache["medias"] = None
        pc = _num(p_casa)
        try:
            return prob_de_kills_por_time(dict(l), cache["medias"],
                                          p_casa=pc if pc is not None and 0 < pc < 1 else None)
        except Exception:
            return None
    alvo = metrica_da_linha(l)
    if alvo is None or calcular_totais is None or prob_over is None:
        return None
    k = ("totais", alvo[1])
    if k not in cache:
        try:
            cache[k] = calcular_totais(time_a, time_b, alvo[1])
        except Exception:
            cache[k] = None
    if cache[k] is None:
        return None
    r = avaliar_linha(dict(l, odd=2.0), cache[k], prob_over=prob_over)
    return None if r is None else r["probabilidade"]


def melhores_linhas(lins: List[Mapping[str, Any]], chance: Callable[[Mapping[str, Any]], Optional[float]]
                    ) -> List[Dict[str, Any]]:
    """Pura (com `chance` de fora): uma linha por (mapa, mercado, lado, pontos) com a MELHOR
    odd entre as casas dele, a Pinnacle sem margem (quando ela tem a MESMA linha) e o
    modelo. `melhor_linha` marca, em cada mercado+lado, a linha de maior valor."""
    por = {}
    for l in lins:
        por.setdefault(_chave_linha(l), []).append(l)
    pin = {k: l["odd"] for k, ls in por.items() for l in ls if l["slug"] == "pinnacle"}

    def _oposto(k):
        per, uni, tipo, tm, lado, pts = k
        if tipo == "spread":
            return (per, uni, tipo, tm, "away" if lado == "home" else "home", -pts)
        return (per, uni, tipo, tm, "under" if lado == "over" else "over", pts)
    fora = []
    for k, ls in por.items():
        casas = sorted((l for l in ls if l["slug"] != "pinnacle"), key=lambda l: -l["odd"])
        if not casas:
            continue
        top = casas[0]
        p_pin = None
        if k in pin and _oposto(k) in pin:
            ia, ib = 1 / pin[k], 1 / pin[_oposto(k)]
            p_pin = ia / (ia + ib)
        p_mod = chance(top)
        fora.append({"periodo": k[0], "unidade": k[1], "tipo": k[2], "time": k[3], "lado": k[4],
                     "pontos": k[5], "odd": top["odd"], "casa": top["casa_aposta"],
                     "outras": [(l["casa_aposta"], l["odd"]) for l in casas[1:]],
                     "p_pinnacle": p_pin, "p_modelo": p_mod,
                     "valor_pin": None if p_pin is None else round((p_pin * top["odd"] - 1) * 100, 1),
                     "valor_mod": None if p_mod is None else round((p_mod * top["odd"] - 1) * 100, 1),
                     "melhor_linha": False})
    grupos = {}
    for r in fora:
        grupos.setdefault(_chave_grupo(r), []).append(r)
    # MELHOR LINHA na logica do resumo (28/10/2026, ele: "bets que tem fair menor que
    # 1,72... eu quero os dois [valor e batida]"): so concorre linha com odd JUSTA abaixo
    # de ODD_MODELO_MAX (~58% ou mais); entre elas, a de maior valor.
    for rs in grupos.values():
        com = [r for r in rs if valor(r) is not None and provavel(r)]
        if com:
            max(com, key=valor)["melhor_linha"] = True
    return fora


def chance(r: Mapping[str, Any]) -> Optional[float]:
    """A chance que manda: a do modelo; sem modelo, a da Pinnacle sem margem."""
    return r["p_modelo"] if r.get("p_modelo") is not None else r.get("p_pinnacle")


def provavel(r: Mapping[str, Any]) -> bool:
    """Odd justa abaixo de 1,72 -- a mesma regra das melhores apostas do resumo."""
    from app_core.modelo_mercado_core import ODD_MODELO_MAX
    p = chance(r)
    return bool(p) and 1 / p < ODD_MODELO_MAX


def valor(r: Mapping[str, Any]) -> Optional[float]:
    """O valor que manda: o do modelo; sem modelo, o contra a Pinnacle."""
    return r["valor_mod"] if r.get("valor_mod") is not None else r.get("valor_pin")


def rotulo_linha(r: Mapping[str, Any], time_a: str, time_b: str) -> tuple:
    """(mercado, aposta) em palavras dele."""
    uni = NOME_UNIDADE.get(r["unidade"], str(r["unidade"]).title())
    pts = f"{r['pontos']:g}".replace(".", ",")
    if r["tipo"] == "spread":
        quem = time_a if r["lado"] == "home" else time_b
        return f"Handicap de {uni.lower()}", f"{quem} {'+' if r['pontos'] > 0 else ''}{pts}"
    lado = "Mais de" if r["lado"] == "over" else "Menos de"
    if r["tipo"] == "team_total":
        quem = time_a if r.get("time") == "home" else time_b
        return f"{uni} de {quem}", f"{lado} {pts}"
    return f"Total de {uni.lower()}", f"{lado} {pts}"


def html_linhas(rs: List[Mapping[str, Any]], time_a: str, time_b: str,
                sinais_fn: Optional[Callable[[Mapping[str, Any]], Optional[str]]] = None) -> str:
    if not rs:
        return ""

    from app_core.oportunidades_core import VANTAGEM_SUSPEITA_PCT

    def _v(x):
        if x is None:
            return '<span class="outras">-</span>'
        if x > VANTAGEM_SUSPEITA_PCT:      # a mesma regra da Linha justa: desconfie do modelo
            return '<span class="v d" title="grande demais: desconfie do numero">{} ⚠</span>'.format(
                f"{x:+.1f}%".replace(".", ","))
        return '<span class="v {}">{}</span>'.format("b" if x > 0 else "r", f"{x:+.1f}%".replace(".", ","))

    def _cel(v, p, falta):
        if not p:
            return '<span class="q">{}</span>'.format(falta)
        return '{}<span class="q">justa {:.2f}</span>'.format(_v(v), 1 / p)
    corpo = ""
    for r in rs:
        merc, aposta = rotulo_linha(r, time_a, time_b)
        outras = " · ".join(f"{escape(c)} {o:.2f}" for c, o in r["outras"][:4])
        sin = ""
        if sinais_fn is not None:
            sin = '<td class="esq" style="font-size:12px">{}</td>'.format(sinais_fn(r) or '<span class="q">-</span>')
        corpo += ('<tr class="{}"><td class="m">Mapa {}</td><td class="esq">{}</td><td class="esq"><b>{}</b>{}</td>'
                  '<td><span class="p">{:.2f}</span><span class="q">{}</span></td>'
                  '<td class="esq outras">{}</td><td>{}</td><td>{}</td>' + sin + '</tr>').format(
            "melhor-linha" if r["melhor_linha"] else "", r["periodo"], escape(merc),
            escape(aposta), ' <span class="top">melhor linha</span>' if r["melhor_linha"] else "",
            r["odd"], escape(r["casa"]), outras or "-",
            _cel(r["valor_pin"], r["p_pinnacle"], "sem esta linha"),
            _cel(r["valor_mod"], r["p_modelo"], "não precifica"))
    return (CSS_CASAS + '<table class="tc"><tr><th>Mapa</th><th style="text-align:left">Mercado</th>'
            '<th style="text-align:left">Aposta</th><th>Melhor odd</th><th style="text-align:left">'
            'Outras casas</th><th>vs Pinnacle</th><th>vs Modelo</th>'
            + ('<th style="text-align:left">Seus sinais (Médias · Adv · Hist · Stats)</th>' if sinais_fn else '')
            + '</tr>' + corpo + '</table>')


def render_suas_casas(st: Any, *, raiz: Path, confronto: Mapping[str, Any],
                      resolver: Optional[Callable[[str], str]], calcular_totais: Optional[Callable] = None,
                      prob_over: Optional[Callable] = None, medias_kills: Optional[Callable] = None,
                      team_games: Any = None, detalhado: bool = False) -> None:
    time_a = str(confronto.get("teamA") or "")
    time_b = str(confronto.get("teamB") or "")
    if not (time_a and time_b):
        return
    doc = carregar(raiz)
    jogo, inv = achar(doc.get("jogos") or [], time_a, time_b, resolver)
    st.markdown("#### Suas casas x Pinnacle · vencedor e ±1,5")
    nomes = [n for _, n in MARCAS if n != "Pinnacle"]
    # o padrao vai para a sessao ANTES do widget: o envelope de multiselect do app
    # (app_core/text_normalization.py) grava [] em chave nova e o `default` nunca valeria
    try:
        st.session_state.setdefault("suas_casas_escolhidas", [n for n in CASAS_PADRAO if n in nomes])
    except Exception:
        pass
    escolhidas = st.multiselect("Casas que você usa (a Pinnacle entra sempre, como régua)", nomes,
                                key="suas_casas_escolhidas")
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
    lins = linhas(jogo, inv, esporte(raiz), p_serie=p_serie, p_mapa=p_mapa, casas_dele=escolhidas)
    # SO A SERIE (28/10/2026, ele: "isso de por mapa e irreal, nao giro isso nunca")
    lins = [l for l in lins if l["periodo"] == 0]
    # e o +-1,5 de mapas das casas dele (29/10/2026), na mesma tabela
    try:
        lins += linhas_handicap_mapas(jogo, inv, esporte(raiz), p_mapa=p_mapa, p_serie=confronto.get("p_series"),
                                      bo=bo, casas_dele=escolhidas)
    except Exception:
        pass
    if not lins:
        st.caption("A OddsPapi tem este jogo, mas nenhuma das casas escolhidas abriu vencedor.")
    else:
        _desenhar(st, html(time_a, time_b, lins, str(doc.get("lido_em") or "")), len(lins))
    if not detalhado:
        # 29/10/2026, ele: "a tela esta muito poluida, varias coisas falando a mesma coisa". As
        # linhas das casas (kills, torres...) estao na Linha justa (casa trocavel) e na Analise
        # de Linha; aqui fica so o vencedor e o +-1,5.
        return
    cache: dict = {}
    rs = melhores_linhas(linhas_de_mapa(jogo, inv, esporte(raiz), casas_dele=escolhidas), lambda l: chance_do_modelo(
        l, time_a=time_a, time_b=time_b, calcular_totais=calcular_totais, prob_over=prob_over,
        medias_kills=medias_kills, p_casa=p_mapa, _cache=cache))
    st.markdown("#### Melhor odd e melhor linha: kills, torres, "
                + ("roshans..." if esporte(raiz) == "dota" else "dragões..."))
    if not rs:
        st.caption("Nenhuma das suas casas abriu linhas de mapa (kills, torres...) para este jogo "
                   "na leitura atual.")
        return
    rs.sort(key=lambda r: (r["periodo"], r["unidade"] != "kills", r["unidade"], r["tipo"],
                           str(r.get("time")), r["lado"], r["pontos"]))
    from app_core.modelo_mercado_core import ODD_MODELO_MAX
    from app_core.oportunidades_core import VANTAGEM_SUSPEITA_PCT
    # O DESTAQUE segue o resumo: so aposta provavel (justa < 1,72), a melhor linha de
    # cada mercado, com valor; os suspeitos vao para o fim, como na Linha justa
    sinais_fn = _sinais_fn(team_games, time_a, time_b) if esporte(raiz) == "lol" else None
    alt = 50
    if sinais_fn is not None:
        _secao_sinais(st, rs, time_a, time_b, sinais_fn, team_games)
    destaque = [r for r in rs if r["melhor_linha"] and (valor(r) or 0) > 0]
    destaque = sorted(destaque, key=lambda r: (valor(r) > VANTAGEM_SUSPEITA_PCT, -valor(r)))[:12]
    _justa = "{:.2f}".format(ODD_MODELO_MAX).replace(".", ",")
    if destaque:
        st.caption("As melhores: odd justa abaixo de {} (chance de ~58% ou mais), a melhor linha de cada "
                   "mercado, na casa que paga mais, da maior vantagem para a menor. ⚠ = acima de {:.0f}%: "
                   "desconfie do modelo antes da casa.".format(_justa, VANTAGEM_SUSPEITA_PCT))
        _desenhar(st, html_linhas(destaque, time_a, time_b, sinais_fn), len(destaque), linha=alt)
    else:
        st.caption("Nenhuma linha provável (odd justa abaixo de {}) das suas casas paga mais do que o "
                   "justo.".format(_justa))
    with st.expander(f"Todas as linhas das suas casas neste jogo ({len(rs)})"):
        _desenhar(st, html_linhas(rs, time_a, time_b, sinais_fn), len(rs), linha=alt)
    st.caption("Melhor odd = a maior entre as suas casas para aquela aposta. vs Pinnacle: só quando a "
               "Pinnacle tem a MESMA linha (sem margem). vs Modelo: a mesma conta da Linha justa. "
               "Melhor linha = em cada mercado e lado, entre as de odd justa abaixo de 1,72, a de maior valor.")


def _sinais_fn(team_games: Any, time_a: str, time_b: str):
    """Celula "Seus sinais" para as linhas de TOTAL DE KILLS (so LoL; os pesos sao do LoL):
    geral e so mapa 1. Ver app_core/sinais_linha_core.py."""
    if team_games is None or getattr(team_games, "empty", True):
        return None
    from app_core.sinais_linha_core import resumo_html, sinais
    cache: dict = {}

    def _sn(L, m1):
        k = (float(L), bool(m1))
        if k not in cache:
            try:
                cache[k] = sinais(team_games, time_a, time_b, float(L), so_mapa1=bool(m1))
            except Exception:
                cache[k] = None
        return cache[k]

    def fn(r):
        if r.get("unidade") != "kills" or r.get("tipo") != "total":
            return None
        geral = resumo_html(_sn(r["pontos"], False), r["lado"], "Geral ", r.get("odd"))
        if int(r.get("periodo") or 0) != 1:
            return geral                  # o recorte "so mapa 1" so vale para a aposta do mapa 1
        return geral + "<br>" + resumo_html(_sn(r["pontos"], True), r["lado"], "Só mapa 1 ", r.get("odd"))
    fn.sinais = _sn
    return fn


def _secao_sinais(st: Any, rs: List[Mapping[str, Any]], time_a: str, time_b: str, sinais_fn, team_games) -> None:
    """PELOS SEUS SINAIS (28/10/2026; regra de 29/10: ENTRA com 3 de 4, 2 de 4 = analisar): as
    linhas de total de kills em que 2 ou mais dos 4
    sinais dele passam -- no geral ou so no mapa 1 --, da que mais paga acima da fair do
    conjunto para a que menos. E a regra dele ("se duas passam eu ja considero entrar")."""
    from app_core.sinais_linha_core import chance, passam
    lista = []
    for r in rs:
        if r.get("unidade") != "kills" or r.get("tipo") != "total":
            continue
        melhor_k, melhor_v = 0, None
        for m1 in ((False, True) if int(r.get("periodo") or 0) == 1 else (False,)):
            sn = sinais_fn.sinais(r["pontos"], m1)
            if not sn:
                continue
            k = sum(1 for v in passam(sn, r["lado"]) if v)
            p = chance(sn, r["lado"])
            v = None if not p else (p * float(r["odd"]) - 1) * 100
            if k >= 2:
                melhor_k = max(melhor_k, k)
                melhor_v = v if melhor_v is None or (v is not None and v > melhor_v) else melhor_v
        if melhor_k >= 2:
            lista.append((melhor_v if melhor_v is not None else -999, melhor_k, r))
    st.markdown("##### Pelos seus sinais: total de kills com 3 de 4 (e as de 2 de 4 para analisar)")
    if not lista:
        st.caption("Nenhuma linha de total de kills das suas casas tem 2 dos 4 sinais passando "
                   "(nem no geral, nem só no mapa 1).")
        return
    lista.sort(key=lambda x: (-x[0], -x[1]))
    pagam = [x[2] for x in lista if x[0] > 0 and x[1] >= 3][:12]
    analisar = [x[2] for x in lista if x[0] > 0 and x[1] == 2]
    resto = [x[2] for x in lista if x[0] <= 0]
    if analisar:
        with st.expander(f"2 de 4 e paga acima da fair: para analisar ({len(analisar)})"):
            _desenhar(st, html_linhas(analisar[:30], time_a, time_b, sinais_fn), min(len(analisar), 30), linha=50)
    rows = pagam
    if not pagam:
        st.caption("Nenhuma linha com 3 de 4 a favor pagando acima da fair do conjunto.")
    if resto:
        with st.expander(f"Com 2+ a favor mas abaixo da fair ({len(resto)})"):
            _desenhar(st, html_linhas(resto[:30], time_a, time_b, sinais_fn), min(len(resto), 30), linha=50)
    if not pagam:
        return
    st.caption("✓ passa · ✗ não passa · · sem dado. Ordem: Médias, Adversário, Histórico (5·20·10), Stats da linha. "
               "\"justa\" = fair do conjunto dos 4 sinais (medida no histórico); o % é a vantagem da odd "
               "da casa sobre ela. Com 3 de 4 a linha bateu ~63-68% no histórico; contra a linha da "
               "Pinnacle os sinais filtram, mas não garantem lucro sozinhos.")
    _desenhar(st, html_linhas(rows, time_a, time_b, sinais_fn), len(rows), linha=50)


def _desenhar(st: Any, corpo: str, n: int, linha: int = 46) -> None:
    from app_core.modelo_mercado_html_core import CSS, pagina
    try:
        import streamlit.components.v1 as _componentes
        _componentes.html(pagina(corpo), height=40 + linha * n + 60, scrolling=False)
    except Exception:
        st.markdown(CSS + corpo, unsafe_allow_html=True)


# ---- LINHA JUSTA COM QUALQUER CASA (29/10/2026) ------------------------------------------
# Ele: "essa ultima tela, onde diz Pinnacle, eu gostaria de conseguir mudar para a casa que eu
# queria dentro daquelas da OddsPapi". As linhas da casa viram o MESMO formato das linhas da
# Pinnacle e passam pela MESMA conta (oportunidades_core.montar_linhas_justas) e pela mesma
# tabela -- so muda de onde vem a odd.
_UNIDADE_PINNACLE = {"kills": "Kills", "turrets": "Towers", "dragons": "Dragons", "barons": "Barons",
                     "inhibitors": "Inhibitors"}


def linhas_formato_pinnacle(jogo: Mapping[str, Any], invertido: bool, esp: str, nome_casa: str,
                            jogo_id: int = -1) -> List[Dict[str, Any]]:
    """As linhas de mapa de UMA casa no formato das linhas da Pinnacle (Time 1 da tela = home). Pura."""
    fora = []
    for l in linhas_de_mapa(jogo, invertido, esp, casas_dele=[nome_casa]):
        if l["casa_aposta"] != nome_casa or int(l["periodo"]) < 1:
            continue
        u = _UNIDADE_PINNACLE.get(l["unidade"])
        if not u:
            continue
        n = int(l["periodo"])
        base = {"jogo": jogo_id, "periodo": n, "unidade": u, "lado": l["lado"], "pontos": l["pontos"],
                "odd": l["odd"], "alternativa": False, "time": None}
        if l["tipo"] == "total":
            fora.append(dict(base, tipo="total", mercado=f"Total - {u} - mapa {n}"))
        elif l["tipo"] == "team_total" and u == "Kills":
            fora.append(dict(base, tipo="team_total", time=l.get("time"), mercado=f"Total do time - Kills - mapa {n}"))
        elif l["tipo"] == "spread" and u == "Kills":
            fora.append(dict(base, tipo="spread", mercado=f"Handicap - Kills - mapa {n}"))
    return fora


def linha_justa_da_casa(*, raiz: Path, confronto: Mapping[str, Any], resolver: Optional[Callable[[str], str]],
                        nome_casa: str, calcular_totais: Optional[Callable], prob_over: Optional[Callable],
                        medias_kills: Optional[Callable], p_mapa_da_casa: Optional[Callable]):
    """(justas, vencedores, motivo) da casa escolhida para o confronto aberto."""
    from app_core.oportunidades_core import montar_linhas_justas
    time_a, time_b = str(confronto.get("teamA") or ""), str(confronto.get("teamB") or "")
    doc = carregar(raiz)
    jogo, inv = achar(doc.get("jogos") or [], time_a, time_b, resolver)
    if jogo is None:
        return [], [], f"{nome_casa} não tem este jogo na leitura atual (OddsPapi)."
    try:
        bo = int(confronto.get("bo") or 1)
    except (TypeError, ValueError):
        bo = 1
    fake = {"id": -1, "casa": time_a, "fora": time_b, "bo": bo, "liga": jogo.get("torneio")}
    linhas_pin = linhas_formato_pinnacle(jogo, inv, esporte(raiz), nome_casa)
    justas = []
    if linhas_pin and calcular_totais is not None and prob_over is not None:
        justas = montar_linhas_justas({"jogos": [fake], "linhas": linhas_pin}, calcular_totais,
                                      prob_over=prob_over, medias_kills=medias_kills,
                                      p_mapa_da_casa=p_mapa_da_casa)
        for x in justas:
            x.setdefault("casa", time_a)
            x.setdefault("fora", time_b)
    # vencedor: serie e mapas, a odd da casa contra o modelo do cartao
    venc = []
    p_map, p_ser = _num(confronto.get("p_map")), _num(confronto.get("p_series"))
    for mid, per in mercados(esporte(raiz)).items():
        par = None
        for l in linhas(jogo, inv, esporte(raiz), casas_dele=[nome_casa]):
            if l["casa"] == nome_casa and l["periodo"] == per:
                par = (l["odd_a"], l["odd_b"])
        if not par:
            continue
        p = p_ser if (per == 0 and bo > 1) else p_map
        r = {"periodo": per, "pin_a": par[0], "pin_b": par[1]}
        if p is not None and 0 < p < 1:
            r.update({"modelo_a": 1 / p, "modelo_b": 1 / (1 - p),
                      "vant_a": round((p * par[0] - 1) * 100, 1), "vant_b": round(((1 - p) * par[1] - 1) * 100, 1)})
        venc.append(r)
    motivo = "" if (justas or venc) else f"{nome_casa} não abriu linhas de mapa para este jogo."
    return justas, venc, motivo
