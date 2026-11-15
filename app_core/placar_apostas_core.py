# -*- coding: utf-8 -*-
"""PLACAR DAS APOSTAS INDICADAS (15/11/2026). Arquivo identico nos dois repositorios.

O CS ja tinha (placar_apostas.py la): o robo grava cada aposta que a tela indica e, quando o jogo
acaba, fecha com lucro ou prejuizo -- e o unico jeito de saber se as indicacoes dao dinheiro.

O QUE E "INDICADA" AQUI. A aposta nas casas dele (Stake, Polymarket) que paga MAIS que o preco
justo. A justa e a da PINNACLE sem margem no mesmo mercado e na mesma linha (a regua dele desde
23/10: "a pinnacle vai ser quase sempre melhor regulada"; nos tres esportes o modelo nao bateu a
Pinnacle no cego). Valor entre VALOR_MIN e VALOR_MAX: abaixo e ruido de arredondamento, acima e
quase sempre odd velha ou time casado errado. Mercados: vencedor da serie e de cada mapa, handicap
de mapas, total de mapas e total de kills do mapa.

COMO FECHA. Pelo resultado OFICIAL do proprio mercado na Polymarket (o evento fechado traz o lado
que pagou) -- vale para os dois esportes, sem casar nome de time com planilha de resultado. Por
isso so entra no placar o jogo que a Polymarket lista. Mapa que nao foi jogado (serie 2 x 0) fecha
meio a meio la: aqui e ANULADA, stake devolvida.

Uma aposta entra UMA vez, pela odd de quando foi vista primeiro (e a que ele teria pego).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from app_core.mercado_vencedor_core import _chave, _num

VALOR_MIN, VALOR_MAX = 1.0, 20.0
TETO_ESTRELA = 1.72                 # a regra dele: destaque para justa abaixo de 1,72
JANELA_H = 36.0
CASAS = ("Stake", "Polymarket")
_ENFEITE = re.compile(r"\b(team|gaming|esports?|e-sports|club|clan|gg)\b", re.I)


def _quando(v: Any) -> Optional[datetime]:
    t = str(v or "").strip().replace("Z", "+00:00").replace(" ", "T")
    for s in (t, t + ":00"):
        try:
            d = datetime.fromisoformat(s)
            return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def mesmo_time(a: Any, b: Any) -> bool:
    """"LGD" x "LGD Gaming", "Spirit" x "Team Spirit": o mesmo time escrito por duas fontes. Pura.
    So decide junto com o OUTRO time do jogo e o horario (jogo_da_pinnacle), nunca sozinha."""
    ka, kb = _chave(a), _chave(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    sa, sb = _chave(_ENFEITE.sub(" ", str(a))), _chave(_ENFEITE.sub(" ", str(b)))
    if sa and sa == sb:
        return True
    curto, longo = sorted((sa or ka, sb or kb), key=len)
    return len(curto) >= 3 and (longo.startswith(curto) or longo.endswith(curto))


def jogo_da_pinnacle(pin: Mapping[str, Any], jogo: Mapping[str, Any]) -> Tuple[Optional[Mapping[str, Any]], bool]:
    """(jogo da Pinnacle, invertido) do mesmo confronto, ate 36 h de diferenca. Pura."""
    nomes1 = [jogo.get("time1")] + list((jogo.get("polymarket_times") or [None, None])[:1])
    nomes2 = [jogo.get("time2")] + list((jogo.get("polymarket_times") or [None, None])[1:2])
    t0 = _quando(jogo.get("inicio_utc"))
    for pj in pin.get("jogos") or []:
        t1 = _quando(pj.get("inicio_utc"))
        if t0 and t1 and abs((t0 - t1).total_seconds()) > JANELA_H * 3600:
            continue
        casa1 = any(mesmo_time(pj.get("casa"), n) for n in nomes1 if n)
        fora2 = any(mesmo_time(pj.get("fora"), n) for n in nomes2 if n)
        casa2 = any(mesmo_time(pj.get("casa"), n) for n in nomes2 if n)
        fora1 = any(mesmo_time(pj.get("fora"), n) for n in nomes1 if n)
        if casa1 and fora2 and not (casa2 and fora1):
            return pj, False
        if casa2 and fora1 and not (casa1 and fora2):
            return pj, True
    return None, False


def _sem_margem(o1: Any, o2: Any) -> Optional[float]:
    a, b = _num(o1), _num(o2)
    if not a or not b or a <= 1 or b <= 1:
        return None
    return (1 / a) / (1 / a + 1 / b)


def reguas(pin: Mapping[str, Any], pj: Mapping[str, Any], inv: bool) -> Dict[tuple, float]:
    """{chave do mercado: chance justa do PRIMEIRO lado} pela Pinnacle sem margem, na orientacao do
    jogo das casas. Chaves: ("ml", mapa), ("hc", handicap do time 1), ("tm", linha),
    ("tk", mapa, linha). Primeiro lado = time 1 (ml, hc) ou Mais (tm, tk). Pura."""
    lados: Dict[tuple, Dict[str, float]] = {}
    for l in pin.get("linhas") or []:
        if l.get("jogo") != pj.get("id"):
            continue
        forma, per, lado = str(l.get("forma") or ""), int(l.get("periodo") or 0), str(l.get("lado") or "").lower()
        met, pts, odd = str(l.get("metrica") or ""), _num(l.get("pontos")), _num(l.get("odd"))
        if not odd or lado not in ("home", "away", "over", "under"):
            continue
        if lado in ("home", "away") and inv:
            lado = "away" if lado == "home" else "home"
        if met == "ml" and forma == "vence":
            k = ("ml", per)
        elif met == "ml" and forma == "handicap" and per == 0 and pts is not None:
            k = ("hc", pts if lado == "home" else -pts)
        elif met == "ml" and forma == "total" and per == 0 and pts is not None:
            k = ("tm", pts)
        elif met == "kills" and forma == "total" and per >= 1 and pts is not None:
            k = ("tk", per, pts)
        else:
            continue
        lados.setdefault(k, {})["1" if lado in ("home", "over") else "2"] = odd
    fora = {}
    for k, v in lados.items():
        p = _sem_margem(v.get("1"), v.get("2"))
        if p is not None:
            fora[k] = p
    return fora


def ofertas(jogo: Mapping[str, Any], esp: str, casas: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Cada lado de cada mercado que as casas dele abriram para o jogo, na orientacao do jogo. Pura.
    {casa, mercado (ml|hc|tm|tk), mapa, linha, lado ("1"|"2"), odd, chave (a da regua), primeiro}."""
    from app_core import suas_casas_core as sc
    casas = list(casas or CASAS)
    fora: List[Dict[str, Any]] = []
    for l in sc.linhas(jogo, False, esp, casas_dele=casas):
        if l["casa"] == "Pinnacle":
            continue
        per = int(l["periodo"])
        for lado, odd in (("1", l["odd_a"]), ("2", l["odd_b"])):
            fora.append({"casa": l["casa"], "mercado": "ml", "mapa": per, "linha": None, "lado": lado,
                         "odd": float(odd), "chave": ("ml", per)})
    for l in sc.linhas_de_mapa(jogo, False, esp, casas_dele=casas):
        if l["casa_aposta"] == "Pinnacle":
            continue
        per, pts, odd = int(l["periodo"]), float(l["pontos"]), float(l["odd"])
        if l["unidade"] == "maps" and l["tipo"] == "spread" and per == 0:
            h1 = pts if l["lado"] == "home" else -pts
            fora.append({"casa": l["casa_aposta"], "mercado": "hc", "mapa": 0, "linha": pts,
                         "lado": "1" if l["lado"] == "home" else "2", "odd": odd, "chave": ("hc", h1)})
        elif l["unidade"] == "maps" and l["tipo"] == "total" and per == 0:
            fora.append({"casa": l["casa_aposta"], "mercado": "tm", "mapa": 0, "linha": pts,
                         "lado": "1" if l["lado"] == "over" else "2", "odd": odd, "chave": ("tm", pts)})
        elif l["unidade"] == "kills" and l["tipo"] == "total" and per >= 1:
            fora.append({"casa": l["casa_aposta"], "mercado": "tk", "mapa": per, "linha": pts,
                         "lado": "1" if l["lado"] == "over" else "2", "odd": odd, "chave": ("tk", per, pts)})
    return fora


def _nome(o: Mapping[str, Any], t1: str, t2: str) -> str:
    quem = t1 if o["lado"] == "1" else t2
    m, mapa, L = o["mercado"], o["mapa"], o["linha"]
    if m == "ml":
        return f"{quem} vence " + ("a série" if not mapa else f"o mapa {mapa}")
    if m == "hc":
        return f"{quem} {L:+g} mapas".replace(".", ",")
    lado = "Mais de" if o["lado"] == "1" else "Menos de"
    if m == "tm":
        return f"{lado} {L:g} mapas".replace(".", ",")
    return f"{lado} {L:g} kills no mapa {mapa}".replace(".", ",")


def candidatos(jogo: Mapping[str, Any], pin: Mapping[str, Any], esp: str,
               casas: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """As apostas com valor do jogo (regua: Pinnacle sem margem), a melhor casa de cada lado. Pura."""
    pj, inv = jogo_da_pinnacle(pin, jogo)
    if pj is None:
        return []
    rg = reguas(pin, pj, inv)
    melhor: Dict[tuple, Dict[str, Any]] = {}
    for o in ofertas(jogo, esp, casas):
        p1 = rg.get(o["chave"])
        if p1 is None:
            continue
        p = p1 if o["lado"] == "1" else 1 - p1
        valor = (p * o["odd"] - 1) * 100
        if not (VALOR_MIN <= valor <= VALOR_MAX):
            continue
        k = (o["mercado"], o["mapa"], o["linha"], o["lado"])
        if k in melhor and melhor[k]["odd"] >= o["odd"]:
            continue
        melhor[k] = {"time1": jogo.get("time1"), "time2": jogo.get("time2"), "inicio_utc": jogo.get("inicio_utc"),
                     "torneio": jogo.get("torneio"), "evento": jogo.get("polymarket_evento"),
                     "times_poly": jogo.get("polymarket_times"),
                     "casa": o["casa"], "mercado": o["mercado"], "mapa": o["mapa"], "linha": o["linha"],
                     "lado": o["lado"], "nome": _nome(o, str(jogo.get("time1")), str(jogo.get("time2"))),
                     "odd": round(o["odd"], 3), "p_justa": round(p, 4), "justa": round(1 / p, 2),
                     "regua": "pinnacle", "valor_pct": round(valor, 1), "estrela": (1 / p) < TETO_ESTRELA}
    return sorted(melhor.values(), key=lambda a: (not a["estrela"], -a["valor_pct"]))


def chave_da_aposta(a: Mapping[str, Any]) -> tuple:
    return (str(a.get("evento")), str(a.get("mercado")), int(a.get("mapa") or 0),
            None if a.get("linha") is None else float(a["linha"]), str(a.get("lado")))


def novas(cands: List[Mapping[str, Any]], gravadas: List[Mapping[str, Any]], agora: datetime,
          esp: str) -> List[Dict[str, Any]]:
    """As que entram no arquivo agora: tem evento na Polymarket (e por ele que fecha), o jogo nao
    comecou e a aposta ainda nao foi gravada. Pura."""
    ja = {chave_da_aposta(g) for g in gravadas}
    fora = []
    for c in cands:
        t = _quando(c.get("inicio_utc"))
        if not c.get("evento") or t is None or t <= agora or chave_da_aposta(c) in ja:
            continue
        ja.add(chave_da_aposta(c))
        fora.append(dict(c, esporte=esp, gravado_em=agora.isoformat(timespec="seconds"), resultado=None, lucro=None))
    return fora


# ---------------------------------------------------------------- fechar pelo mercado da Polymarket
def _lista(v):
    import json
    if isinstance(v, list):
        return v
    try:
        x = json.loads(v or "[]")
        return x if isinstance(x, list) else []
    except (TypeError, ValueError):
        return []


_GAME = re.compile(r"\bGame\s+(\d)\b", re.I)


def _pagou(m: Mapping[str, Any]) -> Optional[int]:
    """0 ou 1 = o lado que pagou; -1 = meio a meio (anulado); None = ainda aberto. Pura."""
    if not m.get("closed"):
        return None
    pr = [_num(x) for x in _lista(m.get("outcomePrices"))]
    if len(pr) != 2 or pr[0] is None or pr[1] is None:
        return None
    if pr[0] >= 0.99 and pr[1] <= 0.01:
        return 0
    if pr[1] >= 0.99 and pr[0] <= 0.01:
        return 1
    if abs(pr[0] - 0.5) < 0.01 and abs(pr[1] - 0.5) < 0.01:
        return -1
    return None


def resultado(a: Mapping[str, Any], evento: Mapping[str, Any]) -> Optional[str]:
    """"ganhou" | "perdeu" | "anulada" | None (mercado ainda aberto ou nao achado). Pura.
    `evento` = o evento da Gamma (com markets); a aposta guarda os nomes dos times como a
    Polymarket escreve (times_poly), na ordem time 1, time 2 do jogo."""
    tp = [str(x) for x in (a.get("times_poly") or [])]
    if len(tp) != 2:
        return None
    meu = _chave(tp[0] if a.get("lado") == "1" else tp[1])
    mercado, mapa, L = a.get("mercado"), int(a.get("mapa") or 0), _num(a.get("linha"))
    for m in evento.get("markets") or []:
        tipo = m.get("sportsMarketType")
        lados = [_chave(x) for x in _lista(m.get("outcomes"))]
        linha = _num(m.get("line"))
        g = _GAME.search(str(m.get("question") or ""))
        jogo_n = int(g.group(1)) if g else 0
        idx = None
        if mercado == "ml" and ((mapa == 0 and tipo == "moneyline") or (mapa and tipo == "child_moneyline" and jogo_n == mapa)):
            idx = lados.index(meu) if meu in lados else None
        elif mercado == "tm" and tipo == "totals" and linha == L:
            idx = 0 if a.get("lado") == "1" else 1
        elif mercado == "tk" and tipo == "kill_over_under_game" and linha == L and jogo_n == mapa:
            idx = 0 if a.get("lado") == "1" else 1
        elif mercado == "hc" and tipo == "map_handicap" and linha is not None and L is not None and len(lados) == 2:
            # o mercado e "X (linha) vs Y (-linha)": a aposta "meu com L" esta nele se bate o lado e a linha
            if lados[0] == meu and abs(linha - L) < 1e-9:
                idx = 0
            elif lados[1] == meu and abs(-linha - L) < 1e-9:
                idx = 1
        if idx is None:
            continue
        pg = _pagou(m)
        if pg is None:
            return None
        return "anulada" if pg == -1 else ("ganhou" if pg == idx else "perdeu")
    return None


def lucro(a: Mapping[str, Any], res: str) -> float:
    return round(float(a["odd"]) - 1, 3) if res == "ganhou" else (-1.0 if res == "perdeu" else 0.0)


def resumo(apostas: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """{n, abertas, fechadas, ganhou, perdeu, anuladas, lucro (em stakes), roi_pct}. Pura."""
    fech = [a for a in apostas if a.get("resultado") in ("ganhou", "perdeu")]
    lu = sum(float(a.get("lucro") or 0.0) for a in fech)
    return {"n": len(apostas), "abertas": sum(1 for a in apostas if not a.get("resultado")),
            "fechadas": len(fech), "ganhou": sum(1 for a in fech if a["resultado"] == "ganhou"),
            "perdeu": sum(1 for a in fech if a["resultado"] == "perdeu"),
            "anuladas": sum(1 for a in apostas if a.get("resultado") == "anulada"),
            "lucro": round(lu, 2), "roi_pct": None if not fech else round(100 * lu / len(fech), 1)}


def texto_do_resumo(r: Mapping[str, Any]) -> str:
    if not r.get("n"):
        return "Placar das apostas indicadas: nenhuma gravada ainda (o robô grava a cada 3 horas)."
    t = f"Placar das apostas indicadas: {r['n']} gravada(s), {r['abertas']} em aberto"
    if r.get("fechadas"):
        t += (f"; {r['fechadas']} fechada(s): {r['ganhou']} ganhou, {r['perdeu']} perdeu, lucro "
              f"{r['lucro']:+.2f} stake(s), ROI {r['roi_pct']:+.1f}%").replace(".", ",")
        if r["fechadas"] < 100:
            t += " — amostra pequena, ainda não prova nada"
    return t + "."
