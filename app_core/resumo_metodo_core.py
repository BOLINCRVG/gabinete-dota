"""A LOGICA DAS APOSTAS DO RESUMO, refeita pelo metodo dele (29/10/2026).

Ele: "refaca a aba de Resumo... so a questao da logica das apostas, o layout e as coisas que ela me
oferece ja estao boas". O layout (cards, Sua odd, Edge, Stake, veredito) fica; muda o NUMERO:

  - totais (kills, torres, dragoes, baroes, inibidores, tempo): a fair e a chance do metodo dele
    (medias, adversario, historico 5/20/10, stats da linha), com os pesos medidos no cego
    (sinais_linha_core.METRICAS). Os passos a favor contam: 3 de 4 = ENTRA, 2 = ANALISAR;
  - vencedor: o da SERIE (ele so aposta a serie), o mesmo numero do card do confronto;
  - combinadas (vence + total): a condicional medida (combinadas_core), nas linhas dos totais;
  - entra na lista o que tem fair <= teto (1,72, a regra dele); a odd das casas dele (Stake +
    Polymarket por padrao) ja vem preenchida em "Sua odd".

Sem o motor do metodo (Dota ainda nao tem) a linha segue a logica antiga: nada quebra.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import pandas as pd

TOTAIS = ("kills", "towers", "dragons", "barons", "inhibitors", "time")
COMBINADAS = ("kills", "time", "towers", "dragons")
ROTULO = {"kills": "Kills", "towers": "Torres", "dragons": "Dragões", "barons": "Barões",
          "inhibitors": "Inibidores", "time": "Tempo"}

# odds das casas: (metrica, linha) -> {"over": (odd, casa), "under": (odd, casa)}; ("ml", 0) ->
# {"A": (odd, casa), "B": (odd, casa)}
OddsCasas = Callable[[str, float], Mapping[str, Tuple[float, str]]]


def _fair(p: Optional[float]) -> Optional[float]:
    return None if p is None or p <= 0 else round(1.0 / p, 2)


def _leaf(metric: str) -> str:
    m = str(metric or "")
    for t in TOTAIS:
        if m == t or m.endswith("_" + t):
            return t
    return m


def _marca_casa(row: Dict[str, Any], casa: Optional[Tuple[float, str]]) -> None:
    if casa and casa[0] and float(casa[0]) > 1.0:
        row["_casa_odd"] = float(casa[0])
        row["_casa_nome"] = str(casa[1])


def _aplica_total(row: Dict[str, Any], tg: pd.DataFrame, a: str, b: str, odds: Optional[OddsCasas],
                  cache: Dict[Any, Any]) -> None:
    from app_core.sinais_linha_core import METRICAS, chance, passam, sinais
    met = _leaf(row.get("_metric"))
    L = float(row.get("_line") or 0.0)
    lado = "over" if int(row.get("_side_ord") or 0) == 1 else "under"
    ch = (met, L)
    if ch not in cache:
        cache[ch] = sinais(tg, a, b, L, so_mapa1=False, metrica=met)
    sn = cache[ch]
    p = chance(sn, lado) if sn else None
    if p is None:
        return
    pas = passam(sn, lado)
    k = sum(1 for x in pas if x is True)
    fair = _fair(p)
    row.update({"Odd (Hist)": fair, "Odd (Mix)": fair, "_p_metodo": float(p), "_passos": int(k),
                "_passos_lista": pas, "_bateu": METRICAS[met]["bateu"][lado].get(k),
                "_instavel": bool(METRICAS[met].get("instavel"))})
    if odds is not None:
        try:
            _marca_casa(row, (odds(met, L) or {}).get(lado))
        except Exception:
            pass


def _aplica_ml(row: Dict[str, Any], p_serie_a: Optional[float], bo: int, odds: Optional[OddsCasas]) -> None:
    if p_serie_a is None or not (0.0 < float(p_serie_a) < 1.0):
        return
    lado = "A" if str(row.get("_metric", "")).endswith("_A") else "B"
    p = float(p_serie_a) if lado == "A" else 1.0 - float(p_serie_a)
    fair = _fair(p)
    row.update({"Mercado": f"ML (Série MD{bo})" if bo > 1 else "ML (Mapa)", "Odd (Modelo)": fair,
                "Odd (Hist)": fair, "Odd (Mix)": fair, "_p_metodo": p, "_passos": None})
    if odds is not None:
        try:
            _marca_casa(row, (odds("ml", 0.0) or {}).get(lado))
        except Exception:
            pass


def combinadas_rows(tg: pd.DataFrame, a: str, b: str, p_mapa_a: Optional[float],
                    linhas: Mapping[str, List[float]], fmt_linha: Callable[[str, float], str]) -> List[Dict[str, Any]]:
    """As combinadas (vence o mapa + total) nas linhas dos totais, com a condicional medida."""
    from app_core.combinadas_core import combinadas
    out: List[Dict[str, Any]] = []
    if p_mapa_a is None:
        return out
    for met in COMBINADAS:
        for L in sorted({float(x) for x in (linhas.get(met) or [])}):
            try:
                r = combinadas(tg, a, b, L, met, p_mapa_a)
            except Exception:
                r = None
            if not r:
                continue
            for c in r["combos"]:
                quem = "A" if c["time"] == a else "B"
                ld = fmt_linha(met, L)
                fair = _fair(c["p"])
                out.append({
                    "Mercado": f"{c['time']} + {ROTULO[met]} ({ld})",
                    "Lado": f"ML & {'Acima' if c['lado'] == 'over' else 'Abaixo'}", "Linha": ld,
                    "Amostra Liga": int(c["hist"][1]), "Amostra H2H": int(c["h2h"][1]),
                    "Liga": "-", "H2H": "-", "Mismatch": "N/A", "Laplace (Liga)": "—", "Laplace (H2H)": "—",
                    "StatsCamps": "-", "StatsCamps %": None, "Odd (Modelo)": fair, "Odd (Hist)": fair,
                    "Odd (Mix)": fair, "Obs Mix": "", "_metric": f"zz_mltot_{quem}_{met}",
                    "_side_ord": 1 if c["lado"] == "over" else 0, "_line": L, "_p_metodo": float(c["p"]),
                    "_passos": None, "_combo_hist": c["hist"], "_combo_h2h": c["h2h"],
                })
    return out


def aplicar_metodo(rows: List[Dict[str, Any]], *, tg: pd.DataFrame, team_a: str, team_b: str,
                   p_serie_a: Optional[float] = None, bo: int = 1, odds: Optional[OddsCasas] = None
                   ) -> List[Dict[str, Any]]:
    """Troca a fair de cada linha pela do metodo; as combinadas antigas saem (o chamador poe as novas)."""
    try:
        import app_core.sinais_linha_core  # noqa: F401  (so o LoL tem o motor)
    except Exception:
        return rows
    cache: Dict[Any, Any] = {}
    fora: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        m = str(r.get("_metric") or "")
        if m.startswith("zz_mltot_"):
            continue
        if m in ("00_ml_A", "00_ml_B"):
            _aplica_ml(r, p_serie_a, int(bo or 1), odds)
        elif _leaf(m) in TOTAIS:
            try:
                _aplica_total(r, tg, team_a, team_b, odds, cache)
            except Exception:
                pass
        fora.append(r)
    return fora


def dentro_do_teto(row: Mapping[str, Any], teto: float) -> bool:
    """A regra dele: a fair (do metodo) ate o teto. Sem fair do metodo: a regra antiga
    (modelo OU hist no teto)."""
    if row.get("_p_metodo") is not None:
        try:
            return float(row.get("Odd (Mix)")) <= float(teto)
        except (TypeError, ValueError):
            return False
    for campo in ("Odd (Hist)", "Odd (Modelo)"):
        try:
            if float(row.get(campo)) <= float(teto):
                return True
        except (TypeError, ValueError):
            continue
    return False


def pontos(row: Mapping[str, Any], edge: Optional[float]) -> Dict[str, float]:
    """O score do metodo (0-100): chance (100*p) + passos (3-4: +10, 2: 0, 0-1: -10) + valor
    (o edge contra a casa, limitado a +-15). Batida e valor juntos, como ele pediu."""
    p = float(row.get("_p_metodo") or 0.0)
    k = row.get("_passos")
    passos = 0.0 if k is None else (10.0 if k >= 3 else (0.0 if k == 2 else -10.0))
    valor = 0.0 if edge is None else max(-15.0, min(15.0, float(edge)))
    total = max(0.0, min(100.0, 100.0 * p + passos + valor))
    return {"chance": round(100.0 * p, 1), "passos": passos, "valor": valor, "total": total}


def veredito_do_metodo(row: Mapping[str, Any], edge: Optional[float]) -> str:
    k = row.get("_passos")
    if k is None:
        return "ENTRA" if (edge is not None and edge > 0) else "sem odd da casa" if edge is None else "odd abaixo da fair"
    if k >= 3 and edge is not None and edge > 0:
        return "ENTRA"
    if k >= 3:
        return "3+ a favor, falta odd" if edge is None else "3+ a favor, odd abaixo da fair"
    if k == 2:
        return "ANALISAR (2 de 4)"
    return "não entra"


def faixa(row: Mapping[str, Any], edge: Optional[float]) -> int:
    """A ordem da lista: 0 ENTRA, 1 ANALISAR ou falta a odd, 2 odd abaixo da fair, 3 nao entra.
    Linha sem o metodo: 1 (a logica antiga decide pelo score)."""
    if row.get("_p_metodo") is None:
        return 1
    v = veredito_do_metodo(row, edge)
    if v == "ENTRA":
        return 0
    if v.startswith("ANALISAR") or "falta odd" in v or v == "sem odd da casa":
        return 1
    if "abaixo da fair" in v:
        return 2
    return 3


def pros_e_contras(row: Mapping[str, Any], edge: Optional[float], odd_casa: Optional[float]) -> Tuple[List[str], List[str]]:
    pros: List[str] = []
    cons: List[str] = []
    p = row.get("_p_metodo")
    k = row.get("_passos")
    v = veredito_do_metodo(row, edge)
    if p is not None:
        pros.append(f"Fair do método {1 / float(p):.2f} ({100 * float(p):.0f}%) — {v}")
    if k is not None:
        nomes = ("Médias", "Adversário", "Histórico", "Stats da linha")
        lst = row.get("_passos_lista") or []
        a_favor = [n for n, x in zip(nomes, lst) if x is True]
        contra = [n for n, x in zip(nomes, lst) if x is False]
        txt = f"{k} de 4 passos a favor" + (f" ({', '.join(a_favor)})" if a_favor else "")
        if row.get("_bateu") is not None:
            txt += f" — com {k} passos, bateu {row['_bateu']}% no cego"
        (pros if k >= 3 else cons).append(txt)
        if contra:
            cons.append("Contra: " + ", ".join(contra))
    if row.get("_combo_hist"):
        h, n = row["_combo_hist"]
        if n:
            pros.append(f"Nos últimos jogos com esse resultado: {h}/{n} bateram o total")
    if row.get("_casa_nome") and odd_casa:
        pros.append(f"Melhor odd nas suas casas: {row['_casa_nome']} {odd_casa:.2f}")
    if edge is not None:
        (pros if edge > 0 else cons).append(f"Valor vs casa: {edge:+.1f}%")
    if row.get("_instavel"):
        cons.append("Mercado instável no cego (poucas linhas de teste) — peso menor")
    return pros, cons


def odds_das_casas(raiz: Any, tg: pd.DataFrame, team_a: str, team_b: str, casas: List[str]) -> Optional[OddsCasas]:
    """A melhor odd do MAPA 1 nas casas dele (totais) e da SERIE (vencedor). None = jogo fora da coleta."""
    from pathlib import Path
    from app_core.seus_sinais_tab_core import linhas_de_kills
    from app_core.suas_casas_core import jogo_das_casas, linhas
    times = sorted({str(x) for x in tg["team"].dropna().unique()})
    jogo, inv = jogo_das_casas(Path(raiz), str(team_a), str(team_b), times)
    if not jogo:
        return None
    cache: Dict[str, Any] = {}

    def _f(met: str, L: float):
        if met == "ml":
            if "ml" not in cache:
                best: Dict[str, Tuple[float, str]] = {}
                for l in linhas(jogo, inv, "lol", casas_dele=casas):
                    if int(l.get("periodo") or 0) != 0:
                        continue
                    for lado, k in (("A", "odd_a"), ("B", "odd_b")):
                        if lado not in best or float(l[k]) > best[lado][0]:
                            best[lado] = (float(l[k]), str(l["casa"]))
                cache["ml"] = best
            return cache["ml"]
        if met not in cache:
            cache[met] = linhas_de_kills(jogo, inv, casas, met)
        return (cache[met].get(float(L)) or {}).get(1) or {}
    return _f
