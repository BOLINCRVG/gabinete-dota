# -*- coding: utf-8 -*-
"""A aba "Modelo x Mercado", reorganizada (23/10/2026).

Pedido dele: "refaz essa aba ... a unica parte que quero deixar desse jeito e
a Linha justa (Fair / Modelo / Pinnacle); o resto eu quero melhor, nao sei
como, faca ai, dai vamos mudando".

A ORDEM, do que mais ajuda a decidir para o que so explica:
  1. PAINEL DO DIA -- quantos jogos, quantas linhas com valor, idade das odds
     de cada casa. Aviso so quando ha problema (casa barrou o robo, odd velha).
  2. ONDE ESTA O VALOR HOJE -- as melhores linhas de TODOS os jogos num ranking
     so. Antes era preciso abrir jogo por jogo para descobrir onde havia valor.
  3. O JOGO ESCOLHIDO -- vencedor nas tres fontes (modelo, Pinnacle,
     Polymarket), as linhas com valor DESTE jogo, e a Linha justa intacta.
  4. DETALHES, recolhidos: mercado inteiro, reguas, diario do robo, Polymarket
     inteira, o que ficou de fora.

Este modulo tem as contas puras (testaveis); o desenho fica em
oportunidades_tab_core.py, que continua sendo o ponto de entrada da aba.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from app_core.oportunidades_core import VANTAGEM_MINIMA_PCT, VANTAGEM_SUSPEITA_PCT


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def nivel(vantagem: Any, *, suspeito: Any = None) -> str:
    """Rotulo curto da vantagem, o mesmo em toda a aba."""
    v = _num(vantagem)
    if v is None:
        return ""
    if suspeito or v > VANTAGEM_SUSPEITA_PCT:
        return "desconfie"
    if v >= 8:
        return "alto"
    if v >= 4:
        return "medio"
    if v >= VANTAGEM_MINIMA_PCT:
        return "baixo"
    return ""


def linhas_com_valor(completo: List[Mapping[str, Any]], *, jogo: Any = None,
                     incluir_suspeitas: bool = True) -> List[Mapping[str, Any]]:
    """Linhas em que o modelo ve vantagem >= VANTAGEM_MINIMA_PCT, da maior para
    a menor. As suspeitas vao para o fim (aparecem, mas nao no topo)."""
    fora = []
    for x in completo or []:
        if jogo is not None and x.get("jogo") != jogo:
            continue
        v = _num(x.get("vantagem_pct"))
        if v is None or v < VANTAGEM_MINIMA_PCT:
            continue
        susp = bool(x.get("suspeito")) or v > VANTAGEM_SUSPEITA_PCT
        if susp and not incluir_suspeitas:
            continue
        fora.append(x)
    return sorted(fora, key=lambda x: (
        bool(x.get("suspeito")) or (_num(x.get("vantagem_pct")) or 0) > VANTAGEM_SUSPEITA_PCT,
        -(_num(x.get("vantagem_pct")) or 0)))


def painel_do_dia(completo: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Os numeros do topo da aba."""
    jogos = {x.get("jogo") for x in completo or []}
    com_modelo = [x for x in completo or [] if _num(x.get("vantagem_pct")) is not None]
    valor = linhas_com_valor(completo, incluir_suspeitas=False)
    return {"jogos": len(jogos), "linhas": len(completo or []),
            "linhas_com_modelo": len(com_modelo), "linhas_com_valor": len(valor),
            "jogos_com_valor": len({x.get("jogo") for x in valor})}


def _br(x: Any, casas: int = 2) -> str:
    v = _num(x)
    return "-" if v is None else f"{v:.{casas}f}".replace(".", ",")


def aposta(x: Mapping[str, Any]) -> str:
    """Nome curto do lado: "Liquid -8,5", "acima de 27,5", "T1"."""
    lado = str(x.get("lado") or "").strip().lower()
    ponto = _num(x.get("pontos"))
    time = str(x.get("time") or "").lower()
    quem = x.get("casa") if time == "home" else (x.get("fora") if time == "away" else None)
    if lado in ("home", "away"):
        nome = str(x.get("casa") if lado == "home" else x.get("fora"))
        if ponto is not None and str(x.get("tipo") or "") == "spread":
            nome += " " + f"{ponto:+.1f}".replace(".", ",")
        return nome
    if lado in ("over", "under"):
        txt = "{} {}".format("acima de" if lado == "over" else "abaixo de", _br(ponto, 1))
        return f"{quem}: {txt}" if quem else txt
    return str(x.get("lado") or "?")


def tabela_de_valor(linhas: List[Mapping[str, Any]], *, com_jogo: bool) -> List[Dict[str, Any]]:
    """As colunas do ranking. Separado da tela para dar para testar."""
    fora = []
    for x in linhas:
        linha = {}
        if com_jogo:
            linha["Jogo"] = "{} x {}".format(x.get("casa") or "?", x.get("fora") or "?")
            linha["Liga"] = x.get("liga") or ""
            linha["Inicio"] = str(x.get("inicio_utc") or "")[5:16].replace("T", " ")
        linha.update({
            "Mercado": x.get("nome_pt") or x.get("mercado") or "",
            "Aposta": aposta(x),
            "Pinnacle paga": _br(x.get("odd_casa")),
            "Modelo (justa)": _br(x.get("odd_modelo")),
            "Vantagem": "-" if _num(x.get("vantagem_pct")) is None
            else f"{_num(x.get('vantagem_pct')):+.1f}%".replace(".", ","),
            "Nivel": nivel(x.get("vantagem_pct"), suspeito=x.get("suspeito")),
        })
        fora.append(linha)
    return fora
