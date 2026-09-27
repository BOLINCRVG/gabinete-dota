# -*- coding: utf-8 -*-
"""Quanto confiar no numero DESTE confronto.

O QUE ESTAVA ERRADO (relatado por ele em 15/09/2026: "sempre aparece C
(Baixa)")
O cartao somava 50 de base e tinha uma penalidade de -15 para "nao tenho o
relatorio de validacao". 50 - 15 = 35, que era exatamente o que aparecia na
tela, todo dia, em qualquer confronto.

A causa: o relatorio (`lv_last_report.json`) era procurado SO na memoria da
sessao. Quem o escreve e a rotina de manutencao; a memoria zera a cada
reinicio do site e a tela nunca lia o arquivo. Nunca achava.

E ler o arquivo NAO era o conserto: o que estava em disco era de 04/03/2026 —
seis meses — e com uma liga so (LPL). Com ele, o cartao pularia de "C (Baixa)
35" para "A (Alta) 90". Trocar um numero falso por um numero falso e
confiante e pior que o defeito.

A FONTE CERTA JA EXISTIA
`evidence/nota_por_liga.json`: 33 ligas, com logloss, acerto, ECE e n,
medidas ha 8 dias. E a mesma que alimenta a linha embaixo do confronto.

O QUE O CARTAO MEDE AGORA (escolha dele, 15/09/2026)
Duas coisas, porque a pergunta "posso confiar neste numero?" depende das duas:
  1. o modelo e honesto NESTA LIGA — calibracao, tamanho da amostra,
     previsibilidade;
  2. o que aconteceu NESTE CONFRONTO — quais camadas agiram, cobertura de
     escalacao, guardas que precisaram intervir.

CALIBRACAO PESA MAIS QUE PREVISIBILIDADE, de proposito. "Confianca" e sobre o
% ser honesto (se o EV que voce calcula e real). Previsibilidade e sobre
quanto edge existe para extrair — importa, mas e outra pergunta, e entra com
peso menor. Medido na propria base: a LPL tem a melhor calibracao de todas e
a nota mais baixa, porque os times sao parelhos.

PRINCIPIO IV: sem medicao utilizavel, o cartao NAO mostra numero. Mostra "—"
e diz o motivo. Foi justamente um numero de "nao sei" com cara de veredito
que enganou por meses.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

# Os limiares de calibracao vem da linha da liga, nao de uma copia: se os dois
# discordassem, a tela diria "honesta" embaixo e "C (Baixa)" em cima.
from app_core.nota_liga_core import _ECE_BOA, _ECE_RUIM

# Mesmo prazo do selo de frescor da linha da liga: 2x a cadencia semanal.
# Medicao mais velha que isso repete o numero antigo com toda a confianca do
# mundo, e o sintoma aparece como "nota estranha", que nao aponta a causa.
LIMITE_DIAS = 14.0


def _num(v: Any) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def pontos_da_liga(linha: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Quanto a qualidade do modelo NAQUELA LIGA soma ou tira.

    Devolve `usavel=False` quando nao ha o que medir — nunca zero disfarcado
    de "mediano".
    """
    if not isinstance(linha, Mapping) or not linha:
        return {"usavel": False, "pontos": 0.0, "notas": [], "motivo": "sem medicao da liga"}

    ece = _num(linha.get("ece"))
    n = _num(linha.get("n"))
    nota = _num(linha.get("nota"))
    if not math.isfinite(ece) or not math.isfinite(n):
        return {"usavel": False, "pontos": 0.0, "notas": [],
                "motivo": "medicao da liga incompleta"}

    pts = 0.0
    notas = []

    # 1. CALIBRACAO — o peso maior. E o que diz se o % e honesto.
    if ece <= _ECE_BOA:
        pts += 20.0
        notas.append("calibracao honesta")
    elif ece <= _ECE_RUIM:
        pts += 5.0
        notas.append("calibracao ok")
    else:
        pts -= 20.0
        notas.append("calibracao ruim")

    # 2. TAMANHO DA AMOSTRA — medicao de 80 jogos nao sustenta veredito.
    if n >= 400:
        pts += 10.0
    elif n >= 200:
        pts += 5.0
    elif n < 100:
        pts -= 10.0
        notas.append("poucos jogos medidos")

    # 3. PREVISIBILIDADE — peso menor, e outra pergunta (quanto edge existe).
    if math.isfinite(nota):
        if nota >= 8.0:
            pts += 10.0
        elif nota >= 7.0:
            pts += 5.0
        elif nota < 6.0:
            pts -= 5.0
            notas.append("liga imprevisivel")

    return {"usavel": True, "pontos": float(pts), "notas": notas, "motivo": "ok",
            "ece": ece, "n": n, "nota": nota}


def escolher_entre_ligas(
    linha_a: Optional[Mapping[str, Any]],
    linha_b: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Num confronto ENTRE LIGAS, qual medicao manda.

    O DEFEITO QUE ISTO CONSERTA (relatado por ele em 15/09/2026): o cartao
    pedia a liga COMPARTILHADA do confronto, e num PRM x TCL nao existe uma —
    entao ele apagava e dizia "nao medida". Apagava justamente onde confianca
    mais importa: entre ligas o modelo precisa ligar duas escalas de rating
    diferentes, que e a conta mais dificil que ele faz.

    A ESCOLHA E A PIOR DAS DUAS, e nao a media. Confianca e sobre o % ser
    honesto; se o modelo e desonesto num dos lados, a odd do confronto herda
    isso. Fazer media deixaria a liga boa maquiar a ruim.

    NAO invento penalidade extra por ser entre ligas. Faz sentido que exista
    uma — a conta e mais dificil —, mas ninguem mediu, e chutar um desconto
    daria um numero com cara de medido. Fica so a pior das duas, e a tela diz
    que e confronto entre ligas.
    """
    qa, qb = pontos_da_liga(linha_a), pontos_da_liga(linha_b)
    if qa["usavel"] and qb["usavel"]:
        pior = linha_a if qa["pontos"] <= qb["pontos"] else linha_b
        return {"linha": pior, "entre_ligas": True,
                "ligas": [str((linha_a or {}).get("liga") or ""),
                          str((linha_b or {}).get("liga") or "")]}
    if qa["usavel"]:
        return {"linha": linha_a, "entre_ligas": False,
                "ligas": [str((linha_a or {}).get("liga") or "")]}
    if qb["usavel"]:
        return {"linha": linha_b, "entre_ligas": False,
                "ligas": [str((linha_b or {}).get("liga") or "")]}
    return {"linha": None, "entre_ligas": False, "ligas": []}


def montar_confianca(
    *,
    linha_liga: Optional[Mapping[str, Any]],
    idade_dias: Optional[float],
    liga: str = "",
    entre_ligas: bool = False,
    ajuste_do_confronto: float = 0.0,
    resumo_do_confronto: str = "",
    n_jogos_dos_times: float = float("nan"),
    limite_dias: float = LIMITE_DIAS,
) -> Dict[str, Any]:
    """O cartao inteiro: nota, pontuacao, fonte e o porque."""
    velha = (idade_dias is not None) and (float(idade_dias) > float(limite_dias))
    q = pontos_da_liga(linha_liga)

    if not q["usavel"] or velha:
        motivo = ("medicao de {:.0f} dias — velha demais para valer".format(idade_dias)
                  if velha else q["motivo"])
        return {
            "disponivel": False,
            "grade": "—",
            "score_text": "—",
            "source": "nao medida",
            "motivo": motivo,
            "porque": ("A confianca vem da medicao por liga. Rode a manutencao "
                       "para atualiza-la."),
        }

    score = 50.0 + float(q["pontos"]) + float(ajuste_do_confronto)

    nte = _num(n_jogos_dos_times)
    if math.isfinite(nte):
        if nte >= 25:
            score += 10.0
        elif nte >= 12:
            score += 5.0
        else:
            score -= 5.0

    score = float(max(0.0, min(100.0, score)))
    grade = "A (Alta)" if score >= 80.0 else ("B (Media)" if score >= 65.0 else "C (Baixa)")

    partes = list(q["notas"])
    if entre_ligas:
        # Dizer que e entre ligas importa: o numero saiu do lado PIOR, e sem
        # isso ele pareceria a medicao de uma liga so.
        partes.insert(0, "confronto entre ligas (vale o lado mais fraco)")
    if str(resumo_do_confronto or "").strip():
        partes.append(str(resumo_do_confronto).strip())

    return {
        "disponivel": True,
        "grade": grade,
        "score": score,
        "score_text": "{:d}/100".format(int(round(score))),
        "source": ("liga:{}".format(liga) if liga else "liga"),
        "idade_dias": (float(idade_dias) if idade_dias is not None else None),
        "pontos_da_liga": float(q["pontos"]),
        "ajuste_do_confronto": float(ajuste_do_confronto),
        "porque": " · ".join(p for p in partes if str(p).strip())[:300],
        "motivo": "ok",
    }
