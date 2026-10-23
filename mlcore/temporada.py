# -*- coding: utf-8 -*-
"""Escolher QUAL temporada a régua mede — sem cair em rótulo esquisito.

╔══ FONTES ═══════════════════════════════════════════════════════════════════╗
║                                                                             ║
║ POR QUE ESTE ARQUIVO EXISTE (06/09/2026):                                   ║
║   `ml_fusion_scorecard.py` e `nota_por_liga.py` escolhiam a temporada com   ║
║   `max(year)`. Em 02/09/2026 a fonte passou a marcar 60 linhas da liga EBL  ║
║   com `year = 2027` (data real 2026-09-02). As duas réguas foram para 2027, ║
║   acharam 5 jogos, e:                                                       ║
║     - a NOTA OFICIAL virou `nan`;                                           ║
║     - a NOTA POR LIGA estourou com KeyError: 'nota'.                        ║
║   E o vigia semanal NÃO disparou, porque `nan` não é maior nem menor que o  ║
║   piso. A nota oficial simplesmente deixou de existir, calada — que é       ║
║   exatamente o que este projeto inteiro foi construído para impedir.        ║
║                                                                             ║
║ 1. AUSÊNCIA DE DADO NÃO PODE VIRAR UM NÚMERO QUALQUER:                      ║
║    "absence of data must not be representable as zero. Our metrics now      ║
║    return null when nothing was measured, carry the failure count alongside ║
║    the result, and print an explicit no-valid-measurement warning instead   ║
║    of a percentage" — e "fail loudly at the boundary".                      ║
║    Por isso aqui: quando nenhuma temporada tem amostra suficiente, isto     ║
║    LEVANTA erro em vez de devolver um palpite, e o chamador sai com código  ║
║    diferente de zero para o CI ver.                                         ║
║    https://thicket.sh/blog/when-your-metric-is-broken-not-your-business     ║
║                                                                             ║
║ 2. RÓTULO FORA DA FAIXA É ANOMALIA DE DADO, E TEM QUE APARECER:             ║
║    a prática de validação de dados em produção define o domínio esperado de ║
║    cada feature (faixa, para inteiro) e trata valor fora dela como          ║
║    ANOMALIA — não como dado normal. É o item DADOS 1 da régua das 28        ║
║    ("expectativas das features num esquema"), que está VERMELHO nos dois    ║
║    repositórios. Enquanto não houver o contrato de dados, esta função ao    ║
║    menos IMPRIME o que descartou, para o rótulo esquisito ficar visível em  ║
║    vez de derrubar a nota em silêncio.                                      ║
║    https://www.tensorflow.org/tfx/data_validation/get_started               ║
║                                                                             ║
║ 3. POR QUE "A MAIS RECENTE COM AMOSTRA", e não "a do calendário":           ║
║    o ano de calendário quebraria em janeiro, quando a temporada nova ainda  ║
║    não tem jogos; e fixar um número exige alguém lembrar de trocar todo     ║
║    ano. "A mais recente que tem amostra suficiente" atravessa a virada de   ║
║    ano sozinha e ignora rótulo de temporada que mal começou.                ║
╚═════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Optional, Tuple

# Abaixo disto uma temporada nao tem amostra para sustentar nota nenhuma. Nao e
# numero cravado por gosto: com 5 jogos (o caso de 2027) a logloss por segmento
# ja saia `nan`, e as notas por liga usam min-n de 80 por liga.
MIN_JOGOS_TEMPORADA = 200


class TemporadaInsuficiente(RuntimeError):
    """Nenhuma temporada tem amostra. Falhar alto e melhor que chutar (fonte 1)."""


def escolher_temporada(
    anos: Iterable[Any],
    *,
    min_n: int = MIN_JOGOS_TEMPORADA,
    explicito: Optional[int] = None,
) -> Tuple[int, dict]:
    """Devolve (temporada, diagnostico).

    `anos` e a coluna `year` inteira, uma entrada por linha/jogo.
    `explicito` vence sempre — mas o diagnostico continua dizendo o tamanho da
    amostra, para quem forcou ver que forcou uma temporada vazia.

    Sem `explicito`: a temporada mais RECENTE que tenha ao menos `min_n`
    linhas. Rotulos com amostra menor entram em `descartadas` para serem
    IMPRESSOS, nao para sumirem (fonte 2).
    """
    contagem: Counter = Counter()
    for a in anos:
        try:
            v = int(a)
        except (TypeError, ValueError):
            continue
        contagem[v] += 1

    if not contagem:
        raise TemporadaInsuficiente(
            "nenhum valor de temporada legivel na coluna `year`")

    diag = {
        "min_n": int(min_n),
        "por_temporada": dict(sorted(contagem.items())),
        "descartadas": [],
        "forcada": explicito is not None,
    }

    if explicito is not None:
        n = contagem.get(int(explicito), 0)
        diag["n"] = n
        diag["temporada"] = int(explicito)
        return int(explicito), diag

    elegiveis = [a for a, n in contagem.items() if n >= int(min_n)]
    descartadas = [(a, n) for a, n in sorted(contagem.items(), reverse=True)
                   if n < int(min_n)]
    diag["descartadas"] = descartadas

    if not elegiveis:
        raise TemporadaInsuficiente(
            f"nenhuma temporada com pelo menos {min_n} linhas; "
            f"presentes: {dict(sorted(contagem.items()))}")

    escolhida = max(elegiveis)
    diag["temporada"] = escolhida
    diag["n"] = contagem[escolhida]
    return escolhida, diag


def aviso(diag: dict) -> str:
    """Texto a IMPRIMIR sempre — inclusive quando esta tudo bem.

    Regua que so fala quando quebra nao ensina ninguem a ler o normal.
    """
    linhas = [f"temporada medida: {diag['temporada']} "
              f"({diag.get('n', 0)} linhas, minimo {diag['min_n']})"]
    if diag.get("forcada"):
        linhas.append("  (FORCADA por --season)")
        if diag.get("n", 0) < diag["min_n"]:
            linhas.append("  ATENCAO: amostra ABAIXO do minimo — a nota nao "
                          "vai significar grande coisa.")
    for ano, n in diag.get("descartadas", []):
        linhas.append(f"  descartada: temporada {ano} tem so {n} linha(s) — "
                      f"rotulo de temporada que mal comecou, ou erro da fonte.")
    return "\n".join(linhas)
