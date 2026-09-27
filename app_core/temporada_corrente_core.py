# -*- coding: utf-8 -*-
"""Qual ano de competicao esta EM CURSO nos dados carregados.

POR QUE ISTO EXISTE (13/09/2026)
O dono abriu o Radar, procurou um time pelo nome exato e a tela disse que nao
existia. O time existia — a LISTA e que tinha 18 times em vez de 394.

Tres telas (Radar, Campeoes, Players) escolhiam a temporada assim:

    ano = max(anos presentes)

e isso ficou errado sem ninguem mexer em nada. As ligas regionais europeias
(EBL, LFL, LIT, PRM) comecaram o split WINTER de 2027 em 28/08/2026. A fonte
marca esses jogos com year=2027, o que esta certo: e a temporada 2027 deles.
Entao, desde 28/08, "o maior ano" passou a ser 2027 — 46 jogos, 18 times —
enquanto a temporada 2026 seguia rodando com 394 times e jogos ate 12/09.

Duas temporadas se sobrepoem no calendario. "A ultima" nao quer dizer "a que
esta sendo jogada", e a diferenca so aparece nesta janela do ano — o que
torna o defeito sazonal e facil de nao ver.

A REGRA
Temporada corrente = a que tem MAIS JOGOS na janela recente dos dados.
Nao ha limiar inventado de "quantos jogos fazem uma temporada": compara-se
volume contra volume. E se corrige sozinha — quando a 2026 acabar e a 2027
estiver a todo vapor, a janela passa a ser dominada pela 2027 sem ninguem
mexer aqui.

A janela e contada a partir da DATA MAIS NOVA DOS DADOS, nunca do relogio da
maquina. Com o relogio, abrir um CSV antigo daria janela vazia e a conta
cairia no ramo de emergencia sem motivo.
"""
from __future__ import annotations

import warnings
from typing import Any, Optional

import pandas as pd

# Seis semanas. Todo split joga toda semana, entao a janela cobre rodadas
# suficientes de TODAS as ligas para que a temporada dominante fique clara —
# e e curta o bastante para uma temporada encerrada sair dela rapido.
JANELA_DIAS_PADRAO = 45


def ano_da_temporada_corrente(
    df: Any,
    *,
    col_ano: str = "year",
    col_data: str = "date",
    janela_dias: int = JANELA_DIAS_PADRAO,
) -> Optional[int]:
    """Ano da temporada em curso, ou None quando nao da para dizer.

    PRINCIPIO IV: sem dado, devolve None — nunca um ano plausivel inventado.
    """
    if df is None or not hasattr(df, "empty") or df.empty:
        return None
    if col_ano not in getattr(df, "columns", []):
        return None

    anos = pd.to_numeric(df[col_ano], errors="coerce")
    if not anos.notna().any():
        return None

    if col_data not in getattr(df, "columns", []):
        return int(anos.max())

    # O aviso de "nao consegui inferir o formato" nao ajuda ninguem aqui: a
    # coluna vem de CSV de terceiro e datas ilegiveis ja estao tratadas logo
    # abaixo. Silenciado SO nesta chamada, para nao esconder aviso de outro.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        datas = pd.to_datetime(df[col_data], errors="coerce")
    if not datas.notna().any():
        # Sem data nao da para saber o que e recente. O maior ano volta a ser
        # o melhor palpite disponivel, e esta dito que e palpite.
        return int(anos.max())

    fim = datas.max()
    inicio = fim - pd.Timedelta(days=int(janela_dias))
    recente = anos[(datas >= inicio) & datas.notna() & anos.notna()]
    if recente.empty:
        return int(anos.max())

    contagem = recente.astype(int).value_counts()
    maior = int(contagem.max())
    # Empate no volume -> fica com o ano maior, que e a temporada mais nova.
    empatados = [int(a) for a, n in contagem.items() if int(n) == maior]
    return max(empatados)


def anos_disponiveis(df: Any, *, col_ano: str = "year") -> list:
    """Todos os anos presentes, do mais novo para o mais velho.

    Serve para a tela OFERECER a outra temporada em vez de esconde-la: o
    conserto do padrao nao pode virar um novo jeito de sumir com times.
    """
    if df is None or not hasattr(df, "empty") or df.empty:
        return []
    if col_ano not in getattr(df, "columns", []):
        return []
    anos = pd.to_numeric(df[col_ano], errors="coerce").dropna()
    if anos.empty:
        return []
    return sorted({int(a) for a in anos}, reverse=True)
