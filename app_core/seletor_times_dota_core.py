# -*- coding: utf-8 -*-
"""Quem aparece no seletor de times do Dota (08/11/2026).

Dois times com o mesmo nome ficam separados: o de menos jogos ganha o sufixo "#id". Certo para
o modelo, ruim no seletor: "HULIGANI" (88 jogos, Premier Series) e "HULIGANI #10225339" (1 jogo
de um time amador, "HULIGANIFAKE" na OpenDota) pareciam o mesmo time, e escolher o segundo dava
jogo "even". Sai do seletor o homonimo sem historia; os jogos dele continuam na base.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Mapping

import pandas as pd

_SUFIXO = re.compile(r"^(.*) #\d+$")
MAX_JOGOS_SEM_HISTORIA = 2
DIAS_PARADO = 30
FRACAO_DO_HOMONIMO = 0.2


def some_do_seletor(nome: str, jogos: Mapping[str, int], ultimo: Mapping[str, pd.Timestamp],
                    hoje: pd.Timestamp) -> bool:
    """So homonimo com sufixo: ate 2 jogos, ou parado ha 30+ dias com 1/5 dos jogos do dono do nome."""
    m = _SUFIXO.match(str(nome))
    if not m or m.group(1) not in jogos:
        return False
    n = int(jogos.get(nome, 0))
    if n <= MAX_JOGOS_SEM_HISTORIA:
        return True
    visto = ultimo.get(nome)
    parado = visto is not None and (hoje - visto).days > DIAS_PARADO
    return parado and n <= FRACAO_DO_HOMONIMO * int(jogos[m.group(1)])


def times_do_seletor(times: Iterable[str], team_games: pd.DataFrame) -> List[str]:
    lista = list(times)
    if team_games is None or team_games.empty or not {"team", "date"}.issubset(team_games.columns):
        return lista
    datas = pd.to_datetime(team_games["date"], errors="coerce")
    g = team_games.assign(_d=datas).groupby(team_games["team"].astype(str))["_d"]
    jogos, ultimo = g.size().to_dict(), g.max().to_dict()
    hoje = datas.max()
    return [t for t in lista if not some_do_seletor(t, jogos, ultimo, hoje)]
