# -*- coding: utf-8 -*-
"""Nota de jogador por DESEMPENHO, por rota — nao pelo resultado da partida.

╔══ O PROBLEMA QUE ISTO RESOLVE ═════════════════════════════════════════════╗
║ A camada de jogadores de producao (mlcore/players_layer.py, update_game)   ║
║ move a nota de cada jogador com `err = y - p`, onde y e "o time venceu".   ║
║ Ou seja: ganhou o time, sobe TODO MUNDO; perdeu, desce todo mundo. Quem    ║
║ jogou muito bem num time que perdeu e punido igual a quem jogou mal.       ║
║                                                                            ║
║ Medido na regua honesta em 11/09/2026, 8 dobras, n=8.494, teste pareado:   ║
║   producao hoje (camada bloqueada) .......... 7,09                         ║
║   + camada ANTIGA (Elo pelo resultado) ...... 7,13   +0,00233              ║
║   + camada NOVA (esta aqui) ................. 7,20   +0,00402              ║
║ e a nova bate a antiga por +0,00169, IC [+0,00049, +0,00289].              ║
╚════════════════════════════════════════════════════════════════════════════╝

╔══ FONTES ══════════════════════════════════════════════════════════════════╗
║ O desenho vem do PandaSkill, que e especifico de LoL profissional e tem o  ║
║ mesmo objetivo: avaliar jogador sem credita-lo so por estar no time que    ║
║ venceu.                                                                    ║
║                                                                            ║
║ Estagio 1, que e o que esta implementado aqui, textualmente:               ║
║   "For each role, we train a machine learning model that predicts the      ║
║    probability of a player winning the game based on their end-game        ║
║    statistics... The predicted probabilities are then transformed into     ║
║    percentiles."                                                           ║
║ https://arxiv.org/html/2501.10049  (aberta em 11/09/2026)                  ║
║                                                                            ║
║ DIFERENCA ASSUMIDA: eles convertem em PERCENTIL; aqui fica o logito cru.   ║
║ Nao foi medido qual e melhor. Fica como candidato, nao como certeza.       ║
║                                                                            ║
║ O QUE NAO FOI IMPLEMENTADO, de proposito: o estagio 2 deles atualiza a     ║
║ habilidade por um confronto de todos-contra-todos entre os 10 jogadores,   ║
║ ordenados pela nota de desempenho. Aqui a nota e a MEDIA ACUMULADA das     ║
║ notas anteriores. E mais pobre — nao sabe contra quem jogou nem carrega    ║
║ incerteza — mas e o que foi MEDIDO. Subir o que nao foi medido seria       ║
║ trocar evidencia por expectativa.                                          ║
║                                                                            ║
║ SEM DECAIMENTO, tambem de proposito: testado em 11/09 com meia-vida de     ║
║ 180 dias (a que o projeto ja usa noutra camada) e a nota PIOROU.           ║
╚════════════════════════════════════════════════════════════════════════════╝

PONTO NO TEMPO, que e o que impede a nota de mentir:

- os pesos por rota e a padronizacao saem SO de jogos ANTERIORES ao corte;
- a nota de um jogador num jogo usa SO os jogos anteriores dele — a soma
  acumulada exclui o jogo corrente de proposito.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from mlcore.players_layer import norm_pos

# Estatisticas de fim de jogo lidas do CSV. Nem todo dump tem todas; o que
# faltar e simplesmente ignorado, e a lista de atributos encolhe.
COLUNAS_ESTATISTICAS: List[str] = [
    "kills", "deaths", "assists", "teamkills", "dpm", "damageshare",
    "earnedgoldshare", "cspm", "visionscore", "golddiffat15",
]

# A ordem importa: e ela que casa os pesos com as colunas na hora de servir.
ATRIBUTOS: List[str] = [
    "kp", "deaths", "dpm", "damageshare", "earnedgoldshare",
    "cspm", "visionscore", "golddiffat15",
]

# Jogos-equivalentes de prior na media acumulada. Segura a nota de quem tem
# poucos jogos perto de zero, em vez de deixar dois jogos bons virarem craque.
BETA = 10.0

# Abaixo disto uma rota nao tem jogo suficiente para aprender peso proprio.
MIN_JOGOS_POR_ROTA = 300
# Abaixo disto nem vale tentar: o conjunto anterior ao corte e pequeno demais.
MIN_JOGOS_TOTAL = 2000


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def carregar_linhas(csvs: Iterable[str]) -> pd.DataFrame:
    """Le as linhas de jogador com as estatisticas de fim de jogo.

    A CHAVE DO JOGADOR e derivada exatamente como em
    `mlcore/players_layer.load_match_player_rows`: `playerid`, caindo para
    `playername` quando vier vazio. Se divergir disso, a nota calculada aqui
    nao casa com a escalacao que o app entrega, e a camada vira ruido.
    """
    base = ["gameid", "date", "participantid", "position", "playerid",
            "playername", "result", "teamname"]
    quadros = []
    for c in csvs:
        espiada = pd.read_csv(c, nrows=1, low_memory=False)
        uso = [x for x in base + COLUNAS_ESTATISTICAS if x in espiada.columns]
        quadros.append(pd.read_csv(c, low_memory=False, usecols=uso))
    df = pd.concat(quadros, ignore_index=True)

    df = df[(_num(df["participantid"]) >= 1) & (_num(df["participantid"]) <= 10)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "position", "result"])

    if "playerid" in df.columns:
        chave = df["playerid"].astype(str).str.strip()
        vazio = chave.eq("") | chave.eq("nan")
        chave = chave.mask(vazio, df.get("playername", pd.Series("", index=df.index)).astype(str).str.strip())
    else:
        chave = df["playername"].astype(str).str.strip()
    df["player_key"] = chave
    df = df[df["player_key"].ne("") & df["player_key"].ne("nan")].copy()

    df["pos_can"] = df["position"].map(norm_pos)
    df = df[df["pos_can"].astype(str).ne("")].copy()
    df["lado"] = np.where(_num(df["participantid"]) <= 5, "blue", "red")
    df = df.sort_values(["date", "gameid"]).reset_index(drop=True)

    # Participacao em abates normaliza por quao sangrento foi o jogo. E melhor
    # que KDA cru, que premia quem joga partida cheia de briga.
    if {"kills", "assists", "teamkills"}.issubset(df.columns):
        tk = _num(df["teamkills"])
        df["kp"] = ((_num(df["kills"]).fillna(0) + _num(df["assists"]).fillna(0))
                    / tk.replace(0, np.nan)).fillna(0.0)
    return df


def _newton_logistico(X: np.ndarray, y: np.ndarray, passos: int = 60) -> Optional[np.ndarray]:
    """Regressao logistica por Newton, com crista minima para nao estourar."""
    w = np.zeros(X.shape[1], dtype=float)
    for _ in range(passos):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30.0, 30.0)))
        g = X.T @ (y - p)
        H = -(X.T * (p * (1.0 - p))) @ X - 1e-6 * np.eye(X.shape[1])
        try:
            passo = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            return None
        w = w + passo
        if not np.all(np.isfinite(w)):
            return None
        if float(np.max(np.abs(passo))) < 1e-9:
            break
    return w


def ajustar(df: pd.DataFrame, *, ate: Optional[date]) -> Dict[str, Any]:
    """Aprende os pesos por rota usando SO os jogos ANTERIORES a `ate`.

    `ate` normalmente e a data de corte do treino. Passar None usa tudo, o que
    so faz sentido fora de avaliacao — dentro dela vazaria futuro.
    """
    atributos = [a for a in ATRIBUTOS if a in df.columns]
    if not atributos:
        raise ValueError("nenhum atributo de desempenho disponivel no CSV")

    if ate is None:
        treino = np.ones(len(df), dtype=bool)
    else:
        treino = (df["date"].dt.date < ate).to_numpy()
    if int(treino.sum()) < MIN_JOGOS_TOTAL:
        raise ValueError(
            "poucas linhas antes do corte ({}) para aprender os pesos".format(int(treino.sum())))

    y = _num(df["result"]).fillna(0).to_numpy(float)
    pesos: Dict[str, List[float]] = {}
    padrao: Dict[str, Dict[str, List[float]]] = {}

    for rota, idx in df.groupby("pos_can", sort=False).indices.items():
        tr = np.asarray([i for i in idx if treino[i]], dtype=int)
        if len(tr) < MIN_JOGOS_POR_ROTA:
            continue
        medias, desvios = [], []
        Xtr = np.zeros((len(tr), len(atributos)), dtype=float)
        for j, a in enumerate(atributos):
            v = _num(df[a]).fillna(0.0).to_numpy(float)
            m = float(v[tr].mean())
            s = float(v[tr].std()) or 1.0
            medias.append(m)
            desvios.append(s)
            Xtr[:, j] = (v[tr] - m) / s
        w = _newton_logistico(np.column_stack([Xtr, np.ones(len(tr))]), y[tr])
        if w is None:
            continue
        pesos[str(rota)] = [float(x) for x in w]
        padrao[str(rota)] = {"media": medias, "desvio": desvios}

    if not pesos:
        raise ValueError("nenhuma rota teve jogos suficientes para aprender peso")

    return {
        "versao": 1,
        "atributos": list(atributos),
        "beta": float(BETA),
        "corte": (ate.isoformat() if ate else None),
        "pesos_por_rota": pesos,
        "padronizacao": padrao,
    }


def calcular_pscore(df: pd.DataFrame, ajuste: Mapping[str, Any]) -> pd.Series:
    """Nota de DESEMPENHO de cada linha (jogador naquele jogo)."""
    atributos = list(ajuste.get("atributos") or [])
    pesos = dict(ajuste.get("pesos_por_rota") or {})
    padrao = dict(ajuste.get("padronizacao") or {})
    out = np.zeros(len(df), dtype=float)

    for rota, idx in df.groupby("pos_can", sort=False).indices.items():
        w = pesos.get(str(rota))
        pad = padrao.get(str(rota))
        if not w or not pad:
            continue
        idx = np.asarray(idx, dtype=int)
        X = np.zeros((len(idx), len(atributos)), dtype=float)
        for j, a in enumerate(atributos):
            v = _num(df[a]).fillna(0.0).to_numpy(float) if a in df.columns else np.zeros(len(df))
            m = float(pad["media"][j])
            s = float(pad["desvio"][j]) or 1.0
            X[:, j] = (v[idx] - m) / s
        out[idx] = np.column_stack([X, np.ones(len(idx))]) @ np.asarray(w, dtype=float)
    return pd.Series(out, index=df.index, name="pscore")


def notas_ponto_no_tempo(df: pd.DataFrame, pscore: pd.Series, *, beta: float = BETA) -> pd.Series:
    """Nota de cada jogador ANTES de cada jogo dele.

    A soma acumulada EXCLUI o jogo corrente de proposito — incluir seria usar o
    desempenho do proprio jogo que se quer prever.
    """
    d = df[["player_key", "date", "gameid"]].copy()
    d["pscore"] = pscore.to_numpy()
    d = d.sort_values(["player_key", "date", "gameid"])
    g = d.groupby("player_key", sort=False)
    nota = (g["pscore"].cumsum() - d["pscore"]) / (g.cumcount() + float(beta))
    return nota.reindex(df.index)


def notas_finais(df: pd.DataFrame, pscore: pd.Series, *, beta: float = BETA) -> Dict[str, Dict[str, float]]:
    """Nota atual de cada jogador+rota, para servir uma partida futura.

    Aqui a soma acumulada INCLUI todos os jogos, porque a partida a prever
    ainda nao aconteceu — nao ha o que excluir.
    """
    d = df[["player_key", "pos_can"]].copy()
    d["pscore"] = pscore.to_numpy()
    ag = d.groupby(["player_key", "pos_can"], sort=False)["pscore"].agg(["sum", "count"])
    out: Dict[str, Dict[str, float]] = {}
    for (chave, rota), linha in ag.iterrows():
        n = int(linha["count"])
        out["{}|||{}".format(chave, rota)] = {
            "n": float(linha["sum"]) / (n + float(beta)),
            "gp": n,
        }
    return out


def indice_de_apelidos(df: pd.DataFrame) -> Dict[str, str]:
    """"nome|||Rota" -> "player_key|||Rota", para quem so tem o nome.

    O QUE ACONTECEU DE VERDADE (13-14/09/2026), e o meu erro no meio
    As notas sao chaveadas por `playerid` (`oe:player:...`), porque id
    sobrevive a troca de nick.

    O caminho da ODD manda ID e sempre funcionou: a escalacao dele vem de
    `lineup_ml_*`, que guarda a chave (o editor mostra o nome, mas grava a
    chave). Conferido no app dirigido em 14/09/2026 com o artefato de
    producao: applied=True, cobertura 5/5.

    O RADAR e que manda NOME: o montador de draft dele usa `latest_roster`,
    que devolve `playername`. Sem indice, nenhuma nota resolvia ali — e era
    por isso que nao dava para mostrar a nota por rota no Radar.

    MEU ERRO, registrado de proposito: em 13/09 eu testei o caminho servido
    com a escalacao do RADAR (nomes), vi 0/5 e anunciei que a camada estava
    inerte em producao. Nao estava. Peguei a lista errada e concluí grande
    demais a partir de um teste so. O indice continua valendo — e do Radar a
    necessidade —, mas o motivo e outro.

    AMBIGUIDADE NAO E RESOLVIDA NO CHUTE. Medido no CSV de 2026: de 2.838
    pares (nome, rota), 8 tem mais de um id — 0,28%. Esses 8 ficam DE FORA do
    indice. Para eles a cobertura cai e a camada nao age, que e o certo:
    escolher "o mais provavel" seria somar a nota do jogador errado.
    """
    if df is None or df.empty:
        return {}
    faltando = [c for c in ("playername", "pos_can", "player_key") if c not in df.columns]
    if faltando:
        return {}

    vistos: Dict[str, set] = {}
    for nome, rota, chave in zip(df["playername"].astype(str),
                                 df["pos_can"].astype(str),
                                 df["player_key"].astype(str)):
        n = nome.strip()
        r = rota.strip()
        k = chave.strip()
        if not n or not r or not k:
            continue
        vistos.setdefault("{}|||{}".format(n.casefold(), r), set()).add(k)

    return {apelido: next(iter(chaves))
            for apelido, chaves in vistos.items() if len(chaves) == 1}


def resolver_chave(
    notas: Mapping[str, Any],
    apelidos: Optional[Mapping[str, str]],
    jogador: str,
    rota: str,
) -> str:
    """A chave de `notas` para este jogador, ou "" se nao der para saber.

    Tenta primeiro o que veio (ja pode ser o id). So depois cai no indice de
    nomes. A ordem importa: quem ja manda o id nao paga por ambiguidade de
    nome de terceiro.
    """
    direta = "{}|||{}".format(str(jogador).strip(), str(rota).strip())
    if direta in notas:
        return direta
    if not apelidos:
        return ""
    alvo = apelidos.get("{}|||{}".format(
        str(jogador).strip().casefold(), str(rota).strip()))
    if not alvo:
        return ""
    por_apelido = "{}|||{}".format(alvo, str(rota).strip())
    return por_apelido if por_apelido in notas else ""


def delta_da_escalacao(
    notas: Mapping[str, Mapping[str, float]],
    *,
    blue_lineup: Mapping[str, str],
    red_lineup: Mapping[str, str],
    apelidos: Optional[Mapping[str, str]] = None,
) -> Tuple[float, int]:
    """Soma das notas do lado azul menos a do vermelho, e quantos foram achados.

    PRINCIPIO IV: jogador sem nota NAO vira zero disfarcado de nota — ele nao
    entra na soma, e a contagem devolvida diz quantos entraram, para quem
    chamar poder decidir se ha cobertura suficiente.
    """
    def _soma(lineup: Mapping[str, str]) -> Tuple[float, int]:
        total, achados = 0.0, 0
        for pos, jogador in (lineup or {}).items():
            rota = norm_pos(pos)
            if not rota or not str(jogador or "").strip():
                continue
            chave = resolver_chave(notas, apelidos, str(jogador), rota)
            reg = notas.get(chave) if chave else None
            if not reg:
                continue
            total += float(reg.get("n", 0.0) or 0.0)
            achados += 1
        return total, achados

    sb, nb = _soma(blue_lineup)
    sr, nr = _soma(red_lineup)
    return float(sb - sr), int(min(nb, nr))


def deltas_por_jogo(df: pd.DataFrame, pscore: pd.Series, *, beta: float = BETA) -> Dict[str, float]:
    """gameid -> (soma das notas do azul) menos (soma das notas do vermelho).

    Usa a nota PONTO NO TEMPO de cada jogador: para cada jogo, so o que ele
    tinha feito ANTES. Serve para ajustar o peso da camada num conjunto grande
    de jogos sem vazar futuro em nenhum deles.

    Diferente de `notas_finais`, que congela a nota num instante so — aqui
    cada jogo enxerga o seu proprio passado, e nada alem dele.
    """
    d = df[["gameid", "lado"]].copy()
    d["nota"] = notas_ponto_no_tempo(df, pscore, beta=beta).to_numpy()
    ag = d.groupby(["gameid", "lado"], sort=False)["nota"].sum().unstack("lado")
    if "blue" not in ag.columns or "red" not in ag.columns:
        return {}
    ag = ag.dropna()
    return {str(k): float(v) for k, v in (ag["blue"] - ag["red"]).items()}


def construir(
    csvs: Iterable[str],
    *,
    ate: Optional[date],
    ate_notas: Optional[date] = None,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Monta o pacote: pesos, padronizacao e nota atual por jogador.

    SAO DOIS CORTES, e confundi-los vaza futuro:

    - `ate` corta o que os PESOS por rota enxergam;
    - `ate_notas` corta o que as NOTAS de cada jogador enxergam.

    Para AJUSTAR o peso da camada no holdout, os dois tem de valer o corte do
    treino: usar notas que ja viram o holdout faria o peso ser escolhido com
    resposta na mao.

    Para SERVIR uma partida futura, `ate_notas` fica None — todo jogo passado
    e passado, e a partida a prever ainda nao aconteceu.
    """
    if df is None:
        df = carregar_linhas(csvs)
    ajuste = ajustar(df, ate=ate)
    if ate_notas is not None:
        d_notas = df[df["date"].dt.date < ate_notas]
    else:
        d_notas = df
    ajuste["notas"] = notas_finais(d_notas, calcular_pscore(d_notas, ajuste))
    ajuste["corte_notas"] = (ate_notas.isoformat() if ate_notas else None)
    # O indice de nomes e montado sobre AS MESMAS linhas das notas, para nao
    # prometer uma chave que nao tem nota do outro lado.
    ajuste["apelidos"] = indice_de_apelidos(d_notas)
    ajuste["n_linhas"] = int(len(df))
    ajuste["n_linhas_notas"] = int(len(d_notas))
    return ajuste
