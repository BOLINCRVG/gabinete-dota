# -*- coding: utf-8 -*-
"""O METODO DELE numa linha de TOTAL (kills, torres, dragoes, baroes, inibidores, tempo).

Para o MENOS de L (no MAIS e o espelho), os dois times juntos (28-29/10/2026, nas
palavras dele):
  1. Medias     -- "a media recente dos dois times tem que estar menor que a linha"
                   (ultimos 10 de cada).
  2. Adversario -- "se as medias estao mais altas e os adversarios eram over: so foi over
                   porque os adversarios eram over; se eram under, a analise deu errado".
                   Conta: a media AJUSTADA pelos adversarios enfrentados (desconta uma
                   fracao -- medida por mercado -- do quanto eles puxavam o total acima
                   da liga) do lado certo da linha.
  3. Historico  -- "5 / 20 / 10 / ano; o ano eu menos ligo": os ultimos 5, 20 e 10 (dos
                   dois somados) com 60%+ do lado. O ano so aparece.
  4. Stats da linha -- "pego no maximo 30% do outro lado": a media dos dois, no
                   campeonato, do OUTRO lado da linha <= 30%.
  Referencia dele: 1,72 / 1,67 (58% / 60%).

MEDIDO por mercado (scratchpad metodo_geral.py, 29/10/2026), cego 2026 com pesos de
2025, linhas perto da media: ver BATEU. A chance (fair) e uma logistica nas mesmas
medidas, ganho sobre so o historico: kills +0,016, torres +0,024, dragoes +0,031,
tempo +0,015, baroes +0,044, inibidores +0,039. Baroes e inibidores (poucos valores
possiveis) tem a contagem de passos instavel: o painel avisa.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pandas as pd

JANELAS = (5, 20, 10)              # a ordem dele
MINIMO_POR_TIME = 5
MINIMO_CAMPEONATO = 3
PASSA_HIST = 0.60
TETO_OUTRO_LADO = 0.30
NOMES = ("Médias", "Adversário", "Histórico", "Stats da linha")

# metrica -> colunas (a favor, contra) ou a coluna do tempo; ajuste do adversario medido;
# pesos (b0, media-L, media_aj-L, logit h5, logit h10, logit h20, logit stats) -> logit P(mais);
# o que bateu no cego com 0..4 passos a favor (linhas perto da media)
METRICAS: Dict[str, Dict[str, Any]] = {
    "kills": {"cols": ("kills_for", "kills_against"), "ajuste": 0.46, "unidade": "kills",
              "pesos": (-0.0068, -0.0278, 0.1472, 0.0046, -0.2024, 0.4476, 0.1338),
              "bateu": {"over": {0: 34, 1: 47, 2: 56, 3: 63, 4: 70}, "under": {0: 41, 1: 53, 2: 59, 3: 67, 4: 71}}},
    "towers": {"cols": ("towers_for", "towers_against"), "ajuste": 0.38, "unidade": "torres",
               "pesos": (-0.0633, 0.0737, 0.5861, 0.0003, -0.4066, 0.5907, 0.1233),
               "bateu": {"over": {0: 32, 1: 45, 2: 56, 3: 68, 4: 72}, "under": {0: 39, 1: 54, 2: 59, 3: 66, 4: 73}}},
    "dragons": {"cols": ("dragons_for", "dragons_against"), "ajuste": 0.42, "unidade": "dragões",
                "pesos": (0.0340, 0.6868, 1.0331, -0.0042, -0.3335, 0.3814, 0.1265),
                "bateu": {"over": {0: 32, 1: 50, 2: 62, 3: 65, 4: 76}, "under": {0: 36, 1: 41, 2: 57, 3: 63, 4: 74}}},
    "barons": {"cols": ("barons_for", "barons_against"), "ajuste": 0.41, "unidade": "barões", "instavel": True,
               "pesos": (0.0562, 1.2615, 1.3517, -0.0504, -0.3025, 0.3516, 0.1531),
               "bateu": {"over": {0: 32, 1: 40, 2: 68, 3: 59, 4: 82}, "under": {0: 50, 1: 32, 2: 46, 3: 63, 4: 73}}},
    "inhibitors": {"cols": ("inhibitors_for", "inhibitors_against"), "ajuste": 0.45, "unidade": "inibidores",
                   "instavel": True,
                   "pesos": (0.0504, 0.8246, 1.0123, -0.0168, -0.3842, 0.7102, 0.0197),
                   "bateu": {"over": {0: 26, 1: 39, 2: 55, 3: 47, 4: 49},
                             "under": {0: 48, 1: 42, 2: 54, 3: 69, 4: 77}}},
    "time": {"cols": None, "ajuste": 0.40, "unidade": "min",
             "pesos": (-0.0942, -0.0637, 0.2318, -0.0090, -0.1413, 0.4924, 0.1270),
             "bateu": {"over": {0: 34, 1: 45, 2: 54, 3: 62, 4: 69}, "under": {0: 42, 1: 54, 2: 62, 3: 67, 4: 71}}},
}
# coluna da Analise de Linha -> metrica
COLUNA_PARA_METRICA = {"total_kills": "kills", "mv2_total_kills": "kills", "mv2_total_towers": "towers",
                       "total_towers": "towers", "mv2_total_dragons": "dragons", "total_dragons": "dragons",
                       "mv2_total_barons": "barons", "total_nashors": "barons",
                       "mv2_total_inhibitors": "inhibitors", "total_inhibitors": "inhibitors",
                       "game_time_min": "time"}


# DOTA (15/11/2026). Arquivo identico nos dois repositorios; no Dota (dota_paths.py na raiz) a
# tabela e a MEDIDA NO DOTA, mesma receita: pesos e ajuste em 2025, cego em 2026 (10,6 mil linhas
# de 3,6 mil jogos, linhas perto da media). Ganho da fair sobre so o historico:
#   kills +0,0024 IC95 [+0,0008; +0,0041]   roshans +0,0027 [+0,0011; +0,0044]   (passaram)
#   torres +0,0013 [-0,0007; +0,0034]   tempo +0,0010 [-0,0008; +0,0029]   barracas -0,0005
#   (empatam com o historico: a fair nao piora, mas o ganho nao esta provado)
# Os passos separam como no LoL: com 3-4 a favor bate 58-69%. Roshans e barracas tem poucos
# valores possiveis (linha 0,5 / 1,5 / 2,5): contagem de passos instavel, o painel avisa.
# "barons" = Roshans e "inhibitors" = barracas (os nomes das colunas do CSV do Dota).
def _e_dota() -> bool:
    from pathlib import Path
    return (Path(__file__).resolve().parents[1] / "dota_paths.py").exists()


E_DOTA = _e_dota()
METRICAS_DOTA: Dict[str, Dict[str, Any]] = {
    "kills": {"cols": ("kills_for", "kills_against"), "ajuste": 0.15, "unidade": "kills",
              "pesos": (-0.0817, -0.0563, 0.1239, 0.0465, -0.3811, 0.5141, 0.1468),
              "bateu": {"over": {0: 40, 1: 47, 2: 51, 3: 58, 4: 66}, "under": {0: 47, 1: 54, 2: 58, 3: 62, 4: 62}}},
    "towers": {"cols": ("towers_for", "towers_against"), "ajuste": 0.60, "unidade": "torres",
               "pesos": (-0.0461, 0.0904, 0.2015, -0.0145, -0.2672, 0.4977, 0.2035),
               "bateu": {"over": {0: 39, 1: 51, 2: 56, 3: 66, 4: 66}, "under": {0: 40, 1: 49, 2: 58, 3: 63, 4: 66}}},
    "barons": {"cols": ("barons_for", "barons_against"), "ajuste": 0.75, "unidade": "roshans", "instavel": True,
               "pesos": (-0.0358, 0.3255, 0.3613, -0.0028, -0.1569, 0.5660, 0.2123),
               "bateu": {"over": {0: 22, 1: 46, 2: 58, 3: 69, 4: 84}, "under": {0: 24, 1: 55, 2: 63, 3: 69, 4: 82}}},
    "inhibitors": {"cols": ("inhibitors_for", "inhibitors_against"), "ajuste": 0.90, "unidade": "barracas",
                   "instavel": True,
                   "pesos": (-0.0161, -0.0458, 0.2417, -0.0239, -0.1207, 0.5718, 0.2308),
                   "bateu": {"over": {0: 41, 1: 52, 2: 60, 3: 69, 4: 73}, "under": {0: 33, 1: 42, 2: 53, 3: 56, 4: 82}}},
    "time": {"cols": None, "ajuste": 0.30, "unidade": "min",
             "pesos": (-0.0577, -0.0157, 0.0953, -0.0304, -0.3071, 0.5926, 0.2245),
             "bateu": {"over": {0: 40, 1: 46, 2: 50, 3: 59, 4: 68}, "under": {0: 48, 1: 54, 2: 57, 3: 62, 4: 65}}},
}
if E_DOTA:
    METRICAS = dict(METRICAS_DOTA)
    COLUNA_PARA_METRICA = {k: v for k, v in COLUNA_PARA_METRICA.items() if v in METRICAS_DOTA}
PESOS = METRICAS["kills"]["pesos"]            # compatibilidade
_CACHE_TEND: Dict[Any, pd.DataFrame] = {}


def _lg(p: float) -> float:
    p = min(max(float(p), 0.02), 0.98)
    return math.log(p / (1 - p))


def _total(d: pd.DataFrame, met: str) -> pd.Series:
    cols = METRICAS[met]["cols"]
    if cols is None:
        t = pd.to_numeric(d.get("game_time_min", pd.Series(dtype=float, index=d.index)), errors="coerce")
        if len(t) and t.isna().all() and "gamelength" in d.columns:
            t = pd.to_numeric(d["gamelength"], errors="coerce") / 60.0
        return t
    return pd.to_numeric(d[cols[0]], errors="coerce") + pd.to_numeric(d[cols[1]], errors="coerce")


def com_tendencia(team_games: pd.DataFrame, met: str) -> pd.DataFrame:
    """team_games + 'tend_<met>': quanto o ADVERSARIO de cada jogo puxava o total acima da liga,
    ANTES do jogo (seus 10 anteriores menos os ultimos 200 da liga). Kills reaproveita a coluna
    de cruzamento_estatistica_core."""
    col = f"tend_{met}"
    if met == "kills" and not E_DOTA:         # no Dota a medicao usou a conta geral, abaixo
        from app_core.cruzamento_estatistica_core import com_tendencia_dos_adversarios
        d = com_tendencia_dos_adversarios(team_games)
        if "tend_adv_kills" in d.columns and col not in d.columns:
            d = d.assign(**{col: d["tend_adv_kills"]})
        return d
    if col in team_games.columns:
        return team_games
    chave = (id(team_games), len(team_games), met)
    if chave in _CACHE_TEND:
        return _CACHE_TEND[chave]
    try:
        d = team_games.copy()
        d["_d"] = pd.to_datetime(d["date"], errors="coerce")
        d["_tot"] = _total(d, met)
        d = d.sort_values(["_d", "gameid"])
        d["_pt"] = d.groupby("team")["_tot"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        d["_pl"] = d.groupby("league")["_tot"].transform(lambda x: x.shift(1).rolling(200, min_periods=30).mean())
        adv = d[["gameid", "team", "_pt"]].drop_duplicates(["gameid", "team"]).rename(
            columns={"team": "opponent", "_pt": "_pa"})
        d = d.merge(adv, on=["gameid", "opponent"], how="left")
        d[col] = (d["_pa"] - d["_pl"]).fillna(0.0)
        d = d.drop(columns=["_d", "_tot", "_pt", "_pl", "_pa"])
    except (KeyError, TypeError, ValueError):
        return team_games
    if len(_CACHE_TEND) > 8:
        _CACHE_TEND.clear()
    _CACHE_TEND[chave] = d
    return d


def _jogos(tg: pd.DataFrame, time: str, so_mapa1: bool) -> pd.DataFrame:
    d = tg[tg["team"].astype(str) == str(time)]
    if so_mapa1 and "map_number" in d.columns:
        d = d[pd.to_numeric(d["map_number"], errors="coerce").fillna(1).astype(int) == 1]
    return d.assign(_d=pd.to_datetime(d["date"], errors="coerce")).dropna(subset=["_d"]).sort_values("_d")


def _camp(d: pd.DataFrame, met: str) -> pd.Series:
    """Os jogos do time no CAMPEONATO atual (liga + split + ano do ultimo jogo)."""
    if d.empty:
        return pd.Series(dtype=float)
    u = d.iloc[-1]
    c = d[(d["league"].astype(str) == str(u["league"])) & (d["split"].astype(str) == str(u.get("split")))
          & (d["_d"].dt.year == u["_d"].year)]
    return _total(c, met).dropna()


def sinais(team_games: pd.DataFrame, time_a: str, time_b: str, linha: float,
           so_mapa1: bool = False, metrica: str = "kills") -> Optional[Dict[str, Any]]:
    """As medidas do metodo dele para a linha L, por time e somadas. None = pouco jogo. Pura."""
    if team_games is None or team_games.empty or metrica not in METRICAS:
        return None
    cfg = METRICAS[metrica]
    tg = com_tendencia(team_games, metrica)
    col_t = f"tend_{metrica}"
    da, db = _jogos(tg, time_a, so_mapa1), _jogos(tg, time_b, so_mapa1)
    if len(da) < MINIMO_POR_TIME or len(db) < MINIMO_POR_TIME:
        return None
    L = float(linha)

    def _time(d):
        tot = _total(d, metrica).dropna()
        tend = pd.to_numeric(d.get(col_t, pd.Series(0.0, index=d.index)), errors="coerce") \
            .reindex(tot.index).fillna(0.0)
        u10 = tot.tail(10)
        t10 = tend.reindex(u10.index).fillna(0.0)
        c = _camp(d, metrica)
        ult = d.iloc[-1]
        return {"media": float(u10.mean()), "media_aj": float((u10 - cfg["ajuste"] * t10).mean()),
                "adv": float(t10.mean()), "n10": int(len(u10)),
                "janelas": {n: (int((tot.tail(n) > L).sum()), int(len(tot.tail(n)))) for n in JANELAS},
                "ano": (int((tot > L).sum()), int(len(tot))),
                "camp": (int((c > L).sum()), int(len(c))),
                "camp_rotulo": " ".join(str(x) for x in (ult["league"], ult.get("split"))
                                        if x is not None and str(x) not in ("", "nan"))}
    a, b = _time(da), _time(db)
    if not (a["n10"] and b["n10"]):
        return None
    media = (a["media"] + b["media"]) / 2
    media_aj = (a["media_aj"] + b["media_aj"]) / 2
    h = {n: (a["janelas"][n][0] + b["janelas"][n][0]) / max(1, a["janelas"][n][1] + b["janelas"][n][1])
         for n in JANELAS}
    camp = None
    if a["camp"][1] >= MINIMO_CAMPEONATO and b["camp"][1] >= MINIMO_CAMPEONATO:
        camp = (a["camp"][0] / a["camp"][1] + b["camp"][0] / b["camp"][1]) / 2     # % MAIS, media dos dois
    out = {"linha": L, "mapa1": bool(so_mapa1), "metrica": metrica, "a": a, "b": b, "media": media,
           "media_aj": media_aj, "h": h, "camp_mais": camp}
    if camp is not None:
        b0, wm, wa, w5, w10, w20, wc = cfg["pesos"]
        z = (b0 + wm * (media - L) + wa * (media_aj - L) + w5 * _lg(h[5]) + w10 * _lg(h[10])
             + w20 * _lg(h[20]) + wc * _lg(camp))
        out["p_over"] = 1.0 / (1.0 + math.exp(-z))
    return out


def passam(sn: Dict[str, Any], lado: str) -> List[Optional[bool]]:
    """[Medias, Adversario, Historico, Stats] a favor do lado ('over'/'under'); None = sem dado."""
    mais = str(lado).lower() == "over"
    L = sn["linha"]
    p1 = sn["media"] > L if mais else sn["media"] < L
    p2 = sn["media_aj"] > L if mais else sn["media_aj"] < L
    hs = [sn["h"][n] if mais else 1 - sn["h"][n] for n in JANELAS]
    p3 = all(x >= PASSA_HIST for x in hs)
    c = sn.get("camp_mais")
    p4 = None if c is None else ((1 - c) if mais else c) <= TETO_OUTRO_LADO
    return [p1, p2, p3, p4]


def chance(sn: Dict[str, Any], lado: str) -> Optional[float]:
    p = sn.get("p_over")
    if p is None:
        return None
    return p if str(lado).lower() == "over" else 1.0 - p


def resumo_html(sn: Optional[Dict[str, Any]], lado: str, rotulo: str = "", odd: Any = None) -> str:
    """'✓ ✓ ✗ ✓ 3/4 · justa 1,64' em HTML curto (para celulas de tabela)."""
    if not sn:
        return '<span class="q">pouco jogo</span>'
    ps = passam(sn, lado)
    marcas = "".join('<span title="{}" style="color:{}">{}</span>'.format(
        n, "var(--boa)" if v else ("var(--fraco)" if v is None else "var(--ruim)"),
        "✓" if v else ("·" if v is None else "✗")) for n, v in zip(NOMES, ps))
    k = sum(1 for v in ps if v)
    tot = sum(1 for v in ps if v is not None)
    p = chance(sn, lado)
    justa = "" if not p else ' · justa {:.2f}'.format(1 / p).replace(".", ",")
    try:
        if p and odd and float(odd) > 1:
            v = (p * float(odd) - 1) * 100
            justa += ' <span style="color:{};font-weight:700">{}</span>'.format(
                "var(--boa)" if v > 0 else "var(--ruim)", f"{v:+.1f}%".replace(".", ","))
    except (TypeError, ValueError):
        pass
    cor = "var(--boa)" if k >= 3 else ("var(--pin)" if k == 2 else "var(--fraco)")
    fraco = 'style="color:var(--fraco);font-size:11px"'
    return '<span style="white-space:nowrap">{}<b>{}</b> <b style="color:{}">{}/{}</b><span {}>{}</span></span>'.format(
        f'<span {fraco}>{rotulo}</span>' if rotulo else "", marcas, cor, k, tot, fraco, justa)



# ---- KILLS POR TIME e HANDICAP DE KILLS (29/10/2026) ---------------------------------------
# Os mesmos 4 passos, com a serie certa (scratchpad metodo_time.py, cego 2026, pesos de 2025):
#   kills de X:      X = kills que X FEZ; Y = kills que Y CEDEU. Ajuste: o quanto os adversarios
#                    enfrentados cediam acima da liga (peso 0,5).
#   handicap de X h: "cobre" = saldo de X maior que -h; X = saldo de X; Y = -saldo de Y. Ajuste:
#                    o saldo medio dos adversarios enfrentados (peso 0,5).
# Ganho da fair sobre so o historico: kills por time +0,011; handicap +0,011. Com 3 a favor,
# linhas perto da media: kills por time 55-60%, handicap 62-68% (4 de 4 quase nao acontece).
# (so LoL: no Dota kills por time e handicap ainda nao foram medidos -- ficam com os numeros do
#  LoL apenas para o modulo carregar; nenhuma tela do Dota os usa.)
METRICAS["kills_time"] = {"cols": None, "ajuste": 0.5, "unidade": "kills",
                          "pesos": (0.0906, 0.0836, 0.0206, 0.0581, -0.2789, 0.3634, 0.3431),
                          "bateu": {"over": {0: 49, 1: 52, 2: 52, 3: 60}, "under": {0: 47, 1: 47, 2: 51, 3: 55}}}
METRICAS["handicap"] = {"cols": None, "ajuste": 0.5, "unidade": "kills",
                        "pesos": (0.1225, -0.1242, 0.1278, 0.0860, 0.1236, 0.3745, 0.3939),
                        "bateu": {"over": {0: 49, 1: 53, 2: 56, 3: 68}, "under": {0: 43, 1: 48, 2: 49, 3: 62}}}
_CACHE_TIME: Dict[Any, pd.DataFrame] = {}


def _com_tend_time(team_games: pd.DataFrame) -> pd.DataFrame:
    """+ 't_ced' (o adversario do jogo cedia quantas kills acima da liga) e 't_sal' (o saldo
    medio do adversario), ambos ANTES do jogo."""
    if "t_ced" in team_games.columns:
        return team_games
    chave = (id(team_games), len(team_games))
    if chave in _CACHE_TIME:
        return _CACHE_TIME[chave]
    d = team_games.copy()
    d["_d"] = pd.to_datetime(d["date"], errors="coerce")
    d["_kf"] = pd.to_numeric(d["kills_for"], errors="coerce")
    d["_ka"] = pd.to_numeric(d["kills_against"], errors="coerce")
    d["_sal"] = d["_kf"] - d["_ka"]
    d = d.sort_values(["_d", "gameid"])
    for col, nome in (("_ka", "ced"), ("_sal", "sal")):
        d[f"_p_{nome}"] = d.groupby("team")[col].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    d["_l_ced"] = d.groupby("league")["_ka"].transform(lambda x: x.shift(1).rolling(200, min_periods=30).mean())
    adv = d[["gameid", "team", "_p_ced", "_p_sal"]].drop_duplicates(["gameid", "team"]).rename(
        columns={"team": "opponent", "_p_ced": "_a_ced", "_p_sal": "_a_sal"})
    d = d.merge(adv, on=["gameid", "opponent"], how="left")
    d["t_ced"] = (d["_a_ced"] - d["_l_ced"]).fillna(0.0)
    d["t_sal"] = d["_a_sal"].fillna(0.0)
    d = d.drop(columns=[c for c in d.columns if c.startswith("_") and c != "_d"]).drop(columns=["_d"])
    _CACHE_TIME.clear()
    _CACHE_TIME[chave] = d
    return d


def sinais_time(team_games: pd.DataFrame, time_x: str, time_y: str, linha: float, mercado: str,
                so_mapa1: bool = False) -> Optional[Dict[str, Any]]:
    """Kills de X (mercado 'kills_time', linha L) ou handicap de X (mercado 'handicap', linha h).
    Mesmo formato de sinais(); 'over' = Mais (kills) / COBRE (handicap). Pura."""
    if team_games is None or team_games.empty or mercado not in ("kills_time", "handicap"):
        return None
    cfg = METRICAS[mercado]
    tg = _com_tend_time(team_games)
    dx, dy = _jogos(tg, time_x, so_mapa1), _jogos(tg, time_y, so_mapa1)
    if len(dx) < MINIMO_POR_TIME or len(dy) < MINIMO_POR_TIME:
        return None
    w = cfg["ajuste"]
    kf = lambda d: pd.to_numeric(d["kills_for"], errors="coerce")
    ka = lambda d: pd.to_numeric(d["kills_against"], errors="coerce")
    if mercado == "kills_time":
        L = float(linha)
        sx, ajx = kf(dx), kf(dx) - w * dx["t_ced"]
        sy, ajy = ka(dy), ka(dy) - w * dy["t_ced"]
    else:
        L = -float(linha)                         # cobre h  <=>  saldo > -h
        sx = kf(dx) - ka(dx)
        ajx = sx + w * dx["t_sal"]
        sy = -(kf(dy) - ka(dy))
        ajy = sy - w * dy["t_sal"]

    def _time(d, serie, aj, rot):
        tot = serie.dropna()
        u10 = tot.tail(10)
        ult = d.iloc[-1]
        c = d[(d["league"].astype(str) == str(ult["league"])) & (d["split"].astype(str) == str(ult.get("split")))
              & (d["_d"].dt.year == ult["_d"].year)]
        cs = serie.reindex(c.index).dropna()
        return {"media": float(u10.mean()), "media_aj": float(aj.reindex(u10.index).mean()),
                # positivo = o numero do time esta INFLADO pelos adversarios (cediam mais / eram fracos)
                "adv": float((u10 - aj.reindex(u10.index)).mean() / (w if w else 1)), "n10": int(len(u10)),
                "janelas": {n: (int((tot.tail(n) > L).sum()), int(len(tot.tail(n)))) for n in JANELAS},
                "ano": (int((tot > L).sum()), int(len(tot))), "camp": (int((cs > L).sum()), int(len(cs))),
                "camp_rotulo": " ".join(str(x) for x in (ult["league"], ult.get("split"))
                                        if x is not None and str(x) not in ("", "nan")), "rotulo": rot}
    if mercado == "kills_time":
        a = _time(dx, sx, ajx, f"{time_x} faz")
        b = _time(dy, sy, ajy, f"{time_y} cede")
    else:
        a = _time(dx, sx, ajx, f"saldo de {time_x}")
        b = _time(dy, sy, ajy, f"{time_y} (saldo invertido)")
    media, media_aj = (a["media"] + b["media"]) / 2, (a["media_aj"] + b["media_aj"]) / 2
    h = {n: (a["janelas"][n][0] + b["janelas"][n][0]) / max(1, a["janelas"][n][1] + b["janelas"][n][1])
         for n in JANELAS}
    camp = None
    if a["camp"][1] >= MINIMO_CAMPEONATO and b["camp"][1] >= MINIMO_CAMPEONATO:
        camp = (a["camp"][0] / a["camp"][1] + b["camp"][0] / b["camp"][1]) / 2
    out = {"linha": L, "mapa1": bool(so_mapa1), "metrica": mercado, "a": a, "b": b, "media": media,
           "media_aj": media_aj, "h": h, "camp_mais": camp, "linha_casa": float(linha)}
    if mercado == "handicap":
        out["adv_palavras"] = ("fracos", "fortes")
    def _n(v, sinal=False):
        v = 0.0 if abs(v) < 1e-9 else float(v)
        t = f"{v:g}".replace(".", ",")
        return ("+" + t) if (sinal and v > 0) else t
    if mercado == "kills_time":
        out["titulos"] = {"over": f"{time_x}: MAIS {_n(linha)}", "under": f"{time_x}: MENOS {_n(linha)}"}
    else:
        out["titulos"] = {"over": f"{time_x} {_n(linha, True)} (cobre)",
                          "under": f"{time_y} {_n(-linha, True)} (cobre)"}
    if camp is not None:
        b0, wm, wa, w5, w10, w20, wc = cfg["pesos"]
        z = (b0 + wm * (media - L) + wa * (media_aj - L) + w5 * _lg(h[5]) + w10 * _lg(h[10])
             + w20 * _lg(h[20]) + wc * _lg(camp))
        out["p_over"] = 1.0 / (1.0 + math.exp(-z))
    return out
