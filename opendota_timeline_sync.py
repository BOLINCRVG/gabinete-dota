# -*- coding: utf-8 -*-
"""Coleta a SERIE POR MINUTO das partidas (ouro/XP/CS/kills aos 10/15/20/25).

Por que existe (16/08/2026): a aba "Durante o mapa (estado -> vitoria)" NUNCA
funcionou no Dota — nao ha um unico modelo em live_models/. No LoL essa camada
vale +9.87 a +29.28pp. O motivo era duplo: o orquestrador de treino nao existe no
clone, e o CSV do Dota nao tinha as fotos por minuto que o modelo consome.

A API TEM o dado: /matches/{id} devolve radiant_gold_adv e radiant_xp_adv com um
ponto por minuto, e cada jogador traz gold_t / xp_t / lh_t completos. Este script
extrai isso e grava com os MESMOS nomes de coluna que o mlcore/live_ml espera
(goldat10, opp_goldat10, ...), para o codigo do ao vivo rodar sem alteracao.

Honestidade sobre o que NAO da para extrair: assistencias por minuto nao existem
na resposta da API. assistsat* fica 0 dos dois lados — o modelo ve a feature
constante e simplesmente nao a usa. Kills sai do kills_log (tem o instante de
cada abate); as mortes de um lado sao os abates do outro.

Idempotente e retomavel, como o opendota_upgrade_objectives: cada rodada pega o
que falta, respeita orcamento de tempo e para sozinha. Sao ~23 mil partidas e a
cota e ~2k/dia, entao isto roda em varias rodadas.

Uso: python opendota_timeline_sync.py [--budget-seconds 280] [--max-matches 400]
"""
from __future__ import annotations

import sys

# Console cp1252 no Windows derruba print com acento (ver find_silent_failures).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

import argparse
import time
from pathlib import Path

import pandas as pd

from opendota_client import _request

HERE = Path(__file__).resolve().parent
STORE = HERE / "timeline_snapshots.csv"
MINUTOS = (10, 15, 20, 25)
CAMPOS = ("goldat", "xpat", "csat", "killsat", "deathsat", "assistsat")


# DRAFT (25/08/2026): vem DE GRACA na mesma chamada que ja fazemos. A literatura
# e consistente em que draft sozinho preve 63-76% e que SINERGIA e COUNTER valem
# mais que taxa de vitoria individual do heroi. O modelo do Dota nao tem heroi
# nenhum hoje (o adaptador grava champion vazio), entao e uma dimensao inteira
# faltando. Colher agora evita refazer 23 mil chamadas depois.
COLS_DRAFT = ("herois_radiant", "herois_dire", "bans_radiant", "bans_dire")


def colunas() -> list[str]:
    out = ["match_id"]
    for t in MINUTOS:
        for c in CAMPOS:
            out += [f"{c}{t}", f"opp_{c}{t}"]
    return out + list(COLS_DRAFT)


def _soma_ate(series_por_jogador: list[list], t: int) -> float:
    """Valor acumulado no minuto t, somado no time. A serie da API e por minuto."""
    tot = 0.0
    for s in series_por_jogador:
        if isinstance(s, list) and len(s) > t:
            try:
                tot += float(s[t])
            except (TypeError, ValueError):
                pass
    return tot


def _kills_ate(jogadores: list[dict], t: int) -> float:
    """Abates do time ate o minuto t, pelo instante de cada um no kills_log."""
    n = 0
    limite = t * 60
    for p in jogadores:
        for k in (p.get("kills_log") or []):
            try:
                if float(k.get("time", 1e9)) <= limite:
                    n += 1
            except (TypeError, ValueError, AttributeError):
                pass
    return float(n)


def extrair(d: dict) -> dict | None:
    """Uma linha de snapshots a partir da resposta de /matches/{id}."""
    jog = d.get("players") or []
    if not jog:
        return None
    rad = [p for p in jog if p.get("isRadiant") is True]
    dire = [p for p in jog if p.get("isRadiant") is False]
    if not rad or not dire:
        return None
    dur = float(d.get("duration") or 0)
    linha: dict = {"match_id": int(d.get("match_id") or 0)}

    # Draft final vem de players[].hero_id (SEMPRE presente). picks_bans traz a
    # ordem e os banimentos, mas falta em modo sem capitaes / partida nao parseada
    # — por isso o draft nao depende dele.
    linha["herois_radiant"] = ",".join(str(p.get("hero_id") or 0) for p in rad)
    linha["herois_dire"] = ",".join(str(p.get("hero_id") or 0) for p in dire)
    pb = d.get("picks_bans") or []
    if isinstance(pb, list) and pb:
        for lado, chave in ((0, "bans_radiant"), (1, "bans_dire")):
            linha[chave] = ",".join(
                str(x.get("hero_id") or 0) for x in pb
                if isinstance(x, dict) and not x.get("is_pick") and x.get("team") == lado)
    else:
        linha["bans_radiant"] = ""
        linha["bans_dire"] = ""
    for t in MINUTOS:
        # partida que acabou antes do minuto t nao tem foto valida ali
        if dur < t * 60:
            for c in CAMPOS:
                linha[f"{c}{t}"] = None
                linha[f"opp_{c}{t}"] = None
            continue
        g_r = _soma_ate([p.get("gold_t") for p in rad], t)
        g_d = _soma_ate([p.get("gold_t") for p in dire], t)
        x_r = _soma_ate([p.get("xp_t") for p in rad], t)
        x_d = _soma_ate([p.get("xp_t") for p in dire], t)
        c_r = _soma_ate([p.get("lh_t") for p in rad], t)
        c_d = _soma_ate([p.get("lh_t") for p in dire], t)
        k_r = _kills_ate(rad, t)
        k_d = _kills_ate(dire, t)
        linha.update({
            f"goldat{t}": g_r, f"opp_goldat{t}": g_d,
            f"xpat{t}": x_r, f"opp_xpat{t}": x_d,
            f"csat{t}": c_r, f"opp_csat{t}": c_d,
            f"killsat{t}": k_r, f"opp_killsat{t}": k_d,
            # mortes de um lado = abates do outro
            f"deathsat{t}": k_d, f"opp_deathsat{t}": k_r,
            # assistencias por minuto NAO existem na API: constante, o modelo ignora
            f"assistsat{t}": 0.0, f"opp_assistsat{t}": 0.0,
        })
    return linha


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-seconds", type=float, default=280.0)
    ap.add_argument("--max-matches", type=int, default=400)
    args = ap.parse_args()

    import os
    os.environ.setdefault("DOTA_IDENTITY_API", "0")   # nao precisa aqui, poupa tempo
    from eval_whr_dota import is_pro_tier, load_matches

    m = load_matches()
    m = m[is_pro_tier(m["tier"])]
    alvo = [int(x) for x in m["match_id"].dropna().unique()]

    feitos: set[int] = set()
    if STORE.exists():
        try:
            _st = pd.read_csv(STORE, low_memory=False)
            # linha colhida ANTES do draft entrar (25/08) precisa ser refeita: so
            # conta como pronta quem ja tem os herois. Sem isto, as ~2 mil
            # primeiras ficariam para sempre sem essa dimensao.
            if "herois_radiant" in _st.columns:
                _ok = _st["herois_radiant"].astype(str).str.strip().ne("").fillna(False)
                _st = _st[_ok]
            else:
                _st = _st.iloc[0:0]
            feitos = {int(x) for x in _st["match_id"].dropna()}
        except (OSError, ValueError, KeyError):
            feitos = set()
    faltam = [x for x in alvo if x not in feitos]
    print(f"universo pro: {len(alvo)} | ja coletadas: {len(feitos)} | faltam: {len(faltam)}")
    if not faltam:
        print("nada a fazer — serie por minuto completa.")
        return 0

    t0 = time.monotonic()
    novas: list[dict] = []
    erros = 0
    for i, mid in enumerate(faltam[: max(1, int(args.max_matches))]):
        if time.monotonic() - t0 > float(args.budget_seconds):
            print(f"orcamento de {args.budget_seconds:.0f}s esgotado — parando (retomavel)")
            break
        try:
            d = _request(f"matches/{mid}")
        except Exception:
            d = None
            erros += 1
        if not isinstance(d, dict):
            erros += 1
            continue
        linha = extrair(d)
        if linha:
            novas.append(linha)
        if (i + 1) % 25 == 0:
            print(f"  {i+1} pedidas | {len(novas)} extraidas | {erros} sem resposta", flush=True)

    if novas:
        df = pd.DataFrame(novas).reindex(columns=colunas())
        if STORE.exists():
            df = pd.concat([pd.read_csv(STORE, low_memory=False), df], ignore_index=True)
            df = df.drop_duplicates("match_id", keep="last")   # a nova (com draft) vence
        tmp = STORE.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, STORE)
        print(f"gravadas {len(novas)} | total no arquivo: {len(df)} | restam ~{len(faltam)-len(novas)}")
    else:
        print(f"nenhuma extraida ({erros} sem resposta) — a cota pode ter acabado; tente depois")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
