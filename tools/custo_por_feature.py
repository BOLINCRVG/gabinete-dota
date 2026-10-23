# -*- coding: utf-8 -*-
"""QUANTO CUSTA CADA CAMADA DO MODELO? (23/10/2026)

Item "nenhuma feature custa demais" da regua das 28 (LoL e Dota em 0: "custo/
latencia por feature nunca foi medido"). O ganho de cada camada ja e medido
pelos gates; faltava o outro lado da conta: o que cada uma custa para servir.

Mede, nos jogos reais mais recentes do CSV:
  - TEMPO por previsao de cada camada, isolada:
      time       -> model.p_blue (rating do time)
      jogadores  -> compute_delta_and_confidence (lineup dos 10 jogadores)
      etapas     -> cada etapa da cadeia servida (mlcore.replay_pre_jogo.cadeia):
                    microssegmento, Platt do time, fusao time+jogadores
  - TAMANHO de cada bloco nos arquivos do modelo (bytes de JSON): o que cada
    camada pesa para carregar.
"Custa demais" aqui = passar de 5 ms por previsao (a tela faz uma por jogo; o
pior caso, a agenda do dia, tem ~40 jogos -> 0,2 s) ou de 20 MB no disco.
Esses tetos sao folga de ordem de grandeza sobre o medido, nao calibracao.

Grava evidence/custo_por_feature.json. Roda na suite (test_custo_por_feature),
que o robo roda todo dia, e falha se uma camada passar do teto. Arquivo
identico nos dois repositorios (o que nao existir num deles e pulado).

FONTE (PDF aberto em 22/10/2026): Breck et al., "The ML Test Score", item de
dados "No feature's cost is too much" ("not only a function of the added
inference latency and RAM usage, but also more upstream data dependencies").
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf

Uso: python tools/custo_por_feature.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

TETO_MS = 5.0
TETO_MB = 20.0
N_JOGOS = 400
EVIDENCIA = RAIZ / "evidence" / "custo_por_feature.json"
ARQUIVOS = ("ml_artifact.json", "players_artifact.json", "players_pscore.json", "params_fitted.json")


def _ms_por_chamada(fn, itens, repeticoes=3) -> float:
    melhor = float("inf")
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        for x in itens:
            fn(x)
        melhor = min(melhor, time.perf_counter() - t0)
    return 1000.0 * melhor / max(1, len(itens))


def tamanhos(raiz: Path) -> dict:
    """{arquivo: {bloco: MB}} dos blocos de topo (e de meta) de cada arquivo."""
    saida = {}
    for nome in ARQUIVOS:
        p = raiz / nome
        if not p.exists():
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        blocos = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "meta" and isinstance(v, dict):
                    for k2, v2 in v.items():
                        blocos[f"meta.{k2}"] = len(json.dumps(v2)) / 1e6
                else:
                    blocos[k] = len(json.dumps(v)) / 1e6
        saida[nome] = {k: round(v, 4) for k, v in sorted(blocos.items(), key=lambda kv: -kv[1])[:8]}
        saida[nome]["_total"] = round(p.stat().st_size / 1e6, 4)
    return saida


def _jogos_recentes(raiz: Path):
    import pandas as pd
    from mlcore.io_oracle import load_oracle_elixir_games
    csvs = sorted((raiz / "CSV_DIARIO").glob("*.csv"))
    if not csvs:
        return None, None
    g = load_oracle_elixir_games(str(csvs[-1]), keep_playoffs=None)
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    return g.sort_values("date").tail(N_JOGOS), csvs[-1]


def tempos(raiz: Path) -> dict:
    from mlcore.artifact import MLArtifact
    from mlcore.tiers import tier_for_league
    saida = {}
    jogos, csv = _jogos_recentes(raiz)
    if jogos is None or jogos.empty:
        return saida
    modelo = MLArtifact.load_json(str(raiz / "ml_artifact.json")).model
    pares = []
    for r in jogos.itertuples(index=False):
        lg = str(getattr(r, "league", "") or "OTHER")
        t = str(tier_for_league(lg))
        pares.append((str(r.blue), str(r.red), lg, t))
    saida["time"] = _ms_por_chamada(
        lambda x: modelo.p_blue(x[0], x[1], blue_league=x[2], red_league=x[2],
                                blue_tier=x[3], red_tier=x[3]), pares)

    try:
        from mlcore.players_layer import (PlayersArtifact, compute_delta_and_confidence,
                                          lineup_for_game, load_match_player_rows)
        pa = raiz / "players_artifact.json"
        art_p = PlayersArtifact.load_json(str(pa)) if pa.exists() else None
        if art_p is not None:
            ids = set(jogos["gameid"].astype(str))
            linhas = load_match_player_rows([str(csv)], keep_playoffs=None, gameids=ids)
            escal = []
            for r in jogos.itertuples(index=False):
                b, rd = lineup_for_game(linhas, str(r.gameid))
                if b and rd:
                    escal.append((b, rd, str(getattr(r, "league", "") or "")))
            if escal:
                saida["jogadores"] = _ms_por_chamada(
                    lambda x: compute_delta_and_confidence(art_p, blue_lineup=x[0], red_lineup=x[1],
                                                           match_league=x[2]), escal)
    except (ImportError, OSError, ValueError, KeyError):
        pass

    try:
        from mlcore.calibration import fit_platt_from_raw_probs
        from mlcore.replay_pre_jogo import Amostra, cadeia
        params = raiz / "params_fitted.json"
        tabela = {}
        if params.exists():
            tabela = (json.loads(params.read_text(encoding="utf-8")).get("microseg_calibration") or {}).get("table") or {}
        amostras = []
        for i, r in enumerate(jogos.itertuples(index=False)):
            amostras.append(Amostra(
                gameid=str(r.gameid), league=str(getattr(r, "league", "") or ""), split="",
                side="Blue", bo="", phase="", month="", y=int(r.blue_win),
                p_team=0.3 + 0.4 * (i % 10) / 10, p_players=0.5, n_team=50.0,
                n_players=20.0, delta_eff=0.0))
        platt = fit_platt_from_raw_probs([a.y for a in amostras], [a.p_team for a in amostras])
        nada = dict(usar_ms1=False, usar_platt_time=False, usar_fusao=False)
        base = _ms_por_chamada(lambda a: cadeia([a], tabela=tabela, platt_time=platt, **nada), amostras)
        for nome, liga in (("etapa_microssegmento", "usar_ms1"),
                           ("etapa_platt_time", "usar_platt_time"),
                           ("etapa_fusao", "usar_fusao")):
            com = dict(nada, **{liga: True})
            ms = _ms_por_chamada(lambda a: cadeia([a], tabela=tabela, platt_time=platt, **com), amostras)
            saida[nome] = max(0.0, ms - base)
    except (ImportError, OSError, ValueError, KeyError, TypeError):
        pass
    return {k: round(v, 4) for k, v in saida.items()}


def medir(raiz: Path = RAIZ) -> dict:
    t, s = tempos(raiz), tamanhos(raiz)
    caros = [f"{k}: {v} ms" for k, v in t.items() if v > TETO_MS]
    caros += [f"{a}: {b['_total']} MB" for a, b in s.items() if b["_total"] > TETO_MB]
    return {"quando": datetime.now().isoformat(timespec="seconds"),
            "teto_ms_por_previsao": TETO_MS, "teto_mb_por_arquivo": TETO_MB,
            "ms_por_previsao": t, "mb_por_bloco": s, "acima_do_teto": caros}


def main() -> int:
    res = medir()
    EVIDENCIA.parent.mkdir(exist_ok=True)
    EVIDENCIA.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ms por previsao, por camada:")
    for k, v in res["ms_por_previsao"].items():
        print(f"  {k:<24}{v:>9.4f}")
    for arq, blocos in res["mb_por_bloco"].items():
        print(f"{arq}: {blocos['_total']} MB (maiores: "
              + ", ".join(f"{k} {v}" for k, v in list(blocos.items())[:3]) + ")")
    print("ACIMA DO TETO: " + "; ".join(res["acima_do_teto"]) if res["acima_do_teto"]
          else "nenhuma camada acima do teto")
    return 1 if res["acima_do_teto"] else 0


if __name__ == "__main__":
    sys.exit(main())
