# -*- coding: utf-8 -*-
"""Adaptador Dota -> schema OracleElixir-like (P1 do redesign, 2026-07-08).

Converte data.csv (linhas-time do OpenDota) no formato de CSV que o app do LoL
consome (team rows, side Blue/Red), para que os modulos app_core rodem IGUAIS
em cima de dados de Dota. Mapa de mercados DECIDIDO pelo usuario:
  barons      <- roshans      (funcao de jogo equivalente)
  inhibitors  <- barracks
  dragons     <- vazio (NaN) ate ele mandar tirar da UI
  towers      <- towers | kills <- teamkills | gamelength <- duration (s)
  side        <- radiant=Blue / dire=Red (a vantagem Radiant vira "blue side")

Regras de qualidade:
- SO tier professional + premium (universo do produto; excluded fica no motor WHR).
  PREMIUM E O TI: ate 16/08/2026 o filtro pegava so "professional" e o
  International inteiro ficava fora do modelo. Ver PRO_TIERS em eval_whr_dota.
- Partidas sem objetivos baixados: towers/barons/inhibitors = NaN (NUNCA 0).
- game = numero do mapa na serie, inferido por (dia, par de times) via start_time.

Uso: python dota_adapter.py  -> escreve CSV_DOTA/2026_Dota_match_data_oe_like.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval_whr_dota import is_pro_tier, load_matches  # noqa: E402

OUT_DIR = HERE / "CSV_DOTA"


def build_oe_like() -> pd.DataFrame:
    # Guarda anti-corrida (2026-07-10): se a esteira estiver escrevendo o
    # data.csv no exato momento da leitura, a base vem truncada (visto: CSV
    # do app saiu pela metade). Rele ate a contagem estabilizar, ficando
    # sempre com a MAIOR leitura. (A escrita do sync agora e atomica; isto
    # e cinto-e-suspensorio.)
    import time as _t
    m = load_matches()
    for _ in range(6):
        _t.sleep(2.0)
        m2 = load_matches()
        n_prev, n_new = m["match_id"].nunique(), m2["match_id"].nunique()
        if n_new > n_prev:
            m = m2
        elif n_new == n_prev:
            break
    m = m[is_pro_tier(m["tier"])].copy()
    m = m.sort_values(["ord_key", "match_id"]).reset_index(drop=True)

    # numero do mapa na serie (mesma inferencia validada nos estudos M1/M2)
    m["date_key"] = m["dt"].dt.date
    m["pair"] = [tuple(sorted(t)) for t in zip(m["tid"], m["oid"])]
    m["game"] = m.groupby(["date_key", "pair"]).cumcount() + 1

    has_obj = m["has_objectives"].astype(str).str.strip().str.lower().isin(["true", "1"])

    def obj(col: str) -> pd.Series:
        vals = pd.to_numeric(m[col], errors="coerce")
        return vals.where(has_obj, np.nan)

    rows = []
    for persp in ("radiant", "dire"):
        is_r = persp == "radiant"
        team_col, opp_col = ("teamname", "opp_teamname") if True else (None, None)
        r = pd.DataFrame({
            "gameid": "DOTA_" + m["match_id"].astype(str),
            "date": m["dt"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "league": m["league"].astype(str),
            "year": m["year"].astype(int),
            "split": "",
            "playoffs": 0,
            "patch": "",
            "game": m["game"].astype(int),
            "side": "Blue" if is_r else "Red",
            "participantid": 100 if is_r else 200,
            "position": "team",
            "datacompleteness": np.where(has_obj, "complete", "partial"),
            "playername": np.nan,
            "champion": np.nan,
            "teamname": (m["teamname"] if is_r else m["opp_teamname"]).astype(str),
            "teamid": (m["tid"] if is_r else m["oid"]).astype(int),
            "result": (m["win"] if is_r else 1 - m["win"]).astype(int),
            "teamkills": pd.to_numeric(m["teamkills" if is_r else "oppkills"], errors="coerce"),
            "teamdeaths": pd.to_numeric(m["oppkills" if is_r else "teamkills"], errors="coerce"),
            "kills": pd.to_numeric(m["teamkills" if is_r else "oppkills"], errors="coerce"),
            "deaths": pd.to_numeric(m["oppkills" if is_r else "teamkills"], errors="coerce"),
            "dragons": np.nan,   # decisao (a): vazio ate ele mandar tirar
            "barons": obj("roshans" if is_r else "opp_roshans"),
            "towers": obj("towers" if is_r else "opp_towers"),
            "opp_towers": obj("opp_towers" if is_r else "towers"),
            "inhibitors": obj("barracks" if is_r else "opp_barracks"),
            "opp_inhibitors": obj("opp_barracks" if is_r else "barracks"),
            # Megacreeps: nasce quando um lado derruba TODAS as 6 rax do
            # adversario. 1 = este time causou megas; NaN quando objetivos
            # nao baixados (nunca 0 falso).
            "megacreeps": (obj("barracks" if is_r else "opp_barracks") >= 6).astype(float).where(
                obj("barracks" if is_r else "opp_barracks").notna()),
            "gamelength": pd.to_numeric(m["duration"], errors="coerce"),
            # First X do Dota (extracao 2026-07-11): 1/0 por time, NaN nos
            # jogos ainda nao re-passados. firstbaron = First ROSHAN.
            "firstblood": (pd.to_numeric(m.get("first_blood"), errors="coerce")
                           if is_r else 1.0 - pd.to_numeric(m.get("first_blood"), errors="coerce")),
            "firsttower": (pd.to_numeric(m.get("first_tower"), errors="coerce")
                           if is_r else 1.0 - pd.to_numeric(m.get("first_tower"), errors="coerce")),
            "firstbaron": (pd.to_numeric(m.get("first_roshan"), errors="coerce")
                           if is_r else 1.0 - pd.to_numeric(m.get("first_roshan"), errors="coerce")),
            "firstdragon": np.nan,
        })
        rows.append(r)

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["date", "gameid", "side"]).reset_index(drop=True)
    return out


def main() -> int:
    df = build_oe_like()
    OUT_DIR.mkdir(exist_ok=True)
    # Mesmo layout do LoL: ano corrente em CSV_DIARIO, anteriores em
    # CSV_HISTORICO — e o seletor de times do app mostra so a season atual.
    diario = HERE / "CSV_DIARIO"
    historico = HERE / "CSV_HISTORICO"
    diario.mkdir(exist_ok=True)
    historico.mkdir(exist_ok=True)
    import os as _os
    cur_year = int(df["year"].max())
    for year, sub in df.groupby("year"):
        folder = diario if int(year) == cur_year else historico
        target = folder / f"{int(year)}_Dota_match_data_oe_like.csv"
        # Escrita ATOMICA: o app/retreino pode estar lendo este CSV agora
        # (corrida real 2026-07-11: retrain dele leu arquivo pela metade
        # durante um regen e caiu com rc=1).
        tmp = target.with_suffix(".csv.tmp")
        sub.to_csv(tmp, index=False)
        _os.replace(tmp, target)
        print(f"OK: {target}  linhas={len(sub)} partidas={sub['gameid'].nunique()} "
              f"times={sub['teamname'].nunique()}")
    tmp_all = OUT_DIR / "all_years_oe_like.csv.tmp"
    df.to_csv(tmp_all, index=False)
    _os.replace(tmp_all, OUT_DIR / "all_years_oe_like.csv")
    with_obj = df["towers"].notna().sum() // 2
    print(f"TOTAL: linhas={len(df)} partidas={df['gameid'].nunique()} "
          f"com_objetivos={with_obj} ligas={df['league'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
