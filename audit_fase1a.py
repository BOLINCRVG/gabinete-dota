"""Fase 1a — auditoria do data.csv pos-backfill: tiers, qualidade, times.

Junta o CSV com /leagues (1 chamada, cacheada) para mapear league_id -> tier
OpenDota (premium / professional / excluded / amateur) e responde:
  1. quantas partidas por tier (o filtro de qualidade do modelo nasce daqui);
  2. top ligas por volume dentro de cada tier;
  3. saude por tier: team_id ausente, nome placeholder, kills/duração zerados;
  4. tamanho da amostra "modelavel" (tier bom + times identificados).
"""
from __future__ import annotations

import pandas as pd

from opendota_client import get_leagues

pd.set_option("display.width", 160)

df = pd.read_csv("data.csv", low_memory=False)
m = df.drop_duplicates("match_id").copy()
m["dt"] = pd.to_datetime(m["date"], errors="coerce", utc=True)

leagues = pd.DataFrame(get_leagues())
leagues = leagues.rename(columns={"leagueid": "league_id"})[["league_id", "tier", "name"]]
m["league_id"] = pd.to_numeric(m["league_id"], errors="coerce").fillna(0).astype(int)
m = m.merge(leagues, on="league_id", how="left", suffixes=("", "_od"))
m["tier"] = m["tier"].fillna("sem_tier")

print("=== partidas por tier (OpenDota) ===")
print(m["tier"].value_counts().to_string())

print("\n=== top 12 ligas por tier ===")
for tier in ["premium", "professional", "excluded", "amateur", "sem_tier"]:
    sub = m[m["tier"] == tier]
    if sub.empty:
        continue
    print(f"\n-- {tier} ({len(sub)} partidas, {sub['league'].nunique()} ligas) --")
    print(sub["league"].value_counts().head(12).to_string())

print("\n=== saude por tier ===")
m["team_id_num"] = pd.to_numeric(m["team_id"], errors="coerce").fillna(0)
m["opp_id_num"] = pd.to_numeric(m["opp_team_id"], errors="coerce").fillna(0)
m["sem_team_id"] = (m["team_id_num"] <= 0) | (m["opp_id_num"] <= 0)
low = m["teamname"].astype(str).str.strip().str.lower()
low_opp = m["opp_teamname"].astype(str).str.strip().str.lower()
m["nome_placeholder"] = low.isin(["radiant", "dire", ""]) | low_opp.isin(["radiant", "dire", ""])
m["kills_zerados"] = (
    pd.to_numeric(m["teamkills"], errors="coerce").fillna(0)
    + pd.to_numeric(m["oppkills"], errors="coerce").fillna(0)
) <= 0
m["dur_zerada"] = pd.to_numeric(m["duration"], errors="coerce").fillna(0) <= 0
health = m.groupby("tier")[["sem_team_id", "nome_placeholder", "kills_zerados", "dur_zerada"]].mean()
print((health * 100).round(1).to_string())

print("\n=== amostra modelavel ===")
ok = m[~m["sem_team_id"] & ~m["nome_placeholder"] & ~m["kills_zerados"] & ~m["dur_zerada"]]
print("todas com times identificados e stats sas:", len(ok))
print(ok["tier"].value_counts().to_string())
print("\ntimes distintos (identificados):", ok["teamname"].nunique())
print("partidas/mes na amostra modelavel:")
print(ok.set_index("dt").resample("ME")["match_id"].count().to_string())
