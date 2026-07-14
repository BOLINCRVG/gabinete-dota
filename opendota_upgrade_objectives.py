# -*- coding: utf-8 -*-
"""Upgrade de objetivos: fetch completo (/matches/{id}) SO para partidas tier
professional que entraram pelo indice (source_mode=opendota_list) e portanto
tem torres/roshan/barracks zerados.

Idempotente e resumivel: cada rodada pega o que ainda falta, respeita orcamento
de tempo e para sozinha. Rodar quantas vezes precisar (a cota da API e ~2k/dia).

Uso: python opendota_upgrade_objectives.py [--budget-seconds 280] [--max-matches 400]
"""
from __future__ import annotations

import argparse
import time
from typing import List

import pandas as pd

from dota_core import csv_path
from opendota_client import get_match
from opendota_sync import _merge_rows, csv_health_report, load_existing_csv, match_to_rows

from eval_whr_dota import load_tiers


def pending_match_ids(df: pd.DataFrame) -> List[int]:
    """Fila do refetch, MAIS RECENTES PRIMEIRO (2026-07-11: a Recent Form dos
    jogos do dia precisa dos objetivos antes do historico de 2025).

    Entram: (a) partidas so-indice (sem objetivos); (b) partidas com objetivos
    baixados ANTES da extracao dos "primeiros" (first_blood ausente) — a
    re-passada preenche o First X com a mesma fila.
    """
    tiers = load_tiers()
    m = df.drop_duplicates("match_id").copy()
    m["league_id"] = pd.to_numeric(m["league_id"], errors="coerce").fillna(0).astype(int)
    m["tier"] = m["league_id"].map(tiers).fillna("sem_tier")
    prof = m["tier"] == "professional"
    so_indice = m.get("source_mode").astype(str) == "opendota_list"
    has_obj = m.get("has_objectives").astype(str).str.strip().str.lower().isin(["true", "1"])
    if "firsts_pass" in m.columns:
        firsts_missing = pd.to_numeric(m["firsts_pass"], errors="coerce").isna()
    else:
        firsts_missing = pd.Series(True, index=m.index)
    mask = prof & (so_indice | (has_obj & firsts_missing))
    sub = m.loc[mask, ["match_id", "date"]].copy()
    sub["match_id"] = pd.to_numeric(sub["match_id"], errors="coerce")
    sub = sub.dropna(subset=["match_id"])
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce", utc=True)
    sub = sub.sort_values("date", ascending=False)
    return sub["match_id"].astype(int).tolist()


def run(budget_seconds: int, max_matches: int) -> dict:
    start = time.monotonic()
    target = csv_path()
    existing = load_existing_csv(target)
    pending = pending_match_ids(existing)
    print(f"pendentes (professional, so-indice): {len(pending)}", flush=True)

    new_rows: List[dict] = []
    done = 0
    failed = 0
    for mid in pending:
        if done >= max_matches or (time.monotonic() - start) > budget_seconds:
            break
        try:
            payload = get_match(mid)
            rows = match_to_rows(payload)
            if rows:
                new_rows.extend(rows)
                done += 1
        except Exception:
            failed += 1
            if failed >= 10:
                print("muitas falhas seguidas — provavelmente cota; parando.", flush=True)
                break
            time.sleep(30)

    df = _merge_rows(existing, new_rows, target)
    report = {
        "upgraded": done,
        "failed": failed,
        "still_pending": max(0, len(pending) - done),
        "elapsed_seconds": round(time.monotonic() - start, 1),
    }
    report.update({k: csv_health_report(target, df=df)[k]
                   for k in ("total_matches", "matches_missing_objectives")})
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-seconds", type=int, default=280)
    ap.add_argument("--max-matches", type=int, default=400)
    args = ap.parse_args()
    report = run(args.budget_seconds, args.max_matches)
    for k, v in report.items():
        print(f"{k}: {v}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
