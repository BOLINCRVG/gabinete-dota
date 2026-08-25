"""Backfill de buracos historicos no data.csv via paginacao real do /proMatches.

O sync_recent_pro_matches so le a primeira pagina do /proMatches (as ~100 mais
recentes) — se o sync ficou semanas sem rodar, os jogos daquele periodo nunca
entram (buracos de abr/mai/jun-2026). Este modulo pagina PARA TRAS com
less_than_match_id ate o cutoff e ingere o que falta.

Modos:
  discover  -> so conta o que falta, por mes (nao escreve nada).
  fill      -> fetch completo (/matches/{id}) de cada faltante, com orcamento de
               tempo; o que nao couber no orcamento entra como linha "partial"
               construida do proprio item da listagem (kills/duracao/resultado
               ok; torres/roshan/barracks zerados + flag missing_objectives,
               mesma convencao do sync normal para objetivos ausentes).

Uso (do proprio diretorio):
  python opendota_backfill_holes.py discover
  python opendota_backfill_holes.py fill --budget-seconds 280
"""
from __future__ import annotations

import sys

# Console cp1252 no Windows derruba print com acento — e este script e rodado
# A MAO no PC dele (o robo roda em Linux/UTF-8 e nunca ve o problema).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

import argparse
import time
from typing import Dict, List, Optional

import pandas as pd

from opendota_client import _request, get_match
from opendota_sync import (
    _merge_rows,
    _seen_match_ids,
    csv_health_report,
    load_existing_csv,
    match_to_rows,
)
from dota_core import csv_path

CUTOFF_UTC = "2026-01-01"
MAX_PAGES = 220


def _pro_matches_page(less_than_match_id: Optional[int]) -> List[dict]:
    params: Dict[str, int] = {}
    if less_than_match_id:
        params["less_than_match_id"] = int(less_than_match_id)
    # o retry interno do _request desiste rapido em 429; aqui o backoff e
    # paciente porque paginar historico nao tem pressa
    for attempt in range(6):
        try:
            data = _request("proMatches", params=params, read_timeout=35)
            return data if isinstance(data, list) else []
        except Exception as exc:
            if "429" in str(exc) and attempt < 5:
                time.sleep(65)
                continue
            raise
    return []


def discover_missing(existing: Optional[pd.DataFrame] = None, verbose: bool = True) -> List[dict]:
    """Pagina o /proMatches ate CUTOFF_UTC e retorna itens ausentes do CSV."""
    df = existing if existing is not None else load_existing_csv()
    seen = _seen_match_ids(df)
    cutoff_ts = int(pd.Timestamp(CUTOFF_UTC, tz="UTC").timestamp())

    missing: List[dict] = []
    less_than: Optional[int] = None
    pages = 0
    scanned = 0
    while pages < MAX_PAGES:
        page = _pro_matches_page(less_than)
        if not page:
            break
        pages += 1
        scanned += len(page)
        oldest_ts = None
        for item in page:
            mid = int((item or {}).get("match_id") or 0)
            ts = int((item or {}).get("start_time") or 0)
            if mid <= 0:
                continue
            if less_than is None or mid < less_than:
                less_than = mid
            if ts > 0:
                oldest_ts = ts if oldest_ts is None else min(oldest_ts, ts)
            if ts < cutoff_ts:
                continue
            if mid not in seen:
                missing.append(item)
        if oldest_ts is not None and oldest_ts < cutoff_ts:
            break
        time.sleep(3.0)  # cortesia com o rate limit (60/min)

    if verbose:
        print(f"paginas={pages} itens_varridos={scanned} faltantes={len(missing)}")
        if missing:
            months = pd.Series(
                [pd.to_datetime(int(m["start_time"]), unit="s", utc=True).strftime("%Y-%m") for m in missing]
            ).value_counts().sort_index()
            for month, count in months.items():
                print(f"  {month}: {count}")
    return missing


def _list_item_rows(item: dict) -> List[dict]:
    """Linhas a partir do item da listagem (sem objectives -> partial).

    source_mode=opendota_list marca que torres/roshan/barracks vieram zerados
    do indice — sao essas linhas que um upgrade posterior (fetch completo por
    liga T1/T2) deve substituir.
    """
    payload = dict(item)
    payload.setdefault("leagueid", payload.get("leagueid") or payload.get("league_id") or 0)
    rows = match_to_rows(payload)
    for row in rows:
        row["source_mode"] = "opendota_list"
    return rows


def fill_missing(budget_seconds: int = 280, full_fetch: bool = True) -> dict:
    start = time.monotonic()
    target = csv_path()
    existing = load_existing_csv(target)
    missing = discover_missing(existing, verbose=True)
    # mais antigos primeiro: preenche o buraco na ordem cronologica
    missing.sort(key=lambda m: int(m.get("start_time") or 0))

    new_rows: List[dict] = []
    fetched_full = 0
    from_list = 0
    failed: List[int] = []
    for item in missing:
        mid = int(item.get("match_id") or 0)
        if full_fetch and (time.monotonic() - start) < budget_seconds:
            try:
                payload = get_match(mid)
                rows = match_to_rows(payload)
                if rows:
                    new_rows.extend(rows)
                    fetched_full += 1
                    continue
            except Exception:
                failed.append(mid)
        rows = _list_item_rows(item)
        if rows:
            new_rows.extend(rows)
            from_list += 1

    df = _merge_rows(existing, new_rows, target)
    report = {
        "mode": "backfill_holes",
        "missing_found": len(missing),
        "fetched_full": fetched_full,
        "from_list_partial": from_list,
        "failed_full_fetch": len(failed),
        "new_rows": len(new_rows),
        "elapsed_seconds": round(time.monotonic() - start, 1),
    }
    report.update(csv_health_report(target, df=df))
    return report


def main() -> None:
    global CUTOFF_UTC, MAX_PAGES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["discover", "fill"])
    parser.add_argument("--budget-seconds", type=int, default=280)
    parser.add_argument("--no-full-fetch", action="store_true",
                        help="so ingere do item da listagem (rapido, tudo partial)")
    parser.add_argument("--cutoff", default=CUTOFF_UTC,
                        help="data-limite p/ tras (ex.: 2025-01-01 = backfill da season passada)")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES)
    args = parser.parse_args()
    CUTOFF_UTC = str(args.cutoff)
    MAX_PAGES = int(args.max_pages)

    if args.mode == "discover":
        discover_missing()
        return
    report = fill_missing(budget_seconds=args.budget_seconds, full_fetch=not args.no_full_fetch)
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
