# -*- coding: utf-8 -*-
"""Sync diario automatico do Gabinete Dota (agendado no Task Scheduler).

Faz, nesta ordem, com orcamento total de ~25min e log em sync_daily.log:
1. sync_recent_pro_matches — pega as partidas novas do indice;
2. backfill de buracos (paginado) — caso o PC tenha ficado dias desligado;
3. rodadas de upgrade de objetivos ate a cota reclamar (para sozinho).

Zero acao do usuario. Se a cota acabar, amanha continua de onde parou.
"""
from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
LOG = HERE / "sync_daily.log"


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main() -> int:
    start = time.monotonic()
    log("=== daily_sync inicio ===")
    try:
        from opendota_sync import sync_recent_pro_matches
        rep = sync_recent_pro_matches(limit=200, professional_only=False, max_seconds=240)
        log(f"[1] recentes: +{rep.get('fetched_matches', 0)} partidas "
            f"(csv={rep.get('total_matches')}, idade={rep.get('latest_match_age_hours')}h)")
    except Exception:
        log("[1] recentes FALHOU:\n" + traceback.format_exc())

    try:
        from opendota_backfill_holes import fill_missing
        rep = fill_missing(budget_seconds=0, full_fetch=False)  # so indice, rapido
        log(f"[2] buracos: {rep.get('missing_found', 0)} faltantes ingeridos do indice")
    except Exception:
        log("[2] buracos FALHOU:\n" + traceback.format_exc())

    try:
        # Campanha P3 (2026-07-09, "faça"): o fetch completo traz objetivos +
        # players/draft (sidecar raw_matches.jsonl). Rodadas até ~50min ou
        # cota — o refetch histórico (~18k partidas de 2025) anda sozinho.
        from opendota_upgrade_objectives import run as upgrade_run
        rounds = 0
        while (time.monotonic() - start) < 3000 and rounds < 10:
            rounds += 1
            rep = upgrade_run(budget_seconds=280, max_matches=300)
            log(f"[3] refetch rodada {rounds}: +{rep.get('upgraded', 0)} "
                f"(pendentes={rep.get('still_pending')})")
            if rep.get("upgraded", 0) == 0 or rep.get("still_pending", 0) == 0:
                break
            time.sleep(15)
    except Exception:
        log("[3] refetch FALHOU:\n" + traceback.format_exc())

    try:
        # [4] Regenera os CSVs que o APP le (CSV_DIARIO/CSV_HISTORICO) a partir
        # do data.csv atualizado — sem isso o app fica defasado mesmo com o
        # dado bruto novo (gap apontado pelo usuario em 2026-07-09).
        from dota_adapter import main as adapter_main
        adapter_main()
        log("[4] adapter: CSVs do app regenerados")
    except Exception:
        log("[4] adapter FALHOU:\n" + traceback.format_exc())

    log(f"=== daily_sync fim ({time.monotonic()-start:.0f}s) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
