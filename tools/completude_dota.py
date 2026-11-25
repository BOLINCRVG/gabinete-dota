# -*- coding: utf-8 -*-
"""COMPLETUDE DA BASE DO DOTA (24/11/2026): toda serie que a Liquipedia da como TERMINADA nos ultimos 21
dias (agenda_dota_terminados.json, gravado por tools/agenda_dota.py) esta na base? As que nao estao vao
para evidence/completude_dota.json com o motivo, e a faixa do topo do app mostra.
Mesmo papel do tools/completude_lol.py. Regra dele: jogo nunca falta calado.
Uso: python tools/completude_dota.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
SAIDA = RAIZ / "evidence" / "completude_dota.json"
CSV = RAIZ / "CSV_DIARIO" / "2026_Dota_match_data_oe_like.csv"


def conferir(series: list, base: pd.DataFrame, resolver) -> list:
    """Pura: as series sem jogo na base (mesmos dois times, de 1 dia antes a 2 dias depois)."""
    t = base[base["position"] == "team"].copy()
    t["dt"] = pd.to_datetime(t["date"], errors="coerce", utc=True)
    jogos = [(g["dt"].min(), frozenset(str(x) for x in g["teamname"])) for _, g in t.groupby("gameid")]
    faltam = []
    for s in series:
        a, b = resolver(s["time_a"]) or "", resolver(s["time_b"]) or ""
        item = {"inicio": str(s["inicio_utc"]).replace(" ", "T").replace("+00", "+00:00"), "a": s["time_a"],
                "b": s["time_b"], "liga": s.get("torneio") or s.get("pagina"), "placar": s.get("placar", "")}
        if not a or not b:
            nao = [x for x, r in ((s["time_a"], a), (s["time_b"], b)) if not r]
            faltam.append({**item, "motivo": "time não reconhecido na base: " + ", ".join(nao)})
            continue
        ini = pd.Timestamp(item["inicio"])
        if not any(par == {a, b} and ini - pd.Timedelta(days=1) <= dt <= ini + pd.Timedelta(days=2) for dt, par in jogos):
            # 25/11/2026: as unicas series "faltando" eram justamente as SEM placar na Liquipedia (terminadas sem
            # resultado): nao houve jogo para coletar -- W.O. ou adiada. O motivo diz isso, nao culpa a coleta.
            faltam.append({**item, "motivo": "não está na base (a coleta da OpenDota ainda não trouxe)" if item["placar"]
                           else "a Liquipedia não tem placar desta série (provável W.O. ou adiada) — sem jogo para coletar"})
    return faltam


def main() -> int:
    try:
        series = json.loads((RAIZ / "agenda_dota_terminados.json").read_text(encoding="utf-8")).get("series") or []
    except (OSError, ValueError):
        print("[completude_dota] sem agenda_dota_terminados.json; nada a conferir")
        return 0
    base = pd.read_csv(CSV, usecols=["gameid", "date", "teamname", "position"], low_memory=False)
    times = sorted(set(base.loc[base["position"] == "team", "teamname"].astype(str)))
    from app_core.gaveta_jogos_core import resolver_padrao
    faltam = conferir(series, base, resolver_padrao(times, RAIZ))
    doc = {"quando": datetime.now(timezone.utc).isoformat(timespec="seconds"), "fonte": "Liquipedia:Matches",
           "janela_dias": 21, "series_conferidas": len(series), "faltando": faltam}
    SAIDA.parent.mkdir(exist_ok=True)
    SAIDA.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[completude_dota] {len(series)} series terminadas; faltam {len(faltam)} na base")
    for f in faltam[:8]:
        print("  -", f["inicio"][:10], f["a"], "x", f["b"], f"({f['liga']}, {f['placar']}):", f["motivo"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
