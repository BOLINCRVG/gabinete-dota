# -*- coding: utf-8 -*-
"""STATUS DA COLETA (24/11/2026): o robo diz o que aconteceu -- inclusive quando FALHA.

Ele: "EU QUERO TER certeza de que o que eu estou vendo na minha tela e a informacao mais FIEL
POSSIVEL [...] Se eu tiver que ficar de tempos em tempos verificando isso, vai acabar com a
confianca do modelo."

Achado que motivou: de 20 a 24/11 o robo do Dota falhou 5 vezes (cobranca, depois o contrato de
dados barrando um CSV com 2.698 jogos a menos, depois tempo esgotado) e o app nao sabia de nada --
so o robo que da certo gravava. Agora o ultimo passo do robo roda SEMPRE (if: always()) e grava
status_coleta_<esporte>.json; a faixa do topo do app (app_core/confianca_base_core.py) le.

Quando o contrato reprova por queda de jogos, compara o CSV novo com o publicado (git HEAD) por liga
e por time: a proxima falha ja chega com a causa. Arquivo identico nos repositorios.
Uso: python tools/status_coleta.py --esporte dota --resultado success|failure|cancelled
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

RAIZ = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "tools" else Path(__file__).resolve().parent
CSV_DIA = {"dota": "CSV_DIARIO/2026_Dota_match_data_oe_like.csv",
           "lol": "CSV_DIARIO/2026_LoL_esports_match_data_from_OraclesElixir.csv"}   # cs: sem contrato
CONTRATO = {"dota": "evidence/contrato_dados_dota.json"}


def _jogos_por(df: pd.DataFrame, col: str) -> pd.Series:
    t = df[df["position"] == "team"]
    return t.drop_duplicates(["gameid", col]).groupby(col)["gameid"].nunique()


def comparar(novo: pd.DataFrame, velho: pd.DataFrame, n: int = 12) -> dict:
    """Pura: onde os jogos sumiram (liga e time), do CSV publicado para o novo."""
    out = {"jogos_antes": int(velho.loc[velho["position"] == "team", "gameid"].nunique()),
           "jogos_agora": int(novo.loc[novo["position"] == "team", "gameid"].nunique())}
    for col in ("league", "teamname"):
        a, b = _jogos_por(velho, col), _jogos_por(novo, col)
        d = (b.reindex(a.index.union(b.index)).fillna(0) - a.reindex(a.index.union(b.index)).fillna(0))
        d = d[d < 0].sort_values().head(n)
        out["perdas_por_" + ("liga" if col == "league" else "time")] = {str(k): int(v) for k, v in d.items()}
    gv = set(velho.loc[velho["position"] == "team", "gameid"])
    gn = set(novo.loc[novo["position"] == "team", "gameid"])
    sumiram = sorted(gv - gn)
    out["jogos_que_sumiram"] = len(sumiram)
    out["exemplos"] = sumiram[:15]
    return out


def _csv_publicado(rel: str) -> pd.DataFrame | None:
    try:
        b = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=RAIZ, capture_output=True, timeout=120).stdout
        return pd.read_csv(io.BytesIO(b), low_memory=False) if b else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esporte", required=True)
    ap.add_argument("--resultado", required=True)
    a = ap.parse_args()
    res = str(a.resultado).strip().lower()
    doc = {"quando": datetime.now(timezone.utc).isoformat(timespec="seconds"), "resultado": res,
           "rodada": (f"{os.environ.get('GITHUB_SERVER_URL', '')}/{os.environ.get('GITHUB_REPOSITORY', '')}"
                      f"/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}") if os.environ.get("GITHUB_RUN_ID") else ""}
    motivo = ""
    arq_c = CONTRATO.get(a.esporte)
    if arq_c and (RAIZ / arq_c).exists():
        try:
            c = json.loads((RAIZ / arq_c).read_text(encoding="utf-8"))
            doc["contrato"] = {k: c.get(k) for k in ("ok", "jogos", "falhas", "jogos_aprovados")}
            if c.get("ok") is False:
                motivo = "contrato de dados reprovou: " + "; ".join(c.get("falhas") or [])
        except (OSError, ValueError):
            pass
    rel = CSV_DIA.get(a.esporte)
    if motivo and rel and (RAIZ / rel).exists():
        velho = _csv_publicado(rel)
        if velho is not None:
            try:
                doc["diagnostico"] = comparar(pd.read_csv(RAIZ / rel, low_memory=False), velho)
            except (OSError, ValueError, KeyError) as e:
                doc["diagnostico"] = {"erro": f"{type(e).__name__}: {e}"}
    if res == "cancelled" and not motivo:
        motivo = "rodada cortada (tempo esgotado ou cancelada) antes de gravar"
    elif res == "failure" and not motivo:
        motivo = "um passo do robo falhou antes de gravar (ver a rodada)"
    doc["motivo"] = motivo
    destino = RAIZ / f"status_coleta_{a.esporte}.json"
    velho_doc = {}
    if destino.exists():
        try:
            velho_doc = json.loads(destino.read_text(encoding="utf-8"))
        except ValueError:
            velho_doc = {}
    doc["ultimo_sucesso"] = doc["quando"] if res == "success" else velho_doc.get("ultimo_sucesso")
    destino.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[status_coleta] {a.esporte}: {res}" + (f" -- {motivo}" if motivo else ""))
    if doc.get("diagnostico"):
        print(json.dumps(doc["diagnostico"], ensure_ascii=False, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
