# -*- coding: utf-8 -*-
"""CONTRATO DE DADOS DO DOTA: o CSV que o adaptador gerou pode treinar o modelo? (22/10/2026)

Item "expectativas das features num esquema" da regua das 28. O LoL tem
tools/run_phase1_data_contract.py; o Dota nao tinha nada entre o dota_adapter.py
e o treino: um CSV com jogo de 3 linhas, dois vencedores ou data quebrada
treinava calado.

Confere o que, medido em 22/10/2026 no CSV real (5.288 jogos), vale sempre:
  - colunas obrigatorias presentes;
  - cada jogo tem exatamente 2 linhas de time, um lado de cada, e 1 vencedor;
  - data legivel e nao no futuro (folga de 1 dia por fuso);
  - time e liga em branco em no maximo 2% das linhas (medido: 0%);
  - duracao entre 5 min e 3 h (medido: 7 a 112 min);
  - o numero de jogos nao caiu mais de 10% desde o ultimo contrato aprovado
    (CSV que encolhe e sync que quebrou, nao jogos que sumiram).

Roda no robo (dados_diarios_dota.yml) depois do adaptador e ANTES do treino:
sai com 1 se falhar, o job para e o modelo publicado continua o antigo.
Grava evidence/contrato_dados_dota.json.

FONTE (PDF aberto em 22/10/2026): Breck et al., "The ML Test Score", item de
dados "Feature expectations are captured in a schema".
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf

Uso: python tools/contrato_dados_dota.py [--csv caminho]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CSV_PADRAO = RAIZ / "CSV_DIARIO" / "2026_Dota_match_data_oe_like.csv"
EVIDENCIA = RAIZ / "evidence" / "contrato_dados_dota.json"
OBRIGATORIAS = ("gameid", "date", "league", "side", "position", "teamname", "result", "gamelength")
MAX_BRANCO = 0.02
DURACAO = (300, 10800)
QUEDA_MAX = 0.10


def conferir(df: pd.DataFrame, jogos_antes: int | None = None, agora: datetime | None = None) -> dict:
    """Pura: devolve {'ok', 'jogos', 'falhas': [...]}."""
    falhas = []
    faltam = [c for c in OBRIGATORIAS if c not in df.columns]
    if faltam:
        return {"ok": False, "jogos": 0, "falhas": [f"colunas faltando: {faltam}"]}
    t = df[df["position"] == "team"]
    jogos = int(t["gameid"].nunique())
    if jogos == 0:
        return {"ok": False, "jogos": 0, "falhas": ["nenhuma linha de time"]}
    por_jogo = t.groupby("gameid")
    n = por_jogo.size()
    if (n != 2).any():
        falhas.append(f"{int((n != 2).sum())} jogo(s) sem exatamente 2 linhas de time")
    lados = por_jogo["side"].nunique()
    if (lados != 2).any():
        falhas.append(f"{int((lados != 2).sum())} jogo(s) com o mesmo lado repetido")
    res = pd.to_numeric(t["result"], errors="coerce")
    if res.isna().any() or not res.isin([0, 1]).all():
        falhas.append("result fora de {0,1}")
    venc = res.groupby(t["gameid"]).sum()
    if (venc != 1).any():
        falhas.append(f"{int((venc != 1).sum())} jogo(s) sem exatamente 1 vencedor")
    datas = pd.to_datetime(t["date"], errors="coerce")
    if datas.isna().any():
        falhas.append(f"{int(datas.isna().sum())} data(s) ilegivel(is)")
    limite = (agora or datetime.now()) + timedelta(days=1)
    if (datas > limite).any():
        falhas.append(f"{int((datas > limite).sum())} data(s) no futuro")
    for col in ("teamname", "league"):
        taxa = float(t[col].isna().mean() + (t[col].astype(str).str.strip() == "").mean())
        if taxa > MAX_BRANCO:
            falhas.append(f"{col} em branco em {taxa:.1%} das linhas (max {MAX_BRANCO:.0%})")
    dur = pd.to_numeric(t["gamelength"], errors="coerce")
    fora = int((dur.isna() | (dur < DURACAO[0]) | (dur > DURACAO[1])).sum())
    if fora:
        falhas.append(f"{fora} linha(s) com duracao fora de {DURACAO} s")
    if jogos_antes and jogos < jogos_antes * (1 - QUEDA_MAX):
        falhas.append(f"jogos cairam de {jogos_antes} para {jogos} (> {QUEDA_MAX:.0%})")
    return {"ok": not falhas, "jogos": jogos, "falhas": falhas}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV_PADRAO))
    a = ap.parse_args()
    antes = None
    try:
        antes = json.loads(EVIDENCIA.read_text(encoding="utf-8")).get("jogos_aprovados")
    except (OSError, ValueError):
        pass
    res = conferir(pd.read_csv(a.csv, low_memory=False), antes)
    saida = {"quando": datetime.now().isoformat(timespec="seconds"), **res,
             "jogos_aprovados": res["jogos"] if res["ok"] else antes}
    EVIDENCIA.parent.mkdir(exist_ok=True)
    EVIDENCIA.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    for f in res["falhas"]:
        print("FALHA:", f)
    print(f"[contrato_dados_dota] {res['jogos']} jogos -> {'APROVADO' if res['ok'] else 'REPROVADO'}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
