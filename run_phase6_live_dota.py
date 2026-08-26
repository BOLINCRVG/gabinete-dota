# -*- coding: utf-8 -*-
"""FASE 6 no Dota: estado do mapa -> vitoria. Treina e MEDE antes de salvar.

A aba "Durante o mapa" nunca funcionou no Dota por dois motivos; este resolve o
segundo e ultimo: nao havia orquestrador de treino (o clone nao tem tools/, e o
app tentava chamar tools/retrain_live_models_all.py todo dia, falhava e gravava
"ultimo retreino: hoje" mesmo assim).

Porque NAO copiei o gate do LoL: aquele tem 319 linhas com microsegmento e Platt
por segmento, maquinaria que o Dota nao possui. Aqui uso a MESMA API
(mlcore.live_ml) com o minimo que o Dota tem.

O GATE (o ponto do script): o modelo de estado so vale se bater a PREVISAO
PRE-JOGO sozinha. Saber que o time favorito esta 5k de ouro na frente aos 15
minutos tem que valer mais que so saber que ele era favorito. Se nao bater, NAO
SALVA — melhor aba vazia que aba mentindo.

Uso: python run_phase6_live_dota.py [--save-models-dir live_models] [--min-n 300]
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

import argparse
import json
import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mlcore.artifact import MLArtifact  # noqa: E402
from mlcore.live_ml import SNAPSHOT_MINUTES, StateToWinModel, build_snapshot_features  # noqa: E402
from mlcore.tiers import tier_for_league  # noqa: E402

GATE = 0.005          # ganho de logloss sobre o pre-jogo para valer a pena


def _ll(ps, ys) -> float:
    tot = 0.0
    for p, y in zip(ps, ys):
        p = min(max(float(p), 1e-9), 1 - 1e-9)
        tot += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return tot / max(1, len(ps))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="ml_artifact.json")
    ap.add_argument("--save-models-dir", default="live_models")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--min-n", type=int, default=300,
                    help="amostra minima por minuto para sequer tentar")
    ap.add_argument("--output", default="FASE6_LIVE_DOTA.json")
    args = ap.parse_args()

    linhas = []
    for pasta in ("CSV_HISTORICO", "CSV_DIARIO"):
        for c in sorted((HERE / pasta).glob("*_oe_like.csv")):
            linhas.append(pd.read_csv(c, low_memory=False))
    if not linhas:
        print("nenhum CSV — rode dota_adapter.py antes")
        return 1
    tr = pd.concat(linhas, ignore_index=True)
    tr["date"] = pd.to_datetime(tr["date"], errors="coerce")
    tr = tr.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # P(vitoria) PRE-JOGO por (gameid, side) — a referencia que o estado precisa bater
    art = MLArtifact.load_json(str(HERE / args.artifact))
    model = art.model
    p_pre: dict[tuple[str, str], float] = {}
    for gid, par in tr.groupby("gameid"):
        b = par[par["side"] == "Blue"]
        r = par[par["side"] == "Red"]
        if b.empty or r.empty:
            continue
        nb, nr = str(b.iloc[0]["teamname"]), str(r.iloc[0]["teamname"])
        lg = str(b.iloc[0].get("league") or "OTHER")
        t = str(tier_for_league(lg))
        try:
            p = float(model.p_blue(nb, nr, blue_league=lg, red_league=lg,
                                   blue_tier=t, red_tier=t))
        except (TypeError, ValueError):
            continue
        p_pre[(str(gid), "Blue")] = p
        p_pre[(str(gid), "Red")] = 1.0 - p

    print(f"linhas-time: {len(tr)} | partidas com previsao pre-jogo: {len(p_pre)//2}\n")
    print(f"{'minuto':>7}{'n treino':>10}{'n val':>8}{'pre-jogo':>11}{'estado':>10}{'ganho':>10}  veredito")

    salvos, resumo = 0, []
    for minuto in SNAPSHOT_MINUTES:
        snaps = build_snapshot_features(tr, minute=minuto,
                                        p_map_pre_by_gameid_side=p_pre,
                                        require_complete=True)
        if len(snaps) < int(args.min_n):
            print(f"{minuto:>7}{len(snaps):>10}{'-':>8}{'-':>11}{'-':>10}{'-':>10}  amostra pequena")
            resumo.append({"minuto": minuto, "n": len(snaps), "veredito": "amostra pequena"})
            continue
        corte = int(len(snaps) * (1.0 - float(args.val_frac)))
        treino, val = snaps[:corte], snaps[corte:]

        m = StateToWinModel(minute=minuto).fit(treino, val)
        ys = [int(s.y) for s in val]
        p_estado = list(m.predict_proba(val))
        p_base = [float(s.p_pre) for s in val]
        ll_base, ll_est = _ll(p_base, ys), _ll(p_estado, ys)
        ganho = ll_base - ll_est
        ok = ganho > GATE
        print(f"{minuto:>7}{len(treino):>10}{len(val):>8}{ll_base:>11.4f}{ll_est:>10.4f}"
              f"{ganho:>+10.4f}  {'SALVA' if ok else 'REJEITADO'}")
        resumo.append({"minuto": minuto, "n_treino": len(treino), "n_val": len(val),
                       "ll_pre_jogo": round(ll_base, 4), "ll_estado": round(ll_est, 4),
                       "ganho": round(ganho, 4), "salvo": bool(ok)})
        if ok:
            destino = HERE / str(args.save_models_dir) / f"minute_{minuto:02d}"
            destino.parent.mkdir(parents=True, exist_ok=True)
            m.save(destino)
            salvos += 1

    print(f"\nmodelos salvos: {salvos} de {len(SNAPSHOT_MINUTES)}")
    if not salvos:
        print("NENHUM passou o gate — a aba segue sem modelo, o que e o certo:")
        print("  melhor aba vazia que aba mentindo.")
    (HERE / str(args.output)).write_text(
        json.dumps({"gate": GATE, "minutos": resumo, "salvos": salvos},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[salvo] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
