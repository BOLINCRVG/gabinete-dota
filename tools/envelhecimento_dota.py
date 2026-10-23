# -*- coding: utf-8 -*-
"""QUANTO CUSTA O MODELO DO DOTA ENVELHECER? (23/10/2026)

Item "impacto do envelhecimento conhecido" da regua das 28 (Dota 0: "a medicao
de 27/08 e do LoL e nao foi replicada no Dota").

Como mede. Treina o modelo do jeito do robo (train_offline, holdout de 30 dias,
SEM aplicar o holdout aos ratings), numa pasta temporaria: e o modelo como
ele estava no dia do corte. Depois percorre os jogos posteriores ao corte em
ordem de data, nos dois bracos, com a MESMA funcao de previsao:
  - CONGELADO: preve com o modelo do corte e nunca aprende;
  - EM DIA: preve, e so entao aprende o resultado (model.update) -- o que o
    passo "Aplicar os jogos recentes" e o incremental fazem no robo.
A diferenca de logloss, por semana desde o corte, e o preco de nao atualizar.
Probabilidade crua (sem Platt) nos dois bracos: compara o modelo, nao o
calibrador.

LIMITE: o update usa a liga do proprio jogo como liga de casa dos dois times
(o incremental de producao usa a liga domestica inferida). Igual nos dois
bracos, entao a comparacao e justa; o numero absoluto pode diferir um pouco.

Grava evidence/envelhecimento_dota.json. Roda a mao (treino ~15 s).

FONTE (PDF aberto em 22/10/2026): Breck et al., "The ML Test Score", item de
modelo "The impact of model staleness is known".
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf

Uso: python tools/envelhecimento_dota.py
"""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from mlcore.artifact import MLArtifact  # noqa: E402
from mlcore.io_oracle import load_oracle_elixir_games  # noqa: E402
from mlcore.tiers import tier_for_league  # noqa: E402

CSVS = ("CSV_HISTORICO/2025_Dota_match_data_oe_like.csv",
        "CSV_DIARIO/2026_Dota_match_data_oe_like.csv")
HOLDOUT = 30
EVIDENCIA = RAIZ / "evidence" / "envelhecimento_dota.json"


def _ll(p: float, y: int) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -math.log(p if y else 1 - p)


def comparar(modelo, jogos: pd.DataFrame, corte) -> pd.DataFrame:
    """Preve cada jogo pos-corte com o modelo congelado e com uma copia que
    aprende jogo a jogo. Devolve uma linha por jogo."""
    em_dia = copy.deepcopy(modelo)
    linhas = []
    for r in jogos.sort_values(["date", "gameid"]).itertuples(index=False):
        b, rd, y = str(r.blue), str(r.red), int(r.blue_win)
        lg = str(getattr(r, "league", "") or "OTHER")
        t = str(tier_for_league(lg))
        kw = dict(blue_league=lg, red_league=lg, blue_tier=t, red_tier=t)
        p_c = float(modelo.p_blue(b, rd, **kw))
        p_d = float(em_dia.p_blue(b, rd, **kw))
        linhas.append({"semana": int((r.date - corte).days // 7) + 1, "y": y,
                       "ll_congelado": _ll(p_c, y), "ll_em_dia": _ll(p_d, y)})
        em_dia.update(b, rd, y, **kw)
    return pd.DataFrame(linhas)


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "corte.json"
        cmd = [sys.executable, "-m", "mlcore.train_offline", "--out", str(out),
               "--holdout-days", str(HOLDOUT),
               # os irmaos do artefato tambem vao para a pasta temporaria
               "--players-out", str(Path(d) / "players.json"),
               "--pscore-out", str(Path(d) / "pscore.json")]
        for c in CSVS:
            cmd += ["--csv", c]
        r = subprocess.run(cmd, cwd=str(RAIZ), capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout[-1500:], r.stderr[-1500:])
            return 1
        art = MLArtifact.load_json(str(out))
    corte = pd.Timestamp(art.meta["cutoff_date"])
    g = pd.concat([load_oracle_elixir_games(str(RAIZ / c), keep_playoffs=None) for c in CSVS],
                  ignore_index=True, join="inner")
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = g.dropna(subset=["date"]).drop_duplicates("gameid", keep="last")
    g = g[g["date"] > corte]
    df = comparar(art.model, g, corte)
    sem = (df.groupby("semana")[["ll_congelado", "ll_em_dia"]].mean()
           .join(df.groupby("semana").size().rename("n")))
    sem["custo"] = sem["ll_congelado"] - sem["ll_em_dia"]
    res = {
        "quando": datetime.now().isoformat(timespec="seconds"),
        "corte": str(corte.date()), "jogos": int(len(df)),
        "logloss_congelado": round(float(df["ll_congelado"].mean()), 4),
        "logloss_em_dia": round(float(df["ll_em_dia"].mean()), 4),
        "custo_total": round(float((df["ll_congelado"] - df["ll_em_dia"]).mean()), 4),
        "por_semana": [{"semana": int(s), "n": int(x.n), "congelado": round(x.ll_congelado, 4),
                        "em_dia": round(x.ll_em_dia, 4), "custo": round(x.custo, 4)}
                       for s, x in sem.iterrows()],
    }
    EVIDENCIA.parent.mkdir(exist_ok=True)
    EVIDENCIA.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"corte {res['corte']}, {res['jogos']} jogos depois dele")
    print(f"logloss congelado {res['logloss_congelado']}  em dia {res['logloss_em_dia']}  "
          f"custo de envelhecer {res['custo_total']:+.4f}")
    for s in res["por_semana"]:
        print(f"  semana {s['semana']}: n={s['n']:4d}  custo {s['custo']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
