# -*- coding: utf-8 -*-
"""Fase 1c (fecho) — primeira NOTA honesta do Dota + teste da hipotese de momento.

1. Replay config* (w2=20, excluded ON, side ON), coletando p_raw prequencial de
   toda partida professional elegivel (ambos >=5 jogos) desde 2026-02-15.
2. NOTA: Platt fitado SO no TUNE (fev15..abr30), aplicado no BLIND (mai+) —
   mesma regua do LoL (_grade_from_logloss: random 0.6931->5.0, 0.55->9.5).
3. MOMENTO (hipotese do usuario: "time ruim ganhou do bom hoje, amanha ganha do
   outro ruim"): residual (y - p) de times vindo de jogo nos ultimos <=2 dias,
   por condicao (upset_win = venceu com p<0.40; upset_loss = perdeu com p>0.60;
   win/loss normais). Se o WHR ja absorve o momento, residual ~0; sinal positivo
   em upset_win = momento REAL alem do rating -> vira termo gateado.
"""
from __future__ import annotations

import math
import sys

# Console cp1252 no Windows derruba qualquer print com acento/emoji — e o estouro
# vem no fim, depois do trabalho todo feito. Ver tools/find_silent_failures.py.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass
from pathlib import Path

import pandas as pd

from dota_paths import ensure_lol_on_path
LOL_REPO = ensure_lol_on_path()

from mlcore.whr_core import WHRFitter  # noqa: E402
from mlcore.calibration import fit_platt_from_raw_probs  # noqa: E402
from eval_whr_dota import SideTracker, load_matches, MIN_GAMES_EVAL, TUNE_FROM, TUNE_TO, BLIND_FROM  # noqa: E402

W2_STAR = 20.0
RANDOM_LL = 0.6931471805599453


def grade(ll: float) -> float:
    return max(0.0, min(10.0, 5.0 + (RANDOM_LL - ll) / (RANDOM_LL - 0.55) * 4.5))


def main() -> int:
    m = load_matches()
    fitter = WHRFitter(w2_elo_per_day=W2_STAR)
    side = SideTracker()
    t_tune_from = pd.Timestamp(TUNE_FROM, tz="UTC")
    t_tune_to = pd.Timestamp(TUNE_TO, tz="UTC")
    t_blind = pd.Timestamp(BLIND_FROM, tz="UTC")

    preds = []          # (dt, p_raw, y)
    momentum_rows = []  # (cond, residual) por perspectiva de time
    last_event: dict[str, tuple[int, str]] = {}
    count = 0

    for r in m.itertuples(index=False):
        a, b = str(r.tid), str(r.oid)
        is_prof = r.tier == "professional"
        p = None
        ga = len(fitter.games.get(a, ()))
        gb = len(fitter.games.get(b, ()))
        eligible = is_prof and ga >= MIN_GAMES_EVAL and gb >= MIN_GAMES_EVAL
        if eligible:
            logit = fitter.rating_nat(a) - fitter.rating_nat(b) + side.offset_nat(int(r.day))
            p = 1.0 / (1.0 + math.exp(-logit))
            y = int(r.win)
            if r.dt >= t_tune_from:
                preds.append((r.dt, p, y))
                for team, p_team, y_team in ((a, p, y), (b, 1.0 - p, 1 - y)):
                    ev = last_event.get(team)
                    if ev and 1 <= (int(r.day) - ev[0]) <= 2:
                        momentum_rows.append((ev[1], y_team - p_team))
        # atualiza estado prequencial
        fitter.add_game(a, b, int(r.win), int(r.day))
        fitter.refresh_pair(a, b, iters=2)
        if is_prof:
            side.add(int(r.day), int(r.win))
            if p is not None:
                y = int(r.win)
                for team, p_team, y_team in ((a, p, y), (b, 1.0 - p, 1 - y)):
                    if y_team == 1:
                        cond = "upset_win" if p_team < 0.40 else "win"
                    else:
                        cond = "upset_loss" if p_team > 0.60 else "loss"
                    last_event[team] = (int(r.day), cond)
        count += 1
        if count % 500 == 0:
            fitter.full_sweeps(sweeps=1, iters=1)

    df = pd.DataFrame(preds, columns=["dt", "p", "y"])
    tune = df[(df["dt"] >= t_tune_from) & (df["dt"] <= t_tune_to)]
    blind = df[df["dt"] >= t_blind]
    print(f"preds: tune={len(tune)} blind={len(blind)}")

    cal = fit_platt_from_raw_probs(tune["y"].tolist(), tune["p"].tolist())
    for label, sub in (("RAW  ", blind), ):
        ll = -(sub["y"] * sub["p"].clip(1e-6, 1 - 1e-6).map(math.log)
               + (1 - sub["y"]) * (1 - sub["p"].clip(1e-6, 1 - 1e-6)).map(math.log)).mean()
        print(f"BLIND {label} ll={ll:.4f} nota={grade(ll):.2f}")
    pc = blind["p"].map(lambda v: float(cal.predict_proba(float(v)))).clip(1e-6, 1 - 1e-6)
    llc = -(blind["y"] * pc.map(math.log) + (1 - blind["y"]) * (1 - pc).map(math.log)).mean()
    acc = ((pc >= 0.5) == (blind["y"] == 1)).mean()
    # ECE 10 bins
    bins = pd.cut(pc, bins=[i / 10 for i in range(11)], include_lowest=True)
    ece = 0.0
    for _, sub in pd.DataFrame({"p": pc, "y": blind["y"], "b": bins}).groupby("b", observed=True):
        if len(sub):
            ece += (len(sub) / len(pc)) * abs(sub["p"].mean() - sub["y"].mean())
    print(f"BLIND PLATT ll={llc:.4f} nota={grade(llc):.2f} acc={acc*100:.1f}% ece={ece:.3f} "
          f"(platt a={cal.a:.3f} b={cal.b:.3f})")

    print("\n=== momento (residual y-p do time, jogo anterior <=2 dias) ===")
    mom = pd.DataFrame(momentum_rows, columns=["cond", "res"])
    for cond in ("upset_win", "win", "upset_loss", "loss"):
        sub = mom[mom["cond"] == cond]["res"]
        if len(sub) < 5:
            continue
        mu, se = sub.mean(), sub.std(ddof=1) / math.sqrt(len(sub))
        z = mu / se if se > 0 else 0.0
        print(f"  {cond:10} n={len(sub):<5} residual_medio={mu*100:+5.1f}pp  z={z:+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
