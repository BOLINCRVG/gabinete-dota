# -*- coding: utf-8 -*-
"""SOBRA SINAL DE PLAYERS NO DOTA? (teste barato antes de construir a camada)

Contexto: no LoL a camada de players mediu peso otimo 0.00 — o core ja tinha
absorvido o sinal. No Dota o core era mais fraco, mas HOJE (16/08/2026) ele foi
muito fortalecido: identidade corrigida + 419 partidas recuperadas, TI de 0.68
para 6.29. Entao a pergunta e outra: ainda sobra sinal, ou o core ja pegou?

NAO uso taxa de vitoria do jogador como proxy de forca — no LoL isso ja foi
flagrado como mentira (e a vitoria do TIME colada em cada jogador). Testo o que a
literatura afirma de forma direta e falsificavel:

  "com 1 jogador trocado, times rendem ~12.7 p.p. ABAIXO do esperado nos
   primeiros 8 jogos, fechando a diferenca la pelo 10o"

Se isso vale na base dele, entao CONTINUIDADE DE ELENCO e sinal que o modelo nao
tem — e vira feature pronta. Se nao vale, a frente fecha por medicao, como no LoL.

Metodo: para cada partida, quantos dos 5 titulares do time jogaram tambem na
partida ANTERIOR daquele time. Depois compara o residuo do modelo (resultado real
menos probabilidade prevista) entre elenco intacto e elenco mexido. Residuo
sistematicamente negativo em quem mexeu = o modelo esta superestimando esses
times = sinal aproveitavel.

RESULTADO (16/08/2026, n=43.189 observacoes time-jogo) -> SEM SINAL, frente fechada:
  0 trocados  n=35484  residuo +0.0020
  1 trocado   n= 2899  residuo -0.0086
  2 trocados  n= 1638  residuo -0.0065
  3+          n= 3168  residuo -0.0073
  mexido - intacto = -0.96 p.p. (erro padrao 0.57) -> z = -1.70
A DIRECAO bate com a literatura, mas o TAMANHO nao: 0.96 p.p. contra os ~12.7
reportados — treze vezes menor e indistinguivel de ruido. A curva temporal
tambem nao tem formato de efeito (a pior faixa e a de 8-10 jogos, nao a de 0-3).
Explicacao provavel: o WHR atualiza rating a cada jogo, entao ja absorve troca de
elenco sozinho — mesmo desfecho do LoL, onde a camada de players mediu peso 0.00.
Rodar de novo se o core mudar de forma relevante.

Uso: python medir_sinal_players.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
DOTA = Path(__file__).resolve().parent
sys.path.insert(0, str(DOTA))


def main() -> int:
    from eval_whr_dota import (canonical_team_names, is_pro_tier, load_matches,
                               roster_identity_map)
    from mlcore.artifact import MLArtifact
    from mlcore.tiers import tier_for_league

    m = load_matches()
    m = m[is_pro_tier(m["tier"])].sort_values("ord_key").reset_index(drop=True)
    canon = canonical_team_names(m)
    rep = roster_identity_map(m)

    def nome(i: int) -> str:
        i = int(i)
        return canon.get(rep.get(i, i), canon.get(i, str(i)))

    # elenco por (match_id, team_id)
    roster: dict[tuple[int, int], frozenset[int]] = {}
    with open(DOTA / "raw_matches.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            mid = int(d.get("match_id") or 0)
            pl = d.get("players") or []
            for tid_raw, is_rad in ((d.get("radiant_team_id"), True),
                                    (d.get("dire_team_id"), False)):
                try:
                    tid = int(tid_raw or 0)
                except (TypeError, ValueError):
                    continue
                if tid <= 0:
                    continue
                accs = {int(p["account_id"]) for p in pl
                        if p.get("account_id") and (p.get("isRadiant") is is_rad)}
                if accs:
                    roster[(mid, tid)] = frozenset(accs)

    art = MLArtifact.load_json(str(DOTA / "ml_artifact.json"))
    model = art.model

    ultimo: dict[int, frozenset[int]] = {}       # identidade -> elenco anterior
    desde_troca: dict[int, int] = defaultdict(lambda: 99)
    linhas = []
    for r in m.itertuples(index=False):
        mid = int(r.match_id)
        a, b = int(r.tid), int(r.oid)
        ra, rb = roster.get((mid, a)), roster.get((mid, b))
        ia, ib = rep.get(a, a), rep.get(b, b)
        na, nb = nome(a), nome(b)
        lg = str(getattr(r, "league", "") or "OTHER")
        t = str(tier_for_league(lg))
        try:
            p = float(model.p_blue(na, nb, blue_league=lg, red_league=lg,
                                   blue_tier=t, red_tier=t))
            y = int(r.win)
        except (TypeError, ValueError):
            continue

        for ident, rost, ehA in ((ia, ra, True), (ib, rb, False)):
            ant = ultimo.get(ident)
            if rost is not None and ant is not None:
                comum = len(rost & ant)
                trocados = 5 - comum
                if trocados > 0:
                    desde_troca[ident] = 0
                else:
                    desde_troca[ident] += 1
                # residuo DESTE time: resultado dele menos o que o modelo previu
                y_t = y if ehA else 1 - y
                p_t = p if ehA else 1 - p
                linhas.append({
                    "trocados": min(trocados, 3),
                    "jogos_desde_troca": min(desde_troca[ident], 20),
                    "residuo": y_t - p_t,
                    "p": p_t, "y": y_t,
                })
            if rost is not None:
                ultimo[ident] = rost

    df = pd.DataFrame(linhas)
    print(f"observacoes time-jogo com elenco conhecido dos dois lados: {len(df)}\n")

    print(f"{'jogadores trocados':<22}{'n':>7}{'residuo medio':>15}{'p medio':>10}{'real':>8}")
    for k, d in df.groupby("trocados"):
        rot = f"{k} trocado(s)" + ("+" if k == 3 else "")
        print(f"{rot:<22}{len(d):>7}{d['residuo'].mean():>+15.4f}{d['p'].mean():>10.3f}{d['y'].mean():>8.3f}")

    print(f"\n--- a literatura diz que o buraco fecha la pelo 10o jogo ---")
    print(f"{'jogos desde a troca':<22}{'n':>7}{'residuo medio':>15}")
    def faixa(n):
        return "0-3" if n <= 3 else ("4-7" if n <= 7 else ("8-10" if n <= 10 else "11+"))
    df["faixa"] = df["jogos_desde_troca"].map(faixa)
    ordem = ["0-3", "4-7", "8-10", "11+"]
    for k in ordem:
        d = df[df["faixa"] == k]
        if len(d):
            print(f"{k:<22}{len(d):>7}{d['residuo'].mean():>+15.4f}")

    intacto = df[df["trocados"] == 0]["residuo"]
    mexido = df[df["trocados"] > 0]["residuo"]
    if len(intacto) and len(mexido):
        dif = mexido.mean() - intacto.mean()
        # erro padrao da diferenca de medias
        se = math.sqrt(mexido.var(ddof=1) / len(mexido) + intacto.var(ddof=1) / len(intacto))
        z = dif / se if se > 0 else 0.0
        print(f"\nmexido - intacto = {dif:+.4f} (em pontos percentuais: {100*dif:+.2f} p.p.)")
        print(f"erro padrao {se:.4f} -> z = {z:+.2f}")
        if abs(z) < 2:
            print("VEREDITO: sem sinal — o core ja absorveu (nao construir a camada).")
        elif dif < 0:
            print("VEREDITO: SINAL — quem mexeu no elenco rende ABAIXO do previsto.")
            print("  Bate com a literatura. Continuidade de elenco vira feature.")
        else:
            print("VEREDITO: sinal INVERTIDO — quem mexeu rende ACIMA. Investigar antes de usar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
