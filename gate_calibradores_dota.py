# -*- coding: utf-8 -*-
"""GATE: onde parar de treinar, e com que calibrador servir a probabilidade.

Duas perguntas de uma vez, porque elas sao a MESMA pergunta vista de dois
lados (Berta, Holzmuller, Jordan & Bach, 2025 — "Rethinking Early Stopping:
Refine, Then Calibrate"):

  ONDE PARAR   erro de calibracao e erro de refinamento NAO sao minimizados
               juntos durante o treino. Parar pela logloss de validacao cai
               num ponto de compromisso subotimo PARA OS DOIS. A receita e
               parar pelo REFINAMENTO (a logloss DEPOIS de recalibrar) e
               deixar a calibracao para o post-hoc.

  COMO SERVIR  temperature (1 parametro) / Platt corrigido (2) / beta (3) —
               modelos aninhados, ver mlcore/calibration.py.

QUEM DECIDE: a LOGLOSS no teste cego. Os calibradores sao monotonicos, e
recalibracao monotonica preserva a resolucao — entao a diferenca de logloss
entre eles E a diferenca de calibracao, sem binning e sem vies. O ECE e
reportado so para leitura e para comparar com a literatura (Kim, Lee & Chung
2020 mediram, em LoL, ECE 4.47% sem calibrar e 1.11% com temperature).

PREVISAO A TESTAR (Niculescu-Mizil & Caruana): boosting empurra as previsoes
para longe de 0 e 1, produzindo diagrama de confiabilidade em forma de
sigmoide — o que faz da familia logistica a certa para arvores impulsionadas.
Se estiver certo, o Platt corrigido vai bem e a beta converge para perto dele
(b ~ a). Se a beta ficar na identidade (a ~ b ~ 1, c ~ 0), o modelo ja saia
calibrado e QUALQUER Platt estava piorando.

UM TREINO POR MINUTO: treina solto ate o teto e depois varre o numero de
arvores. Assim os dois criterios de parada saem do MESMO modelo, e a
comparacao isola o criterio em vez de misturar duas semeaduras diferentes.

Uso: python gate_calibradores_dota.py
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
from mlcore.calibration import (  # noqa: E402
    fit_beta_from_raw_probs,
    fit_platt_corrigido,
    fit_platt_from_raw_probs,
    fit_temperature_from_raw_probs,
)
from mlcore.live_ml import SNAPSHOT_MINUTES, StateToWinModel  # noqa: E402
from mlcore.live_ml.features import build_snapshot_features  # noqa: E402

EPS = 1e-12


def _ll(ps, ys) -> float:
    tot = 0.0
    for p, y in zip(ps, ys):
        q = min(max(float(p), EPS), 1.0 - EPS)
        tot += -(math.log(q) if int(y) == 1 else math.log(1.0 - q))
    return tot / max(1, len(ys))


def _ece(ps, ys, bins: int = 10) -> float:
    """ECE de largura uniforme, 10 caixas — a mesma convencao do artigo com
    que comparamos. So leitura: a decisao e por logloss."""
    caixas: dict[int, list] = {}
    for p, y in zip(ps, ys):
        caixas.setdefault(min(bins - 1, int(float(p) * bins)), []).append((float(p), int(y)))
    n = len(ys)
    return sum((len(v) / n) * abs(sum(q for q, _ in v) / len(v)
                                  - sum(w for _, w in v) / len(v)) for v in caixas.values())


class _Cru:
    @staticmethod
    def predict_proba(p):
        return p


def _aplica(cal, ps):
    return [cal.predict_proba(float(p)) for p in ps]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="ml_artifact.json")
    ap.add_argument("--val-frac", type=float, default=0.25,
                    help="fracao de CALIBRACAO e de TESTE (o resto e treino)")
    ap.add_argument("--min-n", type=int, default=300)
    ap.add_argument("--teto-arvores", type=int, default=1200)
    ap.add_argument("--passo", type=int, default=10, help="granularidade da varredura de arvores")
    ap.add_argument("--out", default="evidence/gate_calibradores_dota.json")
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

    art = MLArtifact.load_json(str(HERE / args.artifact))
    azul = tr[tr["side"] == "Blue"]
    p_pre = {}
    for r in azul.itertuples(index=False):
        gid = str(r.gameid)
        adv = tr[(tr["gameid"] == gid) & (tr["side"] == "Red")]
        if adv.empty:
            continue
        pb = float(art.model.p_blue(str(r.teamname), str(adv.iloc[0]["teamname"])))
        p_pre[(gid, "Blue")] = pb
        p_pre[(gid, "Red")] = 1.0 - pb
    print(f"linhas-time: {len(tr)} | partidas com previsao pre-jogo: {len(p_pre) // 2}")

    calibradores = [
        ("cru", None),
        ("temperature", fit_temperature_from_raw_probs),
        ("platt_atual", fit_platt_from_raw_probs),
        ("platt_corrig", fit_platt_corrigido),
        ("beta", fit_beta_from_raw_probs),
    ]

    saida = []
    # perdas por amostra de cada celula, agregadas nos 4 minutos: e o que
    # permite bootstrap PAREADO (mesma partida nas duas celulas comparadas).
    perdas: dict[tuple, list] = {}
    # previsoes no ponto de parada por refinamento, para o teste JUNTO
    pool: list[dict] = []
    for minuto in SNAPSHOT_MINUTES:
        snaps = build_snapshot_features(tr, minute=minuto,
                                        p_map_pre_by_gameid_side=p_pre,
                                        require_complete=True)
        n = len(snaps)
        if n < int(args.min_n):
            print(f"\nminuto {minuto}: amostra pequena ({n})")
            continue
        c1 = int(n * (1.0 - 2.0 * float(args.val_frac)))
        c2 = int(n * (1.0 - float(args.val_frac)))
        treino, calib, teste = snaps[:c1], snaps[c1:c2], snaps[c2:]
        if len(teste) < 100 or len(calib) < 100:
            print(f"\nminuto {minuto}: amostra pequena")
            continue

        # UM treino, solto ate o teto: os dois criterios de parada saem dele.
        modelo = StateToWinModel(minute=minuto)
        modelo.config.num_boost_round = int(args.teto_arvores)
        modelo.config.early_stopping_rounds = int(args.teto_arvores)  # nunca para sozinho
        modelo.fit(treino, calib)
        booster = modelo.lgb_model
        X_cal, y_cal = modelo._stack(calib)
        X_tes, y_tes = modelo._stack(teste)
        y_cal, y_tes = [int(v) for v in y_cal], [int(v) for v in y_tes]
        total = int(booster.current_iteration())

        # Varredura das arvores: para cada k, a logloss CRUA na calibracao
        # (criterio de hoje) e a de REFINAMENTO (depois de temperature).
        grade = list(range(args.passo, total + 1, args.passo))
        melhor_cru = (float("inf"), grade[0])
        melhor_ref = (float("inf"), grade[0])
        for k in grade:
            p = booster.predict(X_cal, num_iteration=k)
            cru = _ll(p, y_cal)
            t = fit_temperature_from_raw_probs(y_cal, p)
            ref = _ll(_aplica(t, p), y_cal)
            if cru < melhor_cru[0]:
                melhor_cru = (cru, k)
            if ref < melhor_ref[0]:
                melhor_ref = (ref, k)

        print(f"\n=== minuto {minuto} | treino {len(treino)} · calib {len(calib)} · teste {len(teste)} "
              f"| arvores: {total} ===")
        print(f"parada pela logloss crua .....: {melhor_cru[1]:>4} arvores")
        print(f"parada pelo REFINAMENTO ......: {melhor_ref[1]:>4} arvores")
        print(f"{'parada':<12}{'calibrador':<14}{'logloss':>10}{'ECE':>8}{'acerto':>8}   parametros")

        linha_min = {"minuto": minuto, "n_treino": len(treino), "n_calib": len(calib),
                     "n_teste": len(teste), "arvores_total": total,
                     "arvores_cru": melhor_cru[1], "arvores_refinamento": melhor_ref[1],
                     "celulas": []}

        for rotulo, k in (("crua", melhor_cru[1]), ("refinam.", melhor_ref[1])):
            p_cal = booster.predict(X_cal, num_iteration=k)
            p_tes = booster.predict(X_tes, num_iteration=k)
            for nome, ajusta in calibradores:
                cal = _Cru if ajusta is None else ajusta(y_cal, p_cal)
                q = _aplica(cal, p_tes)
                ll, ece = _ll(q, y_tes), _ece(q, y_tes)
                acc = sum(1 for a, b in zip(q, y_tes) if (a >= 0.5) == (b == 1)) / len(y_tes)
                par = "" if ajusta is None else json.dumps(cal.to_dict())
                par = par.replace('"', '').replace('{', '').replace('}', '')
                print(f"{rotulo:<12}{nome:<14}{ll:>10.4f}{ece:>8.3f}{acc:>8.1%}   {par}")
                perdas.setdefault((rotulo, nome), []).extend(
                    [-(math.log(min(max(a, EPS), 1 - EPS)) if b == 1
                       else math.log(1 - min(max(a, EPS), 1 - EPS))) for a, b in zip(q, y_tes)])
                linha_min["celulas"].append({
                    "parada": rotulo, "calibrador": nome, "logloss": round(ll, 5),
                    "ece": round(ece, 4), "acerto": round(acc, 4),
                    "parametros": None if ajusta is None else cal.to_dict()})
        pool.append({
            "minuto": minuto,
            "p_cal": list(booster.predict(X_cal, num_iteration=melhor_ref[1])),
            "y_cal": list(y_cal),
            "p_tes": list(booster.predict(X_tes, num_iteration=melhor_ref[1])),
            "y_tes": list(y_tes),
        })
        saida.append(linha_min)

    if saida:
        print("\n\n=== VEREDITO (logloss media no teste cego, por celula) ===")
        agr: dict[tuple, list] = {}
        for m in saida:
            for c in m["celulas"]:
                agr.setdefault((c["parada"], c["calibrador"]), []).append(c["logloss"])
        base = None
        print(f"{'parada':<12}{'calibrador':<14}{'logloss':>10}{'ganho vs hoje':>15}")
        for (parada, nome), vs in agr.items():
            media = sum(vs) / len(vs)
            if parada == "crua" and nome == "platt_atual":
                base = media
        for (parada, nome), vs in agr.items():
            media = sum(vs) / len(vs)
            g = "" if base is None else f"{base - media:>+15.5f}"
            marca = "  <- hoje" if (parada == "crua" and nome == "platt_atual") else ""
            print(f"{parada:<12}{nome:<14}{media:>10.5f}{g}{marca}")
        print("\nbarra do projeto: +0.002 de logloss para ACCEPT.")

        # BOOTSTRAP PAREADO (2000 reamostragens). Sem isto eu estaria escolhendo
        # a melhor de 10 celulas e chamando de ganho — que e exatamente o vies
        # de selecao por multiplas tentativas (Bailey & Lopez de Prado).
        import random as _rnd

        def ic(pior, melhor, n_boot=2000, semente=11):
            """IC 95% da diferenca media de perda (pior - melhor), PAREADA:
            a mesma partida entra nas duas celulas, entao a variancia do jogo
            em si sai da conta e sobra so a diferenca entre os metodos."""
            d = [x - y for x, y in zip(perdas[pior], perdas[melhor])]
            n = len(d)
            r = _rnd.Random(semente)
            ms = sorted(sum(d[r.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
            return sum(d) / n, ms[int(0.025 * n_boot)], ms[int(0.975 * n_boot)], n

        print()
        print("BOOTSTRAP PAREADO — 2000 reamostragens, IC 95% do ganho de logloss")
        print(f"{'comparacao':<46}{'ganho':>9}{'IC 95%':>24}   veredito")
        for titulo, pior, melhor in (
            ("refinamento + nao calibrar  vs  HOJE", ("crua", "platt_atual"), ("refinam.", "cru")),
            ("parar por refinamento (nenhum calibra)", ("crua", "cru"), ("refinam.", "cru")),
            ("nao calibrar  vs  calibrar com beta", ("refinam.", "beta"), ("refinam.", "cru")),
            ("nao calibrar  vs  Platt de hoje", ("refinam.", "platt_atual"), ("refinam.", "cru")),
        ):
            g, lo, hi, nn = ic(pior, melhor)
            vd = "ACEITA" if lo > 0 else ("zero dentro do IC" if hi > 0 else "PIORA")
            print(f"{titulo:<46}{g:>+9.5f}   [{lo:+.5f}, {hi:+.5f}]   {vd}  n={nn}")
        print("IC cruzando o zero = nao da pra afirmar ganho, por maior que seja a media.")


    # =================================================================
    # O calibrador esta aprendendo RUIDO de ~900 pontos. Duas respostas
    # principiadas, medidas aqui em vez de escolhidas por gosto:
    #
    #  (a) JUNTAR OS MINUTOS. A miscalibracao pode ser propriedade da
    #      FAMILIA de modelo (LightGBM nestas features), nao do minuto.
    #      Se for, um calibrador so, ajustado nos 4 minutos juntos, tem
    #      ~4x o dado e para de perseguir ruido.
    #  (b) ENCOLHER PARA A IDENTIDADE. Se o problema e variancia, mais
    #      regularizacao L2 puxa o ajuste para "nao fazer nada" — que na
    #      familia beta E um ponto valido (a=b=1, c=0). Varredura de L2.
    # =================================================================
    if pool:
        p_cal_all = [p for m in pool for p in m["p_cal"]]
        y_cal_all = [y for m in pool for y in m["y_cal"]]
        n_teste = sum(len(m["y_tes"]) for m in pool)
        print()
        print(f"=== CALIBRADOR JUNTANDO OS MINUTOS (ajuste em {len(y_cal_all)} pontos, "
              f"contra ~{len(pool[0]['y_cal'])} por minuto) ===")
        print(f"{'calibrador':<16}{'L2':>8}{'logloss':>10}{'vs nao calibrar':>18}")

        base_perdas = [pd_ for m in pool for pd_ in
                       [-(math.log(min(max(p, EPS), 1 - EPS)) if y == 1
                          else math.log(1 - min(max(p, EPS), 1 - EPS)))
                        for p, y in zip(m["p_tes"], m["y_tes"])]]
        base = sum(base_perdas) / len(base_perdas)
        print(f"{'(nao calibrar)':<16}{'-':>8}{base:>10.5f}{'':>18}")

        melhor = (0.0, None)
        for nome, ajusta in (("temperature", fit_temperature_from_raw_probs),
                             ("platt_corrig", fit_platt_corrigido),
                             ("beta", fit_beta_from_raw_probs)):
            for l2 in (1e-3, 1e-1, 1.0, 10.0, 100.0):
                cal = ajusta(y_cal_all, p_cal_all, l2=l2)
                perdas_c = [pd_ for m in pool for pd_ in
                            [-(math.log(min(max(cal.predict_proba(p), EPS), 1 - EPS)) if y == 1
                               else math.log(1 - min(max(cal.predict_proba(p), EPS), 1 - EPS)))
                             for p, y in zip(m["p_tes"], m["y_tes"])]]
                ll = sum(perdas_c) / len(perdas_c)
                g = base - ll
                if g > melhor[0]:
                    melhor = (g, (nome, l2, perdas_c))
                print(f"{nome:<16}{l2:>8g}{ll:>10.5f}{g:>+18.5f}")

        if melhor[1] is not None:
            import random as _rnd2
            nome, l2, perdas_c = melhor[1]
            d = [a - b for a, b in zip(base_perdas, perdas_c)]
            n = len(d)
            r = _rnd2.Random(23)
            ms = sorted(sum(d[r.randrange(n)] for _ in range(n)) / n for _ in range(2000))
            lo, hi = ms[50], ms[1949]
            vd = "ACEITA" if lo > 0 else ("zero dentro do IC" if hi > 0 else "PIORA")
            print()
            print(f"melhor junto: {nome} com L2={l2:g} -> ganho {melhor[0]:+.5f} "
                  f"IC95% [{lo:+.5f}, {hi:+.5f}]  {vd}   (n={n_teste})")
            print("Lembrete: aqui eu testei 15 combinacoes. Ganho de melhor-de-15 "
                  "precisa de barra MAIOR que a de um teste unico.")
        else:
            print()
            print("NENHUMA combinacao bateu 'nao calibrar'. Com este tamanho de amostra,")
            print("calibrar custa mais do que ganha — juntar os minutos nao resolveu.")

    out = HERE / str(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"minutos": saida}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[salvo] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
