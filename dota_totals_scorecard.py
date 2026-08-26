# -*- coding: utf-8 -*-
"""REGUA DOS TOTAIS DO DOTA — a Analise de Linha de la nunca foi medida.

Situacao encontrada em 25/08/2026: o motor de linhas (mlcore/lines_engine) roda
no Dota porque e o mesmo codigo do LoL, mas NAO HA regua, nem parametros de
totais, nem live_models/markets. Ou seja, a aba servia numero sem ninguem saber
se prestava — a mesma situacao do pre-jogo antes da dota_scorecard.

Metodo (copiado do totals_scorecard do LoL): replay PREQUENCIAL — para cada mapa,
os perfis dos dois times sao construidos SO com jogos anteriores, e mede-se
P(over linha) contra o resultado real. Sem isso o numero seria in-sample e
inflado, como ja aconteceu duas vezes neste projeto.

LINHAS medidas na base do Dota (23.043 partidas), NAO copiadas do LoL — la sao
27.5 kills e 32.5 min, aqui a mediana e 56 kills e 34.9 min. Copiar teria
produzido lixo com aparencia de medicao.

Nota pela MESMA formula do resto do projeto (Brier -> nota), declarada em
dota_scorecard.py.

RESULTADO 26/08/2026 (2026 cego, 3.010 partidas). A coluna que vale e "vs base"
— ganho sobre chutar a taxa-base do over, porque Brier premia LINHA TORTA:
    kills       +0.0139  nota 8.16   <- o unico com sinal de verdade
    time        +0.0054  nota 7.30
    barons      +0.0038  nota 7.49   (roshans)
    towers      +0.0007  nota 5.33   <- nao sabe nada
    inhibitors  -0.0015  nota 4.96   <- PIOR que a taxa-base (barracas)
Nota geral 6.65 vs 6.12 do LoL, mas a leitura honesta e: so kills presta.

TETO DESTA FRENTE. Tres ataques ja medidos, nenhum passou:
  - Dixon-Coles (LoL, 2026-07-05): REJECT em towers/barons/inhibitors.
  - dispersao por liga (LoL, 2026-07-17): so time ACCEPT, +0.0007.
  - calibracao Platt da probabilidade (aqui, --calibrar): TUDO ~neutro; o
    melhor foi inhibitors +0.0016, abaixo da barra de +0.002.
Distribuicao nao e o problema: roshans e SUB-disperso (var/media 0.74) e
barracas SUPER-disperso (1.27) — nenhuma familia fixa serve aos dois, e a
calibracao, que cobriria ambos, nao pagou. O problema e SINAL: a correlacao
entre media prevista e total real e 0.12 em towers e 0.13 em barracas.
  - prever o total pela DURACAO esperada (--via-duracao): REJECT em kills
    (-0.0355), ~neutro no resto. Melhor caso inhibitors +0.0010.

ONDE O TETO REALMENTE ESTA (medido, e o achado que importa daqui):
os objetivos sao quase FUNCAO DA DURACAO. Correlacao com a duracao REAL:
    roshans 0.87 | torres 0.80 | barracas 0.55 | kills 0.45
Ou seja, nao falta sinal nesses mercados — falta prever a DURACAO. O motor
preve duracao com corr 0.21, e por isso o ataque via duracao nasce morto:
a duracao prevista quase nao varia, entao o total previsto quase nao varia.
Se um dia alguem quiser mexer nos totais do Dota, o alvo unico e
PREVER A DURACAO DA PARTIDA. Um numero melhor ali levanta tres mercados
de uma vez. Enquanto isso nao existir, so kills presta.

Uso: python dota_totals_scorecard.py [--season 2026] [--min-jogos 8] [--calibrar]
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
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mlcore.lines_engine import (  # noqa: E402
    Filters, MatchupTotals, build_profile, matchup_expected_totals,
    prob_over_from_total)
from mlcore.stats_teams import build_team_games  # noqa: E402
from mlcore.calibration import fit_platt_from_raw_probs  # noqa: E402

# (metrica do engine, linha padrao MEDIDA no Dota, coluna do total no jogo)
MERCADOS = [
    ("kills", 56.5, "kills"),
    ("towers", 12.5, "towers"),
    ("barons", 2.5, "barons"),
    ("inhibitors", 4.5, "inhibitors"),
    ("time", 35.0, "gamelength_min"),
]


def nota(brier: float) -> float:
    """MESMA regua dos totais do LoL (tools/totals_scorecard.py, 18/07/2026):
    8.0 no brier de referencia 0.2067. Copiada para os dois numeros serem
    comparaveis — inventar uma segunda regua aqui seria comparar laranja e maca."""
    return max(0.0, min(10.0, 8.0 - 68.0 * (brier - 0.2067)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--min-jogos", type=int, default=8,
                    help="jogos anteriores minimos por time para arriscar previsao")
    ap.add_argument("--via-duracao", action="store_true",
                    help="mede prever o total pela DURACAO esperada (ajuste em 2025, cego na alvo)")
    ap.add_argument("--calibrar", action="store_true",
                    help="mede se calibrar a probabilidade (Platt em 2025, cego em 2026) melhora")
    ap.add_argument("--out", default="evidence/dota_totals_scorecard.json")
    args = ap.parse_args()

    quadros = [pd.read_csv(c, low_memory=False)
               for p in ("CSV_HISTORICO", "CSV_DIARIO")
               for c in sorted((HERE / p).glob("*_oe_like.csv"))]
    if not quadros:
        print("sem CSV — rode dota_adapter.py")
        return 1
    d = pd.concat(quadros, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # totais reais por partida
    b = d[d["side"] == "Blue"].set_index("gameid")
    r = d[d["side"] == "Red"].set_index("gameid")
    reais: dict[str, pd.Series] = {}
    for _m, _l, col in MERCADOS:
        if col == "gamelength_min":
            reais[col] = pd.to_numeric(b["gamelength"], errors="coerce") / 60.0
        else:
            reais[col] = (pd.to_numeric(b[col], errors="coerce")
                          + pd.to_numeric(r[col], errors="coerce"))

    season = int(args.season) if args.season else int(pd.to_numeric(d["year"], errors="coerce").max())
    anos = pd.to_numeric(d["year"], errors="coerce")
    # GATE DA CALIBRACAO: replay tambem nas seasons ANTERIORES, so para AJUSTAR
    # o Platt. A season alvo continua CEGA — o calibrador nunca ve 2026.
    _passado = bool(args.calibrar or args.via_duracao)
    alvo = d[(anos <= season) & (d["side"] == "Blue")] if _passado else         d[(anos == season) & (d["side"] == "Blue")]
    alvo_ano = pd.to_numeric(alvo["year"], errors="coerce").to_numpy()
    print(f"replay prequencial | season {season} | {len(alvo)} partidas\n")

    # O motor come team_games (uma linha por time-jogo, com for/against ja
    # pareado), NAO o CSV cru: build_profile pede as colunas team/kills_for/...
    tg = build_team_games(d)
    tg["date"] = pd.to_datetime(tg["date"], errors="coerce")
    tg = tg.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    # Fatiar "so o passado" por time, com busca binaria: refiltrar o quadro
    # inteiro dentro do laco era O(n^2) (38 mil linhas x 4 mil partidas).
    por_time = {t: g.reset_index(drop=True) for t, g in tg.groupby("team", sort=False)}
    datas = {t: g["date"].to_numpy() for t, g in por_time.items()}

    def passado(time_nome, quando):
        g = por_time.get(time_nome)
        if g is None:
            return None
        corte = int(datas[time_nome].searchsorted(quando.to_datetime64(), side="left"))
        return g.iloc[:corte] if corte >= int(args.min_jogos) else None

    filtros = Filters()
    acc: dict[str, list[tuple[float, int]]] = {m[0]: [] for m in MERCADOS}
    # media prevista x total real: separa "dado quebrado" (corr ~0) de
    # "distribuicao errada" (corr boa, probabilidade ruim).
    mus: dict[str, list[tuple[float, float]]] = {m[0]: [] for m in MERCADOS}
    treino: dict[str, list[tuple[float, int]]] = {m[0]: [] for m in MERCADOS}
    # (duracao prevista, total real) — para o ataque VIA DURACAO
    dur_tr: dict[str, list[tuple[float, float]]] = {m[0]: [] for m in MERCADOS}
    dur_ev: dict[str, list[tuple[float, float, float]]] = {m[0]: [] for m in MERCADOS}
    vistos = 0
    for _i, row in enumerate(alvo.itertuples(index=False)):
        gid = str(row.gameid)
        adv = r["teamname"].get(gid)
        if adv is None:
            continue
        ha, hb = passado(str(row.teamname), row.date), passado(str(adv), row.date)
        if ha is None or hb is None:
            continue
        pa = build_profile(ha, team=str(row.teamname), filters=filtros)
        pb = build_profile(hb, team=str(adv), filters=filtros)
        if min(pa.n_games, pb.n_games) < int(args.min_jogos):
            continue
        tot = matchup_expected_totals(pa, pb)
        vistos += 1
        _tt = tot.get("time")
        _mu_time = float(getattr(_tt, "mean", float("nan"))) if _tt else float("nan")
        for metrica, linha, col in MERCADOS:
            t = tot.get(metrica)
            if t is None:
                continue
            p = prob_over_from_total(t, float(linha))
            y = reais[col].get(gid)
            if p == p and y == y:
                par = (float(p), int(float(y) > float(linha)))
                if int(alvo_ano[_i]) == season:
                    acc[metrica].append(par)
                    mus[metrica].append((float(t.mean), float(y)))
                else:
                    treino[metrica].append(par)
                if _mu_time == _mu_time:
                    if int(alvo_ano[_i]) == season:
                        dur_ev[metrica].append((_mu_time, float(y), float(p)))
                    else:
                        dur_tr[metrica].append((_mu_time, float(y)))

    print(f"{'mercado':<14}{'linha':>7}{'n':>7}{'Brier':>9}{'acerto':>9}{'over real':>11}{'over prev':>11}{'nota':>7}{'vs base':>9}{'corr':>7}")
    saida = []
    for metrica, linha, _c in MERCADOS:
        v = acc[metrica]
        if len(v) < 100:
            print(f"{metrica:<14}{linha:>7.1f}{len(v):>7}   amostra pequena")
            continue
        br = sum((p - y) ** 2 for p, y in v) / len(v)
        # Controle de TAXA-BASE: brier de quem so chuta a frequencia do over.
        # Sem ele, linha torta vira "talento" (ver comentario gemeo no LoL).
        _c = mus[metrica]
        corr = float(pd.Series([a for a, _ in _c]).corr(pd.Series([b for _, b in _c]))) if _c else float("nan")
        _pb = sum(y for _p, y in v) / len(v)
        base = _pb * (1.0 - _pb)
        ac = sum(1 for p, y in v if (p >= .5) == (y == 1)) / len(v)
        saida.append({"mercado": metrica, "linha": linha, "n": len(v),
                      "brier": round(br, 4), "acerto": round(ac, 3), "nota": round(nota(br), 2),
                      "brier_base": round(base, 4), "ganho_base": round(base - br, 4),
                      "corr": round(corr, 3),
                      "over_real": round(sum(y for _p, y in v) / len(v), 3),
                      "over_previsto": round(sum(p for p, _y in v) / len(v), 3)})
        print(f"{metrica:<14}{linha:>7.1f}{len(v):>7}{br:>9.4f}{ac:>9.1%}"
              f"{saida[-1]['over_real']:>11.1%}{saida[-1]['over_previsto']:>11.1%}"
              f"{saida[-1]['nota']:>7.2f}{saida[-1]['ganho_base']:>+9.4f}{saida[-1]['corr']:>7.2f}")

    if args.via_duracao:
        # OS OBJETIVOS SAO QUASE FUNCAO DA DURACAO — medido nesta base: torres
        # 0.80, roshans 0.87, barracas 0.55, kills 0.45 de correlacao com a
        # duracao REAL. Ou seja, o motor nao falha nesses mercados por falta de
        # sinal; falha porque preve mal a duracao (corr 0.21). Aqui: prever o
        # total A PARTIR da duracao esperada bate a media for/against? Reta
        # ajustada nas seasons anteriores, aplicada CEGA na alvo, sd = residuo.
        import statistics as _st
        print()
        print("GATE VIA DURACAO (reta ajustada nas seasons anteriores, cega na alvo)")
        print(f"{'mercado':<14}{'brier V0':>10}{'brier DUR':>11}{'brier BL':>10}"
              f"{'difDUR':>9}{'difBL':>9}  veredito")
        for metrica, linha, _col in MERCADOS:
            tr, ev = dur_tr[metrica], dur_ev[metrica]
            if len(tr) < 500 or len(ev) < 100:
                print(f"{metrica:<14}   amostra pequena")
                continue
            xs = [a for a, _b in tr]
            mx, my = _st.fmean(xs), _st.fmean([b for _a, b in tr])
            sxx = sum((x - mx) ** 2 for x in xs)
            if sxx <= 1e-9:
                print(f"{metrica:<14}   duracao prevista constante")
                continue
            b1 = sum((x - mx) * (y - my) for x, y in tr) / sxx
            b0 = my - b1 * mx
            sd = max(_st.pstdev([y - (b0 + b1 * x) for x, y in tr]), 1e-6)
            b_v0 = b_du = b_bl = 0.0
            for x, y, q in ev:
                alvo_y = 1.0 if y > linha else 0.0
                pdur = prob_over_from_total(
                    MatchupTotals(mean=float(b0 + b1 * x), sd=float(sd), dist="normal"),
                    float(linha))
                b_v0 += (q - alvo_y) ** 2
                b_du += (pdur - alvo_y) ** 2
                b_bl += (0.5 * (pdur + q) - alvo_y) ** 2
            n_ev = len(ev)
            b_v0, b_du, b_bl = b_v0 / n_ev, b_du / n_ev, b_bl / n_ev
            g_du, g_bl = b_v0 - b_du, b_v0 - b_bl
            melhor = max(g_du, g_bl)
            vd = "ACCEPT" if melhor >= 0.002 else ("REJECT" if melhor <= -0.002 else "~neutro")
            print(f"{metrica:<14}{b_v0:>10.4f}{b_du:>11.4f}{b_bl:>10.4f}"
                  f"{g_du:>+9.4f}{g_bl:>+9.4f}  {vd}")

    if args.calibrar:
        print()
        print("GATE DA CALIBRACAO (Platt ajustado em seasons anteriores, cego na alvo)")
        print(f"{'mercado':<14}{'n treino':>9}{'brier V0':>10}{'brier cal':>11}{'ganho':>9}  veredito")
        for metrica, _linha, _col in MERCADOS:
            tr, ev = treino[metrica], acc[metrica]
            if len(tr) < 500 or len(ev) < 100:
                print(f"{metrica:<14}{len(tr):>9}   amostra pequena")
                continue
            cal = fit_platt_from_raw_probs([y for _p, y in tr], [p for p, _y in tr])
            b0 = sum((p - y) ** 2 for p, y in ev) / len(ev)
            b1 = sum((cal.predict_proba(p) - y) ** 2 for p, y in ev) / len(ev)
            g = b0 - b1
            vd = "ACCEPT" if g >= 0.002 else ("REJECT" if g <= -0.002 else "~neutro")
            print(f"{metrica:<14}{len(tr):>9}{b0:>10.4f}{b1:>11.4f}{g:>+9.4f}  {vd}"
                  f"   (a={cal.a:.3f} b={cal.b:+.3f})")

    if saida:
        geral = sum(x["nota"] for x in saida) / len(saida)
        print()
        print(f"NOTA DOS TOTAIS DO DOTA (regua do LoL): {geral:.2f}   [LoL = 6.12]")
        print("  mas a coluna que vale e 'vs base': ganho sobre chutar a taxa-base.")
        print("  <= 0 significa que o motor NAO sabe nada daquele mercado.")
    print(f"\npartidas avaliadas: {vistos}")
    print("leitura: Brier 0.25 = chute (50/50). Abaixo disso o motor sabe algo;")
    print("         'over real' vs 'over previsto' muito distantes = linha mal centrada.")
    out = HERE / str(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"season": season, "mercados": saida},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[salvo] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
