# -*- coding: utf-8 -*-
"""RÉGUA DO DOTA — a nota do modelo por comando (item 2 da auditoria, 16/08/2026).

Por que existe: o LoL tem 11 scorecards; o Dota tinha 8 scripts soltos e NENHUMA
nota de produto. Sem régua, nenhuma mudança aqui pode ser julgada — foi preciso
improvisar medição para descobrir que o TI 2026 estava com nota 0.68. Isso devia
ser um comando, e agora é.

MESMA FÓRMULA do scorecard oficial do LoL (aleatório 0.6931 -> 5.0, elite 0.55 ->
9.5, linear). Está duplicada aqui porque os dois repositórios são separados de
propósito; o teste test_dota_scorecard.py trava as âncoras para as réguas não
divergirem em silêncio.

O QUE ELE REPORTA (e por quê):
  GERAL          — a nota, sem ponderação inventada. O LoL pondera 45/55 porque
                   ele decidiu isso; aqui não há decisão dele, então não invento.
  por familiaridade — quantos jogos o modelo tem de cada lado. CUIDADO: este
                   corte se INVERTE conforme o contexto (ver o aviso impresso na
                   saída) — na temporada, pouco-conhecido é time fraco e o modelo
                   acerta fácil; num LAN de classificação é time que passou pelo
                   funil e o modelo erra feio. Diagnóstico, não semáforo.
  por torneio    — onde confiar, evento a evento (é onde o TI aparecia com 0.68).
                   ESTE é o que serve para decidir aposta, junto do ECE.

Uso:  python dota_scorecard.py [--min-n 60] [--season 2026]
Saída: tabela + evidence/dota_scorecard.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mlcore.artifact import MLArtifact  # noqa: E402
from mlcore.io_oracle import load_oracle_elixir_games  # noqa: E402
from mlcore.tiers import tier_for_league  # noqa: E402

RANDOM_LOGLOSS = 0.6931   # ln(2) — a moeda. Ancora a nota.
ELITE_LOGLOSS = 0.55
CONHECIDO_MIN = 50        # jogos para o modelo "conhecer" um time (medido no TI)


def grade_from_logloss(ll: float) -> float:
    """Mapa transparente: moeda -> 5.0, elite -> 9.5, linear, travado em [0,10].

    IDÊNTICO ao tools/ml_quality_scorecard.py do LoL. Se mexer aqui, mexa lá —
    o teste das âncoras existe para isso não passar batido."""
    if not (isinstance(ll, float) and math.isfinite(ll)):
        return float("nan")
    g = 5.0 + (RANDOM_LOGLOSS - ll) / (RANDOM_LOGLOSS - ELITE_LOGLOSS) * 4.5
    return float(max(0.0, min(10.0, g)))


def _ece(d: pd.DataFrame) -> float:
    bins = [0, .3, .4, .45, .55, .6, .7, 1]
    d = d.copy()
    d["_b"] = pd.cut(d["p"], bins)
    tot = 0.0
    for _, dd in d.groupby("_b", observed=True):
        if len(dd) == 0:
            continue
        tot += (len(dd) / len(d)) * abs(dd["p"].mean() - dd["y"].mean())
    return float(tot)


def _bloco(d: pd.DataFrame) -> dict:
    if d.empty:
        return {"n": 0}
    return {
        "n": int(len(d)),
        "nota": round(float(grade_from_logloss(float(d["ll"].mean()))), 2),
        "logloss": round(float(d["ll"].mean()), 4),
        "acuracia": round(float(((d["p"] >= .5) == (d["y"] >= .5)).mean()), 3),
        "ece": round(_ece(d), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=60, help="amostra mínima por torneio")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--artifact", default="ml_artifact.json")
    args = ap.parse_args()

    art = MLArtifact.load_json(str(HERE / args.artifact))
    model = art.model
    gp = getattr(model, "games_played", {}) or {}

    frames = []
    for pasta in ("CSV_HISTORICO", "CSV_DIARIO"):
        for c in sorted((HERE / pasta).glob("*_oe_like.csv")):
            try:
                d = load_oracle_elixir_games(str(c), keep_playoffs=None)
                if not d.empty:
                    frames.append(d)
            except Exception as e:
                print(f"skip {c.name}: {e}")
    if not frames:
        print("nenhum CSV encontrado — rode dota_adapter.py antes")
        return 1
    g = pd.concat(frames, ignore_index=True, join="inner")
    g["date"] = pd.to_datetime(g["date"], errors="coerce")
    g = g.dropna(subset=["date"]).drop_duplicates("gameid", keep="last")
    season = int(args.season) if args.season else int(pd.to_numeric(g["year"], errors="coerce").max())
    g = g[pd.to_numeric(g["year"], errors="coerce") == season]

    linhas = []
    for r in g.itertuples(index=False):
        try:
            y = int(r.blue_win)
        except (TypeError, ValueError):
            continue
        b, rd = str(r.blue), str(r.red)
        lg = str(getattr(r, "league", "") or "OTHER")
        t = str(tier_for_league(lg))
        p = float(model.p_blue(b, rd, blue_league=lg, red_league=lg, blue_tier=t, red_tier=t))
        linhas.append({
            "torneio": lg, "p": p, "y": y,
            "min_gp": min(int(gp.get(b, 0)), int(gp.get(rd, 0))),
        })
    df = pd.DataFrame(linhas)
    if df.empty:
        print("sem jogos na season")
        return 1
    pc = df["p"].clip(1e-9, 1 - 1e-9)
    df["ll"] = -(df["y"] * pc.apply(math.log) + (1 - df["y"]) * (1 - pc).apply(math.log))

    geral = _bloco(df)
    conhecidos = _bloco(df[df["min_gp"] >= CONHECIDO_MIN])
    novatos = _bloco(df[df["min_gp"] < CONHECIDO_MIN])

    print(f"=== RÉGUA DO DOTA — season {season} ===")
    print(f"NOTA GERAL: {geral['nota']}   [logloss {geral['logloss']}, "
          f"acerto {geral['acuracia']:.1%}, ECE {geral['ece']}, n={geral['n']}]")
    print("  (moeda = 5.0 · elite = 9.5 · mesma régua do LoL)\n")
    print(f"{'segmento':<34}{'n':>6}{'nota':>7}{'logloss':>10}{'acerto':>8}{'ECE':>7}")
    for nome, blk in (("confronto entre times rodados", conhecidos),
                      (f"algum lado com <{CONHECIDO_MIN} jogos", novatos)):
        if blk["n"]:
            print(f"{nome:<34}{blk['n']:>6}{blk['nota']:>7.2f}{blk['logloss']:>10.4f}"
                  f"{blk['acuracia']:>8.3f}{blk['ece']:>7.3f}")
    # ARMADILHA MEDIDA em 16/08/2026, deixada explicita porque enganou a mim
    # primeiro: na temporada normal "time com poucos jogos" e time FRACO apanhando
    # de um forte — atropelo previsivel, nota alta. Num LAN de convite/classificacao
    # (TI) e o oposto: quem tem poucos jogos CLASSIFICOU, e forte, e o modelo erra
    # feio. Mesma metrica, significados invertidos. Nao use este corte como
    # semaforo de aposta; use a linha do TORNEIO, e sobretudo o ECE.
    print("  ATENCAO: pouco-conhecido NAO significa arriscado. Na temporada e time")
    print("  fraco (atropelo previsivel); em LAN de classificacao e time que passou")
    print("  pelo funil (forte, e o modelo erra). Olhe a linha do torneio.")

    tor = []
    for lg, d in df.groupby("torneio"):
        if len(d) < int(args.min_n):
            continue
        blk = _bloco(d)
        blk["torneio"] = lg
        tor.append(blk)
    tor.sort(key=lambda x: x["nota"])
    print(f"\n{'torneio (piores primeiro)':<40}{'n':>6}{'nota':>7}{'ECE':>7}")
    for b in tor[:8]:
        print(f"{b['torneio'][:39]:<40}{b['n']:>6}{b['nota']:>7.2f}{b['ece']:>7.3f}")
    print("...")
    for b in tor[-3:]:
        print(f"{b['torneio'][:39]:<40}{b['n']:>6}{b['nota']:>7.2f}{b['ece']:>7.3f}")

    print("\nleitura: nota alta = torneio previsível (dá pra extrair edge);")
    print("         ECE alto (>0.10) = o % do modelo NÃO é honesto ali, EV não vale.")

    # ===== AO VIVO =====================================================
    # O LoL tem tools/live_scorecard.py; aqui a live e so o state->win (o Dota
    # nao tem first-markets nem player-props), entao ela mora nesta mesma regua
    # em vez de ganhar um arquivo so pra ela.
    vivo = None
    _f6 = HERE / "FASE6_LIVE_DOTA.json"
    if _f6.exists():
        try:
            _d = json.loads(_f6.read_text(encoding="utf-8"))
            mins = [m for m in _d.get("minutos", []) if m.get("salvo")]
        except (OSError, ValueError):
            mins = []
        if mins:
            _ll = sum(float(m["ll_estado"]) for m in mins) / len(mins)
            _pre = sum(float(m["ll_pre_jogo"]) for m in mins) / len(mins)
            _ece = [float(m["ece"]) for m in mins if m.get("ece") is not None]
            vivo = {"nota": round(grade_from_logloss(_ll), 2),
                    "logloss": round(_ll, 4), "logloss_pre_jogo": round(_pre, 4),
                    "ganho": round(_pre - _ll, 4),
                    "ece": round(sum(_ece) / len(_ece), 4) if _ece else None,
                    "minutos": len(mins)}
            print()
            print("=== AO VIVO (estado do mapa -> vitoria) ===")
            print(f"{'minuto':>7}{'pre-jogo':>11}{'estado':>10}{'ganho':>9}{'ECE':>8}{'nota':>7}")
            for m in mins:
                print(f"{int(m['minuto']):>7}{float(m['ll_pre_jogo']):>11.4f}"
                      f"{float(m['ll_estado']):>10.4f}{float(m['ganho']):>+9.4f}"
                      f"{float(m.get('ece', float('nan'))):>8.3f}"
                      f"{grade_from_logloss(float(m['ll_estado'])):>7.2f}")
            print(f"MEDIA: ganho {vivo['ganho']:+.4f} sobre o pre-jogo, ECE {vivo['ece']}")
            print("  o LoL, na mesma regua: ganho +0.1570, ECE 0.015.")
            if vivo["ece"] and vivo["ece"] > 0.025:
                print("  ECE ACIMA do LoL: o % ao vivo daqui e menos honesto — trate")
                print("  o numero como direcao, nao como preco. Melhora sozinho")
                print("  conforme o coletor de fotos por minuto acumula partidas.")

    ev = HERE / "evidence"
    ev.mkdir(exist_ok=True)
    (ev / "dota_scorecard.json").write_text(json.dumps({
        "season": season, "geral": geral,
        "conhecidos": conhecidos, "pouco_conhecidos": novatos,
        "torneios": tor, "ao_vivo": vivo,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    # MESMO schema do evidence/nota_por_liga.json do LoL, de proposito: assim o
    # app_core/nota_liga_core.py roda aqui VERBATIM, sem fork. Se um dia o
    # schema mudar la, muda aqui junto.
    (ev / "nota_por_liga.json").write_text(json.dumps({
        "gerado_em": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "season": season, "nota_geral": geral.get("nota"), "min_n": int(args.min_n),
        "ligas": [{"liga": b["torneio"], "tier": "", "n": b["n"], "nota": b["nota"],
                   "logloss": b["logloss"], "acuracia": b["acuracia"], "ece": b["ece"]}
                  for b in tor],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n[salvo] evidence/dota_scorecard.json + evidence/nota_por_liga.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
