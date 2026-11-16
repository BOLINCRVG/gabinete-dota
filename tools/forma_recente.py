# -*- coding: utf-8 -*-
"""FORMA RECENTE: quanto cada time rendeu acima/abaixo do que o modelo previa nos ultimos 5 mapas.

Pedido dele (16/11/2026): "o time esta tomando varada contra os times piores e o modelo abre odd
even contra um time que vem dando varada nos melhores" (WE no LoL: 7-8 series de 2-0 ate o modelo
reagir; Synapse x Spirit Academy no Dota). Aprovado por ele no mesmo dia ("sim").

Medido as cegas (replay so com o passado; peso ajustado num periodo, lido no seguinte):
  LoL : Elo de producao + MOVDA, ajuste 2025, teste 2026: +0.0073 IC95 [+0.0046; +0.0098] (9.864 mapas)
  Dota: replay WHR das medicoes, teste do periodo cego: +0.0041 IC95 [+0.0017; +0.0065] (3.982 mapas)
  CS  : nao passou (-0.0003): o modelo por jogador ja acompanha a fase. Nao usado.
forma do time = soma de (resultado - p previsto) nos ultimos 5 mapas dele, com o p da HORA do jogo
(replay prequencial -- o p do modelo de hoje ja "sabe" esses jogos e esconderia a forma).
O app soma ao logit do mapa: PESO * (forma_A - forma_B) (app_core/forma_recente_core.py).

Gera forma_recente_<esporte>.json. Roda no robo diario. Arquivo identico nos dois repositorios.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError, ValueError):
    pass

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tools"))
E_DOTA = (RAIZ / "dota_paths.py").exists()
ESPORTE = "dota" if E_DOTA else "lol"
JANELA = 5
PESO = {"lol": 0.172, "dota": 0.098}[ESPORTE]
MEDICAO = {
    "lol": {"medido_em": "2026-11-16", "ganho_logloss": 0.0073, "ic95": [0.0046, 0.0098], "n_teste": 9864,
            "base": "Elo de producao + MOVDA, prequencial, ajuste 2025 / teste 2026"},
    "dota": {"medido_em": "2026-11-16", "ganho_logloss": 0.0041, "ic95": [0.0017, 0.0065], "n_teste": 3982,
             "base": "replay WHR (w2=20) prequencial das medicoes do Dota, tune / cego"},
}[ESPORTE]


def _replay_lol():
    """[(data, time, adversario, venceu, p)] em ordem, com o p da hora (Elo de producao + MOVDA)."""
    import eval_whr as ew
    g = ew.load_games(ew.default_csvs())
    ano = int(g["yr"].max()) + 1
    lp = ew.sos_points_from_train(g, ano, 400.0)
    mov = ew.fit_expected_mov(g, oos_year=ano, league_points=lp)
    elo = ew.EloModel(ew.EloParams(**ew.PROD_PARAMS))
    elo.league_base_points_map = dict(lp)
    am, bm, gm = mov["alpha_m"], mov["beta_m"], mov["gamma_m"]
    out = []
    for r in g.itertuples(index=False):
        bl, rl = r.dl, r.ol
        p = None
        if bl and rl:
            p = float(elo.p_blue(r.team, r.opponent, blue_league=bl, red_league=rl,
                                 blue_tier=ew._tier(bl), red_tier=ew._tier(rl)))
        elo.update(str(r.team), str(r.opponent), int(r.win), blue_league=bl or None, red_league=rl or None,
                   blue_tier=ew._tier(bl) if bl else None, red_tier=ew._tier(rl) if rl else None)
        if p is not None:
            dr = ew._dr_from_p(p, ew.PROD_PARAMS["scale_points"])
            mv = ew._mov_of(r)
            if mv is not None:
                adj = 0.5 * (mv - (am * math.tanh(bm * dr) + gm))
                elo.ratings[str(r.team)] = elo.rating_raw(str(r.team)) + adj
                elo.ratings[str(r.opponent)] = elo.rating_raw(str(r.opponent)) - adj
            out.append((str(getattr(r, "date", "") or ""), str(r.team), str(r.opponent), int(r.win), p))
    return out


def _replay_dota():
    """Idem no Dota (replay WHR das medicoes); nomes pelo teamid do CSV que o app usa."""
    import pandas as pd
    from dota_paths import ensure_lol_on_path
    ensure_lol_on_path()
    from eval_whr_dota import MIN_GAMES_EVAL, SideTracker, load_matches
    from mlcore.whr_core import WHRFitter
    nome = {}
    for c in ("CSV_HISTORICO/2025_Dota_match_data_oe_like.csv", "CSV_DIARIO/2026_Dota_match_data_oe_like.csv"):
        p = RAIZ / c
        if p.exists():
            d = pd.read_csv(p, usecols=["teamid", "teamname", "date", "position"], low_memory=False)
            d = d[d["position"] == "team"].sort_values("date")
            for tid, tn in zip(d["teamid"], d["teamname"]):
                try:
                    nome[str(int(float(tid)))] = str(tn)
                except (TypeError, ValueError):
                    pass
    m = load_matches()
    fit = WHRFitter(w2_elo_per_day=20.0)
    side = SideTracker()
    out = []
    for k, r in enumerate(m.itertuples(index=False)):
        a, b = str(r.tid), str(r.oid)
        prof = r.tier == "professional"
        p = None
        if prof and len(fit.games.get(a, ())) >= MIN_GAMES_EVAL and len(fit.games.get(b, ())) >= MIN_GAMES_EVAL:
            p = 1 / (1 + math.exp(-(fit.rating_nat(a) - fit.rating_nat(b) + side.offset_nat(int(r.day)))))
        fit.add_game(a, b, int(r.win), int(r.day))
        fit.refresh_pair(a, b, iters=2)
        if prof:
            side.add(int(r.day), int(r.win))
            if p is not None:
                na = nome.get(a) or nome.get(str(r.tid_orig)) or str(r.teamname)
                nb = nome.get(b) or nome.get(str(r.oid_orig)) or str(r.opp_teamname)
                out.append((str(r.dt)[:19], na, nb, int(r.win), p))
        if k % 500 == 0:
            fit.full_sweeps(sweeps=1, iters=1)
    return out


def calcular(jogos):
    """{time: {'forma', 'mapas', 'ultimo', 'resultados'}} com os ultimos JANELA mapas de cada time."""
    hist = defaultdict(lambda: deque(maxlen=JANELA))
    for data, a, b, y, p in jogos:
        hist[a].append((data, y, y - p))
        hist[b].append((data, 1 - y, (1 - y) - (1 - p)))
    return {t: {"forma": round(sum(x[2] for x in h), 4), "mapas": len(h), "ultimo": h[-1][0][:10],
                "resultados": "".join("V" if x[1] else "D" for x in h)} for t, h in hist.items()}


def main() -> int:
    jogos = _replay_dota() if E_DOTA else _replay_lol()
    times = calcular(jogos)
    doc = {"gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"), "janela": JANELA,
           "peso": PESO, "medicao": MEDICAO, "n_mapas": len(jogos), "times": times}
    destino = RAIZ / f"forma_recente_{ESPORTE}.json"
    destino.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"[forma_recente] {ESPORTE}: {len(times)} times, {len(jogos)} mapas -> {destino.name} "
          f"({destino.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
