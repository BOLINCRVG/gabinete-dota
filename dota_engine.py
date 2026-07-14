# -*- coding: utf-8 -*-
"""Motor de producao do Dota — config validada na Fase 1/2 (2026-07-08):

- WHR w2=20 Elo2/dia, treino com TODAS as ligas (excluded ON), eval/produto
  so tier professional. Reusa mlcore/whr_core do repo LoL (agnostico).
- Side Radiant: termo rolante prequencial (meia-vida 30d, clamp +-0.4 nat).
- Totais kills: media do confronto (shrink knee=8) blend 50/50 com ataque x
  defesa multiplicativo, + ajuste de mapa>=2 (delta pareado, ~+2.7 kills).
- Totais duracao: media do confronto (SEM ajuste de mapa — REJECT no gate).

Sem calibrador na v1 (Platt PIOROU no blind — raw ja nasce calibrado).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from dota_paths import ensure_lol_on_path

LOL_REPO = ensure_lol_on_path()

from mlcore.whr_core import WHRFitter  # noqa: E402
from eval_whr_dota import SideTracker, load_matches  # noqa: E402
from eval_totals_dota import Decayed, PairedDelta, shrink  # noqa: E402

W2_STAR = 40.0  # re-tune ano-cego 09/07 na base 44k (20-60 ~empatados; 40 otimo)
NAT = math.log(10.0) / 400.0
MIN_GAMES_PRODUCT = 5

# Mercados de totais do Dota (analogos aos do LoL):
#   kills (gateado c/ DC+M2) | duracao (gateado V0) |
#   torres/roshans/barracks: medias de confronto sobre partidas COM objetivos
#   (upgrade em andamento; gates rodam quando a base completar).
DOTA_MARKETS = {
    "kills": {"label": "Kills totais", "obj_only": False},
    "duracao": {"label": "Duracao (min)", "obj_only": False},
    "torres": {"label": "Torres totais", "obj_only": True},
    "roshans": {"label": "Roshans totais", "obj_only": True},
    "barracks": {"label": "Barracks totais", "obj_only": True},
}


def _row_has_objectives(r) -> bool:
    return str(getattr(r, "has_objectives", "")).strip().lower() in ("true", "1")


def _market_value(r, market: str) -> float:
    if market == "kills":
        return float(r.teamkills) + float(r.oppkills)
    if market == "duracao":
        return float(r.duration) / 60.0
    if market == "torres":
        return float(r.towers) + float(r.opp_towers)
    if market == "roshans":
        return float(r.roshans) + float(r.opp_roshans)
    if market == "barracks":
        return float(r.barracks) + float(r.opp_barracks)
    return float("nan")


@dataclass
class TeamInfo:
    team_id: int
    name: str = ""
    league: str = ""
    last_day: int = 0
    prof_games: int = 0
    rating_elo: float = 0.0


@dataclass
class DotaEngine:
    matches: pd.DataFrame = field(repr=False, default=None)
    fitter: WHRFitter = field(repr=False, default=None)
    side: SideTracker = field(repr=False, default=None)
    teams: dict = field(default_factory=dict)
    league_kills: Decayed = field(repr=False, default=None)
    league_dur: Decayed = field(repr=False, default=None)
    tk: dict = field(repr=False, default_factory=dict)
    tk_for: dict = field(repr=False, default_factory=dict)
    tk_against: dict = field(repr=False, default_factory=dict)
    td: dict = field(repr=False, default_factory=dict)
    tobj: dict = field(repr=False, default_factory=dict)      # {market: {team: Decayed}}
    league_obj: dict = field(repr=False, default_factory=dict)  # {market: Decayed}
    m2_kills_delta: float = 0.0
    today: int = 0

    # ---------- build ----------
    @classmethod
    def build(cls) -> "DotaEngine":
        m = load_matches()
        eng = cls(matches=m)
        eng.fitter = WHRFitter(w2_elo_per_day=W2_STAR)
        eng.side = SideTracker()
        eng.league_kills = Decayed()
        eng.league_dur = Decayed()
        d_k = PairedDelta()
        series_count: dict = {}
        series_first: dict = {}
        m = m.assign(pair_key=list(zip(m["dt"].dt.date,
                                       [tuple(sorted(t)) for t in zip(m["tid"], m["oid"])])))
        for r in m.itertuples(index=False):
            a, b = str(r.tid), str(r.oid)
            day = int(r.day)
            eng.fitter.add_game(a, b, int(r.win), day)
            is_prof = r.tier == "professional"
            kills = float(r.teamkills) + float(r.oppkills)
            dur = float(r.duration) / 60.0
            if is_prof and math.isfinite(kills) and math.isfinite(dur):
                eng.side.add(day, int(r.win))
                eng.league_kills.add(day, kills)
                eng.league_dur.add(day, dur)
                for team, kf, ka in ((a, float(r.teamkills), float(r.oppkills)),
                                     (b, float(r.oppkills), float(r.teamkills))):
                    eng.tk.setdefault(team, Decayed()).add(day, kills)
                    eng.tk_for.setdefault(team, Decayed()).add(day, kf)
                    eng.tk_against.setdefault(team, Decayed()).add(day, ka)
                    eng.td.setdefault(team, Decayed()).add(day, dur)
                if _row_has_objectives(r):
                    for mkt in ("torres", "roshans", "barracks"):
                        val = _market_value(r, mkt)
                        if math.isfinite(val):
                            eng.league_obj.setdefault(mkt, Decayed()).add(day, val)
                            store = eng.tobj.setdefault(mkt, {})
                            store.setdefault(a, Decayed()).add(day, val)
                            store.setdefault(b, Decayed()).add(day, val)
                mapno = series_count.get(r.pair_key, 0) + 1
                if mapno == 1:
                    series_first[r.pair_key] = kills
                elif r.pair_key in series_first:
                    d_k.add(kills - series_first[r.pair_key])
                series_count[r.pair_key] = mapno
            # info de time (nome mais recente, so prof conta p/ produto)
            for team_id, name, opp in ((int(r.tid), str(r.teamname), None),
                                       (int(r.oid), str(r.opp_teamname), None)):
                info = eng.teams.get(team_id) or TeamInfo(team_id=team_id)
                if day >= info.last_day:
                    info.last_day = day
                    if name and name.lower() not in ("radiant", "dire"):
                        info.name = name
                    if is_prof:
                        info.league = str(r.league)
                if is_prof:
                    info.prof_games += 1
                eng.teams[team_id] = info
        eng.m2_kills_delta = d_k.value()
        eng.today = int(m["day"].max())
        eng.fitter.full_sweeps(sweeps=30, tol_elo=0.5)
        for tid_, info in eng.teams.items():
            info.rating_elo = eng.fitter.rating_nat(str(tid_)) / NAT
        return eng

    # ---------- produto ----------
    def product_teams(self, min_games: int = MIN_GAMES_PRODUCT,
                      active_days: int = 90) -> list[TeamInfo]:
        cut = self.today - int(active_days)
        out = [t for t in self.teams.values()
               if t.prof_games >= min_games and t.last_day >= cut and t.name]
        return sorted(out, key=lambda t: -t.rating_elo)

    def side_offset(self) -> float:
        return self.side.offset_nat(self.today)

    # Fadiga-diferencial: ACCEPT em 07-08 (232 gatilhos, -6.6pp) foi DERRUBADO
    # no re-teste com a base 2025 (2.136 gatilhos, +0.4pp = zero) — era ruido
    # de amostra pequena. K=0 (sem ajuste na odd); a INFO de carga continua
    # exposta na UI. Re-testar se a base dobrar de novo.
    FATIGUE_K = 0.0
    FATIGUE_TRIG = 3

    def maps_last24h(self, team_id: int, ref_ts: float | None = None) -> int:
        if ref_ts is None:
            ref_ts = pd.Timestamp.utcnow().timestamp()
        m = self.matches
        mask = ((m["tid"] == team_id) | (m["oid"] == team_id)) \
            & (m["ord_key"] > ref_ts - 86400.0) & (m["ord_key"] < ref_ts)
        return int(mask.sum())

    def fatigue_info(self, a_id: int, b_id: int, ref_ts: float | None = None) -> dict:
        ma = self.maps_last24h(a_id, ref_ts)
        mb = self.maps_last24h(b_id, ref_ts)
        d24 = ma - mb
        adj = 0.0
        if self.FATIGUE_K and d24 >= self.FATIGUE_TRIG:
            adj = -self.FATIGUE_K   # A mais carregado -> contra A
        elif self.FATIGUE_K and d24 <= -self.FATIGUE_TRIG:
            adj = self.FATIGUE_K
        return {"maps_a": ma, "maps_b": mb, "d24": d24, "adj_nat": adj}

    def p_map(self, a_id: int, b_id: int, a_side: str | None = None,
              ref_ts: float | None = None) -> float:
        logit = self.fitter.rating_nat(str(a_id)) - self.fitter.rating_nat(str(b_id))
        off = self.side_offset()
        if a_side == "radiant":
            logit += off
        elif a_side == "dire":
            logit -= off
        logit += self.fatigue_info(a_id, b_id, ref_ts)["adj_nat"]
        return 1.0 / (1.0 + math.exp(-logit))

    def series_probs(self, a_id: int, b_id: int, bo: int = 3,
                     ref_ts: float | None = None) -> dict:
        p = self.p_map(a_id, b_id, ref_ts=ref_ts)  # lado neutro (series simetriza)
        if bo == 2:
            return {"2-0": p * p, "1-1": 2 * p * (1 - p), "0-2": (1 - p) ** 2}
        if bo == 3:
            pa = p * p * (3 - 2 * p)
            return {"A": pa, "B": 1 - pa}
        if bo == 5:
            pa = (p ** 3) * (10 - 15 * p + 6 * p * p)
            return {"A": pa, "B": 1 - pa}
        return {"A": p, "B": 1 - p}

    def kills_line(self, a_id: int, b_id: int, mapno: int = 1) -> float | None:
        lk = self.league_kills.mean(self.today)
        if lk is None:
            return None
        a, b = str(a_id), str(b_id)
        def _mean(store, team, prior):
            dec = store.get(team)
            if dec is None:
                return prior
            v = dec.mean(self.today)
            return shrink(v, prior, dec.eff_n()) if v is not None else prior
        v0 = 0.5 * (_mean(self.tk, a, lk) + _mean(self.tk, b, lk))
        half = lk / 2.0
        dc = (_mean(self.tk_for, a, half) * _mean(self.tk_against, b, half) / half
              + _mean(self.tk_for, b, half) * _mean(self.tk_against, a, half) / half)
        est = 0.5 * v0 + 0.5 * dc
        if mapno >= 2:
            est += self.m2_kills_delta
        return est

    def duration_line(self, a_id: int, b_id: int) -> float | None:
        ld = self.league_dur.mean(self.today)
        if ld is None:
            return None
        def _mean(team):
            dec = self.td.get(str(team))
            if dec is None:
                return ld
            v = dec.mean(self.today)
            return shrink(v, ld, dec.eff_n()) if v is not None else ld
        return 0.5 * (_mean(a_id) + _mean(b_id))

    # Gate 2026-07-09 (eval_objetivos_e_janelas, base 100% completa):
    # torres e roshans = MEDIA DA LIGA vence o confronto no blind (roshan e
    # quase aleatorio por jogo; torres tem pouco sinal de time); barracks =
    # media do confronto (V0) ganha. Ajuste de mapa 2+: ~zero p/ objetivos
    # (REJECT — o M1-under/M2-over e fenomeno de KILLS).
    OBJ_LEAGUE_ONLY = ("torres", "roshans")

    def objective_line(self, a_id: int, b_id: int, market: str) -> tuple[float, int] | None:
        """Linha p/ torres/roshans/barracks conforme o gate. Retorna (linha, n_eff)."""
        league = self.league_obj.get(market)
        if league is None:
            return None
        prior = league.mean(self.today)
        if prior is None:
            return None
        if market in self.OBJ_LEAGUE_ONLY:
            return prior, int(league.eff_n())
        store = self.tobj.get(market, {})
        vals, effs = [], []
        for tid_ in (a_id, b_id):
            dec = store.get(str(tid_))
            if dec is None:
                vals.append(prior)
                effs.append(0.0)
                continue
            v = dec.mean(self.today)
            vals.append(shrink(v, prior, dec.eff_n()) if v is not None else prior)
            effs.append(dec.eff_n())
        return 0.5 * (vals[0] + vals[1]), int(min(effs))

    def market_series(self, team_id: int, market: str, opp_id: int | None = None) -> pd.Series:
        """Serie historica (mais recente primeiro) do valor do mercado nos jogos
        professional do time (ou do H2H, se opp_id dado). Base das janelas
        Ano/L10/L5/H2H da Analise de Linha."""
        m = self.matches
        mask = (m["tier"] == "professional") & ((m["tid"] == team_id) | (m["oid"] == team_id))
        if opp_id is not None:
            mask = ((m["tid"] == team_id) & (m["oid"] == opp_id)) | \
                   ((m["tid"] == opp_id) & (m["oid"] == team_id))
        sub = m[mask].sort_values("ord_key", ascending=False)
        if DOTA_MARKETS.get(market, {}).get("obj_only"):
            sub = sub[sub["has_objectives"].astype(str).str.strip().str.lower().isin(["true", "1"])]
        vals = []
        for r in sub.itertuples(index=False):
            v = _market_value(r, market)
            if math.isfinite(v):
                vals.append(v)
        return pd.Series(vals, dtype=float)

    def team_recent(self, team_id: int, n: int = 10) -> pd.DataFrame:
        m = self.matches
        sub = m[((m["tid"] == team_id) | (m["oid"] == team_id))
                & (m["tier"] == "professional")].sort_values("ord_key", ascending=False).head(n)
        rows = []
        for r in sub.itertuples(index=False):
            we = int(r.tid) == int(team_id)
            rows.append({
                "data": r.dt.strftime("%d/%m"),
                "liga": str(r.league)[:28],
                "vs": str(r.opp_teamname if we else r.teamname)[:22],
                "res": "W" if (int(r.win) == 1) == we else "L",
                "kills_totais": int(float(r.teamkills) + float(r.oppkills)),
                "dur_min": round(float(r.duration) / 60.0, 1),
            })
        return pd.DataFrame(rows)

    def h2h(self, a_id: int, b_id: int, n: int = 10) -> pd.DataFrame:
        m = self.matches
        sub = m[(((m["tid"] == a_id) & (m["oid"] == b_id))
                 | ((m["tid"] == b_id) & (m["oid"] == a_id)))].sort_values("ord_key", ascending=False).head(n)
        rows = []
        for r in sub.itertuples(index=False):
            a_is_t = int(r.tid) == int(a_id)
            a_won = (int(r.win) == 1) == a_is_t
            rows.append({
                "data": r.dt.strftime("%d/%m/%y"),
                "liga": str(r.league)[:28],
                "vencedor": "A" if a_won else "B",
                "kills_totais": int(float(r.teamkills) + float(r.oppkills)),
                "dur_min": round(float(r.duration) / 60.0, 1),
            })
        return pd.DataFrame(rows)
