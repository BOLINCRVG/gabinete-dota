# -*- coding: utf-8 -*-
"""WHR (Whole-History Rating, Coulom 2008) — fitter de produção (alavanca N).

Estima por MAP a TRAJETÓRIA de rating de cada time: random walk gaussiano
entre dias de jogo (w2 Elo²/dia) + likelihood Bradley-Terry, prior do primeiro
ponto = pontos SOS da liga doméstica. Newton com Hessiana tridiagonal por time
(coordenada), incremental + varreduras completas.

Validação (2026-07-06): probe prequencial (+0.0114 logloss geral vs Elo
produção+MOVDA) e caminho fiel com Platt (+0.59 nota harness, ECE melhora).
w2*=60 tunado prequencialmente em 2025. Ver tools/eval_whr*.py e
evidence/eval_whr*.log.

Deploy: train_offline --core whr usa este fitter e EXPORTA os ratings finais
para um artefato EloModel comum (ratings já descontados do prior de liga;
params alpha=0 / league_prior_weight=1.0 / scale=400 reconstroem exatamente
p = logistic10((R_a - R_b)/400) do protocolo validado).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

NAT = math.log(10.0) / 400.0   # pontos Elo -> unidades naturais do logit
SIGMA0_ELO = 350.0             # desvio do prior do 1º ponto (Elo)
MAX_STEP = 1.0                 # clamp do passo de Newton (natural)
DEFAULT_W2 = 60.0              # Elo²/dia (tunado 2025, eval cego 2026)


def _solve_tridiag(diag: np.ndarray, off: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = len(diag)
    if n == 1:
        return rhs / diag
    for i in range(1, n):
        m = off[i - 1] / diag[i - 1]
        diag[i] -= m * off[i - 1]
        rhs[i] -= m * rhs[i - 1]
    x = np.zeros(n)
    x[-1] = rhs[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (rhs[i] - off[i] * x[i + 1]) / diag[i]
    return x


class WHRFitter:
    """MAP do histórico completo; ratings em unidades naturais internamente."""

    def __init__(self, *, w2_elo_per_day: float = DEFAULT_W2,
                 league_prior_points: Optional[Dict[str, float]] = None):
        self.w2 = float(w2_elo_per_day) * NAT * NAT
        self.prior_mu = {str(k): float(v) * NAT for k, v in (league_prior_points or {}).items()}
        self.days: Dict[str, List[int]] = {}
        self.r: Dict[str, np.ndarray] = {}
        self.mu0: Dict[str, float] = {}
        self.games: Dict[str, List[Tuple[int, str, int, int]]] = {}
        self.n_games: Dict[str, int] = {}
        self.home_league: Dict[str, str] = {}

    # ------------------ estrutura ------------------
    def _ensure_point(self, team: str, day: int, league: str) -> int:
        if team not in self.days:
            self.days[team] = [day]
            mu = self.prior_mu.get(league, 0.0)
            self.r[team] = np.array([mu], dtype=float)
            self.mu0[team] = mu
            self.games[team] = []
            self.n_games[team] = 0
            self.home_league[team] = str(league or "")
            return 0
        if league:
            self.home_league[team] = str(league)
        ds = self.days[team]
        if day > ds[-1]:
            ds.append(day)
            self.r[team] = np.append(self.r[team], self.r[team][-1])
        return len(ds) - 1

    def add_game(self, blue: str, red: str, blue_win: int, day: int,
                 blue_league: str = "", red_league: str = "") -> None:
        ib = self._ensure_point(blue, day, blue_league)
        ir = self._ensure_point(red, day, red_league)
        y = 1 if int(blue_win) == 1 else 0
        self.games[blue].append((ib, red, ir, y))
        self.games[red].append((ir, blue, ib, 1 - y))
        self.n_games[blue] = self.n_games.get(blue, 0) + 1
        self.n_games[red] = self.n_games.get(red, 0) + 1

    # ------------------ otimização ------------------
    def refresh_team(self, team: str, iters: int = 2) -> float:
        """Newton no timeline do time (opponents fixos). Retorna |passo| máx."""
        ds = self.days.get(team)
        if not ds:
            return 0.0
        n = len(ds)
        r = self.r[team]
        s0 = (SIGMA0_ELO * NAT) ** 2
        max_dx = 0.0
        for _ in range(int(iters)):
            grad = np.zeros(n)
            diag = np.zeros(n)
            off = np.zeros(max(n - 1, 0))
            grad[0] += -(r[0] - self.mu0[team]) / s0
            diag[0] += -1.0 / s0
            for i in range(n - 1):
                var = self.w2 * max(ds[i + 1] - ds[i], 1)
                d = (r[i + 1] - r[i]) / var
                grad[i] += d
                grad[i + 1] -= d
                diag[i] -= 1.0 / var
                diag[i + 1] -= 1.0 / var
                off[i] += 1.0 / var
            for (idx, opp, oidx, y) in self.games[team]:
                ro = self.r[opp][min(oidx, len(self.r[opp]) - 1)]
                p = 1.0 / (1.0 + math.exp(-(r[idx] - ro)))
                grad[idx] += (y - p)
                diag[idx] -= p * (1.0 - p)
            dx = _solve_tridiag(diag.copy(), off.copy(), -grad)
            np.clip(dx, -MAX_STEP, MAX_STEP, out=dx)
            r += dx
            max_dx = max(max_dx, float(np.max(np.abs(dx))) if n else 0.0)
        self.r[team] = r
        return max_dx

    def refresh_pair(self, blue: str, red: str, iters: int = 2) -> None:
        self.refresh_team(blue, iters=iters)
        self.refresh_team(red, iters=iters)

    def full_sweeps(self, sweeps: int = 8, tol_elo: float = 0.5, iters: int = 1) -> int:
        """Varreduras completas até convergir (|passo| máx < tol, em Elo)."""
        tol_nat = float(tol_elo) * NAT
        done = 0
        for s in range(int(sweeps)):
            worst = 0.0
            for t in self.days:
                worst = max(worst, self.refresh_team(t, iters=iters))
            done = s + 1
            if worst < tol_nat:
                break
        return done

    # ------------------ leitura ------------------
    def p_blue(self, blue: str, red: str,
               blue_league: str = "", red_league: str = "") -> float:
        ra = self.rating_nat(blue, blue_league)
        rb = self.rating_nat(red, red_league)
        return 1.0 / (1.0 + math.exp(-(ra - rb)))

    def rating_nat(self, team: str, league: str = "") -> float:
        arr = self.r.get(team)
        if arr is not None and len(arr):
            return float(arr[-1])
        return self.prior_mu.get(str(league or self.home_league.get(team, "")), 0.0)

    def ratings_elo(self) -> Dict[str, float]:
        """{time -> rating final em pontos Elo (prior de liga INCLUSO)}."""
        return {t: float(arr[-1]) / NAT for t, arr in self.r.items() if len(arr)}
