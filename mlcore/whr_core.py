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

MARGEM DE VITORIA (09/2026): a likelihood aceita resultado FRACIONARIO ỹ ∈ [0,1]
no lugar de y ∈ {0,1}. Gradiente (ỹ-p) e Hessiana -p(1-p) ficam intactos — a
Hessiana nao depende do resultado —, entao o Newton segue valido. Prova cega em
2026 (tools/eval_whr_mov.py, evidence/eval_whr_mov_*.log), com ouro aos 15 min
como sinal: logloss +0.0117, acerto 65.0%->65.7%, ganho em 9/9 meses e 28/34
ligas. DESLIGADO por default (mov_lambda=0).

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
                 league_prior_points: Optional[Dict[str, float]] = None,
                 mov_lambda: float = 0.0, mov_folga: float = 0.0,
                 mov_fit: Optional[Dict[str, float]] = None):
        # MARGEM DE VITORIA — resultado fracionario (ver docstring do modulo).
        # mov_lambda = 0.0 (default) deixa ỹ == y: o fitter fica BIT A BIT o
        # de antes de 09/2026. Nada muda em producao sem alguem ligar isto.
        self.mov_lambda = float(mov_lambda)
        self.mov_folga = float(mov_folga)
        self.mov_fit = dict(mov_fit or {})
        self.n_mov_usados = 0
        self.n_mov_faltando = 0
        self.w2 = float(w2_elo_per_day) * NAT * NAT
        self.prior_mu = {str(k): float(v) * NAT for k, v in (league_prior_points or {}).items()}
        self.days: Dict[str, List[int]] = {}
        self.r: Dict[str, np.ndarray] = {}
        self.mu0: Dict[str, float] = {}
        # (idx_do_time, oponente, idx_do_oponente, ỹ) — ỹ é float desde 09/2026
        self.games: Dict[str, List[Tuple[int, str, int, float]]] = {}
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

    def resultado_fracionario(self, blue: str, red: str, blue_win: int,
                              mov: Optional[float],
                              blue_league: str = "", red_league: str = "") -> float:
        """ỹ ∈ [0,1] na perspectiva do blue.

        ỹ = recorta( folga + y·(1-2·folga) + lam·(MOV - E[MOV|Δr]) / desvio )

        - Só a SURPRESA da margem entra: E[MOV|Δr] = a·tanh(b·Δr) + c. Margem
          crua confundiria "dominou porque é melhor" (já está no rating) com
          "dominou acima do esperado" (informação nova).
        - `folga` deixa a vitória normal valer 1-folga, para sobrar teto e o
          modelo conseguir PREMIAR domínio, não só descontar vitória ruim.
        - Sem MOV no jogo, cai no y binário: ausência de dado não vira número.
        """
        y = 1.0 if int(blue_win) == 1 else 0.0
        base = self.mov_folga + y * (1.0 - 2.0 * self.mov_folga)
        if self.mov_lambda == 0.0 or mov is None:
            if self.mov_lambda != 0.0:
                self.n_mov_faltando += 1
            return base
        try:
            m = float(mov)
        except (TypeError, ValueError):
            self.n_mov_faltando += 1
            return base
        if not math.isfinite(m):
            self.n_mov_faltando += 1
            return base
        f = self.mov_fit
        s = float(f.get("s", 0.0))
        if not (math.isfinite(s) and s > 0.0):
            self.n_mov_faltando += 1
            return base
        dr = (self.rating_nat(blue, blue_league) - self.rating_nat(red, red_league))
        esperado = (float(f.get("a", 0.0)) * math.tanh(float(f.get("b", 1.0)) * dr)
                    + float(f.get("c", 0.0)))
        self.n_mov_usados += 1
        return min(max(base + self.mov_lambda * (m - esperado) / s, 0.0), 1.0)

    def add_game(self, blue: str, red: str, blue_win: int, day: int,
                 blue_league: str = "", red_league: str = "",
                 mov: Optional[float] = None) -> None:
        # ỹ tem que sair ANTES de _ensure_point: ele usa o Δr de agora, que é
        # o que o modelo sabia antes deste jogo.
        yf = self.resultado_fracionario(blue, red, blue_win, mov,
                                        blue_league, red_league)
        ib = self._ensure_point(blue, day, blue_league)
        ir = self._ensure_point(red, day, red_league)
        # O oponente recebe 1-ỹ: a simetria sai por construção, não por uma
        # segunda conta que poderia divergir da primeira.
        self.games[blue].append((ib, red, ir, yf))
        self.games[red].append((ir, blue, ib, 1.0 - yf))
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


def ajustar_expectativa_mov(dr_nat: List[float], movs: List[float]) -> Dict[str, float]:
    """Ajusta E[MOV|Δr] = a·tanh(b·Δr) + c por mínimos quadrados (grid em b).

    Devolve também `s` = desvio do resíduo, que é o que normaliza a surpresa:
    sem ele, `lam` teria um significado diferente para cada sinal de margem
    (kills anda em unidades de ~10, ouro aos 15 em unidades de ~2500).

    A tanh não é enfeite: ela SATURA, então uma margem absurda não vira um
    ajuste absurdo. É o que trata placar inflado no fim de jogo.
    Fonte da forma: MOVDA, https://arxiv.org/html/2506.00348 (aberta 08/09/2026).
    """
    X = np.asarray(dr_nat, dtype=float)
    Y = np.asarray(movs, dtype=float)
    ok = np.isfinite(X) & np.isfinite(Y)
    X, Y = X[ok], Y[ok]
    if len(X) < 200:
        raise ValueError(f"amostra insuficiente para ajustar E[MOV]: n={len(X)}")
    melhor: Optional[Dict[str, float]] = None
    for b in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        T = np.tanh(b * X)
        A = np.column_stack([T, np.ones_like(T)])
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        resid = Y - A @ coef
        sse = float(resid @ resid)
        if melhor is None or sse < melhor["sse"]:
            melhor = {"a": float(coef[0]), "b": float(b), "c": float(coef[1]),
                      "s": float(resid.std(ddof=2)), "sse": sse, "n": int(len(X))}
    assert melhor is not None
    return melhor
