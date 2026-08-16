from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math

from .tiers import (
    canonical_league,
    tier_for_league,
    enforce_monotone_caps,
    tier_base_points,
    league_base_points,
)


def _p_from_diff_points(diff_points: float, scale_points: float) -> float:
    # classic Elo base-10 logistic
    try:
        return float(1.0 / (1.0 + 10.0 ** (-float(diff_points) / float(scale_points))))
    except Exception:
        return 0.5


def uncertainty_scale_mult(n_blue: float, n_red: float, c: float, n_prior: float) -> float:
    """Quanto a escala do Elo deve inflar pela incerteza dos dois times (g(RD) do
    Glicko). 1.0 = sem efeito. Ver EloParams.unc_scale_c."""
    c = float(c or 0.0)
    n0 = float(n_prior or 0.0)
    if c <= 0.0 or n0 <= 0.0:
        return 1.0
    ub = n0 / (n0 + max(0.0, float(n_blue)))
    ur = n0 / (n0 + max(0.0, float(n_red)))
    return float(math.sqrt(1.0 + c * (ub + ur)))


def flatten_p_by_uncertainty(p: float, n_blue: float, n_red: float,
                             c: float, n_prior: float) -> float:
    """Achata uma probabilidade JA PRONTA pelo mesmo fator de incerteza.

    Existe porque o core WHR fita o Platt nas probabilidades do proprio WHR, nao
    nas do EloModel: sem aplicar o achatamento tambem ali, o calibrador nao veria
    o efeito e a parte global viraria vies em vez de ser reabsorvida."""
    m = uncertainty_scale_mult(n_blue, n_red, c, n_prior)
    if m == 1.0:
        return float(p)
    q = min(max(float(p), 1e-9), 1.0 - 1e-9)
    return float(1.0 / (1.0 + 10.0 ** (-math.log10(q / (1.0 - q)) / m)))


@dataclass
class EloParams:
    # Team residual Elo update magnitude
    k: float = 26.0
    # Learned offsets (only updated on inter-league / inter-tier games)
    k_league: float = 6.0
    k_tier: float = 3.0
    # Optional side advantage learning (blue-side advantage in points)
    k_side: float = 0.0

    # Elo scale denominator
    scale_points: float = 400.0
    init_rating: float = 0.0

    # Shrink & stability knobs (data-tuned in train_offline)
    # - shrink factor for inference: n/(n+alpha)
    # - k-decay during training: alpha/(alpha+n)
    alpha: float = 80.0

    # Per-update weight decay (pulls parameters toward 0)
    l2_team: float = 0.0002
    l2_league: float = 0.0005
    l2_tier: float = 0.0005
    l2_side: float = 0.0

    # Data-driven caps for GLOBAL ranking only (not applied to ML probability)
    rank_cap_percentile: float = 99.5
    rank_caps_min_teams: int = 20

    # Structural priors (tier/league baselines from mlcore.tiers).
    # 0.0 disables priors; >0.0 enforces stronger inter-league hierarchy.
    tier_prior_weight: float = 0.20
    league_prior_weight: float = 0.10

    # INCERTEZA -> PROBABILIDADE MAIS CHATA (forma do Glicko; achado 16/08/2026).
    # O residual ja encolhe o RATING de quem tem poucos jogos (n/(n+alpha)), mas a
    # PROBABILIDADE continuava confiante: time novo (rating ~0) contra um top vira
    # "o top ganha facil". Medido no TI 2026: quando o mais fraco tem <50 jogos o
    # logloss e 0.984 (PIOR que moeda) contra 0.657 quando o modelo conhece os dois.
    # Aqui a escala do Elo e inflada pela incerteza dos DOIS lados, que e o
    # g(RD) do Glicko: menos jogo -> curva mais achatada -> p mais perto de 50%.
    # 0.0 = desligado e identico ao comportamento legado (default seguro).
    unc_scale_c: float = 0.0
    # Quantos jogos um time precisa para ser considerado "conhecido":
    # u = unc_n_prior/(unc_n_prior + n). NAO reusa o alpha de proposito — no
    # core WHR (default desde 07/07) alpha e zerado para reproduzir o protocolo
    # validado, e amarrar a incerteza nele deixava esta camada morta.
    unc_n_prior: float = 50.0
    # Hard safety caps to avoid disconnected-league drift dominating probabilities.
    league_strength_cap_abs: float = 45.0
    # Multipliers by tier to avoid weaker ecosystems drifting too far.
    league_strength_cap_mult_t1: float = 1.00
    league_strength_cap_mult_t2: float = 0.70
    league_strength_cap_mult_erl: float = 0.45
    league_strength_cap_mult_minor: float = 0.30
    league_strength_cap_mult_other: float = 0.40
    tier_strength_cap_abs: float = 30.0

    # MOVDA (arXiv 2506.00348): termo de margem-de-vitoria no update.
    # R' = R + K(S-E) + mov_lambda*(MOV_real - E_MOV), com
    # E_MOV = mov_alpha*tanh(mov_beta*dR) + mov_gamma (dR = diff implicito na p prevista).
    # mov_lambda=0.0 (default) desliga por completo. Constantes fitadas em 2024-25
    # (tools/eval_mov_elo.py, n=20252, MOV=diff de kills, resid_std=11.8 kills).
    mov_lambda: float = 0.0
    mov_alpha: float = 9.343
    mov_beta: float = 0.005
    mov_gamma: float = -0.056

    # Phase-1 scaffold (feature-flagged): dynamic K adjustment by uncertainty/volatility.
    # Disabled by default to preserve current behavior.
    dynamic_k_enabled: bool = False
    dynamic_k_games_prior: float = 30.0
    dynamic_k_uncertainty_mult_max: float = 1.80
    dynamic_k_volatility_ref: float = 0.30
    dynamic_k_volatility_mult_min: float = 0.80
    dynamic_k_volatility_mult_max: float = 1.35
    volatility_ewma_beta: float = 0.08


class EloModel:
    """Map-level ML model with *learned* league/tier strengths (no hardcoded baselines).

    Total (base) strength for team T:
      base(T) = residual_eff(T) + league_strength(league(T)) + tier_strength(tier(T))

    Where:
      residual_eff(T) = residual_raw(T) * n_games(T)/(n_games(T)+alpha)

    Notes:
    - league_strength/tier_strength are updated ONLY on games where the two sides are from
      different leagues/tiers (inter-league / inter-tier). Domestic games do not affect them.
    - Optional blue-side advantage (side_adv_points) can also be learned.

    Ranking caps:
    - For safety against disconnected graphs (no interleague signal), we compute
      data-driven caps per tier (percentile of base strengths) and apply them ONLY
      to the GLOBAL ranking (never to match probabilities).
    """

    def __init__(self, params: Optional[EloParams] = None):
        self.params = params or EloParams()
        self.ratings: Dict[str, float] = {}
        self.games_played: Dict[str, int] = {}
        self.team_volatility: Dict[str, float] = {}

        self.league_strength: Dict[str, float] = {}
        self.tier_strength: Dict[str, float] = {}
        self.side_adv_points: float = 0.0  # blue-side advantage in points

        # Data-derived league base points (SOS, fit-time). When a league is here
        # it OVERRIDES the global static prior from mlcore.tiers — this keeps the
        # league prior consistent with the learned offset (both fit together) and
        # auto-updates with data. Empty = use the global static priors (legacy).
        self.league_base_points_map: Dict[str, float] = {}

        # Stable mapping inferred during training (home league/tier)
        self.team_meta: Dict[str, Dict[str, str]] = {}

        # Caps (tier -> max base strength) for ranking only
        self.rank_caps_by_tier: Dict[str, float] = {}

        # Counts (useful for reporting)
        self._league_games: Dict[str, int] = {}
        self._tier_games: Dict[str, int] = {}

    # ------------------ helpers ------------------
    def rating_raw(self, team: str) -> float:
        if team not in self.ratings:
            self.ratings[team] = float(self.params.init_rating)
        return float(self.ratings[team])

    def n_games(self, team: str) -> int:
        if team not in self.games_played:
            self.games_played[team] = 0
        return int(self.games_played[team])

    def meta(self, team: str) -> Dict[str, str]:
        if team in self.team_meta:
            return self.team_meta[team]
        return {"league": "OTHER", "tier": "OTHER"}

    def _shrink_factor(self, n: int) -> float:
        a = float(self.params.alpha)
        if a <= 0:
            return 1.0
        return float(n / (n + a))

    def residual_eff(self, team: str) -> float:
        n = self.n_games(team)
        return self.rating_raw(team) * self._shrink_factor(n)

    def _league_base_points(self, league: str) -> float:
        """Data-derived per-model league base points if present; else global static."""
        if league in self.league_base_points_map:
            return float(self.league_base_points_map[league])
        return float(league_base_points(league))

    def _k_decay(self, n: int) -> float:
        a = float(self.params.alpha)
        if a <= 0:
            return 1.0
        return float(a / (a + n))

    def _get_league(self, team: str, league: Optional[str]) -> str:
        if league is not None:
            z = canonical_league(str(league))
            return z if z else "OTHER"
        z = canonical_league(str(self.meta(team).get("league", "OTHER")))
        return z if z else "OTHER"

    def _get_tier(self, team: str, tier: Optional[str], league: Optional[str]) -> str:
        if tier is not None and str(tier).strip():
            return str(tier)
        # try from meta, else mapping from league
        m = self.meta(team)
        t = str(m.get("tier") or "").strip()
        if t:
            return t
        return str(tier_for_league(league or m.get("league")))

    def _league_strength(self, league: str) -> float:
        if league not in self.league_strength:
            self.league_strength[league] = 0.0
        cap = float(max(0.0, float(self._league_strength_cap_for_league(league))))
        v = float(self.league_strength[league])
        if cap <= 0:
            return v
        return float(max(-cap, min(cap, v)))

    def _league_strength_cap_for_league(self, league: str) -> float:
        base = float(max(0.0, float(self.params.league_strength_cap_abs)))
        if base <= 0.0:
            return 0.0
        tr = str(tier_for_league(league))
        if tr == "T1":
            mult = float(self.params.league_strength_cap_mult_t1)
        elif tr == "T2":
            mult = float(self.params.league_strength_cap_mult_t2)
        elif tr == "ERL":
            mult = float(self.params.league_strength_cap_mult_erl)
        elif tr == "MINOR":
            mult = float(self.params.league_strength_cap_mult_minor)
        else:
            mult = float(self.params.league_strength_cap_mult_other)
        mult = float(max(0.0, mult))
        return float(base * mult)

    def _tier_strength(self, tier: str) -> float:
        if tier not in self.tier_strength:
            self.tier_strength[tier] = 0.0
        cap = float(max(0.0, float(self.params.tier_strength_cap_abs)))
        v = float(self.tier_strength[tier])
        if cap <= 0:
            return v
        return float(max(-cap, min(cap, v)))

    def _team_vol(self, team: str) -> float:
        ref = float(max(1e-6, float(self.params.dynamic_k_volatility_ref)))
        if team not in self.team_volatility:
            self.team_volatility[team] = ref
        try:
            v = float(self.team_volatility.get(team, ref))
            return float(max(1e-6, v))
        except Exception:
            return ref

    def _dynamic_k_mult(self, blue: str, red: str) -> float:
        if not bool(self.params.dynamic_k_enabled):
            return 1.0
        n_min = float(min(self.n_games(blue), self.n_games(red)))
        g_prior = float(max(1.0, float(self.params.dynamic_k_games_prior)))
        unc = float(g_prior / (g_prior + max(0.0, n_min)))
        unc_max = float(max(1.0, float(self.params.dynamic_k_uncertainty_mult_max)))
        unc_mult = float(1.0 + unc * (unc_max - 1.0))

        v_ref = float(max(1e-6, float(self.params.dynamic_k_volatility_ref)))
        v_pair = 0.5 * (self._team_vol(blue) + self._team_vol(red))
        v_ratio = float(v_pair / v_ref)
        v_min = float(max(0.10, float(self.params.dynamic_k_volatility_mult_min)))
        v_max = float(max(v_min, float(self.params.dynamic_k_volatility_mult_max)))
        vol_mult = float(max(v_min, min(v_max, v_ratio)))

        out = float(unc_mult * vol_mult)
        return float(max(0.50, min(2.50, out)))

    def _update_team_volatility(self, team: str, abs_err: float) -> None:
        old = self._team_vol(team)
        beta = float(max(0.001, min(1.0, float(self.params.volatility_ewma_beta))))
        ae = float(max(0.0, min(1.0, float(abs_err))))
        self.team_volatility[team] = float(((1.0 - beta) * old) + (beta * ae))

    # ------------------ inference ------------------
    def base_strength(
        self,
        team: str,
        *,
        league: Optional[str] = None,
        tier: Optional[str] = None,
        apply_rank_caps: bool = False,
    ) -> float:
        lg = self._get_league(team, league)
        tr = self._get_tier(team, tier, lg)
        # Learned components + structural priors from league/tier baselines.
        prior_tier = float(self.params.tier_prior_weight) * float(tier_base_points(tr))
        prior_league = float(self.params.league_prior_weight) * float(self._league_base_points(lg))
        base = float(
            self.residual_eff(team)
            + self._league_strength(lg)
            + self._tier_strength(tr)
            + prior_tier
            + prior_league
        )
        if apply_rank_caps and self.rank_caps_by_tier:
            cap = self.rank_caps_by_tier.get(tr)
            if cap is not None:
                base = min(base, float(cap))
        return base

    def total_strength(
        self,
        team: str,
        *,
        league: Optional[str] = None,
        tier: Optional[str] = None,
        apply_rank_caps: bool = False,
    ) -> float:
        # For now, total_strength is the base strength (achievements live outside this model).
        return self.base_strength(team, league=league, tier=tier, apply_rank_caps=apply_rank_caps)

    def components(
        self,
        team: str,
        *,
        league: Optional[str] = None,
        tier: Optional[str] = None,
        apply_rank_caps: bool = False,
    ) -> Dict[str, Any]:
        lg = self._get_league(team, league)
        tr = self._get_tier(team, tier, lg)
        raw = float(self.rating_raw(team))
        n = int(self.n_games(team))
        shrink = float(self._shrink_factor(n))
        eff = float(raw * shrink)
        ls = float(self._league_strength(lg))
        ts = float(self._tier_strength(tr))
        prior_tier = float(self.params.tier_prior_weight) * float(tier_base_points(tr))
        prior_league = float(self.params.league_prior_weight) * float(self._league_base_points(lg))
        base_uncapped = float(eff + ls + ts + prior_tier + prior_league)
        cap = None
        if self.rank_caps_by_tier:
            cap = self.rank_caps_by_tier.get(tr)
        base = base_uncapped
        if apply_rank_caps and cap is not None:
            base = min(base, float(cap))
        return {
            "team": team,
            "league": lg,
            "tier": tr,
            "residual_raw": raw,
            "games_played": n,
            "shrink_factor": shrink,
            "residual_eff": eff,
            "league_strength": ls,
            "tier_strength": ts,
            "tier_prior_points": prior_tier,
            "league_prior_points": prior_league,
            "side_adv_points": float(self.side_adv_points),
            "base_strength_uncapped": base_uncapped,
            "base_strength": base,
            "rank_cap": cap,
        }

    def p_blue(
        self,
        blue: str,
        red: str,
        *,
        blue_league: Optional[str] = None,
        red_league: Optional[str] = None,
        blue_tier: Optional[str] = None,
        red_tier: Optional[str] = None,
    ) -> float:
        sb = self.base_strength(blue, league=blue_league, tier=blue_tier, apply_rank_caps=False)
        sr = self.base_strength(red, league=red_league, tier=red_tier, apply_rank_caps=False)
        diff = float(sb - sr + float(self.side_adv_points))
        return _p_from_diff_points(diff, self._scale_for(blue, red))

    def _scale_for(self, blue: str, red: str) -> float:
        """Escala do Elo inflada pela incerteza dos dois times (ver unc_scale_c).

        u = n0/(n0+n) e "o quanto eu NAO sei" do time: 1 sem jogo nenhum, ~0 com
        muito jogo. Escala maior => curva mais achatada => p mais perto de 50%
        quando o modelo nao tem base pra cravar."""
        return float(self.params.scale_points) * uncertainty_scale_mult(
            self.n_games(blue), self.n_games(red),
            getattr(self.params, "unc_scale_c", 0.0),
            getattr(self.params, "unc_n_prior", 50.0),
        )

    # ------------------ training ------------------
    def update(
        self,
        blue: str,
        red: str,
        blue_win: int,
        *,
        blue_league: Optional[str] = None,
        red_league: Optional[str] = None,
        blue_tier: Optional[str] = None,
        red_tier: Optional[str] = None,
        mov: Optional[float] = None,
    ) -> float:
        """One SGD-like update.

        League/tier strengths are updated only for inter-league / inter-tier games.
        `mov` (margem de vitoria do blue, ex.: diff de kills) so tem efeito quando
        params.mov_lambda != 0 — caso contrario o comportamento e identico ao legado.
        """
        bl = self._get_league(blue, blue_league)
        rl = self._get_league(red, red_league)
        bt = self._get_tier(blue, blue_tier, bl)
        rt = self._get_tier(red, red_tier, rl)

        # Ensure keys exist
        self._league_strength(bl); self._league_strength(rl)
        self._tier_strength(bt); self._tier_strength(rt)

        p = self.p_blue(blue, red, blue_league=bl, red_league=rl, blue_tier=bt, red_tier=rt)
        s = 1.0 if int(blue_win) == 1 else 0.0
        err = float(s - p)

        nb = self.n_games(blue)
        nr = self.n_games(red)
        decay = 0.5 * (self._k_decay(nb) + self._k_decay(nr))

        # Team residuals
        k_team = float(self.params.k) * float(decay) * float(self._dynamic_k_mult(blue, red))
        d_team = k_team * err

        self.ratings[blue] = self.rating_raw(blue) + d_team
        self.ratings[red] = self.rating_raw(red) - d_team

        # MOVDA: surpresa de margem ajusta o rating (zero-sum), igual ao validado
        # em tools/eval_mov_elo.py (sem decay no termo, lambda pequeno).
        if float(self.params.mov_lambda) != 0.0 and mov is not None:
            try:
                _mov_f = float(mov)
            except (TypeError, ValueError):
                _mov_f = float("nan")
            if math.isfinite(_mov_f):
                _p_c = min(max(float(p), 1e-6), 1.0 - 1e-6)
                _dr = float(self.params.scale_points) * math.log10(_p_c / (1.0 - _p_c))
                _e_mov = (
                    float(self.params.mov_alpha) * math.tanh(float(self.params.mov_beta) * _dr)
                    + float(self.params.mov_gamma)
                )
                _adj = float(self.params.mov_lambda) * (_mov_f - _e_mov)
                self.ratings[blue] = self.rating_raw(blue) + _adj
                self.ratings[red] = self.rating_raw(red) - _adj

        # League offsets (only on interleague)
        if bl != rl and float(self.params.k_league) != 0.0:
            d_league = float(self.params.k_league) * err
            self.league_strength[bl] = self._league_strength(bl) + d_league
            self.league_strength[rl] = self._league_strength(rl) - d_league
            self._league_games[bl] = self._league_games.get(bl, 0) + 1
            self._league_games[rl] = self._league_games.get(rl, 0) + 1

        # Tier offsets (only on intertier)
        if bt != rt and float(self.params.k_tier) != 0.0:
            d_tier = float(self.params.k_tier) * err
            self.tier_strength[bt] = self._tier_strength(bt) + d_tier
            self.tier_strength[rt] = self._tier_strength(rt) - d_tier
            self._tier_games[bt] = self._tier_games.get(bt, 0) + 1
            self._tier_games[rt] = self._tier_games.get(rt, 0) + 1

        # Side advantage (optional)
        if float(self.params.k_side) != 0.0:
            self.side_adv_points = float(self.side_adv_points + float(self.params.k_side) * err)

        # Weight decay (L2)
        if float(self.params.l2_team) != 0.0:
            l2 = float(self.params.l2_team)
            self.ratings[blue] = float(self.ratings[blue] - l2 * self.ratings[blue])
            self.ratings[red] = float(self.ratings[red] - l2 * self.ratings[red])
        if float(self.params.l2_league) != 0.0 and bl != rl:
            l2 = float(self.params.l2_league)
            self.league_strength[bl] = float(self.league_strength[bl] - l2 * self.league_strength[bl])
            self.league_strength[rl] = float(self.league_strength[rl] - l2 * self.league_strength[rl])
        if float(self.params.l2_tier) != 0.0 and bt != rt:
            l2 = float(self.params.l2_tier)
            self.tier_strength[bt] = float(self.tier_strength[bt] - l2 * self.tier_strength[bt])
            self.tier_strength[rt] = float(self.tier_strength[rt] - l2 * self.tier_strength[rt])
        if float(self.params.l2_side) != 0.0:
            l2 = float(self.params.l2_side)
            self.side_adv_points = float(self.side_adv_points - l2 * self.side_adv_points)

        # Keep offset components bounded even under sparse/disconnected interleague graphs.
        _lcap_b = float(max(0.0, float(self.params.league_strength_cap_abs)))
        _tcap = float(max(0.0, float(self.params.tier_strength_cap_abs)))
        if _lcap_b > 0.0:
            _lcap_bl = float(self._league_strength_cap_for_league(bl))
            _lcap_rl = float(self._league_strength_cap_for_league(rl))
            self.league_strength[bl] = float(max(-_lcap_bl, min(_lcap_bl, float(self.league_strength.get(bl, 0.0)))))
            self.league_strength[rl] = float(max(-_lcap_rl, min(_lcap_rl, float(self.league_strength.get(rl, 0.0)))))
        if _tcap > 0.0:
            self.tier_strength[bt] = float(max(-_tcap, min(_tcap, float(self.tier_strength.get(bt, 0.0)))))
            self.tier_strength[rt] = float(max(-_tcap, min(_tcap, float(self.tier_strength.get(rt, 0.0)))))

        # Increment games played
        self._update_team_volatility(blue, abs(err))
        self._update_team_volatility(red, abs(err))
        self.games_played[blue] = nb + 1
        self.games_played[red] = nr + 1

        return float(p)

    def fit(self, games_df) -> None:
        """Fit sequentially.

        Expected columns:
          - always: blue, red, blue_win
          - optional: blue_league, red_league, blue_tier, red_tier
          - legacy optional: league, tier  (applied to both teams)
        """
        for row in games_df.itertuples(index=False):
            blue = str(row.blue)
            red = str(row.red)
            y = int(row.blue_win)

            bl = getattr(row, "blue_league", None)
            rl = getattr(row, "red_league", None)
            bt = getattr(row, "blue_tier", None)
            rt = getattr(row, "red_tier", None)

            # legacy fallbacks
            lg = getattr(row, "league", None)
            tg = getattr(row, "tier", None)
            if bl is None and lg is not None:
                bl = lg
            if rl is None and lg is not None:
                rl = lg
            if bt is None and tg is not None:
                bt = tg
            if rt is None and tg is not None:
                rt = tg

            self.update(blue, red, y, blue_league=bl, red_league=rl, blue_tier=bt, red_tier=rt)

    def set_team_meta(self, team_meta: Dict[str, Dict[str, str]]) -> None:
        self.team_meta = {str(t): {"league": str((m or {}).get("league", "OTHER")), "tier": str((m or {}).get("tier", "OTHER"))} for t, m in (team_meta or {}).items()}

    def _center_offsets(self) -> None:
        # Center league/tier offsets to mean ~0 for interpretability (does not affect predictions)
        if self.league_strength:
            vals = list(self.league_strength.values())
            mu = float(sum(vals) / max(1, len(vals)))
            for k in list(self.league_strength.keys()):
                v = float(self.league_strength[k] - mu)
                cap = float(max(0.0, float(self._league_strength_cap_for_league(k))))
                if cap > 0.0:
                    v = float(max(-cap, min(cap, v)))
                self.league_strength[k] = v
        if self.tier_strength:
            vals = list(self.tier_strength.values())
            mu = float(sum(vals) / max(1, len(vals)))
            cap = float(max(0.0, float(self.params.tier_strength_cap_abs)))
            for k in list(self.tier_strength.keys()):
                v = float(self.tier_strength[k] - mu)
                if cap > 0.0:
                    v = float(max(-cap, min(cap, v)))
                self.tier_strength[k] = v

    def compute_rank_caps(self, team_meta: Dict[str, Dict[str, str]]) -> Dict[str, float]:
        """Compute data-driven caps per tier for ranking only.

        Steps:
        1) For each tier, compute a high percentile of base strengths.
        2) If a tier has too few teams, fall back to a conservative cap (max).
        3) Enforce monotonicity across tiers (T1 >= T2 >= ERL >= ...).
        """
        try:
            import numpy as np  # type: ignore
        except Exception:
            np = None  # type: ignore

        caps: Dict[str, float] = {}
        tiers: Dict[str, list[float]] = {}
        for team in (self.ratings or {}).keys():
            m = team_meta.get(team) or {}
            tr = str(m.get("tier") or "OTHER")
            lg = str(m.get("league") or "OTHER")
            s = float(self.base_strength(team, league=lg, tier=tr, apply_rank_caps=False))
            tiers.setdefault(tr, []).append(s)

        min_n = int(self.params.rank_caps_min_teams)
        for tr, vals in tiers.items():
            if len(vals) >= min_n:
                if np is not None:
                    q = float(np.percentile(np.array(vals, dtype=float), float(self.params.rank_cap_percentile)))
                else:
                    # fallback: mean + 3 std
                    mu = sum(vals) / len(vals)
                    var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)
                    q = float(mu + 3.0 * math.sqrt(var))
                caps[tr] = float(q)
            elif len(vals) >= 5:
                # Too few teams for a percentile cap, but still prevent absurd outliers.
                caps[tr] = float(max(vals))

        # Enforce tier order monotonicity (ranking-only guardrail)
        return enforce_monotone_caps(caps)

    # ------------------ serialization ------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": {
                # legacy key kept: k
                "k": float(self.params.k),
                "k_league": float(self.params.k_league),
                "k_tier": float(self.params.k_tier),
                "k_side": float(self.params.k_side),
                "scale_points": float(self.params.scale_points),
                "init_rating": float(self.params.init_rating),
                "alpha": float(self.params.alpha),
                "l2_team": float(self.params.l2_team),
                "l2_league": float(self.params.l2_league),
                "l2_tier": float(self.params.l2_tier),
                "l2_side": float(self.params.l2_side),
                "rank_cap_percentile": float(self.params.rank_cap_percentile),
                "rank_caps_min_teams": int(self.params.rank_caps_min_teams),
                "tier_prior_weight": float(self.params.tier_prior_weight),
                "league_prior_weight": float(self.params.league_prior_weight),
                "unc_scale_c": float(self.params.unc_scale_c),
                "unc_n_prior": float(self.params.unc_n_prior),
                "league_strength_cap_abs": float(self.params.league_strength_cap_abs),
                "league_strength_cap_mult_t1": float(self.params.league_strength_cap_mult_t1),
                "league_strength_cap_mult_t2": float(self.params.league_strength_cap_mult_t2),
                "league_strength_cap_mult_erl": float(self.params.league_strength_cap_mult_erl),
                "league_strength_cap_mult_minor": float(self.params.league_strength_cap_mult_minor),
                "league_strength_cap_mult_other": float(self.params.league_strength_cap_mult_other),
                "tier_strength_cap_abs": float(self.params.tier_strength_cap_abs),
                "dynamic_k_enabled": bool(self.params.dynamic_k_enabled),
                "dynamic_k_games_prior": float(self.params.dynamic_k_games_prior),
                "dynamic_k_uncertainty_mult_max": float(self.params.dynamic_k_uncertainty_mult_max),
                "dynamic_k_volatility_ref": float(self.params.dynamic_k_volatility_ref),
                "dynamic_k_volatility_mult_min": float(self.params.dynamic_k_volatility_mult_min),
                "dynamic_k_volatility_mult_max": float(self.params.dynamic_k_volatility_mult_max),
                "volatility_ewma_beta": float(self.params.volatility_ewma_beta),
                "mov_lambda": float(self.params.mov_lambda),
                "mov_alpha": float(self.params.mov_alpha),
                "mov_beta": float(self.params.mov_beta),
                "mov_gamma": float(self.params.mov_gamma),
            },
            "ratings": self.ratings,
            "games_played": self.games_played,
            "team_volatility": self.team_volatility,
            "league_strength": self.league_strength,
            "tier_strength": self.tier_strength,
            "side_adv_points": float(self.side_adv_points),
            "team_meta": self.team_meta,
            "rank_caps_by_tier": self.rank_caps_by_tier,
            "league_base_points_map": self.league_base_points_map,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EloModel":
        p = dict(d.get("params") or {})
        def _as_bool(v: Any, default: bool = False) -> bool:
            if isinstance(v, bool):
                return bool(v)
            s = str(v or "").strip().lower()
            if s in {"1", "true", "yes", "on"}:
                return True
            if s in {"0", "false", "no", "off"}:
                return False
            return bool(default)
        # Back-compat: old artifacts used params.k only
        model = cls(EloParams(
            k=float(p.get("k", 26.0)),
            k_league=float(p.get("k_league", 6.0)),
            k_tier=float(p.get("k_tier", 3.0)),
            k_side=float(p.get("k_side", 0.0)),
            scale_points=float(p.get("scale_points", 400.0)),
            init_rating=float(p.get("init_rating", 0.0)),
            alpha=float(p.get("alpha", 80.0)),
            l2_team=float(p.get("l2_team", 0.0002)),
            l2_league=float(p.get("l2_league", 0.0005)),
            l2_tier=float(p.get("l2_tier", 0.0005)),
            l2_side=float(p.get("l2_side", 0.0)),
            rank_cap_percentile=float(p.get("rank_cap_percentile", 99.5)),
            rank_caps_min_teams=int(p.get("rank_caps_min_teams", 20)),
            tier_prior_weight=float(p.get("tier_prior_weight", 0.20)),
            league_prior_weight=float(p.get("league_prior_weight", 0.10)),
            unc_scale_c=float(p.get("unc_scale_c", 0.0)),
            unc_n_prior=float(p.get("unc_n_prior", 50.0)),
            league_strength_cap_abs=float(p.get("league_strength_cap_abs", 45.0)),
            league_strength_cap_mult_t1=float(p.get("league_strength_cap_mult_t1", 1.00)),
            league_strength_cap_mult_t2=float(p.get("league_strength_cap_mult_t2", 0.70)),
            league_strength_cap_mult_erl=float(p.get("league_strength_cap_mult_erl", 0.45)),
            league_strength_cap_mult_minor=float(p.get("league_strength_cap_mult_minor", 0.30)),
            league_strength_cap_mult_other=float(p.get("league_strength_cap_mult_other", 0.40)),
            tier_strength_cap_abs=float(p.get("tier_strength_cap_abs", 30.0)),
            dynamic_k_enabled=_as_bool(p.get("dynamic_k_enabled", False), False),
            dynamic_k_games_prior=float(p.get("dynamic_k_games_prior", 30.0)),
            dynamic_k_uncertainty_mult_max=float(p.get("dynamic_k_uncertainty_mult_max", 1.80)),
            dynamic_k_volatility_ref=float(p.get("dynamic_k_volatility_ref", 0.30)),
            dynamic_k_volatility_mult_min=float(p.get("dynamic_k_volatility_mult_min", 0.80)),
            dynamic_k_volatility_mult_max=float(p.get("dynamic_k_volatility_mult_max", 1.35)),
            volatility_ewma_beta=float(p.get("volatility_ewma_beta", 0.08)),
            # MOVDA: artefatos antigos (sem as chaves) caem em lambda=0 = comportamento legado.
            mov_lambda=float(p.get("mov_lambda", 0.0)),
            mov_alpha=float(p.get("mov_alpha", 9.343)),
            mov_beta=float(p.get("mov_beta", 0.005)),
            mov_gamma=float(p.get("mov_gamma", -0.056)),
        ))

        model.ratings = {str(k): float(v) for k, v in (d.get("ratings") or {}).items()}
        model.games_played = {str(k): int(v) for k, v in (d.get("games_played") or {}).items()}
        model.team_volatility = {str(k): float(v) for k, v in (d.get("team_volatility") or {}).items()}
        model.league_strength = {str(k): float(v) for k, v in (d.get("league_strength") or {}).items()}
        model.tier_strength = {str(k): float(v) for k, v in (d.get("tier_strength") or {}).items()}
        try:
            model.side_adv_points = float(d.get("side_adv_points") or 0.0)
        except Exception:
            model.side_adv_points = 0.0

        tm = d.get("team_meta") or {}
        if isinstance(tm, dict):
            model.team_meta = {str(t): {"league": str((m or {}).get("league", "OTHER")), "tier": str((m or {}).get("tier", "OTHER"))} for t, m in tm.items()}

        rc = d.get("rank_caps_by_tier") or {}
        if isinstance(rc, dict):
            model.rank_caps_by_tier = enforce_monotone_caps({str(k): float(v) for k, v in rc.items()})

        lbp = d.get("league_base_points_map") or {}
        if isinstance(lbp, dict):
            model.league_base_points_map = {str(k): float(v) for k, v in lbp.items()}

        # Center offsets for older artifacts that might have drifted
        model._center_offsets()
        return model
