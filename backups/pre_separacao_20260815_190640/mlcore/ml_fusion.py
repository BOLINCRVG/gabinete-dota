from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import math

from .utils import clip_prob, logit, sigmoid


@dataclass(frozen=True)
class FusionConfig:
    # Default (used in "mid" phase)
    team_weight: float = 0.75
    # Phase-aware defaults
    team_weight_early: float = 0.58
    team_weight_mid: float = 0.75
    team_weight_playoffs: float = 0.82
    # In early season (few team games), trust players relatively more.
    early_season_team_weight: float = 0.55
    season_knee_games: float = 18.0
    # How much coverage amplifies/attenuates the players contribution.
    coverage_power: float = 1.5
    # Extra multiplier for players component when transfer signal is high.
    transfer_boost: float = 0.35
    # Keep final output away from 0/1.
    eps: float = 1e-6


def _finite01(x: float) -> bool:
    return math.isfinite(x) and 0.0 < x < 1.0


def fuse_team_players_probs(
    *,
    p_team: float,
    p_players: float,
    coverage: float,
    players_weight_mult: float = 1.0,
    team_games_min: float | None = None,
    transfer_signal: float = 0.0,
    season_phase: str = "auto",  # auto|early|mid|playoffs
    cfg: FusionConfig | None = None,
) -> Dict[str, Any]:
    """Fuse team-model and players-model probabilities in logit space.

    coverage in [0,1] scales players impact. When coverage is low, fusion falls
    back smoothly toward the team model.
    """
    c = cfg or FusionConfig()

    if not _finite01(float(p_team)):
        return {
            "p_fused": float("nan"),
            "w_team": float("nan"),
            "w_players": float("nan"),
            "coverage_eff": float("nan"),
            "reason": "invalid_p_team",
        }

    if not _finite01(float(p_players)):
        return {
            "p_fused": float(clip_prob(float(p_team), eps=float(c.eps))),
            "w_team": 1.0,
            "w_players": 0.0,
            "coverage_eff": 0.0,
            "reason": "invalid_p_players_fallback_team",
        }

    cov = float(coverage)
    if not math.isfinite(cov):
        cov = 0.0
    cov = max(0.0, min(1.0, cov))
    cov_eff = float(cov ** max(0.0, float(c.coverage_power)))
    pwm = float(players_weight_mult) if math.isfinite(float(players_weight_mult)) else 1.0
    pwm = max(0.0, min(4.0, pwm))

    phase = str(season_phase or "auto").strip().lower()
    if phase == "playoffs":
        w_team_base = max(0.0, min(1.0, float(c.team_weight_playoffs)))
    elif phase == "early":
        w_team_base = max(0.0, min(1.0, float(c.team_weight_early)))
    elif phase == "mid":
        w_team_base = max(0.0, min(1.0, float(c.team_weight_mid)))
    else:
        # auto phase from sample size
        if team_games_min is not None and math.isfinite(float(team_games_min)):
            knee = max(1e-6, float(c.season_knee_games))
            g = float(team_games_min)
            if g <= 0.45 * knee:
                phase = "early"
                w_team_base = max(0.0, min(1.0, float(c.team_weight_early)))
            else:
                phase = "mid"
                w_team_base = max(0.0, min(1.0, float(c.team_weight_mid)))
        else:
            phase = "mid"
            w_team_base = max(0.0, min(1.0, float(c.team_weight_mid)))
    if team_games_min is not None and math.isfinite(float(team_games_min)):
        knee = max(1e-6, float(c.season_knee_games))
        progress = max(0.0, min(1.0, float(team_games_min) / knee))
        w_early = max(0.0, min(1.0, float(c.early_season_team_weight)))
        w_team_base = float(w_early + (w_team_base - w_early) * progress)

    ts = 0.0 if not math.isfinite(float(transfer_signal)) else float(transfer_signal)
    ts = max(0.0, min(1.0, ts))

    w_players_raw = (1.0 - w_team_base) * cov_eff * pwm * (1.0 + float(c.transfer_boost) * ts)
    w_team = float(w_team_base)
    w_players = float(w_players_raw)

    s = w_team + w_players
    if s <= 0:
        return {
            "p_fused": float(clip_prob(float(p_team), eps=float(c.eps))),
            "w_team": 1.0,
            "w_players": 0.0,
            "coverage_eff": cov_eff,
            "reason": "zero_weights_fallback_team",
        }

    w_team /= s
    w_players /= s

    z_team = float(logit(float(p_team), eps=float(c.eps)))
    z_players = float(logit(float(p_players), eps=float(c.eps)))
    z_fused = float(w_team * z_team + w_players * z_players)
    p_fused = float(clip_prob(float(sigmoid(z_fused)), eps=float(c.eps)))

    return {
        "p_fused": p_fused,
        "w_team": w_team,
        "w_players": w_players,
        "coverage_eff": cov_eff,
        "players_weight_mult": pwm,
        "season_phase": phase,
        "reason": "ok",
    }
