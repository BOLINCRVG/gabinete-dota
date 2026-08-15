"""Runtime helper for the Phase 7 live market models.

Lazy-loads per-(market × minute) models from disk and exposes a high-level
helper that, given the current game state, returns calibrated probabilities
for over/under lines on common markets:

  - total_kills
  - total_towers
  - total_dragons
  - gamelength_min  (total game time in minutes)

The total_barons model was REJECTED by the gate (delta_RMSE < threshold), so
it is intentionally not exposed here.

Models live under `live_models/markets/{market}/minute_{T}/`.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from mlcore.live_ml import LiveMarketModel, LiveMarketSample, SNAPSHOT_MINUTES


# Markets that PASSED the gate (live_models/markets/<name>/minute_*/)
# total_barons is excluded because all 4 snapshots had delta_RMSE below threshold.
ACTIVE_MARKETS: Tuple[str, ...] = (
    "total_kills", "total_towers", "total_dragons", "gamelength_min",
)


_VALID_MINUTES = set(SNAPSHOT_MINUTES)


def is_live_market_models_enabled(session_state: Mapping[str, Any] | None) -> bool:
    """Markets share the same flag as the state model — if state is ON, markets are too."""
    ss = session_state if isinstance(session_state, Mapping) else {}
    return bool(ss.get("ml_live_state_model_enabled", False))


def markets_dir_from_state(session_state: Mapping[str, Any] | None) -> Path:
    ss = session_state if isinstance(session_state, Mapping) else {}
    base = ss.get("ml_live_state_models_dir", "live_models") or "live_models"
    return Path(str(base)) / "markets"


@lru_cache(maxsize=64)
def _load_market_model_by_sig(markets_dir: str, market: str, minute: int, _mtime_ns: int) -> Optional[LiveMarketModel]:
    """Load real, com o mtime do modelo na chave (via _cached_load_market_model)."""
    p = Path(str(markets_dir)) / str(market) / f"minute_{int(minute):02d}"
    if not p.exists():
        return None
    try:
        return LiveMarketModel.load(p)
    except Exception:
        return None


def _market_model_mtime_ns(markets_dir: str, market: str, minute: int) -> int:
    """mtime do arquivo do modelo (0 se ausente) — entra na chave do cache."""
    p = Path(str(markets_dir)) / str(market) / f"minute_{int(minute):02d}" / "lgb_model.txt"
    try:
        return int(p.stat().st_mtime_ns)
    except OSError:
        return 0


def _cached_load_market_model(markets_dir: str, market: str, minute: int) -> Optional[LiveMarketModel]:
    """Carrega o modelo com cache que se INVALIDA quando ele é retreinado.

    Bug achado com ele (23/07): o cache era só (pasta, mercado, minuto), então
    guardava o None de quando o modelo ainda NÃO existia — e o N/A persistia
    após o retreino até reiniciar o app (F5 não limpa cache de módulo). Agora o
    mtime do arquivo entra na chave: modelo reescrito -> chave nova -> recarrega
    sozinho, sem reiniciar."""
    return _load_market_model_by_sig(
        str(markets_dir), str(market), int(minute),
        _market_model_mtime_ns(markets_dir, market, minute),
    )


def clear_market_model_cache() -> None:
    _load_market_model_by_sig.cache_clear()


def available_markets(markets_dir: str | Path = "live_models/markets") -> Dict[str, Tuple[int, ...]]:
    """Return {market: tuple_of_minutes_with_saved_model}."""
    base = Path(str(markets_dir))
    out: Dict[str, Tuple[int, ...]] = {}
    if not base.exists():
        return out
    for market_dir in sorted(base.iterdir()):
        if not market_dir.is_dir():
            continue
        market = market_dir.name
        minutes = []
        for sub in sorted(market_dir.iterdir()):
            if sub.is_dir() and (sub / "metadata.json").exists():
                # Expect "minute_NN"
                try:
                    minute = int(sub.name.split("_")[-1])
                    minutes.append(minute)
                except (ValueError, IndexError):
                    continue
        if minutes:
            out[market] = tuple(sorted(minutes))
    return out


def _features_from_state(
    *, p_map_pre: float, minute: int,
    gold_diff: float, xp_diff: float, cs_diff: float,
    kills_a: float, kills_b: float,
    deaths_a: float, deaths_b: float,
    assists_a: float, assists_b: float,
    league_tier_int: int = 5, bo: int = 1, playoffs: int = 0,
) -> np.ndarray:
    """Build the canonical market feature vector from live state. Matches
    MARKET_FEATURE_COLS in mlcore.live_ml.market_features.
    """
    p_pre = max(1e-9, min(1.0 - 1e-9, float(p_map_pre)))
    z_pre = math.log(p_pre / (1.0 - p_pre))
    return np.array([
        z_pre,
        float(gold_diff), abs(float(gold_diff)),
        float(xp_diff), float(cs_diff),
        float(kills_a + kills_b), float(kills_a - kills_b),
        float(deaths_a + deaths_b),
        float(assists_a + assists_b),
        float(league_tier_int),
        float(bo), float(playoffs),
    ], dtype=float)


def predict_market(
    *,
    session_state: Mapping[str, Any] | None,
    market: str,
    minute: int,
    line: Optional[float] = None,
    p_map_pre: float = 0.5,
    gold_diff: float = 0.0,
    xp_diff: float = 0.0,
    cs_diff: float = 0.0,
    kills_a: float = 0.0, kills_b: float = 0.0,
    deaths_a: float = 0.0, deaths_b: float = 0.0,
    assists_a: float = 0.0, assists_b: float = 0.0,
    league: str = "", bo: int = 1, is_playoffs: bool = False,
) -> Dict[str, Any]:
    """Return {'mean': pred, 'p_over': float|None, 'sigma': float, 'minute': int}.

    p_over is computed only when `line` is provided. If feature flag is off or
    model is missing, returns {'mean': None, 'p_over': None, ...} so the
    caller can fall back to heuristics.
    """
    out = {"mean": None, "p_over": None, "sigma": None, "minute": int(minute), "market": str(market), "line": line}

    if not is_live_market_models_enabled(session_state):
        out["reason"] = "disabled"
        return out
    if str(market) not in ACTIVE_MARKETS:
        out["reason"] = "market_not_active"
        return out

    # Snap to nearest supported snapshot
    try:
        m = round(float(minute))
    except (TypeError, ValueError):
        out["reason"] = "invalid_minute"
        return out
    supported = sorted(_VALID_MINUTES)
    nearest = min(supported, key=lambda x: abs(x - m))
    if abs(nearest - m) > 3:
        out["reason"] = "minute_out_of_window"
        return out

    markets_dir = str(markets_dir_from_state(session_state))
    model = _cached_load_market_model(markets_dir, str(market), int(nearest))
    if model is None:
        out["reason"] = "model_not_available"
        return out

    # Resolve league tier
    try:
        from mlcore.tiers import tier_for_league
        _tier_int = {"T1": 0, "T2": 1, "T3": 2, "MINOR": 3, "ERL": 4, "OTHER": 5}
        tier_int = _tier_int.get(tier_for_league(str(league or "")), 5)
    except Exception:
        tier_int = 5

    feat = _features_from_state(
        p_map_pre=float(p_map_pre), minute=int(nearest),
        gold_diff=float(gold_diff), xp_diff=float(xp_diff), cs_diff=float(cs_diff),
        kills_a=float(kills_a), kills_b=float(kills_b),
        deaths_a=float(deaths_a), deaths_b=float(deaths_b),
        assists_a=float(assists_a), assists_b=float(assists_b),
        league_tier_int=int(tier_int), bo=int(bo), playoffs=1 if bool(is_playoffs) else 0,
    )
    sample = LiveMarketSample(
        gameid="runtime", minute=int(nearest), market=str(market),
        features=feat, target=0.0,
    )
    try:
        pred_mean = model.predict_mean(sample)
        out["mean"] = float(pred_mean)
        out["sigma"] = float(model.residual_sigma)
        out["minute_used"] = int(nearest)
        if line is not None:
            try:
                out["p_over"] = float(model.prob_over(sample, float(line)))
            except Exception:
                out["p_over"] = None
    except Exception as exc:
        out["reason"] = f"predict_error:{type(exc).__name__}"
        return out
    out["reason"] = "ok"
    return out
