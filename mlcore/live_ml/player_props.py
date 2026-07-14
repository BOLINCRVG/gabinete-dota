"""Phase 10: Player prop predictions (v1 — kills).

For each (player, champion) row in a game, predict the expected number of
kills the player will get, plus residual sigma → P(over X) via normal CDF
(same approach as Fase 7 game-level markets).

Features (per row, BEFORE the game starts):
  - player_kills_mean    -> rolling mean of kills over the player's last N games
  - player_kills_std     -> rolling std
  - champ_kills_mean_pos -> avg kills on this champion in this position across all players
  - position_id          -> top/jng/mid/bot/sup encoded
  - league_tier          -> structural strength of the league
  - bo                   -> 1/3/5
  - playoffs             -> 0/1
  - z_pre                -> logit of team's pre-game win probability

Markets supported (v1):
  - player_kills        (over/under)
Future: player_deaths, player_assists, player_KDA can use the same skeleton.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from mlcore.tiers import tier_for_league


_TIER_INT: Dict[str, int] = {
    "T1": 0, "T2": 1, "T3": 2, "ERL": 3, "MINOR": 4, "OTHER": 5,
}
_POS_INT: Dict[str, int] = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}


# Player prop markets (v1: kills only; deaths/assists planned for v2).
PLAYER_PROP_MARKETS: Tuple[str, ...] = ("player_kills",)


PLAYER_PROP_FEATURE_COLS: Tuple[str, ...] = (
    "player_kills_mean",
    "player_kills_std",
    "champ_kills_mean_pos",
    "position_id",
    "league_tier",
    "bo",
    "playoffs",
    "z_pre",
)


def _logit(p: float) -> float:
    p = max(1e-9, min(1.0 - 1e-9, float(p)))
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class PlayerPropSample:
    """One (player × game) row with pre-game features + observed target."""
    gameid: str
    player: str
    champion: str
    position: str
    market: str
    features: np.ndarray   # shape (len(PLAYER_PROP_FEATURE_COLS),)
    target: float          # observed kills (or other prop target)


def build_player_kill_features(
    player_rows: pd.DataFrame,
    *,
    p_map_pre_by_gameid_side: Mapping[Tuple[str, str], float],
    min_history: int = 5,
    half_life_games: int = 20,
) -> List[PlayerPropSample]:
    """Build per-player rolling features for kills prediction.

    Walks games chronologically per player so the features are causal
    (only past games inform the feature vector at each row).

    p_map_pre_by_gameid_side: {(gameid, "Blue" | "Red"): p_team_a_wins}
    """
    if player_rows.empty:
        return []

    # Normalize column names that we'll use
    df = player_rows[player_rows["position"].isin(_POS_INT.keys())].copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT
    df = df.dropna(subset=["gameid", "playername", "champion", "position", "kills"])
    df = df.sort_values(["date", "gameid"]).reset_index(drop=True)

    # Per-champion-per-position baseline mean kills
    champ_pos_kills_mean: Dict[Tuple[str, str], float] = {}
    grouped = df.groupby(["champion", "position"])["kills"].mean()
    for (champ, pos), v in grouped.items():
        champ_pos_kills_mean[(str(champ), str(pos))] = float(v)
    global_kills_mean = float(df["kills"].mean()) if len(df) else 2.5

    # Per-player rolling kills history. For each row, compute the (causal)
    # weighted mean and std from prior rows of the same player using
    # exponential decay weights with given half-life.
    samples: List[PlayerPropSample] = []
    decay = math.log(2) / max(1, int(half_life_games))
    for player, g in df.groupby("playername", sort=False):
        # Maintain a deque of (k, age_in_games) so weight = exp(-decay*age)
        history_k: List[float] = []
        for _, row in g.iterrows():
            n_hist = len(history_k)
            if n_hist >= min_history:
                weights = np.array([math.exp(-decay * (n_hist - 1 - i))
                                     for i in range(n_hist)], dtype=float)
                ws = weights.sum()
                ks = np.array(history_k, dtype=float)
                w_mean = float((ks * weights).sum() / ws) if ws > 0 else float(np.mean(ks))
                resid = ks - w_mean
                w_var = float((weights * (resid ** 2)).sum() / ws) if ws > 0 else float(np.var(ks))
                w_std = float(math.sqrt(max(0.0, w_var)))
            else:
                # Not enough history — fall back to global mean and a moderate std
                w_mean = float(global_kills_mean)
                w_std = 2.0

            champ = str(row.get("champion") or "")
            pos = str(row.get("position") or "")
            champ_baseline = champ_pos_kills_mean.get((champ, pos), global_kills_mean)
            league = str(row.get("league") or "")
            tier_int = _TIER_INT.get(tier_for_league(league), 5)
            pos_id = _POS_INT.get(pos, 5)
            bo_raw = row.get("game", 1)
            try:
                bo = int(bo_raw)
            except (TypeError, ValueError):
                bo = 1
            playoffs = 1 if str(row.get("playoffs", "") or "").strip().lower() in {"1", "true", "playoffs"} else 0

            side_norm = str(row.get("side", "Blue") or "Blue").strip().title()
            gid = str(row.get("gameid") or "")
            p_pre = float(p_map_pre_by_gameid_side.get((gid, side_norm), 0.5))
            if not (0.0 < p_pre < 1.0):
                p_pre = 0.5

            feat = np.array([
                w_mean, w_std, float(champ_baseline),
                float(pos_id), float(tier_int),
                float(bo), float(playoffs), _logit(p_pre),
            ], dtype=float)

            try:
                target_kills = float(row.get("kills", 0.0) or 0.0)
            except (TypeError, ValueError):
                target_kills = 0.0

            samples.append(PlayerPropSample(
                gameid=gid,
                player=str(player),
                champion=champ,
                position=pos,
                market="player_kills",
                features=feat,
                target=float(target_kills),
            ))

            history_k.append(float(target_kills))

    return samples


try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore


@dataclass
class PlayerPropConfig:
    learning_rate: float = 0.04
    num_leaves: int = 31
    max_depth: int = 6
    min_data_in_leaf: int = 80
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    lambda_l2: float = 0.5
    num_boost_round: int = 1500
    early_stopping_rounds: int = 40


class PlayerPropModel:
    """LightGBM regressor for one player-prop market (v1 = player_kills).

    Output: expected value + residual sigma → P(over X) via normal CDF.
    Same trick as Fase 7 markets — simple and good enough for v1.
    """

    def __init__(self, *, market: str = "player_kills",
                 config: Optional[PlayerPropConfig] = None) -> None:
        if lgb is None:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        if market not in PLAYER_PROP_MARKETS:
            raise ValueError(f"market must be in {PLAYER_PROP_MARKETS}; got {market}")
        self.market = str(market)
        self.config = config or PlayerPropConfig()
        self.lgb_model: Optional[Any] = None
        self.best_iteration: int = 0
        self.residual_sigma: float = 2.0
        self.feature_cols: Tuple[str, ...] = PLAYER_PROP_FEATURE_COLS
        self.train_metadata: Dict[str, Any] = {}

    def fit(self, train_samples: Sequence[PlayerPropSample],
            val_samples: Optional[Sequence[PlayerPropSample]] = None) -> "PlayerPropModel":
        if not train_samples:
            raise ValueError("train_samples is empty")
        X_train, y_train = self._stack(train_samples)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": self.config.learning_rate,
            "num_leaves": self.config.num_leaves,
            "max_depth": self.config.max_depth,
            "min_data_in_leaf": self.config.min_data_in_leaf,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "bagging_freq": self.config.bagging_freq,
            "lambda_l2": self.config.lambda_l2,
            "verbose": -1,
        }
        train_ds = lgb.Dataset(X_train, label=y_train, feature_name=list(self.feature_cols))
        callbacks = [lgb.log_evaluation(0)]
        val_sets = [train_ds]
        val_names = ["train"]
        if val_samples:
            X_val, y_val = self._stack(val_samples)
            val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds, feature_name=list(self.feature_cols))
            val_sets.append(val_ds)
            val_names.append("val")
            callbacks.append(lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds, verbose=False))
        self.lgb_model = lgb.train(
            params, train_ds,
            num_boost_round=self.config.num_boost_round,
            valid_sets=val_sets, valid_names=val_names,
            callbacks=callbacks,
        )
        self.best_iteration = int(self.lgb_model.best_iteration or self.lgb_model.current_iteration())

        cal_samples = val_samples if val_samples else train_samples
        X_cal, y_cal = self._stack(cal_samples)
        preds = self.lgb_model.predict(X_cal, num_iteration=self.best_iteration)
        residuals = (y_cal - preds).astype(float)
        self.residual_sigma = float(max(1e-3, np.std(residuals)))

        self.train_metadata = {
            "market": self.market,
            "n_train": len(train_samples),
            "n_val": len(val_samples) if val_samples else 0,
            "best_iteration": int(self.best_iteration),
            "residual_sigma": float(self.residual_sigma),
            "feature_names": list(self.feature_cols),
            "feature_importance": dict(zip(
                self.feature_cols,
                [float(v) for v in self.lgb_model.feature_importance(importance_type="gain")],
            )),
        }
        return self

    def predict_mean(self, sample: PlayerPropSample) -> float:
        if self.lgb_model is None:
            return 0.0
        X = sample.features.reshape(1, -1)
        return float(self.lgb_model.predict(X, num_iteration=self.best_iteration)[0])

    def predict_mean_batch(self, samples: Sequence[PlayerPropSample]) -> np.ndarray:
        if not samples or self.lgb_model is None:
            return np.array([], dtype=float)
        X = np.vstack([s.features for s in samples])
        return self.lgb_model.predict(X, num_iteration=self.best_iteration)

    def prob_over(self, sample: PlayerPropSample, line: float) -> float:
        pred = self.predict_mean(sample)
        try:
            z = (float(line) - pred) / float(max(self.residual_sigma, 1e-3))
            from math import erf, sqrt
            cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
            return float(max(1e-6, min(1.0 - 1e-6, 1.0 - cdf)))
        except (TypeError, ValueError, OverflowError):
            return float("nan")

    def save(self, path: str | Path) -> None:
        if self.lgb_model is None:
            raise RuntimeError("Model not trained yet")
        d = Path(str(path))
        d.mkdir(parents=True, exist_ok=True)
        self.lgb_model.save_model(str(d / "lgb_model.txt"))
        (d / "metadata.json").write_text(json.dumps({
            "market": self.market,
            "best_iteration": int(self.best_iteration),
            "residual_sigma": float(self.residual_sigma),
            "feature_cols": list(self.feature_cols),
            "train_metadata": self.train_metadata,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PlayerPropModel":
        d = Path(str(path))
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        obj = cls(market=str(meta.get("market") or "player_kills"))
        obj.lgb_model = lgb.Booster(model_file=str(d / "lgb_model.txt"))
        obj.best_iteration = int(meta.get("best_iteration", 0))
        obj.residual_sigma = float(meta.get("residual_sigma", 2.0))
        obj.feature_cols = tuple(meta.get("feature_cols", PLAYER_PROP_FEATURE_COLS))
        obj.train_metadata = dict(meta.get("train_metadata") or {})
        return obj

    @staticmethod
    def _stack(samples: Sequence[PlayerPropSample]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.vstack([s.features for s in samples])
        y = np.array([s.target for s in samples], dtype=float)
        return X, y


__all__ = [
    "PLAYER_PROP_FEATURE_COLS",
    "PLAYER_PROP_MARKETS",
    "PlayerPropConfig",
    "PlayerPropModel",
    "PlayerPropSample",
    "build_player_kill_features",
]
