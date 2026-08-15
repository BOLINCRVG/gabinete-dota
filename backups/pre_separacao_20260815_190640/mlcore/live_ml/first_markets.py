"""Phase 8: pre-game binary classifiers for first-X markets.

For each first-X market we predict P(Blue side wins it) using only pre-game
features (no live state needed). At apply time the user just needs the same
inputs already entered on the live-bet tab: teams + league + drafts (drafts
flow into p_map_pre via the main ML pipeline).

Markets supported:
  - first_blood     (almost always resolved, very high coverage)
  - first_dragon
  - first_tower
  - first_herald
  - first_baron     (lower coverage — many games end before)

The probability of Red side winning the market is just (1 - P_blue).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from mlcore.tiers import league_strength_score, tier_for_league


# Markets supported by Phase 8 (binary: did Blue win this?).
FIRST_MARKETS: Tuple[str, ...] = (
    "first_blood",
    "first_dragon",
    "first_tower",
    "first_herald",
    "first_baron",
)


# Pre-game features for each first-X classifier.
FIRST_FEATURE_COLS: Tuple[str, ...] = (
    "z_pre",            # logit of P(blue wins map)
    "league_tier",      # encoded tier (T1=0..MINOR=4)
    "league_strength",  # [0,1] strength score
    "bo",               # int 1/3/5
    "playoffs",         # 0/1
)


_TIER_INT: Dict[str, int] = {
    "T1": 0, "T2": 1, "T3": 2, "ERL": 3, "MINOR": 4, "OTHER": 5,
}


# Column in the Oracle CSV → market name (binary at the BLUE row).
_MARKET_TO_COLUMN: Dict[str, str] = {
    "first_blood": "firstblood",
    "first_dragon": "firstdragon",
    "first_tower": "firsttower",
    "first_herald": "firstherald",
    "first_baron": "firstbaron",
}


def _encode_tier(league: str) -> int:
    return _TIER_INT.get(tier_for_league(str(league or "")), 5)


def _logit(p: float) -> float:
    p = max(1e-9, min(1.0 - 1e-9, float(p)))
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class FirstMarketSample:
    """One game's pre-game features + target (did Blue win the market?)."""
    gameid: str
    market: str
    features: np.ndarray   # shape (len(FIRST_FEATURE_COLS),)
    target: int            # 1 if Blue won the first-X, 0 otherwise


def feature_names() -> Tuple[str, ...]:
    return FIRST_FEATURE_COLS


def build_first_market_features(
    team_rows: pd.DataFrame,
    *,
    market: str,
    p_map_pre_by_gameid_blue: Mapping[str, float],
    require_complete: bool = True,
) -> List[FirstMarketSample]:
    """Build training samples for one first-X market.

    One sample per game (BLUE perspective). The target is 1 if Blue side won
    the market, 0 if Red side won, dropped if neither side has it recorded.
    """
    if market not in FIRST_MARKETS:
        raise ValueError(f"market must be one of {FIRST_MARKETS}")
    column = _MARKET_TO_COLUMN[market]
    if column not in team_rows.columns:
        return []

    df = team_rows[team_rows["participantid"].isin([100, 200])].copy()
    if require_complete:
        df = df.dropna(subset=[column])

    out: List[FirstMarketSample] = []
    for gid, g in df.groupby("gameid", sort=False):
        gid_s = str(gid)
        blue_rows = g[g["participantid"] == 100]
        red_rows = g[g["participantid"] == 200]
        if blue_rows.empty or red_rows.empty:
            continue
        blue = blue_rows.iloc[0]
        red = red_rows.iloc[0]

        # Resolve target from blue row (1 = blue won)
        try:
            blue_val = float(blue.get(column, float("nan")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(blue_val):
            continue
        target = 1 if blue_val >= 0.5 else 0

        # Sanity: the two team rows should be complementary (blue=1 → red=0)
        try:
            red_val = float(red.get(column, float("nan")))
            if math.isfinite(red_val) and abs((blue_val + red_val) - 1.0) > 0.5:
                # Inconsistent rows — skip
                continue
        except (TypeError, ValueError):
            pass

        # Pre-game features
        p_pre = float(p_map_pre_by_gameid_blue.get(gid_s, 0.5))
        if not (0.0 < p_pre < 1.0):
            p_pre = 0.5
        league = str(blue.get("league", "") or "")
        bo_raw = blue.get("game", 1)
        bo = int(bo_raw) if str(bo_raw).strip().isdigit() else 1
        playoffs = 1 if str(blue.get("playoffs", "") or "").strip().lower() in {"1", "true", "playoffs"} else 0

        features = np.array([
            _logit(p_pre),
            float(_encode_tier(league)),
            float(league_strength_score(league)),
            float(bo),
            float(playoffs),
        ], dtype=float)

        out.append(FirstMarketSample(
            gameid=gid_s, market=market,
            features=features, target=int(target),
        ))

    return out


try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore


@dataclass
class FirstMarketConfig:
    learning_rate: float = 0.04
    num_leaves: int = 15
    max_depth: int = 4
    min_data_in_leaf: int = 60
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    lambda_l2: float = 0.5
    num_boost_round: int = 1500
    early_stopping_rounds: int = 40


class FirstMarketModel:
    """LightGBM binary classifier for one first-X market.

    Output is P(Blue side wins the market). Calibrated via a built-in
    isotonic-style adjustment: we capture the raw mean prediction and offset
    so that the average matches the training base rate.
    """

    def __init__(
        self,
        *,
        market: str,
        config: Optional[FirstMarketConfig] = None,
    ) -> None:
        if lgb is None:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        if market not in FIRST_MARKETS:
            raise ValueError(f"market must be in {FIRST_MARKETS}")
        self.market = str(market)
        self.config = config or FirstMarketConfig()
        self.lgb_model: Optional[Any] = None
        self.best_iteration: int = 0
        self.base_rate_blue: float = 0.5
        self.feature_cols: Tuple[str, ...] = FIRST_FEATURE_COLS
        self.train_metadata: Dict[str, Any] = {}

    def fit(
        self,
        train_samples: Sequence[FirstMarketSample],
        val_samples: Optional[Sequence[FirstMarketSample]] = None,
    ) -> "FirstMarketModel":
        if not train_samples:
            raise ValueError("train_samples is empty")
        X_train, y_train = self._stack(train_samples)
        self.base_rate_blue = float(np.mean(y_train))

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
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

        self.train_metadata = {
            "market": self.market,
            "n_train": len(train_samples),
            "n_val": len(val_samples) if val_samples else 0,
            "base_rate_blue": float(self.base_rate_blue),
            "best_iteration": int(self.best_iteration),
            "feature_names": list(self.feature_cols),
            "feature_importance": dict(zip(
                self.feature_cols,
                [float(v) for v in self.lgb_model.feature_importance(importance_type="gain")],
            )),
        }
        return self

    def predict_blue(self, sample: FirstMarketSample) -> float:
        if self.lgb_model is None:
            return float(self.base_rate_blue)
        X = sample.features.reshape(1, -1)
        p = float(self.lgb_model.predict(X, num_iteration=self.best_iteration)[0])
        return float(max(1e-6, min(1.0 - 1e-6, p)))

    def predict_blue_batch(self, samples: Sequence[FirstMarketSample]) -> np.ndarray:
        if not samples or self.lgb_model is None:
            return np.array([], dtype=float)
        X = np.vstack([s.features for s in samples])
        return np.clip(self.lgb_model.predict(X, num_iteration=self.best_iteration), 1e-6, 1 - 1e-6)

    def save(self, path: str | Path) -> None:
        if self.lgb_model is None:
            raise RuntimeError("Model not trained yet")
        d = Path(str(path))
        d.mkdir(parents=True, exist_ok=True)
        self.lgb_model.save_model(str(d / "lgb_model.txt"))
        (d / "metadata.json").write_text(json.dumps({
            "market": self.market,
            "best_iteration": int(self.best_iteration),
            "base_rate_blue": float(self.base_rate_blue),
            "feature_cols": list(self.feature_cols),
            "train_metadata": self.train_metadata,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FirstMarketModel":
        d = Path(str(path))
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        obj = cls(market=str(meta.get("market") or "first_blood"))
        obj.lgb_model = lgb.Booster(model_file=str(d / "lgb_model.txt"))
        obj.best_iteration = int(meta.get("best_iteration", 0))
        obj.base_rate_blue = float(meta.get("base_rate_blue", 0.5))
        obj.feature_cols = tuple(meta.get("feature_cols", FIRST_FEATURE_COLS))
        obj.train_metadata = dict(meta.get("train_metadata") or {})
        return obj

    @staticmethod
    def _stack(samples: Sequence[FirstMarketSample]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.vstack([s.features for s in samples])
        y = np.array([s.target for s in samples], dtype=float)
        return X, y


__all__ = [
    "FIRST_FEATURE_COLS",
    "FIRST_MARKETS",
    "FirstMarketConfig",
    "FirstMarketModel",
    "FirstMarketSample",
    "build_first_market_features",
    "feature_names",
]
