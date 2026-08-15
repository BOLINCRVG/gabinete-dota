"""LiveMarketModel: per-market, per-snapshot regression model for live markets.

Trained once per (market × minute) pair. Predicts the final game total (kills,
towers, dragons, barons, gamelength_min) given the game state at minute t.

From the regression we derive over/under probabilities for any line via the
empirical residual distribution (fit on the validation set):

    P(total >= line) ≈ 1 - Phi((line - pred) / sigma)

where sigma is the residual standard deviation captured at fit time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None  # type: ignore

from mlcore.live_ml.market_features import (
    LiveMarketSample,
    MARKET_FEATURE_COLS,
    SUPPORTED_MARKETS,
)


@dataclass
class MarketModelConfig:
    """Hyperparameters for the LightGBM market regressor."""

    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = 6
    min_data_in_leaf: int = 30
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 5
    lambda_l2: float = 0.5
    num_boost_round: int = 2000
    early_stopping_rounds: int = 50


class LiveMarketModel:
    """LightGBM regressor for one market × snapshot pair.

    Usage
    -----
        model = LiveMarketModel(market="total_kills", minute=15)
        model.fit(train_samples, val_samples)
        pred = model.predict_mean(sample)
        p_over = model.prob_over(sample, line=27.5)
    """

    def __init__(
        self,
        *,
        market: str,
        minute: int,
        config: Optional[MarketModelConfig] = None,
    ) -> None:
        if lgb is None:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        if market not in SUPPORTED_MARKETS:
            raise ValueError(f"market must be in {SUPPORTED_MARKETS}; got {market}")
        self.market = str(market)
        self.minute = int(minute)
        self.config = config or MarketModelConfig()
        self.lgb_model: Optional[Any] = None
        self.best_iteration: int = 0
        self.residual_sigma: float = 1.0  # set during fit
        self.feature_cols: Tuple[str, ...] = MARKET_FEATURE_COLS
        self.train_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_samples: Sequence[LiveMarketSample],
        val_samples: Optional[Sequence[LiveMarketSample]] = None,
    ) -> "LiveMarketModel":
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

        # Compute residual sigma on val (or train if no val)
        cal_samples = val_samples if val_samples else train_samples
        X_cal, y_cal = self._stack(cal_samples)
        preds = self.lgb_model.predict(X_cal, num_iteration=self.best_iteration)
        residuals = (y_cal - preds).astype(float)
        self.residual_sigma = float(max(1e-3, np.std(residuals)))

        self.train_metadata = {
            "market": self.market,
            "minute": int(self.minute),
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

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_mean(self, sample: LiveMarketSample) -> float:
        """Predict the expected market total for one sample."""
        if self.lgb_model is None:
            return 0.0
        X = sample.features.reshape(1, -1)
        return float(self.lgb_model.predict(X, num_iteration=self.best_iteration)[0])

    def predict_mean_batch(self, samples: Sequence[LiveMarketSample]) -> np.ndarray:
        if not samples or self.lgb_model is None:
            return np.array([], dtype=float)
        X = np.vstack([s.features for s in samples])
        return self.lgb_model.predict(X, num_iteration=self.best_iteration)

    def prob_over(self, sample: LiveMarketSample, line: float) -> float:
        """Return P(market_total > line) given the sample state."""
        pred = self.predict_mean(sample)
        try:
            z = (float(line) - pred) / float(max(self.residual_sigma, 1e-3))
            # 1 - Phi(z) where Phi is the standard normal CDF; use erf identity.
            from math import erf, sqrt
            cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
            return float(max(1e-6, min(1.0 - 1e-6, 1.0 - cdf)))
        except (TypeError, ValueError, OverflowError):
            return float("nan")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        if self.lgb_model is None:
            raise RuntimeError("Model not trained yet")
        d = Path(str(path))
        d.mkdir(parents=True, exist_ok=True)
        self.lgb_model.save_model(str(d / "lgb_model.txt"))
        (d / "metadata.json").write_text(json.dumps({
            "market": self.market,
            "minute": int(self.minute),
            "best_iteration": int(self.best_iteration),
            "residual_sigma": float(self.residual_sigma),
            "feature_cols": list(self.feature_cols),
            "train_metadata": self.train_metadata,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LiveMarketModel":
        d = Path(str(path))
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        obj = cls(market=str(meta.get("market") or "total_kills"),
                  minute=int(meta.get("minute") or 15))
        obj.lgb_model = lgb.Booster(model_file=str(d / "lgb_model.txt"))
        obj.best_iteration = int(meta.get("best_iteration", 0))
        obj.residual_sigma = float(meta.get("residual_sigma", 1.0))
        obj.feature_cols = tuple(meta.get("feature_cols", MARKET_FEATURE_COLS))
        obj.train_metadata = dict(meta.get("train_metadata") or {})
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stack(samples: Sequence[LiveMarketSample]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.vstack([s.features for s in samples])
        y = np.array([s.target for s in samples], dtype=float)
        return X, y
