"""StateToWinModel: per-snapshot LightGBM model for live win-probability prediction.

One model per snapshot minute (10, 15, 20, 25). Each model is trained on
historical games' state at that minute + the pre-game prediction, to predict
the final outcome (P(blue wins) | snapshot @ t).

Output is calibrated via Platt scaling on top of LightGBM probabilities so it
can be used directly as a probability.
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

from mlcore.calibration import PlattCalibrator, fit_platt_from_raw_probs
from mlcore.live_ml.features import LiveSnapshot, feature_names
from mlcore.utils import clip_prob


@dataclass
class StateModelConfig:
    """Hyperparameters for the LightGBM state-to-win model."""

    learning_rate: float = 0.03
    num_leaves: int = 31
    max_depth: int = 6
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 5
    lambda_l2: float = 0.5
    num_boost_round: int = 2000
    early_stopping_rounds: int = 50


class StateToWinModel:
    """One LightGBM model trained for a specific snapshot minute.

    Usage
    -----
        model = StateToWinModel(minute=15)
        model.fit(train_snapshots, val_snapshots)
        p_live = model.predict(snapshot)
    """

    def __init__(self, *, minute: int, config: Optional[StateModelConfig] = None) -> None:
        if lgb is None:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        self.minute = int(minute)
        self.config = config or StateModelConfig()
        self.lgb_model: Optional[Any] = None
        self.platt: Optional[PlattCalibrator] = None
        self.best_iteration: int = 0
        self.feature_cols: Tuple[str, ...] = feature_names()
        self.train_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_snapshots: Sequence[LiveSnapshot],
        val_snapshots: Optional[Sequence[LiveSnapshot]] = None,
    ) -> "StateToWinModel":
        if not train_snapshots:
            raise ValueError("train_snapshots is empty")
        X_train, y_train = self._stack(train_snapshots)

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

        if val_snapshots:
            X_val, y_val = self._stack(val_snapshots)
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

        # Calibrate on validation set if available, else on training set
        cal_snapshots = val_snapshots if val_snapshots else train_snapshots
        X_cal, y_cal = self._stack(cal_snapshots)
        p_raw = self.lgb_model.predict(X_cal, num_iteration=self.best_iteration)
        p_raw = np.clip(p_raw, 1e-9, 1.0 - 1e-9)
        try:
            self.platt = fit_platt_from_raw_probs(list(y_cal), list(p_raw))
        except Exception:
            self.platt = None

        self.train_metadata = {
            "minute": int(self.minute),
            "n_train": len(train_snapshots),
            "n_val": len(val_snapshots) if val_snapshots else 0,
            "best_iteration": int(self.best_iteration),
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

    def predict_proba_one(self, snapshot: LiveSnapshot) -> float:
        """Predict P(snapshot.side wins) for a single snapshot."""
        if self.lgb_model is None:
            return float(snapshot.p_pre)
        X = snapshot.features.reshape(1, -1)
        p_raw = float(self.lgb_model.predict(X, num_iteration=self.best_iteration)[0])
        if self.platt is not None:
            p_raw = float(self.platt.predict_proba(p_raw))
        return float(clip_prob(p_raw))

    def predict_proba(self, snapshots: Sequence[LiveSnapshot]) -> np.ndarray:
        """Batch predict over multiple snapshots."""
        if not snapshots:
            return np.array([], dtype=float)
        if self.lgb_model is None:
            return np.array([s.p_pre for s in snapshots], dtype=float)
        X = np.vstack([s.features for s in snapshots])
        p_raw = self.lgb_model.predict(X, num_iteration=self.best_iteration)
        p_raw = np.clip(p_raw, 1e-9, 1.0 - 1e-9)
        if self.platt is not None:
            p_cal = np.array([self.platt.predict_proba(float(p)) for p in p_raw])
            return np.clip(p_cal, 1e-9, 1.0 - 1e-9)
        return p_raw

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the model to a directory: lgb_model.txt + platt.json + metadata.json."""
        if self.lgb_model is None:
            raise RuntimeError("Model not trained yet")
        d = Path(str(path))
        d.mkdir(parents=True, exist_ok=True)
        self.lgb_model.save_model(str(d / "lgb_model.txt"))
        (d / "metadata.json").write_text(json.dumps({
            "minute": self.minute,
            "best_iteration": self.best_iteration,
            "feature_cols": list(self.feature_cols),
            "train_metadata": self.train_metadata,
        }, indent=2), encoding="utf-8")
        if self.platt is not None:
            (d / "platt.json").write_text(json.dumps(self.platt.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StateToWinModel":
        d = Path(str(path))
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        obj = cls(minute=int(meta.get("minute", 15)))
        obj.lgb_model = lgb.Booster(model_file=str(d / "lgb_model.txt"))
        obj.best_iteration = int(meta.get("best_iteration", 0))
        obj.feature_cols = tuple(meta.get("feature_cols", feature_names()))
        obj.train_metadata = dict(meta.get("train_metadata") or {})
        platt_path = d / "platt.json"
        if platt_path.exists():
            obj.platt = PlattCalibrator.from_dict(json.loads(platt_path.read_text(encoding="utf-8")))
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stack(snapshots: Sequence[LiveSnapshot]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.vstack([s.features for s in snapshots])
        y = np.array([s.y for s in snapshots], dtype=int)
        return X, y
