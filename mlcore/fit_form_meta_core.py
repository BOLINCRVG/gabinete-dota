"""Phase C meta-blend trainer — rewritten 2026-05-30.

REWRITE NOTE: the original source was lost in a cleanup pass. This is a
fresh implementation that produces the same return schema expected by
`mlcore.meta_blend_core.apply_meta_blend_c_model`, but is minimalist:
  - Trains a SINGLE global logistic regression (no per-context submodels)
  - Computes basic OOS metrics via a simple time-ordered holdout
  - Emits the runtime_guard / runtime_stability defaults that downstream
    expects, so apply_meta_blend_c_model can still gate / cap properly

The original module had richer per-(league × patch × phase) context
sub-models. Those are NOT reproduced here — when the global model is
selected, the C-layer still applies, just without per-context refinement.

Public API: _fit_meta_blend_c_model(samples, ...) -> dict
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mlcore.fit_form_core import (
    _fit_logreg_numpy,
    _meta_probs_for_sample,
    _meta_recent_evidence_from_sample,
    _predict_logreg_numpy,
)
from mlcore.fit_samples_core import Sample
from mlcore.meta_blend_features_core import (
    META_BLEND_C_MODEL_FEATURES,
    build_meta_blend_c_model_features,
)
from mlcore.utils import brier, logloss


def _features_for_sample(
    sample: Sample,
    *,
    team_scale: float,
    players_scale: float,
    form_cfg: Dict[str, Any],
) -> Optional[Tuple[List[float], float]]:
    """Build the feature vector + target for one sample. Returns (x, y) or None."""
    try:
        p_anchor, p_adj = _meta_probs_for_sample(
            sample, team_scale=team_scale, players_scale=players_scale, form_cfg=form_cfg
        )
    except Exception:
        return None
    if not (math.isfinite(p_anchor) and math.isfinite(p_adj)):
        return None

    rec_ev = _meta_recent_evidence_from_sample(sample)
    feat = build_meta_blend_c_model_features(
        p_anchor=p_anchor,
        p_adjusted=p_adj,
        n_team_eff=float(sample.n_team),
        n_players_eff=float(sample.n_players),
        recent_evidence=float(rec_ev),
        rd_conf=None,
        hybrid_agree=None,
        bo=int(getattr(sample, "bo", 1) or 1),
        is_playoffs=(str(getattr(sample, "phase", "")).strip().lower() == "playoffs"),
        patch_status=str(getattr(sample, "patch_status", "") or ""),
        interregional=bool(getattr(sample, "interregional", False)),
        unknown_macro=False,
        roster_changed=bool(getattr(sample, "roster_changed", False)),
    )
    if not bool(feat.get("ok")):
        return None
    feat_map = dict(feat.get("features") or {})
    try:
        x = [float(feat_map.get(name, 0.0)) for name in META_BLEND_C_MODEL_FEATURES]
        y = int(sample.y)
    except (TypeError, ValueError, AttributeError):
        return None
    if y not in (0, 1):
        return None
    if not all(math.isfinite(v) for v in x):
        return None
    return x, float(y)


def _stratified_time_holdout(samples: List[Sample], *, holdout_frac: float = 0.2) -> Tuple[List[int], List[int]]:
    """Time-ordered split: first 1-holdout_frac as train, last holdout_frac as test.

    Assumes samples are already sorted by date in the caller (fit_ml_params).
    Returns (train_idx, test_idx).
    """
    n = len(samples)
    if n < 30:
        return list(range(n)), []
    n_test = max(10, int(n * float(holdout_frac)))
    n_train = max(20, n - n_test)
    n_test = n - n_train
    return list(range(n_train)), list(range(n_train, n))


def _logloss_safe(y: List[int], p: List[float]) -> float:
    try:
        return float(logloss(y, p))
    except Exception:
        return float("nan")


def _evaluate(p_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    y_list = [int(v) for v in y_true.tolist()]
    p_list = [float(max(1e-6, min(1.0 - 1e-6, float(v)))) for v in p_pred.tolist()]
    return {
        "logloss": _logloss_safe(y_list, p_list),
        "brier": float(brier(y_list, p_list)) if y_list else float("nan"),
        "n": len(y_list),
    }


def _fit_meta_blend_c_model(
    samples: List[Sample],
    *,
    team_scale: float = 1.0,
    players_scale: float = 1.0,
    min_train_games: int = 30,
    form_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Train the Phase C meta-blend logistic regression.

    Returns a dict in the schema expected by apply_meta_blend_c_model:
      {
        "enabled": bool,
        "version": "meta_blend_c_model_v3",
        "features": [...],
        "coef": [...],
        "intercept": float,
        "x_mean": [...], "x_std": [...],
        "n_total": int,
        "improvement_vs_anchor": float,
        "improvement_vs_adjusted": float,
        "oos_logloss": float,
        "runtime_guard": {...},
        "runtime_stability": {...},
        "meta": {...},
      }
    """
    form_cfg = dict(form_cfg or {})
    feature_names = list(META_BLEND_C_MODEL_FEATURES)
    out_template: Dict[str, Any] = {
        "enabled": False,
        "version": "meta_blend_c_model_v3",
        "features": list(feature_names),
        "coef": [0.0] * len(feature_names),
        "intercept": 0.0,
        "x_mean": [0.0] * len(feature_names),
        "x_std": [1.0] * len(feature_names),
        "n_total": 0,
        "improvement_vs_anchor": float("nan"),
        "improvement_vs_adjusted": float("nan"),
        "oos_logloss": float("nan"),
        "runtime_guard": {
            "enabled": True,
            "min_n_total": 220.0,
            "min_improvement_vs_adjusted": -0.003,
            "min_improvement_vs_anchor": -0.0015,
            "allow_context_override": True,
        },
        "runtime_stability": {
            "enabled": True,
            "cap_pp_lo": 2.2,
            "cap_pp_hi_bo1": 5.2,
            "cap_pp_hi_bo3": 6.3,
            "cap_pp_hi_bo5": 7.2,
            "small_delta_logit": 0.08,
            "tiny_delta_logit": 0.03,
            "cap_pp_small_delta": 3.2,
            "cap_pp_tiny_delta": 2.2,
            "mult_interregional": 0.88,
            "mult_patch_refit": 0.86,
            "mult_patch_warn": 0.94,
            "mult_roster_changed": 0.93,
            "weak_imp_threshold": -0.001,
            "mult_weak_improvement": 0.92,
        },
        "context_models": {},
        "meta": {
            "rows_total": 0,
            "n_train": 0,
            "n_test": 0,
            "_source": "fit_meta_blend_c_model_rewrite_v1",
        },
    }

    if not samples:
        out_template["meta"]["reason"] = "no_samples"
        return out_template

    # Build feature matrix
    X_list: List[List[float]] = []
    y_list: List[float] = []
    p_anchor_list: List[float] = []
    p_adj_list: List[float] = []
    for s in samples:
        built = _features_for_sample(s, team_scale=team_scale, players_scale=players_scale, form_cfg=form_cfg)
        if built is None:
            continue
        x, y = built
        X_list.append(x)
        y_list.append(y)
        # Capture anchor/adjusted for OOS comparison
        try:
            p_anchor, p_adj = _meta_probs_for_sample(s, team_scale=team_scale,
                                                       players_scale=players_scale,
                                                       form_cfg=form_cfg)
            p_anchor_list.append(float(p_anchor))
            p_adj_list.append(float(p_adj))
        except Exception:
            p_anchor_list.append(0.5)
            p_adj_list.append(0.5)

    n_total = len(X_list)
    out_template["n_total"] = int(n_total)
    out_template["meta"]["rows_total"] = int(n_total)
    if n_total < int(min_train_games):
        out_template["meta"]["reason"] = f"insufficient_samples ({n_total} < {min_train_games})"
        return out_template

    # Time-ordered split (samples already sorted by date upstream)
    _train_idx, _test_idx = _stratified_time_holdout(samples[:n_total], holdout_frac=0.2)
    # We need to also reindex into our X_list / y_list which dropped invalid samples.
    # The simplest approach: refit on full data + use last 20% of OUR valid rows as test.
    # That way indexes always line up.
    if n_total >= 40:
        n_test = max(10, int(n_total * 0.2))
        train_slice = slice(0, n_total - n_test)
        test_slice = slice(n_total - n_test, n_total)
    else:
        train_slice = slice(0, n_total)
        test_slice = slice(n_total, n_total)  # empty

    X = np.asarray(X_list, dtype=float)
    y = np.asarray(y_list, dtype=float)

    fit_res = _fit_logreg_numpy(X[train_slice], y[train_slice], l2=0.10, lr=0.08, epochs=260)
    if fit_res.get("error"):
        out_template["meta"]["reason"] = str(fit_res.get("error"))
        return out_template

    # Populate trained coefs
    out_template["coef"] = list(fit_res.get("coef") or [])
    out_template["intercept"] = float(fit_res.get("intercept", 0.0))
    out_template["x_mean"] = list(fit_res.get("x_mean") or [])
    out_template["x_std"] = list(fit_res.get("x_std") or [])
    out_template["enabled"] = True
    out_template["meta"]["n_train"] = int(fit_res.get("n_train") or 0)
    out_template["meta"]["train_logloss"] = float(fit_res.get("train_logloss", float("nan")))
    out_template["meta"]["train_brier"] = float(fit_res.get("train_brier", float("nan")))

    # OOS evaluation: predict on test slice, compare to anchor and adjusted baselines
    if test_slice.stop > test_slice.start:
        X_test = X[test_slice]
        y_test = y[test_slice]
        p_pred = _predict_logreg_numpy(fit_res, X_test)
        eval_pred = _evaluate(p_pred, y_test)
        eval_anchor = _evaluate(np.asarray(p_anchor_list[test_slice], dtype=float), y_test)
        eval_adj = _evaluate(np.asarray(p_adj_list[test_slice], dtype=float), y_test)

        out_template["oos_logloss"] = eval_pred.get("logloss", float("nan"))
        # improvement = baseline_logloss - pred_logloss (positive = pred is better)
        if math.isfinite(eval_pred["logloss"]) and math.isfinite(eval_anchor["logloss"]):
            out_template["improvement_vs_anchor"] = eval_anchor["logloss"] - eval_pred["logloss"]
        if math.isfinite(eval_pred["logloss"]) and math.isfinite(eval_adj["logloss"]):
            out_template["improvement_vs_adjusted"] = eval_adj["logloss"] - eval_pred["logloss"]
        out_template["meta"]["n_test"] = int(eval_pred.get("n", 0))
        out_template["meta"]["test_logloss"] = float(eval_pred.get("logloss", float("nan")))
        out_template["meta"]["test_anchor_logloss"] = float(eval_anchor.get("logloss", float("nan")))
        out_template["meta"]["test_adjusted_logloss"] = float(eval_adj.get("logloss", float("nan")))
        out_template["meta"]["test_brier"] = float(eval_pred.get("brier", float("nan")))

    return out_template


__all__ = ["_fit_meta_blend_c_model"]
