"""Unified ML evaluation harness: holdout splits, bootstrap CI, and acceptance gates.

This package is the foundation for Fase 0+ of the ML quality roadmap. Every new
layer/feature/model must pass through these utilities before being shipped.

Modules
-------
- splits: chronological train/val/test split helpers
- metrics: logloss, ECE, brier with bootstrap CI
- gate: layer acceptance gate (delta_logloss > threshold, p-value < alpha, ECE not worse)

Usage
-----
    from mlcore.ml_eval import build_split, metrics_with_ci, acceptance_gate

    split = build_split(samples, train_years=(2024, 2025), val_year=2026, val_cutoff="2026-06-30")
    metrics = metrics_with_ci(y=split.test.y, p=p_baseline, n_bootstrap=1000)
    decision = acceptance_gate(metrics_baseline, metrics_after, alpha=0.05, min_delta=0.002)
"""
from __future__ import annotations

from mlcore.ml_eval.splits import HoldoutSplit, SampleBucket, build_split
from mlcore.ml_eval.metrics import MetricsCI, bootstrap_ci, ece_score, metrics_with_ci
from mlcore.ml_eval.gate import GateDecision, acceptance_gate

__all__ = [
    "GateDecision",
    "HoldoutSplit",
    "MetricsCI",
    "SampleBucket",
    "acceptance_gate",
    "bootstrap_ci",
    "build_split",
    "ece_score",
    "metrics_with_ci",
]
