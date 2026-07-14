"""Shared numeric/probability helpers for app_core layers."""

from __future__ import annotations

import math

from mlcore.utils import clip_prob, isfinite_num, safe_float
from app_core.ml_probability_adjustments_core import safe_sigmoid

# Re-exported so callers can `from app_core.prob_math_core import safe_float/...`.
# Listed in __all__ so linters don't strip them as "unused".
__all__ = [
    "clip_prob",
    "isfinite_num",
    "safe_float",
    "safe_logit",
    "safe_sigmoid",
]


def safe_logit(p: float) -> float:
    pp = clip_prob(float(p))
    return float(math.log(pp / (1.0 - pp)))
