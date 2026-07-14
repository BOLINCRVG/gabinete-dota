from __future__ import annotations

import math
from .utils import clip

def p_series_from_pmap(p_map: float, bo: int) -> float:
    """Convert per-map win probability to series win probability.

    bo must be an odd integer >= 1.
    """
    p = clip(float(p_map), 1e-12, 1.0 - 1e-12)
    b = int(bo)
    if b < 1 or (b % 2) == 0:
        raise ValueError("bo must be an odd integer >= 1")
    if b == 1:
        return p

    need = (b // 2) + 1
    q = 1.0 - p
    s = 0.0
    for k in range(need, b + 1):
        s += math.comb(b, k) * (p ** k) * (q ** (b - k))
    return float(s)

def prob_win_series(p_map: float, bo: int) -> float:
    """Backward-compatible alias for :func:`p_series_from_pmap`.

    Some UI modules import `prob_win_series`; keep it stable.
    """
    return p_series_from_pmap(p_map, bo)
