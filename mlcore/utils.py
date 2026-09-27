from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np


def safe_float(value: float, default: float = float("nan")) -> float:
    """Convert value to float, returning default if conversion fails or result is non-finite."""
    try:
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def safe_float_0(value: Any) -> float:
    """safe_float with default=0.0; for score calculations where missing means zero."""
    return safe_float(value, 0.0)


def safe_float_or_none(value: object) -> Optional[float]:
    """Return a finite float, or None if the value is invalid or non-finite."""
    try:
        out = float(value)  # type: ignore[arg-type]
        return out if math.isfinite(out) else None
    except Exception:
        return None

def isfinite_num(value: object) -> bool:
    """Return True if value can be converted to a finite float."""
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except Exception:
        return False


def clip(x: float, lo: float, hi: float) -> float:
    """Clamp x to the closed interval [lo, hi]."""
    return lo if x < lo else hi if x > hi else x

def clip_safe(x: Any, lo: float, hi: float) -> float:
    """clip with arbitrary input; non-convertible or non-finite values map to lo."""
    return clip(safe_float(x, lo), lo, hi)

def clip_prob(p: float, eps: float = 1e-6) -> float:
    """Clamp probability p to (eps, 1-eps) to avoid log(0) in loss computations."""
    return clip(float(p), float(eps), 1.0 - float(eps))

def wmean(v: np.ndarray, w: np.ndarray) -> float:
    """Weighted mean over numpy arrays; skips non-finite values and zero/negative weights."""
    m = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not m.any():
        return float("nan")
    s = float(np.sum(w[m]))
    if s <= 1e-12:
        return float(np.mean(v[m]))
    return float(np.sum(v[m] * w[m]) / s)

def sigmoid(z: float) -> float:
    """Numerically stable sigmoid: 1 / (1 + exp(-z))."""
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

def logit(p: float, eps: float = 1e-12) -> float:
    """Log-odds transform: log(p / (1-p)), clipped to avoid infinities."""
    p = clip(p, eps, 1.0 - eps)
    return math.log(p / (1.0 - p))

def prob_to_decimal_odds(p: float, eps: float = 1e-6) -> float:
    """Convert win probability to decimal odds (1/p), clipped away from 0."""
    p2 = clip_prob(p, eps=eps)
    return float(1.0 / p2)

def decimal_odds_to_prob(odds: float) -> float:
    """Convert decimal odds to implied probability (1/odds). Returns NaN for odds <= 1."""
    o = float(odds)
    if not math.isfinite(o) or o <= 1.0:
        return float("nan")
    return float(1.0 / o)

def logloss(y_true: Iterable[int], p: Iterable[float], eps: float = 1e-12) -> float:
    """Binary cross-entropy loss averaged over all samples."""
    total = 0.0
    n = 0
    for yt, pr in zip(y_true, p):
        pr = clip(float(pr), eps, 1.0 - eps)
        total += -(yt * math.log(pr) + (1 - yt) * math.log(1.0 - pr))
        n += 1
    return total / max(n, 1)

def coerce_loud(series: Any, *, nome: str, default: float,
                limite_ruim: float = 0.20, avisar: Any = None) -> Any:
    """to_numeric(errors="coerce").fillna(default), MAS gritando quando o
    preenchimento vira regra em vez de exceção.

    Por que existe (varredura de 16/08/2026): o par coerce+fillna aparece 99 vezes
    no projeto e é silencioso por construção — o que não dá pra ler vira NaN, e o
    NaN vira um número plausível. Numa coluna de probabilidade isso significa que
    "a coluna mudou de nome" se transforma em "todo mundo com 0.5", e o calibrador
    é ajustado em cima disso sem uma linha de aviso.

    O default neutro continua certo para um caso isolado; o que estava faltando é
    perceber quando ele deixou de ser isolado. Acima de `limite_ruim` (20% por
    padrão) isto imprime — o dado segue passando, mas para de passar CALADO.
    """
    import pandas as pd  # local: utils é importado em caminho quente

    s = pd.to_numeric(series, errors="coerce")
    n = int(len(s))
    if n:
        ruins = int(s.isna().sum())
        frac = ruins / n
        if frac > float(limite_ruim):
            msg = (f"[coerce_loud] '{nome}': {ruins}/{n} valores ({100*frac:.0f}%) "
                   f"não puderam ser lidos e viraram {default}. "
                   f"Coluna mudou de formato?")
            (avisar or print)(msg)
    return s.fillna(default)


def brier(y_true: Iterable[int], p: Iterable[float]) -> float:
    """Brier score: mean squared error between predicted probabilities and binary outcomes."""
    total = 0.0
    n = 0
    for yt, pr in zip(y_true, p):
        d = float(pr) - float(yt)
        total += d * d
        n += 1
    return total / max(n, 1)
