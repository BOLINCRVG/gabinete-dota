from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Any

from .utils import sigmoid, logit

@dataclass
class PlattCalibrator:
    """Platt scaling: p = sigmoid(a * x + b), where x is a score (logit of raw p)."""
    a: float = 1.0
    b: float = 0.0

    def predict_proba(self, p_raw: float) -> float:
        x = logit(p_raw)
        return sigmoid(self.a * x + self.b)

    def to_dict(self) -> Dict[str, Any]:
        return {"a": float(self.a), "b": float(self.b)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlattCalibrator":
        return cls(a=float(d.get("a", 1.0)), b=float(d.get("b", 0.0)))


def fit_platt_from_raw_probs(
    y_true: Iterable[int],
    p_raw: Iterable[float],
    *,
    l2: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> PlattCalibrator:
    """Fit Platt scaling parameters a,b using Newton-Raphson on logistic loss.

    We use x = logit(p_raw) as input feature to keep it stable and meaningful.
    Adds small L2 regularization for stability.
    """
    xs = []
    ys = []
    for y, p in zip(y_true, p_raw):
        ys.append(1 if int(y) == 1 else 0)
        xs.append(logit(float(p)))

    n = len(xs)
    if n == 0:
        return PlattCalibrator()
    # If labels are all same, calibration is undefined; return identity.
    if sum(ys) == 0 or sum(ys) == n:
        return PlattCalibrator()

    a = 1.0
    b = 0.0

    for _ in range(max_iter):
        # gradients and Hessian
        ga = 0.0
        gb = 0.0
        haa = l2
        hab = 0.0
        hbb = l2
        for x, y in zip(xs, ys):
            z = a * x + b
            p = sigmoid(z)
            w = p * (1.0 - p)
            ga += (p - y) * x
            gb += (p - y)
            haa += w * x * x
            hab += w * x
            hbb += w

        # Newton step solve 2x2
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = ( hbb * ga - hab * gb) / det
        db = (-hab * ga + haa * gb) / det

        a_new = a - da
        b_new = b - db

        if abs(a_new - a) < tol and abs(b_new - b) < tol:
            a, b = a_new, b_new
            break

        a, b = a_new, b_new

    return PlattCalibrator(a=a, b=b)
