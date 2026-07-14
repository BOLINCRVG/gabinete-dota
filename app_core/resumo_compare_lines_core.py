"""Compare multiple lines of the same market (sweet-spot finder).

When the user enters several lines for one metric (e.g. Kills Under 27.5 /
28.5 / 29.5), this groups them and builds a compact comparison so they can
see which line offers the best trade-off: the historical probability
(1/odd_hist) per line, plus an inline sparkline.

Pure + deterministic: takes resumo rows, returns plain structures the UI renders.
"""
from __future__ import annotations

from typing import Iterable, Optional


_SPARK_TICKS = "▁▂▃▄▅▆▇█"


def _odd_to_prob(odd) -> Optional[float]:
    try:
        o = float(odd)
        if o <= 1.0:
            return None
        return 1.0 / o
    except (TypeError, ValueError):
        return None


def _metric_leaf(metric_raw: str) -> str:
    m = str(metric_raw or "")
    if m.endswith("_ml_A") or m == "ml_A":
        return "ml"
    if m.endswith("_ml_B") or m == "ml_B":
        return "ml"
    if m.startswith("zz_mltot_"):
        return "combo"
    return m


def _side_norm(lado: str) -> str:
    s = str(lado or "").strip().upper()
    if s in ("ACIMA", "OVER"):
        return "Over"
    if s in ("ABAIXO", "UNDER"):
        return "Under"
    return str(lado or "")


def build_line_comparisons(rows: Iterable[dict]) -> list[dict]:
    """Group rows by (metric_leaf, side) and return groups with 2+ lines.

    Each group:
      {
        "metric": "kills", "side": "Under",
        "lines": [
            {"line": 27.5, "odd_hist": 1.78, "p_hist": 0.562, "row_key": "..."},
            ...
        ] sorted by line ascending,
        "best_idx": int,   # index of the line with the highest p_hist
      }
    Only totals markets (with a numeric line) are grouped; ML/combos skipped.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        leaf = _metric_leaf(r.get("_metric") or "")
        if leaf in ("ml", "combo", ""):
            continue
        line_val = r.get("_line")
        try:
            line = float(line_val) if line_val is not None else None
        except (TypeError, ValueError):
            line = None
        if line is None:
            continue
        side = _side_norm(r.get("Lado"))
        odd_hist = r.get("Odd (Hist)")
        p_hist = _odd_to_prob(odd_hist)
        key = (leaf, side)
        groups.setdefault(key, []).append({
            "line": line,
            "odd_hist": (float(odd_hist) if odd_hist not in (None, "", "-") else None),
            "p_hist": p_hist,
            "row_key": f"{r.get('Mercado','')}|{r.get('Lado','')}|{r.get('Linha','')}",
        })

    out: list[dict] = []
    for (metric, side), lines in groups.items():
        if len(lines) < 2:
            continue
        lines_sorted = sorted(lines, key=lambda d: d["line"])
        # Best = highest historical probability (safest line by Hist).
        best_idx = 0
        best_p = -1.0
        for i, ln in enumerate(lines_sorted):
            p = ln.get("p_hist")
            if p is not None and p > best_p:
                best_p = p
                best_idx = i
        out.append({
            "metric": metric,
            "side": side,
            "lines": lines_sorted,
            "best_idx": best_idx,
        })
    # Stable order: by metric name then side.
    out.sort(key=lambda g: (g["metric"], g["side"]))
    return out


def sparkline(values: list[Optional[float]]) -> str:
    """Render a unicode sparkline from a list of values (None → space).

    Scales min..max across the finite values to the 8 tick glyphs.
    """
    finite = [v for v in values if isinstance(v, (int, float))]
    if not finite:
        return ""
    lo, hi = min(finite), max(finite)
    span = hi - lo
    out_chars: list[str] = []
    for v in values:
        if not isinstance(v, (int, float)):
            out_chars.append(" ")
            continue
        if span <= 1e-9:
            out_chars.append(_SPARK_TICKS[len(_SPARK_TICKS) // 2])
            continue
        frac = (v - lo) / span
        idx = round(frac * (len(_SPARK_TICKS) - 1))
        idx = max(0, min(len(_SPARK_TICKS) - 1, idx))
        out_chars.append(_SPARK_TICKS[idx])
    return "".join(out_chars)


__all__ = ["build_line_comparisons", "sparkline"]
