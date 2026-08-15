from __future__ import annotations

import math
from typing import Any, List, Optional


# Canonical labels shared by UI layers to avoid duplicated literals in plays_app.py.
MARKET_TAB_NAMES: List[str] = [
    "Win Rate",
    "Kills",
    "Torres",
    "Dragões",
    "Dragões Elementais",
    "Barões",
    "Inibidores",
    "Tempo",
    "Kills p/ Time",
    "Handicaps",
    "Primeiros Objetivos",
    "Stats",
    "Perfil de Oponente",
    "ML + Totais",
    "Detalhes dos Jogos",
    "Seletor Rápido",
    "Resumo",
]


METRIC_PT = {
    "kills": "Kills",
    "towers": "Torres",
    "dragons": "Dragões",
    "barons": "Barões",
    "inhibitors": "Inibidores",
    "time": "Tempo",
}

SIDE_PT = {
    "over": "Acima",
    "under": "Abaixo",
}


def fmt_prob_pct(p: Optional[float]) -> str:
    """Format a probability in [0, 1] as a percentage string: '64.3%'."""
    if p is None or not math.isfinite(p):
        return "-"
    return f"{100.0 * p:.1f}%"


def fmt_num(x: Optional[float], nd: int = 2) -> str:
    if x is None or not math.isfinite(x):
        return "-"
    return f"{x:.{nd}f}"


def min_to_mmss(x: Optional[float]) -> str:
    if x is None or not math.isfinite(x):
        return "-"
    sec = round(float(x) * 60.0)
    m = sec // 60
    s = sec % 60
    return f"{m:02d}:{s:02d}"


def metric_label_pt(metric_key: str) -> str:
    k = str(metric_key).strip().lower()
    return METRIC_PT.get(k, str(metric_key).strip().title())


def side_label_pt(side: str) -> str:
    s = str(side).strip().lower()
    return SIDE_PT.get(s, str(side).strip().title())


def fmt_line_disp(metric_key: str, line_val: Any) -> str:
    """Format a line value for display. Time is shown as MM:SS."""
    try:
        v = float(line_val)
    except (TypeError, ValueError):
        return str(line_val)
    if str(metric_key).strip().lower() == "time":
        return min_to_mmss(v)
    if abs(v - round(v)) < 1e-9:
        return str(round(v))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def fmt_recortes_disp(items: Any) -> str:
    """Format a list of filter segments for display: ['Ano', '5'] → 'Ano, 5'."""
    if not items:
        return "-"
    out = []
    for it in items:
        s = str(it).strip()
        if s.lower() == "ano":
            s = "Ano"
        out.append(s)
    return ", ".join(out) if out else "-"


def fmt_bytes_short(n: Any) -> str:
    import math as _math
    try:
        x = float(n)
    except (TypeError, ValueError):
        return "-"
    if not _math.isfinite(x) or x < 0:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while x >= 1024.0 and i < (len(units) - 1):
        x /= 1024.0
        i += 1
    if i == 0:
        return f"{int(x)} {units[i]}"
    return f"{x:.1f} {units[i]}"


__all__ = [
    "MARKET_TAB_NAMES",
    "METRIC_PT",
    "SIDE_PT",
    "fmt_bytes_short",
    "fmt_line_disp",
    "fmt_num",
    "fmt_prob_pct",
    "fmt_recortes_disp",
    "metric_label_pt",
    "min_to_mmss",
    "side_label_pt",
]
