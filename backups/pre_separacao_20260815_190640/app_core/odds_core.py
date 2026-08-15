"""Shared odds helpers extracted from plays_app.

This module keeps pure pricing/calibration primitives so UI tabs can reuse
the same logic without copy/paste.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional

import pandas as pd
from app_core.market import prob_from_odd as market_prob_from_odd
from mlcore.utils import prob_to_decimal_odds


def laplace_odds(k: int, n: int, *, floor_odd: float = 1.10) -> Optional[float]:
    """Laplace-smoothed odds: (n+2)/(k+1) clamped to floor_odd. Returns None when n<=0."""
    if int(n) <= 0:
        return None
    odd = float((float(n) + 2.0) / (float(k) + 1.0))
    return float(max(float(floor_odd), odd))


def format_laplace_filtered(
    odds_dict: Dict[str, Any],
    prefix: str,
    side_key: str,
    odd_max: float,
    pick: Optional[str] = None,
) -> str:
    try:
        om = float(odd_max)
    except (TypeError, ValueError):
        om = 999.0
    items = []
    for rec in ("Ano", "5", "10", "15"):
        v = odds_dict.get(f"{prefix}_{rec}_{side_key}")
        if v is None:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if (not math.isfinite(vf)) or vf > om:
            continue
        items.append((rec, vf))
    pick_txt = None
    if pick is not None:
        pick_label = str(pick).strip() or "Liga 15"
        try:
            pick_odd = laplace_pick_odd(odds_dict, side_key, pick_label)
            if pick_odd is None:
                pick_txt = f"pick={pick_label}=-"
            else:
                pick_txt = f"pick={pick_label}={float(pick_odd):.2f}"
        except (IndexError, TypeError, ValueError):
            pick_txt = f"pick={pick_label}=-"
    if not items:
        if pick_txt:
            return f"mu recortes=- | {pick_txt}"
        return "-"
    mu = sum(v for _, v in items) / float(len(items))
    parts = [f"mu recortes={mu:.2f}"]
    for rec, vf in items:
        parts.append(f"{rec}={vf:.2f}")
    if pick_txt:
        parts.append(pick_txt)
    return " | ".join(parts)


def format_scope_odds_clean(
    odds_dict: Dict[str, Any],
    prefix: str,
    side_key: str,
    odd_max: float,
) -> str:
    """User-facing odds line for ONE scope (Liga or H2H), no internal jargon.

    Shows the odds of the recortes (Ano/15/10/5) of THIS scope that are within
    the odd limit, plus their average. There is no "pick" concept — the odds
    are always shown; the limit only decides which recortes qualify (mirroring
    the ✅/❌ badges next to it). Uses the scope's own odds (H2H shows H2H odds,
    not the global pick), fixing the "pick=Liga 15 on an H2H line" bug.

    Returns e.g. "méd 1.62 · Ano 1.55 · 15 1.70" or "nenhum recorte ≤ limite".
    """
    try:
        om = float(odd_max)
    except (TypeError, ValueError):
        om = 999.0
    items: list[tuple[str, float]] = []
    for rec in ("Ano", "15", "10", "5"):
        v = odds_dict.get(f"{prefix}_{rec}_{side_key}")
        if v is None:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if (not math.isfinite(vf)) or vf > om:
            continue
        items.append((rec, vf))
    if not items:
        return "nenhum recorte ≤ limite"
    mu = sum(v for _, v in items) / float(len(items))
    parts = [f"méd {mu:.2f}"]
    for rec, vf in items:
        parts.append(f"{rec} {vf:.2f}")
    return " · ".join(parts)


def odd_to_prob(odd: Any) -> Optional[float]:
    """Convert decimal odds to implied probability. Returns None for invalid or non-finite inputs."""
    try:
        return market_prob_from_odd(float(odd))
    except (TypeError, ValueError):
        return None


def prob_to_odd(p: Any) -> Optional[float]:
    """Convert probability to decimal odds. Returns None for non-finite inputs."""
    try:
        pp = float(p)
        if not math.isfinite(pp):
            return None
        return float(prob_to_decimal_odds(pp))
    except (TypeError, ValueError):
        return None


def laplace_pick_odd(odds_dict: Dict[str, Any], side_key: str, pick: str) -> Optional[float]:
    def _get(scope: str, rec: str) -> Optional[float]:
        return odds_dict.get(f"{scope}_{rec}_{side_key}")

    def _norm_rec(rec: str) -> str:
        r = str(rec).strip()
        return "Ano" if r.lower() == "ano" else r

    pick = str(pick or "Liga 15")
    if pick.startswith("Liga "):
        rec = _norm_rec(pick.split(" ", 1)[1])
        return _get("Liga", rec)
    if pick.startswith("H2H "):
        rec = _norm_rec(pick.split(" ", 1)[1])
        return _get("H2H", rec)
    if pick.startswith(("Média ", "MÃ©dia ", "Media ")):
        rec = _norm_rec(pick.split(" ", 1)[1].split()[-1])
        o1 = _get("Liga", rec)
        o2 = _get("H2H", rec)
        p1 = odd_to_prob(o1)
        p2 = odd_to_prob(o2)
        if p1 is None and p2 is None:
            return None
        if p1 is None:
            return o2
        if p2 is None:
            return o1
        return prob_to_odd(0.5 * (p1 + p2))
    return _get("Liga", "15")


def odd_mix(
    odd_model: Any,
    odd_hist: Any,
    stats_pct: Any,
    *,
    w_model: float,
    w_hist: float,
    w_stats: float,
    model_boost: float = 1.0,
) -> Optional[float]:
    p_m = odd_to_prob(odd_model)
    p_h = odd_to_prob(odd_hist)
    p_s = None
    try:
        if stats_pct is not None and math.isfinite(float(stats_pct)):
            p_s = float(stats_pct) / 100.0
    except Exception:
        p_s = None

    sources = []
    if p_m is not None:
        sources.append((p_m, float(w_model) * max(0.0, float(model_boost))))
    if p_h is not None:
        sources.append((p_h, float(w_hist)))
    if p_s is not None:
        sources.append((p_s, float(w_stats)))
    if not sources:
        return None

    total_w = sum(max(0.0, w) for _, w in sources)
    if total_w <= 0:
        total_w = float(len(sources))
        sources = [(p, 1.0) for p, _ in sources]
    p_mix = sum(p * max(0.0, w) for p, w in sources) / total_w
    return prob_to_odd(p_mix)


def edge_pp_from_odds(odd_mix_v: Any, odd_hist_v: Any, odd_model_v: Any) -> Optional[float]:
    p_mix = odd_to_prob(odd_mix_v)
    p_hist = odd_to_prob(odd_hist_v)
    p_model = odd_to_prob(odd_model_v)
    if p_mix is None:
        return None
    baseline = p_hist if p_hist is not None else p_model
    if baseline is None:
        return None
    return float((p_mix - baseline) * 100.0)


def _fmt_count_odd(k: int, n: int, floor_odd: float) -> str:
    if int(n) <= 0:
        return "-"
    odd = laplace_odds(k, n, floor_odd=floor_odd)
    if odd is None:
        return f"{k}/{n}"
    return f"{k}/{n} ({float(odd):.2f}x)"


def ou_block(
    df_l_year: pd.DataFrame,
    df_l_15: pd.DataFrame,
    df_l_10: pd.DataFrame,
    df_l_5: pd.DataFrame,
    df_h_year: pd.DataFrame,
    df_h_15: pd.DataFrame,
    df_h_10: pd.DataFrame,
    df_h_5: pd.DataFrame,
    *,
    value_col: str,
    line: float,
    floor_odd: float = 1.10,
) -> pd.DataFrame:
    def _counts(df: pd.DataFrame) -> tuple[int, int, int]:
        x = pd.to_numeric(df.get(value_col), errors="coerce").dropna()
        n = int(x.shape[0])
        if n <= 0:
            return 0, 0, 0
        over = int((x > float(line)).sum())
        under = int((x < float(line)).sum())
        return over, under, n

    l_over_y, l_under_y, l_ny = _counts(df_l_year)
    l_over_15, l_under_15, l_n15 = _counts(df_l_15)
    l_over_10, l_under_10, l_n10 = _counts(df_l_10)
    l_over_5, l_under_5, l_n5 = _counts(df_l_5)

    h_over_y, h_under_y, h_ny = _counts(df_h_year)
    h_over_15, h_under_15, h_n15 = _counts(df_h_15)
    h_over_10, h_under_10, h_n10 = _counts(df_h_10)
    h_over_5, h_under_5, h_n5 = _counts(df_h_5)

    return pd.DataFrame(
        {
            "Ano todo": [
                f"Over: {_fmt_count_odd(l_over_y, l_ny, floor_odd)}",
                f"Under: {_fmt_count_odd(l_under_y, l_ny, floor_odd)}",
                f"Over: {_fmt_count_odd(h_over_y, h_ny, floor_odd)}",
                f"Under: {_fmt_count_odd(h_under_y, h_ny, floor_odd)}",
            ],
            "Últimos 15": [
                f"Over: {_fmt_count_odd(l_over_15, l_n15, floor_odd)}",
                f"Under: {_fmt_count_odd(l_under_15, l_n15, floor_odd)}",
                f"Over: {_fmt_count_odd(h_over_15, h_n15, floor_odd)}",
                f"Under: {_fmt_count_odd(h_under_15, h_n15, floor_odd)}",
            ],
            "Últimos 10": [
                f"Over: {_fmt_count_odd(l_over_10, l_n10, floor_odd)}",
                f"Under: {_fmt_count_odd(l_under_10, l_n10, floor_odd)}",
                f"Over: {_fmt_count_odd(h_over_10, h_n10, floor_odd)}",
                f"Under: {_fmt_count_odd(h_under_10, h_n10, floor_odd)}",
            ],
            "Últimos 5": [
                f"Over: {_fmt_count_odd(l_over_5, l_n5, floor_odd)}",
                f"Under: {_fmt_count_odd(l_under_5, l_n5, floor_odd)}",
                f"Over: {_fmt_count_odd(h_over_5, h_n5, floor_odd)}",
                f"Under: {_fmt_count_odd(h_under_5, h_n5, floor_odd)}",
            ],
        },
        index=["Liga - Over", "Liga - Under", "H2H - Over", "H2H - Under"],
    )


def event_block(
    df_l_year: pd.DataFrame,
    df_l_15: pd.DataFrame,
    df_l_10: pd.DataFrame,
    df_l_5: pd.DataFrame,
    df_h_year: pd.DataFrame,
    df_h_15: pd.DataFrame,
    df_h_10: pd.DataFrame,
    df_h_5: pd.DataFrame,
    *,
    event_fn: Callable[[pd.DataFrame], pd.Series],
    label_yes: str = "Yes",
    label_no: str = "No",
    floor_odd: float = 1.10,
) -> pd.DataFrame:
    def _counts(df: pd.DataFrame) -> tuple[int, int, int]:
        ok = event_fn(df).fillna(False).astype(bool)
        n = int(ok.shape[0])
        if n <= 0:
            return 0, 0, 0
        y = int(ok.sum())
        no = int(n - y)
        return y, no, n

    l_y, l_no, l_n = _counts(df_l_year)
    l_y15, l_no15, l_n15 = _counts(df_l_15)
    l_y10, l_no10, l_n10 = _counts(df_l_10)
    l_y5, l_no5, l_n5 = _counts(df_l_5)

    h_y, h_no, h_n = _counts(df_h_year)
    h_y15, h_no15, h_n15 = _counts(df_h_15)
    h_y10, h_no10, h_n10 = _counts(df_h_10)
    h_y5, h_no5, h_n5 = _counts(df_h_5)

    return pd.DataFrame(
        {
            "Ano todo": [
                f"{label_yes}: {_fmt_count_odd(l_y, l_n, floor_odd)}",
                f"{label_no}: {_fmt_count_odd(l_no, l_n, floor_odd)}",
                f"{label_yes}: {_fmt_count_odd(h_y, h_n, floor_odd)}",
                f"{label_no}: {_fmt_count_odd(h_no, h_n, floor_odd)}",
            ],
            "Últimos 15": [
                f"{label_yes}: {_fmt_count_odd(l_y15, l_n15, floor_odd)}",
                f"{label_no}: {_fmt_count_odd(l_no15, l_n15, floor_odd)}",
                f"{label_yes}: {_fmt_count_odd(h_y15, h_n15, floor_odd)}",
                f"{label_no}: {_fmt_count_odd(h_no15, h_n15, floor_odd)}",
            ],
            "Últimos 10": [
                f"{label_yes}: {_fmt_count_odd(l_y10, l_n10, floor_odd)}",
                f"{label_no}: {_fmt_count_odd(l_no10, l_n10, floor_odd)}",
                f"{label_yes}: {_fmt_count_odd(h_y10, h_n10, floor_odd)}",
                f"{label_no}: {_fmt_count_odd(h_no10, h_n10, floor_odd)}",
            ],
            "Últimos 5": [
                f"{label_yes}: {_fmt_count_odd(l_y5, l_n5, floor_odd)}",
                f"{label_no}: {_fmt_count_odd(l_no5, l_n5, floor_odd)}",
                f"{label_yes}: {_fmt_count_odd(h_y5, h_n5, floor_odd)}",
                f"{label_no}: {_fmt_count_odd(h_no5, h_n5, floor_odd)}",
            ],
        },
        index=[
            f"Liga - {label_yes}",
            f"Liga - {label_no}",
            f"H2H - {label_yes}",
            f"H2H - {label_no}",
        ],
    )


def empty_like(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df.iloc[0:0].copy()
    return pd.DataFrame([])


def tail_df(df: Optional[pd.DataFrame], n: int) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return empty_like(df)
    return df.tail(int(n)).copy()


def recency_slices(
    *,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    df_h2h: pd.DataFrame,
    id_col: str = "gameid",
    date_col: str = "date",
    dedupe_league: bool = True,
    backfill_unique_targets: bool = False,
    df_a_year: pd.DataFrame | None = None,
    df_b_year: pd.DataFrame | None = None,
) -> Dict[str, pd.DataFrame]:
    """df_a/df_b alimentam as janelas RECENTES (15/10/5). Quando df_a_year/
    df_b_year vierem, a fatia "Ano" usa ELES (regra do usuário: Ano todo = ano
    inteiro sem filtro de liga; recentes = recorte/campeonato mais recente)."""
    def _league_slice(*parts: pd.DataFrame) -> pd.DataFrame:
        out = pd.concat(list(parts), ignore_index=True)
        if bool(dedupe_league):
            out = out.drop_duplicates(subset=[id_col], keep="last")
        return out.sort_values(date_col)

    def _league_recent_slice(n: int) -> pd.DataFrame:
        if not bool(dedupe_league):
            return _league_slice(df_a.tail(int(n)), df_b.tail(int(n)))
        if not bool(backfill_unique_targets):
            return _league_slice(df_a.tail(int(n)), df_b.tail(int(n)))

        a_sorted = df_a.sort_values(date_col)
        b_sorted = df_b.sort_values(date_col)
        take_a = min(int(n), int(a_sorted.shape[0]))
        take_b = min(int(n), int(b_sorted.shape[0]))
        max_a = int(a_sorted.shape[0])
        max_b = int(b_sorted.shape[0])
        target = int(n) * 2
        total_unique = int(
            pd.concat([a_sorted, b_sorted], ignore_index=True)
            .drop_duplicates(subset=[id_col], keep="last")
            .shape[0]
        )
        target = min(target, total_unique)

        while True:
            merged = _league_slice(a_sorted.tail(take_a), b_sorted.tail(take_b))
            if int(merged.shape[0]) >= target:
                return merged.tail(target).copy()
            grew = False
            if take_a < max_a:
                take_a += 1
                grew = True
            if int(_league_slice(a_sorted.tail(take_a), b_sorted.tail(take_b)).shape[0]) >= target:
                return _league_slice(a_sorted.tail(take_a), b_sorted.tail(take_b)).tail(target).copy()
            if take_b < max_b:
                take_b += 1
                grew = True
            if not grew:
                return merged

    _ya = df_a_year if isinstance(df_a_year, pd.DataFrame) and not df_a_year.empty else df_a
    _yb = df_b_year if isinstance(df_b_year, pd.DataFrame) and not df_b_year.empty else df_b
    league_year = _league_slice(_ya, _yb)
    league_15 = _league_recent_slice(15)
    league_10 = _league_recent_slice(10)
    league_5 = _league_recent_slice(5)
    h2h_year = (df_h2h if isinstance(df_h2h, pd.DataFrame) else pd.DataFrame([])).sort_values(date_col)
    h2h_15 = tail_df(h2h_year, 15).sort_values(date_col)
    h2h_10 = tail_df(h2h_year, 10).sort_values(date_col)
    h2h_5 = tail_df(h2h_year, 5).sort_values(date_col)
    return {
        "league_year": league_year,
        "league_15": league_15,
        "league_10": league_10,
        "league_5": league_5,
        "h2h_year": h2h_year,
        "h2h_15": h2h_15,
        "h2h_10": h2h_10,
        "h2h_5": h2h_5,
    }
