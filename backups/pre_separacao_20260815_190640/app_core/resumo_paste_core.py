"""Paste-capture helpers for the Resumo tab.

Lets the user paste a single text like:

    K27.5/1.85 K28.5/2.10 T11.5/1.65 D4.5/1.90 ML-A/1.45 TM33/1.95

and get back a normalized {(metric, side, line_str) -> odd} dict that the
renderer can cross-match against the rows to fill ``oc_map_cur`` in one shot.

Also exposes Kelly stake helper used by the recommendation cards.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Mapping, Optional


# --- Metric prefix dictionary (kept tiny on purpose). ---------------------
# Order matters: longer aliases first so "ML-A" doesn't match the "M" alone.
_METRIC_ALIASES: tuple[tuple[str, str], ...] = (
    ("MLA", "ml_A"),
    ("ML-A", "ml_A"),
    ("MLB", "ml_B"),
    ("ML-B", "ml_B"),
    ("KILLS", "kills"),
    ("TORRES", "towers"),
    ("TOWERS", "towers"),
    ("DRAKES", "dragons"),
    ("DRAGS", "dragons"),
    ("DRAGONS", "dragons"),
    ("BARONS", "barons"),
    ("BAROES", "barons"),
    ("INIBIDORES", "inhibitors"),
    ("INIBS", "inhibitors"),
    ("INIB", "inhibitors"),
    ("TEMPO", "time"),
    ("TIME", "time"),
    ("TM", "time"),
    ("K", "kills"),
    ("T", "towers"),
    ("D", "dragons"),
    ("B", "barons"),
    ("I", "inhibitors"),
)

# Some entries default to Over when only a line is given (the common case for
# totals markets — users usually shop the Over line). Under can be selected
# explicitly via a "U" suffix: "K27.5U/2.10".
_TOTALS_METRICS = frozenset({"kills", "towers", "dragons", "barons", "inhibitors", "time"})

_SEPARATORS = re.compile(r"[\s,;]+")
_PAIR = re.compile(r"^([A-Z\-]+)([0-9]+(?:[\.:][0-9]+)?)?([UO])?[/=@]([0-9]+(?:[\.,][0-9]+)?)$")


def _normalize_line_token(raw: Optional[str]) -> str:
    """Convert '32:30' to '32.5', '27.5' stays as is, missing returns ''."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    if ":" in s:
        try:
            m, sec = s.split(":", 1)
            return f"{int(m) + (int(sec) / 60.0):.2f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return s
    return s.replace(",", ".")


def _match_prefix(token: str) -> tuple[str, str, str]:
    """Return (metric_key, line_str, side_explicit) for one token.

    side_explicit is "Over" / "Under" / "" when the token has no U/O hint.
    metric_key is "" when the token doesn't match any known prefix.
    """
    t = token.upper().strip()
    if not t:
        return "", "", ""
    # First try the strict regex (PREFIX + optional line + optional U/O + odd separator handled by caller)
    # Here we just split into prefix and remainder for ambiguity-free parsing.
    for alias, key in _METRIC_ALIASES:
        if t.startswith(alias):
            tail = t[len(alias):]
            side = ""
            if tail.endswith("U"):
                side, tail = "Under", tail[:-1]
            elif tail.endswith("O"):
                side, tail = "Over", tail[:-1]
            return key, _normalize_line_token(tail), side
    return "", "", ""


def parse_odds_paste(text: str) -> dict[tuple[str, str, str], float]:
    """Parse a free-form odds paste into a normalized dict.

    Returns ``{(metric, side, line): odd}`` where:
      - metric is one of "kills"/"towers"/"dragons"/"barons"/"inhibitors"/"time"/"ml_A"/"ml_B"
      - side is "Over"/"Under" for totals, "A"/"B" for ML
      - line is a string ("27.5", "33") or "" for ML
      - odd is a positive float > 1.0

    Tokens that don't match are silently dropped — keeps the field forgiving.
    """
    out: dict[tuple[str, str, str], float] = {}
    if not text:
        return out
    for raw in _SEPARATORS.split(str(text).strip()):
        if not raw:
            continue
        # Split body / odd by separator (/, =, @).
        sep_match = re.search(r"[/=@]", raw)
        if not sep_match:
            continue
        body = raw[: sep_match.start()]
        odd_str = raw[sep_match.end():].strip().replace(",", ".")
        try:
            odd_val = float(odd_str)
        except ValueError:
            continue
        if not math.isfinite(odd_val) or odd_val <= 1.0:
            continue
        metric, line, side_hint = _match_prefix(body)
        if not metric:
            continue
        if metric in ("ml_A", "ml_B"):
            side = "A" if metric == "ml_A" else "B"
            out[("ml", side, "")] = odd_val
            continue
        if metric in _TOTALS_METRICS:
            if not line:
                continue
            side = side_hint or "Over"  # default to Over when ambiguous
            out[(metric, side, line)] = odd_val
    return out


# --- Row → key normalization (cross-match parsed paste with rendered rows) -

_SIDE_PT_TO_NORM = {
    "ACIMA": "Over",
    "OVER": "Over",
    "ABAIXO": "Under",
    "UNDER": "Under",
}


def row_signature(
    row: Mapping[str, object], *, team_a: str, team_b: str,
) -> Optional[tuple[str, str, str]]:
    """Build the (metric, side, line) tuple for one resumo row.

    Returns None when the row's market kind isn't recognized (e.g. combos).
    """
    metric = str(row.get("_metric") or "").strip()
    if not metric:
        return None
    lado = str(row.get("Lado") or "").strip()
    line_val = row.get("_line")

    # ML (winner): _metric like "00_ml_A" / "00_ml_B"
    if metric.endswith("_ml_A") or metric == "ml_A":
        return ("ml", "A", "")
    if metric.endswith("_ml_B") or metric == "ml_B":
        return ("ml", "B", "")

    # Totals: _metric is "kills"/"towers"/"dragons"/"barons"/"inhibitors"/"time"
    if metric in _TOTALS_METRICS:
        side_norm = _SIDE_PT_TO_NORM.get(lado.upper(), "")
        if not side_norm:
            return None
        try:
            ln = float(line_val) if line_val is not None else None
        except (TypeError, ValueError):
            ln = None
        if ln is None:
            return None
        # Match the paste's line-string normalization
        line_str = f"{ln:g}"
        return (metric, side_norm, line_str)

    # Combo (ML+Total): skip — too composite to express with one paste token.
    return None


def apply_paste_to_oc_map(
    parsed: dict[tuple[str, str, str], float],
    rows: Iterable[Mapping[str, object]],
    *,
    team_a: str, team_b: str,
) -> dict[str, float]:
    """Cross-match a parsed paste against rows; return {row_key: odd} updates.

    row_key follows the same convention used by the renderer:
        f"{Mercado}|{Lado}|{Linha}"
    """
    updates: dict[str, float] = {}
    if not parsed:
        return updates
    for row in rows:
        sig = row_signature(row, team_a=team_a, team_b=team_b)
        if sig is None:
            continue
        odd = parsed.get(sig)
        if odd is None:
            continue
        key = f"{row.get('Mercado','')}|{row.get('Lado','')}|{row.get('Linha','')}"
        updates[key] = float(odd)
    return updates


# --- Kelly stake (conservative fraction). ---------------------------------

def kelly_stake_fraction(p_true: Optional[float], odd_casa: Optional[float],
                          *, fraction: float = 0.25) -> Optional[float]:
    """Return a 0-1 stake fraction using a fractional Kelly criterion.

    fraction=0.25 means "quarter Kelly" — the standard conservative choice
    for sports betting where p_true is uncertain.

    Returns None if inputs are missing/invalid, 0.0 when Kelly is negative
    (don't bet), and the fractional Kelly otherwise.
    """
    if p_true is None or odd_casa is None:
        return None
    try:
        p = float(p_true); o = float(odd_casa); f = float(fraction)
    except (TypeError, ValueError):
        return None
    if not (0.0 < p < 1.0) or o <= 1.0 or f <= 0.0:
        return None
    b = o - 1.0
    q = 1.0 - p
    k_full = (p * b - q) / b
    if k_full <= 0.0:
        return 0.0
    return float(min(1.0, k_full * f))


# Allowed stake ladder (user works only with these unit sizes).
ALLOWED_UNITS: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def discrete_stake_units(kelly_fraction: Optional[float]) -> Optional[float]:
    """Map a continuous ¼ Kelly fraction (0-1, where 0.01 = 1% of bankroll)
    to one of the user's discrete unit sizes.

    Convention: 1u ≈ 1% of bankroll. ¼ Kelly typically lands in 0.5%-5%,
    matching the ladder below. Returns None when Kelly is too small to bet.

    Mapping (kelly% = kelly_fraction * 100):
      >= 4%   → 2.0u   (pick excepcional)
      >= 3%   → 1.5u
      >= 2.25%→ 1.25u
      >= 1.5% → 1.0u   (padrão)
      >= 1%   → 0.75u
      >= 0.5% → 0.5u   (mínimo)
      < 0.5%  → None   (não apostar)
    """
    if kelly_fraction is None:
        return None
    try:
        kf = float(kelly_fraction)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(kf) or kf <= 0.0:
        return None
    k_pct = kf * 100.0
    if k_pct >= 4.0:
        return 2.0
    if k_pct >= 3.0:
        return 1.5
    if k_pct >= 2.25:
        return 1.25
    if k_pct >= 1.5:
        return 1.0
    if k_pct >= 1.0:
        return 0.75
    if k_pct >= 0.5:
        return 0.5
    return None


def classify_pick(*, edge_hist_pct: Optional[float], conf_score: int) -> str:
    """Classify a pick into one of three buckets for the UI:

      - "recommend"  edge >= 5% AND conf >= 60
      - "watch"      edge >= 0% AND conf >= 40 (and not recommend)
      - "avoid"      anything else (negative edge or very low confidence)

    When edge is None (no Odd Casa filled), fall back to conf only:
      - conf >= 70 → "recommend"
      - conf >= 50 → "watch"
      - else       → "avoid"
    """
    if edge_hist_pct is None:
        if conf_score >= 70:
            return "recommend"
        if conf_score >= 50:
            return "watch"
        return "avoid"
    if edge_hist_pct >= 5.0 and conf_score >= 60:
        return "recommend"
    if edge_hist_pct >= 0.0 and conf_score >= 40:
        return "watch"
    return "avoid"


__all__ = [
    "ALLOWED_UNITS",
    "apply_paste_to_oc_map",
    "classify_pick",
    "discrete_stake_units",
    "kelly_stake_fraction",
    "parse_odds_paste",
    "row_signature",
]
