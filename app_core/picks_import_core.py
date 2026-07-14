"""Pure parsing helpers to backfill the picks tracker from the user's xlsx.

The spreadsheets already carry the OUTCOME (W/L) and the P/L in units, so
imported picks land pre-settled (status won/lost/void, profit_units filled).
We do NOT re-resolve them against the CSV.

What this gives the app:
  - Live ROI-by-market / by-league / by-mapa from the user's real history,
    replacing the hardcoded table in picks_anti_loss_core.

What it does NOT give:
  - Calibration inline (p_hist buckets) — the sheets have no historical-model
    probability, only the bookmaker odd. That stays dependent on new picks.

Only markets the tracker supports are imported (kills/towers/dragons/barons/
inhibitors/time/ml). Combos, handicaps, player props and First-X are skipped
and counted in the report.
"""
from __future__ import annotations

import math
import unicodedata
from typing import Optional


# --- Aposta string → (metric_key, side_key) -------------------------------

def _strip_accents_lower(s: str) -> str:
    """Normalize mojibake/accents: 'Drag�o'/'Dragão' → 'dragao'."""
    s = str(s or "")
    # Drop replacement chars first (mojibake from latin-1/utf-8 mix).
    s = s.replace("�", "").replace("?", "")
    try:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
    except (TypeError, ValueError):
        pass
    return s.strip().lower()


# Metric keyword (normalized) → canonical metric_key. Order matters: check
# longer/more-specific tokens first ("inib" before nothing ambiguous).
_METRIC_TOKENS: tuple[tuple[str, str], ...] = (
    ("kill", "kills"),
    ("torre", "towers"),
    ("drag", "dragons"),     # dragao / drages / dragoes / dragoes elementais
    ("baron", "barons"),
    ("nashor", "barons"),
    ("inib", "inhibitors"),
    ("tempo", "time"),
)


def parse_aposta(aposta: str) -> Optional[tuple[str, str]]:
    """Map a sheet 'Aposta' string to (metric_key, side_key).

    Returns None for markets the tracker does not support:
      - combos: 'ML / Under Kill', 'ML - Over Tempo'
      - handicaps: 'Handicap Kill', 'Handicap Mapa'
      - player props: 'Jogador Under Kill', 'Duel players', 'Pick Jogador'
      - team-specific: 'Time Under Kill'
      - first-X / misc: 'First Blood', 'Outright', 'Proxima Kill', 'F5', etc.
    """
    norm = _strip_accents_lower(aposta)
    if not norm:
        return None

    # Plain ML (winner). Must not be a combo (no '/' or '-' separating markets).
    if norm == "ml":
        return ("ml", "A")  # side resolved later from 'Equipe' vs blue/red

    # Reject combos / handicaps / player / team-specific / misc up front.
    _reject_substrings = (
        "ml /", "ml -", "ml/", "ml-",
        "handicap", "jogador", "duel", "pick ", "dupla", "multipla",
        "time under", "time over", "outright", "primeiro", "proxima",
        "first", "f 5", "f5", "f6", "gols", "chutes", "round", "maps", "mapas",
        "30 - 39",
    )
    for bad in _reject_substrings:
        if bad in norm:
            return None

    # Side: Under / Over.
    if norm.startswith("under "):
        side = "Under"
    elif norm.startswith("over "):
        side = "Over"
    else:
        return None  # unknown shape

    for token, metric in _METRIC_TOKENS:
        if token in norm:
            return (metric, side)
    return None


# --- Line parsing ----------------------------------------------------------

def parse_line(raw: object, *, is_time: bool = False) -> Optional[float]:
    """Parse a 'Linha' cell. Handles '27,5', '-16.5', '31:30' (mm:ss → min)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("nan", "none", "-"):
        return None
    if ":" in s:  # time mm:ss
        try:
            m, sec = s.split(":", 1)
            return float(int(m) + int(sec) / 60.0)
        except (TypeError, ValueError):
            return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


# --- Outcome parsing -------------------------------------------------------

def parse_outcome(wl_raw: object) -> Optional[str]:
    """Map the 'W/L' cell to status. W→won, L→lost, co/CO→void (cashout).

    Returns None for unrecognized values (row skipped)."""
    s = str(wl_raw or "").strip().lower()
    if s == "w":
        return "won"
    if s == "l":
        return "lost"
    if s in ("co", "cashout", "void", "anulado", "v"):
        return "void"
    return None


def _to_float(x: object) -> Optional[float]:
    try:
        if isinstance(x, str):
            x = x.replace(",", ".")
        f = float(x)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def derive_stake_units(*, status: str, profit_units: Optional[float],
                        odd: Optional[float]) -> Optional[float]:
    """Recover stake (in units) from the sheet's P/L (units) + odd.

    won:  profit = stake*(odd-1)  → stake = profit/(odd-1)
    lost: profit = -stake         → stake = -profit
    void: profit = 0              → stake unknown but irrelevant (0 risk)
    """
    if status == "void":
        return 0.0
    if profit_units is None:
        return None
    if status == "lost":
        return abs(float(profit_units))
    if status == "won":
        if odd is None or odd <= 1.0:
            return None
        return float(profit_units) / (float(odd) - 1.0)
    return None


def parse_pick_row(row: dict) -> Optional[dict]:
    """Convert one spreadsheet row (dict) into tracker-Pick kwargs, or None.

    Expected keys: Data, Esporte, Liga, Blue/Casa, Red/Fora, Mapa, Aposta,
    Linha, Equipe, Stake (R$), Odd, W/L, P/L (U).

    Returns a dict ready for new_pick(...) plus pre-settled fields, or None
    when the row isn't an importable single-market LoL bet.
    """
    if str(row.get("Esporte", "")).strip().upper() != "LOL":
        return None

    parsed = parse_aposta(row.get("Aposta"))
    if parsed is None:
        return None
    metric_key, side_key = parsed

    status = parse_outcome(row.get("W/L"))
    if status is None:
        return None

    odd = _to_float(row.get("Odd"))
    profit_u = _to_float(row.get("P/L (U)"))
    stake_u = derive_stake_units(status=status, profit_units=profit_u, odd=odd)
    if stake_u is None:
        return None

    is_time = (metric_key == "time")
    line = None if metric_key == "ml" else parse_line(row.get("Linha"), is_time=is_time)
    # Totals markets need a line; ML doesn't.
    if metric_key != "ml" and line is None:
        return None

    # ML side: figure out which team the bet was on.
    team_a = str(row.get("Blue/Casa", "") or "").strip()
    team_b = str(row.get("Red/Fora", "") or "").strip()
    if metric_key == "ml":
        equipe = str(row.get("Equipe", "") or "").strip()
        eq_norm = _strip_accents_lower(equipe)
        if eq_norm and eq_norm == _strip_accents_lower(team_b):
            side_key = "B"
        else:
            side_key = "A"  # default/blue

    date_raw = row.get("Data")
    ts = ""
    try:
        import pandas as pd  # local import to keep core import-light
        d = pd.to_datetime(date_raw, errors="coerce")
        if pd.notna(d):
            ts = d.isoformat(timespec="seconds")
    except Exception:
        ts = str(date_raw or "")

    return {
        "ts": ts,
        "team_a": team_a,
        "team_b": team_b,
        "league": str(row.get("Liga", "") or "").strip(),
        "metric_key": metric_key,
        "side_key": side_key,
        "line": line,
        "odd_casa": odd if odd is not None else 0.0,
        "stake_units": float(stake_u),
        "status": status,
        "profit_units": float(profit_u) if profit_u is not None else (0.0 if status == "void" else None),
        "source": "xlsx",
        # Map label ("Mapa 1"/"Serie"/"Jogo") — keeps map-level bets on the
        # same day/matchup/line from collapsing into one id during import.
        "mapa": str(row.get("Mapa", "") or "").strip(),
    }


__all__ = [
    "derive_stake_units",
    "parse_aposta",
    "parse_line",
    "parse_outcome",
    "parse_pick_row",
]
