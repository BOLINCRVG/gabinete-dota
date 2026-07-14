"""Online prediction tracking — record predictions + outcomes for live calibration audit.

Each prediction the user makes (pre-game P(team_a wins), live state→p_win,
market predictions, first-X predictions) can be recorded here. When the actual
game outcome is later observed (from the CSV), we can resolve the prediction
and compute calibration metrics over time.

Design
------
- Predictions are appended to a JSONL file (atomic-ish via os.replace).
- One record per prediction event: {gameid, market, ts_predicted, p_predicted,
  context}.
- Outcomes are appended to a separate JSONL: {gameid, market, ts_resolved,
  y_true}.
- The aggregator joins these to produce Brier, logloss, ECE by time window
  and by market.

This is intentionally minimal: no database, no daemons. Files live under
`tracking/` and roll by month for sanity.
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_LOCK = threading.Lock()
_DEFAULT_DIR = Path("tracking")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _yyyymm(ts: str) -> str:
    """Return YYYYMM from an ISO timestamp; falls back to today if invalid."""
    try:
        return ts[:7].replace("-", "")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y%m")


@dataclass
class PredictionRecord:
    gameid: str
    market: str  # "p_win" | "p_win_live" | "total_kills" | "first_blood" | etc.
    p_predicted: float
    ts_predicted: str = field(default_factory=_now_iso)
    line: Optional[float] = None  # for over/under markets
    side: str = "Blue"  # which side the prediction refers to
    minute: Optional[int] = None  # snapshot minute for live predictions
    league: str = ""
    bo: int = 1
    is_playoffs: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeRecord:
    gameid: str
    market: str
    y_true: float  # 0/1 for binary; actual total for regression
    ts_resolved: str = field(default_factory=_now_iso)
    line: Optional[float] = None
    side: str = "Blue"


def predictions_path(base_dir: str | Path = _DEFAULT_DIR, *, when: Optional[str] = None) -> Path:
    base = Path(str(base_dir))
    base.mkdir(parents=True, exist_ok=True)
    yyyymm = _yyyymm(when or _now_iso())
    return base / f"predictions_{yyyymm}.jsonl"


def outcomes_path(base_dir: str | Path = _DEFAULT_DIR, *, when: Optional[str] = None) -> Path:
    base = Path(str(base_dir))
    base.mkdir(parents=True, exist_ok=True)
    yyyymm = _yyyymm(when or _now_iso())
    return base / f"outcomes_{yyyymm}.jsonl"


def record_prediction(record: PredictionRecord, *, base_dir: str | Path = _DEFAULT_DIR) -> None:
    """Append a prediction record to the current month's JSONL."""
    line = json.dumps(asdict(record), ensure_ascii=False)
    path = predictions_path(base_dir, when=record.ts_predicted)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def record_outcome(record: OutcomeRecord, *, base_dir: str | Path = _DEFAULT_DIR) -> None:
    """Append an outcome record to the current month's JSONL."""
    line = json.dumps(asdict(record), ensure_ascii=False)
    path = outcomes_path(base_dir, when=record.ts_resolved)
    with _LOCK:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


def load_predictions(base_dir: str | Path = _DEFAULT_DIR, *,
                      months: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Load prediction records from the given months (defaults to all)."""
    base = Path(str(base_dir))
    if not base.exists():
        return []
    if months:
        files = [base / f"predictions_{m}.jsonl" for m in months]
    else:
        files = sorted(base.glob("predictions_*.jsonl"))
    out: List[Dict[str, Any]] = []
    for f in files:
        out.extend(_read_jsonl(f))
    return out


def load_outcomes(base_dir: str | Path = _DEFAULT_DIR, *,
                   months: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Load outcome records from the given months (defaults to all)."""
    base = Path(str(base_dir))
    if not base.exists():
        return []
    if months:
        files = [base / f"outcomes_{m}.jsonl" for m in months]
    else:
        files = sorted(base.glob("outcomes_*.jsonl"))
    out: List[Dict[str, Any]] = []
    for f in files:
        out.extend(_read_jsonl(f))
    return out


def join_predictions_with_outcomes(
    predictions: List[Dict[str, Any]],
    outcomes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Inner-join by (gameid, market, line) — line None matches None.

    Returns rows with {gameid, market, line, side, p_predicted, y_true,
    ts_predicted, ts_resolved}. Multiple predictions for the same key produce
    multiple joined rows (one per prediction × matching outcome).
    """
    # Index outcomes by (gameid, market, line) — line normalized to None or float
    idx: Dict[Tuple[str, str, Optional[float], str], Dict[str, Any]] = {}
    for o in outcomes:
        try:
            ln = float(o.get("line")) if o.get("line") is not None else None
        except (TypeError, ValueError):
            ln = None
        key = (str(o.get("gameid") or ""), str(o.get("market") or ""),
               ln, str(o.get("side") or "Blue"))
        idx[key] = o
    joined: List[Dict[str, Any]] = []
    for p in predictions:
        try:
            ln = float(p.get("line")) if p.get("line") is not None else None
        except (TypeError, ValueError):
            ln = None
        key = (str(p.get("gameid") or ""), str(p.get("market") or ""),
               ln, str(p.get("side") or "Blue"))
        o = idx.get(key)
        if o is None:
            continue
        joined.append({
            "gameid": p.get("gameid"),
            "market": p.get("market"),
            "line": ln,
            "side": p.get("side"),
            "p_predicted": float(p.get("p_predicted")),
            "y_true": float(o.get("y_true")),
            "minute": p.get("minute"),
            "league": p.get("league"),
            "ts_predicted": p.get("ts_predicted"),
            "ts_resolved": o.get("ts_resolved"),
        })
    return joined


def _brier(p: float, y: float) -> float:
    return float((float(p) - float(y)) ** 2)


def _logloss(p: float, y: float) -> float:
    p = max(1e-6, min(1.0 - 1e-6, float(p)))
    y = float(y)
    return float(-(y * math.log(p) + (1.0 - y) * math.log(1.0 - p)))


def compute_calibration_summary(joined: List[Dict[str, Any]],
                                  *, n_bins: int = 10) -> Dict[str, Any]:
    """Compute Brier, logloss, ECE (binary-style) and bins over joined records.

    Treats regression markets by skipping them (only binary p/y supported here).
    """
    binary = [r for r in joined if 0.0 <= float(r["y_true"]) <= 1.0
              and 0.0 <= float(r["p_predicted"]) <= 1.0]
    n = len(binary)
    if n == 0:
        return {"n": 0, "brier": None, "logloss": None, "ece": None, "bins": []}
    ps = [float(r["p_predicted"]) for r in binary]
    ys = [float(r["y_true"]) for r in binary]
    brier_v = float(sum(_brier(p, y) for p, y in zip(ps, ys)) / n)
    ll_v = float(sum(_logloss(p, y) for p, y in zip(ps, ys)) / n)

    # ECE
    bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": 0,
             "mean_pred": 0.0, "mean_actual": 0.0} for i in range(n_bins)]
    for p, y in zip(ps, ys):
        b = min(n_bins - 1, int(p * n_bins))
        bins[b]["n"] += 1
        bins[b]["mean_pred"] += p
        bins[b]["mean_actual"] += y
    ece = 0.0
    for b in bins:
        if b["n"] == 0:
            continue
        b["mean_pred"] = b["mean_pred"] / b["n"]
        b["mean_actual"] = b["mean_actual"] / b["n"]
        ece += (b["n"] / n) * abs(b["mean_pred"] - b["mean_actual"])

    return {
        "n": n,
        "brier": brier_v,
        "logloss": ll_v,
        "ece": float(ece),
        "bins": bins,
    }


def summarize_by_market(joined: List[Dict[str, Any]],
                         *, n_bins: int = 10) -> Dict[str, Dict[str, Any]]:
    """Compute calibration metrics grouped by market."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in joined:
        grouped.setdefault(str(r["market"]), []).append(r)
    return {
        m: compute_calibration_summary(rows, n_bins=n_bins)
        for m, rows in grouped.items()
    }


def ingest_outcomes_from_csv(csv_path: str | Path, *, base_dir: str | Path = _DEFAULT_DIR) -> int:
    """Read team rows from one Oracle CSV and ingest outcomes for known markets.

    Records outcomes for:
      - p_win (result column)
      - first_blood / first_dragon / first_tower / first_herald / first_baron
      - total_kills, total_towers, total_dragons, total_barons, gamelength_min

    Returns the number of outcomes written. Safe to call repeatedly — the file
    just grows (de-dup is the responsibility of the consumer if needed).
    """
    import pandas as pd
    df = pd.read_csv(str(csv_path), low_memory=False)
    team_rows = df[df["participantid"].isin([100, 200])].copy()
    n_written = 0
    grouped = team_rows.groupby("gameid", sort=False)
    for gid, g in grouped:
        gid_s = str(gid)
        blue = g[g["participantid"] == 100]
        red = g[g["participantid"] == 200]
        if blue.empty or red.empty:
            continue
        blue = blue.iloc[0]
        red = red.iloc[0]
        # Result (blue won?)
        try:
            blue_result = int(blue.get("result", -1))
        except (TypeError, ValueError):
            blue_result = -1
        if blue_result in (0, 1):
            record_outcome(OutcomeRecord(
                gameid=gid_s, market="p_win", side="Blue", y_true=float(blue_result),
            ), base_dir=base_dir)
            n_written += 1
        # First-X
        for col, market in (("firstblood", "first_blood"),
                              ("firstdragon", "first_dragon"),
                              ("firsttower", "first_tower"),
                              ("firstherald", "first_herald"),
                              ("firstbaron", "first_baron")):
            try:
                v = float(blue.get(col, float("nan")))
            except (TypeError, ValueError):
                continue
            if math.isfinite(v) and 0.0 <= v <= 1.0:
                record_outcome(OutcomeRecord(
                    gameid=gid_s, market=market, side="Blue", y_true=float(1 if v >= 0.5 else 0),
                ), base_dir=base_dir)
                n_written += 1
        # Game totals (regression markets — recorded as actuals; line=None)
        try:
            tk = float(blue.get("teamkills", float("nan"))) + float(red.get("teamkills", float("nan")))
            if math.isfinite(tk):
                record_outcome(OutcomeRecord(gameid=gid_s, market="total_kills",
                                              side="Blue", y_true=float(tk)),
                                base_dir=base_dir)
                n_written += 1
        except (TypeError, ValueError):
            pass
        try:
            gl = float(blue.get("gamelength", float("nan")))
            if math.isfinite(gl) and gl > 0:
                record_outcome(OutcomeRecord(gameid=gid_s, market="gamelength_min",
                                              side="Blue", y_true=float(gl / 60.0)),
                                base_dir=base_dir)
                n_written += 1
        except (TypeError, ValueError):
            pass
    return n_written


__all__ = [
    "OutcomeRecord",
    "PredictionRecord",
    "compute_calibration_summary",
    "ingest_outcomes_from_csv",
    "join_predictions_with_outcomes",
    "load_outcomes",
    "load_predictions",
    "outcomes_path",
    "predictions_path",
    "record_outcome",
    "record_prediction",
    "summarize_by_market",
]
