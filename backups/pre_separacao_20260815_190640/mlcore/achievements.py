from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from math import exp, log
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------
# Achievements (manual inputs) + learned weights (points) layer
#
# - You store results in achievements.json (editable in mlcore_app.py).
# - train_offline.py can learn weights_points for each (event, placement)
#   by minimizing logloss with L2 regularization (ridge).
# - Predictions then add:
#     total_strength = base_strength + achievement_boost_points(as_of_date)
#   where boost decays with time (half-life).
# ---------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    # half-life controls how quickly an achievement stops mattering
    # You can tweak these later without retraining Elo residuals (only retrain weights).
    "half_life_days": {
        "worlds": 365,
        "msi": 240,
        "regional": 180,
        "other": 120,
    },
    # Supported placements (UI)
    "placements": ["champion", "runner_up", "top4", "top8", "top16", "playoffs"],
    # Supported event types (UI)
    "events": ["worlds", "msi", "regional", "other"],
    # Optional extra decay aggressiveness:
    # effective_decay = base_decay ** decay_power (power > 1 => faster decay)
    "decay_power": {
        "default": 1.0,
        "worlds": 1.0,
        "msi": 1.0,
        "regional": 1.0,
        "other": 1.0,
    },
}

# Stable feature keys = (event, placement)
def feature_key(event: str, placement: str) -> str:
    e = (event or "other").strip().lower()
    p = (placement or "").strip().lower()
    if not p:
        p = "playoffs"
    return f"{e}:{p}"

def _parse_iso_date(s: Any) -> Optional[date]:
    if s is None:
        return None
    if isinstance(s, date) and not isinstance(s, datetime):
        return s
    if isinstance(s, datetime):
        return s.date()
    try:
        return datetime.fromisoformat(str(s).strip()).date()
    except Exception:
        return None

def _days_between(d1: date, d2: date) -> int:
    return int((d2 - d1).days)

def decay_factor(end_date: date, as_of: date, half_life_days: float) -> float:
    """Exponential decay with half-life. Returns 0..1."""
    if half_life_days <= 0:
        return 0.0
    if as_of < end_date:
        return 0.0  # prevent leakage
    dt_days = _days_between(end_date, as_of)
    # exp(-ln2 * t / half_life)
    return float(exp(-log(2.0) * (dt_days / float(half_life_days))))

@dataclass
class AchievementEntry:
    team: str
    tournament: str
    event: str
    placement: str
    end_date: date
    league: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Optional["AchievementEntry"]:
        team = str(d.get("team") or "").strip()
        if not team:
            return None
        end_d = _parse_iso_date(d.get("end_date") or d.get("date") or "")
        if end_d is None:
            return None
        return cls(
            team=team,
            tournament=str(d.get("tournament") or "").strip(),
            event=str(d.get("event") or "other").strip().lower(),
            placement=str(d.get("placement") or "playoffs").strip().lower(),
            end_date=end_d,
            league=str(d.get("league") or "").strip(),
            notes=str(d.get("notes") or "").strip(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "tournament": self.tournament,
            "event": self.event,
            "placement": self.placement,
            "end_date": self.end_date.isoformat(),
            "league": self.league,
            "notes": self.notes,
        }

def load_achievements(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"config": dict(DEFAULT_CONFIG), "entries": []}
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(dict(d.get("config") or {}))
    entries_raw = list(d.get("entries") or [])
    entries: List[AchievementEntry] = []
    for r in entries_raw:
        if isinstance(r, dict):
            e = AchievementEntry.from_dict(r)
            if e is not None:
                entries.append(e)
    return {"config": cfg, "entries": entries}

def save_achievements(path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(dict(data.get("config") or {}))

    entries_out: List[Dict[str, Any]] = []
    for e in data.get("entries") or []:
        if isinstance(e, AchievementEntry):
            entries_out.append(e.to_dict())
        elif isinstance(e, dict):
            ee = AchievementEntry.from_dict(e)
            if ee is not None:
                entries_out.append(ee.to_dict())
    out = {"config": cfg, "entries": entries_out}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def _half_life_for_event(event: str, cfg: Dict[str, Any]) -> float:
    hl = (cfg.get("half_life_days") or {})
    raw = hl.get(event, hl.get("other", 120))
    try:
        val = float(raw)
    except Exception:
        val = float(hl.get("other", 120) or 120.0)
    if val <= 0.0:
        try:
            fallback = float(hl.get("other", 120) or 120.0)
        except Exception:
            fallback = 120.0
        val = fallback if fallback > 0.0 else 120.0
    return float(val)


def _decay_power_for_event(event: str, cfg: Dict[str, Any]) -> float:
    dp = (cfg.get("decay_power") or {})
    base = float(dp.get("default", 1.0) or 1.0)
    val = float(dp.get(event, base) or base)
    return max(0.25, min(4.0, val))

def build_team_index(data: Dict[str, Any]) -> Dict[str, List[Tuple[str, date, float]]]:
    """
    Build per-team list: (feature_key, end_date, half_life_days).
    """
    cfg = dict(data.get("config") or {})
    idx: Dict[str, List[Tuple[str, date, float]]] = {}
    for e in data.get("entries") or []:
        if isinstance(e, dict):
            e = AchievementEntry.from_dict(e)
        if not isinstance(e, AchievementEntry):
            continue
        fk = feature_key(e.event, e.placement)
        hl = _half_life_for_event(e.event, cfg)
        # decay_power > 1 means faster decay (equivalent to shorter half-life)
        pw = _decay_power_for_event(e.event, cfg)
        if pw > 0:
            hl = float(max(1.0, float(hl) / float(pw)))
        idx.setdefault(e.team, []).append((fk, e.end_date, hl))
    return idx

def team_feature_vector(team: str, as_of: date, team_idx: Dict[str, List[Tuple[str, date, float]]]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for fk, end_d, hl in team_idx.get(team, []):
        if as_of < end_d:
            continue
        w = decay_factor(end_d, as_of, hl)
        if w <= 0:
            continue
        feats[fk] = feats.get(fk, 0.0) + w
    return feats

def diff_vector(blue: str, red: str, as_of: date, team_idx: Dict[str, List[Tuple[str, date, float]]], feature_keys: List[str]) -> np.ndarray:
    """
    Returns an array with feature diffs aligned with feature_keys.
    """
    b = team_feature_vector(blue, as_of, team_idx)
    r = team_feature_vector(red, as_of, team_idx)
    x = np.zeros(len(feature_keys), dtype=np.float32)
    for i, fk in enumerate(feature_keys):
        x[i] = float(b.get(fk, 0.0) - r.get(fk, 0.0))
    return x

def boost_points(feat_vec: Dict[str, float], weights_points: Dict[str, float]) -> float:
    s = 0.0
    for fk, v in feat_vec.items():
        w = float(weights_points.get(fk, 0.0))
        s += w * float(v)
    return float(s)

def team_boost_points(team: str, as_of: date, team_idx: Dict[str, List[Tuple[str, date, float]]], weights_points: Dict[str, float]) -> float:
    feats = team_feature_vector(team, as_of, team_idx)
    return boost_points(feats, weights_points)

# ---------------------------------------------------------------------
# Learning weights (points) via ridge logistic regression on top of Elo
# logit(p) = k * (base_diff_points + X @ w_points)
# where k = ln(10) / scale_points to match Elo base-10.
# ---------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))

def learn_weights_points(
    *,
    train_rows: List[Dict[str, Any]],
    model: Any,
    achievements_data: Dict[str, Any],
    init_weights: Optional[Dict[str, float]] = None,
    scale_points: float = 400.0,
    l2: float = 0.001,
    iters: int = 80,
    lr: float = 0.5,
    max_abs: float = 250.0,
    verbose: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    train_rows: list of dicts with keys: blue, red, blue_win, date (date/datetime/iso).
    model: must provide meta(team) and total_strength(team, league=..., tier=...)
    """
    team_idx = build_team_index(achievements_data)

    # feature universe
    feature_set = set()
    for lst in team_idx.values():
        for fk, _, _ in lst:
            feature_set.add(fk)
    feature_keys = sorted(feature_set)
    if not feature_keys:
        return {}, {"features": [], "note": "no achievement entries found"}

    # Build arrays (vectorized training)
    n = len(train_rows)
    y = np.zeros(n, dtype=np.float32)
    base_diff = np.zeros(n, dtype=np.float32)
    X = np.zeros((n, len(feature_keys)), dtype=np.float32)

    for i, r in enumerate(train_rows):
        b = str(r["blue"])
        rr = str(r["red"])
        y[i] = float(r["blue_win"])
        d = _parse_iso_date(r.get("date"))
        if d is None:
            # fallback: ignore achievements if date missing
            d = date.today()

        mb = model.meta(b)
        mr = model.meta(rr)
        sb = float(model.total_strength(b, league=mb.get("league"), tier=mb.get("tier")))
        sr = float(model.total_strength(rr, league=mr.get("league"), tier=mr.get("tier")))
        base_diff[i] = sb - sr
        X[i, :] = diff_vector(b, rr, d, team_idx, feature_keys)

    # Elo logistic coefficient (base-10)
    k = float(np.log(10.0) / float(scale_points))

    w = np.zeros(len(feature_keys), dtype=np.float32)
    if init_weights:
        for i, fk in enumerate(feature_keys):
            if fk in init_weights:
                w[i] = float(init_weights.get(fk, 0.0))
        w = np.clip(w, -float(max_abs), float(max_abs))
    best = {"loss": float("inf"), "w": None}

    for t in range(int(iters)):
        z = k * (base_diff + X @ w)
        p = _sigmoid(z)
        # logloss
        eps = 1e-12
        loss = float(-np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)) + l2 * float(np.sum(w * w)))
        if loss < best["loss"]:
            best = {"loss": loss, "w": w.copy()}
        # gradient
        g = (k * (X.T @ (p - y)) / float(n)) + (2.0 * l2 * w)
        # simple lr schedule (mild)
        step = float(lr / (1.0 + 0.05 * t))
        w = w - step * g
        w = np.clip(w, -float(max_abs), float(max_abs))
        if verbose and (t % 10 == 0 or t == iters - 1):
            print(f"[ach_learn] iter={t:03d} loss={loss:.6f} step={step:.4f}")

    w_best = best["w"] if best["w"] is not None else w
    weights = {fk: float(w_best[i]) for i, fk in enumerate(feature_keys)}
    report = {
        "features": feature_keys,
        "iters": int(iters),
        "l2": float(l2),
        "lr": float(lr),
        "max_abs": float(max_abs),
        "train_n": int(n),
        "best_loss": float(best["loss"]),
        "note": "weights learned on top of fixed Elo residuals + fixed tier/league baselines",
    }
    return weights, report
