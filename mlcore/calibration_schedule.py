"""Patch/league calibration scheduling helpers (pure functions)."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Any, Dict, List, Optional

import pandas as pd


def _to_dt(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    try:
        s = str(v).strip()
        if not s:
            return None
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        d = datetime.fromisoformat(s)
        return d
    except Exception:
        return None


def _patch_key(patch: Any) -> List[int]:
    s = str(patch or "").strip()
    if not s:
        return []
    nums = [int(x) for x in re.findall(r"\d+", s)]
    return nums


def _patch_gt(a: Any, b: Any) -> bool:
    ka = _patch_key(a)
    kb = _patch_key(b)
    if not ka and not kb:
        return False
    if not kb:
        return True
    if not ka:
        return False
    n = max(len(ka), len(kb))
    ka = ka + [0] * (n - len(ka))
    kb = kb + [0] * (n - len(kb))
    return tuple(ka) > tuple(kb)


def build_patch_snapshot(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Build latest patch snapshot per league from a game-level dataframe.

    Required columns: league, patch, date. Optional: gameid.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {}
    need = {"league", "patch", "date"}
    if not need.issubset(set(df.columns)):
        return {}

    d = df.copy()
    d["league"] = d["league"].astype(str).str.strip()
    d["patch"] = d["patch"].astype(str).str.strip()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])
    d = d[(d["league"] != "") & (d["patch"] != "")]
    if d.empty:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for lg, g in d.groupby("league", sort=False):
        gg = g.sort_values("date")
        last = gg.iloc[-1]
        patch = str(last["patch"])
        # sample size on this patch
        gp = gg[gg["patch"].astype(str).str.strip() == patch]
        if "gameid" in gp.columns:
            n = int(gp["gameid"].astype(str).nunique())
        else:
            n = len(gp)
        out[str(lg)] = {
            "latest_patch": patch,
            "latest_date": str(pd.to_datetime(last["date"]).date().isoformat()),
            "games_on_latest_patch": int(n),
        }
    return out


def recommend_recalibration(
    *,
    current_snapshot: Dict[str, Dict[str, Any]],
    fitted_snapshot: Dict[str, Dict[str, Any]],
    fitted_generated_at: Any,
    max_age_days: int = 7,
    min_games_new_patch: int = 20,
) -> Dict[str, Any]:
    """Compare current and fitted patch snapshots; return status (refit/warn/ok) and reasons."""
    now = datetime.now(timezone.utc)
    dt_fit = _to_dt(fitted_generated_at)
    age_days = None
    if dt_fit is not None:
        try:
            if dt_fit.tzinfo is None:
                dt_fit = dt_fit.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - dt_fit).total_seconds() / 86400.0)
        except Exception:
            age_days = None

    reasons: List[str] = []
    changed: List[Dict[str, Any]] = []

    cur = current_snapshot if isinstance(current_snapshot, dict) else {}
    fit = fitted_snapshot if isinstance(fitted_snapshot, dict) else {}
    for lg, row in cur.items():
        if not isinstance(row, dict):
            continue
        p_cur = str(row.get("latest_patch") or "").strip()
        n_cur = int(row.get("games_on_latest_patch") or 0)
        frow = fit.get(str(lg)) if isinstance(fit.get(str(lg)), dict) else {}
        p_fit = str((frow or {}).get("latest_patch") or "").strip()
        if p_cur and p_fit and _patch_gt(p_cur, p_fit) and n_cur >= int(min_games_new_patch):
            changed.append(
                {
                    "league": str(lg),
                    "patch_current": p_cur,
                    "patch_fitted": p_fit,
                    "games_on_latest_patch": int(n_cur),
                }
            )

    if age_days is None:
        reasons.append("timestamp_fit_invalido")
    elif age_days > float(max_age_days):
        reasons.append(f"fit_antigo_{age_days:.1f}d")
    if changed:
        reasons.append("nova_patch_em_liga")

    if "nova_patch_em_liga" in reasons:
        status = "refit"
    elif any(r.startswith("fit_antigo_") for r in reasons):
        status = "warn"
    elif reasons:
        status = "warn"
    else:
        status = "ok"

    return {
        "status": str(status),
        "reasons": reasons,
        "age_days": (float(age_days) if age_days is not None and math.isfinite(float(age_days)) else None),
        "changed_leagues": changed,
    }
