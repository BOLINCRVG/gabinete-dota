# -*- coding: utf-8 -*-
"""Patch/league drift detection logic for the ML guard."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pandas as pd

from app_core.calibration_schedule import (
    build_patch_snapshot as _build_patch_snapshot,
    recommend_recalibration as _recommend_recalibration,
)


def patch_meta_guard_payload(
    csv_sig: Tuple[str, int, int],
    fit_sig: Tuple[str, int, int],
    *,
    ctx: Mapping[str, Any],
) -> Dict[str, Any]:
    """Computes patch/league drift status for use in the ML guardrail."""
    _load_fitted_params_cached = ctx.get("_load_fitted_params_cached")

    out: Dict[str, Any] = {"status": "warn", "changed_leagues": [], "age_days": None}
    try:
        csv_path, _csz, _cmt = csv_sig
        fit_path, _fsz, _fmt = fit_sig
        cur_snap: Dict[str, Any] = {}
        if str(csv_path or "").strip() and Path(str(csv_path)).exists():
            dfp = pd.read_csv(
                str(csv_path),
                usecols=["gameid", "league", "patch", "date", "participantid"],
                low_memory=False,
            )
            dfp = dfp[pd.to_numeric(dfp.get("participantid"), errors="coerce").isin([100, 200])].copy()
            dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce")
            dfp = dfp.dropna(subset=["date"]).sort_values(["date", "gameid"]).drop_duplicates(subset=["gameid"], keep="last")
            cur_snap = _build_patch_snapshot(dfp)
        fit_data: Dict[str, Any] = {}
        if str(fit_path or "").strip() and Path(str(fit_path)).exists():
            fit_data = _load_fitted_params_cached(fit_sig) or {}
        fit_snap = fit_data.get("patch_snapshot") if isinstance(fit_data, dict) else {}
        fit_ts = fit_data.get("generated_at") if isinstance(fit_data, dict) else None
        rec = _recommend_recalibration(
            current_snapshot=cur_snap if isinstance(cur_snap, dict) else {},
            fitted_snapshot=fit_snap if isinstance(fit_snap, dict) else {},
            fitted_generated_at=fit_ts,
            max_age_days=7,
            min_games_new_patch=20,
        )
        if isinstance(rec, dict):
            out["status"] = str(rec.get("status", "warn"))
            out["changed_leagues"] = list(rec.get("changed_leagues") or [])
            out["age_days"] = rec.get("age_days")
        return out
    except (OSError, KeyError, TypeError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def patch_changed_leagues_set(changed_rows: Any) -> set:
    out: set = set()
    if not isinstance(changed_rows, list):
        return out
    for row in changed_rows:
        if not isinstance(row, dict):
            continue
        lg = str(row.get("league") or row.get("League") or "").strip().upper()
        if lg:
            out.add(lg)
    return out
