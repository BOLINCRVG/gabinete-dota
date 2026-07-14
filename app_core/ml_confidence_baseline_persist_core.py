# -*- coding: utf-8 -*-
"""Builds and persists the ML confidence baseline, with OOS gate protection."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from app_core.ml_confidence_baseline_core import (
    build_ml_confidence_baseline as _build_ml_confidence_baseline,
    write_ml_confidence_baseline as _write_ml_confidence_baseline,
)
from app_core.ml_oos_gate_core import evaluate_confidence_baseline_oos_gate as _evaluate_oos_gate
from app_core.bootstrap_maintenance_state_core import read_json_file_safe as _read_json_file_safe


def generate_and_persist_ml_confidence_baseline(
    *,
    root: Path,
    lv_report: Optional[Dict[str, Any]],
    wf_report: Optional[Dict[str, Any]],
    params_fit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rep = _build_ml_confidence_baseline(lv_report, wf_report, params_fit=params_fit)
    if isinstance(rep, dict) and not rep.get("error"):
        out_p = root / "ml_confidence_baseline.json"
        new_n = int(len((rep.get("by_league") or {})) if isinstance(rep.get("by_league"), dict) else 0)
        prev = dict(_read_json_file_safe(out_p))
        gate = _evaluate_oos_gate(candidate=rep, previous=(prev if isinstance(prev, dict) else None))
        rep["promotion_gate"] = gate
        prev_n = int(len((prev.get("by_league") or {})) if isinstance(prev.get("by_league"), dict) else 0)
        if new_n <= 0 and prev_n > 0:
            keep = dict(prev) if isinstance(prev, dict) else {}
            _warn = list(keep.get("warnings") or [])
            _warn.append("kept_previous_baseline_previous_leagues_gt0_new_leagues_eq0")
            keep["warnings"] = _warn[-20:]
            keep["reused_at"] = datetime.now().isoformat(timespec="seconds")
            keep["reused_reason"] = "new_baseline_has_zero_leagues"
            return keep
        if isinstance(prev, dict) and prev and isinstance(gate, dict) and (not bool(gate.get("pass", True))):
            keep = dict(prev)
            _warn = list(keep.get("warnings") or [])
            _warn.append("kept_previous_baseline_oos_gate_failed")
            keep["warnings"] = _warn[-20:]
            keep["reused_at"] = datetime.now().isoformat(timespec="seconds")
            keep["reused_reason"] = "oos_gate_failed"
            keep["reused_candidate_gate"] = gate
            return keep
        _write_ml_confidence_baseline(out_p, rep)
    return rep
