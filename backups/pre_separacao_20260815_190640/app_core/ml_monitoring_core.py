from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _norm_dec(v: Any) -> str:
    s = str(v or "").strip().lower()
    return s if s in {"promote", "shadow_only", "reject"} else "unknown"


def _safe_bool(v: Any, default: bool = False) -> bool:
    try:
        return bool(v)
    except Exception:
        return bool(default)


def _safe_dt(v: Any) -> datetime | None:
    s = str(v or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _file_meta(path: Path) -> Dict[str, Any]:
    exists = bool(path.exists())
    stat = path.stat() if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "size": (int(stat.st_size) if stat is not None else 0),
        "mtime": (datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds") if stat is not None else ""),
    }


def _latest_matching_file(root: Path, patterns: Iterable[str]) -> Path | None:
    best: Path | None = None
    best_mtime = float("-inf")
    for pat in patterns:
        for p in root.glob(str(pat)):
            if not p.is_file():
                continue
            try:
                mt = float(p.stat().st_mtime)
            except Exception:
                continue
            if mt > best_mtime:
                best = p
                best_mtime = mt
    return best


def _latest_promotion_report(root: Path) -> Path | None:
    base = root / "ops" / "baselines"
    if not base.exists():
        return None
    return _latest_matching_file(base, ["*/promotion_report.json"])


def build_monitor_snapshot(root: str | Path) -> Dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    params_path = root_path / "params_fitted.json"
    phase9_path = root_path / "phase9_players_report.json"
    anchor_path = root_path / "anchor_regression_report.promotion.json"
    prod_refresh_path = root_path / "prod_refresh_report.final.json"
    if not prod_refresh_path.exists():
        prod_refresh_path = root_path / "prod_refresh_report.json"
    baseline_path = _latest_matching_file(
        root_path,
        [
            "phase0_ml_v3_baseline_report.full*.json",
            "phase0_ml_v3_baseline_report.recent*.json",
        ],
    )
    promotion_path = _latest_promotion_report(root_path)

    params_obj = _read_json(params_path)
    phase9_obj = _read_json(phase9_path)
    anchor_obj = _read_json(anchor_path)
    prod_refresh_obj = _read_json(prod_refresh_path)
    baseline_obj = _read_json(baseline_path) if baseline_path is not None else {}
    promotion_obj = _read_json(promotion_path) if promotion_path is not None else {}

    players_ctx_gate = _norm_dec((((params_obj.get("players_context_fit") or {}).get("gate") or {}).get("decision")))
    fused_post_cal_gate = _norm_dec((((params_obj.get("fused_post_calibration") or {}).get("gate") or {}).get("decision")))
    phase9_dec = _norm_dec(phase9_obj.get("decision"))
    anchor_dec = _norm_dec(anchor_obj.get("decision"))
    baseline_dec = _norm_dec(baseline_obj.get("decision"))
    prod_refresh_dec = _norm_dec(prod_refresh_obj.get("final_decision") or prod_refresh_obj.get("decision"))
    promotion_ok = _safe_bool(promotion_obj.get("overall_ok"), False)
    promotion_skip_key_gate = _safe_bool(promotion_obj.get("skip_key_gate"), False)

    blockers: List[str] = []
    if players_ctx_gate != "promote":
        blockers.append(f"players_context_fit={players_ctx_gate}")
    if fused_post_cal_gate != "promote":
        blockers.append(f"fused_post_calibration={fused_post_cal_gate}")
    if baseline_dec != "promote":
        blockers.append(f"baseline={baseline_dec}")
    if phase9_dec not in {"promote", "unknown"}:
        blockers.append(f"phase9_players={phase9_dec}")
    if anchor_dec != "promote":
        blockers.append(f"anchor_regression={anchor_dec}")
    if not promotion_ok:
        blockers.append("promotion_report=not_ok")
    if promotion_skip_key_gate:
        blockers.append("promotion_report=skip_key_gate")

    files = {
        "params_fitted": _file_meta(params_path),
        "phase9_players_report": _file_meta(phase9_path),
        "anchor_report": _file_meta(anchor_path),
        "prod_refresh_report": _file_meta(prod_refresh_path),
        "baseline_report": (_file_meta(baseline_path) if baseline_path is not None else {"path": "", "exists": False, "size": 0, "mtime": ""}),
        "promotion_report": (_file_meta(promotion_path) if promotion_path is not None else {"path": "", "exists": False, "size": 0, "mtime": ""}),
    }

    return {
        "generated_at": utc_now_iso(),
        "root": str(root_path),
        "files": files,
        "sources": {
            "baseline_report": str(baseline_path) if baseline_path is not None else "",
            "promotion_report": str(promotion_path) if promotion_path is not None else "",
        },
        "gates": {
            "players_context_fit": players_ctx_gate,
            "fused_post_calibration": fused_post_cal_gate,
            "phase9_players": phase9_dec,
            "anchor_regression": anchor_dec,
            "baseline": baseline_dec,
            "prod_refresh": prod_refresh_dec,
            "promotion_overall_ok": bool(promotion_ok),
            "promotion_skip_key_gate": bool(promotion_skip_key_gate),
        },
        "health": {
            "overall_ok": bool(not blockers),
            "blockers": blockers,
        },
    }


def summarize_monitor_window(entries: List[Dict[str, Any]], *, days: int = 7, now_iso: str | None = None) -> Dict[str, Any]:
    now_dt = _safe_dt(now_iso) if now_iso else datetime.now(timezone.utc)
    if now_dt is None:
        now_dt = datetime.now(timezone.utc)
    cutoff = now_dt - timedelta(days=max(1, int(days)))

    rows: List[Dict[str, Any]] = []
    for row in list(entries or []):
        if not isinstance(row, dict):
            continue
        dt = _safe_dt(row.get("generated_at"))
        if dt is None or dt < cutoff:
            continue
        rows.append(dict(row))
    rows.sort(key=lambda r: str(r.get("generated_at") or ""))

    healthy = 0
    unhealthy = 0
    streak = 0
    for row in rows:
        if _safe_bool(((row.get("health") or {}).get("overall_ok")), False):
            healthy += 1
        else:
            unhealthy += 1
    for row in reversed(rows):
        if _safe_bool(((row.get("health") or {}).get("overall_ok")), False):
            streak += 1
        else:
            break

    latest = dict(rows[-1]) if rows else {}
    return {
        "window_days": int(days),
        "entries": len(rows),
        "healthy_entries": int(healthy),
        "unhealthy_entries": int(unhealthy),
        "healthy_ratio": (float(healthy) / float(len(rows)) if rows else 0.0),
        "consecutive_healthy_streak": int(streak),
        "latest_overall_ok": _safe_bool(((latest.get("health") or {}).get("overall_ok")), False),
        "latest_blockers": list(((latest.get("health") or {}).get("blockers") or [])),
    }


def update_monitor_history(
    history_payload: Dict[str, Any] | None,
    snapshot: Dict[str, Any],
    *,
    days: int = 7,
    keep_entries: int = 120,
) -> Dict[str, Any]:
    src = history_payload if isinstance(history_payload, dict) else {}
    entries = [dict(x) for x in (src.get("entries") or []) if isinstance(x, dict)]
    entries.append(dict(snapshot or {}))
    entries.sort(key=lambda r: str(r.get("generated_at") or ""))
    if len(entries) > int(max(1, keep_entries)):
        entries = entries[-int(max(1, keep_entries)) :]
    summary = summarize_monitor_window(entries, days=int(days))
    return {
        "generated_at": utc_now_iso(),
        "entries": entries,
        "summary": summary,
    }
