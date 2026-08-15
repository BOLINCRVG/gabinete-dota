from __future__ import annotations

from pathlib import Path
from typing import Any


def decision_banner_parts(value: Any, *, prefix: str) -> dict[str, str]:
    decision = str(value or "unknown").strip().lower()
    label = {"promote": "OK", "shadow_only": "ATENCAO", "reject": "FALHA"}.get(decision, decision.upper() or "UNKNOWN")
    msg_type = "success" if decision == "promote" else ("warning" if decision == "shadow_only" else "error")
    return {
        "decision": decision,
        "label": label,
        "message": f"{prefix}: {label}",
        "message_type": msg_type,
    }


def build_anchor_cases_rows(anchor_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    report = dict(anchor_report or {})
    rows: list[dict[str, Any]] = []
    for item in (report.get("cases") or []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Caso": str(item.get("name", "")),
                "OK": bool(item.get("ok", False)),
                "Mercado": str(item.get("market", "")),
                "Lado": str(item.get("side", "")),
                "Odd": float(item.get("odd_value")) if item.get("odd_value") is not None else None,
                "Min": float(item.get("min_odd")) if item.get("min_odd") is not None else None,
                "Max": float(item.get("max_odd")) if item.get("max_odd") is not None else None,
                "Erro": str(item.get("error", "")),
            }
        )
    return rows


def build_leakage_file_rows(leak_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    report = dict(leak_report or {})
    rows: list[dict[str, Any]] = []
    for item in (report.get("files") or []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Arquivo": Path(str(item.get("path", ""))).name,
                "OK": bool(item.get("ok", False)),
                "Rows": int(item.get("rows") or 0),
                "Errors": " | ".join([str(v) for v in (item.get("errors") or [])[:4]]),
                "Warnings": " | ".join([str(v) for v in (item.get("warnings") or [])[:4]]),
            }
        )
    return rows


def build_parts_rows(parts: dict[str, Any] | None) -> list[dict[str, Any]]:
    payload = dict(parts or {})
    return [{"Check": str(k), "Decision": str(v)} for k, v in payload.items()]


def build_phase12_by_league_rows(phase12_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    report = dict(phase12_report or {})
    rows: list[dict[str, Any]] = []
    by_league = report.get("by_league") or {}
    if not isinstance(by_league, dict):
        return rows
    for league, item in by_league.items():
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Liga": str(league),
                "Rows": int(item.get("rows") or 0),
                "Gain logloss": item.get("gain_logloss"),
                "LL off": ((item.get("off") or {}).get("logloss") if isinstance(item.get("off"), dict) else None),
                "LL on": ((item.get("on") or {}).get("logloss") if isinstance(item.get("on"), dict) else None),
            }
        )
    return rows


def build_stop_rules_banner(backtest_report: dict[str, Any] | None) -> dict[str, Any]:
    report = dict(backtest_report or {})
    promotion = (report.get("promotion_decision") or {}) if isinstance(report.get("promotion_decision"), dict) else {}
    decision = str(promotion.get("decision") or "unknown").strip().lower()
    label = {"promote": "PROMOVER", "shadow_only": "SHADOW ONLY", "reject": "REJEITAR"}.get(decision, decision.upper() or "UNKNOWN")
    message_type = "success" if decision == "promote" else ("warning" if decision == "shadow_only" else "error")
    return {
        "decision": decision,
        "label": label,
        "message_type": message_type,
        "message": f"Decisão stop-rules: {label}",
        "reasons": list(promotion.get("reasons") or []),
    }


def build_stop_rules_major_rows(backtest_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    report = dict(backtest_report or {})
    promotion = (report.get("promotion_decision") or {}) if isinstance(report.get("promotion_decision"), dict) else {}
    checks = (promotion.get("checks") or {}) if isinstance(promotion.get("checks"), dict) else {}
    rows: list[dict[str, Any]] = []
    for item in (checks.get("major_leagues") or []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Liga": str(item.get("league", "")),
                "Rows": int(item.get("rows") or 0),
                "Ganho logloss": float(item.get("gain_logloss_vs_base")) if item.get("gain_logloss_vs_base") is not None else None,
                "Fail drawdown": bool(item.get("fail_drawdown", False)),
            }
        )
    return rows


def build_stop_rules_global_band_rows(backtest_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    report = dict(backtest_report or {})
    monitoring = (report.get("monitoring") or {}) if isinstance(report.get("monitoring"), dict) else {}
    mon_global = (monitoring.get("global") or {}) if isinstance(monitoring.get("global"), dict) else {}
    bands_final = mon_global.get("favorite_odd_bands_final") or []
    bands_base = mon_global.get("favorite_odd_bands_base") or []
    base_by_band = {str(x.get("band", "")): x for x in bands_base if isinstance(x, dict)} if isinstance(bands_base, list) else {}
    rows: list[dict[str, Any]] = []
    for item in bands_final:
        if not isinstance(item, dict):
            continue
        band = str(item.get("band", ""))
        band_base = base_by_band.get(band, {})
        rows.append(
            {
                "Faixa odd fav": band,
                "N": int(item.get("n") or 0),
                "Gap calib final (pp)": float(item.get("calibration_gap_pp")) if item.get("calibration_gap_pp") is not None else None,
                "Gap calib base (pp)": float(band_base.get("calibration_gap_pp")) if band_base.get("calibration_gap_pp") is not None else None,
                "Winrate real %": (float(item.get("winrate_actual")) * 100.0) if item.get("winrate_actual") is not None else None,
                "Prob média %": (float(item.get("p_mean")) * 100.0) if item.get("p_mean") is not None else None,
            }
        )
    return rows


def build_stop_rules_league_band_rows(backtest_report: dict[str, Any] | None, league: str) -> list[dict[str, Any]]:
    report = dict(backtest_report or {})
    monitoring = (report.get("monitoring") or {}) if isinstance(report.get("monitoring"), dict) else {}
    by_league = (monitoring.get("by_league") or {}) if isinstance(monitoring.get("by_league"), dict) else {}
    league_payload = (by_league.get(str(league)) or {}) if isinstance(by_league.get(str(league)), dict) else {}
    rows: list[dict[str, Any]] = []
    for item in (league_payload.get("favorite_odd_bands_final") or []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Faixa odd fav": str(item.get("band", "")),
                "N": int(item.get("n") or 0),
                "Gap calib final (pp)": float(item.get("calibration_gap_pp")) if item.get("calibration_gap_pp") is not None else None,
                "Winrate real %": (float(item.get("winrate_actual")) * 100.0) if item.get("winrate_actual") is not None else None,
                "Prob média %": (float(item.get("p_mean")) * 100.0) if item.get("p_mean") is not None else None,
            }
        )
    return rows
