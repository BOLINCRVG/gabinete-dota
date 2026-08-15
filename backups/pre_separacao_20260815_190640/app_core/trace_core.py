"""Trace helpers (combined ML + Resumo + Players + feature traces) for persistence/audit."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def _as_list(value: Any) -> List[dict]:
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    return []


def coerce_feature_flag(value: Any, default: bool = False) -> bool:
    """Parse feature flags from session/config values without exploding on bad types."""
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        try:
            return bool(int(value))
        except Exception:
            return bool(default)
    s = str(value).strip().lower()
    if s in {"1", "true", "on", "yes", "y", "sim"}:
        return True
    if s in {"0", "false", "off", "no", "n", "nao", "não"}:
        return False
    return bool(default)


def append_feature_trace_event(
    events: Any,
    *,
    feature: str,
    event: str,
    enabled: bool | None = None,
    payload: Any = None,
    dedupe_signature: str | None = None,
    max_items: int = 200,
    generated_at: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Append a small feature-trace event with clipping and dedupe.

    Keeps the structure pure-Python so callers can persist it wherever they want.
    """
    rows: List[Dict[str, Any]] = _as_list(events)
    item: Dict[str, Any] = {
        "generated_at": str(generated_at or datetime.now().isoformat(timespec="seconds")),
        "feature": str(feature or "").strip(),
        "event": str(event or "").strip(),
    }
    if enabled is not None:
        item["enabled"] = bool(enabled)
    if dedupe_signature is not None:
        item["dedupe_signature"] = str(dedupe_signature)
    if isinstance(payload, dict):
        item["payload"] = dict(payload)
    elif payload is not None:
        item["payload"] = {"value": payload}

    if rows:
        last = rows[-1]
        if (
            isinstance(last, dict)
            and str(last.get("feature", "")) == item["feature"]
            and str(last.get("event", "")) == item["event"]
            and str(last.get("dedupe_signature", "")) == str(item.get("dedupe_signature", ""))
        ):
            # Replace the last event to keep the latest timestamp/payload without growing endlessly
            rows[-1] = item
            return rows[-max(1, int(max_items)) :]

    rows.append(item)
    return rows[-max(1, int(max_items)) :]


def build_combined_trace(
    *,
    ml_base_trace: Any,
    resumo_markets_trace: Any,
    players_trace: Any,
    feature_trace: Any = None,
    generated_at: str | None = None,
) -> Dict[str, Any]:
    ml = ml_base_trace if isinstance(ml_base_trace, dict) else {}
    payload: Dict[str, Any] = {
        "generated_at": str(generated_at or datetime.now().isoformat(timespec="seconds")),
        "matchup": (ml.get("matchup") if isinstance(ml.get("matchup"), dict) else {}),
        "filters": (ml.get("filters") if isinstance(ml.get("filters"), dict) else {}),
        "engine": ml.get("engine"),
        "ml_trace": ml,
        "resumo_markets_trace": _as_list(resumo_markets_trace),
        "players_trace": _as_list(players_trace),
        "feature_trace": _as_list(feature_trace),
    }
    return payload
