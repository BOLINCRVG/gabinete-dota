from __future__ import annotations

from typing import Any, List

from app_core.csv_gate_core import rank_decision


def _decision(value: Any, *, fallback: str = "shadow_only") -> str:
    text = str(value or "").strip().lower()
    if text in {"promote", "shadow_only", "reject"}:
        return text
    return fallback


def _review_count(report: dict[str, Any] | None) -> int:
    data = report if isinstance(report, dict) else {}
    raw = data.get("review_count")
    if raw is not None:
        try:
            return int(raw)
        except Exception:
            pass
    rows = list(data.get("rows") or [])
    return sum(1 for row in rows if isinstance(row, dict) and str(row.get("status") or "").strip().lower() == "review")


def evaluate_runtime_canary(
    *,
    matchup_report: dict[str, Any] | None,
    identity_report: dict[str, Any] | None,
    csv_gate_report: dict[str, Any] | None,
) -> dict[str, Any]:
    matchup = matchup_report if isinstance(matchup_report, dict) else {}
    identity = identity_report if isinstance(identity_report, dict) else {}
    csv_gate = csv_gate_report if isinstance(csv_gate_report, dict) else {}

    csv_decision = _decision(csv_gate.get("decision"), fallback="shadow_only")
    matchup_reviews = _review_count(matchup)
    ident_summary = identity.get("summary") if isinstance(identity.get("summary"), dict) else {}
    identity_errors = int(ident_summary.get("error_count") or 0)
    identity_warns = int(ident_summary.get("warn_count") or 0)

    reasons: List[str] = []
    decision = "promote"

    if not matchup:
        decision = "shadow_only"
        reasons.append("missing_matchup_report")
    if not identity:
        decision = "shadow_only"
        reasons.append("missing_identity_report")
    if not csv_gate:
        decision = "shadow_only"
        reasons.append("missing_csv_gate_report")

    if matchup_reviews > 0:
        decision = "reject"
        reasons.append("matchup_integrity_review")
    if identity_errors > 0 and rank_decision("reject") > rank_decision(decision):
        decision = "reject"
        reasons.append("identity_registry_errors")
    if rank_decision(csv_decision) > rank_decision(decision):
        decision = csv_decision
        reasons.append(f"csv_gate:{csv_decision}")
    elif csv_decision != "promote":
        reasons.append(f"csv_gate:{csv_decision}")

    if decision != "reject" and identity_warns > 0:
        decision = "shadow_only"
        reasons.append("identity_registry_warns")

    if not reasons:
        reasons.append("pass")

    return {
        "decision": decision,
        "reasons": reasons,
        "components": {
            "matchup_review_count": matchup_reviews,
            "identity_error_count": identity_errors,
            "identity_warn_count": identity_warns,
            "csv_gate_decision": csv_decision,
        },
    }
