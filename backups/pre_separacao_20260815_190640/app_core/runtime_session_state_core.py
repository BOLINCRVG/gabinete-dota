"""Session-state helpers for maintenance/runtime reports and logs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, MutableMapping


def _stamp(value: Any = None) -> str:
    text = str(value or "").strip()
    if text:
        return text
    return datetime.now().isoformat(timespec="seconds")


def _get_report(session_state: MutableMapping[str, Any], key: str) -> dict[str, Any] | None:
    report = session_state.get(str(key))
    return dict(report) if isinstance(report, dict) else None


def _set_report(
    session_state: MutableMapping[str, Any],
    key: str,
    report: Any,
    *,
    at_key: str | None = None,
    at: Any = None,
) -> dict[str, Any] | None:
    value = dict(report) if isinstance(report, dict) else None
    session_state[str(key)] = value
    if at_key:
        session_state[str(at_key)] = _stamp(at)
    return value


def get_wf_last_report(session_state: MutableMapping[str, Any]) -> dict[str, Any] | None:
    """Retorna o último relatório do pipeline WF armazenado na session_state."""
    return _get_report(session_state, "wf_last_report")


def set_wf_last_report(session_state: MutableMapping[str, Any], report: Any, *, at: Any = None) -> dict[str, Any] | None:
    """Persiste o relatório WF na session_state e registra o timestamp."""
    return _set_report(session_state, "wf_last_report", report, at_key="wf_last_report_at", at=at)


def get_wf_last_report_at(session_state: MutableMapping[str, Any]) -> str:
    """Retorna o timestamp do último relatório WF como string."""
    return str(session_state.get("wf_last_report_at") or "")


def set_wf_last_report_at(session_state: MutableMapping[str, Any], at: Any) -> str:
    """Grava o timestamp do último relatório WF na session_state e o retorna."""
    stamp = _stamp(at)
    session_state["wf_last_report_at"] = stamp
    return stamp


def get_lv_last_report(session_state: MutableMapping[str, Any]) -> dict[str, Any] | None:
    """Retorna o último relatório de layer validation armazenado na session_state."""
    return _get_report(session_state, "lv_last_report")


def set_lv_last_report(session_state: MutableMapping[str, Any], report: Any, *, at: Any = None) -> dict[str, Any] | None:
    """Persiste o relatório de layer validation na session_state e registra o timestamp."""
    return _set_report(session_state, "lv_last_report", report, at_key="lv_last_report_at", at=at)


def get_cal_diag_last_report(session_state: MutableMapping[str, Any]) -> dict[str, Any] | None:
    """Retorna o último relatório de diagnóstico de calibração da session_state."""
    return _get_report(session_state, "cal_diag_last_report")


def set_cal_diag_last_report(session_state: MutableMapping[str, Any], report: Any, *, at: Any = None) -> dict[str, Any] | None:
    """Persiste o relatório de diagnóstico de calibração na session_state."""
    return _set_report(session_state, "cal_diag_last_report", report, at_key="cal_diag_last_report_at", at=at)


def get_ml_confidence_baseline_last_report(session_state: MutableMapping[str, Any]) -> dict[str, Any] | None:
    """Retorna o último relatório de baseline de confiança ML da session_state."""
    return _get_report(session_state, "ml_confidence_baseline_last_report")


def set_ml_confidence_baseline_last_report(
    session_state: MutableMapping[str, Any],
    report: Any,
    *,
    at: Any = None,
) -> dict[str, Any] | None:
    """Persiste o relatório de baseline de confiança ML na session_state."""
    return _set_report(
        session_state,
        "ml_confidence_baseline_last_report",
        report,
        at_key="ml_confidence_baseline_last_report_at",
        at=at,
    )


def get_team_state_artifact_last_report(session_state: MutableMapping[str, Any]) -> dict[str, Any] | None:
    """Retorna o último relatório do artefato de estado de times da session_state."""
    return _get_report(session_state, "team_state_artifact_last_report")


def set_team_state_artifact_last_report(
    session_state: MutableMapping[str, Any],
    report: Any,
    *,
    at: Any = None,
) -> dict[str, Any] | None:
    """Persiste o relatório do artefato de estado de times na session_state."""
    return _set_report(
        session_state,
        "team_state_artifact_last_report",
        report,
        at_key="team_state_artifact_last_report_at",
        at=at,
    )


def get_daily_pipeline_report(session_state: MutableMapping[str, Any]) -> dict[str, Any] | None:
    """Retorna o relatório do pipeline diário armazenado na session_state."""
    return _get_report(session_state, "_daily_pipeline_report")


def set_daily_pipeline_report(session_state: MutableMapping[str, Any], report: Any, *, at: Any = None) -> dict[str, Any] | None:
    """Persiste o relatório do pipeline diário na session_state e registra o timestamp."""
    return _set_report(session_state, "_daily_pipeline_report", report, at_key="_daily_pipeline_report_at", at=at)


def get_runtime_log(session_state: MutableMapping[str, Any], key: str) -> str:
    """Retorna o conteúdo de um log de runtime identificado pela chave fornecida."""
    return str(session_state.get(str(key)) or "")


def set_runtime_log(session_state: MutableMapping[str, Any], key: str, value: Any) -> str:
    """Grava um texto de log de runtime na session_state pela chave fornecida."""
    text = str(value or "")
    session_state[str(key)] = text
    return text


def get_ml_last_log(session_state: MutableMapping[str, Any]) -> str:
    """Retorna o último log de execução ML da session_state."""
    return get_runtime_log(session_state, "ml_last_log")


def set_ml_last_log(session_state: MutableMapping[str, Any], value: Any) -> str:
    """Grava o log de execução ML na session_state."""
    return set_runtime_log(session_state, "ml_last_log", value)


def get_quick_maint_last_log(session_state: MutableMapping[str, Any]) -> str:
    """Retorna o último log de manutenção rápida da session_state."""
    return get_runtime_log(session_state, "quick_maint_last_log")


def set_quick_maint_last_log(session_state: MutableMapping[str, Any], value: Any) -> str:
    """Grava o log de manutenção rápida na session_state."""
    return set_runtime_log(session_state, "quick_maint_last_log", value)


def get_governance_last_log(session_state: MutableMapping[str, Any]) -> str:
    """Retorna o último log de governança da session_state."""
    return get_runtime_log(session_state, "_gov_last_log")


def set_governance_last_log(session_state: MutableMapping[str, Any], value: Any) -> str:
    """Grava o log de governança na session_state."""
    return set_runtime_log(session_state, "_gov_last_log", value)
