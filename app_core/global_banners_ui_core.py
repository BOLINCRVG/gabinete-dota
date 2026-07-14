# -*- coding: utf-8 -*-
"""Global UI banners: maintenance drift alerts and shadow feature trace warnings."""
from __future__ import annotations

from typing import Any, Callable, Dict

from app_core.navigation_state_core import set_nav_pending as _set_nav_pending
from app_core.ml_session_runtime_core import (
    clear_feature_trace_error as _clear_feature_trace_error,
    get_feature_trace_state as _get_feature_trace_state,
)
from app_core.csv_bootstrap_runtime_core import (
    request_auto_run_maintenance as _request_auto_run_maintenance,
)


def _rerun(st: Any) -> None:
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return
    experimental_rerun_fn = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun_fn):
        experimental_rerun_fn()


def render_global_feature_trace_banner(st: Any) -> None:
    """Banner global para falhas de shadow feature/trace (separado de drift/calibração)."""
    _feature_state = _get_feature_trace_state(st.session_state)
    err = str(_feature_state.get("last_error", "") or "").strip()
    if not err:
        return
    err_at = str(_feature_state.get("last_error_at", "") or "").strip()
    body = "Atenção de observabilidade (shadow feature):\n"
    body += f"- Falha no feature trace: {err}"
    if err_at:
        body += f"\n- última ocorrência: {err_at}"
    st.warning(body)
    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        if st.button("Limpar alerta de shadow trace", key="clear_feature_trace_error_banner", width='stretch'):
            _clear_feature_trace_error(st.session_state)
            _rerun(st)
    with c2:
        if st.button("Abrir Parâmetros (Shadow Trace)", key="open_params_shadow_trace_banner", width='stretch'):
            _set_nav_pending(st.session_state, "Parâmetros")
            _rerun(st)


def render_global_maintenance_banner(
    st: Any,
    csv_path_s: str,
    artifact_path_s: str,
    *,
    collect_status_fn: Callable[[str, str], Dict[str, Any]],
) -> None:
    """Banner visível no topo do Plays para drift/calibração/parâmetros."""
    status = collect_status_fn(csv_path_s, artifact_path_s)
    st.session_state["_global_maintenance_status"] = status
    if not bool(status.get("has_issues", False)):
        return

    issues = [str(x) for x in (status.get("alerts") or []) if str(x).strip()]
    issues_preview = issues[:6]
    body = "Aconteceu drift/pendência de calibração:\n- " + "\n- ".join(issues_preview)

    if str(status.get("severity")) == "critical":
        st.error(body)
    else:
        st.warning(body)

    b1, b2 = st.columns([1.3, 1.0])
    with b1:
        if st.button("Aconteceu drift, clique aqui para recalibrar", key="global_recalibrate_cta", type="primary", width='stretch'):
            _set_nav_pending(st.session_state, "Tecnico")
            _request_auto_run_maintenance(st.session_state, status)
            _rerun(st)
    with b2:
        if st.button("Abrir Técnico", key="global_open_tech", width='stretch'):
            _set_nav_pending(st.session_state, "Tecnico")
            _rerun(st)
