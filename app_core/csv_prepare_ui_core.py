# -*- coding: utf-8 -*-
"""UI render helpers for the CSV prepare / bootstrap live-dashboard."""
from __future__ import annotations

import html
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from app_core.ui_labels import fmt_bytes_short as _fmt_bytes_short


def _csv_prepare_status_chip(status: str) -> str:
    m = {
        "ok": "OK",
        "error": "ERRO",
        "running": "RODANDO",
        "pending": "PENDENTE",
    }
    return m.get(str(status or "").lower(), str(status or "").upper())


def _csv_prepare_status_icon(status: str) -> str:
    m = {
        "ok": "[OK]",
        "error": "[X]",
        "running": "[...]",
        "pending": "[ ]",
    }
    return m.get(str(status or "").lower(), "[ ]")


def render_csv_prepare_live_dashboard(
    st: Any,
    target: Any,
    report: Dict[str, Any],
    *,
    steps_plan: List[Dict[str, Any]],
    current_idx: Optional[int] = None,
    current_step: str = "",
) -> None:
    if target is None:
        return
    steps_done = report.get("steps") or []
    total_steps = max(1, len(steps_plan or []))
    ok_count = int(sum(1 for s in steps_done if bool(s.get("ok", False))))
    fail_count = int(sum(1 for s in steps_done if not bool(s.get("ok", False))))
    elapsed_total = float(report.get("elapsed_s", 0.0) or 0.0) if report.get("elapsed_s") is not None else 0.0
    if not report.get("elapsed_s"):
        try:
            _t0 = float(report.get("_t_all0", 0.0) or 0.0)
            if _t0 > 0:
                elapsed_total = max(0.0, time.perf_counter() - _t0)
        except (TypeError, ValueError):
            elapsed_total = 0.0

    with target.container():
        m1, m2, m3, m4, m5 = st.columns([1.4, 0.7, 0.7, 0.9, 1.1])
        m1.metric("Arquivo", Path(str(report.get("csv_path") or "")).name or "-")
        m2.metric("Etapas", f"{len(steps_done)}/{total_steps}")
        m3.metric("OK", ok_count)
        m4.metric("Falhas", fail_count)
        m5.metric("Tempo", f"{elapsed_total:.1f}s")

        if str(current_step or "").strip():
            st.info(f"Executando: {current_idx or '?'} / {total_steps} - {current_step}")
        elif bool(report.get("ok", False)):
            st.success("Preparo do CSV concluído com sucesso.")
        elif report.get("errors"):
            st.error("Preparo do CSV finalizado com falhas.")

        cols = st.columns(2, gap="small")
        for i, p in enumerate(steps_plan or [], start=1):
            col = cols[(i - 1) % 2]
            actual = (steps_done[i - 1] if (i - 1) < len(steps_done) else None)
            if isinstance(actual, dict):
                st_key = "ok" if bool(actual.get("ok", False)) else "error"
                detail = str(actual.get("summary", "") or "")
                t_txt = f"{float(actual.get('elapsed_s', 0.0) or 0.0):.2f}s"
            else:
                if current_idx == i and str(current_step or "") == str(p.get("name") or ""):
                    st_key = "running"
                    detail = "Processando..."
                else:
                    st_key = "pending"
                    detail = "Aguardando"
                t_txt = "-"
            crit_txt = "Crítica" if bool(p.get("critical", False)) else "Opcional"
            icon = _csv_prepare_status_icon(st_key)
            chip = _csv_prepare_status_chip(st_key)
            bg = {
                "ok": "rgba(22,163,74,.12)",
                "error": "rgba(239,68,68,.12)",
                "running": "rgba(59,130,246,.12)",
                "pending": "rgba(148,163,184,.08)",
            }.get(st_key, "rgba(148,163,184,.08)")
            border = {
                "ok": "rgba(22,163,74,.35)",
                "error": "rgba(239,68,68,.35)",
                "running": "rgba(59,130,246,.35)",
                "pending": "rgba(148,163,184,.25)",
            }.get(st_key, "rgba(148,163,184,.25)")
            with col:
                st.markdown(
                    f"""
<div style="border:1px solid {border}; background:{bg}; border-radius:10px; padding:10px 12px; margin:4px 0;">
  <div style="display:flex; justify-content:space-between; gap:8px; align-items:center;">
    <div style="font-weight:600;">{icon} {i}. {p.get("name") or ""!s}</div>
    <div style="opacity:.85; font-size:12px;">{chip}</div>
  </div>
  <div style="opacity:.8; font-size:12px; margin-top:2px;">{crit_txt} | {t_txt}</div>
  <div style="font-size:12px; margin-top:6px; opacity:.95;">{html.escape(detail) if detail else "&nbsp;"}</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )


def render_csv_prepare_artifacts_cards(st: Any, rows: List[Dict[str, Any]]) -> None:
    if not isinstance(rows, list) or not rows:
        return
    st.write("Artifacts relacionados")
    cols = st.columns(2, gap="small")
    for i, r in enumerate(rows, start=1):
        col = cols[(i - 1) % 2]
        ok = bool(r.get("Existe", False))
        border = "rgba(22,163,74,.35)" if ok else "rgba(239,68,68,.35)"
        bg = "rgba(22,163,74,.08)" if ok else "rgba(239,68,68,.08)"
        icon = "[OK]" if ok else "[X]"
        nm = str(r.get("Arquivo") or "")
        path = str(r.get("Caminho") or "")
        with col:
            st.markdown(
                f"""
<div style="border:1px solid {border}; background:{bg}; border-radius:10px; padding:10px 12px; margin:4px 0;">
  <div style="font-weight:600;">{icon} {nm}</div>
  <div style="font-size:12px; opacity:.8; margin-top:4px; word-break:break-all;">{html.escape(path)}</div>
</div>
                """,
                unsafe_allow_html=True,
            )


def render_csv_prepare_report_summary(st: Any, report: Optional[Dict[str, Any]]) -> None:
    if not isinstance(report, dict) or not report:
        return
    csv_path = str(report.get("csv_path") or "")
    sig = report.get("csv_sig") if isinstance(report.get("csv_sig"), (list, tuple)) else None
    ts = str(report.get("finished_at") or report.get("started_at") or "")
    status_txt = "pronto" if bool(report.get("ok", False)) else "com falhas"
    dur = report.get("elapsed_s")
    dur_txt = f"{float(dur):.1f}s" if isinstance(dur, (int, float)) and math.isfinite(float(dur)) else "-"
    file_name = Path(csv_path).name if csv_path else "(sem csv)"
    size_txt = _fmt_bytes_short(sig[1]) if sig and len(sig) >= 2 else "-"
    step_count = len(report.get("steps") or [])
    fail_count = int(sum(1 for s in (report.get("steps") or []) if not bool(s.get("ok", False))))
    c_a, c_b, c_c, c_d = st.columns([1.7, 0.8, 0.8, 1.2])
    with c_a:
        st.caption(f"CSV atual ({status_txt}): `{file_name}`")
    with c_b:
        st.caption(f"Tamanho: {size_txt}")
    with c_c:
        st.caption(f"Falhas: {fail_count}/{step_count}")
    with c_d:
        st.caption(f"Preparo: {dur_txt} | {ts}")
    _steps_plan_view = [{"name": str(s.get("step") or ""), "critical": bool(s.get("critical", False))} for s in (report.get("steps") or [])]
    _tmp_target = st.empty()
    render_csv_prepare_live_dashboard(
        st,
        _tmp_target,
        dict(report),
        steps_plan=_steps_plan_view,
        current_idx=None,
        current_step="",
    )
    _expand_details = bool(report.get("show_details_once", False))
    with st.expander("Detalhes do carregamento do CSV (última troca)", expanded=_expand_details):
        _rows = []
        for s in (report.get("steps") or []):
            if not isinstance(s, dict):
                continue
            _rows.append(
                {
                    "Etapa": str(s.get("step", "")),
                    "Status": "OK" if bool(s.get("ok", False)) else "Erro",
                    "Crítica": bool(s.get("critical", False)),
                    "Tempo (s)": round(float(s.get("elapsed_s", 0.0) or 0.0), 3),
                    "Detalhe": str(s.get("summary", "") or ""),
                }
            )
        if _rows:
            st.dataframe(pd.DataFrame(_rows), width="stretch", hide_index=True)
        if isinstance(report.get("artifacts"), list) and report["artifacts"]:
            render_csv_prepare_artifacts_cards(st, report["artifacts"])
        if report.get("errors"):
            for msg in report.get("errors") or []:
                st.error(str(msg))
    # abre automaticamente só uma vez por troca
    report["show_details_once"] = False
