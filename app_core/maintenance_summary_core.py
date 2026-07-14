from __future__ import annotations

from typing import Any, Dict, List, Mapping


def _short_date(iso: Any) -> str:
    """ISO 'YYYY-MM-DD...' -> 'DD/MM/YYYY' (vazio se não parsear)."""
    s = str(iso or "")[:10]
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return f"{s[8:10]}/{s[5:7]}/{s[0:4]}"
    return ""


def build_maintenance_quality_status(status: Mapping[str, Any]) -> Dict[str, Any]:
    sta = dict(status or {}) if isinstance(status, Mapping) else {}
    severity = str(sta.get("severity") or "").strip().lower()
    need_model = bool(sta.get("need_retrain_model"))
    need_params = bool(sta.get("need_retrain_params"))
    need_wf = bool(sta.get("need_walkforward"))
    need_patch = bool(sta.get("need_patch_recalibration"))
    params_fit_running = bool(sta.get("params_fit_running"))
    params_rt = sta.get("params_fit_runtime") if isinstance(sta.get("params_fit_runtime"), Mapping) else {}
    params_summary = str(params_rt.get("summary") or "").strip()

    model_state = "ready"
    model_text = "Artifact e players alinhados ao CSV ativo."
    if need_model:
        model_state = "pending"
        model_text = "Retreino de modelo ainda pendente."

    operational_state = "ready"
    operational_text = "Leitura diária pronta para operar."
    if need_model and severity == "critical":
        operational_state = "blocked"
        operational_text = "Bloqueada enquanto o modelo base não for atualizado."
    elif params_fit_running:
        operational_state = "running"
        operational_text = (
            f"Refino operacional em andamento: {params_summary}."
            if params_summary
            else "Refino operacional em andamento: parâmetros ainda estão sendo ajustados."
        )
    elif need_params or need_wf:
        operational_state = "pending"
        pending_parts: List[str] = []
        if need_params:
            pending_parts.append("parâmetros")
        if need_wf:
            pending_parts.append("walk-forward")
        operational_text = "Ainda falta concluir " + " + ".join(pending_parts) + "."
    elif need_patch and severity != "critical":
        operational_state = "ready_warn"
        operational_text = "Pronta para operar, com alerta de patch/liga ainda aberto."

    deep_done = bool(sta.get("deep_fit_done"))
    deep_stale = bool(sta.get("deep_fit_stale"))
    deep_at = _short_date(sta.get("deep_fit_at"))
    if not deep_done:
        deep_state = "pending"
        deep_text = "Full fit manual/profundo ainda não foi rodado."
    elif deep_stale:
        deep_state = "ready_warn"
        deep_text = (
            f"Full fit rodado em {deep_at}, mas o modelo mudou depois — vale rodar de novo."
            if deep_at else "Full fit rodado, mas o modelo mudou depois — vale rodar de novo."
        )
    else:
        deep_state = "ready"
        deep_text = (
            f"Full fit rodado em {deep_at} (alinhado ao modelo atual)."
            if deep_at else "Full fit rodado (alinhado ao modelo atual)."
        )

    headline = "Qualidade operacional pronta."
    if operational_state == "blocked":
        headline = "Qualidade operacional bloqueada por pendência crítica."
    elif operational_state in {"running", "pending"}:
        headline = "Qualidade operacional ainda em preparação."
    elif operational_state == "ready_warn":
        headline = "Qualidade operacional pronta, com alertas não críticos."

    return {
        "headline": str(headline),
        "model_base": {
            "state": str(model_state),
            "title": "Modelo base",
            "text": str(model_text),
        },
        "operational": {
            "state": str(operational_state),
            "title": "Qualidade operacional",
            "text": str(operational_text),
        },
        "deep": {
            "state": str(deep_state),
            "title": "Qualidade profunda",
            "text": str(deep_text),
        },
    }


def build_quick_maintenance_alerts(status: Mapping[str, Any]) -> Dict[str, Any]:
    alerts_raw = list(status.get("alerts") or []) if isinstance(status, Mapping) else []
    alerts: List[str] = []
    params_fit_running = bool(status.get("params_fit_running")) if isinstance(status, Mapping) else False

    def _push(msg: str) -> None:
        text = str(msg or "").strip()
        if text and text not in alerts:
            alerts.append(text)

    for msg in alerts_raw:
        text = str(msg or "").strip()
        if not text or "(flag interna)" in text:
            continue
        if text.startswith(("Artifact ausente", "Sem artifact para o CSV")):
            _push("Retreinar modelo: ml_artifact.json nao encontrado.")
        elif text.startswith("Drift de dados: CSV diario mais novo que o `ml_artifact.json`"):
            _push("Retreinar modelo: CSV diario esta mais novo que o ml_artifact.json.")
        elif text.startswith("Artifact antigo"):
            _push("Retreinar modelo: artifact esta antigo (mais de 10 dias).")
        elif text.startswith(("Parametros ausentes", "Parâmetros ausentes")):
            _push("Retreinar parametros: params_fitted.json ausente.")
        elif text.startswith(("Parametros desatualizados vs artifact", "Parâmetros desatualizados vs artifact")):
            if not params_fit_running:
                _push("Retreinar parametros: params_fitted.json esta desatualizado vs artifact.")
        elif text.startswith("Auto-retune mensal pendente"):
            _push("Auto-retune mensal: params_fitted.json e de mes anterior.")
        elif text.startswith(("Retreino de parametros em execucao", "Retreino de parâmetros em execução")):
            _push("Retreino de parametros em execucao.")
        elif text.startswith("Walk-forward desatualizado"):
            _push("Rodar walk-forward: ultimo relatorio tem mais de 7 dias.")
        elif text.startswith("Patch/Liga em REFIT"):
            _push(text)
        elif text.startswith(("Patch/Liga em ATENCAO", "Patch/Liga em ATENÇÃO")):
            _push(text)

    if bool(status.get("need_retrain_model")) and not any(x.startswith("Retreinar modelo:") for x in alerts):
        _push("Retreinar modelo: manutencao pendente.")
    if bool(status.get("need_retrain_params")):
        if params_fit_running:
            _push("Retreino de parametros em execucao.")
        elif not any(x.startswith(("Retreinar parametros:", "Auto-retune mensal:")) for x in alerts):
            _push("Retreinar parametros: manutencao pendente.")
    if bool(status.get("need_walkforward")) and not any(x.startswith("Rodar walk-forward:") for x in alerts):
        _push("Rodar walk-forward: manutencao pendente.")

    return {
        "alerts": alerts,
        "ok": not bool(alerts),
        "need_retrain_model": bool(status.get("need_retrain_model")),
        "need_retrain_params": bool(status.get("need_retrain_params")),
        "need_walkforward": bool(status.get("need_walkforward")),
        "params_fit_running": bool(params_fit_running),
    }
