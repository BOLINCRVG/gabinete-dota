from __future__ import annotations

from typing import Any


def build_confidence_summary(conf: dict[str, Any] | None) -> dict[str, str]:
    payload = dict(conf or {})
    n_total = int(payload.get("n_total") or 0)
    logloss = payload.get("logloss")
    gain_vs_team = payload.get("gain_vs_team_logloss")
    pipeline_adj = float(payload.get("pipeline_adj", 0.0) or 0.0)
    baseline_adj = float(payload.get("baseline_adj", 0.0) or 0.0)
    return {
        "grade": str(payload.get("grade", "C (Baixa)")),
        "score_text": f"{round(float(payload.get('score', 0.0) or 0.0)):d}/100",
        "source": str(payload.get("source", "global")),
        "summary_line": (
            f"Semaforo: n={n_total} | "
            f"logloss={_fmt_num(logloss, 4)} | "
            f"ganho_vs_team={_fmt_num(gain_vs_team, 4)} | "
            f"adj_pipeline={_fmt_num(pipeline_adj, 1)} | "
            f"adj_baseline={_fmt_num(baseline_adj, 1)}."
        ),
        "baseline_line": _build_baseline_line(payload),
        "pipeline_line": _build_pipeline_line(payload),
    }


def _fmt_num(value: Any, decimals: int) -> str:
    try:
        return f"{float(value):.{int(decimals)}f}"
    except Exception:
        return "nan"


def _build_baseline_line(payload: dict[str, Any]) -> str:
    source = str(payload.get("baseline_source") or "").strip()
    if not source:
        return ""
    band = str(payload.get("baseline_band") or "").strip()
    bias = _fmt_num(float(payload.get("baseline_adj", 0.0) or 0.0), 1)
    if band:
        return f"Baseline confianca: {source} | band={band} | bias={bias}."
    return f"Baseline confianca: {source} | bias={bias}."


def _build_pipeline_line(payload: dict[str, Any]) -> str:
    summary = str(payload.get("pipeline_summary") or "").strip()
    if not summary:
        return ""
    method = str(payload.get("pipeline_meta_method") or "").strip()
    if method:
        return f"Camadas A/B/C: {summary} | C={method}"
    return f"Camadas A/B/C: {summary}"


def build_pipeline_step_rows(trace_obj: dict[str, Any] | None) -> list[dict[str, Any]]:
    payload = dict(trace_obj or {})
    pipeline = (payload.get("pipeline") or {}) if isinstance(payload.get("pipeline"), dict) else {}
    rows: list[dict[str, Any]] = []
    for step in (pipeline.get("steps") or []):
        if not isinstance(step, dict):
            continue
        before = step.get("before")
        after = step.get("after")
        try:
            delta_pp = (float(after) - float(before)) * 100.0
        except Exception:
            delta_pp = None
        rows.append(
            {
                "Etapa": str(step.get("stage", "")),
                "Aplicou": bool(step.get("applied", False)),
                "Antes %": (float(before) * 100.0) if before is not None else None,
                "Depois %": (float(after) * 100.0) if after is not None else None,
                "Delta (pp)": delta_pp,
                "Fonte": str(step.get("source", "")) if step.get("source") is not None else "",
            }
        )
    return rows


def build_pipeline_explainability(step_rows: list[dict[str, Any]], conf: dict[str, Any] | None, *, top_n: int = 5) -> dict[str, Any]:
    rows = [dict(x) for x in (step_rows or []) if isinstance(x, dict) and bool(x.get("Aplicou"))]
    for row in rows:
        try:
            row["_abs"] = abs(float(row.get("Delta (pp)")))
        except Exception:
            row["_abs"] = -1.0
    rows = sorted(rows, key=lambda x: float(x.get("_abs", -1.0)), reverse=True)
    top_rows = rows[: max(1, int(top_n))]
    moves: list[str] = []
    for row in top_rows:
        value = row.get("Delta (pp)")
        try:
            moves.append(f"{row.get('Etapa', '')!s} {float(value):.1f}pp")
        except Exception:
            moves.append(f"{row.get('Etapa', '')!s} -pp")
    why: list[str] = []
    payload = dict(conf or {})
    try:
        pipeline_adj = float(payload.get("pipeline_adj", 0.0) or 0.0)
        if pipeline_adj:
            why.append(f"ajuste_pipeline={pipeline_adj:.1f}")
    except Exception:
        pass
    try:
        baseline_adj = float(payload.get("baseline_adj", 0.0) or 0.0)
        if baseline_adj:
            why.append(f"ajuste_baseline={baseline_adj:.1f}")
    except Exception:
        pass
    return {
        "rows": [
            {k: v for k, v in row.items() if k != "_abs"}
            for row in top_rows
        ],
        "moves_line": ("Quem puxou a odd: " + " | ".join(moves)) if moves else "",
        "why_line": ("Por que a confianca mudou: " + " | ".join(why)) if why else "",
    }
