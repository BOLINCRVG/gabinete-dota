# -*- coding: utf-8 -*-
"""Helpers extraídos do plays_app (fatiamento F3, 2026-07-18).

Funções PURAS (não referenciam nada definido no plays_app; sem __file__).
O plays_app importa todas de volta, então o namespace que alimenta o
ctx das abas (globals()->build_render_ctx) segue idêntico.
"""
from __future__ import annotations

from app_core.csv_bootstrap_runtime_core import (
    consume_csv_prepare_refresh as core_consume_csv_prepare_refresh,
)
from app_core.csv_prepare_ui_core import (
    render_csv_prepare_live_dashboard as _render_csv_prepare_live_dashboard,
    render_csv_prepare_report_summary as _render_csv_prepare_report_summary,
)
from app_core.ui_labels import fmt_bytes_short as _fmt_bytes_short
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
import pandas as pd
import time


def ensure_csv_runtime_ready(
    csv_path: str,
    *,
    hist_csvs: Optional[List[str]] = None,
    artifact_path: Optional[str] = None,
    st: Any,
    ctx: Mapping[str, Any],
) -> Dict[str, Any]:
    _file_sig = ctx.get("_file_sig")
    _resolve_players_artifact_path = ctx.get("_resolve_players_artifact_path")
    _FIT_PARAMS_PATH = ctx.get("_FIT_PARAMS_PATH")
    _TEAM_STATE_ARTIFACT_PATH = ctx.get("_TEAM_STATE_ARTIFACT_PATH")
    _ML_CONFIDENCE_BASELINE_PATH = ctx.get("_ML_CONFIDENCE_BASELINE_PATH")
    _csv_header_check = ctx.get("_csv_header_check")
    _csv_diag = ctx.get("_csv_diag")
    _list_teams = ctx.get("_list_teams")
    _load_team_games = ctx.get("_load_team_games")
    _load_player_rows = ctx.get("_load_player_rows")
    APP_ROOT = ctx.get("APP_ROOT")

    csv_path = str(csv_path or "").strip()
    hist_csvs = [str(p) for p in (hist_csvs or []) if str(p or "").strip()]
    artifact_path = str(artifact_path or "").strip()

    sig = _file_sig(csv_path)
    _force = core_consume_csv_prepare_refresh(st.session_state)
    _last_sig_raw = st.session_state.get("_csv_prepare_last_sig")
    _last_sig = tuple(_last_sig_raw) if isinstance(_last_sig_raw, (list, tuple)) else None
    _last_report = st.session_state.get("_csv_prepare_report")
    if (not _force) and _last_sig == tuple(sig) and isinstance(_last_report, dict):
        _render_csv_prepare_report_summary(st, _last_report)
        return _last_report

    started = datetime.now().isoformat(timespec="seconds")
    t_all0 = time.perf_counter()
    report: Dict[str, Any] = {
        "csv_path": csv_path,
        "csv_sig": list(sig),
        "hist_csvs": hist_csvs,
        "artifact_path": artifact_path,
        "started_at": started,
        "steps": [],
        "errors": [],
        "ok": False,
        "show_details_once": True,
    }

    st.markdown("### Carregando novo CSV")
    st.caption("Preparando cache e validando etapas do CSV ativo (arquivo, cabeçalho, diagnóstico, times, team_games e players).")
    st.code(csv_path if csv_path else "(sem caminho)", language="text")
    _prog = st.progress(0, text="Iniciando...")
    _step_cards = st.empty()
    _summary_box = st.empty()

    _artifacts_rows: List[Dict[str, Any]] = []
    if artifact_path:
        _artifacts_rows.append({"Arquivo": "ml_artifact.json", "Caminho": artifact_path, "Existe": bool(Path(artifact_path).exists())})
        # players_artifact so entra no checklist se o deploy atual USA players
        # (meta.players_layer) — o Gabinete Dota deploya sem players ate a P3.
        _expects_players_row = True
        try:
            import json as _json_pl
            _meta_pl = (_json_pl.loads(Path(artifact_path).read_text(encoding="utf-8")).get("meta") or {})
            _expects_players_row = bool(_meta_pl.get("players_layer"))
        except (OSError, ValueError, AttributeError):
            pass
        if _expects_players_row:
            _artifacts_rows.append(
                {
                    "Arquivo": "players_artifact.json",
                    "Caminho": str(_resolve_players_artifact_path(artifact_path, APP_ROOT)),
                    "Existe": bool(Path(str(_resolve_players_artifact_path(artifact_path, APP_ROOT))).exists()),
                }
            )
        _artifacts_rows.append(
            {
                "Arquivo": "params_fitted.json",
                "Caminho": str(_FIT_PARAMS_PATH),
                "Existe": bool(Path(str(_FIT_PARAMS_PATH)).exists()),
            }
        )
        _artifacts_rows.append(
            {
                "Arquivo": "team_state_artifact.json",
                "Caminho": str(_TEAM_STATE_ARTIFACT_PATH),
                "Existe": bool(Path(str(_TEAM_STATE_ARTIFACT_PATH)).exists()),
            }
        )
        _artifacts_rows.append(
            {
                "Arquivo": "ml_confidence_baseline.json",
                "Caminho": str(_ML_CONFIDENCE_BASELINE_PATH),
                "Existe": bool(Path(str(_ML_CONFIDENCE_BASELINE_PATH)).exists()),
            }
        )
    report["artifacts"] = _artifacts_rows
    report["_t_all0"] = float(t_all0)

    _steps_plan = [
        {"name": "Arquivo diário", "critical": True},
        {"name": "CSVs históricos selecionados", "critical": False},
        {"name": "Cabeçalho/colunas", "critical": True},
        {"name": "Diagnóstico rápido (rows/gameids)", "critical": False},
        {"name": "Times (lista/canonização)", "critical": True},
        {"name": "Team games (cache base)", "critical": True},
        {"name": "Players rows (cache Players/Campeões)", "critical": False},
        {"name": "Artifacts relacionados", "critical": False},
    ]
    _total_steps = max(1, len(_steps_plan))

    def _flush_live_cards(*, current_idx: Optional[int] = None, current_step: str = "") -> None:
        _render_csv_prepare_live_dashboard(
            st,
            _step_cards,
            report,
            steps_plan=_steps_plan,
            current_idx=current_idx,
            current_step=current_step,
        )

    def _run_step(step_idx: int, step_name: str, fn, *, critical: bool) -> Any:
        _prog.progress(int((step_idx / _total_steps) * 100), text=f"[{step_idx}/{_total_steps}] {step_name}...")
        _flush_live_cards(current_idx=step_idx, current_step=step_name)
        t0 = time.perf_counter()
        ok = False
        summary = ""
        payload: Any = None
        try:
            payload = fn()
            ok = True
            if isinstance(payload, str):
                summary = payload
            elif isinstance(payload, dict):
                summary = str(payload.get("summary") or payload.get("detail") or payload.get("msg") or "")
            elif payload is None:
                summary = "OK"
            else:
                summary = str(payload)
        except Exception as e:
            ok = False
            summary = f"{type(e).__name__}: {e}"
            if critical:
                report.setdefault("errors", []).append(f"{step_name}: {summary}")
        elapsed = time.perf_counter() - t0
        report["steps"].append(
            {
                "step": step_name,
                "ok": bool(ok),
                "critical": bool(critical),
                "elapsed_s": float(elapsed),
                "summary": summary,
            }
        )
        _flush_live_cards()
        if (not ok) and critical:
            raise RuntimeError(f"{step_name}: {summary}")
        return payload

    def _set_last_prepare_step_summary(summary: str) -> None:
        steps = report.get("steps")
        if not isinstance(steps, list) or not steps:
            return
        last_step = steps[-1]
        if not isinstance(last_step, dict):
            return
        last_step["summary"] = str(summary or "")
        _flush_live_cards()

    try:
        _run_step(
            1,
            "Arquivo diário",
            lambda: (
                (_ for _ in ()).throw(FileNotFoundError("CSV diário não encontrado"))
                if (not csv_path or not Path(csv_path).exists())
                else {
                    "summary": f"{Path(csv_path).name} | {_fmt_bytes_short(sig[1])} | mtime={datetime.fromtimestamp(int(sig[2])).isoformat(timespec='seconds') if int(sig[2] or 0) > 0 else '-'}"
                }
            ),
            critical=True,
        )

        def _hist_summary() -> str:
            if not hist_csvs:
                return "Nenhum histórico selecionado"
            ok_n = 0
            miss_n = 0
            for p in hist_csvs:
                if Path(str(p)).exists():
                    ok_n += 1
                else:
                    miss_n += 1
            if miss_n > 0:
                report.setdefault("errors", []).append(f"Históricos ausentes: {miss_n}")
            return f"{ok_n} encontrado(s), {miss_n} ausente(s)"

        _run_step(2, "CSVs históricos selecionados", _hist_summary, critical=False)

        hdr = _run_step(3, "Cabeçalho/colunas", lambda: _csv_header_check(sig), critical=True)
        if isinstance(hdr, dict):
            if bool(hdr.get("error")):
                raise RuntimeError(str(hdr.get("error")))
            miss = [str(x) for x in (hdr.get("missing_required") or [])]
            if miss:
                raise RuntimeError("Colunas obrigatórias ausentes: " + ", ".join(miss))
            _set_last_prepare_step_summary(f"{int(hdr.get('n_columns', 0))} colunas | obrigatórias OK")

        diag = _run_step(4, "Diagnóstico rápido (rows/gameids)", lambda: _csv_diag(sig), critical=False)
        if isinstance(diag, dict) and diag:
            report["csv_diag"] = diag
            _set_last_prepare_step_summary(
                f"{int(diag.get('rows',0))} linhas | {int(diag.get('unique_gameids',0))} gameids | "
                f"{int(diag.get('games_complete',0))} completos"
            )

        teams_local = _run_step(5, "Times (lista/canonização)", lambda: _list_teams(sig), critical=True)
        if isinstance(teams_local, list):
            report["teams_count"] = len(teams_local)
            _set_last_prepare_step_summary(f"{len(teams_local)} times")

        tg = _run_step(6, "Team games (cache base)", lambda: _load_team_games(sig), critical=True)
        if isinstance(tg, pd.DataFrame):
            report["team_games_shape"] = [int(tg.shape[0]), int(tg.shape[1])]
            _g = int(tg["gameid"].nunique()) if "gameid" in tg.columns else 0
            _lg = int(tg["league"].dropna().astype(str).nunique()) if "league" in tg.columns else 0
            _set_last_prepare_step_summary(f"{int(tg.shape[0])} rows | {_g} jogos | {_lg} ligas")

        dpl = _run_step(7, "Players rows (cache Players/Campeões)", lambda: _load_player_rows(sig), critical=False)
        if isinstance(dpl, pd.DataFrame) and (not dpl.empty):
            _players_n = int(dpl["player"].dropna().astype(str).nunique()) if "player" in dpl.columns else 0
            _champs_n = int(dpl["champ"].dropna().astype(str).nunique()) if "champ" in dpl.columns else 0
            _set_last_prepare_step_summary(f"{int(dpl.shape[0])} rows | {_players_n} players | {_champs_n} champs")

        _run_step(
            8,
            "Artifacts relacionados",
            lambda: f"{sum(1 for r in _artifacts_rows if bool(r.get('Existe')))} de {len(_artifacts_rows)} encontrados",
            critical=False,
        )

        report["ok"] = True
    except Exception as e:
        report["ok"] = False
        report.setdefault("errors", []).append(f"{type(e).__name__}: {e}")

    report["elapsed_s"] = float(time.perf_counter() - t_all0)
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _flush_live_cards()
    st.session_state["_csv_prepare_last_sig"] = list(sig)
    st.session_state["_csv_prepare_report"] = report

    _prog.progress(100, text="Concluído" if bool(report.get("ok", False)) else "Falhou")
    if bool(report.get("ok", False)):
        _summary_box.success(
            f"CSV preparado com sucesso em {report['elapsed_s']:.1f}s. "
            f"Times={int(report.get('teams_count', 0) or 0)} | "
            f"TeamGames={int((report.get('team_games_shape') or [0])[0] or 0)} linhas"
        )
    else:
        _summary_box.error("Falha ao preparar o CSV ativo. Veja as etapas acima.")
        for _err in (report.get("errors") or []):
            st.error(str(_err))
        st.stop()

    _render_csv_prepare_report_summary(st, report)
    return report

