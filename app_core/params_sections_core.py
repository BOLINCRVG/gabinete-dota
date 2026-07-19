# -*- coding: utf-8 -*-
"""Seções da aba Parâmetros (extraídas do render_params_tab de 1.528 linhas).

Refactor 2026-07-19: cada `with st.expander(...)` de 1º nível vira uma
função que recebe (st, ctx) e re-obtém do ctx os mesmos nomes — bloco
verbatim, comportamento idêntico. Valida com E2E + validação visual dele.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path  # noqa: F401 - usado por blocos extraídos
from typing import Any, Dict, List

import pandas as pd


def _render_governanca_ml(*, st, ctx) -> None:
    """Seção da aba Parâmetros extraída de render_params_tab (refactor
    2026-07-19). Bloco movido VERBATIM; os nomes vêm do MESMO ctx."""
    APP_ROOT = ctx.get("APP_ROOT")
    _df_contract_view = ctx.get("_df_contract_view")
    _format_num = ctx.get("_format_num")
    core_append_governance_log = ctx.get("core_append_governance_log")
    core_build_anchor_cases_rows = ctx.get("core_build_anchor_cases_rows")
    core_build_leakage_file_rows = ctx.get("core_build_leakage_file_rows")
    core_build_parts_rows = ctx.get("core_build_parts_rows")
    core_build_phase12_by_league_rows = ctx.get("core_build_phase12_by_league_rows")
    core_build_stop_rules_banner = ctx.get("core_build_stop_rules_banner")
    core_build_stop_rules_global_band_rows = ctx.get("core_build_stop_rules_global_band_rows")
    core_build_stop_rules_league_band_rows = ctx.get("core_build_stop_rules_league_band_rows")
    core_build_stop_rules_major_rows = ctx.get("core_build_stop_rules_major_rows")
    core_combine_governance_decisions = ctx.get("core_combine_governance_decisions")
    core_decision_banner_parts = ctx.get("core_decision_banner_parts")
    core_get_governance_last_log = ctx.get("core_get_governance_last_log")
    core_normalize_governance_decision = ctx.get("core_normalize_governance_decision")
    core_run_governance_commands = ctx.get("core_run_governance_commands")
    core_run_governance_operation = ctx.get("core_run_governance_operation")
    core_set_governance_last_log = ctx.get("core_set_governance_last_log")

    with st.expander("Governanca ML (backtest / promocao / calibração)", expanded=False):
        _bt_report_path = APP_ROOT / "backtest_pipeline_abc_by_league_report.json"
        _anchor_report_path = APP_ROOT / "anchor_regression_report.json"
        _leak_report_path = APP_ROOT / "data_leakage_label_report.json"
        _lacuna_report_path = APP_ROOT / "lacuna_guardrail_report.json"
        _prod_refresh_report_path = APP_ROOT / "prod_refresh_report.json"
        _phase1_report_path = APP_ROOT / "phase1_data_contract_report.json"
        _phase12_report_path = APP_ROOT / "phase12_roster_continuity_report.json"
        _phase4b_report_path = APP_ROOT / "phase4b_marginal_tuning_report.json"
        _phase6_report_path = APP_ROOT / "phase6_series_bo_report.json"
        _phase7_report_path = APP_ROOT / "phase7_layer_uncertainty_report.json"
        _phase8_report_path = APP_ROOT / "phase8_operations_report.json"
        _gov_last_run = str(st.session_state.get("_gov_last_run", "") or "").strip()
        _gov_last_log = str(core_get_governance_last_log(st.session_state) or "").strip()

        def _read_json_safe(_p: Path) -> Dict[str, Any]:
            try:
                if _p.exists():
                    _obj = json.loads(_p.read_text(encoding="utf-8"))
                    return _obj if isinstance(_obj, dict) else {}
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return {}
            return {}

        def _render_status_message(message: Any, message_type: Any = "error") -> None:
            _msg = str(message or "")
            _fn = getattr(st, str(message_type or "error"), None)
            if callable(_fn):
                _fn(_msg)
            else:
                st.caption(_msg)

        _bt_rep = _read_json_safe(_bt_report_path)
        _ar = _read_json_safe(_anchor_report_path)
        _lrp = _read_json_safe(_leak_report_path)
        _lgr = _read_json_safe(_lacuna_report_path)
        _pr = _read_json_safe(_prod_refresh_report_path)
        _p1 = _read_json_safe(_phase1_report_path)
        _p12 = _read_json_safe(_phase12_report_path)
        _p4b = _read_json_safe(_phase4b_report_path)
        _p6 = _read_json_safe(_phase6_report_path)
        _p7 = _read_json_safe(_phase7_report_path)
        _p8 = _read_json_safe(_phase8_report_path)

        _gvr1, _gvr2, _gvr3, _gvr4 = st.columns([1, 1, 1, 1])
        with _gvr1:
            _run_gov_fast = st.button("Rodar governança (rápido)", width='stretch', key="gov_run_fast")
        with _gvr2:
            _run_anchor_only = st.button("Rodar apenas âncoras", width='stretch', key="gov_run_anchor_only")
        with _gvr3:
            _run_lacuna_full = st.button("Rodar lacunas (completo)", width='stretch', key="gov_run_lacuna_full")
        with _gvr4:
            if _gov_last_run:
                st.caption(f"última execução: {_gov_last_run}")
        _gvr5, _gvr6 = st.columns([1.2, 2.8])
        with _gvr5:
            _run_prod_refresh = st.button("Rodar Prod Refresh", width='stretch', key="gov_run_prod_refresh")
        with _gvr6:
            _pr_dec = str((_pr.get("final_decision") if isinstance(_pr, dict) else "") or "").strip().lower()
            _pr_gen = str((_pr.get("generated_at") if isinstance(_pr, dict) else "") or "").strip()
            _pr_action = str((_pr.get("action") if isinstance(_pr, dict) else "") or "").strip()
            if _pr_dec:
                st.caption(f"Prod Refresh atual: {_pr_dec} | ação={_pr_action or '-'} | gerado: {_pr_gen or '-'}")
        _gvp1, _gvp2 = st.columns([1.2, 2.8])
        with _gvp1:
            _run_phase1_gate = st.button("Rodar Fase 1 (data contract)", width='stretch', key="gov_run_phase1_data_contract")
        with _gvp2:
            _p1_dec_hint = str((_p1.get("decision") if isinstance(_p1, dict) else "") or "").strip().lower()
            _p1_gen_hint = str((_p1.get("generated_at") if isinstance(_p1, dict) else "") or "").strip()
            if _p1_dec_hint:
                st.caption(f"Phase 1 atual: {_p1_dec_hint} | gerado: {_p1_gen_hint or '-'}")
        _gvp3, _gvp4 = st.columns([1.2, 2.8])
        with _gvp3:
            _run_phase12 = st.button("Rodar Fase 1.2 (roster A/B)", width='stretch', key="gov_run_phase12_roster")
        with _gvp4:
            _p12_dec = str((_p12.get("decision") if isinstance(_p12, dict) else "") or "").strip().lower()
            _p12_gen = str((_p12.get("generated_at") if isinstance(_p12, dict) else "") or "").strip()
            if _p12_dec:
                st.caption(f"Phase 1.2 atual: {_p12_dec} | gerado: {_p12_gen or '-'}")
        _gvp5, _gvp6 = st.columns([1.2, 2.8])
        with _gvp5:
            _run_phase8_ops = st.button("Rodar Fase 8 (operação)", width='stretch', key="gov_run_phase8_ops")
        with _gvp6:
            _p8_dec = str((_p8.get("decision") if isinstance(_p8, dict) else "") or "").strip().lower()
            _p8_gen = str((_p8.get("generated_at") if isinstance(_p8, dict) else "") or "").strip()
            if _p8_dec:
                st.caption(f"Phase 8 atual: {_p8_dec} | gerado: {_p8_gen or '-'}")

        if _run_gov_fast or _run_anchor_only or _run_lacuna_full or _run_phase1_gate or _run_phase12 or _run_phase8_ops or _run_prod_refresh:
            _cmds: List[List[str]] = []
            if _run_phase1_gate:
                _cmds.append([sys.executable, "tools/run_phase1_data_contract.py", "--timeout", "600"])
            elif _run_phase12:
                _cmds.append([sys.executable, "tools/eval_roster_continuity_phase12.py", "--speed", "fast"])
            elif _run_phase8_ops:
                _cmds.append([sys.executable, "tools/run_phase8_operations_gate.py", "--timeout", "900"])
            elif _run_prod_refresh:
                _cmds.append([sys.executable, "tools/prod_refresh.py", "--mode", "quick", "--phase12-speed", "fast", "--promote-on-pass"])
            else:
                if _run_lacuna_full:
                    _cmds.append([sys.executable, "tools/run_lacuna_guardrails.py"])
                else:
                    _cmds.append([sys.executable, "tools/anchor_regression_suite.py"])
                if _run_gov_fast and (not _run_lacuna_full):
                    _cmds.append([sys.executable, "tools/check_data_leakage_and_labels.py"])
            with st.spinner("Executando governança..."):
                _gov_exec = core_run_governance_commands(_cmds, cwd=APP_ROOT)
            st.session_state["_gov_last_run"] = str(_gov_exec.get("last_run") or datetime.now().isoformat(timespec="seconds"))
            core_set_governance_last_log(st.session_state, "\n\n".join([str(x) for x in (_gov_exec.get("logs") or [])]))
            _gov_last_run = str(st.session_state.get("_gov_last_run", "") or "").strip()
            _gov_last_log = str(core_get_governance_last_log(st.session_state) or "").strip()
            _bt_rep = _read_json_safe(_bt_report_path)
            _ar = _read_json_safe(_anchor_report_path)
            _lrp = _read_json_safe(_leak_report_path)
            _lgr = _read_json_safe(_lacuna_report_path)
            _pr = _read_json_safe(_prod_refresh_report_path)
            _p1 = _read_json_safe(_phase1_report_path)
            _p12 = _read_json_safe(_phase12_report_path)
            _p4b = _read_json_safe(_phase4b_report_path)
            _p6 = _read_json_safe(_phase6_report_path)
            _p7 = _read_json_safe(_phase7_report_path)
            _p8 = _read_json_safe(_phase8_report_path)

        if _gov_last_log:
            with st.expander("Log da última execução de governança", expanded=False):
                st.code(_gov_last_log, language="text")

        _parts = {
            "stop_rules": core_normalize_governance_decision(((_bt_rep.get("promotion_decision") or {}) if isinstance(_bt_rep, dict) else {}).get("decision")),
            "anchors": core_normalize_governance_decision((_ar.get("decision") if isinstance(_ar, dict) else "")),
            "leakage_labels": core_normalize_governance_decision((_lrp.get("decision") if isinstance(_lrp, dict) else "")),
            "lacunas": core_normalize_governance_decision((_lgr.get("decision") if isinstance(_lgr, dict) else "")),
            "prod_refresh": core_normalize_governance_decision((_pr.get("final_decision") if isinstance(_pr, dict) else "")),
            "phase1_data_contract": core_normalize_governance_decision((_p1.get("decision") if isinstance(_p1, dict) else "")),
            "phase12_roster_ab": core_normalize_governance_decision((_p12.get("decision") if isinstance(_p12, dict) else "")),
            "phase4b_marginal_tuning": core_normalize_governance_decision((_p4b.get("decision") if isinstance(_p4b, dict) else "")),
            "phase6_series_bo": core_normalize_governance_decision((_p6.get("decision") if isinstance(_p6, dict) else "")),
            "phase7_layer_uncertainty": core_normalize_governance_decision((_p7.get("decision") if isinstance(_p7, dict) else "")),
            "phase8_operations": core_normalize_governance_decision((_p8.get("decision") if isinstance(_p8, dict) else "")),
        }
        _gov_combo = core_combine_governance_decisions(_parts)
        _final = str(_gov_combo.get("final") or "unknown")
        _final_msg = str(_gov_combo.get("message") or "")
        if _final == "promote":
            _render_status_message(_final_msg, "success")
        elif _final == "shadow_only":
            _render_status_message(_final_msg, "warning")
        else:
            _render_status_message(_final_msg, "error")

        st.caption("Resumo operacional do último backtest full: decisão de promoção (stop-rules) e calibração por faixa de odd.")
        if not _bt_report_path.exists():
            st.caption("Relatório de backtest não encontrado. Rode: python tools\\backtest_pipeline_abc_by_league.py --speed full")
        else:
            if not _bt_rep:
                st.error(f"Falha ao ler {_bt_report_path.name!s}")
            if isinstance(_bt_rep, dict) and _bt_rep:
                _global_bt = _bt_rep.get("global") or {}
                _bt_banner = core_build_stop_rules_banner(_bt_rep)
                _render_status_message(_bt_banner.get("message"), _bt_banner.get("message_type"))
                _g1, _g2, _g3 = st.columns(3)
                with _g1:
                    st.metric("Rows", value=str(int(_bt_rep.get("rows_total") or 0)))
                with _g2:
                    st.metric("Ganho logloss", value=_format_num(float(_global_bt.get("gain_logloss_vs_base", float("nan"))), 4))
                with _g3:
                    st.metric("Ganho brier", value=_format_num(float(_global_bt.get("gain_brier_vs_base", float("nan"))), 4))

                _reasons = _bt_banner.get("reasons") or []
                if isinstance(_reasons, list) and _reasons:
                    st.caption("Motivos: " + " | ".join([str(x) for x in _reasons[:8]]))

                _df_major = pd.DataFrame(core_build_stop_rules_major_rows(_bt_rep))
                if not _df_major.empty:
                    if not _df_major.empty:
                        st.markdown("##### Ligas principais (stop-rules)")
                        st.dataframe(
                            _df_contract_view(
                                _df_major,
                                ["Liga", "Rows", "Ganho logloss", "Fail drawdown"],
                                context="Parâmetros > Governanca ML > Ligas principais",
                                fill_value="-",
                            ),
                            width='stretch',
                            hide_index=True,
                        )

                _mon = _bt_rep.get("monitoring") or {}
                _df_bands = pd.DataFrame(core_build_stop_rules_global_band_rows(_bt_rep))
                if not _df_bands.empty:
                    st.markdown("##### Calibração por faixa (favorito do modelo, global)")
                    st.dataframe(
                        _df_contract_view(
                            _df_bands,
                            ["Faixa odd fav", "N", "Gap calib final (pp)", "Gap calib base (pp)", "Winrate real %", "Prob média %"],
                            context="Parâmetros > Governanca ML > Faixas de odd",
                            fill_value="-",
                        ),
                        width='stretch',
                        hide_index=True,
                    )

                _by_league_mon = _mon.get("by_league") or {}
                if isinstance(_by_league_mon, dict) and _by_league_mon:
                    _lg_opts = ["(global)", *sorted([str(k) for k in _by_league_mon.keys() if str(k).strip()])]
                    if str(st.session_state.get("gov_ml_mon_league_sel", "") or "") not in _lg_opts:
                        st.session_state["gov_ml_mon_league_sel"] = _lg_opts[0]
                    _lg_sel = st.selectbox("Liga (monitoramento por faixa)", _lg_opts, key="gov_ml_mon_league_sel")
                    if _lg_sel != "(global)":
                        _df_lf = pd.DataFrame(core_build_stop_rules_league_band_rows(_bt_rep, _lg_sel))
                        if not _df_lf.empty:
                            st.dataframe(
                                _df_contract_view(
                                    _df_lf,
                                    ["Faixa odd fav", "N", "Gap calib final (pp)", "Winrate real %", "Prob média %"],
                                    context="Parâmetros > Governanca ML > Faixas por liga",
                                    fill_value="-",
                                ),
                                width='stretch',
                                hide_index=True,
                            )

        st.markdown("##### Casos âncora (regressão semântica)")
        st.caption("Valida faixas de odds em confrontos de referência (ex.: LNG/TT, WE/TES, Gen.G/T1).")
        if not _anchor_report_path.exists():
            st.caption("Relatório não encontrado. Rode: python tools\\anchor_regression_suite.py")
        else:
            if not _ar:
                st.error(f"Falha ao ler {_anchor_report_path.name!s}")
            if isinstance(_ar, dict) and _ar:
                _sum_ar = _ar.get("summary") or {}
                _ar_banner = core_decision_banner_parts(_ar.get("decision"), prefix="âncoras")
                _render_status_message(_ar_banner.get("message"), _ar_banner.get("message_type"))
                _ac1, _ac2, _ac3 = st.columns(3)
                with _ac1:
                    st.metric("âncoras (total)", value=str(int(_sum_ar.get("total") or 0)))
                with _ac2:
                    st.metric("OK", value=str(int(_sum_ar.get("ok") or 0)))
                with _ac3:
                    st.metric("FAIL", value=str(int(_sum_ar.get("fail") or 0)))
                _df_ar = pd.DataFrame(core_build_anchor_cases_rows(_ar))
                if not _df_ar.empty:
                    st.dataframe(
                        _df_contract_view(
                            _df_ar,
                            ["Caso", "OK", "Mercado", "Lado", "Odd", "Min", "Max"],
                            optional_cols=["Erro"],
                            context="Parâmetros > Governanca ML > Casos âncora",
                            fill_value="-",
                        ),
                        width='stretch',
                        hide_index=True,
                    )

        st.markdown("##### Qualidade de dados / leakage / labels")
        st.caption("Checks automáticos de qualidade de rótulo e risco de leakage (heurístico e auditável).")
        if not _leak_report_path.exists():
            st.caption("Relatório não encontrado. Rode: python tools\\check_data_leakage_and_labels.py")
        else:
            if not _lrp:
                st.error(f"Falha ao ler {_leak_report_path.name!s}")
            if isinstance(_lrp, dict) and _lrp:
                _lrp_banner = core_decision_banner_parts(_lrp.get("decision"), prefix="Leakage/labels")
                _render_status_message(_lrp_banner.get("message"), _lrp_banner.get("message_type"))
                _sum_lrp = _lrp.get("summary") or {}
                _lc1, _lc2, _lc3, _lc4 = st.columns(4)
                with _lc1:
                    st.metric("CSV files", value=str(int(_sum_lrp.get("csv_files") or 0)))
                with _lc2:
                    st.metric("Rows", value=str(int(_sum_lrp.get("rows_total") or 0)))
                with _lc3:
                    st.metric("Label issues", value=str(int(_sum_lrp.get("label_issues_total") or 0)))
                with _lc4:
                    st.metric("Critical", value=str(int(_sum_lrp.get("critical_issues_total") or 0)))
                _leak_meta = (_lrp.get("leakage_risk") or {})
                _rk = _leak_meta.get("high_risk_raw_columns_present") or []
                if isinstance(_rk, list) and _rk:
                    st.caption("Colunas de outcome presentes no CSV (informativo): " + ", ".join([str(x) for x in _rk[:20]]))
                _df_lrp = pd.DataFrame(core_build_leakage_file_rows(_lrp))
                if not _df_lrp.empty:
                    st.dataframe(
                        _df_contract_view(
                            _df_lrp,
                            ["Arquivo", "OK", "Rows"],
                            optional_cols=["Errors", "Warnings"],
                            context="Parâmetros > Governanca ML > Leakage/labels",
                            fill_value="-",
                        ),
                        width='stretch',
                        hide_index=True,
                    )

        st.markdown("##### Lacunas (guardrail único)")
        st.caption("Decisão consolidada: stop-rules + âncoras + leakage + cobertura + frescor de artifacts.")
        if not _lacuna_report_path.exists():
            st.caption("Relatório não encontrado. Rode: python tools\\run_lacuna_guardrails.py")
        else:
            if not _lgr:
                st.error(f"Falha ao ler {_lacuna_report_path.name!s}")
            if isinstance(_lgr, dict) and _lgr:
                _lgr_banner = core_decision_banner_parts(_lgr.get("decision"), prefix="Lacunas")
                _render_status_message(_lgr_banner.get("message"), _lgr_banner.get("message_type"))

                _parts_lg = _lgr.get("parts") or {}
                _df_parts = pd.DataFrame(core_build_parts_rows(_parts_lg))
                if not _df_parts.empty:
                    st.dataframe(
                        _df_contract_view(
                            _df_parts,
                            ["Check", "Decision"],
                            context="Parâmetros > Governanca ML > Lacunas (parts)",
                            fill_value="-",
                        ),
                        width='stretch',
                        hide_index=True,
                    )

                _cov = _lgr.get("coverage") or {}
                _cov_sum = _cov.get("summary") or {}
                _k1, _k2, _k3, _k4 = st.columns(4)
                with _k1:
                    st.metric("Ligas", value=str(int(_cov_sum.get("league_count") or 0)))
                with _k2:
                    st.metric("Rows total", value=str(int(_cov_sum.get("rows_total") or 0)))
                with _k3:
                    st.metric("Low sample ligas", value=str(int(_cov_sum.get("low_sample_leagues_lt30") or 0)))
                with _k4:
                    st.metric("Blank team rate", value=_format_num(float(_cov_sum.get("blank_team_rate", 0.0) or 0.0) * 100.0, 3) + "%")

        st.markdown("##### Phase 1 - Data Contract")
        st.caption("Gate de consistência de dados pré-ML (leakage/labels + cobertura).")
        if not _phase1_report_path.exists():
            st.caption("Relatório não encontrado. Rode: python tools\\run_phase1_data_contract.py --timeout 600")
        else:
            if not _p1:
                st.error(f"Falha ao ler {_phase1_report_path.name!s}")
            if isinstance(_p1, dict) and _p1:
                _p1_banner = core_decision_banner_parts(_p1.get("decision"), prefix="Fase 1")
                _render_status_message(_p1_banner.get("message"), _p1_banner.get("message_type"))
                _p1_parts = _p1.get("parts") or {}
                _df_p1_parts = pd.DataFrame(core_build_parts_rows(_p1_parts))
                if not _df_p1_parts.empty:
                    st.dataframe(
                        _df_contract_view(
                            _df_p1_parts,
                            ["Check", "Decision"],
                            context="Parâmetros > Governanca ML > Fase 1 (parts)",
                            fill_value="-",
                        ),
                        width='stretch',
                        hide_index=True,
                    )
                _p1_reasons = _p1.get("reasons") or []
                if isinstance(_p1_reasons, list) and _p1_reasons:
                    st.caption("Motivos: " + " | ".join([str(x) for x in _p1_reasons[:8]]))

        st.markdown("##### Phase 1.2 - Roster Continuity A/B")
        st.caption("Valida impacto objetivo da camada roster_continuity_v2 (off vs on) em logloss/brier.")
        if not _phase12_report_path.exists():
            st.caption("Relatório não encontrado. Rode: python tools\\eval_roster_continuity_phase12.py --speed fast")
        else:
            if not _p12:
                st.error(f"Falha ao ler {_phase12_report_path.name!s}")
            if isinstance(_p12, dict) and _p12:
                _p12_banner = core_decision_banner_parts(_p12.get("decision"), prefix="Fase 1.2")
                _render_status_message(_p12_banner.get("message"), _p12_banner.get("message_type"))
                _g_p12 = (_p12.get("global") or {}) if isinstance(_p12.get("global"), dict) else {}
                _k1, _k2, _k3, _k4 = st.columns(4)
                with _k1:
                    st.metric("Rows", value=str(int(_p12.get("rows_total") or 0)))
                with _k2:
                    st.metric("Applied rate", value=f"{_format_num(float(_p12.get('applied_rate', 0.0) or 0.0)*100.0,1)}%")
                with _k3:
                    st.metric("Gain logloss", value=_format_num(float(_g_p12.get("gain_logloss", float("nan"))), 6))
                with _k4:
                    st.metric("Gain brier", value=_format_num(float(_g_p12.get("gain_brier", float("nan"))), 6))
                _df_p12_lg = core_build_phase12_by_league_rows(_p12)
                if _df_p12_lg:
                    _dfp12 = pd.DataFrame(_df_p12_lg).sort_values(["Rows"], ascending=False).head(20)
                    st.dataframe(
                        _df_contract_view(
                            _dfp12,
                            ["Liga", "Rows", "Gain logloss", "LL off", "LL on"],
                            context="Parâmetros > Governanca ML > Fase 1.2 (ligas)",
                            fill_value="-",
                        ),
                        width='stretch',
                        hide_index=True,
                    )
                _p12_reasons = _p12.get("reasons") or []
                if isinstance(_p12_reasons, list) and _p12_reasons:
                    st.caption("Motivos: " + " | ".join([str(x) for x in _p12_reasons[:8]]))

        st.markdown("##### Promoção / Rollback")
        st.caption("Operação de produção com trilha de auditoria: promover baseline (com testes) ou rollback do último baseline.")
        _pr1, _pr2 = st.columns(2)
        with _pr1:
            _run_promote = st.button("Promover baseline agora", width='stretch', key="gov_promote_baseline")
        with _pr2:
            _run_rollback = st.button("Rollback baseline (mais recente)", width='stretch', key="gov_rollback_baseline")

        if _run_promote or _run_rollback:
            with st.spinner("Executando operação..."):
                _op_result = core_run_governance_operation(
                    promote=bool(_run_promote),
                    cwd=APP_ROOT,
                    tag="app_governance",
                    timeout_s=1200,
                )
                _merged_gov = core_append_governance_log(
                    core_get_governance_last_log(st.session_state),
                    _op_result.get("log_line"),
                    now_iso=datetime.now().isoformat(timespec="seconds"),
                )
                core_set_governance_last_log(st.session_state, str(_merged_gov.get("last_log") or ""))
                st.session_state["_gov_last_run"] = str(_merged_gov.get("last_run") or "")
                if bool(_op_result.get("ok")):
                    st.success("Operação concluída com sucesso.")
                else:
                    _err_txt = str(_op_result.get("error") or "").strip()
                    if _err_txt:
                        st.error(f"Falha ao executar operação: {_err_txt}")
                    else:
                        st.error("Operação falhou. Veja o log abaixo.")
