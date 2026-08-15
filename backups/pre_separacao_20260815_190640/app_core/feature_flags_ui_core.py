from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Iterable, Mapping

import pandas as pd
import streamlit as st

from app_core.ml_session_runtime_core import (
    get_feature_trace_state,
    set_feature_trace_events,
)


_FEATURE_FLAG_SPECS: tuple[dict[str, Any], ...] = (
    {
        "label": "Usar ML A-lite como calculo ativo (mapa/serie)",
        "key": "ff_ml_state_alite_live",
        "default": True,
        "help": "Promove o A-lite (estado dinamico + incerteza) para p_map/p_serie exibidos. O motor antigo continua no trace para comparacao.",
        "caption": "Pro: odds mais alinhadas a recencia/estado. Contra: muda o preco exibido (recomendado acompanhar no trace no inicio).",
    },
    {
        "label": "Usar ML A-real (state artifact) como calculo ativo (mapa/serie)",
        "key": "ff_ml_state_areal_live",
        "default": False,
        "help": "Promove o A-real (mu/sigma por time via artifact) para p_map/p_serie exibidos. Requer team_state_artifact.json gerado.",
        "caption": "Pro: estado persistente por time (mu/sigma) com drift por patch/roster. Contra: depende de artifact A-real atualizado.",
    },
    {
        "label": "Aplicar camada B (matchup/style) no ML ativo",
        "key": "ff_ml_style_b_lite_live",
        "default": True,
        "help": "Usa B-real (treinado) quando disponivel; se faltar fit, cai no B-lite. Aplica ajuste contextual por estilo sobre o ML ativo.",
        "caption": "Pro: captura encaixe de estilos sem depender so de ranking/recencia. Contra: exige params_fitted atualizado para usar B-real (com fallback automatico).",
    },
    {
        "label": "Aplicar camada Rank-SOS intra-liga no ML ativo",
        "key": "ff_ml_league_rank_sos_live",
        "default": False,
        "help": "Prior intra-liga ajustado por forca do calendario (SOS). Corrige campanhas enganosas dentro da mesma liga com peso conservador.",
        "caption": "Pro: melhora leitura de top/bottom e perdas para top1 da liga. Contra: so aplica em mesma liga e pode adicionar ruido se a amostra da liga for curta.",
    },
    {
        "label": "Aplicar camada Roster Continuity (identidade/lineup) no ML ativo",
        "key": "ff_ml_roster_continuity_live",
        "default": True,
        "help": "Reduz peso da identidade historica do time quando a continuidade do roster/lineup esta baixa. Usa sinais de lineup/freshness/coverage de forma conservadora.",
        "caption": "Pro: corrige comeco de split/ano e trocas fortes de roster. Contra: depende de lineup/players mapeados para ter mais efeito.",
    },
    {
        "label": "Aplicar calibracao hierarquica (liga/segmento) no ML ativo",
        "key": "ff_ml_hier_calibration_live",
        "default": True,
        "help": "Aplica um ajuste de calibracao leve por liga/segmento usando o baseline de confianca (com shrink por evidencia).",
        "caption": "Pro: melhora calibracao por liga/BO sem mexer brutalmente na odd. Contra: depende do baseline de confianca atualizado.",
    },
    {
        "label": "Aplicar camada SeriesBO (DP-lite) no ML ativo",
        "key": "ff_ml_series_bo_dplite_live",
        "default": True,
        "help": "Ajuste conservador de probabilidade de serie (BO3/BO5) em logit antes do series_gamma, usando contexto/evidencia. Nao altera BO1.",
        "caption": "Pro: melhora leitura de serie sem mexer no p_mapa. Contra: pode adicionar ruido se amostra/evidencia forem fracas.",
    },
    {
        "label": "Aplicar camada de incerteza por camada (LayerUnc) no ML ativo",
        "key": "ff_ml_layer_uncertainty_live",
        "default": True,
        "help": "Agrega incerteza das camadas A/B/Rank-SOS/Roster/HierCal/SeriesBO/C e aplica shrink conservador em logit para evitar exageros em baixa evidencia.",
        "caption": "Pro: reduz overconfidence e odds 'alucinadas' quando as camadas discordam. Contra: pode achatar demais se o peso estiver alto.",
    },
    {
        "label": "Aplicar camada C (meta-blend treinado; fallback C-lite) no ML ativo",
        "key": "ff_ml_meta_c_lite_live",
        "default": False,
        "help": "Usa meta-modelo treinado via walk-forward para combinar A/B/C com o ML base; se faltar fit, cai no C-lite.",
        "caption": "Pro: aprende quando confiar mais em ranking/forma/players. Contra: depende de params_fitted atualizado para usar o modo treinado.",
    },
    {
        "label": "Shadow trace dos paineis de Players",
        "key": "ff_players_panels_shadow_trace",
        "default": True,
        "help": "Registra eventos dos paineis de Players (filtros/contagens) em modo sombra. Nao altera odds.",
        "caption": "Pro: auditoria de uso e comparacao futura. Contra: mais eventos no trace (leve overhead).",
    },
    {
        "label": "Shadow trace A-lite (estado dinamico + incerteza)",
        "key": "ff_ml_shadow_state_alite_trace",
        "default": True,
        "help": "Calcula um shadow alternativo com forca estrutural + recencia + incerteza (Glicko-like simplificado), so para comparacao.",
        "caption": "Pro: testa arquitetura de estado dinamico sem risco. Contra: custo extra leve por analise.",
    },
    {
        "label": "Shadow trace A-real (state artifact mu/sigma)",
        "key": "ff_ml_shadow_state_areal_trace",
        "default": True,
        "help": "Calcula shadow via artifact de estado por time (mu/sigma persistidos) para comparacao com A-lite.",
        "caption": "Pro: valida o A-real em producao sem trocar a odd. Contra: custo leve extra e requer artifact A-real.",
    },
)


def _payload_get_num(row: Any, key: str) -> Any:
    if not isinstance(row, Mapping):
        return None
    payload = row.get("payload")
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _payload_get_str(row: Any, key: str) -> str:
    if not isinstance(row, Mapping):
        return ""
    payload = row.get("payload")
    if not isinstance(payload, Mapping):
        return ""
    value = payload.get(key)
    return str(value) if value is not None else ""


def normalize_feature_trace_rows(events: Any) -> list[dict[str, Any]]:
    rows = [row for row in (events or []) if isinstance(row, dict)] if isinstance(events, list) else []
    if not rows:
        return []
    df = pd.DataFrame(rows)
    for col in ("feature", "event", "generated_at"):
        if col not in df.columns:
            df[col] = "-"
    return df.to_dict(orient="records")


def build_feature_trace_download_payload(events: Any, *, generated_at: str | None = None) -> dict[str, Any]:
    return {
        "generated_at": str(generated_at or datetime.now().isoformat(timespec="seconds")),
        "events": normalize_feature_trace_rows(events),
    }


def build_feature_trace_summary_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = normalize_feature_trace_rows(list(rows))
    if not normalized:
        return []
    df = pd.DataFrame(normalized)
    summary_rows: list[dict[str, Any]] = []
    try:
        groups = df.groupby(["feature", "event"], dropna=False)
        for (feature, event), group in groups:
            last = group.iloc[-1].to_dict() if not group.empty else {}
            summary_rows.append(
                {
                    "Feature": str(feature),
                    "Evento": str(event),
                    "Method": _payload_get_str(last, "method"),
                    "Ocorrencias": len(group),
                    "Ultimo horario": str(last.get("generated_at", "")),
                    "Rows after": _payload_get_num(last, "rows_after"),
                    "Rows total": _payload_get_num(last, "rows_total"),
                    "Fair rows": _payload_get_num(last, "fair_rows_total"),
                    "Min H2H": _payload_get_num(last, "min_h2h_games"),
                }
            )
    except Exception:
        return []
    return summary_rows


def build_feature_trace_events_table(rows: Iterable[Mapping[str, Any]], *, show_payload: bool) -> pd.DataFrame:
    normalized = normalize_feature_trace_rows(list(rows))
    df = pd.DataFrame(normalized)
    if df.empty:
        return pd.DataFrame()
    events_show = pd.DataFrame(
        [
            {
                "Horario": str(row.get("generated_at", "")),
                "Feature": str(row.get("feature", "")),
                "Evento": str(row.get("event", "")),
                "Method": _payload_get_str(row, "method"),
                "Enabled": row.get("enabled"),
                "Dedupe": str(row.get("dedupe_signature", "")),
                "Matchup": _payload_get_str(row, "matchup"),
                "Rows after": _payload_get_num(row, "rows_after"),
                "Rows total": _payload_get_num(row, "rows_total"),
                "Fair rows": _payload_get_num(row, "fair_rows_total"),
                "Min H2H": _payload_get_num(row, "min_h2h_games"),
            }
            for row in df.to_dict(orient="records")
        ]
    )
    if show_payload:
        payload_series = (
            df["payload"].apply(lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else str(value))
            if "payload" in df.columns
            else ""
        )
        events_show["Payload"] = payload_series
    return events_show


def render_feature_flags_shadow_trace_panel(
    *,
    session_state: dict[str, Any],
    persist_combined_trace_file_fn: Callable[[], Any],
    df_contract_view_fn: Callable[..., pd.DataFrame],
) -> None:
    st.markdown("#### Avancado (Feature Flags / Shadow Trace)")
    st.caption("Controles de rollout/sombra e auditoria técnica. Não necessario para uso diario de odds.")

    with st.expander("Feature Flags (shadow / rollout)", expanded=False):
        st.caption("Ative/desative recursos em observacao sem alterar o motor principal. Use para rollout seguro.")
        ff1, ff2 = st.columns([1.4, 1.6])
        with ff1:
            for spec in _FEATURE_FLAG_SPECS:
                if str(spec["key"]) not in session_state:
                    session_state[str(spec["key"])] = bool(spec["default"])
                st.checkbox(
                    spec["label"],
                    key=str(spec["key"]),
                    help=str(spec["help"]),
                )
                st.caption(str(spec["caption"]))
        with ff2:
            feature_state = get_feature_trace_state(session_state)
            events_now = feature_state.get("events", [])
            event_count = len(events_now) if isinstance(events_now, list) else 0
            st.metric("Eventos shadow em sessao", int(event_count))
            fc1, fc2 = st.columns(2)
            with fc1:
                if st.button("Limpar shadow trace", key="ff_clear_shadow_trace", width="stretch"):
                    set_feature_trace_events(session_state, [], ok_at=str(datetime.now().isoformat(timespec="seconds")))
                    try:
                        persist_combined_trace_file_fn()
                    except Exception:
                        pass
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
            with fc2:
                try:
                    payload = build_feature_trace_download_payload(events_now)
                    st.download_button(
                        "Baixar shadow trace (JSON)",
                        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                        file_name="feature_trace_last.json",
                        mime="application/json",
                        key="ff_download_shadow_trace_json",
                        on_click="ignore",
                    )
                except Exception:
                    st.caption("Falha ao preparar download do shadow trace.")

    with st.expander("Shadow Trace / Diff (legivel)", expanded=False):
        st.caption("Visualizacao dos eventos de recursos em modo sombra (filtros, contagens e persistencia).")
        feature_state = get_feature_trace_state(session_state)
        rows = normalize_feature_trace_rows(feature_state.get("events", []))
        if not rows:
            st.caption("Sem eventos shadow na sessao atual.")
            return

        df_ff = pd.DataFrame(rows)
        vf1, vf2, vf3 = st.columns([1.2, 1.2, 1.0])
        with vf1:
            feature_options = ["Todos", *sorted([str(x) for x in df_ff["feature"].dropna().astype(str).unique().tolist() if str(x).strip()])]
            if str(session_state.get("ff_trace_filter_feature", "") or "") not in feature_options:
                session_state["ff_trace_filter_feature"] = "Todos"
            feature_selected = st.selectbox("Filtro feature", feature_options, key="ff_trace_filter_feature")
        with vf2:
            event_options = ["Todos", *sorted([str(x) for x in df_ff["event"].dropna().astype(str).unique().tolist() if str(x).strip()])]
            if str(session_state.get("ff_trace_filter_event", "") or "") not in event_options:
                session_state["ff_trace_filter_event"] = "Todos"
            event_selected = st.selectbox("Filtro evento", event_options, key="ff_trace_filter_event")
        with vf3:
            show_payload = st.checkbox("Mostrar payload bruto", key="ff_trace_show_payload")

        df_view = df_ff.copy()
        if feature_selected != "Todos":
            df_view = df_view[df_view["feature"].astype(str) == str(feature_selected)].copy()
        if event_selected != "Todos":
            df_view = df_view[df_view["event"].astype(str) == str(event_selected)].copy()

        summary_rows = build_feature_trace_summary_rows(df_view.to_dict(orient="records"))
        if summary_rows:
            st.markdown("##### Resumo por recurso/evento")
            df_summary = pd.DataFrame(summary_rows)
            st.dataframe(
                df_contract_view_fn(
                    df_summary,
                    ["Feature", "Evento", "Method", "Ocorrencias", "Ultimo horario", "Rows after", "Rows total", "Fair rows", "Min H2H"],
                    context="Parametros > Shadow Trace resumo",
                    fill_value="-",
                ),
                width="stretch",
                hide_index=True,
            )

        st.markdown("##### Eventos")
        df_events_show = build_feature_trace_events_table(df_view.to_dict(orient="records"), show_payload=bool(show_payload))
        event_cols = ["Horario", "Feature", "Evento", "Method", "Enabled", "Rows after", "Rows total", "Fair rows", "Min H2H", "Dedupe"]
        optional_cols = ["Payload"] if "Payload" in df_events_show.columns else []
        st.dataframe(
            df_contract_view_fn(
                df_events_show,
                event_cols,
                optional_cols=optional_cols,
                context="Parametros > Shadow Trace eventos",
                fill_value="-",
            ),
            width="stretch",
            hide_index=True,
        )
