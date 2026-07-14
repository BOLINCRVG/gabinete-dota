from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping

import pandas as pd

from app_core.championship_stats_core import apply_scope_filters, build_championship_scope_frame, scope_label_from_meta
from app_core.opponent_profile_core import (
    PROFILE_BUCKET_ORDER,
    available_market_labels,
    build_matchup_focus_table,
    build_opponent_reference_table,
    build_team_opponent_profile,
    phase_options_for_market,
    resolve_market_config,
)
from app_core.tab_ui_core import render_tab_section


def _line_options(values: list[float]) -> list[str]:
    out = [""]
    for value in values or []:
        out.append(f"{float(value):g}")
    return out


def _format_metric(metric_key: str, value: Any, *, min_to_mmss_fn: Callable[[Any], str], format_num_fn: Callable[[float, int], str]) -> str:
    try:
        fv = float(value)
    except Exception:
        return "-"
    if metric_key == "time":
        return str(min_to_mmss_fn(fv))
    return str(format_num_fn(float(fv), 1))


def _format_pct(value: Any, format_num_fn: Callable[[float, int], str]) -> str:
    try:
        fv = float(value)
    except Exception:
        return "-"
    return f"{format_num_fn(float(fv), 1)}%"


def _format_z(value: Any, format_num_fn: Callable[[float, int], str]) -> str:
    try:
        fv = float(value)
    except Exception:
        return "-"
    return format_num_fn(float(fv), 2)


def _table_view(
    df_in: pd.DataFrame,
    *,
    metric_key: str,
    min_to_mmss_fn: Callable[[Any], str],
    format_num_fn: Callable[[float, int], str],
) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame([])
    out = df_in.copy()
    if "Confiança" in out.columns:
        out["Confiança"] = out["Confiança"].map(lambda value: _format_pct(value, format_num_fn))
    if "Score Adv." in out.columns:
        out["Score Adv."] = out["Score Adv."].map(
            lambda value: _format_metric(metric_key, value, min_to_mmss_fn=min_to_mmss_fn, format_num_fn=format_num_fn)
        )
    if "Z Adv." in out.columns:
        out["Z Adv."] = out["Z Adv."].map(lambda value: format_num_fn(float(value), 2) if pd.notna(value) else "-")
    if "Média" in out.columns:
        out["Média"] = out["Média"].map(lambda value: _format_metric(metric_key, value, min_to_mmss_fn=min_to_mmss_fn, format_num_fn=format_num_fn))
    if "Delta" in out.columns:
        out["Delta"] = out["Delta"].map(lambda value: _format_metric(metric_key, value, min_to_mmss_fn=min_to_mmss_fn, format_num_fn=format_num_fn))
    if "Delta vs Liga" in out.columns:
        out["Delta vs Liga"] = out["Delta vs Liga"].map(
            lambda value: _format_metric(metric_key, value, min_to_mmss_fn=min_to_mmss_fn, format_num_fn=format_num_fn)
        )
    if "% Over" in out.columns:
        out["% Over"] = out["% Over"].map(lambda value: _format_pct(value, format_num_fn))
    if "% Under" in out.columns:
        out["% Under"] = out["% Under"].map(lambda value: _format_pct(value, format_num_fn))
    preferred = [
        "Atual",
        "Perfil do Oponente",
        "Jogos no bucket",
        "Jogos do time",
        "Amostra",
        "Risco",
        "Confiança",
        "Score Adv.",
        "Z Adv.",
        "Média",
        "Delta",
        "Delta vs Liga",
        "% Over",
        "% Under",
        "Leitura",
        "Exemplos",
    ]
    cols = [col for col in preferred if col in out.columns] + [col for col in out.columns if col not in preferred]
    return out[cols]


def _has_meaningful_rows(df_in: pd.DataFrame | None) -> bool:
    if df_in is None or df_in.empty:
        return False
    return not df_in.dropna(axis=0, how="all").empty


def render_opponent_profile_tab(
    *,
    st: Any,
    session_state: MutableMapping[str, Any],
    team_games: pd.DataFrame | None,
    filters_cls: Any,
    active_filters: Any,
    apply_filters_fn: Callable[[pd.DataFrame, Any], pd.DataFrame],
    ml_info: Mapping[str, Any] | None,
    map_mode_ui: Any,
    team_a: str,
    team_b: str,
    parse_lines_fn: Callable[[str], list[float]],
    parse_time_lines_fn: Callable[[str], list[float]],
    min_to_mmss_fn: Callable[[Any], str],
    format_num_fn: Callable[[float, int], str],
    current_lines_by_metric: Mapping[str, list[float]] | None = None,
) -> None:
    render_tab_section(
        st,
        title="Perfil de Oponente",
        subtitle="Como cada time reage a adversários under, neutros e over no mercado selecionado.",
    )

    df_scope, scope_meta = build_championship_scope_frame(
        session_state=session_state,
        team_games=team_games,
        filters_cls=filters_cls,
        active_filters=active_filters,
        apply_filters_fn=apply_filters_fn,
        ml_info=ml_info,
        map_mode_ui=map_mode_ui,
    )
    if df_scope is None or df_scope.empty:
        st.info("Sem jogos no recorte atual para montar o perfil de oponente.")
        return

    st.caption(f"Escopo: {scope_label_from_meta(scope_meta)}")

    market_labels = available_market_labels()
    current_market = str(session_state.get("opp_profile_market", "Kills") or "Kills")
    if current_market not in market_labels:
        current_market = market_labels[0]
    session_state["opp_profile_market"] = current_market
    market_label = st.selectbox("Mercado", market_labels, key="opp_profile_market")
    market_cfg = resolve_market_config(market_label)
    metric_key = str(market_cfg.get("metric_key", "kills"))

    current_lines = list((current_lines_by_metric or {}).get(metric_key, []) or [])
    line_opts = _line_options(current_lines)
    current_line = str(session_state.get("opp_profile_current_line", "") or "")
    if current_line not in line_opts:
        current_line = line_opts[0]
        session_state["opp_profile_current_line"] = current_line

    c1, c2, c3, c4 = st.columns([1.1, 1.0, 1.0, 1.0])
    with c1:
        selected_current = st.selectbox("Linha do confronto", line_opts, key="opp_profile_current_line")
    with c2:
        custom_line_text = st.text_input("Linha custom", key="opp_profile_custom_line", placeholder="ex: 35.5")
    with c3:
        side_options = ["All", "Blue", "Red"]
        side_value = str(session_state.get("opp_profile_side_filter", "All") or "All")
        if side_value not in side_options:
            side_value = "All"
        session_state["opp_profile_side_filter"] = side_value
        side_filter = st.selectbox("Side", side_options, key="opp_profile_side_filter")
    with c4:
        wl_options = ["All", "Wins", "Losses"]
        wl_value = str(session_state.get("opp_profile_wl_filter", "All") or "All")
        if wl_value not in wl_options:
            wl_value = "All"
        session_state["opp_profile_wl_filter"] = wl_value
        wl_filter = st.selectbox("Win/loss", wl_options, key="opp_profile_wl_filter")

    parser = parse_time_lines_fn if metric_key == "time" else parse_lines_fn
    custom_values = list(parser(str(custom_line_text or "")))
    active_line = float(custom_values[0]) if custom_values else None
    if active_line is None and str(selected_current).strip():
        try:
            active_line = float(selected_current)
        except Exception:
            active_line = None

    df_filtered = apply_scope_filters(df_scope, side_filter=str(side_filter), win_loss_filter=str(wl_filter))
    if df_filtered is None or df_filtered.empty:
        st.info("Sem jogos nesse filtro de Side/Win-loss.")
        return

    reference_table = build_opponent_reference_table(df_filtered, market_label, phase_label="Final")
    if reference_table.empty:
        st.info("Sem base suficiente para classificar os adversários no mercado selecionado.")
        return

    payload_a = build_team_opponent_profile(
        df_filtered,
        team=str(team_a),
        opponent_team=str(team_b),
        market_label=str(market_label),
        line_val=active_line,
        reference_table=reference_table,
        phase_label="Final",
    )
    payload_b = build_team_opponent_profile(
        df_filtered,
        team=str(team_b),
        opponent_team=str(team_a),
        market_label=str(market_label),
        line_val=active_line,
        reference_table=reference_table,
        phase_label="Final",
    )

    if active_line is not None:
        st.caption(
            f"Linha ativa para {market_label}: {float(active_line):g}. "
            "Os buckets usam a média do mercado no recorte atual; os percentuais usam a linha ativa."
        )
    else:
        st.caption(
            "Buckets usam a média do mercado no recorte atual. "
            "Se quiser % Over/% Under, escolha a linha do confronto ou uma linha custom."
        )
    st.caption("`Jogos no bucket` não é o total do time: é só a amostra contra adversários do mesmo perfil do oponente atual.")
    st.caption("`Leitura` compara a média do bucket contra o baseline do próprio time no recorte. `Delta vs Liga` compara o bucket contra a média geral do campeonato/recorte.")

    focus_table = build_matchup_focus_table(payload_a, payload_b)
    st.markdown("##### Leitura do confronto atual")
    st.dataframe(
        _table_view(
            focus_table,
            metric_key=metric_key,
            min_to_mmss_fn=min_to_mmss_fn,
            format_num_fn=format_num_fn,
        ),
        width="stretch",
        hide_index=True,
    )

    phase_options = phase_options_for_market(df_filtered, market_label)
    if len(phase_options) > 1:
        phase_rows: list[pd.DataFrame] = []
        for phase_label in phase_options:
            ref_phase = build_opponent_reference_table(df_filtered, market_label, phase_label=phase_label)
            if ref_phase.empty:
                continue
            phase_payload_a = build_team_opponent_profile(
                df_filtered,
                team=str(team_a),
                opponent_team=str(team_b),
                market_label=str(market_label),
                line_val=active_line,
                reference_table=ref_phase,
                phase_label=phase_label,
            )
            phase_payload_b = build_team_opponent_profile(
                df_filtered,
                team=str(team_b),
                opponent_team=str(team_a),
                market_label=str(market_label),
                line_val=active_line,
                reference_table=ref_phase,
                phase_label=phase_label,
            )
            phase_frame = build_matchup_focus_table(phase_payload_a, phase_payload_b)
            if _has_meaningful_rows(phase_frame):
                phase_rows.append(phase_frame)
        if phase_rows:
            st.markdown("##### Mudança por fase")
            st.caption("Para Kills, a leitura separa o bucket do adversário atual em Final, 0-15, 15-20 e 20+.")
            st.dataframe(
                _table_view(
                    pd.concat(phase_rows, ignore_index=True),
                    metric_key=metric_key,
                    min_to_mmss_fn=min_to_mmss_fn,
                    format_num_fn=format_num_fn,
                ),
                width="stretch",
                hide_index=True,
            )

    col_a, col_b = st.columns(2)
    rows_a = payload_a.get("rows")
    rows_b = payload_b.get("rows")
    if not isinstance(rows_a, pd.DataFrame):
        rows_a = pd.DataFrame([])
    if not isinstance(rows_b, pd.DataFrame):
        rows_b = pd.DataFrame([])

    with col_a:
        st.markdown(f"##### {team_a}")
        st.caption(
            f"Perfil próprio: {payload_a.get('team_bucket', '-')}"
            f" | Adversário atual: {payload_a.get('opponent_bucket', '-')}"
            f" | Jogos totais: {int(payload_a.get('team_games_total') or 0)}"
        )
        st.caption(
            f"Score do adversário atual: {_format_metric(metric_key, payload_a.get('opponent_metric_mean'), min_to_mmss_fn=min_to_mmss_fn, format_num_fn=format_num_fn)}"
            f" | Z-score: {_format_z(payload_a.get('opponent_metric_z'), format_num_fn)}"
        )
        st.dataframe(
            _table_view(
                rows_a,
                metric_key=metric_key,
                min_to_mmss_fn=min_to_mmss_fn,
                format_num_fn=format_num_fn,
            ),
            width="stretch",
            hide_index=True,
        )
        bucket_a_default = str(payload_a.get("opponent_bucket") or PROFILE_BUCKET_ORDER[0])
        bucket_a_options = [bucket for bucket in PROFILE_BUCKET_ORDER if bucket in set(rows_a.get("Perfil do Oponente", pd.Series(dtype=str)).astype(str).tolist())]
        if not bucket_a_options:
            bucket_a_options = list(PROFILE_BUCKET_ORDER)
        bucket_a_index = bucket_a_options.index(bucket_a_default) if bucket_a_default in bucket_a_options else 0
        if str(session_state.get("opp_profile_bucket_pick_a", "") or "") not in bucket_a_options:
            session_state["opp_profile_bucket_pick_a"] = bucket_a_options[bucket_a_index]
        bucket_pick_a = st.selectbox("Auditar bucket", bucket_a_options, key="opp_profile_bucket_pick_a")
        selected_row_a = rows_a[rows_a["Perfil do Oponente"].astype(str) == str(bucket_pick_a)].copy() if not rows_a.empty else pd.DataFrame([])
        if not selected_row_a.empty:
            st.dataframe(
                _table_view(
                    selected_row_a,
                    metric_key=metric_key,
                    min_to_mmss_fn=min_to_mmss_fn,
                    format_num_fn=format_num_fn,
                ),
                width="stretch",
                hide_index=True,
            )
        with st.expander(f"Jogos do bucket selecionado - {team_a}", expanded=False):
            bucket_games_a = payload_a.get("bucket_games") or {}
            focus_games_a = bucket_games_a.get(str(bucket_pick_a)) if isinstance(bucket_games_a, dict) else payload_a.get("focus_games")
            if isinstance(focus_games_a, pd.DataFrame) and not focus_games_a.empty:
                games_view_a = focus_games_a.copy()
                if "Valor" in games_view_a.columns:
                    games_view_a["Valor"] = games_view_a["Valor"].map(
                        lambda value: _format_metric(
                            metric_key,
                            value,
                            min_to_mmss_fn=min_to_mmss_fn,
                            format_num_fn=format_num_fn,
                        )
                    )
                st.dataframe(games_view_a, width="stretch", hide_index=True)
            else:
                st.info("Sem jogos auditáveis nesse bucket.")
    with col_b:
        st.markdown(f"##### {team_b}")
        st.caption(
            f"Perfil próprio: {payload_b.get('team_bucket', '-')}"
            f" | Adversário atual: {payload_b.get('opponent_bucket', '-')}"
            f" | Jogos totais: {int(payload_b.get('team_games_total') or 0)}"
        )
        st.caption(
            f"Score do adversário atual: {_format_metric(metric_key, payload_b.get('opponent_metric_mean'), min_to_mmss_fn=min_to_mmss_fn, format_num_fn=format_num_fn)}"
            f" | Z-score: {_format_z(payload_b.get('opponent_metric_z'), format_num_fn)}"
        )
        st.dataframe(
            _table_view(
                rows_b,
                metric_key=metric_key,
                min_to_mmss_fn=min_to_mmss_fn,
                format_num_fn=format_num_fn,
            ),
            width="stretch",
            hide_index=True,
        )
        bucket_b_default = str(payload_b.get("opponent_bucket") or PROFILE_BUCKET_ORDER[0])
        bucket_b_options = [bucket for bucket in PROFILE_BUCKET_ORDER if bucket in set(rows_b.get("Perfil do Oponente", pd.Series(dtype=str)).astype(str).tolist())]
        if not bucket_b_options:
            bucket_b_options = list(PROFILE_BUCKET_ORDER)
        bucket_b_index = bucket_b_options.index(bucket_b_default) if bucket_b_default in bucket_b_options else 0
        if str(session_state.get("opp_profile_bucket_pick_b", "") or "") not in bucket_b_options:
            session_state["opp_profile_bucket_pick_b"] = bucket_b_options[bucket_b_index]
        bucket_pick_b = st.selectbox("Auditar bucket", bucket_b_options, key="opp_profile_bucket_pick_b")
        selected_row_b = rows_b[rows_b["Perfil do Oponente"].astype(str) == str(bucket_pick_b)].copy() if not rows_b.empty else pd.DataFrame([])
        if not selected_row_b.empty:
            st.dataframe(
                _table_view(
                    selected_row_b,
                    metric_key=metric_key,
                    min_to_mmss_fn=min_to_mmss_fn,
                    format_num_fn=format_num_fn,
                ),
                width="stretch",
                hide_index=True,
            )
        with st.expander(f"Jogos do bucket selecionado - {team_b}", expanded=False):
            bucket_games_b = payload_b.get("bucket_games") or {}
            focus_games_b = bucket_games_b.get(str(bucket_pick_b)) if isinstance(bucket_games_b, dict) else payload_b.get("focus_games")
            if isinstance(focus_games_b, pd.DataFrame) and not focus_games_b.empty:
                games_view_b = focus_games_b.copy()
                if "Valor" in games_view_b.columns:
                    games_view_b["Valor"] = games_view_b["Valor"].map(
                        lambda value: _format_metric(
                            metric_key,
                            value,
                            min_to_mmss_fn=min_to_mmss_fn,
                            format_num_fn=format_num_fn,
                        )
                    )
                st.dataframe(games_view_b, width="stretch", hide_index=True)
            else:
                st.info("Sem jogos auditáveis nesse bucket.")

    with st.expander("Referência dos times no recorte", expanded=False):
        ref_view = reference_table.copy()
        ref_view["Metric Mean"] = ref_view["Metric Mean"].map(
            lambda value: _format_metric(metric_key, value, min_to_mmss_fn=min_to_mmss_fn, format_num_fn=format_num_fn)
        )
        ref_view["Metric Z"] = ref_view["Metric Z"].map(lambda value: format_num_fn(float(value), 2) if pd.notna(value) else "-")
        st.dataframe(ref_view, width="stretch", hide_index=True, height=420)


__all__ = ["render_opponent_profile_tab"]
