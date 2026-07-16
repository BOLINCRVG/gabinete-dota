"""Players tab renderer."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from app_core.tab_filters_core import collect_year_split_league_options, scope_has_playoffs
from app_core.tab_state_core import blank_space_label
from app_core.tab_ui_core import render_tab_hero, render_tab_section
from app_core.ui_design_core import TONE_HTML_COLOR, cyan_section_header, team_badge_html, tone_for_pct
from mlcore.team_artifact_normalize import _norm_team_key
from mlcore.utils import sigmoid as _sigmoid


def _team_rows(dfb: pd.DataFrame, team: str) -> pd.DataFrame:
    """Linhas do time com match exato e fallback normalizado (grafias divergem
    entre o CSV de team-games e o de players)."""
    if dfb is None or dfb.empty or "team" not in dfb.columns:
        return dfb.iloc[0:0] if dfb is not None else pd.DataFrame()
    sub = dfb[dfb["team"].astype(str) == str(team)]
    if sub.empty:
        key = _norm_team_key(team)
        if key:
            sub = dfb[dfb["team"].apply(lambda x: _norm_team_key(x) == key)]
    return sub


def _subhead(st: Any, title: str) -> None:
    """Lightweight cyan-accented sub-header (consistent look, lighter than the
    full gradient bar) for the data sections inside the Players tab."""
    st.markdown(
        f"<div style='margin:14px 0 4px;padding-left:9px;border-left:3px solid #2daac4;"
        f"font-weight:700;font-size:1.02em;color:#2daac4;'>{title}</div>",
        unsafe_allow_html=True,
    )


def _render_players_result_card(
    st: Any, *, team_a: str, team_b: str, odd_a: float, odd_b: float, p_team_a: float,
    p_players_raw: float, p_team_ref: float, w_players: float,
    n_team_eff: float, n_players_eff: float, format_num_fn: Any, format_pct_fn: Any,
) -> None:
    """Headline output: two odd cards colored by win probability + a blend breakdown.

    Replaces the old plain-markdown line + two raw dataframes so the main result
    reads at a glance instead of as crude text.
    """
    pa = float(max(0.0, min(1.0, p_team_a))) * 100.0
    pb = 100.0 - pa
    col_a = TONE_HTML_COLOR.get({"elite": "good", "green": "good", "yellow": "warn", "red": "bad", "neutral": "neutral"}[tone_for_pct(pa)], "#fff")
    col_b = TONE_HTML_COLOR.get({"elite": "good", "green": "good", "yellow": "warn", "red": "bad", "neutral": "neutral"}[tone_for_pct(pb)], "#fff")

    def _odd_card(team: str, odd: float, pct: float, color: str) -> str:
        return (
            "<div style='flex:1;border:1px solid rgba(45,170,196,0.35);border-radius:10px;"
            "padding:12px 14px;background:rgba(45,170,196,0.06);text-align:center;'>"
            f"<div style='display:flex;align-items:center;justify-content:center;gap:8px;margin-bottom:6px;'>"
            f"{team_badge_html(team, size=34)}"
            f"<span style='font-weight:700;font-size:1.02em;'>{team}</span></div>"
            f"<div style='font-size:2.0em;font-weight:800;color:{color};line-height:1;'>{format_num_fn(odd, 2)}</div>"
            f"<div style='opacity:0.7;font-size:0.85em;margin-top:4px;'>{pct:.0f}% de vitória</div>"
            "</div>"
        )

    cyan_section_header(st, "Resultado — ML por players")
    st.markdown(
        "<div style='display:flex;gap:12px;margin:8px 0 6px;'>"
        + _odd_card(str(team_a), odd_a, pa, col_a)
        + _odd_card(str(team_b), odd_b, pb, col_b)
        + "</div>",
        unsafe_allow_html=True,
    )

    def _chip(label: str, value: str) -> str:
        return (
            "<span style='display:inline-block;background:rgba(45,170,196,0.10);border-radius:6px;"
            "padding:3px 9px;margin:2px 4px 2px 0;font-size:0.86em;'>"
            f"<span style='opacity:0.6;'>{label}</span> <b>{value}</b></span>"
        )

    st.markdown(
        "<div style='margin:2px 0 4px;'>"
        + _chip("Players", format_pct_fn(p_players_raw))
        + _chip("Time (ML)", format_pct_fn(p_team_ref))
        + _chip("Blend", format_pct_fn(p_team_a))
        + _chip("Peso players", f"{format_num_fn(100.0 * w_players, 0)}%")
        + _chip("n_time", format_num_fn(n_team_eff, 1))
        + _chip("n_players", format_num_fn(n_players_eff, 1))
        + "</div>",
        unsafe_allow_html=True,
    )


def _team_rating(
    *,
    by_player: pd.DataFrame,
    players: list[str],
    format_num_fn,
) -> tuple[float, pd.DataFrame]:
    if not players:
        return 0.0, pd.DataFrame([])
    sub = by_player[by_player["player"].isin(players)].copy()
    if sub.empty:
        return 0.0, pd.DataFrame([])
    sub["w"] = sub["games"].clip(upper=20).astype(float)
    rating = float((sub["rating"] * sub["w"]).sum() / max(sub["w"].sum(), 1.0))
    view = sub[["player", "games", "wins", "p_smooth", "rating"]].copy()
    view["Win Rate"] = (100.0 * (view["wins"] / view["games"])).apply(lambda x: format_num_fn(x, 1))
    view["p_smooth"] = (100.0 * view["p_smooth"]).apply(lambda x: format_num_fn(x, 1))
    view["rating"] = view["rating"].apply(lambda x: format_num_fn(x, 3))
    view = view.rename(
        columns={"player": "Player", "games": "Games", "wins": "Wins", "p_smooth": "Smooth Win%", "rating": "Rating"}
    )
    return rating, view


def _build_trace_rows(out_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trace_rows = []
    for row in out_rows:
        trace_rows.append(
            {
                "lane": str(row.get("Lane", "")),
                "team": str(row.get("Team", "")),
                "player": str(row.get("Player", "")),
                "mercado": str(row.get("Mercado", "")),
                "lado": str(row.get("Lado", "")),
                "linha": str(row.get("Linha", "")),
                "odd_model": row.get("_odd_model_raw"),
                "odd_hist": row.get("_odd_hist_raw"),
                "odd_h2h": row.get("_odd_h2h_raw"),
                "odd_mix": row.get("_odd_mix_raw"),
                "calib_reason": str(row.get("_calib_reason", "")),
                "calib_source": str(row.get("_calib_source", "")),
                "h2h_n": row.get("_h2h_n"),
                "h2h_used": row.get("_h2h_used"),
                "obs_odds": str(row.get("Obs Odds", "")),
                "obs_mix": str(row.get("Obs Mix", "")),
            }
        )
    return trace_rows


def _render_lineup_value_block(
    *,
    st: Any,
    team_a: str,
    team_b: str,
    csv_sig: Any,
    load_player_rows_fn: Any,
    latest_lineup_names_fn: Any,
    norm_pos_fn: Any,
    fuzzy_match_one_fn: Any,
) -> None:
    """Manchete "quem joga hoje vale quanto" (redesign 2026-07-16): lineup
    habitual dos dois times com percentil de desempenho POR FUNÇÃO —
    independente de vitória (lição PandaSkill). Descritivo, n visível."""
    ta = str(team_a or "").strip()
    tb = str(team_b or "").strip()
    if not ta or not tb:
        return
    try:
        from app_core.players_value_core import (
            build_player_role_scores,
            lineup_value_view,
        )

        rows = load_player_rows_fn(csv_sig)
        if rows is None or rows.empty:
            return
        year = None
        try:
            yr = pd.to_numeric(rows.get("year"), errors="coerce")
            year = int(yr.max()) if yr.notna().any() else None
        except Exception:
            year = None
        scores = build_player_role_scores(rows, year=year)
        if scores.empty:
            return
        rows_year = rows
        if year is not None and "year" in rows.columns:
            rows_year = rows[pd.to_numeric(rows["year"], errors="coerce") == year]

        with st.container(border=True):
            render_tab_section(
                st,
                title=f"Quem joga hoje — {ta} × {tb}",
                subtitle=(
                    f"Lineup habitual + percentil de desempenho dentro da FUNÇÃO "
                    f"({'temporada ' + str(year) if year else 'CSV inteiro'}; KLA, dano/min, CS/min, visão/min; "
                    "independe de vitória — jogador bom em time ruim aparece)."
                ),
            )
            c1, c2 = st.columns(2)
            for col, tm in ((c1, ta), (c2, tb)):
                with col:
                    st.markdown(f"**{tm}**")
                    try:
                        lineup = latest_lineup_names_fn(df_in=rows_year, team_name=tm, norm_pos_fn=norm_pos_fn) or []
                    except Exception:
                        lineup = []
                    lineup = [x for x in lineup if str(x).strip()]
                    if not lineup:
                        st.caption("Sem lineup recente no recorte.")
                        continue
                    v = lineup_value_view(lineup, scores, fuzzy_match_one_fn=fuzzy_match_one_fn)
                    view = pd.DataFrame({
                        "Função": v["pos"].replace("", "?"),
                        "Player": v["player"],
                        "Percentil": v["percentile"].map(
                            lambda x: f"P{x:.0f}" if pd.notna(x) else "sem amostra"
                        ),
                        "J": v["games"].map(lambda n: int(n) if n else "-"),
                    })
                    st.dataframe(view, hide_index=True, use_container_width=True)
                    med = v["percentile"].dropna()
                    if not med.empty:
                        st.caption(f"Mediana do lineup: **P{med.median():.0f}** ({len(med)}/5 com amostra)")
    except Exception:
        # bloco novo NUNCA derruba a aba
        return


def render_players_tab(*, st: Any, ctx: Mapping[str, Any]) -> None:
    session_state = st.session_state
    render_tab_hero(st, title="Players", subtitle="Quem joga hoje vale quanto — e o ML por lineup com fusão Team + Players.")
    _render_lineup_value_block(
        st=st,
        team_a=ctx["teamA"],
        team_b=ctx["teamB"],
        csv_sig=ctx["csv_sig"],
        load_player_rows_fn=ctx["load_player_rows_fn"],
        latest_lineup_names_fn=ctx["latest_lineup_names_fn"],
        norm_pos_fn=ctx["norm_pos_fn"],
        fuzzy_match_one_fn=ctx["fuzzy_match_one_fn"],
    )

    team_games = ctx["team_games"]
    Filters = ctx["Filters"]
    csv_sig = ctx["csv_sig"]
    team_a = ctx["teamA"]
    team_b = ctx["teamB"]
    load_player_rows_fn = ctx["load_player_rows_fn"]
    apply_filters_players_fn = ctx["apply_filters_players_fn"]
    latest_lineup_names_fn = ctx["latest_lineup_names_fn"]
    norm_pos_fn = ctx["norm_pos_fn"]
    parse_players_text_fn = ctx["parse_players_text_fn"]
    resolve_players_fn = ctx["resolve_players_fn"]
    fuzzy_match_one_fn = ctx["fuzzy_match_one_fn"]
    build_lane_rows_fn = ctx["build_lane_rows_fn"]
    ensure_players_filter_state_fn = ctx["ensure_players_filter_state_fn"]
    ensure_players_text_state_fn = ctx["ensure_players_text_state_fn"]
    apply_players_autofill_state_fn = ctx["apply_players_autofill_state_fn"]
    ensure_player_line_input_state_fn = ctx["ensure_player_line_input_state_fn"]
    apply_players_submission_state_fn = ctx["apply_players_submission_state_fn"]
    build_per_player_lines_by_idx_fn = ctx["build_per_player_lines_by_idx_fn"]
    build_player_means_rows_fn = ctx["build_player_means_rows_fn"]
    build_recent_player_stats_rows_fn = ctx["build_recent_player_stats_rows_fn"]
    build_player_market_rows_fn = ctx["build_player_market_rows_fn"]
    append_players_recent_stats_panel_trace_fn = ctx["append_players_recent_stats_panel_trace_fn"]
    append_players_fair_panel_trace_fn = ctx["append_players_fair_panel_trace_fn"]
    append_players_markets_panel_trace_fn = ctx["append_players_markets_panel_trace_fn"]
    append_players_panels_bundle_trace_fn = ctx["append_players_panels_bundle_trace_fn"]
    append_feature_trace_fn = ctx["append_feature_trace_fn"]
    feature_flag_enabled_fn = ctx["feature_flag_enabled_fn"]
    df_contract_view_fn = ctx["df_contract_view_fn"]
    format_num_fn = ctx["format_num_fn"]
    format_pct_fn = ctx["format_pct_fn"]
    format_pct1_fn = ctx["format_pct1_fn"]
    odd_from_prob_fn = ctx["odd_from_prob_fn"]
    state_get_float_fn = ctx["state_get_float_fn"]
    state_get_int_fn = ctx["state_get_int_fn"]
    parse_lines_fn = ctx["parse_lines_fn"]
    fuse_probs_by_precision_fn = ctx["fuse_probs_by_precision_fn"]
    persist_combined_trace_file_fn = ctx["persist_combined_trace_file_fn"]
    playoffs_opt = ctx["playoffs_opt"]

    years, splits, leagues = collect_year_split_league_options(team_games)
    ensure_players_filter_state_fn(session_state)

    def _ensure_choice(key: str, options: list[str], fallback: str | None = None) -> str:
        default = str(fallback if fallback is not None else (options[0] if options else ""))
        current = str(session_state.get(key, default) or default)
        allowed = {str(x) for x in options}
        if current not in allowed:
            current = default
            session_state[key] = current
        return current

    def _on_pl_league_change() -> None:
        session_state["pl_split"] = ""

    def _splits_for_league(league_sel: str) -> list[str]:
        """Splits available for the selected league (fallback: all splits)."""
        try:
            if (league_sel and team_games is not None and not team_games.empty
                    and "league" in team_games.columns and "split" in team_games.columns):
                return sorted(
                    str(x)
                    for x in team_games.loc[team_games["league"].astype(str) == str(league_sel), "split"]
                    .fillna("").astype(str).unique().tolist()
                    if str(x)
                )
        except Exception:
            pass
        return splits

    # Single clean filter row: Ano | Liga | Split | Playoffs.
    with st.container(border=True):
        render_tab_section(st, title="Filtros", subtitle="Feche o recorte antes de abrir lineup, props e medias por jogador.")
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            year_options = [""] + [str(y) for y in years]
            _ensure_choice("pl_year", year_options, "")
            p_year = st.selectbox("Ano", options=year_options, key="pl_year", format_func=blank_space_label)
        with pc2:
            league_options = ["", *leagues]
            _ensure_choice("pl_league", league_options, "")
            p_league = st.selectbox(
                "Liga",
                options=league_options,
                key="pl_league",
                format_func=blank_space_label,
                on_change=_on_pl_league_change,
            )
        # Splits depend on the league just picked; compute before rendering Split.
        splits_pl = _splits_for_league(p_league)
        with pc3:
            split_options = ["", *splits_pl]
            _ensure_choice("pl_split", split_options, "")
            p_split = st.selectbox("Split", options=split_options, key="pl_split", format_func=blank_space_label)
        # Playoffs box appears ONLY when the chosen (liga, ano, split) actually
        # has playoff games — e.g. pick Split 2 and, once playoffs start, the
        # box shows up. Otherwise it stays hidden and no playoff filter applies.
        _has_po = scope_has_playoffs(team_games, league=p_league, year=p_year, split=p_split)
        if _has_po:
            with pc4:
                playoffs_options = ["", "Regular", "Playoffs"]
                _ensure_choice("pl_playoffs", playoffs_options, "")
                p_playoffs = st.selectbox(
                    "Playoffs",
                    options=playoffs_options,
                    key="pl_playoffs",
                    format_func=blank_space_label,
                )
            p_playoffs = st.session_state.get("pl_playoffs", "") or ""
        else:
            p_playoffs = ""

    base_filters = Filters(
        year=None if not p_year else int(p_year),
        league=None if not p_league else p_league,
        split=None if not p_split else p_split,
        playoffs=None if not p_playoffs else (True if p_playoffs == "Playoffs" else False),
    )

    with st.spinner("Carregando players do CSV (1x por sessão / cache)..."):
        df_pl = load_player_rows_fn(csv_sig)

    if df_pl is None or df_pl.empty:
        st.warning("Não achei rows de players nesse CSV.")
        dfb = pd.DataFrame()
    else:
        dfb = apply_filters_players_fn(df_pl, base_filters).copy()

    if dfb.empty:
        st.info("Sem jogos nesse recorte.")
        return

    group_df = dfb.copy()
    group_df["win"] = (pd.to_numeric(group_df.get("result"), errors="coerce").fillna(0).astype(int) > 0).astype(int)
    by_player = group_df.groupby("player").agg(games=("win", "size"), wins=("win", "sum")).reset_index()
    by_player["p_smooth"] = (by_player["wins"] + 2.0) / (by_player["games"] + 4.0)
    by_player["rating"] = np.log(by_player["p_smooth"] / (1.0 - by_player["p_smooth"]))

    vocab = sorted([x for x in by_player["player"].fillna("").astype(str).unique().tolist() if x])
    vocab_a = sorted([x for x in _team_rows(dfb, team_a)["player"].fillna("").astype(str).unique().tolist() if x])
    vocab_b = sorted([x for x in _team_rows(dfb, team_b)["player"].fillna("").astype(str).unique().tolist() if x])
    if not vocab_a:
        vocab_a = vocab
    if not vocab_b:
        vocab_b = vocab

    render_tab_section(st, title="Cole os players", subtitle="Cole os nomes por time ou use o autofill do lineup mais recente.")
    st.caption("Um por linha. O matching é fuzzy (se errar um pouco o nome, ele sugere).")
    a0, a1 = st.columns([1.4, 2.6])
    with a0:
        if st.button("Auto-preencher players dos times", key="pl_autofill_players"):
            auto_a = latest_lineup_names_fn(df_in=dfb, team_name=team_a, norm_pos_fn=norm_pos_fn)
            auto_b = latest_lineup_names_fn(df_in=dfb, team_name=team_b, norm_pos_fn=norm_pos_fn)
            apply_players_autofill_state_fn(session_state, team_a_players=auto_a, team_b_players=auto_b)
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()
    with a1:
        st.caption("Preenche Top/JG/Mid/ADC/Sup com o lineup mais recente encontrado no recorte atual.")

    ensure_players_text_state_fn(session_state)
    players_a = parse_players_text_fn(session_state.get("pl_txtA_applied", ""))
    players_b = parse_players_text_fn(session_state.get("pl_txtB_applied", ""))
    resolved_a, notes_a = resolve_players_fn(
        names=players_a,
        vocab=vocab_a,
        fuzzy_match_fn=lambda nm, vv, c: fuzzy_match_one_fn(nm, vv, cutoff=c),
        cutoff=0.6,
    )
    resolved_b, notes_b = resolve_players_fn(
        names=players_b,
        vocab=vocab_b,
        fuzzy_match_fn=lambda nm, vv, c: fuzzy_match_one_fn(nm, vv, cutoff=c),
        cutoff=0.6,
    )
    resolved_a_view = resolved_a if resolved_a else players_a
    resolved_b_view = resolved_b if resolved_b else players_b

    if notes_a or notes_b:
        with st.expander("Ver matching / sugestões", expanded=False):
            if notes_a:
                st.markdown("**Time A**")
                st.write(notes_a)
            if notes_b:
                st.markdown("**Time B**")
                st.write(notes_b)

    tr_a, df_a = _team_rating(by_player=by_player, players=resolved_a, format_num_fn=format_num_fn)
    tr_b, df_b = _team_rating(by_player=by_player, players=resolved_b, format_num_fn=format_num_fn)

    p_players_raw = _sigmoid(tr_a - tr_b)
    p_team_ref = state_get_float_fn(session_state, "_p_map_used", float("nan"))
    if not math.isfinite(p_team_ref):
        p_team_ref = 0.5
    games_a = int(pd.to_numeric(df_a.get("Games"), errors="coerce").fillna(0).sum()) if not df_a.empty else 0
    games_b = int(pd.to_numeric(df_b.get("Games"), errors="coerce").fillna(0).sum()) if not df_b.empty else 0
    n_players_eff = float(min(games_a, games_b))
    n_team_eff = state_get_float_fn(session_state, "_ml_team_games_min", float("nan"))
    fused = fuse_probs_by_precision_fn(
        p_team=p_team_ref,
        p_players=p_players_raw,
        n_team=n_team_eff,
        n_players=n_players_eff,
        team_scale=state_get_float_fn(session_state, "_fit_team_scale_used", state_get_float_fn(session_state, "_fit_team_scale", 1.0)),
        players_scale=state_get_float_fn(
            st.session_state,
            "_fit_players_scale_used",
            state_get_float_fn(st.session_state, "_fit_players_scale", 1.0),
        ),
        coverage=float(max(0.0, min(1.0, n_players_eff / max(1.0, n_team_eff + n_players_eff)))),
        transfer_signal=0.0,
        season_phase=("playoffs" if str(playoffs_opt).lower() in {"playoffs", "true", "1"} else "auto"),
    )
    p_team_a = float(max(0.02, min(0.98, float(fused.get("p_fused", p_team_ref)))))
    w_players = float(fused.get("w_players", float("nan")))
    odd_a = odd_from_prob_fn(p_team_a)
    odd_b = odd_from_prob_fn(1.0 - p_team_a)

    _render_players_result_card(
        st, team_a=team_a, team_b=team_b, odd_a=odd_a, odd_b=odd_b, p_team_a=p_team_a,
        p_players_raw=p_players_raw, p_team_ref=p_team_ref, w_players=w_players,
        n_team_eff=n_team_eff, n_players_eff=n_players_eff,
        format_num_fn=format_num_fn, format_pct_fn=format_pct_fn,
    )

    with st.expander("Detalhe por jogador (ratings)", expanded=False):
        cta, ctb = st.columns(2)
        with cta:
            st.markdown(f"**{team_a}**")
            if df_a.empty:
                st.info("Nenhum player reconhecido na base.")
            else:
                st.dataframe(df_a, width="stretch", hide_index=True)
        with ctb:
            st.markdown(f"**{team_b}**")
            if df_b.empty:
                st.info("Nenhum player reconhecido na base.")
            else:
                st.dataframe(df_b, width="stretch", hide_index=True)

    st.divider()
    render_tab_section(st, title="Mercados de players", subtitle="Defina linhas por lane e por jogador; a atualizacao so roda quando voce confirmar a analise.")
    st.caption('Defina linhas por lane/jogador. Atualiza apenas ao clicar em "Analisar players".')

    lane_rows = build_lane_rows_fn(resolved_a_view, resolved_b_view, ["top", "jng", "mid", "bot", "sup"])
    ensure_player_line_input_state_fn(session_state, lanes=len(lane_rows))

    with st.form("players_input_form", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.text_area("Time A — players (5)", height=120, key="pl_txtA_input")
        with col_b:
            st.text_area("Time B — players (5)", height=120, key="pl_txtB_input")
        submitted_top = st.form_submit_button("Analisar players", type="primary", key="pl_submit_top")
        st.caption('As lanes e odds abaixo atualizam quando clicar em "Analisar players".')
        st.markdown("##### Linhas individuais (10 jogadores)")
        st.caption("Para cada jogador, informe linhas separadas por vírgula (ex.: 2.5,3.5). O cálculo é individual: passou ou não passou da linha.")
        for idx, row in enumerate(lane_rows):
            c0, c1, c2 = st.columns([1.1, 1.45, 1.45])
            with c0:
                st.markdown(f"**{row['Lane']}**  \nTime A: {row['A'] or '-'}  \nTime B: {row['B'] or '-'}")
            with c1:
                st.markdown(f"**{row['A'] or '-'}**")
                st.text_input(f"Kills A ({row['Lane']})", key=f"pl_in_{idx}_kills_A")
                st.text_input(f"Deaths A ({row['Lane']})", key=f"pl_in_{idx}_deaths_A")
                st.text_input(f"Assists A ({row['Lane']})", key=f"pl_in_{idx}_assists_A")
            with c2:
                st.markdown(f"**{row['B'] or '-'}**")
                st.text_input(f"Kills B ({row['Lane']})", key=f"pl_in_{idx}_kills_B")
                st.text_input(f"Deaths B ({row['Lane']})", key=f"pl_in_{idx}_deaths_B")
                st.text_input(f"Assists B ({row['Lane']})", key=f"pl_in_{idx}_assists_B")
        submitted_bottom = st.form_submit_button("Analisar players", type="primary", key="pl_submit_bottom")
        submitted_players = bool(submitted_top or submitted_bottom)

    if submitted_players:
        apply_players_submission_state_fn(session_state, lanes=len(lane_rows))
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    if not any(str(x.get("A") or "").strip() or str(x.get("B") or "").strip() for x in lane_rows):
        st.info("Informe players para abrir os mercados por lane.")
        return

    wf_m = state_get_float_fn(session_state, "resumo_mix_w_model", 40.0)
    wf_h = state_get_float_fn(session_state, "resumo_mix_w_laplace", 40.0)
    wf_s = state_get_float_fn(session_state, "resumo_mix_w_sc", 20.0)
    min_h2h_players = max(1, state_get_int_fn(session_state, "resumo_min_sample_h2h", 3))
    ff_players_shadow_trace = feature_flag_enabled_fn("ff_players_panels_shadow_trace", default=True)

    med_rows = build_player_means_rows_fn(dfb=dfb, lane_rows=lane_rows, team_a=team_a, team_b=team_b, format_num_fn=format_num_fn)
    if med_rows:
        _subhead(st, "Médias por jogador")
        st.caption("Base estatística usada no modelo de players por lane.")
        dfm = pd.DataFrame(med_rows)
        dfm = df_contract_view_fn(
            dfm,
            ["Lane", "Time", "Player", "Amostra", "Kills (media)", "Kills (sd)", "Deaths (media)", "Deaths (sd)", "Assists (media)", "Assists (sd)"],
            context="Players > Médias",
            fill_value="-",
        )
        st.dataframe(dfm, width="stretch", hide_index=True)
        st.divider()

    recent_rows = build_recent_player_stats_rows_fn(
        dfb=dfb,
        lane_rows=lane_rows,
        team_a=team_a,
        team_b=team_b,
        format_num_fn=format_num_fn,
        max_games_per_player=8,
    )
    if recent_rows:
        _subhead(st, "Stats recentes dos players")
        st.caption("Últimos jogos por player/lane: campeão, resultado, KDA, KP%, duração e adversário.")
        dfr = pd.DataFrame(recent_rows)
        rows_before = len(dfr)
        rf1, rf2, rf3, rf4 = st.columns(4)
        with rf1:
            rf_team_opts = ["Todos", *sorted([x for x in dfr["Time"].astype(str).unique().tolist() if str(x).strip()])]
            _ensure_choice("pl_stats_filter_team", rf_team_opts, "Todos")
            rf_team = st.selectbox("Stats - Time", rf_team_opts, key="pl_stats_filter_team")
        with rf2:
            rf_lane_opts = ["Todos", *sorted(dfr["Lane"].astype(str).unique().tolist())]
            _ensure_choice("pl_stats_filter_lane", rf_lane_opts, "Todos")
            rf_lane = st.selectbox("Stats - Lane", rf_lane_opts, key="pl_stats_filter_lane")
        with rf3:
            rf_player_opts = ["Todos", *sorted(dfr["Player"].astype(str).unique().tolist())]
            _ensure_choice("pl_stats_filter_player", rf_player_opts, "Todos")
            rf_player = st.selectbox("Stats - Player", rf_player_opts, key="pl_stats_filter_player")
        with rf4:
            rf_opp_opts = ["Todos", *sorted(dfr["Oponente"].astype(str).unique().tolist())]
            _ensure_choice("pl_stats_filter_opp", rf_opp_opts, "Todos")
            rf_opp = st.selectbox("Stats - Oponente", rf_opp_opts, key="pl_stats_filter_opp")
        if rf_team != "Todos":
            dfr = dfr[dfr["Time"].astype(str) == str(rf_team)].copy()
        if rf_lane != "Todos":
            dfr = dfr[dfr["Lane"].astype(str) == str(rf_lane)].copy()
        if rf_player != "Todos":
            dfr = dfr[dfr["Player"].astype(str) == str(rf_player)].copy()
        if rf_opp != "Todos":
            dfr = dfr[dfr["Oponente"].astype(str) == str(rf_opp)].copy()
        append_players_recent_stats_panel_trace_fn(
            append_feature_trace_fn=append_feature_trace_fn,
            enabled=bool(ff_players_shadow_trace),
            teamA=team_a,
            teamB=team_b,
            team_filter=rf_team,
            lane_filter=rf_lane,
            player_filter=rf_player,
            opp_filter=rf_opp,
            rows_before=rows_before,
            rows_after=len(dfr),
        )
        dfr = df_contract_view_fn(
            dfr,
            ["Lane", "Time", "Player", "Campeão", "Resultado", "Tempo", "K/D/A", "KDA", "KP%", "Dano", "Oponente", "Data"],
            context="Players > Stats recentes",
            fill_value="-",
        )
        st.dataframe(dfr, width="stretch", hide_index=True)
        st.divider()

    per_player_lines_by_idx = build_per_player_lines_by_idx_fn(
        session_state=session_state,
        lane_rows=lane_rows,
        parse_lines_fn=parse_lines_fn,
    )
    out_rows, fair_rows = build_player_market_rows_fn(
        dfb=dfb,
        lane_rows=lane_rows,
        per_player_lines_by_idx=per_player_lines_by_idx,
        team_a=team_a,
        team_b=team_b,
        wf_m=float(wf_m),
        wf_h=float(wf_h),
        wf_s=float(wf_s),
        odd_from_prob_fn=odd_from_prob_fn,
        format_num_fn=format_num_fn,
        format_pct_fn=format_pct1_fn,
        max_odd=float(session_state.get("resumo_max_odd", 2.0) or 2.0),
        require_model=False,
        min_h2h_games=int(min_h2h_players),
    )
    if not out_rows:
        st.info("Sem dados suficientes para montar mercados de players no recorte atual.")
        return

    if fair_rows:
        _subhead(st, "Fair (Modelo) por jogador")
        st.caption("Odds justas do modelo para cada jogador/linha em Kills, Deaths e Assists.")
        dff = pd.DataFrame(fair_rows)
        ff1, ff2, ff3 = st.columns(3)
        with ff1:
            ff_team_opts = ["Todos", team_a, team_b]
            _ensure_choice("pl_fair_filter_team", ff_team_opts, "Todos")
            ff_team = st.selectbox("Fair - Time", ff_team_opts, key="pl_fair_filter_team")
        with ff2:
            ff_lane_opts = ["Todos", *sorted(dff["Lane"].unique().tolist())]
            _ensure_choice("pl_fair_filter_lane", ff_lane_opts, "Todos")
            ff_lane = st.selectbox("Fair - Lane", ff_lane_opts, key="pl_fair_filter_lane")
        with ff3:
            ff_player_opts = ["Todos", *sorted(dff["Player"].unique().tolist())]
            _ensure_choice("pl_fair_filter_player", ff_player_opts, "Todos")
            ff_player = st.selectbox("Fair - Player", ff_player_opts, key="pl_fair_filter_player")
        if ff_team != "Todos":
            dff = dff[dff["Time"].astype(str) == str(ff_team)].copy()
        if ff_lane != "Todos":
            dff = dff[dff["Lane"].astype(str) == str(ff_lane)].copy()
        if ff_player != "Todos":
            dff = dff[dff["Player"].astype(str) == str(ff_player)].copy()
        append_players_fair_panel_trace_fn(
            append_feature_trace_fn=append_feature_trace_fn,
            enabled=bool(ff_players_shadow_trace),
            teamA=team_a,
            teamB=team_b,
            team_filter=ff_team,
            lane_filter=ff_lane,
            player_filter=ff_player,
            rows_after=len(dff),
            rows_total=len(fair_rows),
        )
        dff = df_contract_view_fn(
            dff,
            ["Lane", "Time", "Player", "Mercado", "Media", "SD", "Linha", "Acima (%)", "Odd Acima", "Abaixo (%)", "Odd Abaixo", "Amostra"],
            context="Players > Fair",
            fill_value="-",
        )
        st.dataframe(dff, width="stretch", hide_index=True)

    st.divider()
    _subhead(st, "Mercados de Players (Modelo/Hist/Mix)")
    dfo = pd.DataFrame(out_rows)
    if not dfo.empty:
        st.caption(
            f'H2H só entra no Odd Mix quando atingir a amostra mínima ({int(min_h2h_players)} jogos). '
            'Se não atingir, confira "Obs Mix". Para "-" em odds (principalmente H2H), veja "Obs Odds".'
        )
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        mkt_team_opts = ["Todos", team_a, team_b]
        _ensure_choice("pl_mkt_filter_team", mkt_team_opts, "Todos")
        team_filter = st.selectbox("Filtro Time", mkt_team_opts, key="pl_mkt_filter_team")
    with f2:
        mkt_lane_opts = ["Todos", *sorted(dfo["Lane"].unique().tolist())]
        _ensure_choice("pl_mkt_filter_lane", mkt_lane_opts, "Todos")
        lane_filter = st.selectbox("Filtro Lane", mkt_lane_opts, key="pl_mkt_filter_lane")
    with f3:
        mkt_market_opts = ["Todos", *sorted(dfo["Mercado"].unique().tolist())]
        _ensure_choice("pl_mkt_filter_market", mkt_market_opts, "Todos")
        market_filter = st.selectbox("Filtro Mercado", mkt_market_opts, key="pl_mkt_filter_market")
    with f4:
        mkt_player_opts = ["Todos", *sorted(dfo["Player"].unique().tolist())]
        _ensure_choice("pl_mkt_filter_player", mkt_player_opts, "Todos")
        player_filter = st.selectbox("Filtro Player", mkt_player_opts, key="pl_mkt_filter_player")
    if team_filter != "Todos":
        dfo = dfo[dfo["Team"].astype(str) == str(team_filter)].copy()
    if lane_filter != "Todos":
        dfo = dfo[dfo["Lane"].astype(str) == str(lane_filter)].copy()
    if market_filter != "Todos":
        dfo = dfo[dfo["Mercado"].astype(str) == str(market_filter)].copy()
    if player_filter != "Todos":
        dfo = dfo[dfo["Player"].astype(str) == str(player_filter)].copy()

    append_players_markets_panel_trace_fn(
        append_feature_trace_fn=append_feature_trace_fn,
        enabled=bool(ff_players_shadow_trace),
        teamA=team_a,
        teamB=team_b,
        team_filter=team_filter,
        lane_filter=lane_filter,
        market_filter=market_filter,
        player_filter=player_filter,
        rows_after=len(dfo),
        rows_total=len(out_rows),
        fair_rows_total=len(fair_rows),
        min_h2h_games=int(min_h2h_players),
    )
    session_state["_ml_trace_players"] = _build_trace_rows(out_rows)
    append_players_panels_bundle_trace_fn(
        append_feature_trace_fn=append_feature_trace_fn,
        enabled=bool(ff_players_shadow_trace),
        teamA=team_a,
        teamB=team_b,
        recent_rows_total=len(recent_rows),
        market_rows_total=len(out_rows),
        fair_rows_total=len(fair_rows),
    )
    persist_combined_trace_file_fn()
    player_cols = ["Lane", "Team", "Player", "Mercado", "Lado", "Linha", "Stats", "H2H", "Odd (Modelo)", "Odd (Hist)", "Odd (H2H)", "Odd (Mix)"]
    if "Obs Odds" in dfo.columns and bool((dfo["Obs Odds"].astype(str).str.strip() != "").any()):
        player_cols.append("Obs Odds")
    player_cols.append("Obs Mix")
    dfo = df_contract_view_fn(
        dfo,
        [c for c in player_cols if c not in {"Obs Odds"}],
        optional_cols=["Obs Odds"],
        context="Players > Mercados",
        fill_value="-",
    )
    st.dataframe(dfo, width="stretch", hide_index=True)
    with st.expander("Trace de calibração (Players)"):
        trace_rows = session_state.get("_ml_trace_players", []) or []
        if trace_rows:
            st.dataframe(pd.DataFrame(trace_rows), width="stretch", hide_index=True)
            st.download_button(
                "Baixar trace (Players) JSON",
                data=json.dumps(trace_rows, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="ml_trace_players.json",
                mime="application/json",
                key="dl_trace_players_json",
                on_click="ignore",
            )
        else:
            st.caption("Sem trace nesta execução.")
