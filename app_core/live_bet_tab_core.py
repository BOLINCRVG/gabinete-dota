from __future__ import annotations

import itertools
from html import escape
from typing import Any, Callable, Mapping, Optional

import pandas as pd

from app_core.ui_cache_core import (
    cached_champion_knowledge_registry as load_champion_knowledge_registry,
)
from app_core.tab_ui_core import render_tab_hero, render_tab_section
from mlcore.utils import safe_float_or_none as _safe_float

from app_core.live_bet_shared_core import (
    VALUE_MARKET_META,
    PLAYER_PROP_META,
    ROLE_ORDER,
    parse_compact_draft_tokens,
    _parse_line_value,
    _parse_names,
    _chip_html,
    _profile_tone,
    _confidence_tone,
    _sample_tone,
    _market_read_tone,
    _resolution_tone,
)
from app_core.live_bet_analysis_core import (
    build_live_bet_pregame_summary,
    build_live_bet_market_value_summary,
    build_live_bet_live_assessment,
)



def _render_profile_comparison(
    st: Any,
    *,
    team_a: str,
    team_b: str,
    cards_a: list[dict[str, Any]],
    cards_b: list[dict[str, Any]],
) -> None:
    if not cards_a and not cards_b:
        st.caption("Sem base suficiente para o perfil comparativo.")
        return
    header_cols = st.columns([0.8, 0.8, 1.1, 1.1])
    with header_cols[0]:
        st.markdown("**Mercado**")
    with header_cols[1]:
        st.markdown("**Resolucao**")
    with header_cols[2]:
        st.markdown(f"**{team_a}**")
    with header_cols[3]:
        st.markdown(f"**{team_b}**")
    for row_a, row_b in itertools.zip_longest(cards_a, cards_b, fillvalue={}):
        market = str(row_a.get("Mercado") or row_b.get("Mercado") or "-")
        resolution = str(row_a.get("Resolucao") or row_b.get("Resolucao") or "-")
        cols = st.columns([0.8, 0.8, 1.1, 1.1])
        with cols[0]:
            st.markdown(f"**{market}**")
        with cols[1]:
            st.markdown(
                _chip_html(resolution, _resolution_tone(resolution)),
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                _chip_html(str(row_a.get("Perfil") or "-"), _profile_tone(str(row_a.get("Perfil") or "-")))
                + _chip_html(str(row_a.get("Amostra") or "-"), _sample_tone(str(row_a.get("Amostra") or "-")))
                + _chip_html(str(row_a.get("Confianca") or "-"), _confidence_tone(str(row_a.get("Confianca") or "-"))),
                unsafe_allow_html=True,
            )
        with cols[3]:
            st.markdown(
                _chip_html(str(row_b.get("Perfil") or "-"), _profile_tone(str(row_b.get("Perfil") or "-")))
                + _chip_html(str(row_b.get("Amostra") or "-"), _sample_tone(str(row_b.get("Amostra") or "-")))
                + _chip_html(str(row_b.get("Confianca") or "-"), _confidence_tone(str(row_b.get("Confianca") or "-"))),
                unsafe_allow_html=True,
            )


def _scope_base_filters(filters_cls: Any, active_filters: Any) -> Any:
    return filters_cls(
        year=getattr(active_filters, "year", None),
        league=None,
        split=getattr(active_filters, "split", None),
        playoffs=getattr(active_filters, "playoffs", None),
    )


def _build_live_scope(
    *,
    team_games: pd.DataFrame | None,
    filters_cls: Any,
    active_filters: Any,
    apply_filters_fn: Callable[[pd.DataFrame, Any], pd.DataFrame],
    team_a: str,
    team_b: str,
    selected_championship: str,
    manual_championship: str,
) -> dict[str, Any]:
    if team_games is None or team_games.empty:
        return {
            "base_scope": pd.DataFrame([]),
            "scope": pd.DataFrame([]),
            "options": ["Auto", "Manual/contexto"],
            "selected": "Auto",
            "context_label": "Auto",
            "scope_note": "Sem base no CSV atual.",
        }

    base_filters = _scope_base_filters(filters_cls, active_filters)
    try:
        base_scope = apply_filters_fn(team_games, base_filters).copy()
    except Exception:
        base_scope = team_games.copy()
    if base_scope is None or base_scope.empty:
        base_scope = pd.DataFrame([])

    relevant: list[str] = []
    if not base_scope.empty and {"team", "league"}.issubset(base_scope.columns):
        relevant_df = base_scope[base_scope["team"].astype(str).isin([str(team_a), str(team_b)])].copy()
        if not relevant_df.empty:
            if "date" in relevant_df.columns:
                relevant_df["date"] = pd.to_datetime(relevant_df["date"], errors="coerce")
                by_league = (
                    relevant_df.dropna(subset=["date"])
                    .groupby("league", dropna=False)["date"]
                    .max()
                    .sort_values(ascending=False)
                )
                relevant = [str(idx).strip() for idx in by_league.index.tolist() if str(idx).strip()]
            else:
                relevant = [str(v).strip() for v in relevant_df["league"].dropna().astype(str).unique().tolist() if str(v).strip()]
    # Todas as ligas do CSV completo como opções extras (depois das relevantes):
    # sem isso, um campeonato futuro dos times (ex.: MSI antes de jogarem nele)
    # nunca aparecia no seletor — só via "Manual/contexto" escondido.
    extras: list[str] = []
    if team_games is not None and not team_games.empty and "league" in team_games.columns:
        _seen = set(relevant)
        extras = sorted({
            str(v).strip() for v in team_games["league"].dropna().astype(str).tolist()
            if str(v).strip() and str(v).strip() not in _seen
        })
    options = ["Auto", *relevant, *extras, "Manual/contexto"]
    selected = str(selected_championship or "").strip()
    if not selected:
        selected = relevant[0] if relevant else "Auto"
    if selected not in options:
        options.insert(1, selected)

    manual_label = str(manual_championship or "").strip()
    scope_df = base_scope.copy()
    scope_note = "Usando todos os confrontos do recorte ativo."
    context_label = selected
    if selected == "Auto":
        context_label = relevant[0] if relevant else "Auto"
        scope_note = "Usando o recorte ativo completo, sem restringir por campeonato."
    elif selected == "Manual/contexto":
        context_label = manual_label or "Manual/contexto"
        if manual_label and "league" in base_scope.columns:
            matched = base_scope[base_scope["league"].astype(str) == manual_label].copy()
            if not matched.empty:
                scope_df = matched
                scope_note = "Campeonato manual encontrado no CSV do recorte."
            else:
                scope_note = "Campeonato manual ainda sem jogos no CSV; usando o recorte ativo como fallback."
        else:
            scope_note = "Campeonato manual sem jogos confirmados; usando o recorte ativo como fallback."
    else:
        if "league" in base_scope.columns:
            matched = base_scope[base_scope["league"].astype(str) == selected].copy()
            if not matched.empty:
                scope_df = matched
                scope_note = "Usando somente os jogos do campeonato selecionado."
            else:
                scope_note = "Campeonato sem jogos no recorte; usando o recorte ativo como fallback."
    return {
        "base_scope": base_scope,
        "scope": scope_df,
        "options": options,
        "selected": selected,
        "context_label": context_label or "Auto",
        "scope_note": scope_note,
    }


def _resolve_latest_players_by_team(df_scope_players: pd.DataFrame | None, team_name: str) -> list[str]:
    if df_scope_players is None or df_scope_players.empty or not str(team_name or "").strip():
        return []
    required = {"team", "player", "pos"}
    if not required.issubset(df_scope_players.columns):
        return []
    team_df = df_scope_players[df_scope_players["team"].astype(str) == str(team_name)].copy()
    if team_df.empty:
        return []
    if "date" in team_df.columns:
        team_df["date"] = pd.to_datetime(team_df["date"], errors="coerce")
    sort_cols: list[str] = []
    ascending: list[bool] = []
    if "date" in team_df.columns:
        sort_cols.append("date")
        ascending.append(False)
    if "gameid" in team_df.columns:
        sort_cols.append("gameid")
        ascending.append(False)
    if sort_cols:
        team_df = team_df.sort_values(sort_cols, ascending=ascending, na_position="last")

    if "gameid" in team_df.columns:
        grouped = team_df.groupby("gameid", dropna=False, sort=False)
        for _, game_df in grouped:
            role_map = {
                str(row.get("pos") or "").strip().lower(): str(row.get("player") or "").strip()
                for _, row in game_df.iterrows()
                if str(row.get("player") or "").strip()
            }
            if all(role_map.get(pos) for pos in ROLE_ORDER):
                return [role_map[pos] for pos in ROLE_ORDER]

    fallback: dict[str, str] = {}
    for _, row in team_df.iterrows():
        pos = str(row.get("pos") or "").strip().lower()
        player = str(row.get("player") or "").strip()
        if pos in ROLE_ORDER and player and pos not in fallback:
            fallback[pos] = player
    return [fallback[pos] for pos in ROLE_ORDER if fallback.get(pos)]






from app_core.live_bet_draft_core import (
    resolve_compact_draft_sides,
    build_live_bet_handicap_summary,
    build_live_bet_player_prop_summary,
    build_live_bet_tactical_notes,
    build_live_bet_draft_analysis,
    _resolve_pick_names,
    _label_player_spot_row,
    _label_player_meta_row,
    _lane_edge_text,
    _build_lane_cards,
    _wr_dual_html,
    _player_role_function,
    _team_role_snapshot,
    _edge_share_text,
)

def _render_draft_board(
    st: Any,
    *,
    team_name: str,
    side_label: str,
    rows: list[dict[str, Any]],
    registry: Mapping[str, Any] | None,
    champ_scope_label: str = "Campeonato",
) -> None:
    def _global_meta_tone(label: str) -> str:
        value = str(label or "").casefold()
        if "premium" in value:
            return "high"
        if "contestado" in value:
            return "medium"
        if "abaixo" in value:
            return "low"
        if "sem base" in value:
            return "sample_none"
        return "neutral"

    with st.container(border=True):
        st.markdown(f"### {side_label} | {team_name}")
        if not rows:
            st.caption("Sem base suficiente para montar o board desse lado.")
            return
        for row in rows:
            role = str(row.get("pos") or "-").upper()
            player = str(row.get("player") or "-")
            champ = str(row.get("champ") or "-")
            function_label = str(row.get("primary_label") or _player_role_function(str(row.get("pos") or ""), champ, registry))
            risk_label = str(row.get("risk_label") or "Seguro")
            player_line = (
                f"<div><strong>{escape(player)}</strong>"
                f"{_wr_dual_html(blue_value=row.get('player_wr_blue'), blue_games=int(row.get('player_games_blue') or 0), red_value=row.get('player_wr_red'), red_games=int(row.get('player_games_red') or 0))}</div>"
            )
            champ_line = (
                f"<div style='margin-top:4px;color:#d8ddea;'>"
                f"{escape(champ)}"
                f"</div>"
            )
            world_line = (
                f"<div style='margin-top:2px;color:#96a0b5;font-size:12px;'>"
                f"Mundo:"
                f"{_wr_dual_html(blue_value=row.get('champ_wr_world_blue'), blue_games=int(row.get('champ_games_world_blue') or 0), red_value=row.get('champ_wr_world_red'), red_games=int(row.get('champ_games_world_red') or 0))}"
                f"</div>"
            )
            scope_line = (
                f"<div style='margin-top:2px;color:#96a0b5;font-size:12px;'>"
                f"{escape(champ_scope_label)}:"
                f"{_wr_dual_html(blue_value=row.get('champ_wr_blue'), blue_games=int(row.get('champ_games_blue') or 0), red_value=row.get('champ_wr_red'), red_games=int(row.get('champ_games_red') or 0))}"
                f"</div>"
            )
            global_meta_label = str(row.get("global_meta_label") or "Situacional")
            league_alert = str(row.get("league_alert") or "").strip()
            meta_line = (
                _chip_html(function_label, "neutral")
                + _chip_html(global_meta_label, _global_meta_tone(global_meta_label))
                + _chip_html(str(row.get("tag") or "-"), _profile_tone(str(row.get("tag") or "-")))
            )
            if league_alert:
                meta_line += _chip_html(league_alert, "medium")
            st.markdown(f"`{role}`", unsafe_allow_html=False)
            st.markdown(player_line + champ_line + world_line + scope_line, unsafe_allow_html=True)
            st.markdown(meta_line, unsafe_allow_html=True)
            st.caption(f"{risk_label} | {row.get('lane_read') or '-'!s}")


def _render_role_card_grid(st: Any, *, team_a: str, team_b: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    for idx in range(0, len(rows), 2):
        col_pair = st.columns(2)
        for inner_idx, row in enumerate(rows[idx : idx + 2]):
            with col_pair[inner_idx]:
                with st.container(border=True):
                    st.caption(str(row.get("Papel") or "-"))
                    team_cols = st.columns(2)
                    with team_cols[0]:
                        st.markdown(f"**{team_a}**")
                        st.write(str(row.get(team_a) or "-"))
                    with team_cols[1]:
                        st.markdown(f"**{team_b}**")
                        st.write(str(row.get(team_b) or "-"))



def _top_combo_rows(df_combo: pd.DataFrame, top_n: int = 2) -> list[dict[str, Any]]:
    if df_combo is None or df_combo.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in df_combo.head(int(top_n)).iterrows():
        rows.append(
            {
                "combo": str(row.get("Combo") or "-"),
                "games": int(_safe_float(row.get("Games")) or 0),
                "wr": str(row.get("Win Rate") or "-"),
                "score_wr": str(row.get("WR Score") or "-"),
                "confidence": str(row.get("Confianca") or "-"),
                "kills": str(row.get("Avg Kills") or "-"),
                "towers": str(row.get("Avg Towers") or "-"),
                "time": str(row.get("Avg Time") or "-"),
            }
        )
    return rows


def _winner_read_text(value: Any, *, team_a: str, team_b: str) -> str:
    winner = str(value or "Equilibrado")
    if winner == str(team_a):
        return f"{team_a} melhor"
    if winner == str(team_b):
        return f"{team_b} melhor"
    return "Equilibrado"


def _comp_market_items(comp_compare: Mapping[str, Any], team_a: str, team_b: str) -> list[tuple[str, str]]:
    return [
        ("Early", _winner_read_text(comp_compare.get("early_team"), team_a=team_a, team_b=team_b)),
        ("Mid", _winner_read_text(comp_compare.get("mid_team"), team_a=team_a, team_b=team_b)),
        ("Late", _winner_read_text(comp_compare.get("late_team"), team_a=team_a, team_b=team_b)),
        ("Objetivos", _winner_read_text(comp_compare.get("objective_team"), team_a=team_a, team_b=team_b)),
        ("Engage", _winner_read_text(comp_compare.get("engage_team"), team_a=team_a, team_b=team_b)),
    ]


def _comp_plan_rows(comp_compare: Mapping[str, Any], team_a: str, team_b: str) -> list[dict[str, str]]:
    plan_profiles = dict(comp_compare.get("plan_profiles") or {})
    plan_a = dict(plan_profiles.get(str(team_a)) or {})
    plan_b = dict(plan_profiles.get(str(team_b)) or {})
    return [
        {
            "Leitura": "Plano",
            str(team_a): str(plan_a.get("identity") or "-"),
            str(team_b): str(plan_b.get("identity") or "-"),
        },
        {
            "Leitura": "CondiÃ§Ã£o de vitÃ³ria",
            str(team_a): str(plan_a.get("win_condition") or "-"),
            str(team_b): str(plan_b.get("win_condition") or "-"),
        },
        {
            "Leitura": "Precisa evitar",
            str(team_a): str(plan_a.get("avoid_condition") or "-"),
            str(team_b): str(plan_b.get("avoid_condition") or "-"),
        },
    ]


def _render_comp_summary(
    st: Any,
    *,
    team_a: str,
    team_b: str,
    comp_a: Mapping[str, Any],
    comp_b: Mapping[str, Any],
    comp_compare: Mapping[str, Any],
) -> None:
    styles = dict(comp_compare.get("styles") or {})
    windows = dict(comp_compare.get("windows") or {})
    c_a, c_b = st.columns(2)
    with c_a:
        with st.container(border=True):
            st.markdown(f"**{team_a}**")
            st.caption(f"Janela: {windows.get(team_a) or comp_a.get('window_label') or '-'}")
            st.caption(f"Estilo: {styles.get(team_a) or comp_a.get('primary_style') or '-'}")
    with c_b:
        with st.container(border=True):
            st.markdown(f"**{team_b}**")
            st.caption(f"Janela: {windows.get(team_b) or comp_b.get('window_label') or '-'}")
            st.caption(f"Estilo: {styles.get(team_b) or comp_b.get('primary_style') or '-'}")
    metric_cols = st.columns(5)
    for idx, (label, value) in enumerate(_comp_market_items(comp_compare, team_a, team_b)):
        with metric_cols[idx]:
            st.metric(label, value)





def _liga_compartilhada(team_games: Any, team_a: str, team_b: str) -> str:
    """A liga DO CONFRONTO, ou "" quando os dois times não são da mesma.

    Mesmo critério do resolve_shared_league_hint (Visão Geral): num jogo entre
    ligas diferentes não existe "a liga da partida", e vazio é a resposta certa —
    é o que faz o painel ao vivo cair no comportamento neutro em vez de assumir a
    liga de um dos lados. Cada time entra com a liga em que MAIS jogou no recorte.
    """
    try:
        if team_games is None or getattr(team_games, "empty", True):
            return ""
        cols = set(getattr(team_games, "columns", []))
        if not {"teamname", "league"} <= cols:
            return ""
        ligas = []
        for t in (team_a, team_b):
            sub = team_games[team_games["teamname"].astype(str) == str(t)]
            if sub.empty:
                return ""
            modo = sub["league"].astype(str).str.strip().replace("", None).dropna()
            if modo.empty:
                return ""
            ligas.append(modo.value_counts().idxmax())
        return str(ligas[0]) if ligas[0] == ligas[1] else ""
    except (KeyError, TypeError, ValueError, AttributeError):
        return ""


def render_live_bet_tab(*, st: Any, ctx: Mapping[str, Any]) -> None:
    render_tab_hero(
        st,
        title="Live-Bet",
        subtitle="A aposta no momento certo: entre mapas (momentum validado), durante o mapa (estado → vitória) e leitura de draft.",
        chips=["Entre mapas", "Durante o mapa", "Draft"],
    )

    team_games = ctx["team_games"]
    filters_cls = ctx["Filters"]
    active_filters = ctx["active_filters"]
    apply_filters_fn = ctx["apply_filters_fn"]
    csv_sig = ctx["csv_sig"]
    state_get_str_fn = ctx["state_get_str_fn"]
    load_player_rows_fn = ctx["load_player_rows_fn"]
    apply_filters_players_fn = ctx["apply_filters_players_fn"]
    champion_vocab_fn = ctx["champion_vocab_fn"]
    fuzzy_match_one_fn = ctx["fuzzy_match_one_fn"]
    format_num_fn = ctx["format_num_fn"]
    min_to_mmss_fn = ctx["min_to_mmss_fn"]

    teams: list[str] = []
    try:
        if team_games is not None and not team_games.empty and "team" in team_games.columns:
            teams = sorted([str(x) for x in team_games["team"].fillna("").astype(str).unique().tolist() if str(x)])
    except Exception:
        teams = []
    team_options = ["", *list(teams)]
    selected_a_seed = str(st.session_state.get("lb_team_a") or "")
    selected_b_seed = str(st.session_state.get("lb_team_b") or "")
    team_a_idx = team_options.index(selected_a_seed) if selected_a_seed in team_options else 0
    team_b_idx = team_options.index(selected_b_seed) if selected_b_seed in team_options else 0

    with st.container(border=True):
        render_tab_section(
            st,
            title="Confronto",
            subtitle="Escolha os times e ajuste o campeonato so quando precisar.",
        )
        c1, c2 = st.columns(2)
        with c1:
            if str(st.session_state.get("lb_team_a", "") or "") not in [str(x) for x in (team_options or [""])]:
                st.session_state["lb_team_a"] = str((team_options or [""])[team_a_idx])
            team_a = st.selectbox(
                "Time 1",
                team_options or [""],
                key="lb_team_a",
                format_func=lambda value: "Selecione..." if not str(value or "").strip() else str(value),
            )
        with c2:
            if str(st.session_state.get("lb_team_b", "") or "") not in [str(x) for x in (team_options or [""])]:
                st.session_state["lb_team_b"] = str((team_options or [""])[team_b_idx])
            team_b = st.selectbox(
                "Time 2",
                team_options or [""],
                key="lb_team_b",
                format_func=lambda value: "Selecione..." if not str(value or "").strip() else str(value),
            )

        scope_seed = _build_live_scope(
            team_games=team_games,
            filters_cls=filters_cls,
            active_filters=active_filters,
            apply_filters_fn=apply_filters_fn,
            team_a=str(team_a),
            team_b=str(team_b),
            selected_championship=str(st.session_state.get("lb_championship", "") or ""),
            manual_championship=str(st.session_state.get("lb_championship_manual", "") or ""),
        )
        options = list(scope_seed.get("options") or ["Auto", "Manual/contexto"])
        selected_seed = str(scope_seed.get("selected") or "Auto")
        seed_idx = options.index(selected_seed) if selected_seed in options else 0
        if str(st.session_state.get("lb_championship", "") or "") not in options:
            st.session_state["lb_championship"] = options[seed_idx]
        championship_selected = st.selectbox("Campeonato", options, key="lb_championship")
        manual_championship = str(st.session_state.get("lb_championship_manual", "") or "")
        if str(championship_selected) == "Manual/contexto":
            if "lb_championship_manual" not in st.session_state:
                st.session_state["lb_championship_manual"] = manual_championship
            manual_championship = st.text_input(
                "Campeonato manual/contexto",
                key="lb_championship_manual",
                placeholder="ex: First Stand 2026",
            )

    scope_payload = _build_live_scope(
        team_games=team_games,
        filters_cls=filters_cls,
        active_filters=active_filters,
        apply_filters_fn=apply_filters_fn,
        team_a=str(team_a),
        team_b=str(team_b),
        selected_championship=str(championship_selected),
        manual_championship=str(manual_championship),
    )
    df_scope = scope_payload["scope"]
    base_scope = scope_payload["base_scope"]
    context_label = str(scope_payload["context_label"] or "Auto")
    scope_note = str(scope_payload["scope_note"] or "")

    if not str(team_a or "").strip() or not str(team_b or "").strip():
        st.info("Selecione os dois times para ativar a leitura pre-jogo e draft.")
        return

    st.caption(f"Contexto ativo: {team_a} x {team_b} | Campeonato: {context_label}")
    if scope_note and str(scope_note).strip():
        st.caption(scope_note)

    # ---- ENTRE MAPAS (série ao vivo) --------------------------------------
    # Primeira entrega do bloco in-game (2026-07-03): momentum de séries medido
    # e aprovado no gate OOS (mapa-2 logloss +0.0084). Marca o placar e recebe
    # a odd justa do próximo mapa (com momentum) + série restante + edge/stake.
    with st.container(border=True):
        render_tab_section(
            st,
            title="Entre mapas (série ao vivo)",
            subtitle="Marque o placar da série: odd justa do próximo mapa com momentum validado + fechar série + edge contra a casa.",
        )
        try:
            from app_core.live_between_maps_core import render_between_maps_block
            render_between_maps_block(
                st=st,
                session_state=st.session_state,
                team_a=str(team_a),
                team_b=str(team_b),
                artifact_path=str(ctx.get("artifact_path", "") or ""),
                format_num_fn=format_num_fn,
            )
        except Exception as _exc_bm:
            try:
                from app_core.ui_notify_core import notify_exception
                notify_exception(st, _exc_bm, context="Falha no bloco entre-mapas")
            except Exception:
                pass

    # DURANTE O MAPA: caixa reservada AQUI (posição 2 da aba) para o painel
    # estado→vitória (Fase 6). O código do painel roda lá no fim (depende de
    # comp_compare/scope calculados no meio), mas renderiza NESTA posição —
    # mesmo padrão de containers da reforma da Visão Geral. Antes ele ficava
    # num expander colapsado no rodapé (o maior ativo validado, escondido).
    _box_durante_mapa = st.container()

    render_tab_section(st, title="Pré-jogo", subtitle="Base do spot sem repetir informação na frente principal.")
    with st.spinner("Calculando resumo pré-jogo..."):
        pregame = build_live_bet_pregame_summary(
            df_scope=df_scope,
            team_a=str(team_a),
            team_b=str(team_b),
            min_to_mmss_fn=min_to_mmss_fn,
            format_num_fn=format_num_fn,
        )
    if pregame["stats_view"] is None or pregame["stats_view"].empty:
        st.info("Sem base suficiente no contexto escolhido para montar a leitura pre-jogo.")
    else:
        st.caption(str(pregame.get("summary_line") or ""))
        with st.expander("Abrir detalhe do prÃ©-jogo", expanded=False):
            st.caption("Cada mercado mostra perfil, amostra e confianca para os dois lados.")
            st.caption(
                "Resolucao mostra o que os dois lados repetem no mercado atual. "
                "Moderado/forte medem a intensidade do perfil no recorte. "
                "Alta/Media/Baixa medem a confianca da leitura pela amostra e pelo afastamento do perfil."
            )
            _render_profile_comparison(
                st,
                team_a=str(team_a),
                team_b=str(team_b),
                cards_a=list(pregame.get("profile_cards_a") or []),
                cards_b=list(pregame.get("profile_cards_b") or []),
            )
            st.dataframe(pregame["stats_view"], width="stretch", hide_index=True)

    render_tab_section(
        st,
        title="Com draft",
        subtitle="Monte o draft e a aba sobe o board dos 10 players com o valor do spot no centro.",
    )
    if str(st.session_state.get("lb_blue_side_team", "") or "") not in {str(team_a), str(team_b)}:
        st.session_state["lb_blue_side_team"] = str(team_a)
    blue_side_team = st.radio(
        "Blue side",
        [str(team_a), str(team_b)],
        horizontal=True,
        key="lb_blue_side_team",
    )
    if "lb_draft_compact" not in st.session_state:
        st.session_state["lb_draft_compact"] = state_get_str_fn(st.session_state, "lb_draft_compact", "")
    compact_draft_text = st.text_area(
        "Draft compacto",
        key="lb_draft_compact",
        height=84,
        placeholder="Ornn DrMundo Ryze Lucian Nami Renekton Qiyana Anivia Varus Alistar",
    )
    st.caption("Primeiros 5 campeões = blue side. Ultimos 5 = red side.")

    with st.spinner("Carregando base de players para analise de draft..."):
        df_players = load_player_rows_fn(csv_sig)
    champion_registry = load_champion_knowledge_registry()
    if df_players is None or df_players.empty:
        st.info("Sem rows de players no CSV atual para a análise de draft.")
        return

    player_filters = _scope_base_filters(filters_cls, active_filters)
    selected_known_league = str(championship_selected) if str(championship_selected) not in {"", "Auto", "Manual/contexto"} else ""
    if selected_known_league:
        player_filters = filters_cls(
            year=getattr(active_filters, "year", None),
            league=str(selected_known_league),
            split=getattr(active_filters, "split", None),
            playoffs=getattr(active_filters, "playoffs", None),
        )
    df_players_scope = apply_filters_players_fn(df_players, player_filters).copy()
    if df_players_scope is None or df_players_scope.empty:
        df_players_scope = df_players.copy()

    team_a_recent_players = _resolve_latest_players_by_team(df_players_scope, str(team_a))
    team_b_recent_players = _resolve_latest_players_by_team(df_players_scope, str(team_b))
    if (str(st.session_state.get("_lb_players_seed_a") or "") != str(team_a)) or not str(st.session_state.get("lb_players_a") or "").strip():
        st.session_state["lb_players_a"] = "\n".join(team_a_recent_players[:5])
        st.session_state["_lb_players_seed_a"] = str(team_a)
    if (str(st.session_state.get("_lb_players_seed_b") or "") != str(team_b)) or not str(st.session_state.get("lb_players_b") or "").strip():
        st.session_state["lb_players_b"] = "\n".join(team_b_recent_players[:5])
        st.session_state["_lb_players_seed_b"] = str(team_b)

    with st.expander("Editar players e picks (opcional)", expanded=False):
        st.caption("Ordem esperada: Top / JG / Mid / ADC / Sup para cada lado.")
        if "lb_champs_a" not in st.session_state:
            st.session_state["lb_champs_a"] = state_get_str_fn(st.session_state, "lb_champs_a", "")
        if "lb_champs_b" not in st.session_state:
            st.session_state["lb_champs_b"] = state_get_str_fn(st.session_state, "lb_champs_b", "")
        d1, d2 = st.columns(2)
        with d1:
            players_a_text = st.text_area(
                f"Players {team_a}",
                height=110,
                key="lb_players_a",
                placeholder="Top\nJG\nMid\nADC\nSup",
            )
            champs_a_text = st.text_area(
                f"Draft {team_a}",
                height=110,
                key="lb_champs_a",
                placeholder="Campeao Top\nCampeao JG\nCampeao Mid\nCampeao ADC\nCampeao Sup",
            )
        with d2:
            players_b_text = st.text_area(
                f"Players {team_b}",
                height=110,
                key="lb_players_b",
                placeholder="Top\nJG\nMid\nADC\nSup",
            )
            champs_b_text = st.text_area(
                f"Draft {team_b}",
                height=110,
                key="lb_champs_b",
                placeholder="Campeao Top\nCampeao JG\nCampeao Mid\nCampeao ADC\nCampeao Sup",
            )

    vocab = champion_vocab_fn(df_players_scope)
    if not vocab:
        vocab = champion_vocab_fn(df_players)
    player_vocab = sorted([x for x in df_players_scope.get("player", pd.Series(dtype=str)).fillna("").astype(str).unique().tolist() if x])

    players_a = []
    players_b = []
    try:
        players_a, _ = _resolve_pick_names(tokens=_parse_names(players_a_text)[:5], vocab=player_vocab, fuzzy_match_one_fn=fuzzy_match_one_fn)
        players_b, _ = _resolve_pick_names(tokens=_parse_names(players_b_text)[:5], vocab=player_vocab, fuzzy_match_one_fn=fuzzy_match_one_fn)
    except Exception:
        players_a, players_b = [], []

    unmatched_a: list[dict[str, Any]] = []
    unmatched_b: list[dict[str, Any]] = []
    if str(compact_draft_text).strip():
        compact_payload = resolve_compact_draft_sides(
            text=str(compact_draft_text),
            vocab=vocab,
            blue_team=str(blue_side_team),
            team_a=str(team_a),
            team_b=str(team_b),
            fuzzy_match_one_fn=fuzzy_match_one_fn,
        )
        picks_a = list(compact_payload.get("picks_a") or [])
        picks_b = list(compact_payload.get("picks_b") or [])
        unmatched_all = list(compact_payload.get("unmatched") or [])
        unmatched_a = unmatched_all
    else:
        picks_a, unmatched_a = _resolve_pick_names(tokens=_parse_names(champs_a_text)[:5], vocab=vocab, fuzzy_match_one_fn=fuzzy_match_one_fn)
        picks_b, unmatched_b = _resolve_pick_names(tokens=_parse_names(champs_b_text)[:5], vocab=vocab, fuzzy_match_one_fn=fuzzy_match_one_fn)

    if unmatched_a or unmatched_b:
        st.warning("Alguns campeões não foram reconhecidos com seguranca. Revise os nomes ou escolha a forma usada no CSV.")
        with st.expander("Ver entradas não reconhecidas", expanded=False):
            for row in (unmatched_a + unmatched_b)[:8]:
                st.write(row)

    if len(picks_a) < 5 or len(picks_b) < 5:
        st.info("Cole os 5 picks de cada lado para ativar a leitura de draft.")
        return

    with st.spinner("Analisando draft, lanes e composição..."):
        draft = build_live_bet_draft_analysis(
            df_scope_players=df_players_scope,
            df_all_players=df_players,
            df_scope_teams=(df_scope if isinstance(df_scope, pd.DataFrame) and not df_scope.empty else base_scope),
            team_a=str(team_a),
            team_b=str(team_b),
            blue_side_team=str(blue_side_team),
            league_context=str(selected_known_league or ""),
            picks_a=picks_a[:5],
            picks_b=picks_b[:5],
            players_a=players_a[:5],
            players_b=players_b[:5],
            format_num_fn=format_num_fn,
            champion_registry=champion_registry,
        )

    verdict = dict(draft.get("verdict") or {})
    lane_view = draft.get("lane_view")
    duos_a = draft.get("duos_a")
    duos_b = draft.get("duos_b")
    trios_a = draft.get("trios_a")
    trios_b = draft.get("trios_b")
    comp_a = dict(draft.get("comp_a") or {})
    comp_b = dict(draft.get("comp_b") or {})
    comp_compare = dict(draft.get("comp_compare") or {})
    draft_confidence = dict(draft.get("draft_confidence") or {})
    player_leadership_a = dict(draft.get("player_leadership_a") or {})
    player_leadership_b = dict(draft.get("player_leadership_b") or {})
    meta_leadership_a = dict(draft.get("meta_leadership_a") or {})
    meta_leadership_b = dict(draft.get("meta_leadership_b") or {})
    combo_support_a = dict(draft.get("combo_support_a") or {})
    combo_support_b = dict(draft.get("combo_support_b") or {})

    cards_a, cards_b = _build_lane_cards(
        lane_view if isinstance(lane_view, pd.DataFrame) else pd.DataFrame([]),
        str(team_a),
        str(team_b),
        champion_registry,
    )
    top_duos_a = _top_combo_rows(duos_a if isinstance(duos_a, pd.DataFrame) else pd.DataFrame([]), top_n=2)
    top_duos_b = _top_combo_rows(duos_b if isinstance(duos_b, pd.DataFrame) else pd.DataFrame([]), top_n=2)
    top_trios_a = _top_combo_rows(trios_a if isinstance(trios_a, pd.DataFrame) else pd.DataFrame([]), top_n=2)
    top_trios_b = _top_combo_rows(trios_b if isinstance(trios_b, pd.DataFrame) else pd.DataFrame([]), top_n=2)
    lane_pairs = list(zip(cards_a, cards_b))

    if str(blue_side_team) == str(team_a):
        blue_team_name, blue_rows = str(team_a), cards_a
        red_team_name, red_rows = str(team_b), cards_b
    else:
        blue_team_name, blue_rows = str(team_b), cards_b
        red_team_name, red_rows = str(team_a), cards_a
    board_scope_label = "Campeonato" if str(championship_selected) not in {"", "Auto"} else "Recorte"

    analysis_cols = st.columns([1.05, 1.35, 1.05])
    with analysis_cols[0]:
        _render_draft_board(st, team_name=blue_team_name, side_label="Blue", rows=blue_rows, registry=champion_registry, champ_scope_label=board_scope_label)
    with analysis_cols[2]:
        _render_draft_board(st, team_name=red_team_name, side_label="Red", rows=red_rows, registry=champion_registry, champ_scope_label=board_scope_label)

    with analysis_cols[1]:
        render_tab_section(
            st,
            title="Leitura central do spot",
            subtitle="O foco aqui Ã© valor: total, handicap e prop de jogador. O bruto fica recolhido abaixo.",
        )
        role_snapshot_a = _team_role_snapshot(cards_a)
        role_snapshot_b = _team_role_snapshot(cards_b)
        tactical_notes = build_live_bet_tactical_notes(
            team_a=str(team_a),
            team_b=str(team_b),
            picks_a=picks_a[:5],
            picks_b=picks_b[:5],
            players_a=players_a[:5],
            players_b=players_b[:5],
            comp_compare=comp_compare,
            lane_view=lane_view if isinstance(lane_view, pd.DataFrame) else pd.DataFrame([]),
            board_rows_a=cards_a,
            board_rows_b=cards_b,
            registry=champion_registry,
            format_num_fn=format_num_fn,
        )
        with st.container(border=True):
            st.markdown(f"### {verdict.get('summary') or 'Sem leitura de draft.'}")
            metric_cols = st.columns(5)
            with metric_cols[0]:
                st.metric("Melhor draft", str(verdict.get("favored_team") or "-"))
            with metric_cols[1]:
                st.metric("Edge", f"{format_num_fn(abs(_safe_float(verdict.get('score_pp')) or 0.0), 1)} p.p.")
            with metric_cols[2]:
                st.metric("Early", _edge_share_text(comp_compare.get("early_delta"), team_a=str(team_a), team_b=str(team_b), scale=7.0))
            with metric_cols[3]:
                st.metric("Late", _edge_share_text(comp_compare.get("late_delta"), team_a=str(team_a), team_b=str(team_b), scale=7.0))
            with metric_cols[4]:
                st.metric("Confianca", str(draft_confidence.get("label") or "Baixa"))
            combo_support_rows: list[dict[str, Any]] = []
            if combo_support_a:
                combo_support_rows.append(
                    {
                        "Apoio de sinergia": str(team_a),
                        "Combo": f"{combo_support_a.get('kind', '-')} | {combo_support_a.get('combo', '-')}",
                        "Leitura": str(combo_support_a.get("reading") or "-"),
                        "Base": f"Jogos {combo_support_a.get('games', 0)} | Score {combo_support_a.get('score_wr', '-') } | {combo_support_a.get('confidence', '-')}",
                    }
                )
            if combo_support_b:
                combo_support_rows.append(
                    {
                        "Apoio de sinergia": str(team_b),
                        "Combo": f"{combo_support_b.get('kind', '-')} | {combo_support_b.get('combo', '-')}",
                        "Leitura": str(combo_support_b.get("reading") or "-"),
                        "Base": f"Jogos {combo_support_b.get('games', 0)} | Score {combo_support_b.get('score_wr', '-') } | {combo_support_b.get('confidence', '-')}",
                        }
                    )
            league_alert_rows: list[dict[str, Any]] = []
            for team_name, rows in ((str(team_a), cards_a), (str(team_b), cards_b)):
                for row in rows:
                    league_alert = str(row.get("league_alert") or "").strip()
                    if not league_alert:
                        continue
                    league_alert_rows.append(
                        {
                            "Leitura campeonato": team_name,
                            "Pick": _label_player_spot_row(row),
                            "Alerta": league_alert,
                            "Mundo": f"B {row.get('champ_wr_world_blue', '-')}% ({int(row.get('champ_games_world_blue') or 0)}) | R {row.get('champ_wr_world_red', '-')}% ({int(row.get('champ_games_world_red') or 0)})",
                            board_scope_label: f"B {row.get('champ_wr_blue', '-')}% ({int(row.get('champ_games_blue') or 0)}) | R {row.get('champ_wr_red', '-')}% ({int(row.get('champ_games_red') or 0)})",
                        }
                    )
            role_rows: list[dict[str, Any]] = []
            for label, key in (("Carrega", "carry"), ("Inicia", "engage"), ("Exposto", "exposed")):
                row_a = dict(role_snapshot_a.get(key) or {})
                row_b = dict(role_snapshot_b.get(key) or {})
                role_rows.append(
                    {
                        "Papel": label,
                        str(team_a): f"{row_a.get('player', '-')} / {row_a.get('champ', '-')}",
                        str(team_b): f"{row_b.get('player', '-')} / {row_b.get('champ', '-')}",
                    }
                )
            for label, key in (
                ("Melhor player", "best_player"),
                ("Carry do spot", "carry_spot"),
                ("Motor de engage", "engage_engine"),
                ("Mais exposto", "most_exposed"),
            ):
                row_a = dict(player_leadership_a.get(key) or {})
                row_b = dict(player_leadership_b.get(key) or {})
                role_rows.append(
                    {
                        "Papel": label,
                        str(team_a): _label_player_spot_row(row_a),
                        str(team_b): _label_player_spot_row(row_b),
                    }
                )
            for label, key in (("Pick premium", "best_global_meta"), ("Pick abaixo do meta", "weak_global_meta")):
                row_a = dict(meta_leadership_a.get(key) or {})
                row_b = dict(meta_leadership_b.get(key) or {})
                role_rows.append(
                    {
                        "Papel": label,
                        str(team_a): _label_player_meta_row(row_a, label_key="global_meta_label"),
                        str(team_b): _label_player_meta_row(row_b, label_key="global_meta_label"),
                    }
                )
            _render_role_card_grid(st, team_a=str(team_a), team_b=str(team_b), rows=role_rows)
            for note in tactical_notes[:4]:
                st.markdown(f"- {note}")
            with st.expander("Abrir leituras auxiliares", expanded=False):
                st.dataframe(pd.DataFrame(_comp_plan_rows(comp_compare, str(team_a), str(team_b))), width="stretch", hide_index=True)
                if combo_support_rows:
                    st.dataframe(pd.DataFrame(combo_support_rows), width="stretch", hide_index=True)
                if league_alert_rows:
                    st.dataframe(pd.DataFrame(league_alert_rows), width="stretch", hide_index=True)

        total_cols = st.columns(2)
        kills_line_key = str(VALUE_MARKET_META["Kills"]["line_key"])
        towers_line_key = str(VALUE_MARKET_META["Torres"]["line_key"])
        if kills_line_key not in st.session_state:
            st.session_state[kills_line_key] = state_get_str_fn(st.session_state, kills_line_key, "")
        if towers_line_key not in st.session_state:
            st.session_state[towers_line_key] = state_get_str_fn(st.session_state, towers_line_key, "")
        if "lb_value_line_hc_kills_t1" not in st.session_state:
            st.session_state["lb_value_line_hc_kills_t1"] = state_get_str_fn(st.session_state, "lb_value_line_hc_kills_t1", "")
        if "lb_value_line_hc_towers_t1" not in st.session_state:
            st.session_state["lb_value_line_hc_towers_t1"] = state_get_str_fn(st.session_state, "lb_value_line_hc_towers_t1", "")
        with total_cols[0]:
            kills_line_raw = st.text_input(
                "Kills",
                key=kills_line_key,
                placeholder="ex: 26.5",
            )
        with total_cols[1]:
            towers_line_raw = st.text_input(
                "Torres",
                key=towers_line_key,
                placeholder="ex: 10.5",
            )
        hc_cols = st.columns(2)
        with hc_cols[0]:
            hc_kills_raw = st.text_input(f"HC Kills {team_a}", key="lb_value_line_hc_kills_t1", placeholder="ex: -4.5")
        with hc_cols[1]:
            hc_towers_raw = st.text_input(f"HC Torres {team_a}", key="lb_value_line_hc_towers_t1", placeholder="ex: -2.5")

        line_values: dict[str, Optional[float]] = {
            "Kills": _parse_line_value(kills_line_raw),
            "Torres": _parse_line_value(towers_line_raw),
            "DragÃµes": None,
            "BarÃµes": None,
            "Tempo": None,
            "HC Kills T1": _parse_line_value(hc_kills_raw),
            "HC Torres T1": _parse_line_value(hc_towers_raw),
        }

        with st.spinner("Avaliando valor dos mercados..."):
            value_summary = build_live_bet_market_value_summary(
                df_scope=(df_scope if isinstance(df_scope, pd.DataFrame) and not df_scope.empty else base_scope),
                team_a=str(team_a),
                team_b=str(team_b),
                blue_side_team=str(blue_side_team),
                pregame=pregame,
                draft=draft,
                line_values=line_values,
                format_num_fn=format_num_fn,
                min_to_mmss_fn=min_to_mmss_fn,
            )
        handicap_summary = build_live_bet_handicap_summary(
            df_scope=(df_scope if isinstance(df_scope, pd.DataFrame) and not df_scope.empty else base_scope),
            team_a=str(team_a),
            team_b=str(team_b),
            blue_side_team=str(blue_side_team),
            draft=draft,
            line_values=line_values,
            format_num_fn=format_num_fn,
        )

        def _market_row(df_in: Any, market_name: str) -> dict[str, Any]:
            if not isinstance(df_in, pd.DataFrame) or df_in.empty:
                return {}
            row = df_in[df_in["Mercado"].astype(str) == str(market_name)].head(1)
            if row.empty:
                return {}
            return dict(row.iloc[0].to_dict())

        def _render_value_callout(title: str, row: Mapping[str, Any]) -> None:
            if not row:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.caption("Sem linha suficiente para analisar esse mercado.")
                return
            with st.container(border=True):
                st.markdown(f"**{title} | Linha {row.get('Linha') or '-'}**")
                st.markdown(f"Justo **{row.get('Justo') or '-'}**")
                st.markdown(
                    _chip_html(str(row.get("Leitura") or "-"), _market_read_tone(str(row.get("Leitura") or "-")))
                    + _chip_html(str(row.get("Confianca") or "-"), _confidence_tone(str(row.get("Confianca") or "-"))),
                    unsafe_allow_html=True,
                )
                st.caption(str(row.get("Drivers") or "-"))
        best_value_row = dict(value_summary.get("best_row") or {})
        if best_value_row:
            with st.container(border=True):
                st.markdown(f"**Maior valor do spot | {best_value_row.get('Mercado') or '-'} | Linha {best_value_row.get('Linha') or '-'}**")
                st.markdown(f"Justo **{best_value_row.get('Justo') or '-'}**")
                st.markdown(
                    _chip_html(str(best_value_row.get("Leitura") or "-"), _market_read_tone(str(best_value_row.get("Leitura") or "-")))
                    + _chip_html(str(best_value_row.get("Confianca") or "-"), _confidence_tone(str(best_value_row.get("Confianca") or "-"))),
                    unsafe_allow_html=True,
                )
                st.caption(str(best_value_row.get("Drivers") or "-"))

        st.caption("Mercados principais")
        main_cols = st.columns(2)
        with main_cols[0]:
            _render_value_callout("Kills", _market_row(value_summary.get("rows"), "Kills"))
        with main_cols[1]:
            _render_value_callout("Torres", _market_row(value_summary.get("rows"), "Torres"))

        st.caption("Handicap")
        hc_render_cols = st.columns(2)
        with hc_render_cols[0]:
            _render_value_callout(f"HC Kills {team_a}", _market_row(handicap_summary.get("rows"), f"HC Kills {team_a}"))
        with hc_render_cols[1]:
            _render_value_callout(f"HC Torres {team_a}", _market_row(handicap_summary.get("rows"), f"HC Torres {team_a}"))

        prop_candidates: list[dict[str, str]] = []
        for team_name, team_side, rows in (
            (str(team_a), "Blue" if str(blue_side_team) == str(team_a) else "Red", cards_a),
            (str(team_b), "Blue" if str(blue_side_team) == str(team_b) else "Red", cards_b),
        ):
            for row in rows:
                prop_candidates.append(
                    {
                        "label": f"{team_name} | {row.get('player', '-')} | {row.get('champ', '-')}",
                        "team": team_name,
                        "side": team_side,
                        "player": str(row.get("player") or ""),
                        "champ": str(row.get("champ") or ""),
                    }
                )
        prop_map = {item["label"]: item for item in prop_candidates}
        prop_select_cols = st.columns([1.4, 0.8, 0.6])
        with prop_select_cols[0]:
            prop_selected = st.selectbox("Prop de jogador", ["", *list(prop_map.keys())], key="lb_prop_focus")
        with prop_select_cols[1]:
            prop_metric = st.selectbox("Mercado do player", list(PLAYER_PROP_META.keys()), key="lb_prop_metric")
        with prop_select_cols[2]:
            if "lb_prop_line_value" not in st.session_state:
                st.session_state["lb_prop_line_value"] = state_get_str_fn(st.session_state, "lb_prop_line_value", "")
            prop_line_raw = st.text_input("Linha player", key="lb_prop_line_value", placeholder="ex: 6.5")
        prop_row = dict(prop_map.get(str(prop_selected) or "") or {})
        kills_means = dict((draft.get("market_means") or {}).get("Kills") or {})
        prop_team_name = str(prop_row.get("team") or "")
        prop_opponent_name = str(team_b) if prop_team_name == str(team_a) else str(team_a)
        if prop_team_name == str(team_a):
            prop_expected_kills = _safe_float(kills_means.get("team_a"))
            prop_expected_deaths = _safe_float(kills_means.get("team_b"))
        else:
            prop_expected_kills = _safe_float(kills_means.get("team_b"))
            prop_expected_deaths = _safe_float(kills_means.get("team_a"))
        prop_summary = build_live_bet_player_prop_summary(
            df_scope_players=df_players_scope,
            team_name=prop_team_name,
            opponent_name=prop_opponent_name,
            player_name=str(prop_row.get("player") or ""),
            champ_name=str(prop_row.get("champ") or ""),
            side_label=str(prop_row.get("side") or ""),
            prop_name=str(prop_metric),
            line_value=prop_line_raw,
            registry=champion_registry,
            comp_compare=comp_compare,
            expected_team_kills=prop_expected_kills,
            expected_team_deaths=prop_expected_deaths,
            format_num_fn=format_num_fn,
        )
        st.caption("Prop de jogador")
        if prop_summary.get("ok"):
            with st.container(border=True):
                st.markdown(
                    f"**{prop_summary.get('player')} | {prop_summary.get('champ')} | {prop_summary.get('prop')} {prop_summary.get('line')}**"
                )
                st.markdown(f"Justo **{prop_summary.get('fair')}**")
                st.markdown(
                    _chip_html(str(prop_summary.get("reading") or "-"), _market_read_tone(str(prop_summary.get("reading") or "-")))
                    + _chip_html(str(prop_summary.get("confidence") or "-"), _confidence_tone(str(prop_summary.get("confidence") or "-"))),
                    unsafe_allow_html=True,
                )
                st.caption(str(prop_summary.get("drivers") or "-"))

        other_rows = value_summary.get("rows")
        if isinstance(other_rows, pd.DataFrame) and not other_rows.empty:
            other_rows = other_rows[~other_rows["Mercado"].astype(str).isin(["Kills", "Torres"])].copy()
            if not other_rows.empty:
                with st.expander("Abrir outros mercados do draft", expanded=False):
                    st.dataframe(other_rows, width="stretch", hide_index=True)

    with st.expander("Abrir leitura estrutural do draft", expanded=False):
        if comp_compare:
            _render_comp_summary(
                st,
                team_a=str(team_a),
                team_b=str(team_b),
                comp_a=comp_a,
                comp_b=comp_b,
                comp_compare=comp_compare,
            )
        for reason in list(verdict.get("reasons") or []):
            st.markdown(f"- {reason}")

    with st.expander("Abrir matchups de lane", expanded=False):
        if lane_pairs:
            summary_rows: list[dict[str, Any]] = []
            for row_a, row_b in lane_pairs:
                summary_rows.append(
                    {
                        "Lane": str(row_a.get("pos") or "-").upper(),
                        team_a: f"{row_a.get('player')} | {row_a.get('champ')}",
                        team_b: f"{row_b.get('player')} | {row_b.get('champ')}",
                        "Leitura": str(row_a.get("lane_read") or _lane_edge_text(row_a.get("edge_pp"), str(team_a), str(team_b))),
                        "Plano do spot": f"{team_a}: {row_a.get('team_plan', '-')} | {team_b}: {row_b.get('team_plan', '-')}",
                        "Confianca": str(row_a.get("lane_confidence") or "-"),
                    }
                )
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
        elif isinstance(lane_view, pd.DataFrame) and not lane_view.empty:
            st.dataframe(lane_view, width="stretch", hide_index=True)
        else:
            st.caption("Sem base suficiente de lane matchups.")

    with st.expander("Duos / Trios (quando valem)", expanded=False):
        combo_cols = st.columns(2)
        with combo_cols[0]:
            st.markdown(f"**{team_a}**")
            if top_duos_a or top_trios_a:
                for row in (top_duos_a[:1] + top_trios_a[:1]):
                    prefix = "Duo" if row in top_duos_a[:1] else "Trio"
                    st.markdown(
                        f"{prefix}: **{row['combo']}** | WR {row['wr']}% | Score {row['score_wr']}% | Jogos {row['games']} | {row['confidence']}"
                    )
            else:
                st.caption("Sem base suficiente de duos/trios desse lado.")
            if isinstance(duos_a, pd.DataFrame) and not duos_a.empty:
                with st.expander("Abrir bruto de duos", expanded=False):
                    st.dataframe(duos_a, width="stretch", hide_index=True)
            if isinstance(trios_a, pd.DataFrame) and not trios_a.empty:
                with st.expander("Abrir bruto de trios", expanded=False):
                    st.dataframe(trios_a, width="stretch", hide_index=True)
        with combo_cols[1]:
            st.markdown(f"**{team_b}**")
            if top_duos_b or top_trios_b:
                for row in (top_duos_b[:1] + top_trios_b[:1]):
                    prefix = "Duo" if row in top_duos_b[:1] else "Trio"
                    st.markdown(
                        f"{prefix}: **{row['combo']}** | WR {row['wr']}% | Score {row['score_wr']}% | Jogos {row['games']} | {row['confidence']}"
                    )
            else:
                st.caption("Sem base suficiente de duos/trios desse lado.")
            if isinstance(duos_b, pd.DataFrame) and not duos_b.empty:
                with st.expander("Abrir bruto de duos", expanded=False):
                    st.dataframe(duos_b, width="stretch", hide_index=True)
            if isinstance(trios_b, pd.DataFrame) and not trios_b.empty:
                with st.expander("Abrir bruto de trios", expanded=False):
                    st.dataframe(trios_b, width="stretch", hide_index=True)

    # ---- Live state-to-win panel (Phase 6) ----------------------------
    # Calibrated P(winner) using the trained LightGBM live models.
    # Renders in a collapsed expander to keep the page tidy.
    try:
        from app_core.live_state_panel_ui_core import render_live_state_panel
        # Try to extract a pre-game prediction for team_a from context.
        p_map_pre_val = None
        try:
            p_map_pre_val = (
                ctx.get("p_map_pre")
                if isinstance(ctx, Mapping) and ctx.get("p_map_pre") is not None
                else st.session_state.get("_p_map_used")
            )
        except Exception:
            p_map_pre_val = None
        # League / BO / playoffs context (best-effort, with safe defaults)
        try:
            # scope_meta NUNCA existiu neste modulo: a guarda locals() era
            # sempre falsa, entao a liga chegava VAZIA ao modelo ao vivo — que
            # no treino recebe a liga de verdade. Divergencia treino/servico,
            # corrigida no repo do LoL em 26/08/2026 e portada para ca quando o
            # varredor de erros do LoL apontou que aqui ficara para tras.
            scope_league = _liga_compartilhada(team_games, str(team_a), str(team_b))
        except Exception:
            scope_league = ""
        try:
            bo_val = int(ctx.get("bo", 1) or 1) if isinstance(ctx, Mapping) else 1
        except Exception:
            bo_val = 1
        try:
            playoffs_val = bool(ctx.get("is_playoffs", False)) if isinstance(ctx, Mapping) else False
        except Exception:
            playoffs_val = False

        with _box_durante_mapa, st.container(border=True):
            render_tab_section(
                st,
                title="Durante o mapa (estado → vitória)",
                subtitle="Ouro/torres/objetivos/tempo do mapa em andamento → P(vitória) calibrada ao vivo (Fase 6, a camada mais validada do projeto).",
            )
            render_live_state_panel(
                st=st, session_state=st.session_state,
                team_a=str(team_a), team_b=str(team_b),
                p_map_pre=p_map_pre_val, league=scope_league,
                bo=bo_val, is_playoffs=playoffs_val,
                comp_compare=comp_compare,
                pregame_resolution={},
                format_num_fn=format_num_fn,
            )
    except Exception as _exc_live_panel:
        try:
            from app_core.ui_notify_core import notify_exception
            notify_exception(st, _exc_live_panel,
                             context="Falha ao renderizar a predição calibrada")
        except Exception:
            pass


__all__ = [
    "build_live_bet_draft_analysis",
    "build_live_bet_live_assessment",
    "build_live_bet_market_value_summary",
    "build_live_bet_pregame_summary",
    "parse_compact_draft_tokens",
    "render_live_bet_tab",
    "resolve_compact_draft_sides",
]

