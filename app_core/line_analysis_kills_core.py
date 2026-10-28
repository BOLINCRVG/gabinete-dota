from __future__ import annotations
"""Painel de KILLS da Analise de Linha — fatiado de line_analysis_tab_core.py
em 16/08/2026.

Por que separar: o arquivo tinha 1594 linhas em apenas CINCO funcoes (275 a 395
cada). As duas de kills somam 768 e sao assunto proprio — verificado por AST que
NAO usam nada das outras tres (dependencia num sentido so), e que de fora so
_render_v3_kills_subpanel e chamado.

Ver tools/superficie_publica.py: rodar ANTES de fatiar, para nao remover
re-export em uso.
"""
"""Análise de Linha tab renderer.

Shows how a specific Over/Under line behaves when a target team
plays each opponent in the league, plus inverse context (how those
same opponents perform against other teams) and a league-wide reference.
"""

import math
from html import escape
from typing import Any, Callable, MutableMapping

import pandas as pd

from app_core.market import parse_lines_text, parse_time_lines_text
from app_core.ui_design_core import cyan_section_header


# Camada de dados/config movida p/ line_analysis_data_core (fatiamento 18/07);
# re-exportada aqui p/ compat dos usos internos e importadores externos.
from app_core.line_analysis_data_core import (  # noqa: F401 - re-export
    _MARKET_KEYS,
    _METRIC_COL,
    _MARKETS_V2_DEFAULT,
    _safe_pct,
    _get_metric_col,
)


# ============================================================================
# V3 view: sub-tabs per market (Kills primeiro como protótipo)
# ============================================================================

# Camada de DADOS/config movida p/ line_analysis_data_core (fatiamento 18/07);
# re-exportada aqui para os usos internos e o import externo (radar_tab_core)
# continuarem achando os nomes sem alteração.
from app_core.line_analysis_data_core import (  # noqa: F401,E402 - re-export
    _augment_with_total_columns,
    _V3_MARKETS,
)

# Do fatiamento de 18/07 (line_analysis_panels_core). O bloco original ficava
# no MEIO do arquivo, entre funcoes — por isso nao veio junto quando copiei o
# cabecalho do topo em 16/08. Import de modulo pode estar em qualquer lugar.
from app_core.line_analysis_panels_core import (  # noqa: F401 - re-export
    _render_balanced_line_analysis,
)


def adicionar_linhas(session_state, lines_key: str, txt_key: str, is_time: bool) -> None:
    """Callback do "Adicionar linha(s)" (26/10/2026): soma as linhas digitadas a lista.

    Callback e nao "if submitted": o clique so vale para UMA execucao do script;
    clicado com o app ainda processando (reruns em cadeia ao abrir o jogo), essa
    execucao era cortada antes do bloco das linhas e a linha sumia -- a caixa
    limpava e nada entrava. O callback roda ANTES do script. Reproduzido e
    corrigido num Chrome de verdade (1 linha -> 1; com o callback, 2)."""
    txt = str(session_state.get(txt_key) or "").strip()
    if not txt:
        return
    parsed = parse_time_lines_text(txt) if is_time else parse_lines_text(txt)
    atual = [float(x) for x in (session_state.get(lines_key) or []) if isinstance(x, (int, float))]
    merged = list(atual)
    for v in parsed:
        if not any(abs(float(v) - float(x)) < 1e-9 for x in merged):
            merged.append(float(v))
    if merged != atual:
        # Sorted ascending so a smaller line sits to the left.
        session_state[lines_key] = sorted(merged)


def _render_kills_line_content(
    *,
    st: Any,
    session_state: MutableMapping[str, Any],
    line_val: float,
    line_idx: int,
    market_label: str,
    market_key: str,
    metric_col: str,
    team_a: str,
    team_b: str,
    df_a_context_all: pd.DataFrame,
    df_b_context_all: pd.DataFrame,
    df_a_context_season: pd.DataFrame,
    df_b_context_season: pd.DataFrame,
    df_h2h_year: pd.DataFrame,
    df_h2h_full: pd.DataFrame,
    df_h2h_year_b: pd.DataFrame | None = None,
    df_a_all_maps: pd.DataFrame,
    df_b_all_maps: pd.DataFrame,
    df_a_full_year: pd.DataFrame = None,
    df_b_full_year: pd.DataFrame = None,
    sinais_tg: pd.DataFrame = None,
    league_scope_for_panel: pd.DataFrame,
    league_scope_a: pd.DataFrame = None,
    league_scope_b: pd.DataFrame = None,
    cross_league: bool = False,
    # Quando cross_league e True porque o RECORTE esvaziou (e nao porque
    # as ligas sao mesmo diferentes), isto vem True e o rotulo muda.
    mesma_liga: bool = False,
    xl_league_a: str = "",
    xl_league_b: str = "",
    num_fmt: Callable[[Any, int], str],
    mmss: Callable[[float], str],
    v3_year: str,
    v3_league: str,
    v3_split: str,
    map_choice: str,
    league_label: str,
    p_map: float | None,
    lines_key: str,
    step_val: float,
    metric_label_short: str = "Kills",
    for_col: str | None = "kills_for",
    against_col: str | None = "kills_against",
    metric_decimals: int = 1,
    is_time_metric: bool = False,
    binary_market: bool = False,
) -> None:
    """Render the full per-line panel content for one specific line value."""
    from app_core.market_v3_analysis_core import (
        build_alerts,
        build_contextual_opponent_profile,
        build_market_stats_table,
        build_recent_games,
        classify_team_archetype,
        compute_h2h_stats,
        compute_map_stats,
        compute_window_stats,
    )
    from app_core.market_v3_consistency_panel_core import render_v3_market_context_panel
    from app_core.market_v3_panel_ui_core import (
        render_alerts_block,
        render_contextual_opponent_profile,
        render_map_block,
        render_v3_stats_table,
    )

    # --------- Linha desta sub-aba (somente exibição) ---------
    # Sem editor +/- (number_input): ele dava a opção de aumentar/diminuir que o
    # usuário não quer E, por ser widget dentro de st.tabs, era resetado ao
    # trocar de aba — corrompendo a lista (a linha "virava" o padrão e duplicava).
    # Agora a linha é só mostrada; para outra linha, use "Adicionar" no topo.
    if bool(binary_market):
        # Marcação sim/não (ex.: Megacreeps no Dota, pedido 2026-07-11): não há
        # linha a exibir/remover — verde = aconteceu, vermelho = não.
        st.markdown(
            f"<div style='font-size:0.78em;opacity:0.6;text-transform:uppercase;letter-spacing:0.04em;'>"
            f"Marcação {escape(str(market_label))}</div>"
            f"<div style='font-weight:800;font-size:1.15em;color:#2daac4;'>aconteceu? (sim/não)</div>",
            unsafe_allow_html=True,
        )
    else:
        _lv_disp = mmss(float(line_val)) if is_time_metric else num_fmt(float(line_val), metric_decimals)
        edit_cols = st.columns([1.4, 1, 4])
        with edit_cols[0]:
            st.markdown(
                f"<div style='font-size:0.78em;opacity:0.6;text-transform:uppercase;letter-spacing:0.04em;'>"
                f"Linha {escape(str(market_label))}</div>"
                f"<div style='font-weight:800;font-size:1.3em;color:#2daac4;'>{escape(str(_lv_disp))}</div>",
                unsafe_allow_html=True,
            )
        with edit_cols[1]:
            if st.button("✕ Remover esta linha",
                           key=f"{lines_key}__rm_{line_idx}",
                           help="Remove esta linha da lista (volta pra padrão se for a última)",
                           disabled=(len(session_state.get(lines_key, [])) <= 1)):
                current = list(session_state.get(lines_key) or [])
                if 0 <= line_idx < len(current):
                    current.pop(line_idx)
                if not current:
                    current = [float(line_val)]
                session_state[lines_key] = current
                st.rerun()
        with edit_cols[2]:
            st.caption("Para analisar outra linha, adicione no campo do topo (aceita várias: 27.5, 28.5, ...).")

    def _cyan_section_header(title: str) -> None:
        cyan_section_header(st, title)

    # ---------- 3-card panel (OVERALL / RECENT FORM / PAST FACEOFFS) ----------
    render_v3_market_context_panel(
        st,
        panel_title=market_label,
        col_name=metric_col,
        team_a=team_a, team_b=team_b,
        df_a_context_all=df_a_context_all,
        df_b_context_all=df_b_context_all,
        df_a_context_season=df_a_context_season,
        df_b_context_season=df_b_context_season,
        df_h2h_year=df_h2h_year,
        df_h2h_context_year_full_a=df_h2h_full,
        df_h2h_year_b=df_h2h_year_b,
        league_scope_df=league_scope_for_panel,
        line=float(line_val),
        min_to_mmss_fn=mmss,
        num_formatter=num_fmt,
        # Handicap: a linha digitada (ex.: -7.5) vira cobertura POR TIME nos 3
        # cards (cobriu X/N com barra), em vez de leitura de total.
        handicap=(str(market_key) == "handicaps"),
        binary=bool(binary_market),
    )

    # ---------- Análise da linha ----------
    _cyan_section_header(f"Análise da linha · {float(line_val):g}")
    # "Ano todo" window uses each team's full-year df (all leagues, no filter);
    # L5/L10/L20 use the selected recorte. See compute_window_stats.
    stats_a = compute_window_stats(df_a_context_all, metric_col=metric_col, line=float(line_val),
                                    full_year_df=df_a_full_year, fallback_df=df_a_context_season)
    stats_b = compute_window_stats(df_b_context_all, metric_col=metric_col, line=float(line_val),
                                    full_year_df=df_b_full_year, fallback_df=df_b_context_season)
    h2h_w = compute_h2h_stats(df_a_context_all, team_b, metric_col=metric_col, line=float(line_val))
    _render_balanced_line_analysis(
        st=st, team_a=team_a, team_b=team_b,
        stats_a=stats_a, stats_b=stats_b, h2h=h2h_w,
        line=float(line_val),
    )
    # OS PASSOS DELE (28-29/10/2026): Medias, Adversario, Historico 5/20/10, Stats da linha --
    # Menos | Mais, com A FAVOR/CONTRA, chance, odd das casas dele e veredito. So LoL; em todos
    # os TOTAIS (kills, torres, dragoes, baroes, inibidores, tempo), cada um com os pesos
    # medidos no proprio mercado (app_core/sinais_linha_core.py); no Dota o modulo nao existe.
    try:
        from app_core.sinais_linha_core import COLUNA_PARA_METRICA as _COL_MET
        from app_core.seus_sinais_tab_core import render_painel_na_analise
    except ImportError:
        _COL_MET, render_painel_na_analise = {}, None
    # ML + TOTAL (29/10/2026): a odd justa da COMBINADA, com o total lido dado quem vence
    # onde isso foi medido (tempo, torres, dragoes). So LoL (app_core/combinadas_core.py).
    if str(market_key) == "ml_totals" and sinais_tg is not None:
        try:
            from app_core.combinadas_core import render_combinadas
        except ImportError:
            render_combinadas = None
        if render_combinadas is not None:
            _cyan_section_header(f"Combinadas · vence + total · {float(line_val):g}")
            try:
                render_combinadas(st=st, tg=sinais_tg, time_a=team_a, time_b=team_b, linha=float(line_val),
                                  p_vence_a=p_map, chave=f"{team_a}_{team_b}_{line_idx}")
            except Exception as _exc:
                st.caption(f"Combinadas: nao deu para calcular ({type(_exc).__name__}).")
    _met = _COL_MET.get(str(metric_col))
    if _met and sinais_tg is not None and str(market_key) not in ("handicaps", "ml_totals"):
        if render_painel_na_analise is not None:
            _cyan_section_header(f"Seus passos · {float(line_val):g}")
            try:
                render_painel_na_analise(st=st, tg=sinais_tg, time_a=team_a, time_b=team_b,
                                         linha=float(line_val), chave=f"{_met}_{team_a}_{team_b}_{line_idx}",
                                         metrica=_met)
            except Exception as _exc:
                st.caption(f"Seus passos: nao deu para calcular ({type(_exc).__name__}).")

    # ---------- Por mapa (panorâmica) ----------
    overall_mean_a_all = float(pd.to_numeric(df_a_all_maps[metric_col], errors="coerce").dropna().mean()) \
        if metric_col in df_a_all_maps.columns else float("nan")
    overall_mean_b_all = float(pd.to_numeric(df_b_all_maps[metric_col], errors="coerce").dropna().mean()) \
        if metric_col in df_b_all_maps.columns else float("nan")
    maps_a = compute_map_stats(df_a_all_maps, metric_col=metric_col, line=float(line_val),
                                 overall_mean=overall_mean_a_all)
    maps_b = compute_map_stats(df_b_all_maps, metric_col=metric_col, line=float(line_val),
                                 overall_mean=overall_mean_b_all)
    if maps_a or maps_b:
        _cyan_section_header("Por mapa (M1 a M5 — sempre comparativo)")
        if maps_a:
            render_map_block(st=st, team_name=team_a, map_stats=maps_a,
                              line=float(line_val), line_label=f"{float(line_val):g}",
                              is_time_metric=bool(is_time_metric))
        if maps_b:
            render_map_block(st=st, team_name=team_b, map_stats=maps_b,
                              line=float(line_val), line_label=f"{float(line_val):g}",
                              is_time_metric=bool(is_time_metric))

    # ---------- Perfil contextual ----------
    # Each team's archetype + opponent pool come from its OWN league scope.
    # Normal mode: both are the same single scope. Cross-league: LCK for A, LPL for B.
    _scope_a = league_scope_a if league_scope_a is not None else league_scope_for_panel
    _scope_b = league_scope_b if league_scope_b is not None else league_scope_for_panel
    archetype_a = classify_team_archetype(
        team_name=team_a, team_df=df_a_context_all,
        league_scope_df=_scope_a, metric_col=metric_col,
    )
    archetype_b = classify_team_archetype(
        team_name=team_b, team_df=df_b_context_all,
        league_scope_df=_scope_b, metric_col=metric_col,
    )
    _cyan_section_header("Perfil contextual · vs grupo de adversários (selecionável)")
    # Per-team opponent-group selector. Default = the OPPONENT's archetype group:
    # each team is shown against opponents like the one it will actually face in
    # this matchup (TT vs LGD's group, LGD vs TT's group). User can switch.
    from app_core.market_v3_analysis_core import (
        OPPONENT_GROUP_CHOICES,
        archetypes_for_group_choice,
        default_group_label_for_archetype,
    )
    _def_a = default_group_label_for_archetype(archetype_b.archetype)  # vs opponent B's group
    _def_b = default_group_label_for_archetype(archetype_a.archetype)  # vs opponent A's group
    # Semear SÓ via session_state (sem index=): default + seed juntos disparam
    # o warning "created with a default value but also had its value set".
    _key_ga = f"la_oppgrp_a_{market_key}_{line_idx}"
    _key_gb = f"la_oppgrp_b_{market_key}_{line_idx}"
    if str(session_state.get(_key_ga, "") or "") not in OPPONENT_GROUP_CHOICES:
        session_state[_key_ga] = _def_a if _def_a in OPPONENT_GROUP_CHOICES else OPPONENT_GROUP_CHOICES[2]
    if str(session_state.get(_key_gb, "") or "") not in OPPONENT_GROUP_CHOICES:
        session_state[_key_gb] = _def_b if _def_b in OPPONENT_GROUP_CHOICES else OPPONENT_GROUP_CHOICES[2]
    _grp_cols = st.columns(2)
    with _grp_cols[0]:
        grp_a = st.selectbox(
            f"Grupo de adversário · {team_a}", OPPONENT_GROUP_CHOICES,
            key=_key_ga,
        )
    with _grp_cols[1]:
        grp_b = st.selectbox(
            f"Grupo de adversário · {team_b}", OPPONENT_GROUP_CHOICES,
            key=_key_gb,
        )
    profile_a = build_contextual_opponent_profile(
        focal_team=team_a, focal_team_df=df_a_context_all,
        league_scope_df=_scope_a,
        target_archetypes=archetypes_for_group_choice(grp_a),
        metric_col=metric_col, line=float(line_val),
    )
    profile_b = build_contextual_opponent_profile(
        focal_team=team_b, focal_team_df=df_b_context_all,
        league_scope_df=_scope_b,
        target_archetypes=archetypes_for_group_choice(grp_b),
        metric_col=metric_col, line=float(line_val),
    )
    render_contextual_opponent_profile(
        st=st, team_a=team_a, team_b=team_b,
        archetype_a=archetype_a, archetype_b=archetype_b,
        profile_a=profile_a, profile_b=profile_b,
        line=float(line_val), line_label=f"{float(line_val):g}",
        is_time_metric=bool(is_time_metric),
    )

    # ---------- Stats da linha ----------
    stats_rows = build_market_stats_table(
        team_a=team_a, team_b=team_b,
        df_a=df_a_context_all, df_b=df_b_context_all,
        league_scope_df=league_scope_for_panel,
        league_scope_a=_scope_a, league_scope_b=_scope_b,
        metric_col=metric_col, line=float(line_val),
        metric_label=metric_label_short,
        for_col=for_col, against_col=against_col,
        metric_decimals=int(metric_decimals),
        is_time_metric=bool(is_time_metric),
        cross_league=bool(cross_league),
    )
    # Cross-league: teams are from different leagues, so the only shared recorte
    # is the YEAR. Show year-only in the header/caption (each team's own league
    # appears in the two Liga columns); otherwise show the full liga·ano·split.
    if cross_league:
        _stats_header = f"Stats da linha · {v3_year}"
        _stats_scope = (
            f"{v3_year} · "
            + ("mesma liga, recorte remontado" if mesma_liga
               else "ligas diferentes (cada time na sua)")
            + f" · Mapa: {map_choice}"
        )
    else:
        _stats_header = f"Stats da linha · {v3_league} · {v3_year} · {v3_split or 'todos splits'}"
        _stats_scope = f"{v3_league} {v3_year} {v3_split or '(todos splits)'} · Mapa: {map_choice}"
    _cyan_section_header(_stats_header)
    render_v3_stats_table(
        st=st, team_a=team_a, team_b=team_b,
        rows=stats_rows,
        scope_label=_stats_scope,
        cross_league=bool(cross_league),
        league_label_a=(f"Liga · {xl_league_a}" if cross_league and xl_league_a else "Liga"),
        league_label_b=(f"Liga · {xl_league_b}" if cross_league and xl_league_b else "Liga"),
    )

    # ---------- First X: probabilidade AJUSTADA PELO CONFRONTO ----------
    # O chip de ML abaixo cobre só kills/torres/dragões/tempo; os "primeiros
    # objetivos" não tinham probabilidade de modelo — só taxa histórica.
    # Aqui entra P(team_a pega o objetivo) ajustada pela diferença de força,
    # e SÓ nos eventos DECIDIDOS (torre/baron): medido em 5.319 jogos OOS
    # (+0.78% e +1.16% de Brier); nos eventos cedo (blood/dragão) o gap é
    # ruído, então a taxa pura fica intacta. Ver first_x_gap_core.
    if str(market_key) == "first_x" and str(metric_col or ""):
        try:
            from app_core.first_x_gap_core import (
                adjust_first_x_prob,
                first_x_gap_supported,
                mu_gap_from_trace,
            )
            _s_evt = pd.to_numeric(df_a_context_all[metric_col], errors="coerce").dropna() \
                if metric_col in getattr(df_a_context_all, "columns", []) else pd.Series(dtype=float)
            if len(_s_evt) > 0:
                _p_base = float(_s_evt.mean())
                _gap = mu_gap_from_trace(session_state.get("_ml_trace_last"),
                                         team=team_a, opponent=team_b)
                _p_adj, _fx_info = adjust_first_x_prob(_p_base, event_col=metric_col, mu_gap=_gap)
                _sup = first_x_gap_supported(metric_col)
                _cyan_section_header(f"Quem pega primeiro — {team_a}")
                if bool(_fx_info.get("applied")):
                    _dpp = float(_fx_info.get("diff_pp") or 0.0)
                    _cor = "#22c55e" if _p_adj >= 0.55 else "#facc15" if _p_adj >= 0.45 else "#ef4444"
                    _nota = (f"ajuste de força vs {team_b}: {_dpp:+.1f} pp "
                             f"(gap {float(_fx_info.get('mu_gap') or 0.0):+.2f})")
                else:
                    _cor, _nota = "#9ca3af", (
                        "evento cedo: a força do time não prevê quem pega (medido ~neutro) — taxa pura"
                        if not _sup else "sem gap disponível — taxa pura")
                # Odd justa dos DOIS lados (pedido dele 22/07: "gerando uma odd,
                # isso para todos os First X"). Justa = 1/p, sem margem.
                _p_sim = float(min(max(_p_adj, 1e-6), 1.0 - 1e-6))
                _odd_sim, _odd_nao = 1.0 / _p_sim, 1.0 / (1.0 - _p_sim)
                st.markdown(
                    f"<div style='display:flex;gap:24px;background:rgba(45,170,196,0.08);"
                    f"border:1px solid rgba(45,170,196,0.3);border-radius:10px;"
                    f"padding:14px 18px;margin-top:8px;align-items:center;flex-wrap:wrap;'>"
                    f"<div><div style='opacity:0.65;font-size:0.78em;text-transform:uppercase;"
                    f"letter-spacing:0.05em;'>Taxa histórica</div>"
                    f"<div style='font-size:1.5em;font-weight:700;'>{100.0*_p_base:.1f}%</div></div>"
                    f"<div><div style='opacity:0.65;font-size:0.78em;text-transform:uppercase;"
                    f"letter-spacing:0.05em;'>P({escape(str(team_a))} pega)</div>"
                    f"<div style='font-size:1.5em;font-weight:700;color:{_cor};'>"
                    f"{100.0*_p_adj:.1f}%</div></div>"
                    f"<div><div style='opacity:0.65;font-size:0.78em;text-transform:uppercase;"
                    f"letter-spacing:0.05em;'>Odd justa · ✓ pega</div>"
                    f"<div style='font-size:1.5em;font-weight:700;color:#22c55e;'>{_odd_sim:.2f}</div></div>"
                    f"<div><div style='opacity:0.65;font-size:0.78em;text-transform:uppercase;"
                    f"letter-spacing:0.05em;'>Odd justa · ✗ não pega</div>"
                    f"<div style='font-size:1.5em;font-weight:700;color:#ef4444;'>{_odd_nao:.2f}</div></div>"
                    f"<div style='opacity:0.7;font-size:0.85em;flex-basis:100%;'>{_nota}"
                    f" · odd justa = 1/p, sem margem da casa</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        except (KeyError, TypeError, ValueError, AttributeError, ImportError):
            pass

    # ---------- ML predict chip ----------
    try:
        from app_core.live_markets_runtime_core import predict_market
        # Map market_key → Fase 7 market name. Phase 7 trained: total_kills,
        # total_towers, total_dragons, gamelength_min (barons rejected, inibs N/A).
        _market_name_map = {
            "kills": "total_kills", "towers": "total_towers",
            "dragons": "total_dragons", "time": "gamelength_min",
        }
        _ml_market_name = _market_name_map.get(market_key)
        res = predict_market(
            session_state=session_state,
            market=_ml_market_name or "total_kills",
            minute=10, line=float(line_val),
            p_map_pre=float(p_map) if p_map is not None else 0.5,
            league=str(league_label or ""), bo=1, is_playoffs=False,
        ) if _ml_market_name else {"mean": None, "p_over": None}
        mean_v = res.get("mean")
        p_over = res.get("p_over")
        if mean_v is not None and p_over is not None:
            p_under = 1.0 - float(p_over)
            _cyan_section_header("Predição ML (complemento)")
            st.markdown(
                f"<div style='display:flex;gap:24px;background:rgba(45,170,196,0.08);"
                f"border:1px solid rgba(45,170,196,0.3);border-radius:10px;"
                f"padding:14px 18px;margin-top:8px;align-items:center;'>"
                f"<div><div style='opacity:0.65;font-size:0.78em;text-transform:uppercase;letter-spacing:0.05em;'>Predição</div>"
                f"<div style='font-size:1.5em;font-weight:700;'>{float(mean_v):.1f}</div></div>"
                f"<div><div style='opacity:0.65;font-size:0.78em;text-transform:uppercase;letter-spacing:0.05em;'>P(Over {float(line_val):g})</div>"
                f"<div style='font-size:1.5em;font-weight:700;"
                f"color:{'#22c55e' if float(p_over) >= 0.55 else '#facc15' if float(p_over) >= 0.45 else '#ef4444'};'>"
                f"{float(p_over)*100:.1f}%</div></div>"
                f"<div><div style='opacity:0.65;font-size:0.78em;text-transform:uppercase;letter-spacing:0.05em;'>P(Under)</div>"
                f"<div style='font-size:1.5em;font-weight:700;'>{float(p_under)*100:.1f}%</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    except (TypeError, ValueError):
        pass

    # ---------- Alertas ----------
    recent_a = build_recent_games(df_a_context_all, league_scope_for_panel,
                                    metric_col=metric_col, line=float(line_val), last_n=10)
    recent_b = build_recent_games(df_b_context_all, league_scope_for_panel,
                                    metric_col=metric_col, line=float(line_val), last_n=10)
    alerts_a = build_alerts(team_name=team_a, window_stats=stats_a, map_stats=maps_a,
                              h2h=h2h_w, recent=recent_a, line=float(line_val),
                              overall_mean=overall_mean_a_all)
    alerts_b = build_alerts(team_name=team_b, window_stats=stats_b, map_stats=maps_b,
                              h2h=h2h_w, recent=recent_b, line=float(line_val),
                              overall_mean=overall_mean_b_all)
    all_alerts = list(alerts_a) + list(alerts_b)
    if all_alerts:
        _cyan_section_header("Alertas")
        render_alerts_block(st=st, alerts=all_alerts)




def _render_v3_kills_subpanel(
    *,
    st: Any,
    session_state: MutableMapping[str, Any],
    team_a: str, team_b: str,
    league_label: str,
    df_a: pd.DataFrame, df_b: pd.DataFrame, df_scope: pd.DataFrame,
    team_games: pd.DataFrame,
    Filters: Any,
    apply_filters_fn: Callable[[pd.DataFrame, Any], pd.DataFrame],
    v3_year: str, v3_league: str, v3_split: str,
    market_cfg: dict,
    p_map: float | None,
    p_series: float | None,
) -> None:
    """Render the Kills (and similar markets) sub-tab content with the 3-card
    panel as the primary view + ML predict chip below."""

    market_key = str(market_cfg["key"])
    market_label = str(market_cfg["label"])
    metric_col = str(market_cfg["metric_col"])
    default_line = float(market_cfg["default_line"])

    # Header
    st.markdown(
        f"<div style='font-size:1.4em;font-weight:700;padding:6px 0 4px;'>"
        f"<span style='color:#2daac4;'>{team_a}</span>"
        f" <span style='opacity:0.6;'>VS</span> "
        f"<span style='color:#2daac4;'>{team_b}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Handicap caption: explain how to read the spread values
    if bool(market_cfg.get("handicap")):
        st.markdown(
            f"<div style='background:rgba(45,170,196,0.06);border-left:3px solid #2daac4;"
            f"padding:8px 12px;border-radius:6px;margin-bottom:10px;font-size:0.9em;'>"
            f"<b>Handicap (spread de kills):</b> mostra kills feitas − kills sofridas, "
            f"do ponto de vista de cada time. Ex.: jogo 20×8 → <b style='color:#22c55e;'>+12</b> pra "
            f"{team_a} e <b style='color:#ef4444;'>−12</b> pra {team_b}. "
            f"Linha <b>0</b> = venceu o mapa (spread &gt; 0). "
            f"Linha <b>+2.5</b> = venceu por 3+ kills."
            f"</div>",
            unsafe_allow_html=True,
        )

    # First X selector — pick which "first" objective to analyze
    if bool(market_cfg.get("first_x_selector")):
        # O host pode fornecer o mapa de objetivos (o Gabinete Dota troca por
        # First Blood/Tower/Roshan); default = os 5 do LoL.
        first_x_opts = dict(market_cfg.get("first_x_opts") or {
            "First Blood":   "firstblood",
            "First Tower":   "firsttower",
            "First Dragon":  "firstdragon",
            "First Herald":  "firstherald",
            "First Baron":   "firstbaron",
        })
        fx_key = f"v3_firstx_choice_{team_a}_{team_b}"
        labels = list(first_x_opts.keys())
        if session_state.get(fx_key) not in labels:
            session_state[fx_key] = labels[0]
        chosen_label = st.radio(
            "Qual primeiro objetivo?", labels, horizontal=True, key=fx_key,
            help="Cada First X é um mercado binário (1 = conseguiu o objetivo).",
        )
        metric_col = first_x_opts[chosen_label]
        market_label = chosen_label  # update label for downstream rendering
        # Override config for the helper.
        # binary_market=True (pedido dele 22/07): First X é marcação SIM/NÃO —
        # "quem pegou primeiro". Reusa a máquina já pronta do binário (feita p/
        # Megacreeps): ✓ verde / ✗ vermelho por jogo em vez de 0/1, legenda
        # "aconteceu / não aconteceu", e some o editor de linha (a "linha 0.5"
        # não significa nada num evento binário).
        market_cfg = {**market_cfg,
                       "metric_col": metric_col,
                       "metric_label": chosen_label,
                       "label": chosen_label,
                       "default_line": 0.5,
                       "for_col": None, "against_col": None,
                       "binary_market": True,
                       "metric_decimals": 0}

    # --------- Map selector (shared across all lines for this market) ---------
    step_val = 0.5 if market_key == "time" else 1.0
    map_opts = ["Todos", "M1", "M2", "M3", "M4", "M5"]
    map_key = f"v3kills_map_{market_key}_{team_a}_{team_b}"
    if session_state.get(map_key) not in map_opts:
        session_state[map_key] = map_opts[0]

    # --------- Lines list — supports multiple lines as nested tabs ----------
    lines_key = f"v3kills_lines_{market_key}_{team_a}_{team_b}"
    if lines_key not in session_state or not isinstance(session_state.get(lines_key), list) or not session_state[lines_key]:
        # A LINHA DO JOGO, nao a padrao (24/10/2026): a principal da Pinnacle
        # para este confronto; sem ela, a media das medias dos dois times.
        # Ver app_core/linha_sugerida_core.py.
        _linha0, _origem0 = float(default_line), ""
        if not bool(market_cfg.get("binary_market")) and not bool(market_cfg.get("handicap")):
            try:
                from app_core.linha_sugerida_core import linha_inicial, serie_recente
                _sa = serie_recente(team_games, str(team_a), market_key)
                _sb = serie_recente(team_games, str(team_b), market_key)
                _linha0, _origem0 = linha_inicial(market_key, str(team_a), str(team_b), _sa, _sb,
                                                  float(default_line))
            except Exception:
                _linha0, _origem0 = float(default_line), ""
        session_state[lines_key] = [float(_linha0)]
        session_state[lines_key + "_origem"] = _origem0
    # Defensive: ensure all entries are floats
    lines_list: list[float] = [float(x) for x in session_state[lines_key] if isinstance(x, (int, float))]
    if not lines_list:
        lines_list = [float(default_line)]
    session_state[lines_key] = lines_list

    # Mercado BINÁRIO (marcação sim/não — ex.: Megacreeps no Dota, pedido
    # 2026-07-11): não existe "linha" a escolher — é fixa em 0.5 e o editor
    # de linhas some. Todo o resto (janelas, % , selos) segue igual.
    if bool(market_cfg.get("binary_market")):
        session_state[lines_key] = [float(default_line)]
        lines_list = [float(default_line)]

    _is_time = (market_key == "time")
    lc1, lc2, lc3 = st.columns([1.6, 2.6, 3])
    with lc1:
        st.markdown(
            f"<div style='padding-top:6px;opacity:0.7;font-size:0.88em;'>"
            f"<b>{len(lines_list)}</b> linha(s) ativa(s) · adicione mais se quiser comparar várias."
            f"</div>",
            unsafe_allow_html=True,
        )
        _origem_txt = str(session_state.get(lines_key + "_origem") or "")
        if _origem_txt:
            st.caption("Linha inicial: {}.".format(_origem_txt))
    with lc2:
        if bool(market_cfg.get("binary_market")):
            st.markdown(
                "<div style='padding-top:6px;opacity:0.7;font-size:0.88em;'>"
                "Marcação <b>sim/não</b> — sem linha para escolher.</div>",
                unsafe_allow_html=True,
            )
        else:
            # Add lines via a FORM: nothing is analyzed until you click — typing in
            # the box does NOT trigger a re-run. Accepts several lines separated by
            # comma (27.5, 28.5, 29.5), decimal with dot. Time also takes 33:30 / 33.5.
            # Callback (adicionar_linhas), nao "if submitted": ver a docstring dela.
            _txt_key = f"{lines_key}__novo"

            with st.form(key=f"{lines_key}__addform", clear_on_submit=True):
                _ph = "ex: 33:30, 34.5" if _is_time else "ex: 27.5, 28.5, 29.5"
                st.text_input(
                    "Adicionar linha(s)",
                    value="",
                    key=_txt_key,
                    placeholder=_ph,
                    help=("Várias de uma vez separadas por vírgula. Decimal com ponto. "
                          + ("Tempo: 33:30 ou 33.5 (minutos)." if _is_time else "Ex.: 27.5, 28.5.")),
                )
                st.form_submit_button("➕ Adicionar linha(s)", width="stretch", on_click=adicionar_linhas,
                                      args=(session_state, lines_key, _txt_key, _is_time))
    with lc3:
        st.markdown("<div style='padding-top:6px;'></div>", unsafe_allow_html=True)
        map_choice = st.radio(
            "Mapa", map_opts, horizontal=True, key=map_key,
            help="Filtra TODAS as seções abaixo. Use 'Todos' pra visão geral.",
        )

    # ---- Cross-league detection: teams from different leagues ----
    # If the selected scope leaves one team empty, evaluate EACH team in its OWN
    # domestic league at the most-recent split (excluding short international
    # events that inflate averages). Opponents/league-ref come from each team's
    # own league. See resolve_team_home_scope.
    _cross_league = bool(df_a is None or df_a.empty or df_b is None or df_b.empty)
    _xl_scope_a = None
    _xl_scope_b = None
    _xl_league_a = ""
    _xl_league_b = ""
    _mesma_liga = False
    if _cross_league:
        from app_core.market_v3_analysis_core import resolve_team_home_scope
        try:
            _yr = int(v3_year)
        except (TypeError, ValueError):
            _yr = None
        _full = _augment_with_total_columns(team_games.copy())
        _ra = resolve_team_home_scope(_full, team_a, year=_yr)
        _rb = resolve_team_home_scope(_full, team_b, year=_yr)
        if _ra["team_df"].empty or _rb["team_df"].empty:
            st.info("Um dos times não tem jogos de liga doméstica neste ano.")
            return
        df_a = _ra["team_df"].copy()
        df_b = _rb["team_df"].copy()
        _xl_scope_a = _ra["league_scope"]
        _xl_scope_b = _rb["league_scope"]
        _xl_league_a = _ra["league"]
        _xl_league_b = _rb["league"]
        # NAO presumir "ligas diferentes" so porque o recorte esvaziou:
        # ver descrever_recorte_por_time.
        from app_core.market_v3_analysis_core import descrever_recorte_por_time
        _desc = descrever_recorte_por_time(_ra, _rb, team_a, team_b)
        _mesma_liga = bool(_desc["mesma_liga"])
        st.info(_desc["texto"])

    # ---- Apply Mapa filter to all DataFrames (keep unfiltered for panoramic "Por mapa" card) ----
    # The column with map order is `map_number` after build_team_games() in
    # mlcore/stats_teams.py (some upstream sources call it `game`). Check both.
    def _filter_by_map(df: pd.DataFrame) -> pd.DataFrame:
        if map_choice == "Todos" or df is None or df.empty:
            return df
        map_col = "map_number" if "map_number" in df.columns else ("game" if "game" in df.columns else None)
        if map_col is None:
            return df
        try:
            target = int(map_choice[1:])
        except (ValueError, IndexError):
            return df
        return df[pd.to_numeric(df[map_col], errors="coerce").fillna(0).astype(int) == target].copy()

    # Keep originals for the "Por mapa" panoramic card (always shows all 5 maps)
    df_a_all_maps = df_a.copy()
    df_b_all_maps = df_b.copy()

    # Full-year df per team (NO league/split filter) — feeds the "Ano todo"
    # window. = everything the team played in the year, across all leagues/
    # tournaments (LCK + EWC + MSI + ...). Map filter still applies.
    try:
        _yr_int = int(v3_year)
    except (TypeError, ValueError):
        _yr_int = None
    _full_aug = _augment_with_total_columns(team_games.copy()) if team_games is not None else pd.DataFrame()
    def _team_full_year(tname: str) -> pd.DataFrame:
        if _full_aug is None or _full_aug.empty or "team" not in _full_aug.columns:
            return pd.DataFrame()
        sub = _full_aug[_full_aug["team"].astype(str) == str(tname)]
        if _yr_int is not None and "year" in sub.columns:
            sub = sub[pd.to_numeric(sub["year"], errors="coerce") == _yr_int]
        return _filter_by_map(sub.copy())
    df_a_full_year = _team_full_year(team_a)
    df_b_full_year = _team_full_year(team_b)

    # Season-in-own-league (all splits of the year, single domestic league) —
    # used for the "Season inteira" card in cross-league matchups, where the
    # selected v3_league only contains ONE of the two teams.
    def _team_year_in_league(tname: str, league_name: str) -> pd.DataFrame:
        if _full_aug is None or _full_aug.empty or "team" not in _full_aug.columns:
            return pd.DataFrame()
        sub = _full_aug[_full_aug["team"].astype(str) == str(tname)]
        if league_name and "league" in sub.columns:
            sub = sub[sub["league"].astype(str) == str(league_name)]
        if _yr_int is not None and "year" in sub.columns:
            sub = sub[pd.to_numeric(sub["year"], errors="coerce") == _yr_int]
        return _filter_by_map(sub.copy())

    df_a = _filter_by_map(df_a)
    df_b = _filter_by_map(df_b)
    df_scope = _filter_by_map(df_scope)
    if df_a.empty or df_b.empty:
        st.info(f"Sem jogos pra {map_choice} no recorte. Tente outro mapa ou 'Todos'.")
        return

    # Build the contexts the panel needs
    #   df_a_context_all: A's games in the current split (or current scope)
    #   df_a_context_season: A's games in the full year (no split filter)
    df_a_context_all = df_a.copy()
    df_b_context_all = df_b.copy()

    # Season-wide (no split) view: same year+league, drop split filter
    try:
        season_filters = Filters(year=int(v3_year), league=(None if v3_league == "S/Camp" else v3_league), split=None, playoffs=None)
        df_season_scope = apply_filters_fn(team_games, season_filters).copy()
        df_season_scope = _augment_with_total_columns(df_season_scope)
        df_season_scope = _filter_by_map(df_season_scope)  # apply Mapa filter
        df_a_context_season = df_season_scope[df_season_scope["team"].astype(str) == str(team_a)].copy()
        df_b_context_season = df_season_scope[df_season_scope["team"].astype(str) == str(team_b)].copy()
    except (TypeError, ValueError, KeyError):
        df_a_context_season = df_a_context_all
        df_b_context_season = df_b_context_all
        df_season_scope = df_scope

    # Cross-league: the single v3_league scope only holds ONE of the two teams,
    # so the other team's "Season inteira" came back empty (0 games). Rebuild
    # each team's season from its OWN domestic league (all splits of the year).
    if _cross_league:
        if _xl_league_a:
            df_a_context_season = _team_year_in_league(team_a, _xl_league_a)
        if _xl_league_b:
            df_b_context_season = _team_year_in_league(team_b, _xl_league_b)

    # H2H: A's games vs B — do ANO INTEIRO em TODAS as ligas. H2H é par-a-par:
    # derivar do recorte de liga escondia confrontos jogados em torneio fora da
    # liga doméstica (visto 2026-07-13: Vici×PlayTime com 2 H2H reais e o card
    # dizendo "sem H2H" no cross-league). Mesma regra do usuário no Resumo.
    try:
        _h2h_base = _augment_with_total_columns(team_games.copy())
        try:
            _h2h_base = _h2h_base[pd.to_numeric(_h2h_base.get("year"), errors="coerce") == int(v3_year)]
        except (TypeError, ValueError):
            pass
        df_h2h_year = _h2h_base[
            (_h2h_base["team"].astype(str) == str(team_a))
            & (_h2h_base["opponent"].astype(str) == str(team_b))
        ].copy()
        df_h2h_year = _filter_by_map(df_h2h_year)
    except (TypeError, ValueError, KeyError):
        df_h2h_year = pd.DataFrame()
    df_h2h_full = df_h2h_year.copy()

    # O MESMO recorte visto do lado do time B (espelho). Pedido dele em
    # 15/09/2026: o card "Past Faceoffs" mostrava so a coluna do time A, e
    # "Gen.G fez 26" sem saber quantas o adversario fez nao diz se foi jogo de
    # sangue ou atropelo. Mesmos filtros, mesmo ano, mesmo mapa — senao os
    # dois lados falariam de recortes diferentes.
    try:
        df_h2h_year_b = _h2h_base[
            (_h2h_base["team"].astype(str) == str(team_b))
            & (_h2h_base["opponent"].astype(str) == str(team_a))
        ].copy()
        df_h2h_year_b = _filter_by_map(df_h2h_year_b)
    except (TypeError, ValueError, KeyError, NameError):
        df_h2h_year_b = pd.DataFrame()

    # Also re-filter context dataframes defensively (Streamlit reruns can lead
    # to stale references; this guarantees the panel never sees mixed-map data).
    df_a_context_all = _filter_by_map(df_a_context_all)
    df_b_context_all = _filter_by_map(df_b_context_all)
    df_a_context_season = _filter_by_map(df_a_context_season)
    df_b_context_season = _filter_by_map(df_b_context_season)

    def _num_fmt(value: Any, decimals: int = 0) -> str:
        try:
            v = float(value)
            if not math.isfinite(v):
                return "—"
            return f"{v:.{int(decimals)}f}"
        except (TypeError, ValueError):
            return "—"

    def _mmss(v: float) -> str:
        try:
            v = float(v)
            if not math.isfinite(v):
                return "—"
            mins = int(v); secs = round((v - mins) * 60)
            return f"{mins}:{secs:02d}"
        except (TypeError, ValueError):
            return "—"

    league_scope_for_panel = df_season_scope if "df_season_scope" in locals() else df_scope
    # Safety: re-apply map filter once more before passing into the panel
    league_scope_for_panel = _filter_by_map(league_scope_for_panel)

    # Per-team league scope: in cross-league mode each team uses its OWN league
    # pool (LCK for T1, LPL for AL); otherwise both share the single scope.
    if _cross_league and _xl_scope_a is not None and _xl_scope_b is not None:
        league_scope_a = _filter_by_map(_augment_with_total_columns(_xl_scope_a.copy()))
        league_scope_b = _filter_by_map(_augment_with_total_columns(_xl_scope_b.copy()))
    else:
        league_scope_a = league_scope_for_panel
        league_scope_b = league_scope_for_panel
    # Drop teams that aren't currently competing (roster-deleted, moved league,
    # etc.). Activity is measured against df_scope (the current split), with a
    # 21-day recency window from its latest game. Keeps focal teams whitelisted
    # so the matchup itself never disappears.
    try:
        from app_core.market_v3_analysis_core import (
            active_teams_in_recent_window,
            filter_to_active_teams,
        )
        _active = active_teams_in_recent_window(df_scope, days=21)
        if _active:
            _active.add(str(team_a)); _active.add(str(team_b))
            league_scope_for_panel = filter_to_active_teams(league_scope_for_panel, _active)
    except (ImportError, TypeError, ValueError, KeyError):
        pass

    # --------- Render: tabs per line OR single render if only one line ---------
    panel_ctx = dict(
        market_label=market_label, market_key=market_key, metric_col=metric_col,
        team_a=team_a, team_b=team_b,
        df_a_context_all=df_a_context_all, df_b_context_all=df_b_context_all,
        df_a_context_season=df_a_context_season, df_b_context_season=df_b_context_season,
        df_h2h_year=df_h2h_year, df_h2h_full=df_h2h_full,
        df_h2h_year_b=df_h2h_year_b,
        df_a_all_maps=df_a_all_maps, df_b_all_maps=df_b_all_maps,
        df_a_full_year=df_a_full_year, df_b_full_year=df_b_full_year,
        sinais_tg=_full_aug,
        league_scope_for_panel=league_scope_for_panel,
        league_scope_a=league_scope_a, league_scope_b=league_scope_b,
        cross_league=_cross_league,
        mesma_liga=_mesma_liga,
        xl_league_a=_xl_league_a, xl_league_b=_xl_league_b,
        num_fmt=_num_fmt, mmss=_mmss,
        v3_year=v3_year, v3_league=v3_league, v3_split=v3_split,
        map_choice=map_choice, league_label=league_label,
        p_map=p_map, lines_key=lines_key,
        step_val=step_val,
        # Market-specific config (defaults to Kills shape if missing)
        metric_label_short=str(market_cfg.get("metric_label", market_label)),
        for_col=market_cfg.get("for_col", "kills_for"),
        against_col=market_cfg.get("against_col", "kills_against"),
        metric_decimals=int(market_cfg.get("metric_decimals", 1)),
        is_time_metric=bool(market_cfg.get("is_time_metric", False)),
        binary_market=bool(market_cfg.get("binary_market", False)),
    )

    if len(lines_list) == 1:
        # Single line — render directly without nested tab overhead
        _render_kills_line_content(
            st=st, session_state=session_state,
            line_val=float(lines_list[0]), line_idx=0,
            **panel_ctx,
        )
    else:
        tab_labels = [f"Linha {l:g}" for l in lines_list]
        tab_refs = st.tabs(tab_labels)
        for idx, (ref, line_val) in enumerate(zip(tab_refs, lines_list)):
            with ref:
                _render_kills_line_content(
                    st=st, session_state=session_state,
                    line_val=float(line_val), line_idx=int(idx),
                    **panel_ctx,
                )
    return  # All per-line content rendered via helper; legacy below is dead.
