"""Análise de Linha tab renderer.

Shows how a specific Over/Under line behaves when a target team
plays each opponent in the league, plus inverse context (how those
same opponents perform against other teams) and a league-wide reference.
"""
from __future__ import annotations

import math
from html import escape
from typing import Any, Callable, MutableMapping, Optional

import pandas as pd

from app_core.market import parse_lines_text, parse_time_lines_text
from app_core.shared_formatters import fmt_delta_pp, fmt_num, fmt_pct, fmt_status
from app_core.ui_design_core import cyan_banner, cyan_section_header
from app_core.tab_filters_core import collect_year_split_league_options, scope_has_playoffs
from app_core.tab_state_core import normalize_select_key, select_placeholder
from app_core.tab_ui_core import render_tab_hero, render_tab_section

from app_core.championship_stats_core import STATS_MARKET_CONFIG

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


def _render_v3_view(
    *,
    st: Any,
    team_games: pd.DataFrame,
    Filters: Any,
    apply_filters_fn: Callable[[pd.DataFrame, Any], pd.DataFrame],
    session_state: MutableMapping[str, Any],
    resumo_renderer: Optional[Callable[[], None]] = None,
) -> None:
    """V3: per-market sub-tabs with consolidated analysis (historic-first + ML complement)."""
    # === Filters at top (liga / ano / split / times) ===
    render_tab_section(st, title="Recorte",
                       subtitle="Pré-preenchido pelo último campeonato dos times. Mude se quiser ver outro recorte.")

    # Auto-fill Liga/Ano/Split based on the MOST RECENT game involving either
    # of the two teams selected at the top. Re-fires when:
    #   1) the matchup changes (different teams), OR
    #   2) lav3_league is currently blank (recovers from resets that wipe
    #      the league pick without invalidating the fingerprint)
    _top_team_a = str(session_state.get("teamA", "") or "").strip()
    _top_team_b = str(session_state.get("teamB", "") or "").strip()
    _matchup_fp = f"{_top_team_a}|{_top_team_b}"
    _last_autofill_fp = str(session_state.get("_lav3_autofill_fp", "") or "")
    _league_blank = not str(session_state.get("lav3_league", "") or "").strip()
    _need_autofill = (_top_team_a and _top_team_b and
                       (_matchup_fp != _last_autofill_fp or _league_blank))

    def _team_match_key(value: Any) -> str:
        """Normalize a team name for cross-comparison, tolerating mojibake and
        casing/whitespace drift. Falls back to the raw string if normalize fails."""
        s = "" if value is None else str(value)
        s = s.strip()
        # Drop replacement char and surrounding non-letters so 'L�S' still
        # casefolds to a comparable key.
        s = s.replace("�", "").replace("?", "")
        # Lowercase ASCII fold (best-effort, ignore non-ascii leftovers).
        try:
            s = s.encode("ascii", "ignore").decode("ascii")
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass
        return s.lower().strip()

    if _need_autofill:
        try:
            if team_games is not None and not team_games.empty and {"team", "date"}.issubset(team_games.columns):
                _team_keys = team_games["team"].astype(str).map(_team_match_key)

                def _recent_games(_keys: set) -> pd.DataFrame:
                    _g = team_games[_team_keys.isin(_keys)].copy()
                    _g["_d"] = pd.to_datetime(_g["date"], errors="coerce")
                    return _g.dropna(subset=["_d"]).sort_values("_d")

                # Recorte segue o TIME 1 (teamA): num cross-league (ex.: EBL x LFL)
                # a liga do recorte deve ser a do PRIMEIRO time, não a de quem
                # jogou mais recentemente. Fallback p/ qualquer um dos dois se o
                # Time 1 não tiver jogos no CSV ativo.
                _matchup_games = _recent_games({_team_match_key(_top_team_a)})
                if _matchup_games.empty:
                    _matchup_games = _recent_games({_team_match_key(_top_team_a), _team_match_key(_top_team_b)})
                if not _matchup_games.empty:
                    _last_row = _matchup_games.iloc[-1]
                    _last_lg = str(_last_row.get("league", "") or "")
                    _last_yr = str(int(_last_row.get("year"))) if pd.notna(_last_row.get("year")) else ""
                    _last_sp = str(_last_row.get("split", "") or "")
                    if _last_lg:
                        session_state["lav3_league"] = _last_lg
                    if _last_yr:
                        session_state["lav3_year"] = _last_yr
                    if _last_sp:
                        session_state["lav3_split"] = _last_sp
                        # Make the split-context key consistent with the new picks
                        session_state["_lav3_split_ctx"] = f"{_last_lg}|{_last_yr}"
            session_state["_lav3_autofill_fp"] = _matchup_fp
        except (TypeError, ValueError, KeyError, AttributeError):
            session_state["_lav3_autofill_fp"] = _matchup_fp

    _, _, leagues = collect_year_split_league_options(team_games)
    fcols = st.columns(4)
    la_league_opts = ["", *leagues]
    with fcols[0]:
        normalize_select_key(session_state, "lav3_league", la_league_opts)
        v3_league = st.selectbox("Liga", la_league_opts, key="lav3_league",
                                   format_func=select_placeholder)
    df_ly = team_games
    if v3_league and team_games is not None and not team_games.empty and "league" in team_games.columns:
        df_ly = team_games[team_games["league"].astype(str) == str(v3_league)]
    years_opt: list[str] = []
    try:
        if df_ly is not None and not df_ly.empty and "year" in df_ly.columns:
            years_opt = [str(y) for y in sorted(
                int(x) for x in pd.to_numeric(df_ly["year"], errors="coerce").dropna().unique()
            )]
    except (TypeError, ValueError, KeyError):
        years_opt = []
    with fcols[1]:
        la_year_opts = ["", *years_opt]
        # Prefer the latest year as default (e.g., 2026), but user can switch.
        if (str(session_state.get("lav3_year", "") or "") not in la_year_opts) and years_opt:
            session_state["lav3_year"] = str(years_opt[-1])
        normalize_select_key(session_state, "lav3_year", la_year_opts)
        v3_year = st.selectbox("Ano", la_year_opts, key="lav3_year",
                                 format_func=select_placeholder)
    df_lys = df_ly
    if v3_year and df_lys is not None and not df_lys.empty and "year" in df_lys.columns:
        try:
            df_lys = df_lys[pd.to_numeric(df_lys["year"], errors="coerce") == int(v3_year)]
        except (TypeError, ValueError, KeyError):
            pass
    # Splits ordered by most recent (latest game date)
    splits_opt: list[str] = []
    latest_split: str = ""
    try:
        if df_lys is not None and not df_lys.empty and "split" in df_lys.columns:
            if "date" in df_lys.columns:
                _date_per_split = (
                    df_lys.assign(_d=pd.to_datetime(df_lys["date"], errors="coerce"))
                          .dropna(subset=["_d"])
                          .groupby("split")["_d"].max()
                          .sort_values(ascending=False)
                )
                splits_opt = [str(x) for x in _date_per_split.index.tolist() if str(x).strip()]
                if splits_opt:
                    latest_split = splits_opt[0]
            else:
                splits_opt = sorted(
                    str(x) for x in df_lys["split"].fillna("").astype(str).unique() if str(x).strip()
                )
                if splits_opt:
                    latest_split = splits_opt[-1]
    except (TypeError, ValueError, KeyError):
        splits_opt = []
    with fcols[2]:
        la_split_opts = ["", *splits_opt]
        # Auto-select latest split on first render OR when league/year change rotates options
        _cur = str(session_state.get("lav3_split", "") or "")
        _ctx_key = "_lav3_split_ctx"
        _ctx_now = f"{v3_league}|{v3_year}"
        if (session_state.get(_ctx_key) != _ctx_now) or (_cur not in la_split_opts):
            session_state["lav3_split"] = latest_split
            session_state[_ctx_key] = _ctx_now
        normalize_select_key(session_state, "lav3_split", la_split_opts)
        v3_split = st.selectbox("Split (opcional)", la_split_opts, key="lav3_split",
                                  format_func=lambda x: "Todos os splits" if x == "" else x)
    # Playoffs box appears ONLY when the chosen (liga, ano, split) actually has
    # playoff games — same pattern as the other tabs (scope_has_playoffs).
    if scope_has_playoffs(team_games, league=v3_league, year=v3_year, split=v3_split):
        with fcols[3]:
            la_po_opts = ["", "Regular", "Playoffs"]
            normalize_select_key(session_state, "lav3_playoffs", la_po_opts)
            v3_playoffs = st.selectbox(
                "Playoffs", la_po_opts, key="lav3_playoffs",
                format_func=lambda x: "Todos" if x == "" else x,
            )
    else:
        v3_playoffs = ""

    if not v3_league or not v3_year:
        st.info("Selecione liga e ano para continuar.")
        return

    sc_filters = Filters(
        year=int(v3_year), league=v3_league,
        split=v3_split if v3_split else None,
        playoffs=None if not v3_playoffs else (v3_playoffs == "Playoffs"),
    )
    df_scope = apply_filters_fn(team_games, sc_filters).copy()
    if df_scope is None or df_scope.empty or "team" not in df_scope.columns:
        st.info("Sem jogos nesse recorte.")
        return

    # Add computed mv2_total_* columns (used by Kills/Towers/Drakes markets)
    df_scope = _augment_with_total_columns(df_scope)

    # === Teams: reuse the top-of-page selection (Time 1 / Time 2) ===
    team_a = str(session_state.get("teamA", "") or "").strip()
    team_b = str(session_state.get("teamB", "") or "").strip()
    if not team_a or not team_b:
        st.info("Selecione os dois times no topo da página (Time 1 / Time 2) e clique em **Analisar Confronto**.")
        return
    if team_a == team_b:
        st.warning("Selecione times diferentes no topo da página.")
        return

    df_a = df_scope[df_scope["team"].astype(str) == str(team_a)].copy()
    df_b = df_scope[df_scope["team"].astype(str) == str(team_b)].copy()
    if df_a.empty and df_b.empty:
        st.info("Nenhum dos times tem jogos no recorte selecionado.")
        return
    # If exactly ONE team is missing from the scope, it's likely a cross-league
    # matchup (teams from different leagues). Don't block here — the per-market
    # subpanel detects this and rebuilds each team in its own domestic league.
    # We only need both teams to exist somewhere in team_games for that to work.
    if df_a.empty or df_b.empty:
        _have_a = bool((team_games["team"].astype(str) == str(team_a)).any()) if team_games is not None and "team" in team_games.columns else False
        _have_b = bool((team_games["team"].astype(str) == str(team_b)).any()) if team_games is not None and "team" in team_games.columns else False
        if not (_have_a and _have_b):
            st.info("Um dos times não tem jogos no histórico.")
            return
        # else: fall through — cross-league handled inside the subpanels.

    # === Sub-tabs per market === (+ Resumo at the very end when injected)
    market_labels = [m["label"] for m in _V3_MARKETS]
    if resumo_renderer is not None:
        market_labels = [*market_labels, "Resumo"]
    sub_tabs = st.tabs(market_labels)
    resumo_tab_ref = sub_tabs[-1] if resumo_renderer is not None else None
    market_sub_tabs = sub_tabs[:-1] if resumo_renderer is not None else sub_tabs

    # ML probabilities (best-effort from session)
    _p_map = session_state.get("_p_map_used")
    try:
        _p_map = float(_p_map) if _p_map is not None else None
        if _p_map is not None and not (0.0 < _p_map < 1.0):
            _p_map = None
    except (TypeError, ValueError):
        _p_map = None
    _p_series = session_state.get("_p_series_used")
    try:
        _p_series = float(_p_series) if _p_series is not None else None
        if _p_series is not None and not (0.0 < _p_series < 1.0):
            _p_series = None
    except (TypeError, ValueError):
        _p_series = None

    for tab_ref, market_cfg in zip(market_sub_tabs, _V3_MARKETS):
        with tab_ref:
            status = market_cfg.get("status", "todo")
            market_key = str(market_cfg.get("key", ""))
            # Vencedor has its own custom renderer (no line-based hit-rate concept)
            if market_key == "winner":
                _render_v3_vencedor_subpanel(
                    st=st, session_state=session_state,
                    team_a=str(team_a), team_b=str(team_b),
                    league_label=str(v3_league),
                    df_a=df_a, df_b=df_b, df_scope=df_scope,
                    team_games=team_games, Filters=Filters,
                    apply_filters_fn=apply_filters_fn,
                    v3_year=v3_year, v3_league=v3_league, v3_split=v3_split,
                    p_map=_p_map, p_series=_p_series,
                )
                continue
            if status != "ready":
                st.markdown(
                    f"<div style='padding:30px;text-align:center;opacity:0.55;'>"
                    f"<div style='font-size:1.4em;font-weight:600;'>"
                    f"Mercado <i>{market_cfg['label']}</i> — em construção</div>"
                    f"<div style='margin-top:8px;'>"
                    f"Replicar do layout aprovado de Kills assim que o protótipo for validado."
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                continue
            _render_v3_kills_subpanel(
                st=st, session_state=session_state,
                team_a=str(team_a), team_b=str(team_b),
                league_label=str(v3_league),
                df_a=df_a, df_b=df_b, df_scope=df_scope,
                team_games=team_games, Filters=Filters,
                apply_filters_fn=apply_filters_fn,
                v3_year=v3_year, v3_league=v3_league, v3_split=v3_split,
                market_cfg=market_cfg,
                p_map=_p_map, p_series=_p_series,
            )

    # Resumo (final sub-tab) — same renderer used previously inside Consistência
    if resumo_tab_ref is not None and resumo_renderer is not None:
        with resumo_tab_ref:
            # Broad catch is intentional: this is a render boundary. The Resumo
            # renderer is injected by plays_app and may fail for many reasons;
            # we surface the error in the sub-tab instead of crashing the whole
            # Análise de Linha page.
            try:
                resumo_renderer()
            except Exception as exc:
                st.error(f"Falha ao renderizar Resumo: {exc}")


def _render_v3_vencedor_subpanel(
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
    p_map: float | None,
    p_series: float | None,
) -> None:
    """Vencedor sub-tab: ML probabilities (Mapa/MD3/MD5) + Win Rate stats.

    Mirrors the Win Rate block from Consistência das linhas but with v3 styling
    (cyan banner headers, side-by-side bordered cards)."""

    # Header
    st.markdown(
        f"<div style='font-size:1.4em;font-weight:700;padding:6px 0 4px;'>"
        f"<span style='color:#2daac4;'>{team_a}</span>"
        f" <span style='opacity:0.6;'>VS</span> "
        f"<span style='color:#2daac4;'>{team_b}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ML probabilities chip (Mapa / MD3 / MD5) — uses session_state values
    _render_ml_chip_vencedor(st=st, team_a=team_a, team_b=team_b,
                               p_map=p_map, p_series=p_series)

    # Map filter
    map_opts = ["Todos", "M1", "M2", "M3", "M4", "M5"]
    map_key = f"v3vencedor_map_{team_a}_{team_b}"
    if session_state.get(map_key) not in map_opts:
        session_state[map_key] = map_opts[0]
    st.markdown("<div style='padding-top:10px;'></div>", unsafe_allow_html=True)
    map_choice = st.radio("Mapa", map_opts, horizontal=True, key=map_key,
                            help="Filtra todas as seções por mapa.")

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

    # Cross-league: if one team is absent from the scope, rebuild each team in
    # its own domestic league (most-recent split, excluding short internationals).
    if df_a is None or df_a.empty or df_b is None or df_b.empty:
        from app_core.market_v3_analysis_core import resolve_team_home_scope
        try:
            _yr = int(v3_year)
        except (TypeError, ValueError):
            _yr = None
        _ra = resolve_team_home_scope(team_games, team_a, year=_yr)
        _rb = resolve_team_home_scope(team_games, team_b, year=_yr)
        if not _ra["team_df"].empty and not _rb["team_df"].empty:
            df_a = _ra["team_df"].copy()
            df_b = _rb["team_df"].copy()
            st.info(
                f"⚔️ Times de ligas diferentes — {escape(team_a)} ({escape(_ra['league'])}) "
                f"vs {escape(team_b)} ({escape(_rb['league'])}), cada um na sua liga."
            )

    df_a = _filter_by_map(df_a)
    df_b = _filter_by_map(df_b)
    if df_a.empty or df_b.empty:
        st.info(f"Sem jogos pra {map_choice} no recorte.")
        return

    # Season-wide view: per-team (cross-league safe — uses each team's own rows)
    try:
        df_a_season = df_a.copy()
        df_b_season = df_b.copy()
    except (TypeError, ValueError, KeyError):
        df_a_season = df_a.copy()
        df_b_season = df_b.copy()

    # H2H do ANO INTEIRO em todas as ligas (par-a-par; recorte de liga escondia
    # confrontos em torneios fora da liga doméstica — fix 2026-07-13, mesmo do
    # painel de mercados).
    try:
        _h2h_all = team_games
        if "year" in _h2h_all.columns and "date" in df_a_season.columns and len(df_a_season):
            _yr_ref = pd.to_datetime(df_a_season["date"], errors="coerce").dt.year.max()
            if pd.notna(_yr_ref):
                _h2h_all = _h2h_all[pd.to_numeric(_h2h_all["year"], errors="coerce") == int(_yr_ref)]
        df_h2h = _h2h_all[
            (_h2h_all["team"].astype(str) == str(team_a))
            & (_h2h_all["opponent"].astype(str) == str(team_b))
        ].copy()
    except (TypeError, ValueError, KeyError):
        df_h2h = df_a_season[df_a_season["opponent"].astype(str) == str(team_b)].copy() \
            if "opponent" in df_a_season.columns else pd.DataFrame()

    # ---------- 3-card panel: OVERALL / RECENT FORM / PAST FACEOFFS ----------
    def _cyan_banner(title: str) -> str:
        return cyan_banner(title)

    def _wr_tuple(df_in: pd.DataFrame) -> tuple[int, int, int, float]:
        if df_in is None or df_in.empty or "win" not in df_in.columns:
            return 0, 0, 0, float("nan")
        w = pd.to_numeric(df_in["win"], errors="coerce")
        m = w.notna()
        n = int(m.sum())
        if n == 0:
            return 0, 0, 0, float("nan")
        k = int((w[m] > 0.5).sum())
        return n, k, n - k, float(k / n)

    def _side_wr(df_in: pd.DataFrame, side_token: str) -> float:
        if df_in is None or df_in.empty or "side" not in df_in.columns:
            return float("nan")
        side_ser = df_in["side"].astype(str).str.strip().str.lower()
        sub = df_in[side_ser == str(side_token).strip().lower()].copy()
        _, _, _, wr = _wr_tuple(sub)
        return float(wr)

    def _fmt_wr(wr: float) -> str:
        if not math.isfinite(float(wr)):
            return "—"
        return f"{100.0 * float(wr):.1f}%"

    def _wr_color(wr: float) -> str:
        if not math.isfinite(float(wr)):
            return "#9ca3af"
        p = 100.0 * float(wr)
        if p >= 65.0:
            return "#22c55e"
        if p >= 40.0:
            return "#facc15"
        return "#ef4444"

    c_over, c_recent, c_h2h = st.columns(3)

    # OVERALL
    with c_over:
        st.markdown(_cyan_banner("OVERALL"), unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;font-weight:600;"
                f"color:#2daac4;font-size:1.0em;margin-bottom:8px;'>"
                f"<span>{team_a}</span><span>{team_b}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='opacity:0.65;font-size:0.8em;font-style:italic;margin:6px 0 2px;'>Split atual</div>",
                          unsafe_allow_html=True)
            nA, wA, lA, wrA = _wr_tuple(df_a)
            nB, wB, lB, wrB = _wr_tuple(df_b)
            nA2, wA2, lA2, wrA2 = _wr_tuple(df_a_season)
            nB2, wB2, lB2, wrB2 = _wr_tuple(df_b_season)

            def _stat_row(label: str, val_a: str, val_b: str, color_a: str | None = None, color_b: str | None = None) -> str:
                cha = f"color:{color_a};font-weight:600;" if color_a else ""
                chb = f"color:{color_b};font-weight:600;" if color_b else ""
                return (f"<div style='display:grid;grid-template-columns:1.3fr 1fr 1fr;"
                        f"padding:4px 6px;border-bottom:1px solid rgba(255,255,255,0.05);'>"
                        f"<div style='opacity:0.8;'>{label}</div>"
                        f"<div style='text-align:center;{cha}'>{val_a}</div>"
                        f"<div style='text-align:center;{chb}'>{val_b}</div></div>")

            rows_html = (
                _stat_row("Win Rate", _fmt_wr(wrA), _fmt_wr(wrB), _wr_color(wrA), _wr_color(wrB))
                + _stat_row("# Games", str(nA), str(nB))
                + _stat_row("Record", f"{wA}W-{lA}L", f"{wB}W-{lB}L")
                + _stat_row("Blue Side", _fmt_wr(_side_wr(df_a, "blue")),
                              _fmt_wr(_side_wr(df_b, "blue")))
                + _stat_row("Red Side", _fmt_wr(_side_wr(df_a, "red")),
                              _fmt_wr(_side_wr(df_b, "red")))
            )
            st.markdown(rows_html, unsafe_allow_html=True)
            st.markdown("<div style='opacity:0.65;font-size:0.8em;font-style:italic;margin:10px 0 2px;'>Season inteira</div>",
                          unsafe_allow_html=True)
            rows_html2 = (
                _stat_row("Win Rate", _fmt_wr(wrA2), _fmt_wr(wrB2), _wr_color(wrA2), _wr_color(wrB2))
                + _stat_row("# Games", str(nA2), str(nB2))
                + _stat_row("Record", f"{wA2}W-{lA2}L", f"{wB2}W-{lB2}L")
            )
            st.markdown(rows_html2, unsafe_allow_html=True)

    # RECENT FORM
    with c_recent:
        st.markdown(_cyan_banner("RECENT FORM"), unsafe_allow_html=True)
        with st.container(border=True):
            def _recent_rows_html(df_in: pd.DataFrame, n: int = 10) -> str:
                if df_in is None or df_in.empty:
                    return "<div style='opacity:0.5;'>— sem jogos —</div>"
                d = df_in.sort_values("date").tail(int(n)).iloc[::-1]
                parts = []
                for _, r in d.iterrows():
                    opp = str(r.get("opponent", "") or "-")
                    wv = pd.to_numeric(pd.Series([r.get("win")]), errors="coerce").iloc[0]
                    won = pd.notna(wv) and float(wv) > 0.5
                    mark_color = "#22c55e" if won else "#ef4444"
                    mark = "W" if won else "L"
                    parts.append(
                        f"<div style='padding:3px 0;'>"
                        f"<span style='opacity:0.85;'>{opp}</span>"
                        f"<span style='margin-left:8px;display:inline-block;padding:1px 8px;"
                        f"border-radius:8px;background:rgba(45,170,196,0.10);color:{mark_color};"
                        f"font-weight:700;font-size:0.85em;'>{mark}</span>"
                        f"</div>"
                    )
                return "".join(parts)

            _, _, _, wrA5 = _wr_tuple(df_a.tail(5))
            _, _, _, wrA10 = _wr_tuple(df_a.tail(10))
            _, _, _, wrB5 = _wr_tuple(df_b.tail(5))
            _, _, _, wrB10 = _wr_tuple(df_b.tail(10))
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;color:#2daac4;"
                f"font-weight:600;font-size:1.0em;margin-bottom:8px;'>"
                f"<span>{team_a}</span><span>{team_b}</span></div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;font-size:0.85em;"
                f"opacity:0.85;padding-bottom:6px;border-bottom:1px solid rgba(255,255,255,0.08);'>"
                f"<div>Last 5: <b>{_fmt_wr(wrA5)}</b> · Last 10: <b>{_fmt_wr(wrA10)}</b></div>"
                f"<div>Last 5: <b>{_fmt_wr(wrB5)}</b> · Last 10: <b>{_fmt_wr(wrB10)}</b></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            rc_a, rc_b = st.columns(2)
            with rc_a:
                st.markdown(_recent_rows_html(df_a, 10), unsafe_allow_html=True)
            with rc_b:
                st.markdown(_recent_rows_html(df_b, 10), unsafe_allow_html=True)

    # PAST FACEOFFS
    with c_h2h:
        st.markdown(_cyan_banner("PAST FACEOFFS"), unsafe_allow_html=True)
        with st.container(border=True):
            nH, wH, lH, wrH = _wr_tuple(df_h2h)
            st.markdown(
                f"<div style='font-weight:600;color:#2daac4;font-size:1.0em;margin-bottom:8px;'>"
                f"{team_a} vs {team_b}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='display:flex;gap:14px;padding-bottom:8px;"
                f"border-bottom:1px solid rgba(255,255,255,0.08);'>"
                f"<div><div style='opacity:0.6;font-size:0.78em;text-transform:uppercase;'>H2H GAMES</div>"
                f"<div style='font-size:1.4em;font-weight:700;color:#2daac4;'>{int(nH)}</div></div>"
                f"<div><div style='opacity:0.6;font-size:0.78em;text-transform:uppercase;'>RECORD</div>"
                f"<div style='font-size:1.4em;font-weight:700;'>{wH}W-{lH}L</div></div>"
                f"<div><div style='opacity:0.6;font-size:0.78em;text-transform:uppercase;'>{team_a}</div>"
                f"<div style='font-size:1.4em;font-weight:700;color:{_wr_color(wrH)};'>{_fmt_wr(wrH)}</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if nH > 0:
                d = df_h2h.sort_values("date").tail(10).iloc[::-1]
                parts = []
                for _, r in d.iterrows():
                    dt = pd.to_datetime(r.get("date"), errors="coerce")
                    dts = dt.strftime("%Y-%m-%d") if pd.notna(dt) else "—"
                    wv = pd.to_numeric(pd.Series([r.get("win")]), errors="coerce").iloc[0]
                    winner = team_a if (pd.notna(wv) and float(wv) > 0.5) else team_b
                    color = "#22c55e" if winner == team_a else "#ef4444"
                    parts.append(
                        f"<div style='padding:3px 0;display:flex;justify-content:space-between;'>"
                        f"<span style='opacity:0.7;font-size:0.88em;'>{dts}</span>"
                        f"<span style='font-weight:700;color:{color};'>{winner}</span>"
                        f"</div>"
                    )
                st.markdown("".join(parts), unsafe_allow_html=True)
            else:
                st.markdown("<div style='opacity:0.55;'>— sem H2H —</div>", unsafe_allow_html=True)


# Helpers movidos p/ line_analysis_panels_core (fatiamento 18/07); re-importados
# aqui para o namespace/ctx do app seguir idêntico.
from app_core.line_analysis_panels_core import (  # noqa: E402,F401 - re-export
    _render_balanced_line_analysis,
    _render_ml_chip_vencedor,
)


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
    df_a_all_maps: pd.DataFrame,
    df_b_all_maps: pd.DataFrame,
    df_a_full_year: pd.DataFrame = None,
    df_b_full_year: pd.DataFrame = None,
    league_scope_for_panel: pd.DataFrame,
    league_scope_a: pd.DataFrame = None,
    league_scope_b: pd.DataFrame = None,
    cross_league: bool = False,
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
                                    full_year_df=df_a_full_year)
    stats_b = compute_window_stats(df_b_context_all, metric_col=metric_col, line=float(line_val),
                                    full_year_df=df_b_full_year)
    h2h_w = compute_h2h_stats(df_a_context_all, team_b, metric_col=metric_col, line=float(line_val))
    _render_balanced_line_analysis(
        st=st, team_a=team_a, team_b=team_b,
        stats_a=stats_a, stats_b=stats_b, h2h=h2h_w,
        line=float(line_val),
    )

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
        _stats_scope = f"{v3_year} · ligas diferentes (cada time na sua) · Mapa: {map_choice}"
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
        session_state[lines_key] = [float(default_line)]
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
    with lc2:
        if bool(market_cfg.get("binary_market")):
            st.markdown(
                "<div style='padding-top:6px;opacity:0.7;font-size:0.88em;'>"
                "Marcação <b>sim/não</b> — sem linha para escolher.</div>",
                unsafe_allow_html=True,
            )
            _submitted = False
            new_txt = ""
        else:
            # Add lines via a FORM: nothing is analyzed until you click — typing in
            # the box does NOT trigger a re-run. Accepts several lines separated by
            # comma (27.5, 28.5, 29.5), decimal with dot. Time also takes 33:30 / 33.5.
            with st.form(key=f"{lines_key}__addform", clear_on_submit=True):
                _ph = "ex: 33:30, 34.5" if _is_time else "ex: 27.5, 28.5, 29.5"
                new_txt = st.text_input(
                    "Adicionar linha(s)",
                    value="",
                    placeholder=_ph,
                    help=("Várias de uma vez separadas por vírgula. Decimal com ponto. "
                          + ("Tempo: 33:30 ou 33.5 (minutos)." if _is_time else "Ex.: 27.5, 28.5.")),
                )
                _submitted = st.form_submit_button("➕ Adicionar linha(s)", width="stretch")
        if _submitted and str(new_txt or "").strip():
            parsed = parse_time_lines_text(new_txt) if _is_time else parse_lines_text(new_txt)
            merged = list(lines_list)
            for v in parsed:
                if not any(abs(float(v) - float(x)) < 1e-9 for x in merged):
                    merged.append(float(v))
            if merged != list(lines_list):
                # Sorted ascending so a smaller line sits to the left.
                session_state[lines_key] = sorted(merged)
                st.rerun()
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
        st.info(
            f"⚔️ Times de ligas diferentes — cada um avaliado na sua liga "
            f"(split mais recente): **{escape(team_a)}** ({escape(_xl_league_a)} · "
            f"{escape(_ra['split'] or '—')}) vs **{escape(team_b)}** "
            f"({escape(_xl_league_b)} · {escape(_rb['split'] or '—')}). "
            f"Oponentes e médias da liga são de cada liga respectiva."
        )

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
        season_filters = Filters(year=int(v3_year), league=v3_league, split=None, playoffs=None)
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
        df_a_all_maps=df_a_all_maps, df_b_all_maps=df_b_all_maps,
        df_a_full_year=df_a_full_year, df_b_full_year=df_b_full_year,
        league_scope_for_panel=league_scope_for_panel,
        league_scope_a=league_scope_a, league_scope_b=league_scope_b,
        cross_league=_cross_league,
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


def render_line_analysis_tab(
    *,
    st: Any,
    team_games: pd.DataFrame,
    Filters: Any,
    apply_filters_fn: Callable[[pd.DataFrame, Any], pd.DataFrame],
    min_to_mmss_fn: Callable[[Any], str],
    session_state: MutableMapping[str, Any],
    resumo_renderer: Optional[Callable[[], None]] = None,
) -> None:
    render_tab_hero(
        st,
        title="Análise de Linha",
        subtitle="Veja como uma linha Over/Under se comporta quando um time joga contra cada oponente da liga — com contexto inverso e referência do campeonato.",
        eyebrow="Scout",
    )

    # Em que mercado o numero desta aba presta (medido 26/08/2026 — ate entao
    # ela servia cinco mercados sem ninguem saber se algum prestava).
    try:
        from pathlib import Path as _P

        from app_core.nota_mercados_core import render_confianca_mercados as _conf_mkt

        _conf_mkt(st=st, root=_P(__file__).resolve().parents[1])
    except (ImportError, OSError, TypeError, ValueError):
        pass

    # v3 is the only supported view now. v2/Clássica foram aposentados.
    _render_v3_view(
        st=st, team_games=team_games, Filters=Filters,
        apply_filters_fn=apply_filters_fn, session_state=session_state,
        resumo_renderer=resumo_renderer,
    )
    return

    # ------------------------------------------------------------------ #
    # Recorte
    # ------------------------------------------------------------------ #
    render_tab_section(st, title="Recorte", subtitle="Liga e ano são obrigatórios. Split e Playoffs são opcionais.")

    _, _, leagues = collect_year_split_league_options(team_games)
    scope_cols = st.columns(4)

    la_league_opts = ["", *leagues]

    def _on_league_change() -> None:
        session_state["la_year"] = ""
        session_state["la_split"] = ""
        session_state["la_team"] = ""

    with scope_cols[0]:
        normalize_select_key(session_state, "la_league", la_league_opts)
        la_league = st.selectbox(
            "Liga",
            la_league_opts,
            key="la_league",
            format_func=select_placeholder,
            on_change=_on_league_change,
        )

    df_ly = team_games
    if la_league and team_games is not None and not team_games.empty and "league" in team_games.columns:
        df_ly = team_games[team_games["league"].astype(str) == str(la_league)]

    years_opt: list[str] = []
    try:
        if df_ly is not None and not df_ly.empty and "year" in df_ly.columns:
            years_opt = [str(y) for y in sorted(int(x) for x in pd.to_numeric(df_ly["year"], errors="coerce").dropna().unique())]
    except (TypeError, ValueError, KeyError):
        years_opt = []

    def _on_year_change() -> None:
        session_state["la_split"] = ""
        session_state["la_team"] = ""

    with scope_cols[1]:
        la_year_opts = ["", *years_opt]
        normalize_select_key(session_state, "la_year", la_year_opts)
        la_year = st.selectbox(
            "Ano",
            la_year_opts,
            key="la_year",
            format_func=select_placeholder,
            on_change=_on_year_change,
        )

    df_lys = df_ly
    if la_year and df_lys is not None and not df_lys.empty and "year" in df_lys.columns:
        try:
            df_lys = df_lys[pd.to_numeric(df_lys["year"], errors="coerce") == int(la_year)]
        except (TypeError, ValueError, KeyError):
            pass

    splits_opt: list[str] = []
    try:
        if df_lys is not None and not df_lys.empty and "split" in df_lys.columns:
            splits_opt = sorted(str(x) for x in df_lys["split"].fillna("").astype(str).unique() if str(x).strip())
    except (TypeError, ValueError, KeyError):
        splits_opt = []

    with scope_cols[2]:
        la_split_opts = ["", *splits_opt]
        normalize_select_key(session_state, "la_split", la_split_opts)
        la_split = st.selectbox(
            "Split (opcional)",
            la_split_opts,
            key="la_split",
            format_func=lambda x: "Todos os splits" if x == "" else x,
        )

    with scope_cols[3]:
        normalize_select_key(session_state, "la_playoffs", ["", "Regular", "Playoffs"])
        la_playoffs = st.selectbox(
            "Playoffs",
            ["", "Regular", "Playoffs"],
            key="la_playoffs",
            format_func=select_placeholder,
        )

    if not la_league:
        st.info("Selecione uma liga para continuar.")
        return
    if not la_year:
        st.info("Selecione um ano para continuar.")
        return

    sc_filters = Filters(
        year=int(la_year),
        league=la_league,
        split=la_split if la_split else None,
        playoffs=None if not la_playoffs else (la_playoffs == "Playoffs"),
    )
    df_scope = apply_filters_fn(team_games, sc_filters).copy()

    if df_scope.empty or "team" not in df_scope.columns or "opponent" not in df_scope.columns:
        st.info("Sem jogos nesse recorte.")
        return

    # ------------------------------------------------------------------ #
    # Configuração da linha
    # ------------------------------------------------------------------ #
    render_tab_section(st, title="Configuração da linha", subtitle="Defina o time alvo, mercado, direção e valor.")

    all_teams = sorted(df_scope["team"].dropna().astype(str).unique().tolist())
    line_cols = st.columns([1.3, 1.0, 0.7, 1.0])

    la_team_opts = ["", *all_teams]
    normalize_select_key(session_state, "la_team", la_team_opts)

    with line_cols[0]:
        la_team = st.selectbox(
            "Time alvo",
            la_team_opts,
            key="la_team",
            format_func=select_placeholder,
        )

    market_labels = list(STATS_MARKET_CONFIG.keys())
    normalize_select_key(session_state, "la_market", market_labels, fallback=market_labels[0])

    with line_cols[1]:
        la_market = st.selectbox("Mercado", market_labels, key="la_market")

    normalize_select_key(session_state, "la_direction", ["Under", "Over"], fallback="Under")

    with line_cols[2]:
        la_direction = st.selectbox("Direção", ["Under", "Over"], key="la_direction")

    with line_cols[3]:
        la_line_text = st.text_input("Linha", key="la_line_value", placeholder="ex: 29.5")

    if not la_team:
        st.info("Selecione o time alvo.")
        return

    try:
        la_line = float(str(la_line_text or "").strip().replace(",", "."))
    except (TypeError, ValueError):
        st.info("Digite um valor de linha válido (ex: 29.5).")
        return

    metric_col = _get_metric_col(la_market)
    if not metric_col or metric_col not in df_scope.columns:
        st.warning(f"Métrica '{la_market}' não disponível nos dados desse recorte.")
        return

    df_target = df_scope[df_scope["team"].astype(str) == str(la_team)].copy()
    if df_target.empty:
        st.info(f"Sem jogos para '{la_team}' nesse recorte.")
        return

    # ------------------------------------------------------------------ #
    # Seção 1 — Linha por oponente
    # ------------------------------------------------------------------ #
    render_tab_section(
        st,
        title=f"{la_direction} {la_line:g} — {la_market} por oponente",
        subtitle=f"Cada linha é um adversário de {la_team}. Ordenado pela % {la_direction} (mais favorável primeiro).",
    )

    rows_by_opp: list[dict] = []
    for opp_name, grp in df_target.groupby("opponent"):
        s = pd.to_numeric(grp[metric_col], errors="coerce").dropna()
        n = len(s)
        if n == 0:
            continue
        avg_val = float(s.mean())
        pct = _safe_pct(s, la_line, la_direction)
        rows_by_opp.append({
            "Oponente": str(opp_name),
            "Jogos": n,
            f"Média {la_market}": avg_val,
            f"% {la_direction}": pct,
            "Status": fmt_status(pct),
            "_pct": pct,
            "_n": n,
        })

    if not rows_by_opp:
        st.info("Sem confrontos individuais encontrados.")
        return

    df_opp = (
        pd.DataFrame(rows_by_opp)
        .sort_values("_pct", ascending=False)
        .reset_index(drop=True)
    )

    low_n = df_opp[df_opp["_n"] < 3]
    if not low_n.empty:
        names = ", ".join(low_n["Oponente"].tolist())
        st.caption(f"Menos de 3 jogos (instável): {names}")

    df_opp_view = df_opp.drop(columns=["_pct", "_n"]).copy()
    if la_market == "Tempo":
        df_opp_view[f"Média {la_market}"] = df_opp_view[f"Média {la_market}"].apply(min_to_mmss_fn)
    else:
        df_opp_view[f"Média {la_market}"] = df_opp_view[f"Média {la_market}"].apply(fmt_num)
    df_opp_view[f"% {la_direction}"] = df_opp_view[f"% {la_direction}"].apply(fmt_pct)

    st.dataframe(df_opp_view, width="stretch", hide_index=True)

    # ------------------------------------------------------------------ #
    # Seção 2 — Contexto inverso
    # ------------------------------------------------------------------ #
    render_tab_section(
        st,
        title="Contexto inverso — oponentes vs outros times",
        subtitle=(
            "Mostra como cada oponente se comporta na mesma linha quando joga contra times que NÃO são o time alvo. "
            "Delta positivo = aquele oponente vai mais para esse lado especificamente contra o time alvo."
        ),
    )

    df_others = df_scope[df_scope["team"].astype(str) != str(la_team)].copy()
    rows_inv: list[dict] = []
    for _, row in df_opp.iterrows():
        opp_name = str(row["Oponente"])
        pct_vs_target = float(row["_pct"])
        n_vs_target = int(row["_n"])

        df_opp_vs_others = df_others[
            (df_others["team"].astype(str) == opp_name) &
            (df_others["opponent"].astype(str) != str(la_team))
        ].copy()
        s_others = pd.to_numeric(df_opp_vs_others[metric_col], errors="coerce").dropna() if not df_opp_vs_others.empty else pd.Series(dtype=float)
        n_others = len(s_others)
        pct_others = _safe_pct(s_others, la_line, la_direction) if n_others > 0 else float("nan")
        avg_others = float(s_others.mean()) if n_others > 0 else float("nan")

        delta = (pct_vs_target - pct_others) if (math.isfinite(pct_vs_target) and math.isfinite(pct_others)) else float("nan")

        rows_inv.append({
            "Oponente": opp_name,
            f"Jogos vs {la_team}": n_vs_target,
            f"% {la_direction} vs {la_team}": fmt_pct(pct_vs_target),
            "Jogos vs outros": n_others,
            f"% {la_direction} vs outros": fmt_pct(pct_others),
            f"Média {la_market} vs outros": min_to_mmss_fn(avg_others) if la_market == "Tempo" else fmt_num(avg_others),
            "Delta (alvo − outros)": fmt_delta_pp(delta),
        })

    st.dataframe(pd.DataFrame(rows_inv), width="stretch", hide_index=True)

    # ------------------------------------------------------------------ #
    # Referência da liga
    # ------------------------------------------------------------------ #
    render_tab_section(
        st,
        title="Referência da liga",
        subtitle="Comportamento geral da linha no campeonato e o agregado do time alvo contra todos os oponentes.",
    )

    s_league = pd.to_numeric(df_scope[metric_col], errors="coerce").dropna()
    s_target_all = pd.to_numeric(df_target[metric_col], errors="coerce").dropna()

    def _ref_row(label: str, s: pd.Series) -> dict:
        n = len(s)
        avg_v = float(s.mean()) if n > 0 else float("nan")
        pct_v = _safe_pct(s, la_line, la_direction) if n > 0 else float("nan")
        return {
            "Escopo": label,
            "Jogos": n,
            f"Média {la_market}": min_to_mmss_fn(avg_v) if la_market == "Tempo" else fmt_num(avg_v),
            f"% {la_direction}": fmt_pct(pct_v),
        }

    df_ref = pd.DataFrame([
        _ref_row("Liga (todos os times)", s_league),
        _ref_row(f"{la_team} (todos os oponentes)", s_target_all),
    ])
    st.dataframe(df_ref, width="stretch", hide_index=True)

    # ------------------------------------------------------------------ #
    # Seção 4 — Primeiros objetivos
    # ------------------------------------------------------------------ #
    _FIRST_OBJ_COLS = [
        ("firstdragon", "Primeiro Dragão"),
        ("firstherald",  "Primeiro Herald"),
        ("firsttower",   "Primeira Torre"),
        ("firstblood",   "First Blood"),
        ("firstbaron",   "Primeiro Baron"),
    ]

    available_obj = [(col, label) for col, label in _FIRST_OBJ_COLS if col in df_scope.columns]
    if available_obj:
        render_tab_section(
            st,
            title="Primeiros objetivos do time alvo",
            subtitle=f"Taxa com que {la_team} consegue cada primeiro objetivo no recorte selecionado, comparado à média da liga.",
        )
        st.caption("O CSV não registra o timestamp do primeiro objetivo — apenas quem o conseguiu. A taxa abaixo indica com que frequência o time chega lá primeiro.")

        rows_obj: list[dict] = []
        for col, label in available_obj:
            s_team_obj = pd.to_numeric(df_target[col], errors="coerce").dropna()
            s_league_obj = pd.to_numeric(df_scope[col], errors="coerce").dropna()
            n_team = len(s_team_obj)
            n_league = len(s_league_obj)
            rate_team = 100.0 * float(s_team_obj.mean()) if n_team > 0 else float("nan")
            rate_league = 100.0 * float(s_league_obj.mean()) if n_league > 0 else float("nan")
            delta = (rate_team - rate_league) if (math.isfinite(rate_team) and math.isfinite(rate_league)) else float("nan")
            rows_obj.append({
                "Objetivo": label,
                f"Taxa {la_team}": fmt_pct(rate_team),
                "Taxa da liga": fmt_pct(rate_league),
                "Delta (time − liga)": fmt_delta_pp(delta),
            })

        st.dataframe(pd.DataFrame(rows_obj), width="stretch", hide_index=True)
