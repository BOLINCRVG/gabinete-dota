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

from app_core.shared_formatters import fmt_delta_pp, fmt_num, fmt_pct, fmt_status
from app_core.ui_design_core import cyan_banner
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



# Painel de kills movido para line_analysis_kills_core (16/08/2026).
from app_core.line_analysis_kills_core import (  # noqa: F401 - superficie publica
    _render_kills_line_content,
    _render_v3_kills_subpanel,
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
        # Ano padrao: o do CALENDARIO, se a liga tem jogos nele; senao o mais
        # recente. 23/10/2026: o Oracle's Elixir ja marca jogos de fim de
        # setembro como 2027 (pre-temporada do CBLOL, LIT); "o maior ano" caia
        # num 2027 de 10 jogos e a tela dizia "Sem jogos".
        if (str(session_state.get("lav3_year", "") or "") not in la_year_opts) and years_opt:
            session_state["lav3_year"] = ano_padrao(years_opt)
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


def ano_padrao(anos: list[str], hoje=None) -> str:
    """O ano do calendario se estiver na lista; senao o mais recente."""
    from datetime import date as _date
    atual = str((hoje or _date.today()).year)
    return atual if atual in anos else str(anos[-1])

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
            # NAO presumir "ligas diferentes" so porque o recorte esvaziou:
            # ver descrever_recorte_por_time.
            from app_core.market_v3_analysis_core import descrever_recorte_por_time
            _desc = descrever_recorte_por_time(_ra, _rb, team_a, team_b)
            st.info(_desc["texto"])

    df_a = _filter_by_map(df_a)
    df_b = _filter_by_map(df_b)
    if df_a.empty or df_b.empty:
        st.info(f"Sem jogos pra {map_choice} no recorte.")
        return

    # Season-wide view: MESMO ano/liga, SEM o filtro de split.
    # BUG achado com ele (24/07): o try e o except faziam a mesma coisa
    # (df_a.copy(), que ja vem filtrado por split) — o card "Season inteira"
    # mostrava numeros IDENTICOS ao "Split atual" (ex.: Gen.G 10 jogos nos dois,
    # tendo 118 no ano). Mesmo padrao do bloco que ja funciona (df_a_context_season).
    try:
        _season_filters = Filters(year=int(v3_year), league=v3_league, split=None, playoffs=None)
        _df_season_scope = apply_filters_fn(team_games, _season_filters).copy()
        _df_season_scope = _augment_with_total_columns(_df_season_scope)
        _df_season_scope = _filter_by_map(_df_season_scope)
        df_a_season = _df_season_scope[_df_season_scope["team"].astype(str) == str(team_a)].copy()
        df_b_season = _df_season_scope[_df_season_scope["team"].astype(str) == str(team_b)].copy()
        # cross-league: o scope de UMA liga so tem um dos times — reconstroi cada
        # um na liga dele (mesmo fix do outro bloco).
        if df_a_season.empty:
            df_a_season = df_a.copy()
        if df_b_season.empty:
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

    # Em que mercado o numero desta aba presta. A regua dos totais media
    # cada mercado desde 07/2026, mas so a NOTA aparecia — e nota alta pode
    # ser linha torta. O que vale e o ganho sobre a taxa-base, e ele agora
    # chega aqui.
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
