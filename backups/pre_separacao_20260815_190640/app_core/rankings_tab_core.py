"""Rankings tab renderer."""

from __future__ import annotations

from html import escape
from typing import Any, Callable

import pandas as pd

from app_core.tab_filters_core import collect_year_split_league_options, scope_has_playoffs
from app_core.tab_state_core import ensure_blank_keys, normalize_select_key, reset_keys, select_placeholder
from app_core.tab_ui_core import render_tab_hero, render_tab_section
from app_core.ui_design_core import team_badge_html
from mlcore.team_artifact_normalize import _norm_team_key


def _render_rank_grid(
    st: Any,
    df: pd.DataFrame,
    *,
    format_num_fn: Callable[[float, int], str],
    metric_col: str,
    metric_label: str,
    show_tier: bool = True,
    top_n: int = 200,
) -> None:
    """Render a ranking as a cyan HTML grid: # | Time (badge) | Liga | [Tier] | <metric>.

    Pre-sorted by the source; top-3 get medals. Scrollable for long lists.
    """
    if df is None or df.empty:
        st.info("Sem dados para essa fonte/recorte. Troque a fonte acima ou ajuste o recorte.")
        return
    d = df.head(int(top_n)).copy().reset_index(drop=True)
    grid = "0.5fr 2.4fr 0.9fr" + (" 0.7fr" if show_tier else "") + " 1fr"

    def _num(v: Any) -> str:
        try:
            f = float(v)
            return format_num_fn(f, 0) if f == f else "—"  # f==f filters NaN
        except (TypeError, ValueError):
            return "—"

    tier_head = "<div style='text-align:center;opacity:0.7;'>Tier</div>" if show_tier else ""
    header = (
        f"<div style='display:grid;grid-template-columns:{grid};"
        "background:rgba(45,170,196,0.14);padding:7px 10px;font-weight:700;"
        "font-size:0.78em;text-transform:uppercase;letter-spacing:0.05em;opacity:0.9;"
        "position:sticky;top:0;'>"
        "<div style='text-align:center;'>#</div>"
        "<div>Time</div>"
        "<div style='text-align:center;opacity:0.7;'>Liga</div>"
        f"{tier_head}"
        f"<div style='text-align:center;color:#2daac4;'>{escape(metric_label)}</div>"
        "</div>"
    )
    rows = []
    for i, r in d.iterrows():
        bg = "rgba(255,255,255,0.02)" if (i % 2 == 0) else "rgba(255,255,255,0.045)"
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else str(i + 1)
        team = str(r.get("team", "") or "")
        league = escape(str(r.get("league", "") or "—"))
        tier_cell = (f"<div style='text-align:center;opacity:0.75;'>{escape(str(r.get('tier','') or '—'))}</div>"
                     if show_tier else "")
        val = _num(r.get(metric_col))
        rows.append(
            f"<div style='display:grid;grid-template-columns:{grid};align-items:center;"
            f"padding:5px 10px;background:{bg};border-bottom:1px solid rgba(255,255,255,0.04);'>"
            f"<div style='text-align:center;font-weight:700;opacity:0.85;'>{medal}</div>"
            f"<div style='display:flex;align-items:center;gap:8px;'>{team_badge_html(team, size=24)}"
            f"<span style='font-weight:600;'>{escape(team)}</span></div>"
            f"<div style='text-align:center;opacity:0.8;'>{league}</div>"
            f"{tier_cell}"
            f"<div style='text-align:center;font-weight:700;color:#2daac4;'>{val}</div>"
            "</div>"
        )
    st.markdown(
        "<div style='border:1px solid rgba(45,170,196,0.25);border-radius:10px;overflow:hidden;'>"
        f"{header}"
        f"<div style='max-height:560px;overflow-y:auto;'>{''.join(rows)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _rating_row(df_all: pd.DataFrame, team: str) -> Any:
    if df_all is None or df_all.empty or "team" not in df_all.columns:
        return None
    r = df_all[df_all["team"].astype(str) == str(team)]
    return r.iloc[0] if not r.empty else None


def _render_rating_comparison(
    st: Any,
    df_all: pd.DataFrame,
    team_a: str,
    team_b: str,
    *,
    format_num_fn: Callable[[float, int], str],
) -> None:
    """Cards lado a lado com a força (ML core v2) de dois times + veredito.

    Serve para comparar times de ligas DIFERENTES (cross-league): a força do v2
    já embute SOS/força-de-liga, então comparar o total_strength é justo entre
    ligas (ao contrário de um Elo cru, que não é comparável entre regiões)."""
    ra = _rating_row(df_all, team_a)
    rb = _rating_row(df_all, team_b)
    if ra is None or rb is None:
        st.info("Algum dos times não está no ranking do modelo. Use a busca acima para conferir o nome exato.")
        return

    def _f(v: Any) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    cols = st.columns(2)
    for col, r in zip(cols, (ra, rb)):
        with col:
            _s = _f(r.get("total_strength"))
            _s_txt = format_num_fn(_s, 1) if _s == _s else "—"  # NaN check
            st.markdown(
                "<div style='border:1px solid rgba(45,170,196,0.25);border-radius:10px;"
                "padding:12px 14px;background:rgba(45,170,196,0.05);'>"
                f"<div style='font-weight:800;font-size:1.15em;color:#2daac4;'>{escape(str(r.get('team', '')))}</div>"
                f"<div style='opacity:0.7;font-size:0.85em;margin:2px 0 8px;'>"
                f"{escape(str(r.get('league', '') or '—'))} · {escape(str(r.get('tier', '') or '—'))}</div>"
                f"<div style='font-size:1.7em;font-weight:800;'>{_s_txt}</div>"
                f"<div style='opacity:0.6;font-size:0.8em;'>força ML · #{int(r.get('rank')) if str(r.get('rank','')).strip().isdigit() else '—'} no ranking geral</div>"
                "</div>",
                unsafe_allow_html=True,
            )

    sa, sb = _f(ra.get("total_strength")), _f(rb.get("total_strength"))
    if sa == sa and sb == sb:  # both finite
        diff = sa - sb
        if abs(diff) < 1e-9:
            _verdict = "Empate técnico de força."
        else:
            _winner = str(ra.get("team", "")) if sa > sb else str(rb.get("team", ""))
            _verdict = f"**{_winner}** é mais forte por **{format_num_fn(abs(diff), 1)}** de força."
        st.caption(f"{_verdict} (força ML core v2 = Elo residual + força de liga/tier + conquistas; comparável entre ligas).")


def render_rankings_tab(
    *,
    st: Any,
    team_games: pd.DataFrame,
    Filters: Any,
    artifact_path: str,
    as_of_date: Any,
    half_life_days: float,
    mlcore_rank_table_v2_fn: Callable[..., pd.DataFrame],
    elo_rank_table_fn: Callable[..., pd.DataFrame],
    format_num_fn: Callable[[float, int], str],
    team_a: str = "",
    team_b: str = "",
) -> None:
    render_tab_hero(
        st,
        title="Rankings",
        subtitle="Tabela principal do ML core v2 com fallback de Elo para auditoria e cobertura.",
        eyebrow="Power ratings",
        chips=[f"As of {as_of_date}", f"Half-life {format_num_fn(float(half_life_days), 0)}d"],
    )
    render_tab_section(
        st,
        title="Recorte do ranking",
        subtitle="Filtre liga, ano, split e playoffs antes de abrir a tabela principal.",
    )

    _, _, leagues = collect_year_split_league_options(team_games)
    ensure_blank_keys(st.session_state, ["rank_league_visual", "rank_year_visual", "rank_split_visual", "rank_po_visual"])

    def _on_rank_league_change() -> None:
        reset_keys(st.session_state, ["rank_year_visual", "rank_split_visual"])

    c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.0, 1.0])
    with c1:
        rank_league = st.selectbox(
            "Liga",
            options=["", *leagues],
            key="rank_league_visual",
            format_func=select_placeholder,
            on_change=_on_rank_league_change,
        )

    df_opt = team_games
    if rank_league and df_opt is not None and not df_opt.empty and "league" in df_opt.columns:
        df_opt = df_opt[df_opt["league"].astype(str) == str(rank_league)].copy()

    years = []
    try:
        if df_opt is not None and not df_opt.empty and "year" in df_opt.columns:
            years = sorted([int(x) for x in pd.to_numeric(df_opt["year"], errors="coerce").dropna().unique().tolist()])
    except Exception:
        years = []

    with c2:
        normalize_select_key(st.session_state, "rank_year_visual", [""] + [str(y) for y in years])
        rank_year = st.selectbox(
            "Ano",
            options=[""] + [str(y) for y in years],
            key="rank_year_visual",
            format_func=select_placeholder,
        )

    df_opt2 = df_opt
    if rank_year and df_opt2 is not None and not df_opt2.empty and "year" in df_opt2.columns:
        try:
            df_opt2 = df_opt2[pd.to_numeric(df_opt2["year"], errors="coerce") == int(rank_year)].copy()
        except Exception:
            pass

    splits = []
    try:
        if df_opt2 is not None and not df_opt2.empty and "split" in df_opt2.columns:
            splits = sorted([str(x) for x in df_opt2["split"].fillna("").astype(str).unique().tolist() if str(x)])
    except Exception:
        splits = []

    with c3:
        normalize_select_key(st.session_state, "rank_split_visual", ["", *splits])
        rank_split = st.selectbox(
            "Split",
            options=["", *splits],
            key="rank_split_visual",
            format_func=select_placeholder,
        )

    # Playoffs box appears ONLY when the chosen (liga, ano, split) has playoff
    # games — same pattern as the other tabs (scope_has_playoffs).
    if scope_has_playoffs(team_games, league=rank_league, year=rank_year, split=rank_split):
        with c4:
            normalize_select_key(st.session_state, "rank_po_visual", ["", "Regular", "Playoffs"])
            rank_playoffs = st.selectbox(
                "Playoffs",
                options=["", "Regular", "Playoffs"],
                key="rank_po_visual",
                format_func=select_placeholder,
            )
        rank_playoffs = st.session_state.get("rank_po_visual", "") or ""
    else:
        rank_playoffs = ""

    rank_filters = Filters(
        year=None if not rank_year else int(rank_year),
        league=None if not rank_league else rank_league,
        split=None if not rank_split else rank_split,
        playoffs=None if not rank_playoffs else (True if rank_playoffs == "Playoffs" else False),
    )

    render_tab_section(
        st,
        title="Ranking principal",
        subtitle="Escolha a fonte: ML core v2 (ratings do modelo) ou Elo (do CSV, cobertura total).",
    )

    # Fonte explícita (sem fallback silencioso) — você sempre sabe o que vê.
    _src_opts = {
        "ML core v2 (ratings do modelo)": "v2",
        "Elo (do CSV — cobertura total)": "elo",
    }
    _src_keys = list(_src_opts.keys())
    if str(st.session_state.get("rank_source", "") or "") not in _src_keys:
        st.session_state["rank_source"] = _src_keys[0]
    s1, s2 = st.columns([1.6, 1.0])
    with s1:
        _src_label = st.selectbox("Fonte do ranking", _src_keys, key="rank_source")
    rank_source = _src_opts.get(_src_label, "v2")
    if "rank_top_n" not in st.session_state:
        st.session_state["rank_top_n"] = 200
    with s2:
        top_n = st.slider("Top N", min_value=50, max_value=400, step=50, key="rank_top_n")

    def _apply_recorte(df: pd.DataFrame) -> pd.DataFrame:
        out = df
        if rank_league and ("league" in out.columns):
            out = out[out["league"].astype(str) == str(rank_league)].copy()
        if rank_year and ("year" in out.columns):
            try:
                out = out[pd.to_numeric(out["year"], errors="coerce") == int(rank_year)].copy()
            except Exception:
                pass
        if rank_split and ("split" in out.columns):
            out = out[out["split"].astype(str) == str(rank_split)].copy()
        if rank_playoffs and ("playoffs" in out.columns):
            try:
                want = str(rank_playoffs) == "Playoffs"
                po = out["playoffs"]
                ser = po if po.dtype == bool else pd.to_numeric(po, errors="coerce").fillna(0).astype(int).astype(bool)
                out = out[ser == want].copy()
            except Exception:
                pass
        return out

    if rank_source == "v2":
        df_v2 = _apply_recorte(mlcore_rank_table_v2_fn(artifact_path, as_of=as_of_date, top_n=int(max(int(top_n), 600))))
        if df_v2.empty:
            st.info("ML core v2 sem dados nesse recorte. Troque para **Elo** na fonte acima, ou ajuste o recorte.")
        else:
            _render_rank_grid(
                st, df_v2, format_num_fn=format_num_fn,
                metric_col="total_strength", metric_label="Força",
                show_tier=True, top_n=int(top_n),
            )
            with st.expander("Ver colunas técnicas (breakdown do ML v2)", expanded=False):
                st.caption("Decomposição da força: elo residual, força de liga/tier, base e boost por conquistas.")
                st.dataframe(df_v2.head(int(top_n)), width="stretch", height=420)
    else:
        try:
            df_elo = _apply_recorte(elo_rank_table_fn(
                team_games, filters=rank_filters, half_life_days=float(half_life_days),
                k_factor=24.0, base_rating=1500.0,
            ))
        except Exception:
            df_elo = pd.DataFrame()
        if df_elo.empty:
            st.info("Sem jogos suficientes nesse recorte para o Elo. Ajuste o recorte ou troque a fonte.")
        else:
            _render_rank_grid(
                st, df_elo, format_num_fn=format_num_fn,
                metric_col="elo", metric_label="Elo",
                show_tier=False, top_n=int(top_n),
            )

    # Tabela COMPLETA (todos os times do artefato, sem corte de Top N) — base
    # para a busca e a comparação. Times novos/fracos (ex.: Magaza, ~691º) não
    # aparecem no Top N mas EXISTEM no modelo; aqui dá p/ achá-los e compará-los.
    df_all = mlcore_rank_table_v2_fn(artifact_path, as_of=as_of_date, top_n=0)

    st.divider()
    render_tab_section(
        st,
        title="Buscar time",
        subtitle="Ache qualquer time pelo nome — força e posição completas, sem corte de Top N nem recorte.",
    )
    _q = st.text_input("Nome do time", key="rank_team_search", placeholder="ex: Magaza")
    if str(_q or "").strip():
        if df_all.empty:
            st.info("Sem ratings no artefato do modelo.")
        else:
            _qn = _norm_team_key(_q)
            _hits = df_all[df_all["team"].apply(lambda t: _qn in _norm_team_key(t))]
            if _hits.empty:
                st.info("Nenhum time encontrado com esse nome.")
            else:
                _scols = [c for c in ["rank", "team", "league", "tier", "total_strength"] if c in _hits.columns]
                st.dataframe(_hits[_scols].head(50), width="stretch", hide_index=True)

    render_tab_section(
        st,
        title="Comparar rating entre ligas",
        subtitle="Força de dois times lado a lado (cross-league). Começa com os times do confronto atual; troque à vontade.",
    )
    if df_all.empty:
        st.info("Sem ratings no artefato do modelo.")
    else:
        _all_teams = sorted(df_all["team"].astype(str).unique().tolist())

        def _match_team(name: str) -> str:
            _nn = _norm_team_key(name)
            for _t in _all_teams:
                if _norm_team_key(_t) == _nn:
                    return _t
            return ""

        # Default = times do confronto. Re-seed quando o confronto muda; entre
        # trocas de confronto, mantém a escolha manual do usuário.
        _seed = f"{team_a}|{team_b}"
        if st.session_state.get("_rank_cmp_seed") != _seed:
            st.session_state["_rank_cmp_seed"] = _seed
            _ma, _mb = _match_team(team_a), _match_team(team_b)
            if _ma:
                st.session_state["rank_cmp_a"] = _ma
            if _mb:
                st.session_state["rank_cmp_b"] = _mb
        for _ck in ("rank_cmp_a", "rank_cmp_b"):
            if st.session_state.get(_ck) not in _all_teams:
                st.session_state[_ck] = _all_teams[0]

        _c1, _c2 = st.columns(2)
        with _c1:
            _ta = st.selectbox("Time A", _all_teams, key="rank_cmp_a")
        with _c2:
            _tb = st.selectbox("Time B", _all_teams, key="rank_cmp_b")
        _render_rating_comparison(st, df_all, str(_ta), str(_tb), format_num_fn=format_num_fn)

    st.divider()
    with st.expander("Avançado — Elo (ajuste K / força por liga)", expanded=False):
        st.caption("Elo é atualizado jogo a jogo; jogos mais recentes valem mais (half-life).")
        c_a, c_b, c_c = st.columns([1, 1, 2])
        with c_a:
            k_factor = st.number_input("K (agressividade)", min_value=4.0, max_value=64.0, value=24.0, step=1.0)
        with c_b:
            base_rating = st.number_input("Base Elo", min_value=1000.0, max_value=2000.0, value=1500.0, step=10.0)
        with c_c:
            st.markdown("**Dica:** K menor = ranking mais estável | K maior = reage mais rápido.")

        elo_df = elo_rank_table_fn(
            team_games,
            filters=rank_filters,
            half_life_days=float(half_life_days),
            k_factor=float(k_factor),
            base_rating=float(base_rating),
        )
        if elo_df.empty:
            st.info("Sem jogos suficientes nesse recorte para calcular ranking.")
            return

        st.dataframe(elo_df.head(50), width="stretch", height=520)
        if "league" not in elo_df.columns or elo_df["league"].astype(str).str.len().sum() <= 0:
            return

        st.markdown("#### Força média por liga (no recorte)")
        try:
            elo_num = pd.to_numeric(elo_df["elo"], errors="coerce")
        except Exception:
            elo_num = pd.Series([None] * len(elo_df))
        tmp = elo_df.copy()
        tmp["elo_num"] = elo_num
        by_league = tmp.dropna(subset=["elo_num"]).groupby("league", dropna=False)["elo_num"].agg(["mean", "count"]).reset_index()
        by_league = by_league.sort_values("mean", ascending=False)
        by_league["mean"] = by_league["mean"].apply(lambda x: format_num_fn(x, 0))
        st.dataframe(by_league, width="stretch", height=260)
