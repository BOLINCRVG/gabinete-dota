"""StatsCamps tab renderer."""

from __future__ import annotations

import math
from html import escape as _esc
from typing import Any, Callable, Optional

import pandas as pd

from app_core.statscamps_state_core import (
    ensure_statscamps_filter_state,
    normalize_statscamps_select_state,
    reset_statscamps_league_dependents,
    sync_statscamps_delta_targets,
)
from app_core.tab_filters_core import collect_year_split_league_options, scope_has_playoffs
from app_core.tab_state_core import normalize_select_key, select_placeholder
from app_core.tab_ui_core import render_tab_hero, render_tab_section


def _sc_team_match_key(value: Any) -> str:
    """Normalize a team name for tolerant comparison (mojibake/casing safe)."""
    s = ("" if value is None else str(value)).strip()
    s = s.replace("�", "").replace("?", "")
    try:
        s = s.encode("ascii", "ignore").decode("ascii")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    return s.lower().strip()


def escolha_inicial_da_comparacao(
    *,
    opcoes: Any,
    time_a: Any,
    time_b: Any,
    ja_escolhido: Any,
    semeado_antes: Any,
) -> Optional[list]:
    """Quais times a comparacao deve ja vir marcando. None = nao mexer.

    Pedido do dono em 13/09/2026: a comparacao do StatsCamps comecava vazia e
    ele tinha de remarcar, na mao, os mesmos dois times que ja estavam
    escolhidos no Confronto logo acima.

    A regra tem de seguir o confronto SEM atropelar escolha feita a mao:
      - ninguem mexeu ainda  -> marca os dois do confronto;
      - trocou o confronto e a comparacao continuava como foi semeada
        -> acompanha;
      - o dono mexeu na comparacao (inclusive para esvaziar) -> nao toca mais.

    Time que nao esta na lista de opcoes (filtro de liga/ano/split diferente)
    NAO e forcado: o Streamlit recusaria o valor, e forcar seria mentir sobre
    o recorte. Comparacao e tolerante a caixa e a acento quebrado, como o
    resto desta tela.
    """
    disponiveis = [str(o) for o in (opcoes or [])]
    por_chave: dict = {}
    for o in disponiveis:
        por_chave.setdefault(_sc_team_match_key(o), o)

    alvo: list = []
    for t in (time_a, time_b):
        achado = por_chave.get(_sc_team_match_key(t))
        if achado and achado not in alvo:
            alvo.append(achado)
    if not alvo:
        return None

    if ja_escolhido is None:
        return alvo
    atual = [str(x) for x in (ja_escolhido or [])]
    semeado = [str(x) for x in (semeado_antes or [])]
    if atual == semeado and alvo != semeado:
        return alvo
    return None


def compute_statscamps_autofill(
    team_games: pd.DataFrame,
    team_a: str,
    team_b: str,
) -> dict:
    """Pure helper: decide Liga/Ano/Split pre-fill from the two selected teams.

    Rule: only pre-fill when BOTH teams' most-recent game is in the SAME league.
    Returns {"league","year","split"} (year/split as strings) when the two
    teams share a league; otherwise {} (caller leaves filters blank).

    Deterministic — no wall-clock, no side effects. Tolerant to mojibake in
    team names via _sc_team_match_key.
    """
    if team_games is None or getattr(team_games, "empty", True):
        return {}
    if not {"team", "date", "league"}.issubset(team_games.columns):
        return {}
    ka = _sc_team_match_key(team_a)
    kb = _sc_team_match_key(team_b)
    if not ka or not kb:
        return {}

    keys = team_games["team"].astype(str).map(_sc_team_match_key)

    def _latest_row(team_key: str):
        sub = team_games[keys == team_key].copy()
        if sub.empty:
            return None
        sub["_d"] = pd.to_datetime(sub["date"], errors="coerce")
        sub = sub.dropna(subset=["_d"]).sort_values("_d")
        if sub.empty:
            return None
        return sub.iloc[-1]

    row_a = _latest_row(ka)
    row_b = _latest_row(kb)
    if row_a is None or row_b is None:
        return {}
    lg_a = str(row_a.get("league", "") or "")
    lg_b = str(row_b.get("league", "") or "")
    if not lg_a or lg_a != lg_b:
        return {}  # different leagues → leave blank

    # Same league: pick the most recent game of either team in that league.
    sub = team_games[keys.isin({ka, kb}) & (team_games["league"].astype(str) == lg_a)].copy()
    sub["_d"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.dropna(subset=["_d"]).sort_values("_d")
    if sub.empty:
        return {"league": lg_a}
    last = sub.iloc[-1]
    out = {"league": lg_a}
    if pd.notna(last.get("year")):
        try:
            out["year"] = str(int(last.get("year")))
        except (TypeError, ValueError):
            pass
    sp = str(last.get("split", "") or "")
    if sp:
        out["split"] = sp
    return out


# Métricas mostradas no comparativo v3 (rótulo, coluna em df_tab_raw,
# "maior é melhor?" → define a cor do vencedor da métrica).
_SC_COMPARE_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("Win Rate", "Win Rate", True),
    ("Kills feitas", "Kills", True),
    ("Mortes", "Deaths", False),
    ("Torres feitas", "Towers", True),
    ("Combined Kills", "C. Kills", True),
    ("Combined Torres", "C. Towers", True),
    ("Combined Drakes", "C. Dragons", True),
    ("Combined Nashor", "C. Nashors", True),
    ("FB%", "FB%", True),
    ("FT%", "FT%", True),
)


def _sc_team_row(df_tab_raw: pd.DataFrame, team: str) -> Optional[dict]:
    """Return the raw (numeric) stats row for a team, tolerant to mojibake/casing."""
    if df_tab_raw is None or df_tab_raw.empty or "Team" not in df_tab_raw.columns:
        return None
    want = _sc_team_match_key(team)
    keys = df_tab_raw["Team"].astype(str).map(_sc_team_match_key)
    hit = df_tab_raw[keys == want]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def render_statscamps_compare_cards(
    *, st: Any, df_tab_raw: pd.DataFrame,
    team_a: str, team_b: str,
    min_to_mmss_fn: Callable[[Any], str],
    format_num_fn: Callable[[float, int], str],
) -> bool:
    """Render the v3-style comparative card for the two matchup teams.

    Returns True when rendered (both teams present in scope), False otherwise
    so the caller can decide whether to keep going. No-op when either team
    isn't in the current scope (e.g. cross-league matchup with blank filters).
    """
    if not team_a or not team_b:
        return False
    row_a = _sc_team_row(df_tab_raw, team_a)
    row_b = _sc_team_row(df_tab_raw, team_b)
    if row_a is None or row_b is None:
        return False

    # Cyan banner (estilo v3).
    st.markdown(
        "<div style='margin-top:6px;background:linear-gradient(180deg,#2daac4 0%,#1f8ba1 100%);"
        "padding:8px 14px;border-radius:8px 8px 0 0;font-weight:700;letter-spacing:0.05em;"
        "font-size:0.92em;color:white;text-transform:uppercase;'>"
        f"Comparativo do confronto · {_esc(str(team_a))} vs {_esc(str(team_b))}</div>",
        unsafe_allow_html=True,
    )

    def _num(v: Any) -> Optional[float]:
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    def _fmt(label: str, v: Optional[float]) -> str:
        if v is None:
            return "—"
        if label == "Win Rate" or label.endswith("%"):
            return f"{v:.0f}%"
        return format_num_fn(v, 1)

    with st.container(border=True):
        ga = int(_num(row_a.get("# Games")) or 0)
        gb = int(_num(row_b.get("# Games")) or 0)
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;font-size:0.85em;opacity:0.7;'>"
            f"<span><b style='color:#2daac4;'>{_esc(str(team_a))}</b> · {ga} jogos</span>"
            f"<span>{gb} jogos · <b style='color:#2daac4;'>{_esc(str(team_b))}</b></span></div>",
            unsafe_allow_html=True,
        )
        for label, col, higher_better in _SC_COMPARE_METRICS:
            va = _num(row_a.get(col))
            vb = _num(row_b.get(col))
            # Decide winner color.
            a_color, b_color = "#e5e7eb", "#e5e7eb"
            if va is not None and vb is not None and va != vb:
                a_wins = (va > vb) if higher_better else (va < vb)
                if a_wins:
                    a_color = "#22c55e"
                else:
                    b_color = "#22c55e"
            st.markdown(
                f"<div style='display:flex;align-items:center;padding:3px 0;"
                f"border-top:1px solid rgba(255,255,255,0.05);'>"
                f"<div style='flex:1;text-align:right;font-weight:700;color:{a_color};'>{_fmt(label, va)}</div>"
                f"<div style='flex:1.2;text-align:center;font-size:0.82em;opacity:0.65;"
                f"text-transform:uppercase;letter-spacing:0.03em;'>{_esc(label)}</div>"
                f"<div style='flex:1;text-align:left;font-weight:700;color:{b_color};'>{_fmt(label, vb)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    return True


def render_statscamps_tab(
    *,
    st: Any,
    team_games: pd.DataFrame,
    Filters: Any,
    apply_filters_fn: Callable[[pd.DataFrame, Any], pd.DataFrame],
    parse_lines_fn: Callable[[str], list[float]],
    parse_time_lines_fn: Callable[[str], list[float]],
    min_to_mmss_fn: Callable[[Any], str],
    format_num_fn: Callable[[float, int], str],
    df_contract_view_fn: Callable[..., pd.DataFrame],
) -> None:
    session_state = st.session_state
    render_tab_hero(
        st,
        title="StatsCamps",
        subtitle="Tabela estilo op.gg por campeonato/liga com comparacao entre times.",
        eyebrow="Scout",
    )
    render_tab_section(
        st,
        title="Recorte e thresholds",
        subtitle="Pré-preenchido pelos dois times do confronto quando jogam a mesma liga. Ajuste se quiser outro recorte.",
    )

    _, _, leagues = collect_year_split_league_options(team_games)
    base_cols = st.columns(4)
    ensure_statscamps_filter_state(session_state)

    # Auto-fill Liga/Ano/Split a partir dos dois times já escolhidos no topo.
    # Só preenche quando AMBOS jogam a mesma liga (liga do jogo mais recente
    # de cada um). Se forem de ligas diferentes, deixa vazio para o usuário
    # escolher manualmente. Re-dispara quando o matchup muda OU quando a liga
    # está em branco (recupera de resets).
    _sc_team_a = str(session_state.get("teamA", "") or "").strip()
    _sc_team_b = str(session_state.get("teamB", "") or "").strip()
    _sc_fp = f"{_sc_team_a}|{_sc_team_b}"
    _sc_last_fp = str(session_state.get("_sc_autofill_fp", "") or "")
    _sc_league_blank = not str(session_state.get("sc_league", "") or "").strip()
    _sc_need = (_sc_team_a and _sc_team_b and
                (_sc_fp != _sc_last_fp or _sc_league_blank))

    if _sc_need:
        try:
            _af = compute_statscamps_autofill(team_games, _sc_team_a, _sc_team_b)
            if _af.get("league"):
                session_state["sc_league"] = _af["league"]
                if _af.get("year"):
                    session_state["sc_year"] = _af["year"]
                if _af.get("split"):
                    session_state["sc_split"] = _af["split"]
            # Ligas diferentes / sem match → _af vazio: não força nada.
        finally:
            session_state["_sc_autofill_fp"] = _sc_fp

    def _on_sc_league_change() -> None:
        reset_statscamps_league_dependents(session_state)

    with base_cols[0]:
        normalize_select_key(session_state, "sc_league", ["", *leagues])
        sc_league = st.selectbox(
            "Liga",
            options=["", *leagues],
            key="sc_league",
            format_func=select_placeholder,
            on_change=_on_sc_league_change,
        )

    df_opt = team_games
    if sc_league and df_opt is not None and not df_opt.empty and "league" in df_opt.columns:
        df_opt = df_opt[df_opt["league"].astype(str) == str(sc_league)].copy()

    years_opt: list[int] = []
    try:
        if df_opt is not None and not df_opt.empty and "year" in df_opt.columns:
            years_opt = sorted([int(x) for x in pd.to_numeric(df_opt["year"], errors="coerce").dropna().unique().tolist()])
    except Exception:
        years_opt = []

    with base_cols[1]:
        # Normalizar SÓ o Ano aqui. NÃO chamar o normalize completo com splits=[]:
        # isso validava sc_split contra [""] e ZERAVA o split escolhido a cada
        # rerun (bug: "seleciono o split e ele ignora"). O split é normalizado
        # mais abaixo (linha ~301) já com as opções reais (splits_opt).
        normalize_select_key(session_state, "sc_year", ["", *[str(y) for y in years_opt]])
        sc_year = st.selectbox(
            "Ano",
            options=[""] + [str(y) for y in years_opt],
            key="sc_year",
            format_func=select_placeholder,
        )

    df_opt2 = df_opt
    if sc_year and df_opt2 is not None and not df_opt2.empty and "year" in df_opt2.columns:
        try:
            df_opt2 = df_opt2[pd.to_numeric(df_opt2["year"], errors="coerce") == int(sc_year)].copy()
        except Exception:
            pass

    splits_opt: list[str] = []
    try:
        if df_opt2 is not None and not df_opt2.empty and "split" in df_opt2.columns:
            splits_opt = sorted([str(x) for x in df_opt2["split"].fillna("").astype(str).unique().tolist() if str(x)])
    except Exception:
        splits_opt = []

    with base_cols[2]:
        normalize_statscamps_select_state(session_state, years=[str(y) for y in years_opt], splits=splits_opt)
        if splits_opt:
            sc_split = st.selectbox(
                "Split",
                options=["", *splits_opt],
                key="sc_split",
                format_func=select_placeholder,
            )
        else:
            # CAMPEONATO SEM SPLIT (25/10/2026): o Mundial (WLDs) vem sem split
            # no CSV; a aba pedia "Selecione um split" com a lista vazia e
            # ficava travada. Sem split nos dados, usa todos os jogos do ano.
            st.selectbox("Split", options=["sem split (campeonato unico)"], disabled=True,
                         key="sc_split_vazio")
            sc_split = ""

    # Playoffs box appears ONLY when the chosen (liga, ano, split) has playoff
    # games — same pattern as the other tabs (scope_has_playoffs).
    if scope_has_playoffs(team_games, league=sc_league, year=sc_year, split=sc_split):
        with base_cols[3]:
            normalize_statscamps_select_state(session_state, years=[str(y) for y in years_opt], splits=splits_opt)
            sc_playoffs = st.selectbox(
                "Playoffs",
                options=["", "Regular", "Playoffs"],
                key="sc_playoffs",
                format_func=select_placeholder,
            )
        sc_playoffs = session_state.get("sc_playoffs", "") or ""
    else:
        sc_playoffs = ""

    sc_ready = True
    if not sc_league:
        st.info("Selecione uma liga para exibir StatsCamps.")
        sc_ready = False
    if not sc_year:
        st.info("Selecione um ano para exibir StatsCamps.")
        sc_ready = False
    if not sc_split and splits_opt:
        st.info("Selecione um split para exibir StatsCamps.")
        sc_ready = False
    if not sc_ready:
        return

    map_opts = ["All maps", "Map 1", "Map 2", "Map 3", "Map 4", "Map 5"]
    c_m1, c_m2 = st.columns([1, 2])
    with c_m1:
        normalize_select_key(session_state, "sc_map", map_opts, fallback=map_opts[0])
        sc_map = st.selectbox("Mapa", options=map_opts, key="sc_map")
    with c_m2:
        st.caption("As medias e percentuais são por mapa; cada linha do CSV representa um mapa.")

    ct1, ct2, ct3 = st.columns(3)
    with ct1:
        th_kills = st.text_input(
            "KILLS thresholds (C. Kills) — ex: 23.5,25.5,27.5,29.5",
            key="sc_th_kills",
        )
        th_towers = st.text_input(
            "TOWERS thresholds (C. Towers) — ex: 10.5,11.5,12.5",
            key="sc_th_towers",
        )
    with ct2:
        th_dragons = st.text_input(
            "DRAGONS thresholds (C. Dragons) — ex: 4.5,5.5",
            key="sc_th_dragons",
        )
        th_barons = st.text_input(
            "NASHORS thresholds (C. Nashors) — ex: 1.5",
            key="sc_th_barons",
        )
    with ct3:
        th_inhib = st.text_input(
            "INHIB thresholds (C. Inhib) — ex: 1.5",
            key="sc_th_inhib",
        )
        th_time = st.text_input(
            "TEMPO thresholds (Game Time) — ex: 32:30, 35, 28.5",
            key="sc_th_time",
            help="Aceita 32:30, 32.5 ou 32. O padrão é 32:30.",
        )

    sc_filters = Filters(
        year=None if not sc_year else int(sc_year),
        league=None if not sc_league else sc_league,
        split=None if not sc_split else sc_split,
        playoffs=None if not sc_playoffs else (True if sc_playoffs == "Playoffs" else False),
    )
    df_sc = apply_filters_fn(team_games, sc_filters).copy()
    if sc_map != "All maps" and "map_number" in df_sc.columns:
        try:
            map_number = int(sc_map.split()[-1])
            df_sc = df_sc[df_sc["map_number"] == map_number].copy()
        except Exception:
            pass
    if df_sc.empty:
        st.info("Sem jogos nesse recorte.")
        return

    def _pct(series_bool: pd.Series) -> float:
        try:
            return 100.0 * float(series_bool.mean())
        except Exception:
            return float("nan")

    # A linha digitada SOMA aos padroes (pedido dele 24/07): digitar 33.5 nao pode
    # apagar as colunas 23.5/25.5/... — so adiciona a nova.
    from app_core.stats_config import STATS_MARKET_CONFIG as _SMC

    def _th_tempo(txt: Any) -> list:
        """Linhas de tempo, que aceitam 32:30 alem de 32.5.

        O parser comum nao entende "32:30" — ele quebraria em 32 e 30, e o
        dono ia digitar o horario e ver duas linhas erradas aparecerem.
        """
        cfg = _SMC.get("Tempo")
        base = parse_time_lines_fn(str(cfg["default_thresholds"])) if cfg else []
        return sorted(set(list(base) + list(parse_time_lines_fn(txt))))

    def _th(txt: Any, market: str) -> list:
        # .get, nao [] (26/08/2026): o dota_overrides troca "Dragões" por
        # "Megacreeps" no registro, e esta aba pedia a chave literal — a aba
        # inteira estourava com KeyError desde 10/07 e ninguem viu, porque os
        # testes de UI de la nem chegavam a rodar (procuravam plays_app.py).
        # Mercado ausente = so as linhas digitadas, coluna vazia. E o que ele
        # decidiu pro Dota: "dragoes fica na UI (vazio) ate mandar tirar".
        cfg = _SMC.get(market)
        base = parse_lines_fn(str(cfg["default_thresholds"])) if cfg else []
        return sorted(set(list(base) + list(parse_lines_fn(txt))))

    th_k = _th(th_kills, "Kills")
    th_t = _th(th_towers, "Torres")
    th_d = _th(th_dragons, "Dragões")
    th_b = _th(th_barons, "Nashors")
    th_i = _th(th_inhib, "Inibidores")
    th_tm = _th_tempo(th_time)
    agg_rows = []
    for team_name, group in df_sc.groupby("team"):
        group = group.copy()
        games_n = int(group.shape[0])
        win_rate = float(group["win"].mean()) if games_n else float("nan")
        row = {
            "Team": team_name,
            "# Games": games_n,
            "Win Rate": 100.0 * win_rate if math.isfinite(win_rate) else float("nan"),
            "Game Time": float(group["game_time_min"].mean()) if "game_time_min" in group.columns else float("nan"),
            "FB%": _pct(group.get("firstblood", pd.Series([0] * games_n)).astype(float) > 0.5),
            "FRH%": _pct(group.get("firstherald", pd.Series([0] * games_n)).astype(float) > 0.5),
            "FT%": _pct(group.get("firsttower", pd.Series([0] * games_n)).astype(float) > 0.5),
            "FD%": _pct(group.get("firstdragon", pd.Series([0] * games_n)).astype(float) > 0.5),
            "FNASH%": _pct(group.get("firstbaron", pd.Series([0] * games_n)).astype(float) > 0.5),
            "Gold": float(group["gold_for"].mean()) if "gold_for" in group.columns else float("nan"),
            "Kills": float(group["kills_for"].mean()) if "kills_for" in group.columns else float("nan"),
            "Deaths": float(group["deaths_for"].mean()) if "deaths_for" in group.columns else float("nan"),
            "Towers": float(group["towers_for"].mean()) if "towers_for" in group.columns else float("nan"),
            "C. Gold": float(group["total_gold"].mean()) if "total_gold" in group.columns else float("nan"),
            "C. Kills": float(group["total_kills"].mean()) if "total_kills" in group.columns else float("nan"),
            "C. Towers": float(group["total_towers"].mean()) if "total_towers" in group.columns else float("nan"),
            "C. Dragons": float(group["total_dragons"].mean()) if "total_dragons" in group.columns else float("nan"),
            "C. Nashors": float(group["total_nashors"].mean()) if "total_nashors" in group.columns else float("nan"),
            "C. Inhib": float(group["total_inhibitors"].mean()) if "total_inhibitors" in group.columns else float("nan"),
        }
        for line in th_k:
            row[f"% Kills > {line}"] = _pct(pd.to_numeric(group["total_kills"], errors="coerce") > float(line))
        for line in th_t:
            row[f"% Towers > {line}"] = _pct(pd.to_numeric(group["total_towers"], errors="coerce") > float(line))
        for line in th_d:
            row[f"% Dragons > {line}"] = _pct(pd.to_numeric(group["total_dragons"], errors="coerce") > float(line))
        for line in th_b:
            row[f"% Nashors > {line}"] = _pct(pd.to_numeric(group["total_nashors"], errors="coerce") > float(line))
        for line in th_i:
            row[f"% Inhib > {line}"] = _pct(pd.to_numeric(group["total_inhibitors"], errors="coerce") > float(line))
        for line in th_tm:
            # O rotulo mostra o horario (32:30), nao o decimal (32.5) — e como
            # a linha e vendida na casa.
            row[f"% Tempo > {min_to_mmss_fn(float(line))}"] = _pct(
                pd.to_numeric(group.get("game_time_min"), errors="coerce") > float(line)
            )
        agg_rows.append(row)

    df_tab = pd.DataFrame(agg_rows)
    df_tab_raw = df_tab.copy()

    if not df_tab.empty:
        df_tab = df_tab.sort_values(["Win Rate", "# Games"], ascending=[False, False])
        if "Game Time" in df_tab.columns:
            df_tab["Game Time"] = df_tab["Game Time"].apply(min_to_mmss_fn)
        for col in df_tab.columns:
            if col in ["Team", "Game Time", "# Games"]:
                continue
            df_tab[col] = df_tab[col].apply(
                lambda x: format_num_fn(x, 1) if x is not None and math.isfinite(float(x)) else "-"
            )

    render_tab_section(
        st, title="Betting stats do campeonato",
        subtitle="Todas as colunas (estilo gol.gg) — as % usam as SUAS linhas dos campos acima. Role pro lado pra ver tudo.",
    )
    # Ordem gol.gg; as colunas "% ..." (linhas dele) vao no fim, na ordem gerada.
    _order = [c for c in ("Team", "# Games", "Win Rate", "Game Time", "FB%", "FRH%", "FT%",
                          "FD%", "FNASH%", "Gold", "Kills", "Deaths", "Towers", "C. Gold",
                          "C. Kills", "C. Towers", "C. Dragons", "C. Nashors", "C. Inhib")
              if c in df_tab.columns]
    _order += [c for c in df_tab.columns if str(c).startswith("% ") and c not in _order]
    st.dataframe(df_tab[_order], width="stretch", height=600, hide_index=True)

    render_tab_section(
        st,
        title="Comparar times",
        subtitle="Selecione 2 ou mais times para ver a tabela lado a lado e o delta por metrica.",
    )
    all_teams_sc = df_tab_raw["Team"].tolist() if "Team" in df_tab_raw.columns else []

    # A comparacao ja nasce com os dois times do Confronto. Tem de ser ANTES do
    # widget: o Streamlit so aceita escrever na chave dele antes de ele existir
    # no desenho. `_sc_compare_semeado` guarda o que foi semeado, e e o que
    # permite saber se o dono mexeu na mao depois.
    _alvo_cmp = escolha_inicial_da_comparacao(
        opcoes=all_teams_sc,
        time_a=session_state.get("teamA"),
        time_b=session_state.get("teamB"),
        ja_escolhido=session_state.get("sc_compare_sel"),
        semeado_antes=session_state.get("_sc_compare_semeado"),
    )
    if _alvo_cmp is not None:
        session_state["sc_compare_sel"] = list(_alvo_cmp)
        session_state["_sc_compare_semeado"] = list(_alvo_cmp)

    selected = st.multiselect("Selecione 2+ times", options=all_teams_sc, key="sc_compare_sel")
    if not selected or len(selected) < 2:
        return

    tab_cmp, tab_delta = st.tabs(["Tabela", "Delta (comparacao)"])
    with tab_cmp:
        # Cards SEPARADOS por time (sem Gold/C.Gold) + so as % das linhas digitadas.
        _card_metrics = ["Win Rate", "Game Time", "Kills", "Deaths", "Towers",
                         "C. Kills", "C. Towers", "C. Dragons", "C. Nashors",
                         "FB%", "FRH%", "FT%", "FD%", "FNASH%"]

        def _cell(m: str, v: Any) -> str:
            try:
                fv = float(v)
                if not math.isfinite(fv):
                    return "-"
            except (TypeError, ValueError):
                return "-"
            if m == "Game Time":
                return min_to_mmss_fn(fv)
            if m == "Win Rate" or str(m).endswith("%"):
                return f"{fv:.0f}%"
            return format_num_fn(fv, 1)

        def _row_of(tm: str):
            r = df_tab_raw[df_tab_raw["Team"].astype(str) == str(tm)]
            return r.iloc[0] if not r.empty else None

        # Paleta categórica por time — validada (CVD ΔE 14+, banda de luminância
        # e contraste no fundo escuro): scripts/validate_palette.js do skill dataviz.
        _PAL = ["#2695ad", "#8b5cf6", "#c2760a", "#4d9e4d", "#c05a8a", "#7a8fb8"]

        def _num_of(tm: str, m: str):
            _r = _row_of(tm)
            try:
                v = float(_r.get(m)) if _r is not None else float("nan")
                return v if math.isfinite(v) else None
            except (TypeError, ValueError):
                return None

        # "melhor" por métrica (Deaths e Game Time: menor é melhor) — destaque
        # sem depender só de cor (o valor vencedor fica em negrito + ponto).
        _lower_better = {"Deaths", "Game Time"}

        for _i, (_co, _tm) in enumerate(zip(st.columns(len(selected)), selected)):
            _r = _row_of(_tm)
            if _r is None:
                continue
            _cor = _PAL[_i % len(_PAL)]
            with _co, st.container(border=True):
                st.markdown(
                    f"<div style='border-left:3px solid {_cor};padding-left:8px;margin-bottom:6px'>"
                    f"<div style='font-weight:700;font-size:1.05em;line-height:1.2'>{_tm}</div>"
                    f"<div style='opacity:.55;font-size:.78em'>{int(_r.get('# Games') or 0)} jogos</div></div>",
                    unsafe_allow_html=True)
                for _m in _card_metrics:
                    if _m not in _r:
                        continue
                    _vals = [v for v in (_num_of(t, _m) for t in selected) if v is not None]
                    _mine = _num_of(_tm, _m)
                    _best = (_mine is not None and len(_vals) > 1 and
                             (_mine == (min(_vals) if _m in _lower_better else max(_vals))))
                    _mark = f"<span style='color:{_cor}'>●</span> " if _best else ""
                    _w = "700" if _best else "500"
                    st.markdown(
                        f"<div style='font-size:.85em;display:flex;justify-content:space-between;"
                        f"padding:2px 0;border-top:1px solid rgba(255,255,255,.04)'>"
                        f"<span style='opacity:.55'>{_m}</span>"
                        f"<span style='font-weight:{_w}'>{_mark}{_cell(_m, _r.get(_m))}</span></div>",
                        unsafe_allow_html=True)

        _line_cols = [c for c in df_tab_raw.columns if str(c).startswith("% ")]
        if _line_cols:
            st.markdown("##### Comparativo das linhas")
            # Barras (magnitude entre entidades) em vez de tabela de texto: dá pra
            # ler quem bate mais a linha de relance. Legenda direta + valor ao lado
            # da barra (identidade nunca só por cor).
            _leg = " · ".join(
                f"<span style='color:{_PAL[i % len(_PAL)]}'>●</span> "
                f"<span style='opacity:.75'>{_esc(str(t))}</span>"
                for i, t in enumerate(selected))
            st.markdown(f"<div style='font-size:.82em;margin-bottom:6px'>{_leg}</div>",
                        unsafe_allow_html=True)
            _html = ["<div style='display:flex;flex-direction:column;gap:7px'>"]
            for _lc in _line_cols:
                _label = str(_lc).replace("% ", "")
                _html.append(
                    "<div style='display:flex;align-items:center;gap:10px'>"
                    f"<div style='width:120px;font-size:.82em;opacity:.7;text-align:right'>{_esc(_label)}</div>"
                    "<div style='flex:1;display:flex;flex-direction:column;gap:2px'>")
                for _i, _tm in enumerate(selected):
                    _v = _num_of(_tm, _lc)
                    _cor = _PAL[_i % len(_PAL)]
                    _pct = max(0.0, min(100.0, float(_v))) if _v is not None else 0.0
                    _txt = f"{_pct:.0f}%" if _v is not None else "-"
                    _html.append(
                        "<div style='display:flex;align-items:center;gap:8px' "
                        f"title='{_esc(str(_tm))} · {_esc(_label)}: {_txt}'>"
                        "<div style='flex:1;height:9px;background:rgba(255,255,255,.05);border-radius:4px'>"
                        f"<div style='width:{_pct:.1f}%;height:100%;background:{_cor};border-radius:4px'></div></div>"
                        f"<div style='width:38px;font-size:.8em;font-weight:600'>{_txt}</div></div>")
                _html.append("</div></div>")
            _html.append("</div>")
            st.markdown("".join(_html), unsafe_allow_html=True)

    with tab_delta:
        c_a, c_b = st.columns(2)
        default_a = selected[0] if selected else ""
        default_b = selected[1] if len(selected) > 1 else ""
        current_a, current_b = sync_statscamps_delta_targets(
            session_state,
            selected=selected,
            default_a=default_a,
            default_b=default_b,
        )

        with c_a:
            if str(session_state.get("sc_delta_a", "") or "") not in selected:
                session_state["sc_delta_a"] = current_a if current_a in selected else (selected[0] if selected else "")
            team_a = st.selectbox(
                "Time A",
                options=selected,
                key="sc_delta_a",
            )
        opts_b = [t for t in selected if t != team_a] or selected
        with c_b:
            cand_b = current_b
            if cand_b == team_a:
                cand_b = opts_b[0] if opts_b else ""
            if cand_b not in opts_b and opts_b:
                cand_b = opts_b[0]
            if str(session_state.get("sc_delta_b", "") or "") not in opts_b:
                session_state["sc_delta_b"] = cand_b
            team_b = st.selectbox(
                "Time B",
                options=opts_b,
                key="sc_delta_b",
            )

        if (not team_a) or (not team_b) or (team_a == team_b):
            st.info("Selecione dois times diferentes para ver Delta.")
            return

        try:
            row_a = df_tab_raw.loc[df_tab_raw["Team"] == team_a].iloc[0]
            row_b = df_tab_raw.loc[df_tab_raw["Team"] == team_b].iloc[0]
        except Exception:
            row_a, row_b = None, None

        def _is_num(value: Any) -> bool:
            try:
                return value is not None and math.isfinite(float(value))
            except Exception:
                return False

        def _fmt_val(col: str, value: Any) -> str:
            if col == "Game Time":
                return min_to_mmss_fn(value) if _is_num(value) else "-"
            if col == "# Games":
                try:
                    return str(int(value)) if _is_num(value) else "-"
                except Exception:
                    return "-"
            return format_num_fn(value, 1) if _is_num(value) else "-"

        def _fmt_delta_num(col: str, a: Any, b: Any) -> str:
            if not _is_num(a) or not _is_num(b):
                return "-"
            delta = float(a) - float(b)
            if col == "# Games":
                return f"{round(delta):+d}"
            return f"{delta:+.1f}"

        def _fmt_delta_time(a: Any, b: Any) -> str:
            if not _is_num(a) or not _is_num(b):
                return "-"
            seconds = round((float(a) - float(b)) * 60.0)
            sign = "+" if seconds > 0 else "-" if seconds < 0 else ""
            seconds = abs(seconds)
            mm = seconds // 60
            ss = seconds % 60
            return f"{sign}{mm:02d}:{ss:02d}"

        if row_a is None or row_b is None:
            st.warning("Não consegui montar Delta para esse par (dados faltando).")
            return

        # Layout A | métrica | B com o vencedor destacado (mesma linguagem visual
        # dos cards): identidade por cor E por negrito, delta ao lado. Cores da
        # paleta validada (validate_palette.js: contraste + CVD no fundo escuro).
        _CA, _CB = "#2695ad", "#8b5cf6"
        # métricas onde MENOR é melhor
        _LOWER = {"Deaths", "Game Time"}
        # neutras: não faz sentido "vencer" (amostra/contexto)
        _NEUTRAL = {"# Games"}

        st.markdown(
            "<div style='display:flex;align-items:center;padding:6px 10px;margin-top:4px;"
            "border-radius:8px 8px 0 0;background:rgba(255,255,255,.04);font-size:.8em;"
            "text-transform:uppercase;letter-spacing:.05em;opacity:.75'>"
            f"<div style='flex:1;text-align:right;color:{_CA};font-weight:700'>{_esc(str(team_a))}</div>"
            "<div style='flex:1.1;text-align:center'>métrica</div>"
            f"<div style='flex:1;text-align:left;color:{_CB};font-weight:700'>{_esc(str(team_b))}</div>"
            "<div style='width:88px;text-align:right'>delta</div></div>",
            unsafe_allow_html=True,
        )

        _html = []
        for col in df_tab_raw.columns:
            if col in ("Team", "Gold", "C. Gold"):  # gold nao ajuda a decidir linha (pedido dele)
                continue
            a_val, b_val = row_a.get(col), row_b.get(col)
            d_val = _fmt_delta_time(a_val, b_val) if col == "Game Time" else _fmt_delta_num(col, a_val, b_val)

            _wa = _wb = False
            if col not in _NEUTRAL and _is_num(a_val) and _is_num(b_val) and float(a_val) != float(b_val):
                _a_wins = (float(a_val) < float(b_val)) if col in _LOWER else (float(a_val) > float(b_val))
                _wa, _wb = _a_wins, not _a_wins
            _sa = f"color:{_CA};font-weight:700" if _wa else "opacity:.75"
            _sb = f"color:{_CB};font-weight:700" if _wb else "opacity:.75"
            _dc = "#9ca3af" if col in _NEUTRAL or not (_wa or _wb) else (_CA if _wa else _CB)
            _html.append(
                "<div style='display:flex;align-items:center;padding:5px 10px;"
                "border-top:1px solid rgba(255,255,255,.05);font-size:.9em'>"
                f"<div style='flex:1;text-align:right;{_sa}'>{_esc(_fmt_val(col, a_val))}</div>"
                f"<div style='flex:1.1;text-align:center;opacity:.5;font-size:.86em'>{_esc(str(col))}</div>"
                f"<div style='flex:1;text-align:left;{_sb}'>{_esc(_fmt_val(col, b_val))}</div>"
                f"<div style='width:88px;text-align:right;color:{_dc};font-size:.85em'>{_esc(str(d_val))}</div></div>"
            )
        st.markdown("".join(_html), unsafe_allow_html=True)
