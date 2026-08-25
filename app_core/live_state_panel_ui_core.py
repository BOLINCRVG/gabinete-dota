"""UI panel for calibrated live-bet predictions.

Default mode requires only what's already on the live-bet tab (the two
teams, the league, and the 10 draft picks). The headline P(team A wins)
is taken from the pre-game pipeline (`session_state["_p_map_used"]` or
`ctx["p_map_pre"]`), which already incorporates draft, roster, BO and
league context.

Market baselines (kills/towers/dragons/duration) are produced by the
Phase 7 LightGBM models evaluated at minute 10 with neutral state, so
the user gets sensible totals + over/under probabilities without
typing anything else.

An optional "Modo avançado" expander reveals the live state inputs
(minute, gold diff, kills, towers, dragons, ...) so the user can refine
the prediction with the Phase 6/7 state-aware models when they want.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping, Optional

from app_core.live_bet_analysis_core import build_live_bet_live_assessment


_SUPPORTED_MINUTES = (10, 15, 20, 25)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _state_key(team_a: str, team_b: str, field: str) -> str:
    """Build a stable session_state key scoped to the current matchup."""
    return f"lb_live_{field}_{str(team_a)[:30]}_{str(team_b)[:30]}"


def _init_state(session_state: MutableMapping[str, Any], team_a: str, team_b: str) -> None:
    """Initialize default values for all advanced-mode state inputs."""
    defaults = {
        "minute": 15.0,
        "gold_diff": 0.0,
        "xp_diff": 0.0,
        "cs_diff": 0.0,
        "kills_a": 0,
        "kills_b": 0,
        "deaths_a": 0,
        "deaths_b": 0,
        "assists_a": 0,
        "assists_b": 0,
        "towers_a": 0,
        "towers_b": 0,
        "dragons_a": 0,
        "dragons_b": 0,
        "barons_a": 0,
        "barons_b": 0,
        "inhibitors_a": 0,
        "inhibitors_b": 0,
    }
    for field, value in defaults.items():
        key = _state_key(team_a, team_b, field)
        if key not in session_state:
            session_state[key] = value


def _state_is_filled(session_state: Mapping[str, Any], team_a: str, team_b: str) -> bool:
    """True if the user typed something in the advanced state inputs (any non-default)."""
    minute = float(session_state.get(_state_key(team_a, team_b, "minute"), 0.0) or 0.0)
    if minute > 0.0 and minute != 15.0:
        return True
    for f in ("gold_diff", "xp_diff", "cs_diff",
              "kills_a", "kills_b", "deaths_a", "deaths_b",
              "assists_a", "assists_b",
              "towers_a", "towers_b", "dragons_a", "dragons_b",
              "barons_a", "barons_b", "inhibitors_a", "inhibitors_b"):
        v = session_state.get(_state_key(team_a, team_b, f), 0)
        try:
            if float(v) != 0.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def render_live_state_panel(
    *,
    st: Any,
    session_state: MutableMapping[str, Any],
    team_a: str,
    team_b: str,
    p_map_pre: float | None = None,
    league: str = "",
    bo: int = 1,
    is_playoffs: bool = False,
    comp_compare: Mapping[str, Any] | None = None,
    pregame_resolution: Mapping[str, Any] | None = None,
    format_num_fn: Optional[Any] = None,
) -> dict:
    """Render the calibrated prediction panel.

    Default view shows P(team_a wins) from the pre-game pipeline plus
    market baselines computed at minute=10 with neutral state. The
    advanced expander reveals live-state inputs that refine the
    prediction through the Phase 6/7 models.
    """
    if not str(team_a).strip() or not str(team_b).strip():
        st.info("Selecione os dois times no topo para liberar a predição calibrada.")
        return {}

    _init_state(session_state, team_a, team_b)

    _render_freshness_banner(
        st=st, session_state=session_state,
        # sufixo por confronto: o Radar multi-jogo renderiza este painel em
        # várias abas — chave fixa duplicaria o widget e derrubaria a página
        key_suffix=f"_{str(team_a)[:16]}_{str(team_b)[:16]}",
    )

    fmt_num = format_num_fn if callable(format_num_fn) else (lambda v, d=1: f"{float(v):.{int(d)}f}")

    p_pre_val = _safe_float(p_map_pre, 0.5)
    if not (0.0 < p_pre_val < 1.0):
        p_pre_val = 0.5
    p_pre_provided = p_map_pre is not None and 0.0 < _safe_float(p_map_pre, -1.0) < 1.0

    advanced_active = _state_is_filled(session_state, team_a, team_b)

    assessment: dict[str, Any] = {}
    state_values: dict[str, float] = {}
    if advanced_active:
        with st.container():
            st.markdown("### Estado do jogo ao vivo")
            st.caption(
                "Modo avançado ativo — o modelo recalcula a predição usando os snapshots "
                "@10/15/20/25 minutos. Para voltar à leitura pré-jogo apenas, zere os campos."
            )
            state_values = _render_state_inputs(st=st, session_state=session_state, team_a=team_a, team_b=team_b)

        try:
            assessment = build_live_bet_live_assessment(
                team_a=str(team_a), team_b=str(team_b),
                df_scope=None,
                minute=state_values["minute"],
                gold_diff=state_values["gold_diff"],
                kills_a=state_values["kills_a"], kills_b=state_values["kills_b"],
                towers_a=state_values["towers_a"], towers_b=state_values["towers_b"],
                dragons_a=state_values["dragons_a"], dragons_b=state_values["dragons_b"],
                barons_a=state_values["barons_a"], barons_b=state_values["barons_b"],
                inhibitors_a=state_values["inhibitors_a"], inhibitors_b=state_values["inhibitors_b"],
                comp_compare=comp_compare or {},
                pregame_resolution=pregame_resolution or {},
                baselines=None,
                session_state=session_state,
                p_map_pre=p_pre_val,
                league=str(league or ""),
                bo=int(bo or 1),
                is_playoffs=bool(is_playoffs),
                xp_diff=state_values["xp_diff"], cs_diff=state_values["cs_diff"],
                deaths_a=state_values["deaths_a"], deaths_b=state_values["deaths_b"],
                assists_a=state_values["assists_a"], assists_b=state_values["assists_b"],
            )
        except Exception as exc:
            st.error(f"Falha ao calcular a análise live: {type(exc).__name__}: {exc}")
            assessment = {}

    _render_headline(
        st=st,
        team_a=team_a,
        team_b=team_b,
        p_pre_val=p_pre_val,
        p_pre_provided=p_pre_provided,
        assessment=assessment,
        fmt_num=fmt_num,
    )

    minute_for_markets = float(state_values.get("minute", 10.0)) if advanced_active else 10.0
    _render_live_markets_section(
        st=st, session_state=session_state,
        minute=minute_for_markets,
        p_map_pre=p_pre_val, league=league,
        bo=int(bo or 1), is_playoffs=bool(is_playoffs),
        gold_diff=state_values.get("gold_diff", 0.0),
        xp_diff=state_values.get("xp_diff", 0.0),
        cs_diff=state_values.get("cs_diff", 0.0),
        kills_a=state_values.get("kills_a", 0.0),
        kills_b=state_values.get("kills_b", 0.0),
        deaths_a=state_values.get("deaths_a", 0.0),
        deaths_b=state_values.get("deaths_b", 0.0),
        assists_a=state_values.get("assists_a", 0.0),
        assists_b=state_values.get("assists_b", 0.0),
        team_a=str(team_a), team_b=str(team_b),
        advanced_active=advanced_active,
    )

    _render_first_markets_section(
        st=st, session_state=session_state,
        p_map_pre=p_pre_val, league=league,
        bo=int(bo or 1), is_playoffs=bool(is_playoffs),
        team_a=str(team_a), team_b=str(team_b),
    )

    if not advanced_active:
        with st.expander("Modo avançado — refinar com estado ao vivo", expanded=False):
            st.caption(
                "Preencha o estado atual da partida (minuto, gold diff, kills, etc.) "
                "para o modelo refinar a predição com os snapshots @10/15/20/25 minutos."
            )
            state_values = _render_state_inputs(st=st, session_state=session_state, team_a=team_a, team_b=team_b)

    if assessment:
        summary = str(assessment.get("summary") or "")
        if summary:
            st.markdown(f"**Leitura:** {summary}")
        conversion = str(assessment.get("conversion") or "")
        reversal = str(assessment.get("reversal_risk") or "")
        if conversion or reversal:
            c_left, c_right = st.columns(2)
            with c_left:
                if conversion:
                    st.markdown(f"**Conversão:** {conversion}")
            with c_right:
                if reversal:
                    st.markdown(f"**Risco de reversão:** {reversal}")

    return assessment


def _render_state_inputs(
    *,
    st: Any,
    session_state: MutableMapping[str, Any],
    team_a: str,
    team_b: str,
) -> dict[str, float]:
    """Render the full grid of advanced state inputs. Returns the values dict."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        minute_val = st.number_input(
            "Minuto", min_value=0.0, max_value=60.0, step=1.0,
            value=float(session_state[_state_key(team_a, team_b, "minute")]),
            key=_state_key(team_a, team_b, "minute"),
            help="Minuto atual do jogo (recomendado: 10, 15, 20 ou 25).",
        )
    with c2:
        gold_diff = st.number_input(
            f"Gold diff ({team_a} − {team_b})",
            min_value=-30000.0, max_value=30000.0, step=100.0,
            value=float(session_state[_state_key(team_a, team_b, "gold_diff")]),
            key=_state_key(team_a, team_b, "gold_diff"),
        )
    with c3:
        xp_diff = st.number_input(
            f"XP diff ({team_a} − {team_b})",
            min_value=-30000.0, max_value=30000.0, step=100.0,
            value=float(session_state[_state_key(team_a, team_b, "xp_diff")]),
            key=_state_key(team_a, team_b, "xp_diff"),
        )
    with c4:
        cs_diff = st.number_input(
            f"CS diff ({team_a} − {team_b})",
            min_value=-500.0, max_value=500.0, step=5.0,
            value=float(session_state[_state_key(team_a, team_b, "cs_diff")]),
            key=_state_key(team_a, team_b, "cs_diff"),
        )

    kc1, kc2, kc3, kc4, kc5, kc6 = st.columns(6)
    with kc1:
        kills_a = st.number_input(
            f"Kills {team_a}", min_value=0, max_value=100, step=1,
            value=int(session_state[_state_key(team_a, team_b, "kills_a")]),
            key=_state_key(team_a, team_b, "kills_a"),
        )
    with kc2:
        kills_b = st.number_input(
            f"Kills {team_b}", min_value=0, max_value=100, step=1,
            value=int(session_state[_state_key(team_a, team_b, "kills_b")]),
            key=_state_key(team_a, team_b, "kills_b"),
        )
    with kc3:
        deaths_a = st.number_input(
            f"Deaths {team_a}", min_value=0, max_value=100, step=1,
            value=int(session_state[_state_key(team_a, team_b, "deaths_a")]),
            key=_state_key(team_a, team_b, "deaths_a"),
        )
    with kc4:
        deaths_b = st.number_input(
            f"Deaths {team_b}", min_value=0, max_value=100, step=1,
            value=int(session_state[_state_key(team_a, team_b, "deaths_b")]),
            key=_state_key(team_a, team_b, "deaths_b"),
        )
    with kc5:
        assists_a = st.number_input(
            f"Assists {team_a}", min_value=0, max_value=200, step=1,
            value=int(session_state[_state_key(team_a, team_b, "assists_a")]),
            key=_state_key(team_a, team_b, "assists_a"),
        )
    with kc6:
        assists_b = st.number_input(
            f"Assists {team_b}", min_value=0, max_value=200, step=1,
            value=int(session_state[_state_key(team_a, team_b, "assists_b")]),
            key=_state_key(team_a, team_b, "assists_b"),
        )

    oc1, oc2, oc3, oc4, oc5, oc6, oc7, oc8 = st.columns(8)
    with oc1:
        towers_a = st.number_input(f"Towers {team_a}", min_value=0, max_value=11, step=1,
            value=int(session_state[_state_key(team_a, team_b, "towers_a")]),
            key=_state_key(team_a, team_b, "towers_a"))
    with oc2:
        towers_b = st.number_input(f"Towers {team_b}", min_value=0, max_value=11, step=1,
            value=int(session_state[_state_key(team_a, team_b, "towers_b")]),
            key=_state_key(team_a, team_b, "towers_b"))
    with oc3:
        dragons_a = st.number_input(f"Drakes {team_a}", min_value=0, max_value=10, step=1,
            value=int(session_state[_state_key(team_a, team_b, "dragons_a")]),
            key=_state_key(team_a, team_b, "dragons_a"))
    with oc4:
        dragons_b = st.number_input(f"Drakes {team_b}", min_value=0, max_value=10, step=1,
            value=int(session_state[_state_key(team_a, team_b, "dragons_b")]),
            key=_state_key(team_a, team_b, "dragons_b"))
    with oc5:
        barons_a = st.number_input(f"Barons {team_a}", min_value=0, max_value=5, step=1,
            value=int(session_state[_state_key(team_a, team_b, "barons_a")]),
            key=_state_key(team_a, team_b, "barons_a"))
    with oc6:
        barons_b = st.number_input(f"Barons {team_b}", min_value=0, max_value=5, step=1,
            value=int(session_state[_state_key(team_a, team_b, "barons_b")]),
            key=_state_key(team_a, team_b, "barons_b"))
    with oc7:
        inhibitors_a = st.number_input(f"Inhibs {team_a}", min_value=0, max_value=3, step=1,
            value=int(session_state[_state_key(team_a, team_b, "inhibitors_a")]),
            key=_state_key(team_a, team_b, "inhibitors_a"))
    with oc8:
        inhibitors_b = st.number_input(f"Inhibs {team_b}", min_value=0, max_value=3, step=1,
            value=int(session_state[_state_key(team_a, team_b, "inhibitors_b")]),
            key=_state_key(team_a, team_b, "inhibitors_b"))

    return {
        "minute": float(minute_val),
        "gold_diff": float(gold_diff),
        "xp_diff": float(xp_diff),
        "cs_diff": float(cs_diff),
        "kills_a": float(kills_a), "kills_b": float(kills_b),
        "deaths_a": float(deaths_a), "deaths_b": float(deaths_b),
        "assists_a": float(assists_a), "assists_b": float(assists_b),
        "towers_a": float(towers_a), "towers_b": float(towers_b),
        "dragons_a": float(dragons_a), "dragons_b": float(dragons_b),
        "barons_a": float(barons_a), "barons_b": float(barons_b),
        "inhibitors_a": float(inhibitors_a), "inhibitors_b": float(inhibitors_b),
    }


def _render_headline(
    *,
    st: Any,
    team_a: str,
    team_b: str,
    p_pre_val: float,
    p_pre_provided: bool,
    assessment: Mapping[str, Any],
    fmt_num: Any,
) -> None:
    """Render the headline P(team_a wins) + state edge metric."""
    live_p = assessment.get("live_p_win") if assessment else None
    state_edge = assessment.get("state_edge") if assessment else None
    leader = str(assessment.get("leader") or "") if assessment else ""

    st.markdown("---")
    m1, m2, m3 = st.columns([1.5, 1.0, 1.0])
    with m1:
        if live_p is not None:
            pct_a = 100.0 * float(live_p)
            pct_b = 100.0 - pct_a
            delta_pp = pct_a - 100.0 * p_pre_val
            st.metric(
                label=f"P({team_a} vence) — modelo calibrado",
                value=f"{pct_a:.1f}%",
                delta=f"{delta_pp:+.1f} pp vs pré-jogo",
                help=(
                    "Probabilidade refinada com o estado ao vivo. "
                    "Treinada nos snapshots históricos @10/15/20/25 minutos."
                ),
            )
            st.caption(f"P({team_b} vence): **{pct_b:.1f}%**")
        elif p_pre_provided:
            pct_a = 100.0 * float(p_pre_val)
            pct_b = 100.0 - pct_a
            st.metric(
                label=f"P({team_a} vence) — pré-jogo",
                value=f"{pct_a:.1f}%",
                help=(
                    "Probabilidade pré-jogo do pipeline principal (já incorpora time, liga, BO e draft). "
                    "Preencha o estado ao vivo no modo avançado para refinar."
                ),
            )
            st.caption(f"P({team_b} vence): **{pct_b:.1f}%**")
        else:
            st.metric(
                label="P(vence) — pré-jogo",
                value="N/A",
                help=(
                    "Rode a aba Plays para calcular a probabilidade pré-jogo. "
                    "Ela aparece aqui automaticamente."
                ),
            )
            st.caption("Sem prior pré-jogo disponível.")
    with m2:
        edge_val = _safe_float(state_edge, 0.0)
        st.metric(
            label="State edge (heurística)",
            value=fmt_num(edge_val, 2),
            help="Indicador heurístico do estado ao vivo (só fica diferente de zero no modo avançado).",
        )
    with m3:
        st.metric(
            label="Lado na frente",
            value=leader or "—",
        )


def _render_live_markets_section(
    *,
    st: Any,
    session_state: MutableMapping[str, Any],
    minute: float,
    p_map_pre: float,
    league: str,
    bo: int,
    is_playoffs: bool,
    gold_diff: float, xp_diff: float, cs_diff: float,
    kills_a: float, kills_b: float,
    deaths_a: float, deaths_b: float,
    assists_a: float, assists_b: float,
    team_a: str, team_b: str,
    advanced_active: bool = False,
) -> None:
    """Render the per-market over/under predictions.

    In default mode (no live state), uses minute=10 + neutral state so the
    Phase 7 models return a sensible baseline. In advanced mode, uses the
    live state values, which gives the gate-validated state-aware refinement.
    """
    try:
        from app_core.live_markets_runtime_core import predict_market, ACTIVE_MARKETS
    except Exception:
        return

    default_lines = {
        "total_kills": 27.5,
        "total_towers": 11.5,
        "total_dragons": 4.5,
        "gamelength_min": 32.5,
    }
    market_labels = {
        "total_kills": "Total de kills",
        "total_towers": "Total de torres",
        "total_dragons": "Total de drakes",
        "gamelength_min": "Duração (min)",
    }

    st.markdown("---")
    title = "Mercados calibrados (refinados com estado)" if advanced_active else "Mercados calibrados (baseline pré-jogo)"
    st.markdown(f"#### {title}")
    if advanced_active:
        st.caption(
            "Predição do total final + P(over linha) por mercado usando o estado ao vivo. "
            "Treinados @10/15/20/25 com ganhos de até −43% RMSE no kills @25min."
        )
    else:
        st.caption(
            "Baseline pré-jogo do total final por mercado. "
            "Para refinar com o estado da partida, ative o modo avançado abaixo."
        )

    cols = st.columns(len(ACTIVE_MARKETS))
    for i, market in enumerate(ACTIVE_MARKETS):
        with cols[i]:
            key = f"lb_live_market_line_{market}_{str(team_a)[:20]}_{str(team_b)[:20]}"
            if key not in session_state:
                session_state[key] = float(default_lines.get(market, 0.0))
            # Sem value= — semear session E passar default juntos dispara o
            # warning "created with a default value but also had its value set".
            line_val = st.number_input(
                f"Linha {market_labels.get(market, market)}",
                min_value=0.0, max_value=200.0, step=0.5,
                key=key,
            )
            r = predict_market(
                session_state=session_state,
                market=market, minute=minute, line=line_val,
                p_map_pre=p_map_pre,
                gold_diff=gold_diff, xp_diff=xp_diff, cs_diff=cs_diff,
                kills_a=kills_a, kills_b=kills_b,
                deaths_a=deaths_a, deaths_b=deaths_b,
                assists_a=assists_a, assists_b=assists_b,
                league=league, bo=bo, is_playoffs=is_playoffs,
            )
            mean_v = r.get("mean")
            p_over = r.get("p_over")
            sigma = r.get("sigma")
            if mean_v is None:
                st.metric(label=market_labels.get(market, market),
                          value="N/A",
                          help=f"Reason: {r.get('reason', 'unknown')}")
            else:
                st.metric(
                    label=f"{market_labels.get(market, market)} — predição",
                    value=f"{mean_v:.2f}",
                    help=f"σ residual: {sigma:.2f}" if sigma else None,
                )
                if p_over is not None:
                    pct_over = 100.0 * float(p_over)
                    pct_under = 100.0 - pct_over
                    st.markdown(
                        f"**Over {line_val:g}: {pct_over:.1f}%**  ·  Under: {pct_under:.1f}%"
                    )


def _render_first_markets_section(
    *,
    st: Any,
    session_state: MutableMapping[str, Any],
    p_map_pre: float,
    league: str,
    bo: int,
    is_playoffs: bool,
    team_a: str, team_b: str,
) -> None:
    """Render pre-game first-X market predictions (Phase 8).

    Pure pre-game prediction — no state needed. Uses only the same inputs
    already on the live-bet tab (teams + league + drafts via p_map_pre).
    """
    try:
        from app_core.live_first_markets_runtime_core import predict_first_market, ACTIVE_FIRST_MARKETS
    except Exception:
        return

    market_labels = {
        "first_blood": "First Blood",
        "first_dragon": "First Dragon",
        "first_tower": "First Tower",
        "first_herald": "First Herald",
        "first_baron": "First Baron",
    }

    st.markdown("---")
    st.markdown("#### Mercados 'First X' (pré-jogo)")
    st.caption(
        "P(time vence o primeiro X) — modelo binário calibrado. "
        "Usa só pré-jogo (não precisa do estado ao vivo)."
    )

    cols = st.columns(len(ACTIVE_FIRST_MARKETS))
    for i, market in enumerate(ACTIVE_FIRST_MARKETS):
        with cols[i]:
            r = predict_first_market(
                session_state=session_state,
                market=market, p_map_pre=p_map_pre,
                league=league, bo=bo, is_playoffs=is_playoffs,
                team_a_is_blue=True,
            )
            p_team_a = r.get("p_team_a")
            label = market_labels.get(market, market)
            if p_team_a is None:
                st.metric(label=label, value="N/A",
                          help=f"Reason: {r.get('reason', 'unknown')}")
            else:
                pct_a = 100.0 * float(p_team_a)
                pct_b = 100.0 - pct_a
                st.metric(
                    label=f"{label} — {team_a}",
                    value=f"{pct_a:.1f}%",
                    help=f"{team_b}: {pct_b:.1f}%",
                )
                st.caption(f"{team_b}: **{pct_b:.1f}%**")


def _render_freshness_banner(*, st: Any, session_state: MutableMapping[str, Any], key_suffix: str = "") -> None:
    """Render a small banner showing whether the live models are up to date.

    Falls back silently if status helpers aren't importable (defensive).
    """
    try:
        from app_core.live_models_status_core import (
            assess_freshness, is_retrain_running, is_training_capable_env,
            launch_retrain_background, retrain_status,
        )
    except Exception:
        return

    # Na Streamlit Cloud (1GB de RAM) o treino trava/é morto: os modelos vêm
    # VERSIONADOS no git e o robô os mantém — a nuvem só carrega, nunca treina.
    pode_treinar = is_training_capable_env()

    csv_paths_raw = session_state.get("_csv_paths_used") or []
    if not csv_paths_raw:
        from pathlib import Path as _P
        candidates = []
        for d in ("CSV_HISTORICO", "CSV_DIARIO"):
            dp = _P(d)
            if dp.is_dir():
                for f in sorted(dp.glob("*.csv")):
                    candidates.append(str(f))
        csv_paths_raw = candidates

    models_dir = session_state.get("ml_live_state_models_dir") or "live_models"

    try:
        fs = assess_freshness(models_dir, csv_paths_raw)
    except Exception:
        return

    state = fs.state
    if state == "fresh":
        st.success(f"✓ {fs.message}")
    elif state == "stale":
        st.warning(f"⚠ {fs.message}")
    elif state == "very_stale":
        st.error(f"⚠ {fs.message}")
    else:
        st.info(f"ℹ {fs.message}")

    try:
        running = is_retrain_running(models_dir)
    except Exception:
        running = False

    # AUTO-RETREINO (22/07): se os modelos ao vivo estão velhos e ninguém está
    # treinando, o app dispara SOZINHO em background (trava de 6h evita relançar
    # a cada F5). É o que faz ele "não precisar mais saber do ao vivo".
    auto_disparado = False
    auto_falhou = ""
    if not running and pode_treinar:
        try:
            from app_core.live_models_status_core import (
                hours_since_auto_retrain, mark_auto_retrain_attempt, should_auto_retrain,
            )
            if should_auto_retrain(
                state=str(state), running=running,
                hours_since_attempt=hours_since_auto_retrain(models_dir),
                min_cooldown_hours=6.0,
            ):
                cutoff = session_state.get("_live_models_val_cutoff", "2026-03-31")
                info = launch_retrain_background(val_cutoff=str(cutoff), save_models_dir=str(models_dir))
                mark_auto_retrain_attempt(models_dir)   # marca a TENTATIVA (respeita o cooldown mesmo falhando)
                if info.get("ok"):
                    running = True
                    auto_disparado = True
                else:
                    # NAO engolir: era assim que o Dota ficava sem NENHUM modelo ao
                    # vivo — o app tentava todo dia, falhava, gravava o carimbo e
                    # seguia. Quem olhasse o last_auto_retrain.txt via a data de hoje
                    # e concluia que estava tudo certo (achado 16/08/2026).
                    auto_falhou = str(info.get("error") or "motivo desconhecido")
        except Exception as _exc_auto:
            # a conveniencia nunca derruba o painel, mas tambem nao some calada
            auto_falhou = f"{type(_exc_auto).__name__}: {_exc_auto}"

    _fresh = str(state) == "fresh"
    cols = st.columns([2, 1, 1])
    with cols[0]:
        if auto_falhou:
            st.error(
                "🚨 **O retreino automático do modelo ao vivo FALHOU** — "
                f"{auto_falhou}. A aba de estado ao vivo fica sem modelo até isso "
                "ser resolvido; o número dela não deve ser usado."
            )
        if auto_disparado:
            st.warning(
                "⏳ **Modelo ao vivo desatualizado — atualizando agora, favor não fechar o app.** "
                "Leva alguns minutos; pode usar o resto normalmente e dar F5 de vez em quando — "
                "os números aparecem quando terminar. Se fechar, **nada estraga**: ele continua "
                "em background ou recomeça sozinho na próxima abertura (o modelo só é salvo quando "
                "o treino termina inteiro)."
            )
        elif running and not _fresh:
            st.info("Modelo em retreino em background (favor não fechar até terminar). "
                    "Atualize a página em alguns minutos.")
        elif running and _fresh:
            # Já fresco + um processo rodando = retreino redundante finalizando.
            # Não contradizer o 'atualizados' verde acima (report dele 22/07).
            st.caption("Um retreino extra ainda finaliza em background — inofensivo; "
                       "os números acima já estão atualizados, pode usar normal.")
        elif state in ("stale", "very_stale", "no_manifest", "no_models"):
            if pode_treinar:
                st.caption("Clique em 'Retreinar agora' para regenerar os modelos com os dados mais recentes.")
            else:
                # Na nuvem não dá pra treinar: o robô do GitHub Actions atualiza.
                st.caption("Os modelos são atualizados automaticamente pelo robô — "
                           "nada a fazer aqui, pode usar normal.")

    with cols[1]:
        # Botão só quando faz sentido: nada rodando, modelos NÃO frescos, E o
        # ambiente CONSEGUE treinar (na nuvem o treino trava, então nem aparece).
        # Antes, o botão aparecia mesmo com modelo fresco e um clique disparava um
        # retreino redundante (foi o que ele pegou: F5 pós-treino -> clicou -> 2º).
        if pode_treinar and not running and not _fresh and st.button(
            "Retreinar agora", key=f"lb_live_retrain_btn{key_suffix}", help="Roda tools/run_phase6_live_gate.py em background"):
            try:
                cutoff = session_state.get("_live_models_val_cutoff", "2026-03-31")
                info = launch_retrain_background(
                    val_cutoff=str(cutoff), save_models_dir=str(models_dir),
                )
                if info.get("ok"):
                    st.success(f"Retreino iniciado (pid={info.get('pid')}). Atualize a página em ~5 min.")
                else:
                    st.error(f"Falha ao iniciar retreino: {info.get('error')}")
            except Exception as exc:
                st.error(f"Erro ao iniciar retreino: {type(exc).__name__}: {exc}")

    with cols[2]:
        if running:
            try:
                rs = retrain_status(models_dir)
                if rs.get("log_tail"):
                    with st.expander("Log do retreino", expanded=False):
                        st.code(rs["log_tail"])
            except Exception as _exc_log:
                from app_core.ui_notify_core import notify
                notify(st, "warn", "Log do retreino indisponível",
                       detail=f"{type(_exc_log).__name__}: {_exc_log}")


__all__ = ["render_live_state_panel"]
