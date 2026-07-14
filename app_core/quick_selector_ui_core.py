from __future__ import annotations

import math
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from app_core.consistency_ui_core import render_market_mode_section


_QUICK_SIDE_OPTIONS: tuple[str, ...] = ("Blue", "Red", "Ambos")
_QUICK_RESULT_OPTIONS: tuple[str, ...] = ("Vit\u00f3ria", "Derrota", "Ambos")


def _normalize_choice(
    session_state: MutableMapping[str, Any],
    *,
    key: str,
    options: Sequence[Any],
    default: Any,
) -> str:
    normalized = [str(value) for value in list(options or [])]
    fallback = str(default or (normalized[0] if normalized else ""))
    current = str(session_state.get(key, fallback) or fallback)
    if current not in normalized:
        current = fallback if fallback in normalized else (normalized[0] if normalized else "")
        session_state[key] = current
    return current


def _normalize_multiselect(
    session_state: MutableMapping[str, Any],
    *,
    key: str,
    options: Sequence[Any],
) -> list[Any]:
    raw_value = session_state.get(key, [])
    current = list(raw_value) if isinstance(raw_value, (list, tuple, set)) else []
    valid = [value for value in current if value in list(options or [])]
    if list(options or []) and not valid:
        valid = [next(iter(options or []))]
    if (key not in session_state) or (valid != current):
        session_state[key] = list(valid)
    return list(valid)


def resolve_quick_selector_state(
    *,
    session_state: MutableMapping[str, Any],
    team_a: str,
    team_b: str,
    kills: Sequence[float],
    towers: Sequence[float],
    dragons: Sequence[float],
    barons: Sequence[float],
    inhib: Sequence[float],
    time_m: Sequence[float],
    team_a_kills_lines: Sequence[float],
    team_b_kills_lines: Sequence[float],
    build_metric_options_fn: Callable[..., Mapping[str, tuple[str, str, Sequence[float]]]],
) -> dict[str, Any]:
    team_options = [str(team_a), str(team_b)]
    team_sel = _normalize_choice(
        session_state,
        key="quick_combo_team",
        options=team_options,
        default=str(team_a),
    )
    side_sel_raw = _normalize_choice(
        session_state,
        key="quick_combo_side",
        options=_QUICK_SIDE_OPTIONS,
        default="Blue",
    )
    result_sel = _normalize_choice(
        session_state,
        key="quick_combo_result",
        options=_QUICK_RESULT_OPTIONS,
        default="Vit\u00f3ria",
    )

    metric_opts = dict(
        build_metric_options_fn(
            str(team_a),
            str(team_b),
            list(kills or []),
            list(towers or []),
            list(dragons or []),
            list(barons or []),
            list(inhib or []),
            list(time_m or []),
            list(team_a_kills_lines or []),
            list(team_b_kills_lines or []),
            str(team_sel),
        )
        or {}
    )
    metric_names = list(metric_opts.keys())
    metric_name = _normalize_choice(
        session_state,
        key="quick_combo_metric",
        options=metric_names,
        default=(metric_names[0] if metric_names else ""),
    )
    metric_key, value_col, line_values = metric_opts.get(metric_name, ("kills", "total_kills", []))
    line_values = list(line_values or [])
    line_sel_multi = _normalize_multiselect(
        session_state,
        key="quick_combo_line_multi",
        options=line_values,
    )
    session_state.setdefault("quick_combo_line_manual", "")

    return {
        "team_sel": str(team_sel),
        "side_sel_raw": str(side_sel_raw),
        "result_sel": str(result_sel),
        "metric_opts": metric_opts,
        "metric_names": metric_names,
        "metric_name": str(metric_name),
        "metric_key": str(metric_key),
        "value_col": str(value_col),
        "line_values": line_values,
        "line_sel_multi": line_sel_multi,
        "manual_lines_raw": str(session_state.get("quick_combo_line_manual", "") or ""),
    }


def resolve_quick_selector_lines(
    *,
    line_sel_multi: Sequence[Any],
    manual_lines_raw: str,
    metric_key: str,
    parse_lines_fn: Callable[[str], Sequence[Any]],
    parse_time_lines_fn: Callable[[str], Sequence[Any]],
    num_cast_fn: Callable[[Any], float | None],
) -> list[float]:
    selected_lines: list[float] = []
    for raw_value in list(line_sel_multi or []):
        parsed = num_cast_fn(raw_value)
        if parsed is not None:
            selected_lines.append(float(parsed))

    parser = parse_time_lines_fn if str(metric_key) == "time" else parse_lines_fn
    for raw_value in list(parser(str(manual_lines_raw or "")) or []):
        parsed = num_cast_fn(raw_value)
        if parsed is not None:
            selected_lines.append(float(parsed))

    return sorted(set(float(value) for value in selected_lines if math.isfinite(float(value))))


def render_quick_selector_controls(
    *,
    st: Any,
    session_state: MutableMapping[str, Any],
    team_a: str,
    team_b: str,
    kills: Sequence[float],
    towers: Sequence[float],
    dragons: Sequence[float],
    barons: Sequence[float],
    inhib: Sequence[float],
    time_m: Sequence[float],
    team_a_kills_lines: Sequence[float],
    team_b_kills_lines: Sequence[float],
    build_metric_options_fn: Callable[..., Mapping[str, tuple[str, str, Sequence[float]]]],
    parse_lines_fn: Callable[[str], Sequence[Any]],
    parse_time_lines_fn: Callable[[str], Sequence[Any]],
    num_cast_fn: Callable[[Any], float | None],
) -> dict[str, Any]:
    render_market_mode_section(
        st,
        title="Leitura",
        subtitle="Monte um cen\u00e1rio operacional e compare Modelo, Hist\u00f3rico e Mix na linha escolhida.",
    )

    state = resolve_quick_selector_state(
        session_state=session_state,
        team_a=team_a,
        team_b=team_b,
        kills=kills,
        towers=towers,
        dragons=dragons,
        barons=barons,
        inhib=inhib,
        time_m=time_m,
        team_a_kills_lines=team_a_kills_lines,
        team_b_kills_lines=team_b_kills_lines,
        build_metric_options_fn=build_metric_options_fn,
    )

    metric_opts = dict(state["metric_opts"])
    metric_names = list(state["metric_names"])
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        team_sel = st.selectbox("Time", [team_a, team_b], key="quick_combo_team")
    with c2:
        side_sel_raw = st.selectbox("Lado", list(_QUICK_SIDE_OPTIONS), key="quick_combo_side")
    with c3:
        result_sel = st.selectbox("Resultado", list(_QUICK_RESULT_OPTIONS), key="quick_combo_result")
    with c4:
        if metric_names:
            metric_name = st.selectbox("Mercado", metric_names, key="quick_combo_metric")
        else:
            metric_name = ""
            st.caption("Sem mercados dispon\u00edveis")

    metric_key, value_col, line_values = metric_opts.get(metric_name, ("kills", "total_kills", []))
    line_values = list(line_values or [])
    with c5:
        if line_values:
            line_sel_multi = st.multiselect(
                "Linha (pr\u00e9-definida)",
                options=line_values,
                key="quick_combo_line_multi",
            )
        else:
            line_sel_multi = []
            if session_state.get("quick_combo_line_multi"):
                session_state["quick_combo_line_multi"] = []
            st.caption("Sem linhas pr\u00e9-definidas")

    manual_lines_raw = st.text_input(
        "Linha manual (1 ou mais, separadas por v\u00edrgula)",
        key="quick_combo_line_manual",
        placeholder="Ex.: 26.5, 27.5 (ou 31:30 para Tempo)",
    )
    selected_lines = resolve_quick_selector_lines(
        line_sel_multi=line_sel_multi,
        manual_lines_raw=str(manual_lines_raw or ""),
        metric_key=str(metric_key),
        parse_lines_fn=parse_lines_fn,
        parse_time_lines_fn=parse_time_lines_fn,
        num_cast_fn=num_cast_fn,
    )

    return {
        "team_sel": str(team_sel),
        "side_sel_raw": str(side_sel_raw),
        "result_sel": str(result_sel),
        "metric_name": str(metric_name),
        "metric_key": str(metric_key),
        "value_col": str(value_col),
        "line_values": line_values,
        "line_sel_multi": list(line_sel_multi or []),
        "manual_lines_raw": str(manual_lines_raw or ""),
        "selected_lines": selected_lines,
    }


__all__ = [
    "render_quick_selector_controls",
    "resolve_quick_selector_lines",
    "resolve_quick_selector_state",
]
