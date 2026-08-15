from __future__ import annotations

from typing import Any, Callable


def render_lineup_selector_column(
    *,
    st_module: Any,
    team_label: str,
    side: str,
    choices: dict[str, list[str]] | None,
    display_map: dict[str, str] | None,
    pos_list_ui: list[str],
    prefix: str,
    use_ml_lineup: bool,
    lineup_key: str,
    safe_key_fn: Callable[[str], str],
    save_lineup_for_matchup_fn: Callable[[str], None],
) -> None:
    st_module.markdown(f"##### {team_label}")
    choices = dict(choices or {})
    display_map = dict(display_map or {})
    for pos in pos_list_ui:
        key = f"{prefix}_{side}_{safe_key_fn(pos)}"
        opts = ["", *list(choices.get(pos, []) or [])]
        cur = str(st_module.session_state.get(key, "") or "")
        idx = opts.index(cur) if cur in opts else 0
        if use_ml_lineup:
            st_module.selectbox(
                f"{pos} (ADC)" if pos == "Bot" else pos,
                options=opts,
                index=idx,
                key=key,
                format_func=lambda pid, _display_map=display_map: "" if not pid else str(_display_map.get(str(pid), pid)),
                on_change=save_lineup_for_matchup_fn,
                args=(lineup_key,),
            )
        else:
            st_module.selectbox(
                pos,
                options=opts,
                index=idx,
                key=key,
                on_change=save_lineup_for_matchup_fn,
                args=(lineup_key,),
            )
