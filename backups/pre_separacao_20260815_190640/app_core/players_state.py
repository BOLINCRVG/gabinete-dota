"""State/parsing helpers for Players tab UI."""

from __future__ import annotations

import re
from typing import Any, Callable, MutableMapping


def parse_players_text(txt: str) -> list[str]:
    return [t.strip() for t in re.split(r"[\n\t,;]+", str(txt or "").strip()) if t.strip()]


def resolve_players(
    *,
    names: list[str],
    vocab: list[str],
    fuzzy_match_fn: Callable[[str, list[str], float], tuple[Any, list[str]]],
    cutoff: float = 0.6,
) -> tuple[list[str], list[dict]]:
    resolved: list[str] = []
    notes: list[dict] = []
    for nm in names:
        m, sugg = fuzzy_match_fn(nm, vocab, cutoff)
        if m is None:
            notes.append({"entrada": nm, "match": None, "sugestoes": list(sugg or [])[:5]})
        else:
            resolved.append(str(m))
            if str(nm).strip() != str(m):
                notes.append({"entrada": nm, "match": str(m), "sugestoes": list(sugg or [])[:5]})
    return resolved, notes


def build_lane_rows(players_a: list[str], players_b: list[str], pos_order: list[str] | None = None) -> list[dict]:
    order = list(pos_order or ["top", "jng", "mid", "bot", "sup"])
    rows: list[dict] = []
    for i, pos in enumerate(order):
        rows.append(
            {
                "Lane": str(pos).upper(),
                "Pos": str(pos),
                "A": str(players_a[i]) if i < len(players_a) else "",
                "B": str(players_b[i]) if i < len(players_b) else "",
            }
        )
    return rows


def ensure_players_text_state(session_state: MutableMapping[str, Any]) -> None:
    if "pl_txtA_applied" not in session_state:
        session_state["pl_txtA_applied"] = str(session_state.get("pl_txtA", "") or "")
    if "pl_txtB_applied" not in session_state:
        session_state["pl_txtB_applied"] = str(session_state.get("pl_txtB", "") or "")
    if "pl_txtA_input" not in session_state:
        session_state["pl_txtA_input"] = str(session_state.get("pl_txtA_applied", "") or "")
    if "pl_txtB_input" not in session_state:
        session_state["pl_txtB_input"] = str(session_state.get("pl_txtB_applied", "") or "")


def ensure_players_filter_state(session_state: MutableMapping[str, Any]) -> None:
    for key in ("pl_year", "pl_league", "pl_split", "pl_playoffs"):
        if key not in session_state or session_state.get(key) in ("All", None):
            session_state[key] = ""


def apply_players_autofill_state(
    session_state: MutableMapping[str, Any],
    *,
    team_a_players: list[str] | None = None,
    team_b_players: list[str] | None = None,
) -> None:
    if team_a_players:
        txt_a = "\n".join([str(x) for x in team_a_players if str(x).strip()])
        if txt_a:
            session_state["pl_txtA"] = txt_a
            session_state["pl_txtA_input"] = txt_a
            session_state["pl_txtA_applied"] = txt_a
    if team_b_players:
        txt_b = "\n".join([str(x) for x in team_b_players if str(x).strip()])
        if txt_b:
            session_state["pl_txtB"] = txt_b
            session_state["pl_txtB_input"] = txt_b
            session_state["pl_txtB_applied"] = txt_b


def ensure_player_line_input_state(
    session_state: MutableMapping[str, Any],
    *,
    lanes: int = 5,
    metrics: tuple[str, ...] = ("kills", "deaths", "assists"),
    sides: tuple[str, ...] = ("A", "B"),
) -> None:
    for i in range(int(lanes)):
        for metric in metrics:
            for side in sides:
                in_key = f"pl_in_{i}_{metric}_{side}"
                ap_key = f"pl_applied_{i}_{metric}_{side}"
                old_key = f"pl_duel_{i}_{metric}_{side}"
                if ap_key not in session_state:
                    session_state[ap_key] = str(session_state.get(old_key, "") or "")
                if in_key not in session_state:
                    session_state[in_key] = str(session_state.get(ap_key, "") or "")


def apply_players_submission_state(
    session_state: MutableMapping[str, Any],
    *,
    lanes: int = 5,
    metrics: tuple[str, ...] = ("kills", "deaths", "assists"),
    sides: tuple[str, ...] = ("A", "B"),
) -> None:
    session_state["pl_txtA_applied"] = str(session_state.get("pl_txtA_input", "") or "")
    session_state["pl_txtB_applied"] = str(session_state.get("pl_txtB_input", "") or "")
    for i in range(int(lanes)):
        for metric in metrics:
            for side in sides:
                session_state[f"pl_applied_{i}_{metric}_{side}"] = str(
                    session_state.get(f"pl_in_{i}_{metric}_{side}", "") or ""
                )
    session_state["pl_txtA"] = session_state["pl_txtA_applied"]
    session_state["pl_txtB"] = session_state["pl_txtB_applied"]


def build_per_player_lines_by_idx(
    *,
    session_state: MutableMapping[str, Any],
    lane_rows: list[dict],
    parse_lines_fn: Callable[[str], list[float]],
) -> dict[int, dict[str, dict[str, list[float]]]]:
    out: dict[int, dict[str, dict[str, list[float]]]] = {}
    for i, _rr in enumerate(lane_rows):
        out[i] = {
            "A": {
                "Kills": parse_lines_fn(session_state.get(f"pl_applied_{i}_kills_A", "")),
                "Deaths": parse_lines_fn(session_state.get(f"pl_applied_{i}_deaths_A", "")),
                "Assists": parse_lines_fn(session_state.get(f"pl_applied_{i}_assists_A", "")),
            },
            "B": {
                "Kills": parse_lines_fn(session_state.get(f"pl_applied_{i}_kills_B", "")),
                "Deaths": parse_lines_fn(session_state.get(f"pl_applied_{i}_deaths_B", "")),
                "Assists": parse_lines_fn(session_state.get(f"pl_applied_{i}_assists_B", "")),
            },
        }
    return out
