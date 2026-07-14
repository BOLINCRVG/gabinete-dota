import unittest

from app_core.quick_selector_ui_core import resolve_quick_selector_lines, resolve_quick_selector_state


def _build_metric_options(
    team_a,
    team_b,
    kills,
    towers,
    dragons,
    barons,
    inhib,
    time_m,
    team_a_kills_lines,
    team_b_kills_lines,
    team_sel,
):
    return {
        "Kills (Total)": ("kills", "total_kills", list(kills or [])),
        "Tempo": ("time", "game_time_min", list(time_m or [])),
        "Kills do Time selecionado": (
            "team_kills",
            "kills_for",
            list(team_a_kills_lines if str(team_sel) == str(team_a) else team_b_kills_lines or []),
        ),
    }


class QuickSelectorUICoreTests(unittest.TestCase):
    def test_resolve_quick_selector_state_normalizes_invalid_values(self) -> None:
        session_state = {
            "quick_combo_team": "Outro",
            "quick_combo_side": "Green",
            "quick_combo_result": "Empate",
            "quick_combo_metric": "Mercado invalido",
            "quick_combo_line_multi": [31.5, 999.5],
        }

        out = resolve_quick_selector_state(
            session_state=session_state,
            team_a="Team A",
            team_b="Team B",
            kills=[30.5, 31.5],
            towers=[],
            dragons=[],
            barons=[],
            inhib=[],
            time_m=[31.0],
            team_a_kills_lines=[12.5],
            team_b_kills_lines=[14.5],
            build_metric_options_fn=_build_metric_options,
        )

        self.assertEqual(out["team_sel"], "Team A")
        self.assertEqual(out["side_sel_raw"], "Blue")
        self.assertEqual(out["result_sel"], "Vit\u00f3ria")
        self.assertEqual(out["metric_name"], "Kills (Total)")
        self.assertEqual(out["line_sel_multi"], [31.5])
        self.assertEqual(session_state["quick_combo_line_multi"], [31.5])
        self.assertEqual(session_state["quick_combo_line_manual"], "")

    def test_resolve_quick_selector_state_refreshes_team_specific_lines(self) -> None:
        session_state = {
            "quick_combo_team": "Team B",
            "quick_combo_metric": "Kills do Time selecionado",
            "quick_combo_line_multi": [12.5],
        }

        out = resolve_quick_selector_state(
            session_state=session_state,
            team_a="Team A",
            team_b="Team B",
            kills=[30.5],
            towers=[],
            dragons=[],
            barons=[],
            inhib=[],
            time_m=[],
            team_a_kills_lines=[12.5],
            team_b_kills_lines=[14.5],
            build_metric_options_fn=_build_metric_options,
        )

        self.assertEqual(out["metric_name"], "Kills do Time selecionado")
        self.assertEqual(out["line_values"], [14.5])
        self.assertEqual(out["line_sel_multi"], [14.5])

    def test_resolve_quick_selector_lines_merges_and_dedupes_values(self) -> None:
        out = resolve_quick_selector_lines(
            line_sel_multi=[27.5, 26.5, None],
            manual_lines_raw="27.5, 28.5",
            metric_key="kills",
            parse_lines_fn=lambda _text: [27.5, 28.5, float("nan")],
            parse_time_lines_fn=lambda _text: [99.0],
            num_cast_fn=lambda value: (float(value) if value is not None else None),
        )

        self.assertEqual(out, [26.5, 27.5, 28.5])

    def test_resolve_quick_selector_lines_uses_time_parser_for_time_metric(self) -> None:
        out = resolve_quick_selector_lines(
            line_sel_multi=[],
            manual_lines_raw="31:30, 32:00",
            metric_key="time",
            parse_lines_fn=lambda _text: [99.0],
            parse_time_lines_fn=lambda _text: [31.5, 32.0],
            num_cast_fn=lambda value: float(value),
        )

        self.assertEqual(out, [31.5, 32.0])


if __name__ == "__main__":
    unittest.main()
