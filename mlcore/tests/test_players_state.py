import unittest

from app_core.players_state import (
    apply_players_autofill_state,
    apply_players_submission_state,
    build_lane_rows,
    build_per_player_lines_by_idx,
    ensure_player_line_input_state,
    ensure_players_filter_state,
    ensure_players_text_state,
    parse_players_text,
    resolve_players,
)


class TestPlayersState(unittest.TestCase):
    def test_parse_players_text(self):
        txt = "Faker\nOner, Zeus; Gumayusi\tKeria"
        got = parse_players_text(txt)
        self.assertEqual(got, ["Faker", "Oner", "Zeus", "Gumayusi", "Keria"])

    def test_resolve_players(self):
        def _fake_fuzzy(name, vocab, cutoff):
            if name.lower() == "faker":
                return "Faker", ["Faker"]
            return None, ["FakeR", "Faker"]

        resolved, notes = resolve_players(
            names=["faker", "desconhecido"],
            vocab=["Faker", "Oner"],
            fuzzy_match_fn=_fake_fuzzy,
            cutoff=0.6,
        )
        self.assertEqual(resolved, ["Faker"])
        self.assertEqual(len(notes), 2)
        self.assertEqual(notes[0]["match"], "Faker")
        self.assertIsNone(notes[1]["match"])

    def test_build_lane_rows(self):
        rows = build_lane_rows(["A1", "A2"], ["B1"], ["top", "jng"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["Lane"], "TOP")
        self.assertEqual(rows[0]["A"], "A1")
        self.assertEqual(rows[0]["B"], "B1")
        self.assertEqual(rows[1]["A"], "A2")
        self.assertEqual(rows[1]["B"], "")

    def test_state_init_and_apply(self):
        st = {"pl_txtA": "A\nB", "pl_txtB": "C\nD"}
        ensure_players_text_state(st)
        self.assertEqual(st["pl_txtA_applied"], "A\nB")
        self.assertEqual(st["pl_txtB_input"], "C\nD")

        ensure_player_line_input_state(st, lanes=1)
        self.assertIn("pl_in_0_kills_A", st)
        st["pl_in_0_kills_A"] = "2.5,3.5"
        st["pl_txtA_input"] = "AX\nBX"
        apply_players_submission_state(st, lanes=1)
        self.assertEqual(st["pl_applied_0_kills_A"], "2.5,3.5")
        self.assertEqual(st["pl_txtA"], "AX\nBX")

    def test_filter_state_and_autofill(self):
        st = {"pl_year": "All", "pl_league": None}
        ensure_players_filter_state(st)
        self.assertEqual(st["pl_year"], "")
        self.assertEqual(st["pl_league"], "")
        self.assertEqual(st["pl_split"], "")
        self.assertEqual(st["pl_playoffs"], "")

        apply_players_autofill_state(st, team_a_players=["Zeus", "Oner"], team_b_players=["Bin"])
        self.assertEqual(st["pl_txtA"], "Zeus\nOner")
        self.assertEqual(st["pl_txtA_input"], "Zeus\nOner")
        self.assertEqual(st["pl_txtB_applied"], "Bin")

    def test_build_per_player_lines_by_idx(self):
        st = {
            "pl_applied_0_kills_A": "2.5,3.5",
            "pl_applied_0_deaths_A": "",
            "pl_applied_0_assists_A": "4.5",
            "pl_applied_0_kills_B": "1.5",
            "pl_applied_0_deaths_B": "2.5",
            "pl_applied_0_assists_B": "",
        }
        lane_rows = [{"Lane": "TOP", "Pos": "top", "A": "A_TOP", "B": "B_TOP"}]

        def _parse_lines(v):
            return [float(x.strip()) for x in str(v).split(",") if str(x).strip()]

        out = build_per_player_lines_by_idx(session_state=st, lane_rows=lane_rows, parse_lines_fn=_parse_lines)
        self.assertEqual(out[0]["A"]["Kills"], [2.5, 3.5])
        self.assertEqual(out[0]["A"]["Deaths"], [])
        self.assertEqual(out[0]["B"]["Deaths"], [2.5])


if __name__ == "__main__":
    unittest.main()
