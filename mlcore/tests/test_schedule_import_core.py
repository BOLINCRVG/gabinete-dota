import tempfile
import unittest
from pathlib import Path

from app_core.schedule_import_core import (
    load_schedule_aliases,
    normalize_schedule_team_key,
    parse_schedule_text,
    resolve_schedule_team,
    suggest_schedule_teams,
    upsert_schedule_alias,
)


class ScheduleImportCoreTests(unittest.TestCase):
    def test_parse_schedule_text_extracts_games(self) -> None:
        text = "League IconLCKFlag Icon\n22/01\n05:00\nBRIONBRION\nHanwha Life Esports\n-\n-\n"
        rows = parse_schedule_text(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["league"], "LCK")
        self.assertEqual(rows[0]["teamA_raw"], "BRION")

    def test_parse_schedule_text_extracts_games_from_repeated_match_block(self) -> None:
        text = (
            "EMEA Masters 2026 Winter\n"
            "EMEA Masters 2026 Winter\n"
            "16:00\n"
            "Colossal Gaming\n"
            "Colossal Gaming\n"
            "—\n"
            "NightBirds\n"
            "NightBirds\n"
            "—\n"
        )
        rows = parse_schedule_text(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["league"], "EMEA Masters 2026 Winter")
        self.assertEqual(rows[0]["time"], "16:00")
        self.assertEqual(rows[0]["teamA_raw"], "Colossal Gaming")
        self.assertEqual(rows[0]["teamB_raw"], "NightBirds")

    def test_alias_roundtrip_and_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aliases.json"
            state = {}
            upsert_schedule_alias(state, path, "Hanwha Life", "Hanwha Life Esports")
            aliases = load_schedule_aliases(state, path)
            self.assertIn(normalize_schedule_team_key("Hanwha Life"), aliases)
            resolved, ok = resolve_schedule_team("Hanwha Life", ["Hanwha Life Esports"], aliases)
            self.assertTrue(ok)
            self.assertEqual(resolved, "Hanwha Life Esports")

    def test_suggestions_are_ranked(self) -> None:
        out = suggest_schedule_teams("T1A", ["T1", "Talon", "Team Heretics"])
        self.assertIn("T1", out[:2])


if __name__ == "__main__":
    unittest.main()


class TestResolverNomesDaCasa27Out(unittest.TestCase):
    """27/10/2026, gaveta do Dota: "Nao consegui casar PVISION x Liquid"."""

    def test_prefixo_team(self):
        from app_core.schedule_import_core import resolve_schedule_team
        self.assertEqual(resolve_schedule_team("Liquid", ["Team Liquid", "Tundra"], {}), ("Team Liquid", True))
        self.assertEqual(resolve_schedule_team("Team Spirit", ["Spirit"], {}), ("Spirit", True))

    def test_prefixo_team_ambiguo_nao_chuta(self):
        from app_core.schedule_import_core import resolve_schedule_team
        # "Team Falcons" e "TEAM FALCONS": duas opcoes com prefixo -> nao escolhe
        self.assertEqual(resolve_schedule_team("Falcons", ["Team Falcons", "TEAM FALCONS"], {})[1], False)

    def test_apelido_vale_mais_que_o_nome_exato(self):
        from app_core.schedule_import_core import resolve_schedule_team
        times = ["PVISION", "PARIVISION"]
        self.assertEqual(resolve_schedule_team("PVISION", times, {}), ("PVISION", True))
        self.assertEqual(resolve_schedule_team("PVISION", times, {"pvision": "PARIVISION"}), ("PARIVISION", True))
