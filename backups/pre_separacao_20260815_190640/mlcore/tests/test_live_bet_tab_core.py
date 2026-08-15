from __future__ import annotations

import unittest

import pandas as pd

from app_core.live_bet_tab_core import (
    _build_lane_cards,
    build_live_bet_draft_analysis,
    build_live_bet_live_assessment,
    build_live_bet_market_value_summary,
    build_live_bet_player_prop_summary,
    build_live_bet_pregame_summary,
    build_live_bet_tactical_notes,
    parse_compact_draft_tokens,
    resolve_compact_draft_sides,
)


class LiveBetTabCoreTest(unittest.TestCase):
    def _champion_registry(self) -> dict[str, object]:
        return {
            "champions": {
                "gnar": {"name": "Gnar", "styles": ["skirmish", "teamfight"], "window": "mid", "traits": {"early": 1.2, "mid": 1.4, "late": 0.8, "lane_priority": 1.1, "all_in": 0.4, "teamfight": 1.2, "skirmish": 1.1, "objective_control": 0.4}},
                "vi": {"name": "Vi", "styles": ["engage", "dive"], "window": "mid", "traits": {"early": 1.1, "mid": 1.3, "late": 0.6, "engage": 1.4, "dive": 1.3, "objective_control": 1.0, "skirmish": 1.2}},
                "orianna": {"name": "Orianna", "styles": ["teamfight", "peel"], "window": "late", "traits": {"early": 0.3, "mid": 1.0, "late": 1.5, "poke": 0.5, "waveclear": 1.2, "teamfight": 1.5, "peel": 1.0}},
                "jinx": {"name": "Jinx", "styles": ["siege", "teamfight"], "window": "late", "traits": {"early": 0.1, "mid": 1.0, "late": 1.8, "range": 1.5, "siege": 1.4, "objective_control": 1.2, "teamfight": 1.0}},
                "nautilus": {"name": "Nautilus", "styles": ["engage", "frontline"], "window": "mid", "traits": {"early": 0.8, "mid": 1.0, "late": 0.5, "engage": 1.4, "frontline": 1.0, "teamfight": 1.0, "objective_control": 0.3}},
                "ksante": {"name": "Ksante", "styles": ["frontline", "teamfight"], "window": "late", "traits": {"early": 0.3, "mid": 0.9, "late": 1.4, "frontline": 1.5, "teamfight": 1.0, "lane_priority": 0.2}},
                "xinzhao": {"name": "Xin Zhao", "styles": ["skirmish", "dive"], "window": "early", "traits": {"early": 1.3, "mid": 1.0, "late": 0.2, "all_in": 0.8, "dive": 1.1, "objective_control": 0.7, "skirmish": 1.4}},
                "ahri": {"name": "Ahri", "styles": ["pick", "poke"], "window": "mid", "traits": {"early": 0.8, "mid": 1.3, "late": 0.8, "lane_priority": 0.8, "pick": 1.4, "mobility": 1.2, "poke": 0.7}},
                "xayah": {"name": "Xayah", "styles": ["teamfight", "peel"], "window": "late", "traits": {"early": 0.2, "mid": 0.9, "late": 1.6, "range": 1.0, "teamfight": 1.2, "peel": 0.8}},
                "rakan": {"name": "Rakan", "styles": ["engage", "peel"], "window": "mid", "traits": {"early": 0.8, "mid": 1.2, "late": 0.7, "engage": 1.2, "peel": 0.8, "mobility": 1.0, "teamfight": 0.8}},
            },
            "aliases": {},
        }

    def test_parse_compact_draft_tokens_supports_mixed_spacing_and_multiword_names(self) -> None:
        vocab = ["Ornn", "Dr. Mundo", "Ryze", "Lucian", "Nami", "Renekton", "Qiyana", "Anivia", "Varus", "Alistar", "Miss Fortune"]
        parsed = parse_compact_draft_tokens(
            "Ornn DrMundo Ryze Lucian Nami Renekton Qiyana Anivia Varus Alistar",
            vocab,
        )
        self.assertEqual(parsed[:5], ["Ornn", "Dr. Mundo", "Ryze", "Lucian", "Nami"])
        self.assertEqual(parsed[5:10], ["Renekton", "Qiyana", "Anivia", "Varus", "Alistar"])

        parsed_multiword = parse_compact_draft_tokens("Miss Fortune Nami Ornn Varus Alistar", vocab, max_picks=5)
        self.assertEqual(parsed_multiword[0], "Miss Fortune")

    def test_resolve_compact_draft_sides_assigns_first_five_to_blue_side(self) -> None:
        vocab = ["Ornn", "Dr. Mundo", "Ryze", "Lucian", "Nami", "Renekton", "Qiyana", "Anivia", "Varus", "Alistar"]
        out = resolve_compact_draft_sides(
            text="Ornn DrMundo Ryze Lucian Nami Renekton Qiyana Anivia Varus Alistar",
            vocab=vocab,
            blue_team="Team B",
            team_a="Team A",
            team_b="Team B",
            fuzzy_match_one_fn=lambda token, choices, threshold: (token if token in choices else next((c for c in choices if c.replace('.', '') == token), None), []),
        )
        self.assertEqual(out["blue_side_team"], "Team B")
        self.assertEqual(out["red_side_team"], "Team A")
        self.assertEqual(out["picks_b"], ["Ornn", "Dr. Mundo", "Ryze", "Lucian", "Nami"])
        self.assertEqual(out["picks_a"], ["Renekton", "Qiyana", "Anivia", "Varus", "Alistar"])

    def test_build_live_bet_pregame_summary_returns_team_rows_and_profiles(self) -> None:
        df_scope = pd.DataFrame(
            [
                {"team": "Team A", "opponent": "Team C", "league": "Cup 2026", "date": "2026-03-01", "win": 1, "game_time_min": 30, "total_kills": 34, "total_towers": 12, "total_dragons": 5, "total_nashors": 2, "total_inhibitors": 1},
                {"team": "Team A", "opponent": "Team D", "league": "Cup 2026", "date": "2026-03-02", "win": 1, "game_time_min": 31, "total_kills": 33, "total_towers": 11, "total_dragons": 4, "total_nashors": 1, "total_inhibitors": 1},
                {"team": "Team B", "opponent": "Team E", "league": "Cup 2026", "date": "2026-03-01", "win": 0, "game_time_min": 35, "total_kills": 22, "total_towers": 8, "total_dragons": 3, "total_nashors": 1, "total_inhibitors": 0},
                {"team": "Team B", "opponent": "Team F", "league": "Cup 2026", "date": "2026-03-02", "win": 1, "game_time_min": 34, "total_kills": 24, "total_towers": 9, "total_dragons": 3, "total_nashors": 1, "total_inhibitors": 0},
                {"team": "Team C", "opponent": "Team A", "league": "Cup 2026", "date": "2026-03-01", "win": 0, "game_time_min": 30, "total_kills": 34, "total_towers": 12, "total_dragons": 5, "total_nashors": 2, "total_inhibitors": 1},
                {"team": "Team D", "opponent": "Team A", "league": "Cup 2026", "date": "2026-03-02", "win": 0, "game_time_min": 31, "total_kills": 33, "total_towers": 11, "total_dragons": 4, "total_nashors": 1, "total_inhibitors": 1},
                {"team": "Team E", "opponent": "Team B", "league": "Cup 2026", "date": "2026-03-01", "win": 1, "game_time_min": 35, "total_kills": 22, "total_towers": 8, "total_dragons": 3, "total_nashors": 1, "total_inhibitors": 0},
                {"team": "Team F", "opponent": "Team B", "league": "Cup 2026", "date": "2026-03-02", "win": 0, "game_time_min": 34, "total_kills": 24, "total_towers": 9, "total_dragons": 3, "total_nashors": 1, "total_inhibitors": 0},
            ]
        )
        out = build_live_bet_pregame_summary(
            df_scope=df_scope,
            team_a="Team A",
            team_b="Team B",
            min_to_mmss_fn=lambda x: f"{int(float(x)):02d}:00",
            format_num_fn=lambda x, d: f"{float(x):.{int(d)}f}",
        )
        self.assertFalse(out["stats_view"].empty)
        self.assertFalse(out["profile_view"].empty)
        self.assertTrue(len(out["profile_cards_a"]) > 0)
        self.assertTrue(len(out["profile_cards_b"]) > 0)
        self.assertFalse(out["profile_matrix"].empty)
        self.assertIn("Resolucao", out["profile_matrix"].columns)
        self.assertTrue(any("Resolucao" in row for row in out["profile_cards_a"]))
        self.assertTrue(str(out["summary_line"]).startswith("Pre-jogo:"))
        self.assertEqual(len(out["quick_cards"]), 4)

    def test_build_live_bet_market_value_summary_returns_rows_and_best_value(self) -> None:
        df_scope = pd.DataFrame(
            [
                {"team": "Team A", "opponent": "Team B", "side": "Blue", "total_kills": 34, "total_towers": 12, "total_dragons": 5, "total_nashors": 2, "game_time_min": 31},
                {"team": "Team A", "opponent": "Team C", "side": "Blue", "total_kills": 32, "total_towers": 11, "total_dragons": 4, "total_nashors": 1, "game_time_min": 30},
                {"team": "Team B", "opponent": "Team A", "side": "Red", "total_kills": 34, "total_towers": 12, "total_dragons": 5, "total_nashors": 2, "game_time_min": 31},
                {"team": "Team B", "opponent": "Team D", "side": "Red", "total_kills": 30, "total_towers": 10, "total_dragons": 4, "total_nashors": 1, "game_time_min": 32},
            ]
        )
        pregame = build_live_bet_pregame_summary(
            df_scope=df_scope,
            team_a="Team A",
            team_b="Team B",
            min_to_mmss_fn=lambda x: f"{int(float(x)):02d}:00",
            format_num_fn=lambda x, d: f"{float(x):.{int(d)}f}",
        )
        draft = {
            "market_calls": {
                "Kills": "Favorece over",
                "Torres": "Mais torres",
                "Drag?es": "Mais dragoes",
                "Bar?es": "Mais baroes",
                "Tempo": "Alonga",
            },
            "market_means": {
                "Kills": {"team_a": 33.0, "team_b": 32.0, "scope": 32.5},
                "Torres": {"team_a": 11.5, "team_b": 11.0, "scope": 11.2},
                "Drag?es": {"team_a": 4.6, "team_b": 4.4, "scope": 4.5},
                "Bar?es": {"team_a": 1.5, "team_b": 1.4, "scope": 1.4},
                "Tempo": {"team_a": 31.3, "team_b": 31.7, "scope": 31.2},
            },
            "comp_compare": {
                "tower_total_call": "Menos torres",
                "tower_total_bias": -0.55,
                "tower_total_reason": "Pouca side/split real para quebrar o mapa.",
                "dragons_total_call": "Mais dragoes",
                "dragons_total_bias": 0.20,
                "dragons_total_reason": "Os drafts conseguem girar varias vezes no dragon.",
                "barons_total_call": "Menos baroes",
                "barons_total_bias": -0.14,
                "barons_total_reason": "O jogo tende a resolver cedo demais para muitos baraos.",
            },
        }
        out = build_live_bet_market_value_summary(
            df_scope=df_scope,
            team_a="Team A",
            team_b="Team B",
            blue_side_team="Team A",
            pregame=pregame,
            draft=draft,
            line_values={"Kills": 30.5, "Torres": 10.5, "DragÃµes": 4.5, "BarÃµes": 1.5, "Tempo": 31.5},
            format_num_fn=lambda x, d: f"{float(x):.{int(d)}f}",
            min_to_mmss_fn=lambda x: f"{int(float(x)):02d}:00",
        )
        self.assertFalse(out["rows"].empty)
        self.assertIn("Mercado", out["rows"].columns)
        self.assertIn("Justo", out["rows"].columns)
        self.assertIn("Drivers", out["rows"].columns)
        self.assertIsNotNone(out["best_row"])
        towers_row = out["rows"][out["rows"]["Mercado"].astype(str) == "Torres"].head(1)
        self.assertFalse(towers_row.empty)
        self.assertIn("Macro Menos torres", str(towers_row.iloc[0]["Drivers"]))
        dragons_row = out["rows"][out["rows"]["Mercado"].astype(str).str.contains("Drag", na=False)].head(1)
        self.assertFalse(dragons_row.empty)
        self.assertIn("Macro Mais dragoes", str(dragons_row.iloc[0]["Drivers"]))
        barons_row = out["rows"][out["rows"]["Mercado"].astype(str).str.contains("Bar", na=False)].head(1)
        self.assertFalse(barons_row.empty)
        self.assertIn("Macro Menos baroes", str(barons_row.iloc[0]["Drivers"]))

    def test_build_live_bet_draft_analysis_returns_verdict_and_combo_tables(self) -> None:
        df_players = pd.DataFrame(
            [
                {"gameid": 1, "team": "Team A", "player": "Top A", "pos": "top", "side": "Blue", "champ": "Gnar", "result": 1},
                {"gameid": 1, "team": "Team A", "player": "Jng A", "pos": "jng", "side": "Blue", "champ": "Vi", "result": 1},
                {"gameid": 1, "team": "Team A", "player": "Mid A", "pos": "mid", "side": "Blue", "champ": "Orianna", "result": 1},
                {"gameid": 1, "team": "Team A", "player": "Adc A", "pos": "bot", "side": "Blue", "champ": "Jinx", "result": 1},
                {"gameid": 1, "team": "Team A", "player": "Sup A", "pos": "sup", "side": "Blue", "champ": "Nautilus", "result": 1},
                {"gameid": 1, "team": "Team B", "player": "Top B", "pos": "top", "side": "Red", "champ": "Ksante", "result": 0},
                {"gameid": 1, "team": "Team B", "player": "Jng B", "pos": "jng", "side": "Red", "champ": "Xin Zhao", "result": 0},
                {"gameid": 1, "team": "Team B", "player": "Mid B", "pos": "mid", "side": "Red", "champ": "Ahri", "result": 0},
                {"gameid": 1, "team": "Team B", "player": "Adc B", "pos": "bot", "side": "Red", "champ": "Xayah", "result": 0},
                {"gameid": 1, "team": "Team B", "player": "Sup B", "pos": "sup", "side": "Red", "champ": "Rakan", "result": 0},
                {"gameid": 2, "team": "Team A", "player": "Top A", "pos": "top", "side": "Blue", "champ": "Gnar", "result": 1},
                {"gameid": 2, "team": "Team A", "player": "Jng A", "pos": "jng", "side": "Blue", "champ": "Vi", "result": 1},
                {"gameid": 2, "team": "Team A", "player": "Mid A", "pos": "mid", "side": "Blue", "champ": "Orianna", "result": 1},
                {"gameid": 2, "team": "Team A", "player": "Adc A", "pos": "bot", "side": "Blue", "champ": "Jinx", "result": 1},
                {"gameid": 2, "team": "Team A", "player": "Sup A", "pos": "sup", "side": "Blue", "champ": "Nautilus", "result": 1},
                {"gameid": 2, "team": "Team B", "player": "Top B", "pos": "top", "side": "Red", "champ": "Ksante", "result": 0},
                {"gameid": 2, "team": "Team B", "player": "Jng B", "pos": "jng", "side": "Red", "champ": "Xin Zhao", "result": 0},
                {"gameid": 2, "team": "Team B", "player": "Mid B", "pos": "mid", "side": "Red", "champ": "Ahri", "result": 0},
                {"gameid": 2, "team": "Team B", "player": "Adc B", "pos": "bot", "side": "Red", "champ": "Xayah", "result": 0},
                {"gameid": 2, "team": "Team B", "player": "Sup B", "pos": "sup", "side": "Red", "champ": "Rakan", "result": 0},
                {"gameid": 3, "team": "Team A", "player": "Top A", "pos": "top", "side": "Blue", "champ": "Gnar", "result": 1},
                {"gameid": 3, "team": "Team A", "player": "Jng A", "pos": "jng", "side": "Blue", "champ": "Vi", "result": 1},
                {"gameid": 3, "team": "Team A", "player": "Mid A", "pos": "mid", "side": "Blue", "champ": "Orianna", "result": 1},
                {"gameid": 3, "team": "Team A", "player": "Adc A", "pos": "bot", "side": "Blue", "champ": "Jinx", "result": 1},
                {"gameid": 3, "team": "Team A", "player": "Sup A", "pos": "sup", "side": "Blue", "champ": "Nautilus", "result": 1},
                {"gameid": 3, "team": "Team B", "player": "Top B", "pos": "top", "side": "Red", "champ": "Ksante", "result": 0},
                {"gameid": 3, "team": "Team B", "player": "Jng B", "pos": "jng", "side": "Red", "champ": "Xin Zhao", "result": 0},
                {"gameid": 3, "team": "Team B", "player": "Mid B", "pos": "mid", "side": "Red", "champ": "Ahri", "result": 0},
                {"gameid": 3, "team": "Team B", "player": "Adc B", "pos": "bot", "side": "Red", "champ": "Xayah", "result": 0},
                {"gameid": 3, "team": "Team B", "player": "Sup B", "pos": "sup", "side": "Red", "champ": "Rakan", "result": 0},
                {"gameid": 4, "team": "Team A", "player": "Top A", "pos": "top", "side": "Blue", "champ": "Gnar", "result": 1},
                {"gameid": 4, "team": "Team A", "player": "Jng A", "pos": "jng", "side": "Blue", "champ": "Vi", "result": 1},
                {"gameid": 4, "team": "Team A", "player": "Mid A", "pos": "mid", "side": "Blue", "champ": "Orianna", "result": 1},
                {"gameid": 4, "team": "Team A", "player": "Adc A", "pos": "bot", "side": "Blue", "champ": "Jinx", "result": 1},
                {"gameid": 4, "team": "Team A", "player": "Sup A", "pos": "sup", "side": "Blue", "champ": "Nautilus", "result": 1},
                {"gameid": 4, "team": "Team B", "player": "Top B", "pos": "top", "side": "Red", "champ": "Ksante", "result": 0},
                {"gameid": 4, "team": "Team B", "player": "Jng B", "pos": "jng", "side": "Red", "champ": "Xin Zhao", "result": 0},
                {"gameid": 4, "team": "Team B", "player": "Mid B", "pos": "mid", "side": "Red", "champ": "Ahri", "result": 0},
                {"gameid": 4, "team": "Team B", "player": "Adc B", "pos": "bot", "side": "Red", "champ": "Xayah", "result": 0},
                {"gameid": 4, "team": "Team B", "player": "Sup B", "pos": "sup", "side": "Red", "champ": "Rakan", "result": 0},
                {"gameid": 5, "team": "Team A", "player": "Top A", "pos": "top", "side": "Blue", "champ": "Gnar", "result": 0},
                {"gameid": 5, "team": "Team A", "player": "Jng A", "pos": "jng", "side": "Blue", "champ": "Vi", "result": 0},
                {"gameid": 5, "team": "Team A", "player": "Mid A", "pos": "mid", "side": "Blue", "champ": "Orianna", "result": 0},
                {"gameid": 5, "team": "Team A", "player": "Adc A", "pos": "bot", "side": "Blue", "champ": "Jinx", "result": 0},
                {"gameid": 5, "team": "Team A", "player": "Sup A", "pos": "sup", "side": "Blue", "champ": "Nautilus", "result": 0},
                {"gameid": 5, "team": "Team B", "player": "Top B", "pos": "top", "side": "Red", "champ": "Ksante", "result": 1},
                {"gameid": 5, "team": "Team B", "player": "Jng B", "pos": "jng", "side": "Red", "champ": "Xin Zhao", "result": 1},
                {"gameid": 5, "team": "Team B", "player": "Mid B", "pos": "mid", "side": "Red", "champ": "Ahri", "result": 1},
                {"gameid": 5, "team": "Team B", "player": "Adc B", "pos": "bot", "side": "Red", "champ": "Xayah", "result": 1},
                {"gameid": 5, "team": "Team B", "player": "Sup B", "pos": "sup", "side": "Red", "champ": "Rakan", "result": 1},
            ]
        )
        df_teams = pd.DataFrame(
            [
                {"gameid": 1, "team": "Team A", "total_kills": 35, "total_towers": 12, "game_time_min": 31},
                {"gameid": 2, "team": "Team A", "total_kills": 33, "total_towers": 11, "game_time_min": 30},
                {"gameid": 3, "team": "Team A", "total_kills": 34, "total_towers": 12, "game_time_min": 30},
                {"gameid": 4, "team": "Team A", "total_kills": 32, "total_towers": 11, "game_time_min": 29},
                {"gameid": 5, "team": "Team A", "total_kills": 28, "total_towers": 9, "game_time_min": 33},
                {"gameid": 1, "team": "Team B", "total_kills": 35, "total_towers": 12, "game_time_min": 31},
                {"gameid": 2, "team": "Team B", "total_kills": 33, "total_towers": 11, "game_time_min": 30},
                {"gameid": 3, "team": "Team B", "total_kills": 34, "total_towers": 12, "game_time_min": 30},
                {"gameid": 4, "team": "Team B", "total_kills": 32, "total_towers": 11, "game_time_min": 29},
                {"gameid": 5, "team": "Team B", "total_kills": 28, "total_towers": 9, "game_time_min": 33},
            ]
        )
        out = build_live_bet_draft_analysis(
            df_scope_players=df_players,
            df_scope_teams=df_teams,
            team_a="Team A",
            team_b="Team B",
            picks_a=["Gnar", "Vi", "Orianna", "Jinx", "Nautilus"],
            picks_b=["Ksante", "Xin Zhao", "Ahri", "Xayah", "Rakan"],
            players_a=["Top A", "Jng A", "Mid A", "Adc A", "Sup A"],
            players_b=["Top B", "Jng B", "Mid B", "Adc B", "Sup B"],
            format_num_fn=lambda x, d: f"{float(x):.{int(d)}f}",
            champion_registry=self._champion_registry(),
        )
        self.assertFalse(out["lane_view"].empty)
        self.assertFalse(out["duos_a"].empty)
        self.assertIn("summary", out["verdict"])
        self.assertIn("Kills", out["market_calls"])
        self.assertTrue(any("Drag" in str(key) for key in out["market_calls"].keys()))
        self.assertTrue(any("Bar" in str(key) for key in out["market_calls"].keys()))
        self.assertIn("WR Score", out["duos_a"].columns)
        self.assertTrue(out["comp_compare"])
        self.assertIn("Leitura Lane", out["lane_view"].columns)
        self.assertIn("Early", out["lane_view"].columns)
        self.assertIn("Late", out["lane_view"].columns)
        self.assertIn("Plano Team A", out["lane_view"].columns)
        self.assertIn("Plano Team B", out["lane_view"].columns)
        self.assertIn("Meta Team A", out["lane_view"].columns)
        self.assertIn("Meta Team B", out["lane_view"].columns)
        self.assertIn("Meta Global Team A", out["lane_view"].columns)
        self.assertIn("Meta Global Team B", out["lane_view"].columns)
        self.assertIn("Alerta Liga Team A", out["lane_view"].columns)
        self.assertIn("Alerta Liga Team B", out["lane_view"].columns)
        self.assertIn("draft_confidence", out)
        self.assertIn("label", out["draft_confidence"])
        self.assertIn("player_leadership_a", out)
        self.assertIn("player_leadership_b", out)
        self.assertIn("best_player", out["player_leadership_a"])
        self.assertIn("carry_spot", out["player_leadership_b"])
        self.assertIn("meta_leadership_a", out)
        self.assertIn("meta_leadership_b", out)
        self.assertIn("best_meta", out["meta_leadership_a"])
        self.assertIn("best_global_meta", out["meta_leadership_a"])
        self.assertIn("weak_global_meta", out["meta_leadership_b"])
        self.assertIn("plan_profiles", out["comp_compare"])
        self.assertTrue(str(((out["comp_compare"].get("plan_profiles") or {}).get("Team A") or {}).get("win_condition")))
        self.assertIn("combo_support_a", out)
        self.assertIn("combo_support_b", out)
        self.assertIsInstance(out.get("combo_support_a"), dict)

    def test_build_live_bet_player_prop_summary_uses_spot_team_context_for_kills(self) -> None:
        df_players = pd.DataFrame(
            [
                {"gameid": 1, "date": "2026-03-01", "team": "Team A", "player": "Adc A", "pos": "bot", "champ": "Jinx", "side": "Blue", "kills": 4, "deaths": 2, "assists": 3, "teamkills": 15, "teamdeaths": 10},
                {"gameid": 2, "date": "2026-03-02", "team": "Team A", "player": "Adc A", "pos": "bot", "champ": "Jinx", "side": "Blue", "kills": 5, "deaths": 1, "assists": 4, "teamkills": 16, "teamdeaths": 8},
                {"gameid": 3, "date": "2026-03-03", "team": "Team A", "player": "Adc A", "pos": "bot", "champ": "Jinx", "side": "Red", "kills": 6, "deaths": 3, "assists": 2, "teamkills": 18, "teamdeaths": 11},
                {"gameid": 4, "date": "2026-03-04", "team": "Team A", "player": "Adc A", "pos": "bot", "champ": "Jinx", "side": "Blue", "kills": 7, "deaths": 2, "assists": 5, "teamkills": 20, "teamdeaths": 9},
                {"gameid": 5, "date": "2026-03-05", "team": "Team A", "player": "Adc A", "pos": "bot", "champ": "Ezreal", "side": "Blue", "kills": 3, "deaths": 2, "assists": 6, "teamkills": 14, "teamdeaths": 9},
                {"gameid": 6, "date": "2026-03-06", "team": "Team A", "player": "Adc A", "pos": "bot", "champ": "Jinx", "side": "Blue", "kills": 6, "deaths": 2, "assists": 4, "teamkills": 19, "teamdeaths": 10},
            ]
        )
        out = build_live_bet_player_prop_summary(
            df_scope_players=df_players,
            team_name="Team A",
            opponent_name="Team B",
            player_name="Adc A",
            champ_name="Jinx",
            side_label="Blue",
            prop_name="Kills",
            line_value=6.5,
            registry=self._champion_registry(),
            comp_compare={"kills_delta": 1.2, "late_team": "Team A", "engage_team": "Team B"},
            expected_team_kills=19.0,
            expected_team_deaths=13.0,
            format_num_fn=lambda x, d: f"{float(x):.{int(d)}f}",
        )
        self.assertTrue(out["ok"])
        self.assertIsNotNone(out["share_context"])
        self.assertGreater(float(out["share_context"]), 5.0)
        self.assertEqual(float(out["expected_team_kills"]), 19.0)
        self.assertIn("Kills time 19.0", str(out["drivers"]))
        self.assertIn("Kill share", str(out["drivers"]))

    def test_build_live_bet_player_prop_summary_uses_expected_deaths_for_death_props(self) -> None:
        df_players = pd.DataFrame(
            [
                {"gameid": 1, "date": "2026-03-01", "team": "Team A", "player": "Sup A", "pos": "sup", "champ": "Nautilus", "side": "Blue", "kills": 1, "deaths": 5, "assists": 10, "teamkills": 15, "teamdeaths": 12},
                {"gameid": 2, "date": "2026-03-02", "team": "Team A", "player": "Sup A", "pos": "sup", "champ": "Nautilus", "side": "Blue", "kills": 0, "deaths": 6, "assists": 11, "teamkills": 16, "teamdeaths": 13},
                {"gameid": 3, "date": "2026-03-03", "team": "Team A", "player": "Sup A", "pos": "sup", "champ": "Nautilus", "side": "Red", "kills": 1, "deaths": 4, "assists": 9, "teamkills": 13, "teamdeaths": 10},
                {"gameid": 4, "date": "2026-03-04", "team": "Team A", "player": "Sup A", "pos": "sup", "champ": "Nautilus", "side": "Blue", "kills": 1, "deaths": 5, "assists": 12, "teamkills": 17, "teamdeaths": 11},
            ]
        )
        out = build_live_bet_player_prop_summary(
            df_scope_players=df_players,
            team_name="Team A",
            opponent_name="Team B",
            player_name="Sup A",
            champ_name="Nautilus",
            side_label="Blue",
            prop_name="Deaths",
            line_value=4.5,
            registry=self._champion_registry(),
            comp_compare={"kills_delta": 0.8, "late_team": "Team B", "engage_team": "Team A"},
            expected_team_kills=15.0,
            expected_team_deaths=14.0,
            format_num_fn=lambda x, d: f"{float(x):.{int(d)}f}",
        )
        self.assertTrue(out["ok"])
        self.assertEqual(float(out["expected_team_deaths"]), 14.0)
        self.assertIn("Deaths esperadas 14.0", str(out["drivers"]))
        self.assertIn("Death share", str(out["drivers"]))

    def test_lane_cards_and_tactical_notes_identify_carry_engage_and_exposed_roles(self) -> None:
        lane_view = pd.DataFrame(
            [
                {
                    "Pos": "bot",
                    "Team A Player": "Adc A",
                    "Team A Champ": "Jinx",
                    "WR Player Team A": "64.0",
                    "WR Liga Team A": "52.0",
                    "N Player Team A": 8,
                    "WR Player Blue Team A": "68.0",
                    "WR Player Red Team A": "54.0",
                    "WR Champ Blue Team A": "55.0",
                    "WR Champ Red Team A": "50.0",
                    "N Player Blue Team A": 5,
                    "N Player Red Team A": 3,
                    "N Champ Blue Team A": 20,
                    "N Champ Red Team A": 18,
                    "Team B Player": "Adc B",
                    "Team B Champ": "Xayah",
                    "WR Player Team B": "58.0",
                    "WR Liga Team B": "51.0",
                    "N Player Team B": 7,
                    "WR Player Blue Team B": "61.0",
                    "WR Player Red Team B": "55.0",
                    "WR Champ Blue Team B": "52.0",
                    "WR Champ Red Team B": "49.0",
                    "N Player Blue Team B": 4,
                    "N Player Red Team B": 3,
                    "N Champ Blue Team B": 19,
                    "N Champ Red Team B": 16,
                    "Leitura Lane": "Lane Team A",
                    "Early": "Team A",
                    "Late": "Team A",
                    "Confianca Lane": "Alta",
                    "Edge (p.p.)": "6.0",
                    "Plano Team A": "Jinx pressiona wave e placa cedo.",
                    "Plano Team B": "Xayah vira condicao forte de teamfight e objetivo.",
                },
                {
                    "Pos": "sup",
                    "Team A Player": "Sup A",
                    "Team A Champ": "Nautilus",
                    "WR Player Team A": "60.0",
                    "WR Liga Team A": "50.0",
                    "N Player Team A": 10,
                    "WR Player Blue Team A": "63.0",
                    "WR Player Red Team A": "56.0",
                    "WR Champ Blue Team A": "53.0",
                    "WR Champ Red Team A": "50.0",
                    "N Player Blue Team A": 6,
                    "N Player Red Team A": 4,
                    "N Champ Blue Team A": 22,
                    "N Champ Red Team A": 21,
                    "Team B Player": "Sup B",
                    "Team B Champ": "Rakan",
                    "WR Player Team B": "57.0",
                    "WR Liga Team B": "50.0",
                    "N Player Team B": 9,
                    "WR Player Blue Team B": "58.0",
                    "WR Player Red Team B": "56.0",
                    "WR Champ Blue Team B": "51.0",
                    "WR Champ Red Team B": "50.0",
                    "N Player Blue Team B": 5,
                    "N Player Red Team B": 4,
                    "N Champ Blue Team B": 19,
                    "N Champ Red Team B": 19,
                    "Leitura Lane": "Equilibrado",
                    "Early": "Equilibrado",
                    "Late": "Team B",
                    "Confianca Lane": "Media",
                    "Edge (p.p.)": "1.0",
                    "Plano Team A": "Nautilus acha engage e roam cedo.",
                    "Plano Team B": "Rakan continua abrindo luta e pick depois.",
                },
            ]
        )
        cards_a, cards_b = _build_lane_cards(lane_view, "Team A", "Team B", self._champion_registry())
        jinx_row = next(row for row in cards_a if str(row.get("champ")) == "Jinx")
        naut_row = next(row for row in cards_a if str(row.get("champ")) == "Nautilus")
        self.assertEqual(str(jinx_row.get("primary_label")), "carrega")
        self.assertIn(str(jinx_row.get("risk_label")), {"Exposto", "Muito exposto"})
        self.assertEqual(str(naut_row.get("primary_label")), "inicia")
        self.assertIn("Jinx", str(jinx_row.get("team_plan") or ""))

        notes = build_live_bet_tactical_notes(
            team_a="Team A",
            team_b="Team B",
            picks_a=["Gnar", "Vi", "Orianna", "Jinx", "Nautilus"],
            picks_b=["Ksante", "Xin Zhao", "Ahri", "Xayah", "Rakan"],
            players_a=["Top A", "Jng A", "Mid A", "Adc A", "Sup A"],
            players_b=["Top B", "Jng B", "Mid B", "Adc B", "Sup B"],
            comp_compare={"early_delta": 1.2, "late_delta": 0.4, "tower_total_reason": "", "reasons": []},
            lane_view=lane_view,
            board_rows_a=cards_a,
            board_rows_b=cards_b,
            registry=self._champion_registry(),
            format_num_fn=lambda x, d: f"{float(x):.{int(d)}f}",
        )
        notes_text = " | ".join(notes)
        self.assertIn("Condicao de dano Team A: Adc A (Jinx).", notes_text)
        self.assertIn("Se Sup A (Nautilus) achar acesso em Adc B (Xayah)", notes_text)

    def test_build_player_spot_leadership_prefers_carry_for_best_player_when_context_fits(self) -> None:
        df_players = pd.DataFrame(
            [
                {"gameid": 1, "date": "2026-03-01", "team": "Team A", "player": "Top A", "pos": "top", "champ": "Gnar", "side": "Blue", "result": 1, "kills": 2, "deaths": 2, "assists": 5, "teamkills": 12, "teamdeaths": 10, "damagetochampions": 18000, "visionscore": 22, "gamelength": 1860},
                {"gameid": 1, "date": "2026-03-01", "team": "Team A", "player": "Adc A", "pos": "bot", "champ": "Jinx", "side": "Blue", "result": 1, "kills": 7, "deaths": 1, "assists": 4, "teamkills": 12, "teamdeaths": 10, "damagetochampions": 29000, "visionscore": 14, "gamelength": 1860},
                {"gameid": 1, "date": "2026-03-01", "team": "Team A", "player": "Sup A", "pos": "sup", "champ": "Nautilus", "side": "Blue", "result": 1, "kills": 1, "deaths": 4, "assists": 10, "teamkills": 12, "teamdeaths": 10, "damagetochampions": 9000, "visionscore": 65, "gamelength": 1860},
                {"gameid": 2, "date": "2026-03-02", "team": "Team A", "player": "Top A", "pos": "top", "champ": "Gnar", "side": "Blue", "result": 1, "kills": 3, "deaths": 2, "assists": 4, "teamkills": 15, "teamdeaths": 11, "damagetochampions": 19000, "visionscore": 24, "gamelength": 1920},
                {"gameid": 2, "date": "2026-03-02", "team": "Team A", "player": "Adc A", "pos": "bot", "champ": "Jinx", "side": "Blue", "result": 1, "kills": 8, "deaths": 2, "assists": 5, "teamkills": 15, "teamdeaths": 11, "damagetochampions": 32000, "visionscore": 16, "gamelength": 1920},
                {"gameid": 2, "date": "2026-03-02", "team": "Team A", "player": "Sup A", "pos": "sup", "champ": "Nautilus", "side": "Blue", "result": 1, "kills": 0, "deaths": 5, "assists": 12, "teamkills": 15, "teamdeaths": 11, "damagetochampions": 10000, "visionscore": 70, "gamelength": 1920},
            ]
        )
        out = build_live_bet_draft_analysis(
            df_scope_players=df_players,
            df_scope_teams=pd.DataFrame([{"gameid": 1, "team": "Team A", "total_kills": 30, "total_towers": 11, "game_time_min": 31}]),
            team_a="Team A",
            team_b="Team B",
            blue_side_team="Team A",
            picks_a=["Gnar", "Vi", "Orianna", "Jinx", "Nautilus"],
            picks_b=["Ksante", "Xin Zhao", "Ahri", "Xayah", "Rakan"],
            players_a=["Top A", "Jng A", "Mid A", "Adc A", "Sup A"],
            players_b=["Top B", "Jng B", "Mid B", "Adc B", "Sup B"],
            format_num_fn=lambda x, d: f"{float(x):.{int(d)}f}",
            champion_registry=self._champion_registry(),
        )
        lead_a = dict(out["player_leadership_a"])
        meta_a = dict(out["meta_leadership_a"])
        self.assertEqual(str((lead_a.get("best_player") or {}).get("player")), "Adc A")
        self.assertEqual(str((lead_a.get("carry_spot") or {}).get("champ")), "Jinx")
        self.assertEqual(str((lead_a.get("engage_engine") or {}).get("champ")), "Nautilus")
        self.assertEqual(str((meta_a.get("best_meta") or {}).get("champ")), "Jinx")
        self.assertEqual(str((meta_a.get("best_global_meta") or {}).get("champ")), "Jinx")
        self.assertIn(str((meta_a.get("best_global_meta") or {}).get("global_meta_label")), {"Premium", "Contestado", "Situacional", "Sem base"})

    def test_build_live_bet_live_assessment_detects_window_and_shift(self) -> None:
        df_scope = pd.DataFrame(
            [
                {"team": "Team A", "win": 1, "golddiffat15_for": 1800, "golddiffat20_for": 2200, "firstdragon": 1, "firsttower": 1},
                {"team": "Team A", "win": 1, "golddiffat15_for": 1500, "golddiffat20_for": 2400, "firstdragon": 1, "firsttower": 1},
                {"team": "Team A", "win": 1, "golddiffat15_for": 1600, "golddiffat20_for": 2100, "firstdragon": 1, "firsttower": 0},
                {"team": "Team A", "win": 0, "golddiffat15_for": -1200, "golddiffat20_for": -1800, "firstdragon": 0, "firsttower": 0},
                {"team": "Team B", "win": 0, "golddiffat15_for": -1800, "golddiffat20_for": -2300, "firstdragon": 0, "firsttower": 0},
                {"team": "Team B", "win": 0, "golddiffat15_for": -1400, "golddiffat20_for": -2100, "firstdragon": 0, "firsttower": 0},
                {"team": "Team B", "win": 0, "golddiffat15_for": -1600, "golddiffat20_for": -2200, "firstdragon": 1, "firsttower": 0},
                {"team": "Team B", "win": 1, "golddiffat15_for": 1300, "golddiffat20_for": 1700, "firstdragon": 1, "firsttower": 1},
            ]
        )
        out = build_live_bet_live_assessment(
            team_a="Team A",
            team_b="Team B",
            df_scope=df_scope,
            minute=23,
            gold_diff=1800,
            kills_a=11,
            kills_b=6,
            towers_a=6,
            towers_b=2,
            dragons_a=3,
            dragons_b=1,
            barons_a=1,
            barons_b=0,
            inhibitors_a=1,
            inhibitors_b=0,
            comp_compare={
                "early_team": "Team A",
                "mid_team": "Team A",
                "late_team": "Team B",
                "objective_team": "Team A",
                "engage_team": "Team A",
                "kills_call": "Team A acelera",
                "towers_call": "Team A pressiona",
                "dragons_call": "Team A setup",
                "barons_call": "Team A setup",
                "time_call": "Neutro",
            },
            pregame_resolution={"Kills": "Under", "Torres": "Choque", "Tempo": "Neutro"},
            baselines={"kills": 28.0, "towers": 9.5, "dragons": 4.0, "barons": 1.2, "inhibitors": 0.8, "time": 31.0},
        )
        self.assertEqual(out["window"], "20+")
        self.assertEqual(out["leader"], "Team A")
        self.assertIn("Team A", out["conversion"])
        self.assertIn("Kills", out["market_live"])
        self.assertIn("Inibidores", out["market_live"])
        self.assertEqual(out["soul_point"], "Team A em soul point")
        self.assertEqual(out["soul_pressure"], "Soul iminente para Team A")
        self.assertEqual(out["baron_pressure"], "Team A ja converteu pressao de barao")
        self.assertTrue(any(str(row).startswith("Kills:") for row in out["shifts"]))
        self.assertIn("live_confidence", out)
        self.assertIn("conversion_history_a", out)
        self.assertIn("conversion_history_b", out)
        self.assertEqual(out["live_confidence"]["label"], "Media")
        self.assertIn("Team A", out["conversion_history_a"]["summary"])
        self.assertIn("first dragon", out["conversion_history_a"]["summary"])
        self.assertGreaterEqual(int(out["conversion_history_a"].get("source_games") or 0), int(out["conversion_history_a"].get("games") or 0))


if __name__ == "__main__":
    unittest.main()
