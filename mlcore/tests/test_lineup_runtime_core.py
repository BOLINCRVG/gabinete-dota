import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from app_core.lineup_runtime_core import (
    autofill_lineup_state,
    build_manual_lineup_choices,
    build_ml_lineup_choices,
    lineup_continuity_from_details,
    resolve_players_layer_config,
    score_manual_lineup_team,
    transfer_signal_from_details,
)


class LineupRuntimeCoreTests(unittest.TestCase):
    def test_build_manual_lineup_choices_and_score(self):
        df_imp = pd.DataFrame(
            [
                {"team": "T1", "pos": "Top", "player": "Zeus", "gp": 10, "score": 2.0},
                {"team": "T1", "pos": "Jungle", "player": "Oner", "gp": 9, "score": 1.5},
            ]
        )
        df_pr = pd.DataFrame(
            [
                {"team": "T1", "pos": "Top", "player": "Zeus"},
                {"team": "T1", "pos": "Jungle", "player": "Oner"},
            ]
        )
        choices = build_manual_lineup_choices(
            df_imp=df_imp,
            df_pr=df_pr,
            team_ui="T1",
            team_key="T1",
            pos_canon=["Top", "Jungle"],
            norm_pos_fn=lambda x: str(x),
        )
        self.assertEqual(choices["Top"][0], "Zeus")
        score = score_manual_lineup_team(
            team_key="T1",
            lineup={"Top": "Zeus", "Jungle": "Oner"},
            df_imp=df_imp,
            pos_canon=["Top", "Jungle"],
            min_gp=5,
            best_match_fn=lambda q, xs, cutoff=0.0: q,
        )
        self.assertEqual(score[1], 2)

    def test_build_ml_lineup_choices(self):
        df_pr = pd.DataFrame(
            [
                {"team": "T1", "pos": "Bot", "player": "Gumayusi", "playerid": "1", "gameid": "g1", "date": "2026-01-01"},
                {"team": "T1", "pos": "Bot", "player": "Guma", "playerid": "1", "gameid": "g2", "date": "2026-01-02"},
            ]
        )
        choices, disp = build_ml_lineup_choices(
            df_pr=df_pr,
            team_ui="T1",
            pos_canon_ml=["Bot"],
            norm_pos_ml_fn=lambda x: str(x),
        )
        self.assertEqual(choices["Bot"], ["1"])
        self.assertIn("Gumayusi", disp["1"])

    def test_autofill_resolve_and_signals(self):
        state = {}
        called = {"saved": False}

        def _latest(df, team, filters, league_fixed_value, pos_col, player_col, pos_list):
            return {str(pos): f"{team}-{player_col}-{pos}" for pos in pos_list}

        ok = autofill_lineup_state(
            session_state=state,
            lineup_key="abc",
            do_auto=True,
            df_pr=pd.DataFrame([{"x": 1}]),
            team_a="A",
            team_b="B",
            filters={},
            league_fixed_value="LCK",
            pos_canon=["Top"],
            pos_canon_ml=["Top"],
            safe_key_fn=lambda x: x.lower(),
            latest_lineup_for_team_fn=_latest,
            save_lineup_for_matchup_fn=lambda key: called.__setitem__("saved", True),
        )
        self.assertTrue(ok)
        self.assertTrue(called["saved"])
        self.assertEqual(state["lineup_A_top"], "A-player-Top")

        cfg = resolve_players_layer_config(
            artifact_path="C:/tmp/definitely_missing_ml_artifact.json",
            session_state={"ml_players_w_lineup": 1.4},
            load_artifact_cached_fn=lambda sig: (_ for _ in ()).throw(RuntimeError("x")),
            artifact_sig_fn=lambda path: (path,),
        )
        self.assertFalse(cfg["enabled"])
        self.assertEqual(cfg["weight_source"], "ui")

        sig = transfer_signal_from_details(
            [{"used": True, "blue_dbg": {"last_tier": "A", "cur_tier": "B"}}],
            tier_priority_fn=lambda x: {"A": 1, "B": 2}[x],
        )
        self.assertGreater(sig, 0.0)
        cont = lineup_continuity_from_details(
            [{"used": True, "blue_dbg": {"gp": 12, "transfer_scale": 1.0}}],
            freshness_avg=0.7,
            cov_ratio=0.8,
        )
        self.assertGreater(cont, 0.0)

    def test_resolve_players_layer_config_uses_artifact_sibling_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifact_path = root / "custom" / "ml_artifact.json"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("{}", encoding="utf-8")
            sibling_players = artifact_path.parent / "players_artifact.json"
            sibling_players.write_text("{}", encoding="utf-8")
            cfg = resolve_players_layer_config(
                artifact_path=str(artifact_path),
                session_state={"ml_players_w_lineup": 1.25},
                load_artifact_cached_fn=lambda _sig: SimpleNamespace(meta={}),
                artifact_sig_fn=lambda path: (path,),
            )
        self.assertTrue(cfg["enabled"])
        self.assertEqual(Path(cfg["players_path"]), sibling_players)


if __name__ == "__main__":
    unittest.main()
