import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from app_core.ml_artifact_pricing_core import (
    mlcore_p_map_v2,
    p_from_diff_points,
    strength_breakdown,
    team_meta_from_artifact,
)


def _clip_prob(p: float) -> float:
    return float(max(1e-6, min(1.0 - 1e-6, float(p))))


class _FakeCalibrator:
    def __init__(self, add: float = 0.0, fail: bool = False):
        self.add = float(add)
        self.fail = bool(fail)

    def predict_proba(self, p: float) -> float:
        if self.fail:
            raise RuntimeError("calibrator boom")
        return float(p) + self.add


class _FakeModel:
    def __init__(self):
        self.ratings = {"Blue": 10.0, "Red": 0.0}
        self.games_played = {"Blue": 20, "Red": 10}
        self.team_meta = {"Fallback": {"league": "LCK", "tier": "T1"}}
        self.params = SimpleNamespace(scale_points=400.0, alpha=80.0)

    def rating_raw(self, team: str) -> float:
        return {"Blue": 20.0, "Red": 0.0}.get(team, 0.0)

    def residual_eff(self, team: str) -> float:
        return {"Blue": 10.0, "Red": 0.0}.get(team, 0.0)

    def n_games(self, team: str) -> int:
        return {"Blue": 20, "Red": 10}.get(team, 0)

    def _shrink_factor(self, n_games: int) -> float:
        return 0.5 if int(n_games) > 0 else 0.0

    def p_blue(self, _blue: str, _red: str, **_kwargs) -> float:
        return 0.64


class _FakeModelWithComponents(_FakeModel):
    def components(self, team: str, **_kwargs):
        if team == "Blue":
            return {
                "residual_raw": 20.0,
                "residual_eff": 10.0,
                "games_played": 20,
                "shrink_factor": 0.5,
                "league_strength": 12.0,
                "tier_strength": 8.0,
                "tier_prior_points": 1.0,
                "league_prior_points": 0.7,
                "base_strength": 31.7,
            }
        return {
            "residual_raw": 0.0,
            "residual_eff": 0.0,
            "games_played": 0,
            "shrink_factor": 0.0,
            "league_strength": 0.0,
            "tier_strength": 0.0,
            "tier_prior_points": 1.0,
            "league_prior_points": 0.7,
            "base_strength": 1.7,
        }


class TestMlArtifactPricingCore(unittest.TestCase):
    def test_team_meta_from_artifact_falls_back_to_model(self):
        art = SimpleNamespace(meta={}, model=_FakeModel())
        out = team_meta_from_artifact(art)
        self.assertEqual(out["Fallback"]["league"], "LCK")

    def test_p_from_diff_points_midpoint(self):
        self.assertAlmostEqual(p_from_diff_points(0.0, scale_points=400.0, clip_prob_fn=_clip_prob), 0.5, places=9)

    def test_strength_breakdown_uses_hints_and_achievements(self):
        art = SimpleNamespace(model=_FakeModel())
        out = strength_breakdown(
            art=art,
            team="Blue",
            as_of="2026-03-07",
            team_meta={"Blue": {"league": "CBLOL", "tier": "T2"}},
            ach_layer={"enabled": True, "team_idx": {}, "weights_points": {"x": 1.0}},
            norm_league_fn=lambda s: str(s or "").strip(),
            tier_from_league_fn=lambda lg: "T1" if lg == "LCK" else "MINOR",
            tier_base_points_fn=lambda tr: 5.0 if tr == "T1" else 1.0,
            league_base_points_fn=lambda lg: 7.0 if lg == "LCK" else 2.0,
            team_boost_points_fn=lambda *_args: 3.0,
            league_hint="LCK",
            tier_hint=None,
        )
        self.assertEqual(out["league"], "LCK")
        self.assertEqual(out["tier"], "T1")
        self.assertAlmostEqual(float(out["shrink_factor"]), 0.5, places=9)
        self.assertAlmostEqual(float(out["total_strength"]), 10.0 + 5.0 + 7.0 + 3.0, places=9)

    def test_strength_breakdown_prefers_model_components_when_available(self):
        art = SimpleNamespace(model=_FakeModelWithComponents())
        out = strength_breakdown(
            art=art,
            team="Blue",
            as_of="2026-03-07",
            team_meta={"Blue": {"league": "CBLOL", "tier": "T2"}},
            ach_layer={"enabled": True, "team_idx": {}, "weights_points": {"x": 1.0}},
            norm_league_fn=lambda s: str(s or "").strip(),
            tier_from_league_fn=lambda lg: "T1" if lg == "LCK" else "MINOR",
            tier_base_points_fn=lambda tr: 5.0 if tr == "T1" else 1.0,
            league_base_points_fn=lambda lg: 7.0 if lg == "LCK" else 2.0,
            team_boost_points_fn=lambda *_args: 3.0,
            league_hint="LCK",
            tier_hint=None,
        )
        self.assertAlmostEqual(float(out["tier_strength"]), 8.0, places=9)
        self.assertAlmostEqual(float(out["league_strength"]), 12.0, places=9)
        self.assertAlmostEqual(float(out["tier_base"]), 1.0, places=9)
        self.assertAlmostEqual(float(out["league_base"]), 0.7, places=9)
        self.assertAlmostEqual(float(out["tier_base_raw"]), 5.0, places=9)
        self.assertAlmostEqual(float(out["league_base_raw"]), 7.0, places=9)
        self.assertAlmostEqual(float(out["base_strength"]), 31.7, places=9)
        self.assertAlmostEqual(float(out["total_strength"]), 34.7, places=9)

    def test_mlcore_p_map_v2_prefers_model_and_calibrator(self):
        fake_art = SimpleNamespace(
            model=_FakeModel(),
            calibrator=_FakeCalibrator(add=0.01),
            meta={"team_meta": {"Blue": {"league": "LCK", "tier": "T1"}, "Red": {"league": "LPL", "tier": "T1"}}},
            version="v1",
        )
        with TemporaryDirectory() as td:
            art_path = Path(td) / "ml_artifact.json"
            art_path.write_text("{}", encoding="utf-8")
            out = mlcore_p_map_v2(
                artifact_path=str(art_path),
                team_blue="Blue",
                team_red="Red",
                as_of="2026-03-07",
                clip_prob_fn=_clip_prob,
                artifact_sig_fn=lambda p: ("sig", p),
                load_artifact_cached_fn=lambda _sig: fake_art,
                team_meta_from_artifact_fn=team_meta_from_artifact,
                ach_layer_for_artifact_fn=lambda _art, _path, _override: {"enabled": False, "path": "", "weights_points": {}, "team_idx": {}},
                strength_breakdown_fn=lambda **kwargs: strength_breakdown(
                    norm_league_fn=lambda s: str(s or "").strip(),
                    tier_from_league_fn=lambda lg: "T1" if lg in {"LCK", "LPL"} else "MINOR",
                    tier_base_points_fn=lambda _tr: 0.0,
                    league_base_points_fn=lambda _lg: 0.0,
                    team_boost_points_fn=lambda *_args: 0.0,
                    **kwargs,
                ),
                p_from_diff_points_fn=lambda diff, scale_points=400.0: p_from_diff_points(diff, scale_points=scale_points, clip_prob_fn=_clip_prob),
            )
        self.assertAlmostEqual(float(out["p_raw"]), 0.64, places=9)
        self.assertAlmostEqual(float(out["p_cal"]), 0.65, places=9)
        self.assertEqual(str(out["ml_pricing_source"]), "artifact_model_p_blue")

    def test_mlcore_p_map_v2_falls_back_when_model_and_calibrator_fail(self):
        class _NoModel(_FakeModel):
            def p_blue(self, *_args, **_kwargs) -> float:
                raise RuntimeError("boom")

        fake_art = SimpleNamespace(
            model=_NoModel(),
            calibrator=_FakeCalibrator(fail=True),
            meta={"team_meta": {"Blue": {"league": "LCK", "tier": "T1"}, "Red": {"league": "LCK", "tier": "T1"}}},
            version="v2",
        )
        with TemporaryDirectory() as td:
            art_path = Path(td) / "ml_artifact.json"
            art_path.write_text("{}", encoding="utf-8")
            out = mlcore_p_map_v2(
                artifact_path=str(art_path),
                team_blue="Blue",
                team_red="Red",
                as_of="2026-03-07",
                clip_prob_fn=_clip_prob,
                artifact_sig_fn=lambda p: ("sig", p),
                load_artifact_cached_fn=lambda _sig: fake_art,
                team_meta_from_artifact_fn=team_meta_from_artifact,
                ach_layer_for_artifact_fn=lambda _art, _path, _override: {"enabled": False, "path": "", "weights_points": {}, "team_idx": {}},
                strength_breakdown_fn=lambda **kwargs: strength_breakdown(
                    norm_league_fn=lambda s: str(s or "").strip(),
                    tier_from_league_fn=lambda _lg: "T1",
                    tier_base_points_fn=lambda _tr: 0.0,
                    league_base_points_fn=lambda _lg: 0.0,
                    team_boost_points_fn=lambda *_args: 0.0,
                    **kwargs,
                ),
                p_from_diff_points_fn=lambda diff, scale_points=400.0: p_from_diff_points(diff, scale_points=scale_points, clip_prob_fn=_clip_prob),
            )
        self.assertEqual(str(out["ml_pricing_source"]), "reconstructed_diff")
        self.assertEqual(str(out["calibration_error"]), "calibrator_failed_fallback_to_p_raw")


if __name__ == "__main__":
    unittest.main()
