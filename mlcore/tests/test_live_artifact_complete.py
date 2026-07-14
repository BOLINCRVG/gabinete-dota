"""Guard: the deployed ml_artifact.json must be STRUCTURALLY COMPLETE.

The regression fixtures pin exact predictions to the artifact's sha256, but if a
rebuild silently drops a piece (e.g. forgetting --fit-players removes the players
layer) and someone regenerates the fixtures on that broken state, the regression
tests would happily pin the regression. This test is independent of the fixtures:
it asserts the LIVE artifact has all the parts the fused product/nota depend on,
so an incomplete deploy fails loudly instead of silently changing the odds.

Skips when ml_artifact.json is absent (e.g. fresh checkout / CI without data),
mirroring the regression tests.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from mlcore.artifact import MLArtifact

ROOT = Path(__file__).resolve().parents[2]


class TestLiveArtifactComplete(unittest.TestCase):

    def setUp(self):
        self.art_path = ROOT / "ml_artifact.json"
        if not self.art_path.exists():
            self.skipTest("ml_artifact.json not present.")
        self.art = MLArtifact.load_json(str(self.art_path))

    def test_elo_params_present(self):
        p = self.art.model.params
        core = str((self.art.meta or {}).get("core") or "elo")
        # Core knobs the validated structure relies on (presence/sanity, not exact
        # values — those legitimately move when retuned).
        self.assertGreater(float(p.k), 0.0)
        self.assertGreater(float(p.scale_points), 0.0)
        if core == "whr":
            # WHR (alavanca N, deploy 2026-07-07): a estrutura VALIDADA é
            # alpha=0 (sem shrink — o MAP já pondera), priors estáticos 0
            # (prior de liga vive DENTRO do rating, lpw=0 = imune a hints),
            # dynk/MOVDA inertes e scale 400 (protocolo do gate).
            self.assertEqual(float(p.alpha), 0.0, "core=whr exige alpha=0 (sem shrink)")
            self.assertEqual(float(p.league_prior_weight), 0.0, "core=whr exige lpw=0 (hint-proof)")
            self.assertEqual(float(p.scale_points), 400.0, "core=whr exige scale=400")
            self.assertFalse(bool(p.dynamic_k_enabled), "core=whr exige dynk off")
            self.assertEqual(float(getattr(p, "mov_lambda", 0.0) or 0.0), 0.0, "core=whr exige MOVDA off")
        else:
            self.assertGreater(float(p.alpha), 0.0)
            self.assertIsInstance(bool(p.dynamic_k_enabled), bool)
        self.assertGreater(float(p.league_strength_cap_abs), 0.0)

    def test_sos_league_map_baked(self):
        # SOS (lever A) bakes data-derived league base points into the model.
        self.assertTrue(
            self.art.model.league_base_points_map,
            "league_base_points_map vazio — SOS nao foi bakeado (rebuild sem SOS?).",
        )

    def test_calibrator_present(self):
        self.assertIsNotNone(self.art.calibrator, "Sem calibrador Platt no artefato.")

    def test_players_layer_present(self):
        # The fused product (the odds the user bets) needs the players layer.
        # A rebuild without --fit-players drops this and silently changes fusion.
        meta = dict(self.art.meta or {})
        pl = dict(meta.get("players_layer") or {})
        self.assertTrue(
            pl, "meta.players_layer ausente — rebuild SEM --fit-players? (regressao silenciosa na fusao)"
        )
        self.assertIn("w_lineup", pl, "players_layer sem w_lineup fitado.")
        self.assertTrue(
            (ROOT / "players_artifact.json").exists(),
            "players_artifact.json ausente, mas o artefato referencia players_layer.",
        )


if __name__ == "__main__":
    unittest.main()
