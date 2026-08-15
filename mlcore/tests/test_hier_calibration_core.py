import math
import unittest

from app_core.hier_calibration_core import (
    _node_base_stats,
    _node_n,
    _node_se,
    _node_stats,
    _sh,
    apply_hier_calibration_live,
)


class NodeBaseStatsTests(unittest.TestCase):
    """Base comum extraída (21/07): _node_stats e _pair_stats duplicavam este
    cálculo; agora os dois derivam daqui. Golden provou saída idêntica."""

    _KW = dict(sigma0=4.0, fdr_q_alpha=0.10)

    def test_campos_base_presentes(self):
        base = _node_base_stats({"score_bias": 1.0, "wf_n": 100}, **self._KW)
        self.assertEqual(set(base), {"bias", "n", "se", "var", "z", "p", "q", "fdr_reject"})

    def test_node_stats_contem_a_base(self):
        node = {"score_bias": 2.0, "wf_n": 200, "score_z": 2.5}
        base = _node_base_stats(node, **self._KW)
        full = _node_stats(node, k_shrink=120.0, z_gate=1.96, **self._KW)
        for k, v in base.items():
            self.assertEqual(full[k], v, f"{k}: _node_stats diverge da base")

    def test_no_vazio_nao_quebra(self):
        base = _node_base_stats({}, **self._KW)
        self.assertEqual(base["bias"], 0.0)
        self.assertEqual(base["n"], 0.0)


class HierHelperTests(unittest.TestCase):
    """Helpers extraídos de apply_hier_calibration_live (21/07, era nested;
    hiercal_golden provou saída idêntica). Agora testáveis isolados."""

    def test_node_n_pega_primeiro_positivo(self):
        self.assertEqual(_node_n({"wf_n": 300, "n": 5}), 300.0)  # wf_n primeiro
        self.assertEqual(_node_n({"n": 42}), 42.0)
        self.assertEqual(_node_n({}), 0.0)
        self.assertEqual(_node_n("lixo"), 0.0)

    def test_node_se_direto_ou_derivado(self):
        self.assertEqual(_node_se({"score_se": 0.5}, 4.0), 0.5)           # direto
        self.assertAlmostEqual(_node_se({"score_var": 0.25}, 4.0), 0.5)   # de var
        # sem se/var/n: cai no piso max(0.8, sigma0)
        self.assertEqual(_node_se({}, 4.0), 4.0)
        # com n: sigma0/sqrt(n)
        self.assertAlmostEqual(_node_se({"wf_n": 100}, 4.0), 0.4, places=9)

    def test_sh_shrink_por_amostra(self):
        self.assertAlmostEqual(_sh(100, 100), 0.5, places=9)  # n==k -> 0.5
        self.assertGreater(_sh(400, 100), _sh(100, 100))       # mais n -> menos shrink
        self.assertEqual(_sh(0, 100), 0.0)

    def test_node_stats_significancia_e_peso(self):
        _KW = dict(k_shrink=120.0, sigma0=4.0, z_gate=1.96, fdr_q_alpha=0.10)
        # viés forte, amostra grande, z alto -> aplicado com peso positivo
        forte = _node_stats({"score_bias": 2.0, "wf_n": 300, "score_z": 3.0}, **_KW)
        self.assertGreater(forte["weight"], 0.0)
        self.assertGreater(abs(forte["bias_adj"]), 0.0)
        self.assertTrue(forte["fdr_reject"])
        # z baixo -> sig_mult encolhe o viés ajustado
        fraco = _node_stats({"score_bias": 2.0, "wf_n": 300, "score_z": 0.3}, **_KW)
        self.assertLess(abs(fraco["bias_adj"]), abs(forte["bias_adj"]))

    def test_node_stats_campos_finitos(self):
        st = _node_stats({"score_bias": 1.0, "wf_n": 50}, k_shrink=120.0, sigma0=4.0, z_gate=1.96, fdr_q_alpha=0.10)
        for k in ("bias", "bias_adj", "n", "se", "var", "z", "weight", "sig_mult"):
            self.assertTrue(math.isfinite(st[k]), f"{k} não finito")


class TestHierCalibrationCore(unittest.TestCase):
    def _baseline(self):
        return {
            "global": {"score_bias": 0.5, "wf_n": 800},
            "by_league": {
                "LCK": {"score_bias": 1.0, "wf_n": 300, "quality_band": "ok"},
                "LPL": {"score_bias": -0.6, "wf_n": 340, "quality_band": "ok"},
            },
            "league_class": {"LCK": "major", "LPL": "major"},
            "by_segment": {
                "league:LCK|bo:5|po:1|inter:0": {"score_bias": 2.0, "wf_n": 120, "quality_band": "good"},
            },
            "interleague_pairs": {
                "inter:KR->CN|bo:3|po:0|inter:1": {"score_bias": 1.2, "wf_n": 220, "quality_band": "good"},
                "inter:CN->KR|bo:3|po:0|inter:1": {"score_bias": -1.2, "wf_n": 220, "quality_band": "good"},
                "inter:CN->KR|inter:1": {"score_bias": -0.8, "wf_n": 220, "quality_band": "good"},
            },
            "thresholds": {
                "seg_shrink_k": 120.0,
                "league_shrink_k": 260.0,
                "global_shrink_k": 400.0,
                "hier_decay": 0.8,
                "inter_shrink_k": 320.0,
                "fine_z_hi": 1.20,
                "fine_z_xhi": 1.85,
                "fine_shrink_hi": 0.05,
                "fine_shrink_xhi": 0.09,
                "fine_profiles": {
                    "default": {"fine_z_hi": 1.20, "fine_z_xhi": 1.85, "fine_shrink_hi": 0.05, "fine_shrink_xhi": 0.09},
                    "same_league": {"fine_z_hi": 1.25, "fine_z_xhi": 1.95, "fine_shrink_hi": 0.045, "fine_shrink_xhi": 0.08},
                    "interregional": {"fine_z_hi": 1.10, "fine_z_xhi": 1.70, "fine_shrink_hi": 0.07, "fine_shrink_xhi": 0.12},
                },
                "fine_profile_by_league_same": {},
                "fine_profile_by_league_inter": {},
            },
        }

    def test_segment_scope_preferred_when_available(self):
        out = apply_hier_calibration_live(
            p_map_in=0.60,
            league="LCK",
            bo=5,
            playoffs=True,
            interregional=False,
            n_team_eff=50,
            n_players_eff=30,
            patch_status="ok",
            baseline_payload=self._baseline(),
        )
        self.assertIn(str(out.get("method")), ("hier_calibration_v3",))
        self.assertIn("league:LCK|bo:5|po:1|inter:0", str(out.get("scope_used")))
        self.assertTrue(isinstance(out.get("hier"), dict))
        self.assertTrue(isinstance((out.get("hier") or {}).get("weights"), dict))

    def test_fallback_to_league_when_segment_missing(self):
        b = self._baseline()
        b["by_segment"] = {}
        out = apply_hier_calibration_live(
            p_map_in=0.57,
            league="LCK",
            bo=3,
            playoffs=False,
            interregional=False,
            n_team_eff=40,
            n_players_eff=20,
            baseline_payload=b,
        )
        self.assertEqual(str(out.get("scope_used")), "league:LCK")

    def test_fallback_to_global_when_league_missing(self):
        b = self._baseline()
        b["by_league"] = {}
        b["by_segment"] = {}
        out = apply_hier_calibration_live(
            p_map_in=0.55,
            league="UNKNOWN",
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=30,
            n_players_eff=10,
            baseline_payload=b,
        )
        self.assertEqual(str(out.get("scope_used")), "global")
        self.assertIn(str(out.get("reason")), ("ok", "tiny_move", "zero_or_missing_bias"))

    def test_interregional_uses_league_pair_context(self):
        out = apply_hier_calibration_live(
            p_map_in=0.62,
            league="INTER",
            league_a="LCK",
            league_b="LPL",
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=42,
            n_players_eff=22,
            patch_status="ok",
            baseline_payload=self._baseline(),
        )
        inter = out.get("interleague") if isinstance(out.get("interleague"), dict) else {}
        self.assertEqual(str(out.get("league_a")), "LCK")
        self.assertEqual(str(out.get("league_b")), "LPL")
        self.assertEqual(str(inter.get("macro_a")), "KR")
        self.assertEqual(str(inter.get("macro_b")), "CN")
        self.assertIn(str(inter.get("reliability_band")), {"low", "mid", "high"})
        self.assertTrue(float(inter.get("dir_cap", 0.0)) > 0.0)
        self.assertIn(str(out.get("reason")), ("ok", "tiny_move", "zero_or_missing_bias"))

    def test_runtime_fine_present_on_extreme_zone(self):
        out = apply_hier_calibration_live(
            p_map_in=0.92,
            league="LCK",
            league_a="LCK",
            league_b="LCK",
            bo=1,
            playoffs=False,
            interregional=False,
            n_team_eff=4,
            n_players_eff=0,
            patch_status="warn",
            baseline_payload=self._baseline(),
        )
        fine = out.get("runtime_fine") if isinstance(out.get("runtime_fine"), dict) else {}
        self.assertIn(str(fine.get("band")), ("high", "extreme"))
        self.assertTrue(float(fine.get("fine_shrink", 0.0)) > 0.0)
        self.assertIn(str(fine.get("profile_used")), {"default", "same_league", "interregional"})

    def test_interregional_prefers_league_pair_signal_when_available(self):
        b = self._baseline()
        b["interleague_pairs"]["league_pair:LCK->LPL|bo:3|po:0|inter:1"] = {
            "score_bias": 3.5,
            "wf_n": 900,
            "quality_band": "strong",
        }
        b["interleague_pairs"]["inter:KR->CN|bo:3|po:0|inter:1"] = {
            "score_bias": -0.2,
            "wf_n": 1200,
            "quality_band": "strong",
        }
        out = apply_hier_calibration_live(
            p_map_in=0.61,
            league="INTER",
            league_a="LCK",
            league_b="LPL",
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=40,
            n_players_eff=18,
            patch_status="ok",
            baseline_payload=b,
        )
        inter = out.get("interleague") if isinstance(out.get("interleague"), dict) else {}
        pair_scope = str(inter.get("pair_scope") or "")
        self.assertIn("league_pair:", pair_scope)
        self.assertTrue(float(inter.get("pair_bias", 0.0)) > 0.0)
        self.assertIn(str(inter.get("pair_scope_kind")), {"league_pair", "blend"})
        self.assertTrue(float(inter.get("league_pair_support_ev", 0.0)) > 0.0)

    def test_runtime_fine_profile_override_for_interregional(self):
        b = self._baseline()
        b["thresholds"]["fine_profiles"] = {
            "interregional": {
                "fine_z_hi": 1.05,
                "fine_z_xhi": 1.50,
                "fine_shrink_hi": 0.08,
                "fine_shrink_xhi": 0.12,
            }
        }
        out = apply_hier_calibration_live(
            p_map_in=0.84,
            league="INTER",
            league_a="LCK",
            league_b="LPL",
            bo=1,
            playoffs=False,
            interregional=True,
            n_team_eff=8,
            n_players_eff=2,
            patch_status="warn",
            baseline_payload=b,
        )
        fine = out.get("runtime_fine") if isinstance(out.get("runtime_fine"), dict) else {}
        self.assertEqual(str(fine.get("profile_used")), "interregional")
        self.assertTrue(float(fine.get("z_hi", 9.9)) <= 1.05 + 1e-9)

    def test_runtime_fine_context_specific_profile_maps(self):
        b = self._baseline()
        b["thresholds"]["fine_profiles"]["major_stable"] = {
            "fine_z_hi": 1.35,
            "fine_z_xhi": 2.10,
            "fine_shrink_hi": 0.03,
            "fine_shrink_xhi": 0.06,
        }
        b["thresholds"]["fine_profiles"]["strict_low_sample"] = {
            "fine_z_hi": 0.95,
            "fine_z_xhi": 1.45,
            "fine_shrink_hi": 0.09,
            "fine_shrink_xhi": 0.14,
        }
        b["thresholds"]["fine_profile_by_league_same"] = {"LCK": "major_stable"}
        b["thresholds"]["fine_profile_by_league_inter"] = {"LCK": "strict_low_sample"}

        same = apply_hier_calibration_live(
            p_map_in=0.82,
            league="LCK",
            league_a="LCK",
            league_b="LCK",
            bo=3,
            playoffs=False,
            interregional=False,
            n_team_eff=28,
            n_players_eff=8,
            patch_status="ok",
            baseline_payload=b,
        )
        inter = apply_hier_calibration_live(
            p_map_in=0.82,
            league="INTER",
            league_a="LCK",
            league_b="LPL",
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=28,
            n_players_eff=8,
            patch_status="ok",
            baseline_payload=b,
        )
        f_same = same.get("runtime_fine") if isinstance(same.get("runtime_fine"), dict) else {}
        f_inter = inter.get("runtime_fine") if isinstance(inter.get("runtime_fine"), dict) else {}
        self.assertEqual(str(f_same.get("profile_used")), "major_stable")
        self.assertEqual(str(f_inter.get("profile_used")), "strict_low_sample")
        self.assertEqual(str(f_same.get("profile_source")), "league_override")
        self.assertEqual(str(f_inter.get("profile_source")), "league_override")

    def test_runtime_fine_inter_reliability_low_is_stricter(self):
        b_low = self._baseline()
        b_low["interleague_pairs"] = {}
        b_high = self._baseline()
        # Force very high pair evidence in one key.
        b_high["interleague_pairs"]["inter:KR->CN|bo:3|po:0|inter:1"] = {
            "score_bias": 0.8,
            "wf_n": 3200,
            "quality_band": "strong",
        }

        out_low = apply_hier_calibration_live(
            p_map_in=0.88,
            league="INTER",
            league_a="LCK",
            league_b="LPL",
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=12,
            n_players_eff=3,
            patch_status="warn",
            baseline_payload=b_low,
        )
        out_high = apply_hier_calibration_live(
            p_map_in=0.88,
            league="INTER",
            league_a="LCK",
            league_b="LPL",
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=12,
            n_players_eff=3,
            patch_status="warn",
            baseline_payload=b_high,
        )
        inter_low = out_low.get("interleague") if isinstance(out_low.get("interleague"), dict) else {}
        inter_high = out_high.get("interleague") if isinstance(out_high.get("interleague"), dict) else {}
        self.assertTrue(float(inter_low.get("inter_shrink", 0.0)) >= float(inter_high.get("inter_shrink", 0.0)))
        self.assertTrue(abs(float(inter_low.get("inter_dir", 0.0))) <= abs(float(inter_high.get("inter_dir", 0.0))))
        self.assertTrue(float(inter_low.get("significance_mult", 0.0)) <= float(inter_high.get("significance_mult", 0.0)))
        self.assertTrue(float(inter_low.get("pair_support_ev", 0.0)) <= float(inter_high.get("pair_support_ev", 0.0)))
        f_low = out_low.get("runtime_fine") if isinstance(out_low.get("runtime_fine"), dict) else {}
        f_high = out_high.get("runtime_fine") if isinstance(out_high.get("runtime_fine"), dict) else {}
        self.assertIn(str(f_low.get("inter_reliability_band")), {"na", "low", "mid", "high"})
        self.assertIn(str(f_high.get("inter_reliability_band")), {"na", "low", "mid", "high"})

    def test_interregional_pair_specific_support_reduces_shrink_vs_macro_only(self):
        b_macro = self._baseline()
        b_macro["interleague_pairs"] = {
            "inter:KR->CN|bo:3|po:0|inter:1": {
                "score_bias": 1.0,
                "score_se": 0.35,
                "wf_n": 1600,
                "quality_band": "strong",
            }
        }
        b_pair = self._baseline()
        b_pair["interleague_pairs"] = dict(b_macro["interleague_pairs"])
        b_pair["interleague_pairs"]["league_pair:LCK->LPL|bo:3|po:0|inter:1"] = {
            "score_bias": 1.4,
            "score_se": 0.30,
            "wf_n": 480,
            "quality_band": "good",
        }

        out_macro = apply_hier_calibration_live(
            p_map_in=0.60,
            league="INTER",
            league_a="LCK",
            league_b="LPL",
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=24,
            n_players_eff=10,
            patch_status="ok",
            baseline_payload=b_macro,
        )
        out_pair = apply_hier_calibration_live(
            p_map_in=0.60,
            league="INTER",
            league_a="LCK",
            league_b="LPL",
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=24,
            n_players_eff=10,
            patch_status="ok",
            baseline_payload=b_pair,
        )
        inter_macro = out_macro.get("interleague") if isinstance(out_macro.get("interleague"), dict) else {}
        inter_pair = out_pair.get("interleague") if isinstance(out_pair.get("interleague"), dict) else {}
        self.assertEqual(str(inter_macro.get("pair_scope_kind")), "macro_pair")
        self.assertIn(str(inter_pair.get("pair_scope_kind")), {"league_pair", "blend"})
        self.assertTrue(float(inter_pair.get("inter_ev", 0.0)) >= float(inter_macro.get("inter_ev", 0.0)))
        self.assertTrue(float(inter_pair.get("inter_shrink", 1.0)) <= float(inter_macro.get("inter_shrink", 1.0)))
        self.assertTrue(abs(float(inter_pair.get("inter_dir", 0.0))) >= abs(float(inter_macro.get("inter_dir", 0.0))))

    def test_interregional_significance_gate_shrinks_weak_pair_bias(self):
        b = self._baseline()
        b["interleague_pairs"] = {
            "league_pair:LCK->LPL|bo:3|po:0|inter:1": {
                "score_bias": 2.0,
                "score_se": 5.0,  # weak signal (|z|=0.4)
                "wf_n": 80,
                "quality_band": "low_sample",
            }
        }
        out = apply_hier_calibration_live(
            p_map_in=0.64,
            league="INTER",
            league_a="LCK",
            league_b="LPL",
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=20,
            n_players_eff=8,
            patch_status="ok",
            baseline_payload=b,
        )
        inter = out.get("interleague") if isinstance(out.get("interleague"), dict) else {}
        self.assertTrue(float(inter.get("significance_mult", 1.0)) < 1.0)
        self.assertTrue(abs(float(inter.get("pair_bias", 0.0))) < abs(float(inter.get("pair_bias_raw", 0.0))))

    def test_interregional_inverse_variance_blend_prefers_low_se_signal(self):
        b = self._baseline()
        b["interleague_pairs"]["league_pair:LCK->LPL|bo:3|po:0|inter:1"] = {
            "score_bias": 2.0,
            "score_se": 0.30,
            "wf_n": 220,
            "quality_band": "good",
        }
        b["interleague_pairs"]["inter:KR->CN|bo:3|po:0|inter:1"] = {
            "score_bias": -2.0,
            "score_se": 2.00,
            "wf_n": 2000,
            "quality_band": "strong",
        }
        out = apply_hier_calibration_live(
            p_map_in=0.61,
            league="INTER",
            league_a="LCK",
            league_b="LPL",
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=26,
            n_players_eff=10,
            patch_status="ok",
            baseline_payload=b,
        )
        inter = out.get("interleague") if isinstance(out.get("interleague"), dict) else {}
        self.assertIn("blend_ivw:", str(inter.get("pair_scope") or ""))
        self.assertTrue(float(inter.get("pair_bias_raw", 0.0)) > 0.5)

    def test_interregional_q_gate_dampens_non_rejected_signal(self):
        b = self._baseline()
        b["interleague_pairs"]["inter:KR->CN|bo:3|po:0|inter:1"] = {
            "score_bias": 2.4,
            "score_se": 0.50,
            "score_q": 0.40,
            "fdr_reject": False,
            "wf_n": 2200,
            "quality_band": "strong",
        }
        out = apply_hier_calibration_live(
            p_map_in=0.63,
            league="INTER",
            league_a="LCK",
            league_b="LPL",
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=30,
            n_players_eff=12,
            patch_status="ok",
            baseline_payload=b,
        )
        inter = out.get("interleague") if isinstance(out.get("interleague"), dict) else {}
        self.assertTrue(float(inter.get("q_mult", 1.0)) < 1.0)
        self.assertTrue(abs(float(inter.get("pair_bias", 0.0))) < abs(float(inter.get("pair_bias_raw", 0.0))))

    def test_interregional_canonicalizes_league_aliases(self):
        b = self._baseline()
        b["interleague_pairs"] = {
            "league_pair:LCKC->LDL|bo:3|po:0|inter:1": {
                "score_bias": 1.3,
                "score_se": 0.40,
                "wf_n": 800,
                "quality_band": "good",
            }
        }
        out = apply_hier_calibration_live(
            p_map_in=0.58,
            league="INTER",
            league_a="LCK CL",
            league_b="LPL CL",
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=24,
            n_players_eff=9,
            patch_status="ok",
            baseline_payload=b,
        )
        inter = out.get("interleague") if isinstance(out.get("interleague"), dict) else {}
        self.assertEqual(str(out.get("league_a")), "LCKC")
        self.assertEqual(str(out.get("league_b")), "AL")
        self.assertIn("LCKC->LDL", str(inter.get("pair_scope") or ""))

    def test_same_league_q_gate_dampens_bias(self):
        b = self._baseline()
        b["by_segment"] = {
            "league:LCK|bo:3|po:0|inter:0": {
                "score_bias": 4.0,
                "score_se": 0.8,
                "score_q": 0.35,
                "fdr_reject": False,
                "wf_n": 500,
                "quality_band": "good",
            }
        }
        out = apply_hier_calibration_live(
            p_map_in=0.60,
            league="LCK",
            league_a="LCK",
            league_b="LCK",
            bo=3,
            playoffs=False,
            interregional=False,
            n_team_eff=30,
            n_players_eff=12,
            patch_status="ok",
            baseline_payload=b,
        )
        hier = out.get("hier") if isinstance(out.get("hier"), dict) else {}
        st = hier.get("stats") if isinstance(hier.get("stats"), dict) else {}
        self.assertTrue(float(st.get("segment_q", 0.0)) > 0.10)
        self.assertTrue(float(st.get("segment_sig_mult", 1.0)) < 1.0)

    def test_same_league_canonical_alias_scope(self):
        b = self._baseline()
        b["by_league"]["LCKC"] = {"score_bias": 1.4, "wf_n": 450, "quality_band": "good"}
        out = apply_hier_calibration_live(
            p_map_in=0.56,
            league="LCK CL",
            league_a="LCK CL",
            league_b="LCK CL",
            bo=3,
            playoffs=False,
            interregional=False,
            n_team_eff=20,
            n_players_eff=9,
            patch_status="ok",
            baseline_payload=b,
        )
        self.assertEqual(str(out.get("league_a")), "LCKC")
        self.assertEqual(str(out.get("scope_used")), "league:LCKC")

    def test_prefers_class_patch_segment_when_available(self):
        b = self._baseline()
        b["by_segment"] = {
            "class:same_major|patch:warn|bo:3|po:0|inter:0": {
                "score_bias": 2.2,
                "wf_n": 540,
                "quality_band": "good",
            }
        }
        b["by_league"]["LCK"] = {"score_bias": 0.9, "wf_n": 280, "quality_band": "ok"}
        b["league_class"] = {"LCK": "major"}
        out = apply_hier_calibration_live(
            p_map_in=0.61,
            league="LCK",
            league_a="LCK",
            league_b="LCK",
            bo=3,
            playoffs=False,
            interregional=False,
            n_team_eff=30,
            n_players_eff=10,
            patch_status="warn",
            baseline_payload=b,
        )
        self.assertEqual(str(out.get("scope_used")), "class:same_major|patch:warn|bo:3|po:0|inter:0")
        hier = out.get("hier") if isinstance(out.get("hier"), dict) else {}
        self.assertEqual(str(hier.get("segment_class")), "same_major")
        self.assertEqual(str(hier.get("patch_tag")), "warn")

    def test_prefers_inter_class_patch_segment_when_available(self):
        b = self._baseline()
        b["by_league"]["TCL"] = {"score_bias": 0.2, "wf_n": 90, "quality_band": "low_sample"}
        b["league_class"] = {"LCK": "major", "TCL": "minor"}
        b["by_segment"] = {
            "class:inter_mixed|patch:refit|bo:5|po:0|inter:1": {
                "score_bias": -1.6,
                "wf_n": 210,
                "quality_band": "ok",
            }
        }
        out = apply_hier_calibration_live(
            p_map_in=0.64,
            league="INTER",
            league_a="LCK",
            league_b="TCL",
            bo=5,
            playoffs=False,
            interregional=True,
            n_team_eff=20,
            n_players_eff=7,
            patch_status="refit",
            baseline_payload=b,
        )
        self.assertEqual(str(out.get("scope_used")), "class:inter_mixed|patch:refit|bo:5|po:0|inter:1")
        hier = out.get("hier") if isinstance(out.get("hier"), dict) else {}
        self.assertEqual(str(hier.get("segment_class")), "inter_mixed")
        self.assertEqual(str(hier.get("patch_tag")), "refit")

    def test_global_patch_segment_fallback(self):
        b = self._baseline()
        b["by_league"] = {}
        b["league_class"] = {}
        b["by_segment"] = {
            "global|patch:warn|bo:1|po:0|inter:0": {
                "score_bias": 1.1,
                "wf_n": 600,
                "quality_band": "good",
            }
        }
        out = apply_hier_calibration_live(
            p_map_in=0.58,
            league="UNKNOWN",
            league_a="UNKNOWN",
            league_b="UNKNOWN",
            bo=1,
            playoffs=False,
            interregional=False,
            n_team_eff=12,
            n_players_eff=4,
            patch_status="WARN",
            baseline_payload=b,
        )
        self.assertEqual(str(out.get("scope_used")), "global|patch:warn|bo:1|po:0|inter:0")

    def test_interregional_pair_reliability_relaxes_runtime_shrink(self):
        b_low = self._baseline()
        b_high = self._baseline()
        b_low["interleague_pairs"]["league_pair:LCK->LPL|bo:3|po:0|inter:1"] = {
            "score_bias": 2.4,
            "wf_n": 900,
            "quality_band": "strong",
            "pair_reliability": 0.25,
            "class_relation": "major_major",
        }
        b_high["interleague_pairs"]["league_pair:LCK->LPL|bo:3|po:0|inter:1"] = {
            "score_bias": 2.4,
            "wf_n": 900,
            "quality_band": "strong",
            "pair_reliability": 0.92,
            "class_relation": "major_major",
        }
        out_low = apply_hier_calibration_live(
            p_map_in=0.61,
            league="INTER",
            league_a="LCK",
            league_b="LPL",
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=40,
            n_players_eff=18,
            patch_status="ok",
            baseline_payload=b_low,
        )
        out_high = apply_hier_calibration_live(
            p_map_in=0.61,
            league="INTER",
            league_a="LCK",
            league_b="LPL",
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=40,
            n_players_eff=18,
            patch_status="ok",
            baseline_payload=b_high,
        )
        inter_low = out_low.get("interleague") if isinstance(out_low.get("interleague"), dict) else {}
        inter_high = out_high.get("interleague") if isinstance(out_high.get("interleague"), dict) else {}
        self.assertTrue(float(inter_high.get("pair_reliability", 0.0)) > float(inter_low.get("pair_reliability", 0.0)))
        self.assertTrue(float(inter_high.get("inter_shrink", 1.0)) < float(inter_low.get("inter_shrink", 1.0)))
        self.assertTrue(float(inter_high.get("dir_cap", 0.0)) > float(inter_low.get("dir_cap", 0.0)))

    def test_interregional_uses_tier_pair_when_league_pair_missing(self):
        b = self._baseline()
        b["interleague_pairs"] = {
            "tier_pair:T1->T2|bo:3|po:0|inter:1": {
                "score_bias": 1.6,
                "wf_n": 700,
                "quality_band": "good",
                "pair_reliability": 0.78,
                "class_relation": "major_major",
                "strength_gap": 0.18,
                "strength_band_a": "major",
                "strength_band_b": "upper",
            },
            "inter:EMEA->BR|bo:3|po:0|inter:1": {
                "score_bias": 0.2,
                "wf_n": 1500,
                "quality_band": "strong",
                "pair_reliability": 0.40,
                "class_relation": "major_major",
            },
        }
        b["league_class"] = {"LEC": "major", "CBLOL": "major"}
        out = apply_hier_calibration_live(
            p_map_in=0.59,
            league="INTER",
            league_a="LEC",
            league_b="CBLOL",
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=26,
            n_players_eff=11,
            patch_status="ok",
            baseline_payload=b,
        )
        inter = out.get("interleague") if isinstance(out.get("interleague"), dict) else {}
        self.assertIn(str(inter.get("pair_scope_kind")), {"tier_pair", "blend"})
        self.assertTrue(float(inter.get("tier_pair_support_ev", 0.0)) > 0.0)
        self.assertTrue(float(inter.get("pair_bias", 0.0)) > 0.0)
        self.assertTrue(float(inter.get("strength_gap", 0.0)) > 0.0)
        self.assertEqual(str(inter.get("strength_band_a")), "major")
        self.assertEqual(str(inter.get("strength_band_b")), "upper")

    def test_strong_cross_strength_caps_adverse_move_against_stronger_side(self):
        b = self._baseline()
        b["interleague_pairs"] = {
            "league_pair:LCK->CBLOL|bo:5|po:0|inter:1": {
                "score_bias": -4.2,
                "wf_n": 900,
                "quality_band": "good",
                "pair_reliability": 0.92,
                "class_relation": "major_major",
                "strength_gap": 0.20,
                "strength_band_a": "elite",
                "strength_band_b": "regional",
            }
        }
        out = apply_hier_calibration_live(
            p_map_in=0.84,
            league="INTER",
            league_a="LCK",
            league_b="CBLOL",
            bo=5,
            playoffs=False,
            interregional=True,
            n_team_eff=32,
            n_players_eff=12,
            patch_status="ok",
            baseline_payload=b,
        )
        self.assertTrue(bool(out.get("applied", False)))
        adverse_pp = (0.84 - float(out.get("p_after"))) * 100.0
        self.assertLessEqual(adverse_pp, 1.000001)

    def test_interregional_transfer_features_adjust_runtime(self):
        payload = {
            "by_league": {
                "LCK": {
                    "league_reliability": 0.84,
                    "league_depth": 0.76,
                    "league_translation_score": 0.66,
                    "league_domestic_inflation": 0.21,
                },
                "LPL": {
                    "league_reliability": 0.71,
                    "league_depth": 0.72,
                    "league_translation_score": 0.58,
                    "league_domestic_inflation": 0.31,
                },
            },
            "by_team": {
                "t1": {"entity_key": "t1", "team_domestic_power": 0.84, "team_transfer_power": 0.80, "sample_games": 24},
                "blg": {"entity_key": "blg", "team_domestic_power": 0.79, "team_transfer_power": 0.70, "sample_games": 24},
            },
            "team_alias_index": {"t1": "t1", "blg": "blg", "bilibiligaming": "blg"},
        }
        base = apply_hier_calibration_live(
            p_map_in=0.60,
            league="INTER",
            league_a="LCK",
            league_b="LPL",
            team_a="T1",
            team_b="Bilibili Gaming",
            team_aliases_a=["T1"],
            team_aliases_b=["BLG"],
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=24,
            n_players_eff=10,
            patch_status="ok",
            baseline_payload=self._baseline(),
        )
        boosted = apply_hier_calibration_live(
            p_map_in=0.60,
            league="INTER",
            league_a="LCK",
            league_b="LPL",
            team_a="T1",
            team_b="Bilibili Gaming",
            team_aliases_a=["T1"],
            team_aliases_b=["BLG"],
            bo=3,
            playoffs=False,
            interregional=True,
            n_team_eff=24,
            n_players_eff=10,
            patch_status="ok",
            baseline_payload=self._baseline(),
            league_transfer_features=payload,
        )
        inter_base = base.get("interleague") if isinstance(base.get("interleague"), dict) else {}
        inter_boost = boosted.get("interleague") if isinstance(boosted.get("interleague"), dict) else {}
        tf = inter_boost.get("transfer_features") if isinstance(inter_boost.get("transfer_features"), dict) else {}
        self.assertTrue(bool(tf.get("applied", False)))
        self.assertTrue(float(inter_boost.get("inter_shrink", 1.0)) <= float(inter_base.get("inter_shrink", 1.0)))
        self.assertTrue(float(inter_boost.get("inter_dir", 0.0)) >= float(inter_base.get("inter_dir", 0.0)))


if __name__ == "__main__":
    unittest.main()
