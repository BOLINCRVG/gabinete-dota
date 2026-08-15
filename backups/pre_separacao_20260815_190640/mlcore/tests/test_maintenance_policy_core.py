import unittest

from app_core.maintenance_policy_core import (
    adjust_needs_after_incremental,
    decide_maintenance_policy,
    force_params_after_model_refresh,
    resolve_maintenance_preferences,
    seed_maintenance_needs,
)


class TestMaintenancePolicyCore(unittest.TestCase):
    def test_resolve_maintenance_preferences(self):
        prefs = resolve_maintenance_preferences({"auto_maint_fast_mode": False, "auto_maint_cmd_timeout_s": 900}, force_full=True)
        self.assertEqual(prefs["cmd_timeout_s"], 900)
        self.assertEqual(prefs["params_timeout_s"], 14000)
        self.assertFalse(prefs["fast_mode_user_pref"])
        # force_full (botão) NÃO força mais o profile full do params (só o modelo).
        self.assertFalse(prefs["policy_force_full"])
        self.assertFalse(prefs["policy_force_fast"])

    def test_params_full_profile_only_from_explicit_setting(self):
        # O deep-fit de params (profile full) vem só do setting auto_maint_force_full.
        on = resolve_maintenance_preferences({"auto_maint_force_full": True}, force_full=False)
        self.assertTrue(on["policy_force_full"])
        off = resolve_maintenance_preferences({}, force_full=True)
        self.assertFalse(off["policy_force_full"])

    def test_resolve_maintenance_preferences_respects_params_timeout_override(self):
        prefs = resolve_maintenance_preferences(
            {
                "auto_maint_fast_mode": True,
                "auto_maint_cmd_timeout_s": 900,
                "auto_maint_params_timeout_s": 2400,
                "auto_maint_force_fast": True,
            },
            force_full=False,
        )
        self.assertEqual(prefs["cmd_timeout_s"], 900)
        self.assertEqual(prefs["params_timeout_s"], 2400)
        self.assertTrue(prefs["policy_force_fast"])

    def test_seed_maintenance_needs(self):
        params, wf = seed_maintenance_needs(False, False, False)
        self.assertTrue(params)
        self.assertTrue(wf)
        params2, wf2 = seed_maintenance_needs(True, False, False)
        self.assertTrue(params2)
        self.assertTrue(wf2)

    def test_adjust_needs_after_incremental(self):
        out = adjust_needs_after_incremental(
            model_ok=True,
            new_games_inc=0,
            need_patch_recalibration=False,
            need_retrain_params=True,
            need_walkforward=True,
        )
        self.assertFalse(out["need_retrain_params"])
        self.assertFalse(out["need_walkforward"])
        self.assertIn("incremental sem novos jogos", out["log_message"])

    def test_force_params_after_model_refresh(self):
        out = force_params_after_model_refresh(
            need_retrain_model=True,
            need_retrain_params=False,
            model_updated_this_cycle=True,
        )
        self.assertTrue(out["need_retrain_params"])
        self.assertTrue(out["forced"])

    def test_decide_maintenance_policy(self):
        out = decide_maintenance_policy(
            policy_force_full=False,
            policy_force_fast=False,
            need_retrain_model=False,
            need_retrain_params=True,
            need_walkforward=False,
            need_patch_recalibration=False,
            model_backfill_suspected=False,
            model_had_full_retrain=False,
            model_new_games=5,
            fast_new_games_max=30,
            fast_mode_user_pref=True,
        )
        self.assertEqual(out["profile"], "fast")
        self.assertEqual(out["reason"], "params_only_fast")

        out_backfill = decide_maintenance_policy(
            policy_force_full=False,
            policy_force_fast=False,
            need_retrain_model=False,
            need_retrain_params=True,
            need_walkforward=False,
            need_patch_recalibration=False,
            model_backfill_suspected=True,
            model_had_full_retrain=False,
            model_new_games=32,
            fast_new_games_max=30,
            fast_mode_user_pref=True,
        )
        self.assertEqual(out_backfill["profile"], "fast")
        self.assertEqual(out_backfill["reason"], "params_only_fast")

        out2 = decide_maintenance_policy(
            policy_force_full=False,
            policy_force_fast=False,
            need_retrain_model=True,
            need_retrain_params=True,
            need_walkforward=True,
            need_patch_recalibration=False,
            model_backfill_suspected=False,
            model_had_full_retrain=False,
            model_new_games=100,
            fast_new_games_max=30,
            fast_mode_user_pref=True,
        )
        self.assertEqual(out2["profile"], "full")
        self.assertIn("new_games_gt_fast_max", out2["reason"])

    def test_decide_maintenance_policy_force_fast(self):
        out = decide_maintenance_policy(
            policy_force_full=False,
            policy_force_fast=True,
            need_retrain_model=True,
            need_retrain_params=True,
            need_walkforward=True,
            need_patch_recalibration=True,
            model_backfill_suspected=False,
            model_had_full_retrain=False,
            model_new_games=100,
            fast_new_games_max=30,
            fast_mode_user_pref=True,
        )
        self.assertEqual(out["profile"], "fast")
        self.assertEqual(out["reason"], "force_fast_flag")

    def test_decide_maintenance_policy_force_full_wins_over_force_fast(self):
        out = decide_maintenance_policy(
            policy_force_full=True,
            policy_force_fast=True,
            need_retrain_model=False,
            need_retrain_params=True,
            need_walkforward=False,
            need_patch_recalibration=False,
            model_backfill_suspected=False,
            model_had_full_retrain=False,
            model_new_games=1,
            fast_new_games_max=30,
            fast_mode_user_pref=True,
        )
        self.assertEqual(out["profile"], "full")
        self.assertEqual(out["reason"], "force_full_flag")


if __name__ == "__main__":
    unittest.main()
