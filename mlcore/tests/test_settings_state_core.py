import unittest

from app_core.settings_state_core import get_settings_sidebar_state, queue_settings_sidebar_action


class TestGetSettingsSidebarState(unittest.TestCase):

    def test_defaults_empty_state(self):
        snapshot = get_settings_sidebar_state({})
        self.assertEqual(snapshot["cfg_profile"], "Padrão")
        self.assertTrue(snapshot["ui_fast_flow"])

    def test_reads_cfg_profile_from_state(self):
        snapshot = get_settings_sidebar_state({"cfg_profile": "Avançado"})
        self.assertEqual(snapshot["cfg_profile"], "Avançado")

    def test_reads_ui_fast_flow_false(self):
        snapshot = get_settings_sidebar_state({"ui_fast_flow": False})
        self.assertFalse(snapshot["ui_fast_flow"])

    def test_none_cfg_profile_falls_back_to_default(self):
        snapshot = get_settings_sidebar_state({"cfg_profile": None})
        self.assertEqual(snapshot["cfg_profile"], "Padrão")

    def test_returns_dict(self):
        self.assertIsInstance(get_settings_sidebar_state({}), dict)

    def test_required_keys_present(self):
        snapshot = get_settings_sidebar_state({})
        self.assertIn("cfg_profile", snapshot)
        self.assertIn("ui_fast_flow", snapshot)


class TestQueueSettingsSidebarAction(unittest.TestCase):

    def test_sets_key_to_true_by_default(self):
        state = {}
        queue_settings_sidebar_action(state, "_pending_reload_settings")
        self.assertTrue(state["_pending_reload_settings"])

    def test_sets_key_to_custom_value(self):
        state = {}
        queue_settings_sidebar_action(state, "_pending_apply_preset", "Avançado")
        self.assertEqual(state["_pending_apply_preset"], "Avançado")

    def test_overwrites_existing_key(self):
        state = {"_pending_apply_preset": "Padrão"}
        queue_settings_sidebar_action(state, "_pending_apply_preset", "Avançado")
        self.assertEqual(state["_pending_apply_preset"], "Avançado")

    def test_multiple_actions_coexist(self):
        state = {}
        queue_settings_sidebar_action(state, "_pending_reload_settings")
        queue_settings_sidebar_action(state, "_pending_apply_preset", "Custom")
        self.assertTrue(state["_pending_reload_settings"])
        self.assertEqual(state["_pending_apply_preset"], "Custom")

    def test_numeric_value(self):
        state = {}
        queue_settings_sidebar_action(state, "my_key", 42)
        self.assertEqual(state["my_key"], 42)


if __name__ == "__main__":
    unittest.main()
