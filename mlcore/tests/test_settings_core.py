import tempfile
import unittest
from datetime import date
from pathlib import Path

from app_core.settings_core import (
    apply_settings_to_session,
    init_settings_once,
    json_safe,
    load_settings_from_disk,
    process_pending_settings_actions,
    save_settings_to_disk,
    settings_path,
)


class TestSettingsPath(unittest.TestCase):

    def test_builds_path(self):
        p = settings_path(Path("/root"), "cfg.json")
        self.assertEqual(p, Path("/root/cfg.json"))

    def test_returns_path_type(self):
        p = settings_path(Path("/root"), "cfg.json")
        self.assertIsInstance(p, Path)


class TestJsonSafe(unittest.TestCase):

    def test_none_stays_none(self):
        self.assertIsNone(json_safe(None))

    def test_bool_kept(self):
        self.assertEqual(json_safe(True), True)

    def test_int_kept(self):
        self.assertEqual(json_safe(42), 42)

    def test_float_kept(self):
        self.assertEqual(json_safe(3.14), 3.14)

    def test_string_kept(self):
        self.assertEqual(json_safe("hello"), "hello")

    def test_date_to_isoformat(self):
        result = json_safe(date(2025, 6, 1))
        self.assertEqual(result, "2025-06-01")

    def test_list_recursed(self):
        result = json_safe([1, "a", None])
        self.assertEqual(result, [1, "a", None])

    def test_dict_recursed(self):
        result = json_safe({"a": 1, "b": "x"})
        self.assertEqual(result, {"a": 1, "b": "x"})

    def test_unknown_type_to_str(self):
        class Custom:
            def __str__(self):
                return "custom"
        result = json_safe(Custom())
        self.assertEqual(result, "custom")

    def test_nested_structure(self):
        result = json_safe({"key": [1, 2, date(2025, 1, 1)]})
        self.assertEqual(result, {"key": [1, 2, "2025-01-01"]})


class TestSaveAndLoadSettings(unittest.TestCase):

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = settings_path(Path(td), "cfg.json")
            ss = {"cfg_profile": "Padrao", "n_sims": 12345}
            save_settings_to_disk(p, ss, persist_keys=["cfg_profile", "n_sims"])
            data = load_settings_from_disk(p)
            self.assertEqual(data.get("cfg_profile"), "Padrao")
            self.assertEqual(int(data.get("n_sims")), 12345)

    def test_load_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "nonexistent.json"
            data = load_settings_from_disk(p)
            self.assertEqual(data, {})

    def test_only_persist_keys_saved(self):
        with tempfile.TemporaryDirectory() as td:
            p = settings_path(Path(td), "cfg.json")
            ss = {"cfg_profile": "Padrao", "ephemeral": "not_saved"}
            save_settings_to_disk(p, ss, persist_keys=["cfg_profile"])
            data = load_settings_from_disk(p)
            self.assertIn("cfg_profile", data)
            self.assertNotIn("ephemeral", data)


class TestApplySettingsToSession(unittest.TestCase):

    def test_applies_persist_key(self):
        ss = {"a": 1}
        apply_settings_to_session(ss, {"b": 2, "c": 3}, persist_keys=["b"], force=False)
        self.assertEqual(ss.get("b"), 2)

    def test_ignores_non_persist_key(self):
        ss = {"a": 1}
        apply_settings_to_session(ss, {"b": 2, "c": 3}, persist_keys=["b"], force=False)
        self.assertIsNone(ss.get("c"))


class TestProcessPendingSettingsActions(unittest.TestCase):

    def test_pending_actions_dispatched(self):
        ss = {
            "_pending_apply_preset": "Padrao",
            "_pending_apply_excellent": True,
            "_pending_apply_twenty_ten": True,
        }
        calls = {"preset": 0, "excellent": 0, "twenty_ten": 0, "reset": 0}

        process_pending_settings_actions(
            ss,
            load_settings_fn=lambda: {"x": 1},
            apply_settings_fn=lambda _d, _f: None,
            apply_recommended_preset_fn=lambda _p: calls.__setitem__("preset", calls["preset"] + 1),
            apply_excellent_preset_fn=lambda: calls.__setitem__("excellent", calls["excellent"] + 1),
            apply_twenty_ten_preset_fn=lambda: calls.__setitem__("twenty_ten", calls["twenty_ten"] + 1),
            reset_all_tuning_fn=lambda: calls.__setitem__("reset", calls["reset"] + 1),
            param_defaults={"a": 1},
        )
        self.assertEqual(calls["preset"], 1)
        self.assertEqual(calls["excellent"], 1)
        self.assertEqual(calls["twenty_ten"], 1)


class TestInitSettingsOnce(unittest.TestCase):

    def test_boots_on_first_call(self):
        ss = {}
        calls = {"load": 0, "pending": 0}
        init_settings_once(
            ss,
            load_settings_fn=lambda: calls.__setitem__("load", calls["load"] + 1) or {},
            apply_settings_fn=lambda _d, _f: None,
            process_pending_fn=lambda: calls.__setitem__("pending", calls["pending"] + 1),
        )
        self.assertTrue(ss.get("_app_settings_booted"))
        self.assertEqual(calls["load"], 1)

    def test_second_call_skips_load(self):
        ss = {}
        calls = {"load": 0}
        fn = lambda: calls.__setitem__("load", calls["load"] + 1) or {}
        init_settings_once(ss, load_settings_fn=fn, apply_settings_fn=lambda _d, _f: None, process_pending_fn=lambda: None)
        init_settings_once(ss, load_settings_fn=fn, apply_settings_fn=lambda _d, _f: None, process_pending_fn=lambda: None)
        self.assertEqual(calls["load"], 1)


if __name__ == "__main__":
    unittest.main()
