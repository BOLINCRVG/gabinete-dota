import unittest

from app_core.navigation_state_core import (
    consume_flash_saved_profile_msg,
    consume_nav_pending,
    get_dismiss_update_banner,
    get_flash_saved_profile_msg,
    get_nav_main,
    get_nav_pending,
    get_ui_page_mode,
    normalize_nav_main,
    resolve_nav_page,
    set_dismiss_update_banner,
    set_flash_saved_profile_msg,
    set_nav_main,
    set_nav_pending,
    set_ui_page_mode,
)


class NavigationStateCoreTests(unittest.TestCase):
    def test_nav_roundtrip(self) -> None:
        state = {}
        self.assertEqual(get_nav_pending(state), "")
        set_nav_pending(state, "Tecnico")
        self.assertEqual(consume_nav_pending(state), "Tecnico")
        self.assertEqual(get_nav_pending(state), "")
        self.assertEqual(get_nav_main(state), "Plays")
        self.assertEqual(set_nav_main(state, "Tecnico"), "Tecnico")
        # Live-Bet saiu do nav (2026-07: fundida no Radar) -> normaliza pro default
        self.assertEqual(set_nav_main(state, "Live-Bet"), "Plays")
        self.assertEqual(normalize_nav_main("Parâmetros"), "Plays")
        self.assertEqual(get_ui_page_mode(state), "Fluxo Rápido")
        self.assertEqual(set_ui_page_mode(state, "Completo"), "Completo")

    def test_flash_and_banner_flags(self) -> None:
        state = {}
        set_flash_saved_profile_msg(state, "ok")
        self.assertEqual(get_flash_saved_profile_msg(state), "ok")
        self.assertEqual(consume_flash_saved_profile_msg(state), "ok")
        self.assertEqual(get_flash_saved_profile_msg(state), "")
        self.assertFalse(get_dismiss_update_banner(state))
        self.assertTrue(set_dismiss_update_banner(state, True))

    def test_resolve_nav_page_applies_pending_and_alias(self) -> None:
        state = {"_nav_pending": "Parametro antigo"}
        self.assertEqual(resolve_nav_page(state), "Plays")
        self.assertEqual(get_nav_main(state), "Plays")
        state["_nav_pending"] = "Radar"
        self.assertEqual(resolve_nav_page(state), "Radar")
        self.assertEqual(get_nav_main(state), "Radar")
        state["nav_main"] = "Calibracoes"
        self.assertEqual(resolve_nav_page(state), "Tecnico")


if __name__ == "__main__":
    unittest.main()
