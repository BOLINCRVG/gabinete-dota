from __future__ import annotations

import re
import unittest

from app_core.streamlit_apptest_core import new_apptest_for_file, quiet_streamlit_bare_mode_warnings, run_apptest_quiet


class UIMojibakeSmoke(unittest.TestCase):
    _PATTERN = re.compile("\\u00c3|[A-Za-z\\u00c0-\\u00ff]\\?|\\?[A-Za-z\\u00c0-\\u00ff]")

    def _collect_suspects(self, at) -> list[str]:
        suspects: list[str] = []
        seen: set[str] = set()
        collections = [
            (at.radio, "label"),
            (at.selectbox, "label"),
            (at.button, "label"),
            (at.checkbox, "label"),
            (at.text_input, "label"),
            (at.text_area, "label"),
            (at.slider, "label"),
            (at.markdown, "value"),
            (at.caption, "value"),
            (at.header, "value"),
            (at.subheader, "value"),
        ]
        for coll, attr in collections:
            for w in coll:
                txt = str(getattr(w, attr, "") or "")
                if self._PATTERN.search(txt):
                    if txt in seen:
                        continue
                    seen.add(txt)
                    suspects.append(txt)
        return suspects

    def _assert_clean_render(self, configure_state=None) -> None:
        try:
            from streamlit.testing.v1 import AppTest  # type: ignore  # noqa: F401 - probe de disponibilidade (guarda skipTest)
        except Exception:
            self.skipTest("streamlit.testing.v1 unavailable")

        quiet_streamlit_bare_mode_warnings()
        at = new_apptest_for_file("plays_app.py")
        if callable(configure_state):
            configure_state(at)
        run_apptest_quiet(at, 240)

        self.assertEqual(len(at.exception), 0, f"App raised exceptions: {at.exception}")
        suspects = self._collect_suspects(at)
        self.assertFalse(
            suspects,
            "Mojibake-like UI text detected:\n"
            + "\n".join((s.encode("unicode_escape", "ignore").decode("ascii", "ignore") for s in suspects[:12])),
        )

    def test_no_mojibake_on_bootstrap_screen(self) -> None:
        self._assert_clean_render()

    def test_no_mojibake_on_results_with_broken_mode_and_tab(self) -> None:
        def _setup(at) -> None:
            at.session_state["_probe_skip_csv_bootstrap_gate"] = True
            at.session_state["ui_page_mode"] = "Fluxo R?pido"
            at.session_state["selected_tab_names"] = ["Consist?ncia de Mercado (Laplace)"]
            at.session_state["flow_stage"] = "results"
            at.session_state["_analysis_force_once"] = True

        self._assert_clean_render(_setup)


if __name__ == "__main__":
    unittest.main()
