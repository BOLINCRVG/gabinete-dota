import unittest

from app_core.tab_ui_core import build_tab_hero_html, build_tab_section_html


class TestBuildTabHeroHtml(unittest.TestCase):

    def test_contains_hero_tab_class(self):
        html = build_tab_hero_html(title="Players", subtitle="ML por lineup")
        self.assertIn("gb-hero-tab", html)

    def test_title_in_output(self):
        html = build_tab_hero_html(title="Players", subtitle="sub")
        self.assertIn("Players", html)

    def test_subtitle_in_output(self):
        html = build_tab_hero_html(title="T", subtitle="My subtitle")
        self.assertIn("My subtitle", html)

    def test_chips_rendered(self):
        html = build_tab_hero_html(title="T", subtitle="S", chips=["BO3", "2 jogos"])
        self.assertIn("BO3", html)
        self.assertIn("gb-chip-row", html)

    def test_no_chips_no_chip_row(self):
        html = build_tab_hero_html(title="T", subtitle="S")
        self.assertNotIn("gb-chip-row", html)

    def test_eyebrow_in_output(self):
        html = build_tab_hero_html(title="T", subtitle="S", eyebrow="Liga")
        self.assertIn("Liga", html)
        self.assertIn("gb-eyebrow", html)

    def test_no_eyebrow_no_eyebrow_div(self):
        html = build_tab_hero_html(title="T", subtitle="S")
        self.assertNotIn("gb-eyebrow", html)

    def test_html_escaped_title(self):
        html = build_tab_hero_html(title="<script>", subtitle="S")
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_returns_string(self):
        self.assertIsInstance(build_tab_hero_html(title="T", subtitle="S"), str)

    def test_empty_chips_filtered(self):
        html = build_tab_hero_html(title="T", subtitle="S", chips=["", "  "])
        self.assertNotIn("gb-chip-row", html)


class TestBuildTabSectionHtml(unittest.TestCase):

    def test_section_head_class(self):
        html = build_tab_section_html(title="Recorte")
        self.assertIn("gb-section-head", html)

    def test_title_in_output(self):
        html = build_tab_section_html(title="Recorte")
        self.assertIn("Recorte", html)

    def test_subtitle_in_output(self):
        html = build_tab_section_html(title="T", subtitle="Feche a liga e o split")
        self.assertIn("Feche a liga e o split", html)

    def test_eyebrow_in_output(self):
        html = build_tab_section_html(title="T", eyebrow="Filtros")
        self.assertIn("Filtros", html)
        self.assertIn("gb-eyebrow", html)

    def test_no_subtitle_no_subtitle_div(self):
        html = build_tab_section_html(title="T")
        self.assertNotIn("gb-section-sub", html)

    def test_returns_string(self):
        self.assertIsInstance(build_tab_section_html(title="T"), str)

    def test_html_escaped_title(self):
        html = build_tab_section_html(title="<b>bold</b>")
        self.assertNotIn("<b>", html)


if __name__ == "__main__":
    unittest.main()
