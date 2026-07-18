"""Tests for the unified ui_notify_core helper."""
from __future__ import annotations

import unittest

from app_core.ui_notify_core import notify, notify_caption, notify_exception


class _FakeSt:
    def __init__(self):
        self.infos = []
        self.successes = []
        self.warnings = []
        self.errors = []
        self.captions = []

    def info(self, msg, **k): self.infos.append(str(msg))
    def success(self, msg, **k): self.successes.append(str(msg))
    def warning(self, msg, **k): self.warnings.append(str(msg))
    def error(self, msg, **k): self.errors.append(str(msg))
    def caption(self, msg, **k): self.captions.append(str(msg))


class TestNotify(unittest.TestCase):

    def test_info_uses_info_method(self):
        fake = _FakeSt()
        notify(fake, "info", "Hello world")
        self.assertEqual(len(fake.infos), 1)
        self.assertIn("Hello world", fake.infos[0])
        self.assertIn("ℹ", fake.infos[0])

    def test_success_uses_success_method(self):
        fake = _FakeSt()
        notify(fake, "success", "Done")
        self.assertEqual(len(fake.successes), 1)
        self.assertIn("✓", fake.successes[0])

    def test_warn_uses_warning_method(self):
        fake = _FakeSt()
        notify(fake, "warn", "Stale data")
        self.assertEqual(len(fake.warnings), 1)
        self.assertIn("⚠", fake.warnings[0])

    def test_error_uses_error_method(self):
        fake = _FakeSt()
        notify(fake, "error", "Bad input")
        self.assertEqual(len(fake.errors), 1)
        self.assertIn("✕", fake.errors[0])

    def test_detail_rendered_as_caption(self):
        fake = _FakeSt()
        notify(fake, "warn", "Header", detail="Some explanation")
        self.assertEqual(len(fake.warnings), 1)
        self.assertEqual(len(fake.captions), 1)
        self.assertIn("Some explanation", fake.captions[0])

    def test_no_detail_no_caption(self):
        fake = _FakeSt()
        notify(fake, "info", "Only message")
        self.assertEqual(len(fake.captions), 0)

    def test_custom_icon_overrides_default(self):
        fake = _FakeSt()
        notify(fake, "info", "Reload", icon="🔄")
        self.assertIn("🔄", fake.infos[0])
        self.assertNotIn("ℹ", fake.infos[0])

    def test_unknown_level_falls_back_to_info(self):
        fake = _FakeSt()
        notify(fake, "nonsense", "Hello")  # type: ignore[arg-type]
        self.assertEqual(len(fake.infos), 1)


class TestNotifyCaption(unittest.TestCase):

    def test_caption_only(self):
        fake = _FakeSt()
        notify_caption(fake, "small text")
        self.assertEqual(len(fake.captions), 1)
        self.assertEqual(len(fake.infos), 0)


class TestNotifyException(unittest.TestCase):

    def test_shows_error_with_type_and_message(self):
        fake = _FakeSt()
        try:
            raise ValueError("invalid X")
        except ValueError as exc:
            notify_exception(fake, exc, context="Tentando carregar X")
        self.assertEqual(len(fake.errors), 1)
        self.assertIn("Tentando carregar X", fake.errors[0])
        self.assertEqual(len(fake.captions), 1)
        self.assertIn("ValueError", fake.captions[0])
        self.assertIn("invalid X", fake.captions[0])

    def test_exception_without_message(self):
        fake = _FakeSt()
        try:
            raise RuntimeError()
        except RuntimeError as exc:
            notify_exception(fake, exc)
        self.assertIn("sem mensagem", fake.captions[0])


if __name__ == "__main__":
    unittest.main()
