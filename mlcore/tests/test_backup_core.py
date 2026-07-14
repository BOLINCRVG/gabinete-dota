"""Tests for app_core/backup_core.py — backup_file_if_exists."""
import tempfile
import unittest
from pathlib import Path

from app_core.backup_core import backup_file_if_exists


class TestBackupFileIfExists(unittest.TestCase):

    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as app_root:
            result = backup_file_if_exists(
                "/nonexistent/file.json",
                tag="test",
                app_root=app_root,
            )
            self.assertEqual(result, "")

    def test_directory_input_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = backup_file_if_exists(tmpdir, tag="test", app_root=tmpdir)
            self.assertEqual(result, "")

    def test_backup_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "data.json"
            src.write_text('{"x": 1}', encoding="utf-8")
            result = backup_file_if_exists(src, tag="manual", app_root=tmpdir)
            self.assertTrue(len(result) > 0)
            self.assertTrue(Path(result).exists())

    def test_backup_in_backups_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "model.json"
            src.write_text("data")
            result = backup_file_if_exists(src, tag="auto", app_root=tmpdir)
            self.assertIn("backups", result)

    def test_backup_preserves_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "store.json"
            content = '{"version": "1.2.3"}'
            src.write_text(content, encoding="utf-8")
            result = backup_file_if_exists(src, tag="test", app_root=tmpdir)
            self.assertEqual(Path(result).read_text(encoding="utf-8"), content)

    def test_backup_preserves_extension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "artifact.json"
            src.write_text("{}")
            result = backup_file_if_exists(src, tag="v1", app_root=tmpdir)
            self.assertTrue(result.endswith(".json"))

    def test_tag_sanitized_in_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "data.json"
            src.write_text("{}")
            result = backup_file_if_exists(src, tag="my tag/test!", app_root=tmpdir)
            fname = Path(result).name
            # Special chars should be replaced
            self.assertNotIn("/", fname)
            self.assertNotIn("!", fname)

    def test_returns_string(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "f.json"
            src.write_text("{}")
            result = backup_file_if_exists(src, tag="t", app_root=tmpdir)
            self.assertIsInstance(result, str)

    def test_default_tag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "file.json"
            src.write_text("{}")
            result = backup_file_if_exists(src, app_root=tmpdir)
            self.assertIn("manual", Path(result).name)


if __name__ == "__main__":
    unittest.main()
