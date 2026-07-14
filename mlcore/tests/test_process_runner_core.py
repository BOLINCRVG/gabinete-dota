"""Tests for app_core/process_runner_core.py — run_cmd subprocess runner."""
import subprocess
import sys
import unittest

from app_core.process_runner_core import run_cmd, _no_window_kwargs


class TestNoWindowKwargs(unittest.TestCase):
    """Console-window suppression that fixes the 'shell popping up' bug."""

    def test_windows_returns_create_no_window(self):
        from unittest import mock
        with mock.patch.object(sys, "platform", "win32"):
            kw = _no_window_kwargs()
        self.assertIn("creationflags", kw)
        expected = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
        self.assertEqual(kw["creationflags"], expected)

    def test_non_windows_returns_empty(self):
        from unittest import mock
        for plat in ("linux", "darwin"):
            with mock.patch.object(sys, "platform", plat):
                self.assertEqual(_no_window_kwargs(), {})

    def test_kwargs_popen_compatible(self):
        # Whatever is returned must be accepted by Popen on this platform.
        result = run_cmd([sys.executable, "-c", "print('ok')"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("ok", result.stdout)


class TestRunCmd(unittest.TestCase):

    def test_successful_command_returns_zero_rc(self):
        result = run_cmd([sys.executable, "-c", "print('hello')"])
        self.assertEqual(result.returncode, 0)

    def test_successful_command_stdout_contains_output(self):
        result = run_cmd([sys.executable, "-c", "print('hello world')"])
        self.assertIn("hello world", result.stdout)

    def test_failed_command_nonzero_rc(self):
        result = run_cmd([sys.executable, "-c", "raise SystemExit(1)"])
        self.assertNotEqual(result.returncode, 0)

    def test_returns_completed_process(self):
        result = run_cmd([sys.executable, "-c", "pass"])
        self.assertIsInstance(result, subprocess.CompletedProcess)

    def test_stdout_is_string(self):
        result = run_cmd([sys.executable, "-c", "print('x')"])
        self.assertIsInstance(result.stdout, str)

    def test_stderr_is_string(self):
        result = run_cmd([sys.executable, "-c", "pass"])
        self.assertIsInstance(result.stderr, str)

    def test_invalid_command_returns_error(self):
        result = run_cmd(["nonexistent_program_xyz_abc"])
        self.assertNotEqual(result.returncode, 0)

    def test_heartbeat_called(self):
        calls = []
        run_cmd(
            [sys.executable, "-c", "import time; time.sleep(1.5)"],
            heartbeat_cb=lambda t: calls.append(t),
            heartbeat_interval_s=0.5,
        )
        # With 1.5s sleep and 0.5s interval, at least 1 heartbeat must fire
        self.assertGreater(len(calls), 0)

    def test_heartbeat_receives_elapsed_time(self):
        times = []
        run_cmd(
            [sys.executable, "-c", "import time; time.sleep(1.5)"],
            heartbeat_cb=lambda t: times.append(t),
            heartbeat_interval_s=0.5,
        )
        if times:
            self.assertGreaterEqual(times[0], 0.0)

    def test_multiline_output(self):
        code = "for i in range(5): print(i)"
        result = run_cmd([sys.executable, "-c", code])
        self.assertEqual(result.returncode, 0)
        self.assertIn("4", result.stdout)


if __name__ == "__main__":
    unittest.main()
