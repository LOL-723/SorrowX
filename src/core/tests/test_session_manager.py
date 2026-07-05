import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from cli import commands
from session.manager import SessionManager


class SessionManagerTests(unittest.TestCase):
    def test_new_session_creates_sibling_session_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = _manager(root)

            first = manager.new_session()
            second = manager.new_session()

            self.assertEqual(first, "session_1")
            self.assertEqual(second, "session_2")
            self.assertEqual(manager.current_session(), "session_2")
            self.assertTrue((root / "storage" / "session_memory" / "session_1").is_dir())
            self.assertTrue((root / "storage" / "session_memory" / "session_2").is_dir())
            self.assertTrue((root / "trace" / "session_trace" / "session_1").is_dir())
            self.assertTrue((root / "trace" / "session_trace" / "session_2").is_dir())
            self.assertFalse(
                (
                    root
                    / "storage"
                    / "session_memory"
                    / "session_1"
                    / "session_2"
                ).exists()
            )

    def test_switch_rejects_unknown_or_nested_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _manager(Path(temp_dir))
            manager.new_session()

            with self.assertRaises(ValueError):
                manager.switch_session("session_1/session_2")
            with self.assertRaises(ValueError):
                manager.switch_session("..")
            with self.assertRaises(ValueError):
                manager.switch_session("session_2")

    def test_ensure_current_session_reuses_existing_current(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _manager(Path(temp_dir))

            created = manager.ensure_current_session()
            reused = manager.ensure_current_session()

            self.assertEqual(created, "session_1")
            self.assertEqual(reused, "session_1")
            self.assertEqual(manager.list_sessions(), ["session_1"])

    def test_delete_current_session_is_blocked_and_preserves_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = _manager(root)
            session_id = manager.new_session()
            paths = manager.paths_for(session_id)
            paths.memory_path.write_text("memory", encoding="utf-8")
            (paths.trace_dir / "run-1.jsonl").write_text("trace", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "cannot delete current session"):
                manager.delete_session(session_id)

            self.assertEqual(manager.current_session(), "session_1")
            self.assertEqual(manager.list_sessions(), ["session_1"])
            self.assertTrue(paths.memory_dir.exists())
            self.assertTrue(paths.trace_dir.exists())
            self.assertTrue(manager.current_session_path.exists())

    def test_delete_non_current_session_preserves_current_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = _manager(root)
            first = manager.new_session()
            second = manager.new_session()

            deleted = manager.delete_session(first)

            self.assertEqual(deleted, "session_1")
            self.assertEqual(manager.current_session(), second)
            self.assertEqual(manager.list_sessions(), ["session_2"])


class SessionCliTests(unittest.TestCase):
    def test_session_command_creates_lists_and_switches_current_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _manager(Path(temp_dir))
            stream = StringIO()

            with (
                patch.object(commands, "get_session_manager", return_value=manager),
                redirect_stdout(stream),
            ):
                self.assertEqual(commands.session_command(["new"]), 0)
                self.assertEqual(commands.session_command(["new"]), 0)
                self.assertEqual(commands.session_command(["list"]), 0)
                self.assertEqual(commands.session_command(["switch", "session_1"]), 0)
                self.assertEqual(commands.session_command(["current"]), 0)

            output = stream.getvalue()
            self.assertIn("Current session: session_1", output)
            self.assertIn("Current session: session_2", output)
            self.assertIn("* session_2", output)
            self.assertTrue(output.rstrip().endswith("session_1"))

    def test_session_del_command_deletes_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _manager(Path(temp_dir))
            manager.new_session()
            manager.new_session()
            stream = StringIO()

            with (
                patch.object(commands, "get_session_manager", return_value=manager),
                redirect_stdout(stream),
            ):
                self.assertEqual(commands.session_command(["del", "session_1"]), 0)
                self.assertEqual(commands.session_command(["current"]), 0)

            self.assertIn("Deleted session: session_1", stream.getvalue())
            self.assertTrue(stream.getvalue().rstrip().endswith("session_2"))

    def test_session_del_command_prints_current_session_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _manager(Path(temp_dir))
            manager.new_session()
            stream = StringIO()

            with (
                patch.object(commands, "get_session_manager", return_value=manager),
                redirect_stdout(stream),
            ):
                self.assertEqual(commands.session_command(["del", "session_1"]), 1)

            self.assertIn("cannot delete current session: session_1", stream.getvalue())


def _manager(root: Path) -> SessionManager:
    return SessionManager(
        current_session_path=root / "session" / "current_session.json",
        memory_root=root / "storage" / "session_memory",
        trace_root=root / "trace" / "session_trace",
    )


if __name__ == "__main__":
    unittest.main()
