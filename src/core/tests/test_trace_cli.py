import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from cli import commands
from core.trace.reader import TraceSummary


class FakeSessionManager:
    def ensure_current_session(self) -> str:
        return "session_1"

    def trace_dir(self, session_id: str) -> Path:
        return Path("session_trace") / session_id


class TraceCliTests(unittest.TestCase):
    def test_trace_command_lists_runs(self) -> None:
        stream = StringIO()
        fake_session_manager = FakeSessionManager()
        with (
            patch.object(
                commands,
                "get_session_manager",
                return_value=fake_session_manager,
            ),
            patch.object(
                commands,
                "list_traces",
                return_value=[
                    TraceSummary(
                        run_id="run-1",
                        entry_count=2,
                        started_at="2026-07-04T10:00:00.001+00:00",
                        finished_at="2026-07-04T10:00:00.851+00:00",
                    )
                ],
            ),
            redirect_stdout(stream),
        ):
            exit_code = commands.trace_command([])

        self.assertEqual(exit_code, 0)
        self.assertIn("run-1  entries=2", stream.getvalue())

    def test_trace_show_command_prints_entries(self) -> None:
        stream = StringIO()
        fake_session_manager = FakeSessionManager()
        with (
            patch.object(
                commands,
                "get_session_manager",
                return_value=fake_session_manager,
            ),
            patch.object(
                commands,
                "read_trace",
                return_value=[
                    {
                        "ts": "2026-07-04T10:00:00.001+00:00",
                        "direction": "CLIENT_TO_CORE",
                        "payload": {
                            "method": "agent.run",
                            "goal": "总结 README.md 的主要章节",
                        },
                    }
                ],
            ),
            redirect_stdout(stream),
        ):
            exit_code = commands.trace_command(["show", "run-1"])

        self.assertEqual(exit_code, 0)
        self.assertIn(
            '10:00:00.001 CLIENT→CORE method=agent.run goal="总结 README.md 的主要章节"',
            stream.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
