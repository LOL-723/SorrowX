import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from cli import commands
from daemon import main as daemon_main
from llm.Agent.memory import (
    CONTEXT_MEMORY_SUMMARY_MAX_CHARS,
    ContextMemory,
    ContextMemorySummaryUpdate,
)
from llm.Agent.memory_summary import refresh_context_memory_summary
from session.manager import SessionManager


class ContextMemorySummaryTests(unittest.TestCase):
    def test_refresh_uses_latest_ten_records_and_keeps_raw_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ContextMemory(Path(temp_dir) / "context_memory.jsonl")
            for index in range(12):
                memory.remember(question=f"question-{index}", final_answer=f"answer-{index}")

            seen_records: list[dict[str, str]] = []

            def summarize(records: list[dict[str, str]]) -> str:
                seen_records.extend(records)
                return "最近结论"

            update = memory.refresh_summary(summarize=summarize)

            self.assertEqual(update.status, "updated")
            self.assertEqual(
                [item["question"] for item in seen_records],
                [f"question-{index}" for index in range(2, 12)],
            )
            self.assertEqual(len(memory.load()), 12)
            self.assertEqual(memory.load_summary(), "最近结论")
            self.assertTrue(memory.summary_path.exists())

    def test_refresh_skips_model_when_latest_ten_records_are_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ContextMemory(Path(temp_dir) / "context_memory.jsonl")
            memory.remember(question="question", final_answer="answer")
            memory.refresh_summary(summarize=lambda _: "stable summary")

            update = memory.refresh_summary(
                summarize=lambda _: self.fail("summarizer should not be called")
            )

            self.assertEqual(update.status, "unchanged")
            self.assertEqual(update.summary, "stable summary")

    def test_refresh_limits_summary_body_to_three_hundred_characters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ContextMemory(Path(temp_dir) / "context_memory.jsonl")
            memory.remember(question="question", final_answer="answer")

            update = memory.refresh_summary(summarize=lambda _: "甲" * 350)

            self.assertEqual(len(update.summary), CONTEXT_MEMORY_SUMMARY_MAX_CHARS)
            self.assertEqual(len(memory.load_summary()), CONTEXT_MEMORY_SUMMARY_MAX_CHARS)

    def test_force_refresh_calls_model_even_when_source_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ContextMemory(Path(temp_dir) / "context_memory.jsonl")
            memory.remember(question="question", final_answer="answer")
            memory.refresh_summary(summarize=lambda _: "old summary")

            update = memory.refresh_summary(summarize=lambda _: "new summary", force=True)

            self.assertEqual(update.status, "updated")
            self.assertEqual(memory.load_summary(), "new summary")

    def test_model_summary_uses_ordered_records_and_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ContextMemory(Path(temp_dir) / "context_memory.jsonl")
            memory.remember(question="same question", final_answer="first answer")
            memory.remember(question="same question", final_answer="latest answer")

            with patch(
                "llm.Agent.memory_summary._chat_completion",
                return_value=json.dumps(
                    {"summary": "第1次为 first answer；第2次最新为 latest answer"}
                ),
            ) as completion:
                update = refresh_context_memory_summary(memory, force=True)

            payload = json.loads(completion.call_args.kwargs["user_message"])
            self.assertEqual([record["sequence"] for record in payload["records"]], [1, 2])
            self.assertIn("latest", update.summary)

    def test_check_memory_command_reads_summary_without_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _manager(Path(temp_dir))
            session_id = manager.new_session()
            memory = ContextMemory(manager.memory_path(session_id))
            memory.remember(question="question", final_answer="answer")
            memory.refresh_summary(summarize=lambda _: "saved summary")
            output = StringIO()

            with (
                patch.object(commands, "get_session_manager", return_value=manager),
                redirect_stdout(output),
            ):
                self.assertEqual(commands.check_memory_command([]), 0)

            self.assertEqual(memory.load_summary(), "saved summary")
            self.assertIn("saved summary", output.getvalue())

    def test_update_memory_command_forces_summary_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _manager(Path(temp_dir))
            session_id = manager.new_session()
            memory_path = manager.memory_path(session_id)
            update = ContextMemorySummaryUpdate(
                status="updated",
                path=memory_path.with_name("context_memory_summary.md"),
                summary="fresh summary",
                record_count=3,
            )
            output = StringIO()

            with (
                patch.object(commands, "get_session_manager", return_value=manager),
                patch.object(
                    commands,
                    "refresh_context_memory_summary",
                    return_value=update,
                ) as refresh,
                redirect_stdout(output),
            ):
                self.assertEqual(commands.update_memory_command([]), 0)

            refresh.assert_called_once()
            self.assertTrue(refresh.call_args.kwargs["force"])
            self.assertIn("fresh summary", output.getvalue())

    def test_daemon_startup_refreshes_current_session_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _manager(Path(temp_dir))
            session_id = manager.new_session()

            with (
                patch("session.manager.get_session_manager", return_value=manager),
                patch("llm.Agent.memory_summary.refresh_context_memory_summary") as refresh,
            ):
                daemon_main._refresh_current_session_memory()

            refreshed_memory = refresh.call_args.args[0]
            self.assertEqual(refreshed_memory.path, manager.memory_path(session_id))


if __name__ == "__main__":
    unittest.main()


def _manager(root: Path) -> SessionManager:
    return SessionManager(
        current_session_path=root / "session" / "current_session.json",
        memory_root=root / "storage" / "session_memory",
        trace_root=root / "trace" / "session_trace",
    )
