import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from llm.Agent.AgentRuntime import AgentRequest, AgentRuntime, EngineResult
from llm.Agent.memory import ContextMemory, MemoryRefreshScheduler
from session.manager import SessionManager


class MemoryReliabilityTests(unittest.TestCase):
    def test_next_runtime_sees_saved_answer_before_background_summary(self) -> None:
        class RecordingEngine:
            def __init__(self) -> None:
                self.contexts: list[str] = []

            def execute(self, request, context, emit):
                self.contexts.append(context.context_memory)
                return EngineResult(answer=f"answer for {request.goal}")

        class NoopScheduler:
            def schedule(self, memory, *, on_failure=None):
                return False

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _manager(Path(temp_dir))
            session_id = manager.new_session()
            engine = RecordingEngine()
            runtime = AgentRuntime(
                engine=engine,
                session_manager=manager,
                memory_scheduler=NoopScheduler(),
            )
            runtime.run(AgentRequest(goal="first", session_id=session_id))
            runtime.run(AgentRequest(goal="second", session_id=session_id))

        self.assertEqual(engine.contexts[0], "")
        self.assertIn("answer for first", engine.contexts[1])

    def test_failed_refresh_is_suppressed_until_a_new_record_arrives(self) -> None:
        attempts: list[int] = []

        def fail_refresh(memory: ContextMemory, target: int):
            attempts.append(target)
            raise RuntimeError("summary unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ContextMemory(Path(temp_dir) / "context_memory.jsonl")
            for index in range(5):
                memory.remember(question=f"q{index}", final_answer=f"a{index}")
            scheduler = MemoryRefreshScheduler(refresh=fail_refresh)
            self.assertTrue(scheduler.schedule(memory))
            self.assertTrue(scheduler.wait_for_idle())
            self.assertEqual(attempts, [5])
            self.assertIn("a4", memory.load_context())
            self.assertFalse(scheduler.schedule(memory))
            self.assertEqual(attempts, [5])

            memory.remember(question="q5", final_answer="a5")
            self.assertTrue(scheduler.schedule(memory))
            self.assertTrue(scheduler.wait_for_idle())
            self.assertEqual(attempts, [5, 6])

    def test_failed_agent_does_not_write_long_term_memory(self) -> None:
        class FailingEngine:
            def execute(self, request, context, emit):
                raise RuntimeError("agent failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = _manager(root)
            session_id = manager.new_session()
            result = AgentRuntime(engine=FailingEngine(), session_manager=manager).run(
                AgentRequest(goal="cannot finish", session_id=session_id)
            )
            records = ContextMemory(manager.memory_path(session_id)).load_records()

        self.assertEqual(result.status, "failed")
        self.assertEqual(records, [])

    def test_memory_persistence_failure_keeps_answer_and_emits_event(self) -> None:
        class SuccessfulEngine:
            def execute(self, request, context, emit):
                return EngineResult(answer="answer")

        events: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _manager(Path(temp_dir))
            session_id = manager.new_session()
            runtime = AgentRuntime(
                engine=SuccessfulEngine(),
                session_manager=manager,
                extra_handlers=[events.append],
            )
            with patch.object(ContextMemory, "remember", side_effect=OSError("disk unavailable")):
                result = runtime.run(AgentRequest(goal="finish", session_id=session_id))

        self.assertEqual(result.status, "finished")
        self.assertEqual(result.answer, "answer")
        self.assertIn("agent.memory.failed", [event["type"] for event in events])

    def test_rolling_summary_batches_history_then_only_new_records(self) -> None:
        batches: list[list[int]] = []

        def summarize(previous: str, records: list[dict[str, object]]) -> str:
            sequences = [int(record["sequence"]) for record in records]
            batches.append(sequences)
            return f"{previous};{','.join(map(str, sequences))}".strip(";")

        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ContextMemory(Path(temp_dir) / "context_memory.jsonl")
            for index in range(25):
                memory.remember(question=f"q{index}", final_answer=f"a{index}")
            update = memory.refresh_rolling_summary(summarize=summarize)
            self.assertEqual(update.covered_through_sequence, 25)
            self.assertEqual(batches, [list(range(1, 11)), list(range(11, 21)), list(range(21, 26))])
            self.assertFalse(memory.status().needs_fallback)

            memory.remember(question="q25", final_answer="a25")
            memory.refresh_rolling_summary(summarize=summarize, target_sequence=26)
            self.assertEqual(batches[-1], [26])
            self.assertEqual(memory.status().summary_covered_sequence, 26)

    def test_concurrent_writes_keep_contiguous_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ContextMemory(Path(temp_dir) / "context_memory.jsonl")
            threads = [
                threading.Thread(
                    target=memory.remember,
                    kwargs={"question": f"q{index}", "final_answer": f"a{index}"},
                )
                for index in range(20)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            records = memory.load_records()

        self.assertEqual([record["sequence"] for record in records], list(range(1, 21)))

    def test_legacy_records_remain_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "context_memory.jsonl"
            path.write_text('{"question":"old","final_answer":"answer"}\n', encoding="utf-8")
            memory = ContextMemory(path)
            memory.remember(question="new", final_answer="latest", run_id="run-2")
            records = memory.load_records()

        self.assertEqual([record["sequence"] for record in records], [1, 2])
        self.assertEqual(records[0]["schema_version"], 1)
        self.assertEqual(records[1]["run_id"], "run-2")


def _manager(root: Path) -> SessionManager:
    return SessionManager(
        current_session_path=root / "session" / "current_session.json",
        memory_root=root / "storage" / "session_memory",
        trace_root=root / "trace" / "session_trace",
    )


if __name__ == "__main__":
    unittest.main()
