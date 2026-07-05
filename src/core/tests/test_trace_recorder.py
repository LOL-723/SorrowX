import tempfile
import unittest
from pathlib import Path

from trace.formatting import format_trace_entry, format_trace_summary
from trace.reader import TraceSummary, read_trace
from trace.recorder import TraceRecorder, trace_run


class TraceRecorderTests(unittest.TestCase):
    def test_recorder_writes_lightweight_run_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = TraceRecorder(records_dir=temp_dir, enabled=True)
            request = recorder.prepare_client_to_core(
                {
                    "jsonrpc": "2.0",
                    "id": "request-1",
                    "method": "agent.run",
                    "params": {
                        "client": "test",
                        "goal": "summarize README",
                    },
                }
            )
            recorder.write_prepared("run-1", request)
            recorder.record_core_event(
                {
                    "type": "agent.loop.thought",
                    "run_id": "run-1",
                    "step_id": "step_1",
                    "thought": "this should not be stored in trace",
                }
            )
            call_id = recorder.record_core_to_llm(
                "run-1",
                model="test-model",
                message_count=3,
                tool_count=1,
            )
            recorder.record_llm_to_core(
                "run-1",
                call_id=call_id,
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 47,
                    "total_tokens": 57,
                },
            )
            recorder.close()

            entries = read_trace("run-1", records_dir=temp_dir)
            trace_path_exists = (Path(temp_dir) / "run-1.jsonl").exists()

        self.assertEqual(
            [entry["direction"] for entry in entries],
            ["CLIENT_TO_CORE", "CORE", "CORE_TO_LLM", "LLM_TO_CORE"],
        )
        self.assertEqual(entries[0]["payload"]["method"], "agent.run")
        self.assertEqual(entries[1]["payload"]["event_type"], "agent.loop.thought")
        self.assertNotIn("thought", entries[1]["payload"])
        self.assertEqual(entries[2]["payload"]["message_count"], 3)
        self.assertEqual(entries[2]["payload"]["tool_call_count"], 1)
        self.assertEqual(entries[3]["payload"]["completion_tokens"], 47)
        self.assertTrue(trace_path_exists)

    def test_trace_formatting_matches_cli_shape(self) -> None:
        summary = TraceSummary(
            run_id="run-1",
            entry_count=2,
            started_at="2026-07-04T10:00:00.001+00:00",
            finished_at="2026-07-04T10:00:00.851+00:00",
        )
        self.assertEqual(
            format_trace_summary(summary),
            "run-1  entries=2  started=10:00:00.001  finished=10:00:00.851",
        )
        self.assertEqual(
            format_trace_entry(
                {
                    "ts": "2026-07-04T10:00:00.009+00:00",
                    "direction": "CORE_TO_LLM",
                    "payload": {
                        "message_count": 3,
                        "tool_call_count": 1,
                    },
                }
            ),
            "10:00:00.009 CORE→LLM msgs=3 tools=1",
        )

    def test_recorder_routes_all_trace_entries_to_session_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recorder = TraceRecorder(
                records_dir=root / "records",
                session_records_root=root / "session_trace",
                enabled=True,
            )
            request = recorder.prepare_client_to_core(
                {
                    "jsonrpc": "2.0",
                    "id": "request-1",
                    "method": "agent.run",
                    "params": {
                        "client": "test",
                        "goal": "summarize README",
                        "session_id": "session_1",
                    },
                }
            )
            recorder.write_prepared("run-1", request, session_id="session_1")
            recorder.record_core_event(
                {
                    "type": "agent.loop.thought",
                    "run_id": "run-1",
                    "session_id": "session_1",
                    "step_id": "step_1",
                    "thought": "this should not be stored in trace",
                }
            )
            with trace_run("run-1", session_id="session_1"):
                call_id = recorder.record_core_to_llm(
                    "run-1",
                    model="test-model",
                    message_count=3,
                    tool_count=1,
                )
                recorder.record_llm_to_core(
                    "run-1",
                    call_id=call_id,
                    usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 47,
                        "total_tokens": 57,
                    },
                )
            recorder.close()

            session_dir = root / "session_trace" / "session_1"
            entries = read_trace("run-1", records_dir=session_dir)
            session_trace_path_exists = (session_dir / "run-1.jsonl").exists()
            global_trace_path_exists = (root / "records" / "run-1.jsonl").exists()

        self.assertEqual(
            [entry["direction"] for entry in entries],
            ["CLIENT_TO_CORE", "CORE", "CORE_TO_LLM", "LLM_TO_CORE"],
        )
        self.assertTrue(session_trace_path_exists)
        self.assertFalse(global_trace_path_exists)
        self.assertEqual({entry.get("session_id") for entry in entries}, {"session_1"})


if __name__ == "__main__":
    unittest.main()
