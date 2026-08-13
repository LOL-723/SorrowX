import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from daemon.events import EventBus
from daemon.handlers import _run_agent_in_background
from llm.Agent.AgentRuntime import AgentResult


class DaemonAgentRuntimeTests(unittest.TestCase):
    def test_background_run_uses_shared_agent_runtime(self) -> None:
        class FakeRuntime:
            def __init__(self) -> None:
                self.calls = []

            def run(self, request, *, run_id, event_bus):
                self.calls.append((request, run_id, event_bus))
                return AgentResult(
                    run_id=run_id,
                    status="finished",
                    answer="answer",
                    error=None,
                    events_path=Path("runs") / run_id / "events.jsonl",
                )

        runtime = FakeRuntime()
        state = SimpleNamespace(event_bus=EventBus())
        with patch("llm.Agent.AgentRuntime.get_agent_runtime", return_value=runtime):
            asyncio.run(
                _run_agent_in_background(
                    state=state,
                    run_id="run-1",
                    goal="goal",
                    session_id="session_1",
                )
            )

        self.assertEqual(len(runtime.calls), 1)
        request, run_id, event_bus = runtime.calls[0]
        self.assertEqual(request.goal, "goal")
        self.assertEqual(request.session_id, "session_1")
        self.assertEqual(run_id, "run-1")
        self.assertIs(event_bus, state.event_bus)


if __name__ == "__main__":
    unittest.main()
