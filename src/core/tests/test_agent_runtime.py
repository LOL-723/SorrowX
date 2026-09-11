import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from daemon.events import EventBus
from llm.Agent.AgentEngine import AgentLoopEngine
from llm.Agent.AgentRuntime import AgentRequest, AgentRuntime
from llm.Agent.memory import ContextMemory, MemoryRefreshScheduler
from llm.Agent.scheduler import AgentScheduler
from session.manager import SessionManager


class AgentRuntimeTests(unittest.TestCase):
    def test_runtime_executes_agent_loop_and_writes_events(self) -> None:
        seen_context_memory: list[str] = []

        def fake_planner(state):
            seen_context_memory.append(state.get("context_memory", ""))
            return {
                "plan": [
                    {
                        "step_id": "step_1",
                        "task": "answer the goal",
                        "status": "pending",
                        "result": None,
                        "retry_count": 0,
                    }
                ],
                "plan_revision": 1,
                "plan_updates": [],
                "planner_mode": "initial",
                "agent_status": "running",
                "logs": [{"node": "planner_node", "message": "plan created"}],
            }

        def fake_select(state):
            if state.get("step_results"):
                return {
                    "should_continue_next": "finish",
                    "current_step_index": 1,
                    "current_step_id": None,
                    "agent_status": "running",
                    "logs": state.get("logs", [])
                    + [{"node": "select_next_step_node", "message": "done"}],
                }
            return {
                "should_continue_next": "continue",
                "current_step_index": 0,
                "current_step_id": "step_1",
                "agent_status": "running",
                "logs": state.get("logs", [])
                + [{"node": "select_next_step_node", "message": "selected"}],
            }

        def fake_loop(state):
            state["_event_callback"](
                {
                    "type": "agent.loop.thought",
                    "step_id": "step_1",
                    "turn": 1,
                    "decision": "finish",
                    "signal": "none",
                    "tool": "none",
                    "thought": "ready to answer",
                }
            )
            plan = list(state["plan"])
            plan[0] = dict(plan[0], status="done", result="step answer")
            return {
                "plan": plan,
                "step_results": [
                    {
                        "step_id": "step_1",
                        "task": "answer the goal",
                        "result": "step answer",
                    }
                ],
                "agent_status": "running",
                "logs": state.get("logs", [])
                + [{"node": "agent_loop_node", "message": "step completed"}],
            }

        seen_events = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = _manager(root)
            session_id = manager.new_session()
            memory = ContextMemory(manager.memory_path(session_id))
            memory.remember(question="old question", final_answer="old answer")
            memory.refresh_summary(summarize=lambda _: "old memory summary")
            scheduler = MemoryRefreshScheduler(
                refresh=lambda saved_memory, target: saved_memory.refresh_rolling_summary(
                    summarize=lambda previous, _: previous or "refreshed summary",
                    target_sequence=target,
                )
            )
            runtime = AgentRuntime(
                engine=AgentLoopEngine(),
                runs_dir=root / "runs",
                extra_handlers=[seen_events.append],
                session_manager=manager,
                memory_scheduler=scheduler,
            )
            with (
                patch("llm.Agent.AgentEngine.planner_node", side_effect=fake_planner),
                patch("llm.Agent.AgentEngine.select_next_step_node", side_effect=fake_select),
                patch("llm.Agent.AgentEngine.agent_loop_node", side_effect=fake_loop),
                patch("llm.Agent.AgentEngine._summarize_agent_answer", return_value="final answer"),
            ):
                result = runtime.run(AgentRequest(goal="complex task", session_id=session_id))

            self.assertEqual(result.status, "finished")
            self.assertEqual(result.answer, "final answer")
            self.assertTrue(result.events_path.exists())
            self.assertEqual(seen_context_memory, ["old memory summary"])
            self.assertEqual(
                memory.load()[-1],
                {"question": "complex task", "final_answer": "final answer"},
            )
            self.assertTrue(scheduler.wait_for_idle())
            file_events = [
                json.loads(line)
                for line in result.events_path.read_text(encoding="utf-8").splitlines()
            ]

        event_types = [event["type"] for event in file_events]
        self.assertEqual(event_types[0], "run.started")
        self.assertIn("agent.step.started", event_types)
        self.assertIn("agent.loop.thought", event_types)
        self.assertIn("agent.step.finished", event_types)
        self.assertIn("agent.answer", event_types)
        self.assertEqual(event_types[-1], "run.finished")
        self.assertEqual([event["type"] for event in seen_events], event_types)
        self.assertEqual({event.get("session_id") for event in file_events}, {"session_1"})

    def test_runtime_returns_failed_result_and_emits_failure_events(self) -> None:
        class FailingEngine:
            def execute(self, request, context, emit):
                raise RuntimeError("engine failed")

        seen_events = []
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manager = _manager(root)
            session_id = manager.new_session()
            runtime = AgentRuntime(
                engine=FailingEngine(),
                runs_dir=root / "runs",
                extra_handlers=[seen_events.append],
                session_manager=manager,
            )

            result = runtime.run(AgentRequest(goal="fail", session_id=session_id))

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error, "engine failed")
        self.assertEqual(
            [event["type"] for event in seen_events],
            ["run.started", "agent.error", "run.finished"],
        )
        self.assertEqual(seen_events[-1]["status"], "failed")

    def test_runtime_rejects_unknown_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = AgentRuntime(
                engine=AgentLoopEngine(),
                session_manager=_manager(Path(temp_dir)),
            )
            with self.assertRaisesRegex(ValueError, "session does not exist"):
                runtime.run(AgentRequest(goal="question", session_id="session_1"))

    def test_runtime_creates_run_scoped_scheduler_and_shuts_it_down(self) -> None:
        schedulers: list[AgentScheduler] = []
        worker_handles = []
        root_thread_id = threading.get_ident()

        def scheduler_factory(*, run_id: str, max_workers: int) -> AgentScheduler:
            scheduler = AgentScheduler(run_id=run_id, max_workers=max_workers)
            schedulers.append(scheduler)
            return scheduler

        class WorkerSubmittingEngine:
            def execute(self, request, context, emit):
                self.assert_root_thread = threading.get_ident()
                worker_handles.append(schedulers[-1].submit(lambda: "worker result"))
                return type("Result", (), {"answer": "root result"})()

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _manager(Path(temp_dir))
            session_id = manager.new_session()
            engine = WorkerSubmittingEngine()
            runtime = AgentRuntime(
                engine=engine,
                session_manager=manager,
                memory_scheduler=_NoopMemoryScheduler(),
                scheduler_max_workers=1,
                scheduler_factory=scheduler_factory,
            )
            result = runtime.run(
                AgentRequest(goal="run", session_id=session_id),
                run_id="run-lifecycle",
            )

        self.assertEqual(result.status, "finished")
        self.assertEqual(engine.assert_root_thread, root_thread_id)
        self.assertEqual(len(schedulers), 1)
        self.assertEqual(schedulers[0].run_id, "run-lifecycle")
        self.assertEqual(schedulers[0].max_workers, 1)
        self.assertTrue(schedulers[0].is_shutdown)
        self.assertEqual(worker_handles[0].future.result(), "worker result")

    def test_runtime_shuts_down_scheduler_when_root_fails(self) -> None:
        schedulers: list[AgentScheduler] = []

        def scheduler_factory(*, run_id: str, max_workers: int) -> AgentScheduler:
            scheduler = AgentScheduler(run_id=run_id, max_workers=max_workers)
            schedulers.append(scheduler)
            return scheduler

        class FailingEngine:
            def execute(self, request, context, emit):
                raise RuntimeError("root failed")

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = _manager(Path(temp_dir))
            session_id = manager.new_session()
            runtime = AgentRuntime(
                engine=FailingEngine(),
                session_manager=manager,
                memory_scheduler=_NoopMemoryScheduler(),
                scheduler_factory=scheduler_factory,
            )
            result = runtime.run(AgentRequest(goal="fail", session_id=session_id))

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error, "root failed")
        self.assertEqual(len(schedulers), 1)
        self.assertTrue(schedulers[0].is_shutdown)


def _manager(root: Path) -> SessionManager:
    return SessionManager(
        current_session_path=root / "session" / "current_session.json",
        memory_root=root / "storage" / "session_memory",
        trace_root=root / "trace" / "session_trace",
    )


class _NoopMemoryScheduler:
    def schedule(self, memory, *, on_failure=None):
        return False


if __name__ == "__main__":
    unittest.main()
