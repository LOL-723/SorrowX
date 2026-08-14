import unittest
from pathlib import Path
from unittest.mock import patch

from llm.Agent.AgentEngine import (
    MAX_AGENT_NODE_ITERATIONS,
    AgentGraphEngine,
    AgentLoopEngine,
    _summarize_agent_answer,
)
from llm.Agent.AgentRuntime import AgentRequest, AgentRunContext, get_agent_runtime


class AgentGraphEngineTests(unittest.TestCase):
    def test_graph_matches_legacy_for_multi_step_run(self) -> None:
        legacy = self._run_success(AgentLoopEngine)
        graph = self._run_success(AgentGraphEngine)

        self.assertEqual(graph["answer"], legacy["answer"])
        self.assertEqual(graph["events"], legacy["events"])
        self.assertEqual(graph["calls"], legacy["calls"])
        self.assertEqual(graph["calls"], {"planner": 1, "select": 3, "loop": 2, "finalize": 1})
        self.assertEqual(graph["context_memory"], ["remembered context"])
        self.assertEqual(graph["tool_context"].run_id, "run_1")
        self.assertEqual(graph["tool_context"].session_id, "session_1")

    def test_graph_matches_legacy_replan_routes(self) -> None:
        for planner_mode in ("replan", "step_replan"):
            with self.subTest(planner_mode=planner_mode):
                legacy = self._run_replan(AgentLoopEngine, planner_mode)
                graph = self._run_replan(AgentGraphEngine, planner_mode)

                self.assertEqual(graph["answer"], legacy["answer"])
                self.assertEqual(graph["events"], legacy["events"])
                self.assertEqual(graph["planner_modes"], legacy["planner_modes"])
                self.assertEqual(graph["planner_modes"], [None, planner_mode])
                self.assertEqual(graph["calls"], {"planner": 2, "select": 3, "loop": 2, "finalize": 1})

    def test_graph_matches_legacy_planner_failure(self) -> None:
        legacy = self._run_planner_failure(AgentLoopEngine)
        graph = self._run_planner_failure(AgentGraphEngine)

        self.assertEqual(graph, legacy)
        self.assertEqual(graph["error"], "planner failed")
        self.assertEqual(
            [event["type"] for event in graph["events"]],
            ["agent.node.started", "agent.log", "agent.node.finished"],
        )

    def test_graph_matches_legacy_agent_loop_failure(self) -> None:
        legacy = self._run_agent_loop_failure(AgentLoopEngine)
        graph = self._run_agent_loop_failure(AgentGraphEngine)

        self.assertEqual(graph, legacy)
        self.assertEqual(graph["error"], "react failed")
        self.assertEqual(
            [event["type"] for event in graph["events"]],
            [
                "agent.node.started",
                "agent.log",
                "agent.node.finished",
                "agent.node.started",
                "agent.log",
                "agent.node.finished",
                "agent.step.started",
                "agent.node.started",
                "agent.log",
                "agent.node.finished",
            ],
        )

    def test_graph_uses_legacy_iteration_limit_before_langgraph_limit(self) -> None:
        legacy = self._run_until_iteration_limit(AgentLoopEngine)
        graph = self._run_until_iteration_limit(AgentGraphEngine)

        self.assertEqual(graph["error"], legacy["error"])
        self.assertEqual(graph["error"], "agent exceeded graph iteration limit")
        self.assertEqual(graph["planner_calls"], MAX_AGENT_NODE_ITERATIONS)
        self.assertEqual(graph["events"], legacy["events"])

    def test_graph_is_compiled_once_per_engine_instance(self) -> None:
        engine = AgentGraphEngine()
        compiled_graph = engine._compiled_graph

        self._run_success_with_engine(engine)
        self._run_success_with_engine(engine)

        self.assertIs(engine._compiled_graph, compiled_graph)

    def test_default_runtime_uses_graph_engine(self) -> None:
        with patch("llm.Agent.AgentRuntime._default_runtime", None):
            runtime = get_agent_runtime()
        self.assertIsInstance(runtime.engine, AgentGraphEngine)

    def test_final_summary_preserves_json_and_raw_fallback_behavior(self) -> None:
        state = {"question": "question", "plan": [], "step_results": [], "failed_tools": []}
        with patch(
            "llm.Agent.AgentEngine._chat_completion",
            return_value='{"final_answer":"structured answer"}',
        ):
            self.assertEqual(_summarize_agent_answer(state), "structured answer")
        with patch("llm.Agent.AgentEngine._chat_completion", return_value="raw answer"):
            self.assertEqual(_summarize_agent_answer(state), "raw answer")

    def _run_success(self, engine_type):
        return self._run_success_with_engine(engine_type())

    def _run_success_with_engine(self, engine):
        calls = {"planner": 0, "select": 0, "loop": 0, "finalize": 0}
        seen_context_memory = []
        seen_tool_context = []
        events = []

        def fake_planner(state):
            calls["planner"] += 1
            seen_context_memory.append(state.get("context_memory"))
            return {
                "plan": [
                    self._step("step_1", "first task"),
                    self._step("step_2", "second task"),
                ],
                "plan_revision": 1,
                "plan_updates": [],
                "planner_mode": "initial",
                "agent_status": "running",
                "logs": state.get("logs", []) + [{"node": "planner_node", "message": "plan created"}],
            }

        def fake_select(state):
            calls["select"] += 1
            for index, step in enumerate(state["plan"]):
                if step["status"] == "pending":
                    return {
                        "current_step_index": index,
                        "current_step_id": step["step_id"],
                        "should_continue_next": "continue",
                        "agent_status": "running",
                        "logs": state.get("logs", []) + [{"node": "select_next_step_node", "message": "selected"}],
                    }
            return {
                "current_step_index": len(state["plan"]),
                "current_step_id": None,
                "should_continue_next": "finish",
                "agent_status": "running",
                "logs": state.get("logs", []) + [{"node": "select_next_step_node", "message": "done"}],
            }

        def fake_loop(state):
            calls["loop"] += 1
            seen_tool_context.append(state.get("tool_context"))
            plan = [dict(step) for step in state["plan"]]
            index = state["current_step_index"]
            plan[index] = dict(plan[index], status="done", result=f"result {index + 1}")
            return {
                "plan": plan,
                "step_results": state.get("step_results", [])
                + [{"step_id": plan[index]["step_id"], "task": plan[index]["task"], "result": plan[index]["result"]}],
                "agent_status": "running",
                "logs": state.get("logs", []) + [{"node": "agent_loop_node", "message": "completed"}],
            }

        def fake_finalize(_state):
            calls["finalize"] += 1
            return "final answer"

        with (
            patch("llm.Agent.AgentEngine.planner_node", side_effect=fake_planner),
            patch("llm.Agent.AgentEngine.select_next_step_node", side_effect=fake_select),
            patch("llm.Agent.AgentEngine.agent_loop_node", side_effect=fake_loop),
            patch("llm.Agent.AgentEngine._summarize_agent_answer", side_effect=fake_finalize),
        ):
            result = engine.execute(self._request(), self._context(), events.append)

        return {
            "answer": result.answer,
            "events": events,
            "calls": calls,
            "context_memory": seen_context_memory,
            "tool_context": seen_tool_context[0],
        }

    def _run_replan(self, engine_type, planner_mode):
        calls = {"planner": 0, "select": 0, "loop": 0, "finalize": 0}
        planner_modes = []
        events = []

        def fake_planner(state):
            calls["planner"] += 1
            planner_modes.append(state.get("planner_mode"))
            return {
                "plan": [self._step("step_1", "revised task")],
                "plan_revision": calls["planner"],
                "planner_mode": "initial",
                "agent_status": "running",
                "logs": state.get("logs", []) + [{"node": "planner_node", "message": "planned"}],
            }

        def fake_select(state):
            calls["select"] += 1
            step = state["plan"][0]
            if step["status"] == "done":
                return {
                    "current_step_index": 1,
                    "current_step_id": None,
                    "should_continue_next": "finish",
                    "agent_status": "running",
                    "logs": state.get("logs", []) + [{"node": "select_next_step_node", "message": "done"}],
                }
            return {
                "current_step_index": 0,
                "current_step_id": "step_1",
                "should_continue_next": "continue",
                "agent_status": "running",
                "logs": state.get("logs", []) + [{"node": "select_next_step_node", "message": "selected"}],
            }

        def fake_loop(state):
            calls["loop"] += 1
            if calls["loop"] == 1:
                return {
                    "planner_mode": planner_mode,
                    "agent_status": "running",
                    "logs": state.get("logs", []) + [{"node": "agent_loop_node", "message": "request replan"}],
                }
            plan = [dict(state["plan"][0], status="done", result="revised result")]
            return {
                "plan": plan,
                "step_results": [{"step_id": "step_1", "task": "revised task", "result": "revised result"}],
                "agent_status": "running",
                "logs": state.get("logs", []) + [{"node": "agent_loop_node", "message": "completed"}],
            }

        def fake_finalize(_state):
            calls["finalize"] += 1
            return "replanned answer"

        with (
            patch("llm.Agent.AgentEngine.planner_node", side_effect=fake_planner),
            patch("llm.Agent.AgentEngine.select_next_step_node", side_effect=fake_select),
            patch("llm.Agent.AgentEngine.agent_loop_node", side_effect=fake_loop),
            patch("llm.Agent.AgentEngine._summarize_agent_answer", side_effect=fake_finalize),
        ):
            result = engine_type().execute(self._request(), self._context(), events.append)

        return {
            "answer": result.answer,
            "events": events,
            "calls": calls,
            "planner_modes": planner_modes,
        }

    def _run_planner_failure(self, engine_type):
        events = []

        def failing_planner(state):
            return {
                "agent_status": "failed",
                "phase": "failed",
                "error": "planner failed",
                "logs": state.get("logs", []) + [{"node": "planner_node", "message": "planner failed"}],
            }

        with patch("llm.Agent.AgentEngine.planner_node", side_effect=failing_planner):
            with self.assertRaises(RuntimeError) as raised:
                engine_type().execute(self._request(), self._context(), events.append)
        return {"error": str(raised.exception), "events": events}

    def _run_agent_loop_failure(self, engine_type):
        events = []

        def fake_planner(state):
            return {
                "plan": [self._step("step_1", "failing task")],
                "planner_mode": "initial",
                "agent_status": "running",
                "logs": state.get("logs", []) + [{"node": "planner_node", "message": "planned"}],
            }

        def fake_select(state):
            return {
                "current_step_index": 0,
                "current_step_id": "step_1",
                "should_continue_next": "continue",
                "agent_status": "running",
                "logs": state.get("logs", []) + [{"node": "select_next_step_node", "message": "selected"}],
            }

        def failing_loop(state):
            return {
                "agent_status": "failed",
                "phase": "failed",
                "error": "react failed",
                "logs": state.get("logs", []) + [{"node": "agent_loop_node", "message": "react failed"}],
            }

        with (
            patch("llm.Agent.AgentEngine.planner_node", side_effect=fake_planner),
            patch("llm.Agent.AgentEngine.select_next_step_node", side_effect=fake_select),
            patch("llm.Agent.AgentEngine.agent_loop_node", side_effect=failing_loop),
        ):
            with self.assertRaises(RuntimeError) as raised:
                engine_type().execute(self._request(), self._context(), events.append)
        return {"error": str(raised.exception), "events": events}

    def _run_until_iteration_limit(self, engine_type):
        events = []
        planner_calls = 0

        def endless_planner(state):
            nonlocal planner_calls
            planner_calls += 1
            return {
                "plan": [self._step("step_1", "never selected")],
                "planner_mode": "replan",
                "agent_status": "running",
                "logs": state.get("logs", []) + [{"node": "planner_node", "message": "replan"}],
            }

        with patch("llm.Agent.AgentEngine.planner_node", side_effect=endless_planner):
            with self.assertRaises(RuntimeError) as raised:
                engine_type().execute(self._request(), self._context(), events.append)
        return {
            "error": str(raised.exception),
            "events": events,
            "planner_calls": planner_calls,
        }

    @staticmethod
    def _step(step_id, task):
        return {
            "step_id": step_id,
            "task": task,
            "status": "pending",
            "result": None,
            "retry_count": 0,
        }

    @staticmethod
    def _request():
        return AgentRequest(goal="test goal", session_id="session_1")

    @staticmethod
    def _context():
        return AgentRunContext(
            run_id="run_1",
            session_id="session_1",
            context_memory_path=Path("memory.jsonl"),
            context_memory="remembered context",
            workspace_root=Path.cwd(),
        )


if __name__ == "__main__":
    unittest.main()
