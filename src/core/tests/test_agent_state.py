import json
import tempfile
import unittest
from unittest.mock import patch

from llm.Agent.memory import append_plan_visualization
from llm.Agent.nodes import agent_loop_node, planner_node
from llm.Agent.state import (
    AgentState,
    MAX_PLAN_STEPS,
    PlanStep,
)


class AgentStateTest(unittest.TestCase):
    def _run_agent_loop_with_responses(self, responses: list[str]) -> AgentState:
        plan = [PlanStep(step_id="step_1", task="Validate model output").model_dump()]

        with patch("llm.Agent.nodes.agent_loop._chat_completion", side_effect=responses):
            return agent_loop_node(
                {
                    "question": "Validate model output",
                    "plan": plan,
                    "current_step_index": 0,
                    "current_step_id": "step_1",
                    "logs": [],
                }
            )

    @staticmethod
    def _loop_response(
        *,
        thought: str = "Inspect current evidence",
        decide_type: str = "finish",
        signal: str | None = None,
        no_finding: int = 0,
        answer: str = "done",
    ) -> str:
        return json.dumps(
            {
                "thought": thought,
                "decide_type": decide_type,
                "Signal": signal,
                "no_finding": no_finding,
                "tool_name": None,
                "arguments": {},
                "answer": answer,
            }
        )

    def test_overturning_signal_requests_single_full_replan(self) -> None:
        observation = json.dumps(
            {
                "tool_name": "search",
                "result": "the earlier assumption is false",
                "error": None,
            }
        )
        state: AgentState = {
            "question": "Investigate failure",
            "plan": [PlanStep(step_id="step_1", task="Check likely cause").model_dump()],
            "current_step_index": 0,
            "current_step_id": "step_1",
            "react_results": [
                {
                    "thought": "Assume the cause is configuration",
                    "decide_type": "tool_call",
                    "Signal": None,
                    "no_finding": 0,
                    "tool_name": "search",
                    "arguments": {},
                    "observation": observation,
                    "answer": "",
                }
            ],
            "logs": [],
        }

        with patch(
            "llm.Agent.nodes.agent_loop._chat_completion",
            return_value=self._loop_response(
                thought="The tool result overturns the current plan",
                decide_type="think",
                signal="overturning",
                answer="",
            ),
        ):
            result = agent_loop_node(state)

        self.assertEqual(result["agent_status"], "running")
        self.assertEqual(result["planner_mode"], "replan")
        self.assertEqual(result["replan_count"], 1)
        self.assertEqual(result["last_tool_observation"], observation)
        self.assertEqual(result["react_results"], [])

    def test_overturning_signal_fails_after_single_replan_used(self) -> None:
        state: AgentState = {
            "question": "Investigate failure",
            "plan": [PlanStep(step_id="step_1", task="Check likely cause").model_dump()],
            "current_step_index": 0,
            "current_step_id": "step_1",
            "replan_count": 1,
            "logs": [],
        }

        with patch(
            "llm.Agent.nodes.agent_loop._chat_completion",
            return_value=self._loop_response(
                decide_type="think",
                signal="overturning",
                answer="",
            ),
        ):
            result = agent_loop_node(state)

        self.assertEqual(result["agent_status"], "failed")
        self.assertIn("overturning replan more than once", result["error"])

    def test_no_finding_count_accumulates_without_trigger_before_six(self) -> None:
        state: AgentState = {
            "question": "Investigate failure",
            "plan": [PlanStep(step_id="step_1", task="Check likely cause").model_dump()],
            "current_step_index": 0,
            "current_step_id": "step_1",
            "no_finding_counts": {"step_1": 4},
            "logs": [],
        }

        with patch(
            "llm.Agent.nodes.agent_loop._chat_completion",
            return_value=self._loop_response(no_finding=1),
        ):
            result = agent_loop_node(state)

        self.assertEqual(result["agent_status"], "running")
        self.assertNotEqual(result.get("planner_mode"), "step_replan")
        self.assertEqual(result["no_finding_counts"]["step_1"], 5)

    def test_no_finding_count_triggers_step_replan_at_six(self) -> None:
        state: AgentState = {
            "question": "Investigate failure",
            "plan": [PlanStep(step_id="step_1", task="Check likely cause").model_dump()],
            "current_step_index": 0,
            "current_step_id": "step_1",
            "no_finding_counts": {"step_1": 5},
            "logs": [],
        }

        with patch(
            "llm.Agent.nodes.agent_loop._chat_completion",
            return_value=self._loop_response(no_finding=1),
        ):
            result = agent_loop_node(state)

        self.assertEqual(result["agent_status"], "running")
        self.assertEqual(result["planner_mode"], "step_replan")
        self.assertEqual(result["step_replan_count"], 1)
        self.assertEqual(result["no_finding_counts"]["step_1"], 6)

    def test_no_finding_count_resets_on_connected_thought(self) -> None:
        state: AgentState = {
            "question": "Investigate failure",
            "plan": [PlanStep(step_id="step_1", task="Check likely cause").model_dump()],
            "current_step_index": 0,
            "current_step_id": "step_1",
            "no_finding_counts": {"step_1": 5},
            "logs": [],
        }

        with patch(
            "llm.Agent.nodes.agent_loop._chat_completion",
            return_value=self._loop_response(no_finding=0),
        ):
            result = agent_loop_node(state)

        self.assertEqual(result["agent_status"], "running")
        self.assertEqual(result["no_finding_counts"]["step_1"], 0)

    def test_finding_missing_signal_fails_after_single_step_replan_used(self) -> None:
        state: AgentState = {
            "question": "Investigate failure",
            "plan": [PlanStep(step_id="step_1", task="Check likely cause").model_dump()],
            "current_step_index": 0,
            "current_step_id": "step_1",
            "step_replan_count": 1,
            "no_finding_counts": {"step_1": 5},
            "logs": [],
        }

        with patch(
            "llm.Agent.nodes.agent_loop._chat_completion",
            return_value=self._loop_response(no_finding=1),
        ):
            result = agent_loop_node(state)

        self.assertEqual(result["agent_status"], "failed")
        self.assertIn("finding_missing step replan more than once", result["error"])

    def test_planner_replan_replaces_entire_plan(self) -> None:
        response = json.dumps(
            {
                "reason": "Tool evidence overturned the first plan.",
                "steps": [
                    {"step_id": "step_1", "task": "Use overturned evidence"},
                    {"step_id": "step_2", "task": "Answer with new evidence"},
                ],
            }
        )
        state: AgentState = {
            "question": "Investigate failure",
            "planner_mode": "replan",
            "plan_revision": 1,
            "plan": [PlanStep(step_id="step_1", task="Old step").model_dump()],
            "last_tool_observation": "new evidence",
            "logs": [],
        }

        with (
            patch("llm.Agent.nodes.planner._chat_completion", return_value=response),
            patch("llm.Agent.nodes.planner.append_plan_visualization"),
        ):
            result = planner_node(state)

        self.assertEqual(result["agent_status"], "running")
        self.assertEqual(result["planner_mode"], "initial")
        self.assertEqual(result["plan_revision"], 2)
        self.assertEqual([step["task"] for step in result["plan"]], [
            "Use overturned evidence",
            "Answer with new evidence",
        ])
        self.assertEqual(result["current_step_index"], 0)
        self.assertEqual(result["current_step_id"], "step_1")

    def test_planner_payload_includes_context_memory(self) -> None:
        response = json.dumps(
            {
                "steps": [
                    {"step_id": "step_1", "task": "Use relevant memory to plan"},
                ],
            }
        )
        context_memory = [
            {
                "question": "Historical question",
                "final_answer": "Historical answer",
            }
        ]
        captured_payload = {}

        def fake_chat_completion(**kwargs):
            captured_payload.update(json.loads(kwargs["user_message"]))
            return response

        state: AgentState = {
            "question": "Plan with memory",
            "context_memory": context_memory,
            "logs": [],
        }

        with (
            patch("llm.Agent.nodes.planner._chat_completion", side_effect=fake_chat_completion),
            patch("llm.Agent.nodes.planner.append_plan_visualization"),
        ):
            result = planner_node(state)

        self.assertEqual(result["agent_status"], "running")
        self.assertEqual(captured_payload["context_memory"], context_memory)

    def test_visualization_snapshots_use_separator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            plan_path = f"{temp_dir}/plan.md"

            append_plan_visualization(
                [{"step_id": "step_1", "task": "Test plan"}],
                path=plan_path,
            )

            with open(plan_path, encoding="utf-8") as file:
                plan_content = file.read()

        self.assertTrue(plan_content.startswith("\n#$#\n"))
        self.assertIn('"task": "Test plan"', plan_content)

    def test_planner_step_replan_preserves_completed_steps(self) -> None:
        response = json.dumps(
            {
                "reason": "Current step is missing a finding.",
                "steps": [
                    {"step_id": "step_1", "task": "Replacement current step"},
                    {"step_id": "step_2", "task": "Replacement final step"},
                ],
            }
        )
        completed = PlanStep(
            step_id="step_1",
            task="Completed step",
            status="done",
            result="completed evidence",
        ).model_dump()
        state: AgentState = {
            "question": "Investigate failure",
            "planner_mode": "step_replan",
            "plan_revision": 1,
            "plan": [
                completed,
                PlanStep(step_id="step_2", task="Current bad step").model_dump(),
                PlanStep(step_id="step_3", task="Old later step").model_dump(),
            ],
            "current_step_index": 1,
            "current_step_id": "step_2",
            "no_finding_counts": {"step_2": 6},
            "logs": [],
        }

        with (
            patch("llm.Agent.nodes.planner._chat_completion", return_value=response),
            patch("llm.Agent.nodes.planner.append_plan_visualization"),
        ):
            result = planner_node(state)

        self.assertEqual(result["agent_status"], "running")
        self.assertEqual(result["plan"][0], completed)
        self.assertEqual(result["plan"][1]["step_id"], "step_2")
        self.assertEqual(result["plan"][1]["task"], "Replacement current step")
        self.assertEqual(result["plan"][2]["step_id"], "step_3")
        self.assertEqual(result["plan"][2]["task"], "Replacement final step")
        self.assertEqual(result["current_step_index"], 1)
        self.assertEqual(result["current_step_id"], "step_2")
        self.assertEqual(result["no_finding_counts"], {})

    def test_planner_still_rejects_too_many_steps(self) -> None:
        response = json.dumps(
            {
                "steps": [
                    {"step_id": f"step_{index}", "task": f"Step {index}"}
                    for index in range(1, MAX_PLAN_STEPS + 2)
                ]
            }
        )
        state: AgentState = {
            "question": "Investigate failure",
            "logs": [],
        }

        with patch("llm.Agent.nodes.planner._chat_completion", return_value=response):
            result = planner_node(state)

        self.assertEqual(result["agent_status"], "failed")
        self.assertIn(f"more than {MAX_PLAN_STEPS} steps", result["error"])



if __name__ == "__main__":
    unittest.main()
