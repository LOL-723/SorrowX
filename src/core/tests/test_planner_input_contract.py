import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from llm.Agent.memory import ContextMemory, OneRunMemory
from llm.Agent.nodes.planner import _parse_agent_plan, _planner_payload, planner_node
from llm.Agent.nodes.universal import _available_tool_summaries, _available_tools
from llm.Agent.prompt import AGENT_LOOP_PROMPT, PLANNER_PROMPT
from llm.Agent.state import FINDING_MISSING_THRESHOLD


class PlannerInputContractTests(unittest.TestCase):
    def test_initial_payload_uses_summary_and_has_no_replan_fields(self) -> None:
        payload = _planner_payload(
            state={"context_memory_summary": "摘要" * 200},
            question="question",
            planner_mode="initial",
        )

        self.assertEqual(len(payload["context_memory_summary"]), 300)
        self.assertNotIn("context_memory", payload)
        self.assertNotIn("replan_signal", payload)
        self.assertNotIn("no_finding_count", payload)

    def test_overturning_payload_has_explicit_signal(self) -> None:
        payload = _planner_payload(
            state={
                "plan": [self._step("step_1", "old")],
                "step_results": [{"step_id": "step_0", "result": "done"}],
                "last_tool_observation": "new evidence",
                "replan_context": {
                    "signal": "overturning",
                    "react_results": [{"observation": "new evidence"}],
                },
            },
            question="question",
            planner_mode="replan",
        )

        self.assertEqual(payload["replan_signal"], "overturning")
        self.assertEqual(payload["last_tool_observation"], "new evidence")
        self.assertNotIn("no_finding_count", payload)

    def test_finding_missing_payload_has_explicit_signal_without_count(self) -> None:
        payload = _planner_payload(
            state={
                "plan": [self._step("step_1", "current")],
                "current_step_index": 0,
                "current_step_id": "step_1",
                "replan_context": {
                    "signal": "finding_missing",
                    "current_step": {"step_id": "step_1", "task": "current"},
                    "react_results": [{"thought": "new direction"}],
                },
            },
            question="question",
            planner_mode="step_replan",
        )

        self.assertEqual(payload["replan_signal"], "finding_missing")
        self.assertNotIn("no_finding_count", payload)

    def test_invalid_mode_signal_pair_fails_before_model_call(self) -> None:
        with patch("llm.Agent.nodes.planner._chat_completion") as completion:
            update = planner_node(
                {
                    "question": "question",
                    "planner_mode": "replan",
                    "plan": [self._step("step_1", "old")],
                    "replan_context": {"signal": "finding_missing"},
                }
            )

        completion.assert_not_called()
        self.assertEqual(update["agent_status"], "failed")
        self.assertIn("requires replan_signal overturning", update["error"])

    def test_planner_tools_are_name_and_description_only(self) -> None:
        summaries = _available_tool_summaries()

        self.assertTrue(summaries)
        self.assertTrue(all(set(item) == {"name", "description"} for item in summaries))
        self.assertIn("input_schema", _available_tools()[0])

    def test_prompt_is_compact_and_uses_shared_threshold(self) -> None:
        self.assertLessEqual(len(PLANNER_PROMPT), 2_000)
        self.assertIn("stage-level cognitive or work objective", PLANNER_PROMPT)
        self.assertIn("replan_signal", PLANNER_PROMPT)
        self.assertIn(f"count reaches {FINDING_MISSING_THRESHOLD}", AGENT_LOOP_PROMPT)

    def test_plan_reason_remains_supported(self) -> None:
        plan = _parse_agent_plan(
            '{"reason":"new evidence","steps":[{"step_id":"step_1","task":"goal"}]}'
        )

        self.assertEqual(plan.reason, "new evidence")

    @staticmethod
    def _step(step_id: str, task: str) -> dict[str, object]:
        return {
            "step_id": step_id,
            "task": task,
            "status": "pending",
            "result": None,
            "retry_count": 0,
        }


class PlannerMemoryContractTests(unittest.TestCase):
    def test_valid_summary_excludes_uncovered_raw_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ContextMemory(Path(temp_dir) / "context_memory.jsonl")
            memory.remember(question="covered", final_answer="covered answer")
            memory.refresh_rolling_summary(
                summarize=lambda _previous, _records: "stable summary"
            )
            memory.remember(question="uncovered", final_answer="raw fallback answer")

            self.assertEqual(memory.load_valid_summary(), "stable summary")
            self.assertIn("raw fallback answer", memory.load_context())
            self.assertNotIn("raw fallback answer", memory.load_valid_summary())

    def test_missing_summary_does_not_fall_back_to_raw_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = ContextMemory(Path(temp_dir) / "context_memory.jsonl")
            memory.remember(question="question", final_answer="raw answer")

            self.assertEqual(memory.load_valid_summary(), "")
            self.assertIn("raw answer", memory.load_context())


class FindingMissingThresholdTests(unittest.TestCase):
    def test_three_consecutive_missing_findings_trigger_replan(self) -> None:
        memory = OneRunMemory(OneRunMemory.initial_state(question="question"))

        self.assertIsNone(memory.update_no_finding(step_id="step_1", no_finding=1))
        self.assertIsNone(memory.update_no_finding(step_id="step_1", no_finding=1))
        self.assertEqual(
            memory.update_no_finding(step_id="step_1", no_finding=1),
            "finding_missing",
        )

    def test_related_turn_resets_missing_finding_count(self) -> None:
        memory = OneRunMemory(OneRunMemory.initial_state(question="question"))

        self.assertIsNone(memory.update_no_finding(step_id="step_1", no_finding=1))
        self.assertIsNone(memory.update_no_finding(step_id="step_1", no_finding=1))
        self.assertIsNone(memory.update_no_finding(step_id="step_1", no_finding=0))
        self.assertIsNone(memory.update_no_finding(step_id="step_1", no_finding=1))
        self.assertIsNone(memory.update_no_finding(step_id="step_1", no_finding=1))
        self.assertEqual(
            memory.update_no_finding(step_id="step_1", no_finding=1),
            "finding_missing",
        )


if __name__ == "__main__":
    unittest.main()
