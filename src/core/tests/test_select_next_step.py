import unittest

from llm.Agent.nodes.select_next_step import select_next_step_node


class SelectNextStepTest(unittest.TestCase):
    def test_selects_first_pending_step_without_llm_decision(self) -> None:
        state = {
            "plan": [
                {
                    "step_id": "step_1",
                    "task": "done step",
                    "status": "done",
                    "result": "done",
                    "retry_count": 0,
                },
                {
                    "step_id": "step_2",
                    "task": "next step",
                    "status": "pending",
                    "result": None,
                    "retry_count": 0,
                },
                {
                    "step_id": "step_3",
                    "task": "later step",
                    "status": "pending",
                    "result": None,
                    "retry_count": 0,
                },
            ],
            "logs": [],
        }

        update = select_next_step_node(state)

        self.assertEqual(update["should_continue_next"], "continue")
        self.assertEqual(update["current_step_index"], 1)
        self.assertEqual(update["current_step_id"], "step_2")
        self.assertEqual(update["agent_status"], "running")
        self.assertEqual(update["phase"], "selecting_step")
        self.assertEqual(update["logs"][-1]["message"], "next pending step selected")

    def test_finishes_when_no_pending_step_exists(self) -> None:
        state = {
            "plan": [
                {
                    "step_id": "step_1",
                    "task": "done step",
                    "status": "done",
                    "result": "done",
                    "retry_count": 0,
                }
            ],
            "logs": [],
        }

        update = select_next_step_node(state)

        self.assertEqual(update["should_continue_next"], "finish")
        self.assertEqual(update["current_step_index"], 1)
        self.assertIsNone(update["current_step_id"])
        self.assertEqual(update["logs"][-1]["message"], "no pending step found")


if __name__ == "__main__":
    unittest.main()
