import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.routes_agent import router
from llm.Agent.AgentRuntime import AgentResult
from schemas.llm import AgentResponse


class AgentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

    def test_ask_requires_session_id(self) -> None:
        response = self.client.post("/agent/ask", data={"message": "hello"})

        self.assertEqual(response.status_code, 422)

    def test_ask_returns_runtime_result_shape(self) -> None:
        class FakeRuntime:
            def run(self, request):
                self.request = request
                return AgentResult(
                    run_id="run-1",
                    status="finished",
                    answer="answer",
                    error=None,
                    events_path=Path("runs/run-1/events.jsonl"),
                )

        runtime = FakeRuntime()
        with patch("api.routes_agent.get_agent_runtime", return_value=runtime):
            response = self.client.post(
                "/agent/ask",
                data={"message": "hello", "session_id": "session_1"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"run_id": "run-1", "status": "finished", "message": "answer"},
        )
        self.assertEqual(runtime.request.goal, "hello")
        self.assertEqual(runtime.request.session_id, "session_1")

    def test_ask_returns_bad_request_for_invalid_session(self) -> None:
        class FakeRuntime:
            def run(self, request):
                raise ValueError("session does not exist: session_9")

        with patch("api.routes_agent.get_agent_runtime", return_value=FakeRuntime()):
            response = self.client.post(
                "/agent/ask",
                data={"message": "hello", "session_id": "session_9"},
            )

        self.assertEqual(response.status_code, 400)

    def test_agent_response_only_allows_finished_status(self) -> None:
        with self.assertRaises(ValidationError):
            AgentResponse(run_id="run-1", status="failed", message="answer")


if __name__ == "__main__":
    unittest.main()
