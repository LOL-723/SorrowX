import json
import tempfile
from threading import Event
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict

from llm.Agent.nodes.agent_loop import _observation_recoverable
from llm.Agent.nodes.universal import _available_tools
from llm.Agent.tools import BUILTIN_TOOL_REGISTRY
from llm.Agent.tools.contracts import FILESYSTEM_READ, ToolContext, ToolSpec
from llm.Agent.tools.executor import ToolExecutor
from llm.Agent.tools.registry import ToolRegistry


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


class ToolGovernanceTests(unittest.TestCase):
    def test_builtin_registry_has_all_tools_and_model_schemas(self) -> None:
        self.assertEqual(
            set(BUILTIN_TOOL_REGISTRY.names()),
            {
                "get_current_time", "calculate_expression", "get_today_weather",
                "list_dir", "read_file", "write_file", "run_tests",
            },
        )
        tools = {item["name"]: item for item in _available_tools()}
        self.assertIn("properties", tools["read_file"]["input_schema"])
        self.assertIn("filesystem.read", tools["read_file"]["permissions"])
        self.assertTrue(tools["write_file"]["side_effect"])

    def test_invalid_arguments_do_not_call_handler(self) -> None:
        called = []
        executor = ToolExecutor(ToolRegistry([
            ToolSpec("test", "test", _Input, frozenset(), False, None, lambda **_: called.append(True)),
        ]))
        result = executor.execute("test", {"unexpected": "x"}, self._context())
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, "invalid_arguments")
        self.assertEqual(called, [])
        self.assertFalse(result.recoverable)

    def test_permission_denied_and_unknown_tool_do_not_call_handler(self) -> None:
        called = []
        executor = ToolExecutor(ToolRegistry([
            ToolSpec("test", "test", _Input, frozenset({FILESYSTEM_READ}), False, None, lambda **_: called.append(True)),
        ]))
        denied = executor.execute("test", {"value": "x"}, self._context())
        unknown = executor.execute("absent", {}, self._context())
        self.assertEqual(denied.error.code, "permission_denied")
        self.assertEqual(unknown.error.code, "unknown_tool")
        self.assertEqual(called, [])

    def test_context_workspace_root_is_injected_and_business_errors_are_recoverable(self) -> None:
        seen = []
        def handler(value: str, workspace_root: Path):
            seen.append(workspace_root)
            return {"error": "temporary tool failure", "value": value}
        executor = ToolExecutor(ToolRegistry([
            ToolSpec("test", "test", _Input, frozenset(), False, None, handler),
        ]))
        with tempfile.TemporaryDirectory() as temp_dir:
            context = ToolContext("run_1", "session_1", Path(temp_dir), frozenset())
            result = executor.execute("test", {"value": "x"}, context)
        self.assertEqual(seen, [Path(temp_dir)])
        self.assertEqual(result.error.code, "tool_error")
        self.assertTrue(result.recoverable)

    def test_audit_records_only_argument_keys(self) -> None:
        events = []
        class Recorder:
            def record(self, *args, **kwargs):
                events.append((args, kwargs))
        executor = ToolExecutor(ToolRegistry([
            ToolSpec("test", "test", _Input, frozenset(), False, None, lambda **_: {"content": "result"}),
        ]))
        with patch("llm.Agent.tools.executor.get_trace_recorder", return_value=Recorder()):
            executor.execute("test", {"value": "secret source content"}, self._context())
        payload = events[0][0][2]
        serialized = json.dumps(payload)
        self.assertEqual(payload["argument_keys"], ["value"])
        self.assertNotIn("secret source content", serialized)
        self.assertNotIn("result", serialized)

    def test_spec_timeout_stops_waiting_and_uses_canonical_error(self) -> None:
        def slow_handler(**_):
            Event().wait(1.1)
            return {"content": "too late"}

        executor = ToolExecutor(ToolRegistry([
            ToolSpec("slow", "slow", _Input, frozenset(), False, 1, slow_handler),
        ]))
        result = executor.execute("slow", {"value": "x"}, self._context())
        self.assertFalse(result.ok)
        self.assertEqual(result.error.code, "timeout")
        self.assertTrue(result.recoverable)

    def test_audit_captures_contract_metadata_without_argument_values(self) -> None:
        events = []
        class Recorder:
            def record(self, *args, **kwargs):
                events.append((args, kwargs))
        executor = ToolExecutor(ToolRegistry([
            ToolSpec("test", "test", _Input, frozenset({FILESYSTEM_READ}), True, 3, lambda **_: {}),
        ]))
        with patch("llm.Agent.tools.executor.get_trace_recorder", return_value=Recorder()):
            executor.execute("test", {"value": "secret"}, ToolContext("run_1", "session_1", Path.cwd(), frozenset({FILESYSTEM_READ})))
        payload = events[0][0][2]
        self.assertTrue(payload["side_effect"])
        self.assertEqual(payload["required_permissions"], ["filesystem.read"])
        self.assertEqual(payload["timeout_seconds"], 3)
        self.assertNotIn("secret", json.dumps(payload))

    def test_nonrecoverable_observation_is_not_subagent_eligible(self) -> None:
        self.assertFalse(_observation_recoverable('{"error":"denied","recoverable":false}'))
        self.assertTrue(_observation_recoverable('{"error":"failed","recoverable":true}'))

    @staticmethod
    def _context() -> ToolContext:
        return ToolContext("run_1", "session_1", Path.cwd(), frozenset())


if __name__ == "__main__":
    unittest.main()
