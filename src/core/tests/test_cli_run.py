import unittest
from unittest.mock import patch

from cli import commands
from core.ipc.protocol import make_event_push, make_result_response


class FakeStreamClient:
    messages: list[dict] = []
    instances: list["FakeStreamClient"] = []

    def __init__(self, *, host, port, read_timeout=None):
        self.host = host
        self.port = port
        self.read_timeout = read_timeout
        self.sent_requests: list[tuple[str, dict | None]] = []
        FakeStreamClient.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def send_request(self, method, params=None):
        self.sent_requests.append((method, params))
        return f"request-{len(self.sent_requests)}"

    def read_message(self):
        if not FakeStreamClient.messages:
            raise AssertionError("no fake stream messages left")
        return FakeStreamClient.messages.pop(0)


class CliRunCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeStreamClient.messages = []
        FakeStreamClient.instances = []

    def test_run_command_dispatches_to_daemon_stream(self) -> None:
        FakeStreamClient.messages = [
            make_result_response(
                "request-1",
                {
                    "subscription_id": "sub-1",
                    "topics": ["*"],
                    "run_id": None,
                    "replayed_count": 0,
                },
            ),
            make_result_response(
                "request-2",
                {
                    "run_id": "run-1",
                    "status": "started",
                },
            ),
            make_event_push(
                {
                    "type": "run.started",
                    "run_id": "run-1",
                    "goal": "hello agent",
                }
            ),
            make_event_push(
                {
                    "type": "run.finished",
                    "run_id": "run-1",
                    "status": "finished",
                }
            ),
        ]

        with (
            patch.object(commands, "ensure_daemon_running") as ensure_daemon,
            patch.object(commands, "CoreStreamClient", FakeStreamClient),
            patch.object(commands, "CliEventPrinter") as printer_class,
        ):
            printer = printer_class.return_value

            exit_code = commands.run_command(["hello", "agent"])

        self.assertEqual(exit_code, 0)
        ensure_daemon.assert_called_once()
        self.assertEqual(len(FakeStreamClient.instances), 1)
        stream = FakeStreamClient.instances[0]
        self.assertEqual(stream.sent_requests[0][0], "event.subscribe")
        self.assertEqual(stream.sent_requests[0][1]["topics"], ["*"])
        self.assertEqual(stream.sent_requests[1][0], "agent.run")
        self.assertEqual(stream.sent_requests[1][1]["goal"], "hello agent")
        printer.handle.assert_any_call(
            {
                "type": "run.started",
                "run_id": "run-1",
                "goal": "hello agent",
            }
        )
        printer.handle.assert_any_call(
            {
                "type": "run.finished",
                "run_id": "run-1",
                "status": "finished",
            }
        )


if __name__ == "__main__":
    unittest.main()
