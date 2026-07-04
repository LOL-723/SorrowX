import asyncio
import contextlib
import os
import socket
import subprocess
import sys
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ipc.client import CoreClient
from ipc.protocol import (
    decode_message,
    encode_message,
    make_request,
    read_event_push,
    read_result_response,
)


CORE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CORE_ROOT.parents[1]


class DaemonIpcTests(unittest.TestCase):
    def test_daemon_replies_to_ping_and_shutdown(self) -> None:
        port = _free_port()
        process = _start_daemon(port)
        try:
            ping = _wait_for_ping(port)
            self.assertEqual(ping.result["server"], "sorrow-core")
            self.assertEqual(ping.result["port"], port)
            self.assertGreaterEqual(ping.elapsed_ms, 0)

            shutdown = CoreClient(port=port, timeout=2).request("core.shutdown")
            self.assertEqual(shutdown.result["status"], "shutting_down")
            process.wait(timeout=5)
            self.assertEqual(process.returncode, 0)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)


class DaemonAgentRunIpcTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_run_returns_run_id_then_pushes_events(self) -> None:
        port = _free_port()

        class FakeAgentRuner:
            def run(self, goal, *, run_id=None, event_bus=None):
                event_bus.publish(
                    {
                        "type": "run.started",
                        "run_id": run_id,
                        "goal": goal,
                    }
                )
                event_bus.publish(
                    {
                        "type": "run.finished",
                        "run_id": run_id,
                        "status": "finished",
                        "answer": "ok",
                        "error": None,
                    }
                )

        from daemon.main import run_daemon

        fake_agent_runner_module = types.ModuleType("llm.Agent.AgentRuner")
        fake_agent_runner_module.AgentRuner = FakeAgentRuner
        fake_agent_runner_module.new_run_id = lambda: "run-test"
        with patch.dict(
            sys.modules,
            {"llm.Agent.AgentRuner": fake_agent_runner_module},
        ):
            daemon_task = asyncio.create_task(run_daemon(port=port))
            try:
                await asyncio.to_thread(_wait_for_ping, port)
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(
                    encode_message(
                        make_request(
                            "event.subscribe",
                            {
                                "client": "test-client",
                                "topics": ["*"],
                            },
                            request_id="request-subscribe",
                        )
                    )
                )
                await writer.drain()

                subscription = read_result_response(
                    decode_message(await reader.readline()),
                    expected_id="request-subscribe",
                )
                self.assertEqual(subscription["topics"], ["*"])

                writer.write(
                    encode_message(
                        make_request(
                            "agent.run",
                            {
                                "client": "test-client",
                                "goal": "hello agent",
                            },
                            request_id="request-run",
                        )
                    )
                )
                await writer.drain()

                response = decode_message(await reader.readline())
                result = read_result_response(response, expected_id="request-run")
                self.assertEqual(result["run_id"], "run-test")

                started = read_event_push(decode_message(await reader.readline()))
                finished = read_event_push(decode_message(await reader.readline()))
                self.assertEqual(started["type"], "run.started")
                self.assertEqual(started["run_id"], "run-test")
                self.assertEqual(finished["type"], "run.finished")
                self.assertEqual(finished["status"], "finished")

                writer.write(
                    encode_message(
                        make_request("core.shutdown", request_id="request-shutdown")
                    )
                )
                await writer.drain()
                shutdown = read_result_response(
                    decode_message(await reader.readline()),
                    expected_id="request-shutdown",
                )
                self.assertEqual(shutdown["status"], "shutting_down")
                writer.close()
                await writer.wait_closed()
                await asyncio.wait_for(daemon_task, timeout=5)
            finally:
                if not daemon_task.done():
                    daemon_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await daemon_task


def _start_daemon(port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CORE_ROOT)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "daemon.main",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_for_ping(port: int):
    deadline = time.monotonic() + 5
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return CoreClient(port=port, timeout=0.5).request("core.ping")
        except Exception as exc:
            last_error = exc
            time.sleep(0.05)
    raise AssertionError(f"daemon did not reply to ping: {last_error}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    unittest.main()
