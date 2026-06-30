import os
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

from ipc.client import CoreClient


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
