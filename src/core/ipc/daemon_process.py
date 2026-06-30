import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ipc.client import CoreClient, get_default_host, get_default_port
from set.runtime import EXPECTED_PYTHON, PROJECT_ROOT


CORE_ROOT = PROJECT_ROOT / "src" / "core"
DAEMON_STORAGE_DIR = CORE_ROOT / "storage" / "daemon"
DAEMON_PID_PATH = DAEMON_STORAGE_DIR / "sorrow-core.pid"
DAEMON_LOG_PATH = DAEMON_STORAGE_DIR / "sorrow-core.log"
STARTUP_TIMEOUT_SECONDS = 8.0


def ensure_daemon_running(
    *,
    host: str | None = None,
    port: int | None = None,
    timeout: float = STARTUP_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    host = host or get_default_host()
    port = port if port is not None else get_default_port()
    existing = _try_ping(host=host, port=port)
    if existing is not None:
        return existing

    process = _start_daemon_process(host=host, port=port)
    DAEMON_PID_PATH.write_text(str(process.pid), encoding="utf-8")

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"daemon exited during startup with code {process.returncode}; "
                f"see {DAEMON_LOG_PATH}"
            )
        ready = _try_ping(host=host, port=port)
        if ready is not None:
            return ready
        try:
            time.sleep(0.1)
        except Exception as exc:
            last_error = exc

    reason = f": {last_error}" if last_error else ""
    raise TimeoutError(f"daemon did not become ready within {timeout:.1f}s{reason}")


def _try_ping(*, host: str, port: int) -> dict[str, Any] | None:
    try:
        return CoreClient(host=host, port=port, timeout=0.5).request("core.ping").result
    except OSError:
        return None
    except TimeoutError:
        return None
    except Exception:
        return None


def _start_daemon_process(*, host: str, port: int) -> subprocess.Popen[Any]:
    DAEMON_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    python_exe = EXPECTED_PYTHON if EXPECTED_PYTHON.exists() else Path(sys.executable)
    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_pythonpath(CORE_ROOT, env.get("PYTHONPATH", ""))

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    with DAEMON_LOG_PATH.open("a", encoding="utf-8") as log_file:
        return subprocess.Popen(
            [
                str(python_exe),
                "-m",
                "daemon.main",
                "--host",
                host,
                "--port",
                str(port),
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            close_fds=True,
        )


def _prepend_pythonpath(path: Path, existing: str) -> str:
    path_text = str(path)
    if not existing:
        return path_text
    parts = existing.split(os.pathsep)
    if path_text in parts:
        return existing
    return os.pathsep.join([path_text, *parts])
