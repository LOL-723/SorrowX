import argparse
from typing import Callable

from cli.event_rendering import CliEventPrinter
from cli.formatting import format_ping_reply
from core.ipc.client import CoreClient, CoreStreamClient, get_default_host, get_default_port
from core.ipc.daemon_process import ensure_daemon_running
from core.ipc.protocol import is_event_push, read_event_push, read_result_response


CommandHandler = Callable[[list[str]], int]
CLIENT_NAME = "sorrow-cli/0.1.0"


def ping_command(argv: list[str]) -> int:
    host = get_default_host()
    port = get_default_port()
    ensure_daemon_running(host=host, port=port)
    rpc_result = CoreClient(host=host, port=port).request(
        "core.ping",
        {"client": CLIENT_NAME},
    )
    print(format_ping_reply(host=host, port=port, rpc_result=rpc_result), flush=True)
    return 0


def shutdown_command(argv: list[str]) -> int:
    host = get_default_host()
    port = get_default_port()
    try:
        rpc_result = CoreClient(host=host, port=port).request(
            "core.shutdown",
            {"client": CLIENT_NAME},
        )
    except OSError:
        print(f"Daemon is not running at {host}:{port}.", flush=True)
        return 0
    print(
        "Daemon shutting down: "
        f"{host}:{port} uptime={_format_seconds(rpc_result.result.get('uptime_ms', 0))}",
        flush=True,
    )
    return 0


def run_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="sorrow run")
    parser.add_argument("goal", nargs="+")
    args = parser.parse_args(argv)
    goal = " ".join(args.goal).strip()
    host = get_default_host()
    port = get_default_port()
    printer = CliEventPrinter()
    try:
        ensure_daemon_running(host=host, port=port)
        return _run_agent_stream(host=host, port=port, goal=goal, printer=printer)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Agent run failed: {exc}", flush=True)
        return 1


def _run_agent_stream(
    *,
    host: str,
    port: int,
    goal: str,
    printer: CliEventPrinter,
) -> int:
    pending_events: list[dict[str, object]] = []
    run_id: str | None = None

    with CoreStreamClient(host=host, port=port, read_timeout=None) as client:
        subscribe_request_id = client.send_request(
            "event.subscribe",
            {
                "client": CLIENT_NAME,
                "topics": ["*"],
            },
        )
        _read_result_ignoring_events(
            client,
            expected_id=subscribe_request_id,
            pending_events=pending_events,
        )

        request_id = client.send_request(
            "agent.run",
            {
                "client": CLIENT_NAME,
                "goal": goal,
            },
        )

        while run_id is None:
            message = client.read_message()
            if is_event_push(message):
                pending_events.append(read_event_push(message))
                continue

            result = read_result_response(message, expected_id=request_id)
            if not isinstance(result, dict):
                raise ValueError("agent.run result must be an object")
            raw_run_id = result.get("run_id")
            if not isinstance(raw_run_id, str) or not raw_run_id:
                raise ValueError("agent.run result must include run_id")
            run_id = raw_run_id

        for event in pending_events:
            exit_code = _handle_run_event(event, run_id=run_id, printer=printer)
            if exit_code is not None:
                return exit_code

        while True:
            message = client.read_message()
            if not is_event_push(message):
                continue
            event = read_event_push(message)
            exit_code = _handle_run_event(event, run_id=run_id, printer=printer)
            if exit_code is not None:
                return exit_code


def _read_result_ignoring_events(
    client: CoreStreamClient,
    *,
    expected_id: str | int,
    pending_events: list[dict[str, object]],
) -> dict[str, object]:
    while True:
        message = client.read_message()
        if is_event_push(message):
            pending_events.append(read_event_push(message))
            continue
        result = read_result_response(message, expected_id=expected_id)
        if not isinstance(result, dict):
            raise ValueError("RPC result must be an object")
        return result


def _handle_run_event(
    event: dict[str, object],
    *,
    run_id: str,
    printer: CliEventPrinter,
) -> int | None:
    if event.get("run_id") != run_id:
        return None
    printer.handle(event)
    if event.get("type") != "run.finished":
        return None
    return 0 if event.get("status") == "finished" else 1


def _format_seconds(uptime_ms: object) -> str:
    try:
        return f"{float(uptime_ms) / 1000:.1f}s"
    except (TypeError, ValueError):
        return "unknown"


COMMANDS: dict[str, CommandHandler] = {
    "ping": ping_command,
    "run": run_command,
    "shutdown": shutdown_command,
}
