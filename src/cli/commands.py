import argparse
from typing import Callable

from cli.event_rendering import CliEventPrinter
from cli.formatting import format_ping_reply
from core.session.manager import get_session_manager
from core.trace.formatting import format_trace_entry, format_trace_summary
from core.trace.reader import list_traces, read_trace
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
        session_id = get_session_manager().ensure_current_session()
        ensure_daemon_running(host=host, port=port)
        return _run_agent_stream(
            host=host,
            port=port,
            goal=goal,
            session_id=session_id,
            printer=printer,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Agent run failed: {exc}", flush=True)
        return 1


def trace_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="sorrow trace")
    subparsers = parser.add_subparsers(dest="action")
    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("run_id")
    args = parser.parse_args(argv)

    session_id = get_session_manager().ensure_current_session()
    if args.action == "show":
        return _trace_show_command(args.run_id, session_id=session_id)
    return _trace_list_command(session_id=session_id)


def _trace_list_command(*, session_id: str) -> int:
    records_dir = get_session_manager().trace_dir(session_id)
    summaries = list_traces(records_dir=records_dir)
    if not summaries:
        print("No trace records found.", flush=True)
        return 0
    for summary in summaries:
        print(format_trace_summary(summary), flush=True)
    return 0


def _trace_show_command(run_id: str, *, session_id: str) -> int:
    records_dir = get_session_manager().trace_dir(session_id)
    try:
        entries = read_trace(run_id, records_dir=records_dir)
    except FileNotFoundError as exc:
        print(str(exc), flush=True)
        return 1
    except ValueError as exc:
        print(f"Invalid run_id: {exc}", flush=True)
        return 1
    for entry in entries:
        print(format_trace_entry(entry), flush=True)
    return 0


def session_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="sorrow session")
    subparsers = parser.add_subparsers(dest="action")
    subparsers.add_parser("new")
    switch_parser = subparsers.add_parser("switch")
    switch_parser.add_argument("session_id")
    del_parser = subparsers.add_parser("del")
    del_parser.add_argument("session_id")
    subparsers.add_parser("list")
    subparsers.add_parser("current")
    args = parser.parse_args(argv)

    manager = get_session_manager()
    if args.action == "new":
        session_id = manager.new_session()
        print(f"Current session: {session_id}", flush=True)
        return 0
    if args.action == "switch":
        try:
            session_id = manager.switch_session(args.session_id)
        except ValueError as exc:
            print(str(exc), flush=True)
            return 1
        print(f"Current session: {session_id}", flush=True)
        return 0
    if args.action == "del":
        try:
            session_id = manager.delete_session(args.session_id)
        except ValueError as exc:
            print(str(exc), flush=True)
            return 1
        except OSError as exc:
            print(
                f"cannot delete session: {args.session_id}; "
                f"session files may still be in use ({exc})",
                flush=True,
            )
            return 1
        print(f"Deleted session: {session_id}", flush=True)
        return 0
    if args.action == "list":
        sessions = manager.list_sessions()
        if not sessions:
            print("No sessions found.", flush=True)
            return 0
        current = manager.current_session()
        for session_id in sessions:
            marker = "*" if session_id == current else " "
            print(f"{marker} {session_id}", flush=True)
        return 0
    if args.action == "current":
        session_id = manager.current_session()
        if session_id is None:
            print("No current session.", flush=True)
            return 0
        print(session_id, flush=True)
        return 0

    parser.print_help()
    return 1


def _run_agent_stream(
    *,
    host: str,
    port: int,
    goal: str,
    session_id: str,
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
                "session_id": session_id,
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
    "session": session_command,
    "shutdown": shutdown_command,
    "trace": trace_command,
}
