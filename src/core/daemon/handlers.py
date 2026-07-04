import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

from daemon.events import normalize_topics
from daemon.state import DaemonState
from ipc.protocol import (
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    ProtocolError,
    make_error_response,
    make_result_response,
    validate_request,
)


@dataclass(frozen=True)
class HandlerResult:
    response: dict[str, Any]
    should_shutdown: bool = False


Handler = Callable[[dict[str, Any], DaemonState, str | None], dict[str, Any]]


def dispatch_rpc(
    message: dict[str, Any],
    state: DaemonState,
    *,
    client_id: str | None = None,
) -> HandlerResult:
    request_id = message.get("id")
    try:
        validate_request(message)
        method = message["method"]
        handler = _HANDLERS.get(method)
        if handler is None:
            return HandlerResult(
                response=make_error_response(
                    request_id,
                    code=METHOD_NOT_FOUND,
                    message=f"method not found: {method}",
                )
            )
        result = handler(message.get("params") or {}, state, client_id)
        return HandlerResult(
            response=make_result_response(request_id, result),
            should_shutdown=method == "core.shutdown",
        )
    except ProtocolError as exc:
        return HandlerResult(
            response=make_error_response(request_id, code=exc.code, message=str(exc))
        )
    except Exception as exc:
        return HandlerResult(
            response=make_error_response(
                request_id,
                code=INTERNAL_ERROR,
                message="internal daemon error",
                data=str(exc),
            )
        )


def handle_ping(
    params: dict[str, Any],
    state: DaemonState,
    client_id: str | None,
) -> dict[str, Any]:
    return {
        "server": "sorrow-core",
        "server_version": state.version,
        "host": state.host,
        "port": state.port,
        "uptime_ms": state.uptime_ms(),
        "started_at": state.iso_started_at(),
        "received_at": datetime.now(UTC).isoformat(),
        "client": params.get("client"),
    }


def handle_shutdown(
    params: dict[str, Any],
    state: DaemonState,
    client_id: str | None,
) -> dict[str, Any]:
    return {
        "server": "sorrow-core",
        "server_version": state.version,
        "status": "shutting_down",
        "uptime_ms": state.uptime_ms(),
        "received_at": datetime.now(UTC).isoformat(),
    }


def handle_event_subscribe(
    params: dict[str, Any],
    state: DaemonState,
    client_id: str | None,
) -> dict[str, Any]:
    if client_id is None:
        raise ProtocolError("event.subscribe requires a connected client")
    try:
        topics = normalize_topics(params.get("topics"))
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc

    run_id = params.get("run_id")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ProtocolError("run_id must be a non-empty string")

    client_name = params.get("client")
    if client_name is not None and not isinstance(client_name, str):
        raise ProtocolError("client must be a string")

    subscription = state.event_hub.subscribe(
        client_id=client_id,
        topics=topics,
        run_id=run_id.strip() if isinstance(run_id, str) else None,
        client_name=client_name,
    )
    return {
        "subscription_id": subscription.subscription_id,
        "topics": list(subscription.topics),
        "run_id": subscription.run_id,
        "replayed_count": 0,
    }


def handle_agent_run(
    params: dict[str, Any],
    state: DaemonState,
    client_id: str | None,
) -> dict[str, Any]:
    goal = params.get("goal")
    if not isinstance(goal, str) or not goal.strip():
        raise ProtocolError("goal must be a non-empty string")

    client_name = params.get("client")
    if client_name is not None and not isinstance(client_name, str):
        raise ProtocolError("client must be a string")

    from llm.Agent.AgentRuner import new_run_id

    run_id = new_run_id()

    task = asyncio.create_task(
        _run_agent_in_background(
            state=state,
            run_id=run_id,
            goal=goal.strip(),
        )
    )
    state.active_runs[run_id] = task
    task.add_done_callback(lambda completed: state.active_runs.pop(run_id, None))

    return {
        "run_id": run_id,
        "status": "started",
    }


async def _run_agent_in_background(
    *,
    state: DaemonState,
    run_id: str,
    goal: str,
) -> None:
    def run_sync() -> None:
        from llm.Agent.AgentRuner import AgentRuner

        runner = AgentRuner()
        try:
            runner.run(goal, run_id=run_id, event_bus=state.event_bus)
        except Exception:
            return

    try:
        await asyncio.to_thread(run_sync)
    except Exception as exc:
        state.event_bus.publish(
            _event("agent.error", run_id, error=str(exc), source="daemon")
        )
        state.event_bus.publish(
            _event(
                "run.finished",
                run_id,
                status="failed",
                answer="",
                error=str(exc),
            )
        )


def _event(event_type: str, run_id: str, **payload: Any) -> dict[str, Any]:
    return {
        "type": event_type,
        "run_id": run_id,
        "ts": datetime.now(UTC).isoformat(),
        **payload,
    }


_HANDLERS: dict[str, Handler] = {
    "agent.run": handle_agent_run,
    "core.ping": handle_ping,
    "core.shutdown": handle_shutdown,
    "event.subscribe": handle_event_subscribe,
}
