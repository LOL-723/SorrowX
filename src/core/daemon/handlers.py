from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable

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


Handler = Callable[[dict[str, Any], DaemonState], dict[str, Any]]


def dispatch_rpc(message: dict[str, Any], state: DaemonState) -> HandlerResult:
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
        result = handler(message.get("params") or {}, state)
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


def handle_ping(params: dict[str, Any], state: DaemonState) -> dict[str, Any]:
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


def handle_shutdown(params: dict[str, Any], state: DaemonState) -> dict[str, Any]:
    return {
        "server": "sorrow-core",
        "server_version": state.version,
        "status": "shutting_down",
        "uptime_ms": state.uptime_ms(),
        "received_at": datetime.now(UTC).isoformat(),
    }


_HANDLERS: dict[str, Handler] = {
    "core.ping": handle_ping,
    "core.shutdown": handle_shutdown,
}
