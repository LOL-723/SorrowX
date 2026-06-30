import json
import uuid
from dataclasses import dataclass
from typing import Any


JSONRPC_VERSION = "2.0"
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603


class ProtocolError(ValueError):
    def __init__(self, message: str, *, code: int = INVALID_REQUEST):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class JsonRpcError(Exception):
    code: int
    message: str
    data: Any | None = None

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def next_request_id() -> str:
    return uuid.uuid4().hex


def make_request(
    method: str,
    params: dict[str, Any] | None = None,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    if not method:
        raise ProtocolError("method is required")
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id or next_request_id(),
        "method": method,
        "params": params or {},
    }


def make_result_response(request_id: str | int | None, result: Any) -> dict[str, Any]:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "result": result,
    }


def make_error_response(
    request_id: str | int | None,
    *,
    code: int,
    message: str,
    data: Any | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "error": error,
    }


def encode_message(message: dict[str, Any]) -> bytes:
    return (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def decode_message(line: bytes | str) -> dict[str, Any]:
    text = line.decode("utf-8") if isinstance(line, bytes) else line
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError("invalid JSON", code=PARSE_ERROR) from exc
    if not isinstance(data, dict):
        raise ProtocolError("message must be a JSON object")
    return data


def validate_request(message: dict[str, Any]) -> None:
    if message.get("jsonrpc") != JSONRPC_VERSION:
        raise ProtocolError("jsonrpc must be 2.0")
    if "id" not in message:
        raise ProtocolError("id is required")
    method = message.get("method")
    if not isinstance(method, str) or not method:
        raise ProtocolError("method is required")
    params = message.get("params", {})
    if params is not None and not isinstance(params, dict):
        raise ProtocolError("params must be an object")


def read_result_response(
    message: dict[str, Any],
    *,
    expected_id: str | int,
) -> Any:
    if message.get("jsonrpc") != JSONRPC_VERSION:
        raise ProtocolError("response jsonrpc must be 2.0")
    if message.get("id") != expected_id:
        raise ProtocolError("response id does not match request id")
    if "error" in message:
        error = message["error"]
        if not isinstance(error, dict):
            raise ProtocolError("response error must be an object")
        raise JsonRpcError(
            code=int(error.get("code", INTERNAL_ERROR)),
            message=str(error.get("message", "unknown error")),
            data=error.get("data"),
        )
    if "result" not in message:
        raise ProtocolError("response result is required")
    return message["result"]
