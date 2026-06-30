import os
import socket
import time
from dataclasses import dataclass
from typing import Any

from ipc.protocol import (
    decode_message,
    encode_message,
    make_request,
    read_result_response,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7437


@dataclass(frozen=True)
class RpcCallResult:
    result: dict[str, Any]
    elapsed_ms: float


class CoreClient:
    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        timeout: float = 3.0,
    ):
        self.host = host or get_default_host()
        self.port = port if port is not None else get_default_port()
        self.timeout = timeout

    def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> RpcCallResult:
        request = make_request(method, params)
        started_at = time.perf_counter()
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            sock.settimeout(self.timeout)
            sock.sendall(encode_message(request))
            response_line = _recv_line(sock, timeout=self.timeout)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        response = decode_message(response_line)
        result = read_result_response(response, expected_id=request["id"])
        if not isinstance(result, dict):
            raise ValueError("RPC result must be an object")
        return RpcCallResult(result=result, elapsed_ms=elapsed_ms)


def get_default_host() -> str:
    return os.environ.get("SORROW_HOST", DEFAULT_HOST)


def get_default_port() -> int:
    raw_port = os.environ.get("SORROW_PORT")
    if raw_port is None:
        return DEFAULT_PORT
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError("SORROW_PORT must be an integer") from exc
    if port < 1 or port > 65535:
        raise ValueError("SORROW_PORT must be between 1 and 65535")
    return port


def _recv_line(sock: socket.socket, *, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        chunk = sock.recv(1)
        if not chunk:
            break
        chunks.append(chunk)
        if chunk == b"\n":
            return b"".join(chunks)
    raise TimeoutError("timed out waiting for daemon response")
