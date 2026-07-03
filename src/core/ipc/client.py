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


class CoreStreamClient:
    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        connect_timeout: float = 3.0,
        read_timeout: float | None = None,
    ) -> None:
        self.host = host or get_default_host()
        self.port = port if port is not None else get_default_port()
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self._sock: socket.socket | None = None

    def __enter__(self) -> "CoreStreamClient":
        self.connect()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def connect(self) -> None:
        if self._sock is not None:
            return
        self._sock = socket.create_connection(
            (self.host, self.port),
            timeout=self.connect_timeout,
        )
        self._sock.settimeout(self.read_timeout)

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        finally:
            self._sock = None

    def send_request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> str | int:
        sock = self._require_socket()
        request = make_request(method, params)
        sock.sendall(encode_message(request))
        return request["id"]

    def read_message(self, *, timeout: float | None = None) -> dict[str, Any]:
        sock = self._require_socket()
        response_line = _recv_line(
            sock,
            timeout=self.read_timeout if timeout is None else timeout,
        )
        return decode_message(response_line)

    def _require_socket(self) -> socket.socket:
        if self._sock is None:
            raise RuntimeError("stream client is not connected")
        return self._sock


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


def _recv_line(sock: socket.socket, *, timeout: float | None) -> bytes:
    deadline = time.monotonic() + timeout if timeout is not None else None
    sock.settimeout(timeout)
    chunks: list[bytes] = []
    while deadline is None or time.monotonic() < deadline:
        try:
            chunk = sock.recv(1)
        except socket.timeout as exc:
            raise TimeoutError("timed out waiting for daemon response") from exc
        if not chunk:
            raise ConnectionError("daemon connection closed")
        chunks.append(chunk)
        if chunk == b"\n":
            return b"".join(chunks)
    raise TimeoutError("timed out waiting for daemon response")
