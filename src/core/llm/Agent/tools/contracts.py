"""Stable contracts shared by every Agent tool provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel


Permission = Literal[
    "filesystem.read",
    "filesystem.write",
    "network.http",
    "process.execute",
]

FILESYSTEM_READ: Permission = "filesystem.read"
FILESYSTEM_WRITE: Permission = "filesystem.write"
NETWORK_HTTP: Permission = "network.http"
PROCESS_EXECUTE: Permission = "process.execute"
ALL_PERMISSIONS: frozenset[Permission] = frozenset(
    {FILESYSTEM_READ, FILESYSTEM_WRITE, NETWORK_HTTP, PROCESS_EXECUTE}
)


@dataclass(frozen=True)
class ToolContext:
    """Invocation metadata supplied by the host, never by model arguments."""

    run_id: str | None
    session_id: str | None
    workspace_root: Path
    granted_permissions: frozenset[Permission] = ALL_PERMISSIONS


class ToolAdapter(Protocol):
    """Provider boundary used by local tools today and MCP tools later."""

    def invoke(
        self,
        spec: "ToolSpec",
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> Any: ...


@dataclass(frozen=True)
class ToolError:
    code: str
    message: str
    recoverable: bool

    def model_dump(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "recoverable": self.recoverable}


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    ok: bool
    data: Any = None
    error: ToolError | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def recoverable(self) -> bool:
        return self.error is None or self.error.recoverable

    def to_observation(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "ok": self.ok,
            "result": self.data,
            "error": self.error.message if self.error else None,
            "error_code": self.error.code if self.error else None,
            "recoverable": self.recoverable,
            "metadata": self.metadata,
        }


ToolHandler = Callable[..., Any]


@dataclass(frozen=True)
class ToolSpec:
    """Declarative, provider-independent contract for an Agent-callable tool."""

    name: str
    description: str
    input_model: type[BaseModel]
    permissions: frozenset[Permission]
    side_effect: bool
    timeout_seconds: int | None
    handler: ToolHandler

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("tool name is required")
        if not self.description or not self.description.strip():
            raise ValueError(f"tool description is required: {self.name}")
        if self.timeout_seconds is not None and (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or self.timeout_seconds <= 0
        ):
            raise ValueError(f"tool timeout must be a positive integer: {self.name}")
