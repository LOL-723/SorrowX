"""Central validation, authorization, execution and audit path for tools."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from time import monotonic
from typing import Any

from pydantic import ValidationError

from llm.Agent.tools.contracts import ToolAdapter, ToolContext, ToolError, ToolResult
from llm.Agent.tools.local_adapter import LocalToolAdapter
from llm.Agent.tools.registry import ToolRegistry
from trace.recorder import get_trace_recorder


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, *, adapter: ToolAdapter | None = None) -> None:
        self.registry = registry
        self.adapter = adapter or LocalToolAdapter()

    def execute(self, tool_name: str, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        started = monotonic()
        spec = self.registry.get(tool_name)
        if spec is None:
            return self._finish(ToolResult(tool_name, False, error=ToolError("unknown_tool", f"unknown tool: {tool_name}", False)), arguments, context, started)
        missing = spec.permissions.difference(context.granted_permissions)
        if missing:
            return self._finish(ToolResult(tool_name, False, error=ToolError("permission_denied", f"tool requires permissions: {', '.join(sorted(missing))}", False)), arguments, context, started)
        try:
            validated = spec.input_model.model_validate(arguments).model_dump(exclude_none=True)
        except ValidationError as exc:
            return self._finish(ToolResult(tool_name, False, error=ToolError("invalid_arguments", str(exc), False)), arguments, context, started)
        try:
            value = self._invoke(spec, validated, context)
        except FutureTimeoutError:
            return self._finish(
                ToolResult(
                    tool_name,
                    False,
                    error=ToolError(
                        "timeout",
                        f"tool exceeded its {spec.timeout_seconds}-second timeout",
                        True,
                    ),
                ),
                arguments,
                context,
                started,
            )
        except Exception as exc:
            return self._finish(ToolResult(tool_name, False, error=ToolError("execution_error", str(exc), True)), arguments, context, started)
        if isinstance(value, dict) and isinstance(value.get("error"), str) and value["error"].strip():
            error_code = "timeout" if value.get("timed_out") else "tool_error"
            return self._finish(ToolResult(tool_name, False, data=value, error=ToolError(error_code, value["error"], True)), arguments, context, started)
        return self._finish(ToolResult(tool_name, True, data=value), arguments, context, started)

    def _invoke(self, spec, arguments: dict[str, Any], context: ToolContext) -> Any:
        """Apply the contract timeout consistently across all adapters.

        A local Python handler cannot be forcefully killed safely.  On timeout it
        is detached and its result is ignored; process/MCP adapters should also
        enforce cancellation at their transport boundary.
        """
        if spec.timeout_seconds is None:
            return self.adapter.invoke(spec, arguments, context)
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-tool")
        future = pool.submit(self.adapter.invoke, spec, arguments, context)
        try:
            return future.result(timeout=spec.timeout_seconds)
        finally:
            # Never block the caller while a non-cooperative local handler exits.
            pool.shutdown(wait=False, cancel_futures=True)

    def _finish(self, result: ToolResult, arguments: dict[str, Any], context: ToolContext, started: float) -> ToolResult:
        elapsed_ms = round((monotonic() - started) * 1000, 2)
        metadata = {**result.metadata, "duration_ms": elapsed_ms}
        result = ToolResult(result.tool_name, result.ok, result.data, result.error, metadata)
        self._audit(result, arguments, context)
        return result

    def _audit(self, result: ToolResult, arguments: dict[str, Any], context: ToolContext) -> None:
        # Trace only stable metadata and argument keys: source or generated file content never enters audit logs.
        spec = self.registry.get(result.tool_name)
        get_trace_recorder().record(
            context.run_id,
            "TOOL",
            {
                "tool": result.tool_name,
                "status": "success" if result.ok else "failed",
                "error_code": result.error.code if result.error else None,
                "side_effect": spec.side_effect if spec else None,
                "required_permissions": sorted(spec.permissions) if spec else [],
                "timeout_seconds": spec.timeout_seconds if spec else None,
                "recoverable": result.recoverable,
                "duration_ms": result.metadata["duration_ms"],
                "argument_keys": sorted(arguments),
            },
            session_id=context.session_id,
        )
