"""Adapter for the project's in-process Python tools."""

from __future__ import annotations

from typing import Any

from llm.Agent.tools.contracts import ToolContext, ToolSpec


class LocalToolAdapter:
    """Adapter for the legacy in-process provider.

    An MCP adapter must implement the same ``invoke`` contract, so validation,
    permissions, failure mapping and audit logging remain owned by ToolExecutor.
    """
    def invoke(self, spec: ToolSpec, arguments: dict[str, Any], context: ToolContext) -> Any:
        return spec.handler(**arguments, workspace_root=context.workspace_root)
