"""The single registry for Agent-callable tools."""

from __future__ import annotations

from typing import Iterable

from llm.Agent.tools.contracts import Permission, ToolSpec


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool registration: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def require(self, name: str) -> ToolSpec:
        spec = self.get(name)
        if spec is None:
            raise KeyError(name)
        return spec

    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def available_for(
        self,
        granted_permissions: frozenset[Permission],
        *,
        excluded_names: set[str] | None = None,
    ) -> list[ToolSpec]:
        excluded = excluded_names or set()
        return [
            spec for spec in self._specs.values()
            if spec.name not in excluded and spec.permissions.issubset(granted_permissions)
        ]

    def model_tools(
        self,
        granted_permissions: frozenset[Permission],
        *,
        excluded_names: set[str] | None = None,
    ) -> list[dict[str, object]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_model.model_json_schema(),
                "permissions": sorted(spec.permissions),
                "side_effect": spec.side_effect,
            }
            for spec in self.available_for(granted_permissions, excluded_names=excluded_names)
        ]
