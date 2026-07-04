from datetime import datetime
from typing import Any

from .reader import TraceSummary


def format_trace_summary(summary: TraceSummary) -> str:
    started = _format_clock(summary.started_at) if summary.started_at else "unknown"
    finished = _format_clock(summary.finished_at) if summary.finished_at else "unknown"
    return (
        f"{summary.run_id}  entries={summary.entry_count}  "
        f"started={started}  finished={finished}"
    )


def format_trace_entry(entry: dict[str, Any]) -> str:
    clock = _format_clock(entry.get("ts"))
    direction = entry.get("direction")
    payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}

    if direction == "CLIENT_TO_CORE":
        return _join_parts(
            clock,
            "CLIENT→CORE",
            _field("method", payload.get("method")),
            _quoted_field("goal", payload.get("goal")),
        )
    if direction == "CORE":
        return _join_parts(
            clock,
            "CORE",
            _field("event_type", payload.get("event_type")),
            _field("node", payload.get("node")),
            _field("step", payload.get("step_id")),
            _field("decision", payload.get("decision")),
            _field("tool", payload.get("tool")),
        )
    if direction == "CORE_TO_CLIENT":
        if payload.get("message_type") == "event":
            return _join_parts(
                clock,
                "CORE→CLIENT",
                _field("event", payload.get("event_type")),
                _field("sub", payload.get("subscription_id")),
            )
        return _join_parts(
            clock,
            "CORE→CLIENT",
            _field("run_id", payload.get("run_id")),
            _field("status", payload.get("status")),
            _field("error", payload.get("error")),
        )
    if direction == "CORE_TO_LLM":
        return _join_parts(
            clock,
            "CORE→LLM",
            _field("msgs", payload.get("message_count")),
            _field("tools", payload.get("tool_call_count")),
        )
    if direction == "LLM_TO_CORE":
        return _join_parts(
            clock,
            "LLM→CORE",
            _field("out_tokens", payload.get("completion_tokens")),
            _field("total_tokens", payload.get("total_tokens")),
            _field("error", payload.get("error")),
        )
    return _join_parts(clock, str(direction or "UNKNOWN"))


def _format_clock(value: object) -> str:
    if not isinstance(value, str) or not value:
        return "??:??:??.???"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value[:12]
    return f"{parsed:%H:%M:%S}.{parsed.microsecond // 1000:03d}"


def _join_parts(*parts: str | None) -> str:
    return " ".join(part for part in parts if part)


def _field(name: str, value: object) -> str | None:
    if value is None:
        return None
    return f"{name}={value}"


def _quoted_field(name: str, value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'{name}="{escaped}"'
