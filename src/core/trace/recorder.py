import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, TextIO


TRACE_DIR = Path(__file__).resolve().parent
DEFAULT_RECORDS_DIR = TRACE_DIR / "records"
_current_run_id: ContextVar[str | None] = ContextVar(
    "sorrow_trace_run_id",
    default=None,
)
_default_recorder: "TraceRecorder | None" = None


class TraceRecorder:
    def __init__(
        self,
        records_dir: str | Path | None = None,
        *,
        enabled: bool | None = None,
    ) -> None:
        self.records_dir = Path(records_dir) if records_dir is not None else DEFAULT_RECORDS_DIR
        self.enabled = _trace_enabled() if enabled is None else enabled
        self._lock = RLock()
        self._seq = 0
        self._files: dict[str, TextIO] = {}

    def prepare_client_to_core(self, message: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return self._make_entry(
            "CLIENT_TO_CORE",
            {
                "message_type": "request",
                "id": message.get("id"),
                "method": message.get("method"),
                **_summarize_params(message.get("params")),
            },
        )

    def write_prepared(self, run_id: str | None, entry: dict[str, Any] | None) -> None:
        if not self.enabled or entry is None or not _valid_run_id(run_id):
            return
        try:
            prepared = dict(entry)
            prepared["run_id"] = run_id
            self._write(run_id, prepared)
        except Exception:
            return

    def record_core_event(self, event: dict[str, Any]) -> None:
        run_id = event.get("run_id")
        if not _valid_run_id(run_id):
            return
        self.record(
            run_id,
            "CORE",
            {
                "message_type": "agent_event",
                **_summarize_event(event),
            },
        )

    def record_core_to_client_reply(
        self,
        run_id: str | None,
        response: dict[str, Any],
    ) -> None:
        if not _valid_run_id(run_id):
            return
        self.record(
            run_id,
            "CORE_TO_CLIENT",
            {
                "message_type": "reply",
                "id": response.get("id"),
                **_summarize_response(response),
            },
        )

    def record_core_to_client_event(
        self,
        event: dict[str, Any],
        *,
        subscription_id: str | None,
        client_id: str | None,
    ) -> None:
        run_id = event.get("run_id")
        if not _valid_run_id(run_id):
            return
        self.record(
            run_id,
            "CORE_TO_CLIENT",
            {
                "message_type": "event",
                "subscription_id": subscription_id,
                "client_id": client_id,
                **_summarize_event(event),
            },
        )

    def record_core_to_llm(
        self,
        run_id: str | None,
        *,
        model: str | None,
        message_count: int,
        tool_count: int,
    ) -> str | None:
        if not self.enabled or not _valid_run_id(run_id):
            return None
        entry = self._make_entry(
            "CORE_TO_LLM",
            {
                "model": model,
                "message_count": message_count,
                "tool_call_count": tool_count,
            },
        )
        call_id = f"llm-{entry['seq']}"
        entry["payload"]["call_id"] = call_id
        self.write_prepared(run_id, entry)
        return call_id

    def record_llm_to_core(
        self,
        run_id: str | None,
        *,
        call_id: str | None,
        usage: Any = None,
        error: str | None = None,
    ) -> None:
        if not _valid_run_id(run_id):
            return
        payload = {
            "call_id": call_id,
            **_summarize_usage(usage),
        }
        if error:
            payload["error"] = _short_text(error)
        self.record(run_id, "LLM_TO_CORE", payload)

    def record(self, run_id: str | None, direction: str, payload: dict[str, Any]) -> None:
        if not self.enabled or not _valid_run_id(run_id):
            return
        try:
            self._write(run_id, self._make_entry(direction, payload, run_id=run_id))
        except Exception:
            return

    def close(self) -> None:
        with self._lock:
            files = list(self._files.values())
            self._files.clear()
        for file in files:
            try:
                file.close()
            except Exception:
                pass

    def _make_entry(
        self,
        direction: str,
        payload: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            seq = self._seq
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "seq": seq,
            "direction": direction,
            "payload": payload,
        }
        if run_id is not None:
            entry["run_id"] = run_id
        return entry

    def _write(self, run_id: str, entry: dict[str, Any]) -> None:
        with self._lock:
            file = self._file_for_run(run_id)
            file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            file.flush()

    def _file_for_run(self, run_id: str) -> TextIO:
        file = self._files.get(run_id)
        if file is not None and not file.closed:
            return file
        self.records_dir.mkdir(parents=True, exist_ok=True)
        path = self.records_dir / f"{run_id}.jsonl"
        file = path.open("a", encoding="utf-8", buffering=1)
        self._files[run_id] = file
        return file


def get_trace_recorder() -> TraceRecorder:
    global _default_recorder
    if _default_recorder is None:
        _default_recorder = TraceRecorder()
    return _default_recorder


def current_run_id() -> str | None:
    return _current_run_id.get()


@contextmanager
def trace_run(run_id: str) -> Iterator[None]:
    token = _current_run_id.set(run_id)
    try:
        yield
    finally:
        _current_run_id.reset(token)


def _trace_enabled() -> bool:
    value = os.environ.get("SORROW_TRACE", "1").strip().lower()
    return value not in {"0", "false", "off", "no"}


def _valid_run_id(run_id: object) -> bool:
    return isinstance(run_id, str) and bool(run_id) and Path(run_id).name == run_id


def _summarize_params(params: Any) -> dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    summary: dict[str, Any] = {}
    for key in ("client", "goal", "run_id", "topics"):
        if key in params:
            summary[key] = _short_value(params[key])
    return summary


def _summarize_response(response: dict[str, Any]) -> dict[str, Any]:
    if isinstance(response.get("result"), dict):
        result = response["result"]
        summary: dict[str, Any] = {}
        for key in ("run_id", "status", "subscription_id", "topics", "replayed_count"):
            if key in result:
                summary[key] = _short_value(result[key])
        return summary
    if isinstance(response.get("error"), dict):
        error = response["error"]
        return {
            "error_code": error.get("code"),
            "error": _short_text(str(error.get("message", ""))),
        }
    return {}


def _summarize_event(event: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "event_type": event.get("type"),
    }
    for key in (
        "run_id",
        "node",
        "status",
        "phase",
        "step_id",
        "turn",
        "decision",
        "signal",
        "tool",
        "source",
    ):
        if key in event:
            summary[key] = _short_value(event[key])
    return summary


def _summarize_usage(usage: Any) -> dict[str, Any]:
    return {
        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
        "completion_tokens": _usage_value(usage, "completion_tokens"),
        "total_tokens": _usage_value(usage, "total_tokens"),
    }


def _usage_value(usage: Any, key: str) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)
    return value if isinstance(value, int) else None


def _short_value(value: Any) -> Any:
    if isinstance(value, str):
        return _short_text(value)
    if isinstance(value, list):
        return [_short_value(item) for item in value[:16]]
    if isinstance(value, dict):
        return {str(key): _short_value(value[key]) for key in list(value)[:16]}
    return value


def _short_text(value: str, *, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"
