import json
import hashlib
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from llm.Agent.state import AgentLoopSignal, AgentState, PlanStepState


DEFAULT_CONTEXT_MEMORY_PATH = (
    Path(__file__).resolve().parents[2] / "storage" / "context_memory.jsonl"
)
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_VISUALIZATION_PATH = WORKSPACE_ROOT / "plan.md"
VISUALIZATION_SEPARATOR = "#$#"
CONTEXT_MEMORY_SUMMARY_FILENAME = "context_memory_summary.md"
CONTEXT_MEMORY_STATE_FILENAME = "memory_state.json"
CONTEXT_MEMORY_SUMMARY_MAX_CHARS = 300
CONTEXT_MEMORY_SUMMARY_SOURCE_LIMIT = 10
CONTEXT_MEMORY_FALLBACK_MAX_CHARS = 6_000
CONTEXT_MEMORY_RECORD_FIELD_MAX_CHARS = 600
MEMORY_SCHEMA_VERSION = 2

_MEMORY_LOCKS_GUARD = threading.Lock()
_MEMORY_LOCKS: dict[Path, tuple[threading.RLock, threading.Lock]] = {}


@dataclass(frozen=True)
class ContextMemorySummaryUpdate:
    status: str
    path: Path
    summary: str
    record_count: int
    covered_through_sequence: int = 0


@dataclass(frozen=True)
class ContextMemoryStatus:
    raw_sequence: int
    summary_covered_sequence: int
    last_auto_attempted_sequence: int
    failed_at_sequence: int | None
    refresh_status: str
    needs_fallback: bool


def append_plan_visualization(
    plan: list[dict[str, Any]],
    *,
    path: str | Path | None = None,
) -> None:
    _append_visualization_snapshot(
        path=Path(path) if path is not None else DEFAULT_PLAN_VISUALIZATION_PATH,
        value=plan,
    )


def _append_visualization_snapshot(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(
            "\n"
            f"{VISUALIZATION_SEPARATOR}\n"
            "```json\n"
            f"{json.dumps(value, ensure_ascii=False, indent=2, default=str)}\n"
            "```\n"
        )


class OneRunMemory:
    def __init__(self, state: AgentState):
        self.react_results = list(state.get("react_results", []))
        self.step_results = list(state.get("step_results", []))
        self.overthink_counts = dict(state.get("overthink_counts", {}))
        self.no_finding_counts = dict(state.get("no_finding_counts", {}))
        self.failed_tools = list(state.get("failed_tools", []))
        self.subagent_results = list(state.get("subagent_results", []))
        self.agent_depth = int(state.get("agent_depth", 0) or 0)

    @classmethod
    def initial_state(
        cls,
        *,
        question: str,
        logs: list[dict[str, Any]] | None = None,
    ) -> AgentState:
        return {
            "question": question,
            "logs": list(logs or []),
            "failed_tools": [],
            "overthink_counts": {},
            "no_finding_counts": {},
            "subagent_results": [],
            "agent_depth": 0,
        }

    def last_tool_observation(self) -> str | None:
        for react_result in reversed(self.react_results):
            observation = react_result.get("observation")
            if isinstance(observation, str) and observation.strip():
                return observation
        return None

    def tool_calls(self) -> list[str]:
        tool_calls: list[str] = []
        for react_result in self.react_results:
            tool_name = react_result.get("tool_name")
            if not isinstance(tool_name, str):
                continue
            if tool_name not in tool_calls:
                tool_calls.append(tool_name)
        return tool_calls

    def append_loop_result(self, loop_result: dict[str, Any]) -> None:
        self.react_results.append(loop_result)

    def reset_react_results(self) -> None:
        self.react_results = []

    def update_no_finding(
        self,
        *,
        step_id: str,
        no_finding: int,
    ) -> AgentLoopSignal | None:
        if no_finding == 0:
            self.no_finding_counts[step_id] = 0
            return None

        next_count = self.no_finding_counts.get(step_id, 0) + 1
        self.no_finding_counts[step_id] = next_count
        if next_count >= 6:
            return "finding_missing"
        return None

    def record_overthink(self, step_id: str) -> int:
        next_count = self.overthink_counts.get(step_id, 0) + 1
        self.overthink_counts[step_id] = next_count
        return next_count

    def record_failed_tool(self, tool_name: str) -> None:
        if tool_name not in self.failed_tools:
            self.failed_tools.append(tool_name)

    def append_step_result(
        self,
        *,
        step_id: str,
        task: str,
        result: str,
    ) -> None:
        self.step_results.append(
            {
                "step_id": step_id,
                "task": task,
                "result": result,
            }
        )

    def append_subagent_result(
        self,
        *,
        step_id: str,
        task: str,
        status: str,
        result: str | None = None,
        error: str | None = None,
    ) -> None:
        item: dict[str, Any] = {
            "step_id": step_id,
            "task": task,
            "status": status,
        }
        if result is not None:
            item["result"] = result
        if error is not None:
            item["error"] = error
        self.subagent_results.append(item)

    def trigger_trace(self, loop_result: dict[str, Any]) -> list[dict[str, Any]]:
        return self.react_results + [loop_result]

    def state_fields(self) -> AgentState:
        return {
            "react_results": self.react_results,
            "step_results": self.step_results,
            "tool_calls": self.tool_calls(),
            "failed_tools": self.failed_tools,
            "overthink_counts": self.overthink_counts,
            "no_finding_counts": self.no_finding_counts,
            "subagent_results": self.subagent_results,
            "agent_depth": self.agent_depth,
        }

    def apply_to_state(self, state: AgentState) -> AgentState:
        updated: AgentState = {**state, **self.state_fields()}
        return updated

    def subagent_state(
        self,
        *,
        parent_state: AgentState,
        question: str,
        step_id: str,
        task: str,
        subagent_step: PlanStepState,
    ) -> AgentState:
        return {
            "question": question,
            "context_memory": parent_state.get("context_memory", ""),
            "plan": [subagent_step],
            "current_step_index": 0,
            "current_step_id": step_id,
            "react_results": list(self.react_results),
            "step_results": list(self.step_results),
            "failed_tools": list(self.failed_tools),
            "overthink_counts": dict(self.overthink_counts),
            "no_finding_counts": dict(self.no_finding_counts),
            "agent_depth": 1,
            "logs": parent_state.get("logs", []),
        }


class ContextMemory:
    """Session-scoped durable memory with a raw archive and a rolling summary."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_CONTEXT_MEMORY_PATH
        self._lock, self._summary_lock = _locks_for(self.path)

    def remember(
        self,
        *,
        question: str,
        final_answer: str,
        run_id: str | None = None,
        created_at: str | None = None,
    ) -> list[dict[str, str]]:
        """Append one completed run. Empty questions intentionally create no record."""
        if not question.strip():
            return self.load()
        with self._lock:
            records = self._load_records_locked()
            state = self._read_state_locked(records)
            sequence = len(records) + 1
            record = {
                "schema_version": MEMORY_SCHEMA_VERSION,
                "sequence": sequence,
                "run_id": run_id or "",
                "created_at": created_at or datetime.now(UTC).isoformat(),
                "question": question,
                "final_answer": final_answer,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
                os.fsync(file.fileno())
            state.update({"raw_sequence": sequence, "refresh_status": "dirty"})
            self._write_state_locked(state)
            records.append(record)
            return _legacy_records(records)

    @property
    def summary_path(self) -> Path:
        return self.path.with_name(CONTEXT_MEMORY_SUMMARY_FILENAME)

    @property
    def state_path(self) -> Path:
        return self.path.with_name(CONTEXT_MEMORY_STATE_FILENAME)

    def load_summary(self) -> str:
        with self._lock:
            summary, _ = self._read_summary_file_locked()
            return summary

    def load_context(self) -> str:
        """Return valid summary plus raw fallback. This method never schedules work."""
        with self._lock:
            records = self._load_records_locked()
            state = self._read_state_locked(records)
            summary, metadata = self._read_summary_file_locked()
            covered = int(state["summary_covered_sequence"])
            valid = self._summary_is_valid_locked(records, state, summary, metadata)
            if valid and covered >= len(records):
                return summary
            additions = records[covered:] if valid else records
            fallback = _format_record_fallback(additions)
            if valid and summary and fallback:
                return f"已有记忆摘要：\n{summary}\n\n尚未摘要的最近对话：\n{fallback}"
            return fallback

    def status(self) -> ContextMemoryStatus:
        with self._lock:
            records = self._load_records_locked()
            state = self._read_state_locked(records)
            summary, metadata = self._read_summary_file_locked()
            valid = self._summary_is_valid_locked(records, state, summary, metadata)
            return ContextMemoryStatus(
                raw_sequence=len(records),
                summary_covered_sequence=int(state["summary_covered_sequence"]),
                last_auto_attempted_sequence=int(state["last_auto_attempted_sequence"]),
                failed_at_sequence=_optional_nonnegative_int(state.get("failed_at_sequence")),
                refresh_status=str(state["refresh_status"]),
                needs_fallback=not valid or int(state["summary_covered_sequence"]) < len(records),
            )

    def reserve_auto_refresh(self) -> int | None:
        """Reserve one newer sequence for automatic refresh, or suppress it."""
        with self._lock:
            records = self._load_records_locked()
            state = self._read_state_locked(records)
            raw_sequence = len(records)
            covered = int(state["summary_covered_sequence"])
            last_attempted = int(state["last_auto_attempted_sequence"])
            failed_at = _optional_nonnegative_int(state.get("failed_at_sequence"))
            if (
                raw_sequence <= covered
                or raw_sequence <= last_attempted
                or (failed_at is not None and raw_sequence <= failed_at)
            ):
                return None
            state.update(
                {
                    "last_auto_attempted_sequence": raw_sequence,
                    "refresh_status": "scheduled",
                }
            )
            self._write_state_locked(state)
            return raw_sequence

    def mark_refreshing(self, target_sequence: int) -> None:
        with self._lock:
            records = self._load_records_locked()
            state = self._read_state_locked(records)
            if len(records) >= target_sequence:
                state["refresh_status"] = "refreshing"
                self._write_state_locked(state)

    def mark_refresh_failed(self, target_sequence: int) -> None:
        with self._lock:
            records = self._load_records_locked()
            state = self._read_state_locked(records)
            state.update(
                {
                    "refresh_status": "failed",
                    "failed_at_sequence": min(target_sequence, len(records)),
                }
            )
            self._write_state_locked(state)

    def refresh_rolling_summary(
        self,
        *,
        summarize: Callable[[str, list[dict[str, Any]]], str],
        force: bool = False,
        target_sequence: int | None = None,
    ) -> ContextMemorySummaryUpdate:
        """Merge at most ten new records per model call into a bounded summary."""
        with self._summary_lock:
            with self._lock:
                records = self._load_records_locked()
                state = self._read_state_locked(records)
                summary, metadata = self._read_summary_file_locked()
                if not records:
                    return ContextMemorySummaryUpdate("empty", self.summary_path, "", 0, 0)
                raw_sequence = len(records)
                target = min(target_sequence or raw_sequence, raw_sequence)
                valid = self._summary_is_valid_locked(records, state, summary, metadata)
                covered = int(state["summary_covered_sequence"]) if valid and not force else 0
                current_summary = summary if valid and not force else ""
                if covered >= target and not force:
                    return ContextMemorySummaryUpdate(
                        "unchanged", self.summary_path, current_summary, target, covered
                    )

            while covered < target:
                next_covered = min(covered + CONTEXT_MEMORY_SUMMARY_SOURCE_LIMIT, target)
                batch = records[covered:next_covered]
                generated_summary = _normalize_memory_summary(summarize(current_summary, batch))
                if not generated_summary:
                    raise ValueError("memory summary cannot be empty")
                with self._lock:
                    latest_records = self._load_records_locked()
                    # A single summary lock serializes writers; records may still be appended.
                    source_records = latest_records[:next_covered]
                    if len(source_records) != next_covered:
                        raise RuntimeError("memory records changed before summary commit")
                    metadata = {
                        "schema_version": MEMORY_SCHEMA_VERSION,
                        "covered_through_sequence": next_covered,
                        "source_digest": _memory_records_digest(source_records),
                        "record_count": next_covered,
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                    self._write_summary_locked(generated_summary, metadata)
                    state = self._read_state_locked(latest_records)
                    state.update(
                        {
                            "summary_covered_sequence": next_covered,
                            "refresh_status": "idle" if next_covered >= len(latest_records) else "dirty",
                            "failed_at_sequence": None,
                        }
                    )
                    self._write_state_locked(state)
                current_summary = generated_summary
                covered = next_covered
            return ContextMemorySummaryUpdate(
                "updated", self.summary_path, current_summary, target, covered
            )

    # Kept for compatibility with callers that explicitly request the legacy
    # latest-ten summarizer. Runtime and CLI use refresh_rolling_summary instead.
    def refresh_summary(
        self,
        *,
        summarize: Callable[[list[dict[str, str]]], str],
        force: bool = False,
    ) -> ContextMemorySummaryUpdate:
        with self._summary_lock:
            with self._lock:
                records = self._load_records_locked()
                if not records:
                    return ContextMemorySummaryUpdate("empty", self.summary_path, "", 0, 0)
                source = records[-CONTEXT_MEMORY_SUMMARY_SOURCE_LIMIT:]
                digest = _memory_records_digest(source)
                existing, metadata = self._read_summary_file_locked()
                if not force and metadata.get("legacy_source_digest") == digest:
                    return ContextMemorySummaryUpdate("unchanged", self.summary_path, existing, len(source), len(records))
            generated = _normalize_memory_summary(summarize(_legacy_records(source)))
            if not generated:
                raise ValueError("memory summary cannot be empty")
            with self._lock:
                all_records = self._load_records_locked()
                covered = len(all_records)
                metadata = {
                    "schema_version": MEMORY_SCHEMA_VERSION,
                    "covered_through_sequence": covered,
                    "source_digest": _memory_records_digest(all_records),
                    "legacy_source_digest": digest,
                    "record_count": covered,
                    "updated_at": datetime.now(UTC).isoformat(),
                }
                self._write_summary_locked(generated, metadata)
                state = self._read_state_locked(all_records)
                state.update(
                    {
                        "summary_covered_sequence": covered,
                        "refresh_status": "idle",
                        "failed_at_sequence": None,
                    }
                )
                self._write_state_locked(state)
            return ContextMemorySummaryUpdate("updated", self.summary_path, generated, len(source), covered)

    def load(self) -> list[dict[str, str]]:
        with self._lock:
            return _legacy_records(self._load_records_locked())

    def load_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(record) for record in self._load_records_locked()]

    def _load_records_locked(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                question, final_answer = data.get("question"), data.get("final_answer")
                if not isinstance(question, str) or not isinstance(final_answer, str):
                    continue
                sequence = len(records) + 1
                records.append(
                    {
                        "schema_version": data.get("schema_version", 1),
                        "sequence": sequence,
                        "run_id": data.get("run_id", ""),
                        "created_at": data.get("created_at", ""),
                        "question": question,
                        "final_answer": final_answer,
                    }
                )
        return records

    def _read_state_locked(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        state = _default_state(len(records))
        if self.state_path.exists():
            try:
                loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                loaded = {}
            if isinstance(loaded, dict):
                state.update({key: loaded[key] for key in state if key in loaded})
        state["raw_sequence"] = len(records)
        state["summary_covered_sequence"] = min(
            _nonnegative_int(state.get("summary_covered_sequence")), len(records)
        )
        state["last_auto_attempted_sequence"] = min(
            _nonnegative_int(state.get("last_auto_attempted_sequence")), len(records)
        )
        failed_at = _optional_nonnegative_int(state.get("failed_at_sequence"))
        state["failed_at_sequence"] = min(failed_at, len(records)) if failed_at is not None else None
        return state

    def _write_state_locked(self, state: dict[str, Any]) -> None:
        payload = _default_state(_nonnegative_int(state.get("raw_sequence")))
        payload.update(state)
        payload["schema_version"] = MEMORY_SCHEMA_VERSION
        _atomic_write_text(self.state_path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")

    def _read_summary_file_locked(self) -> tuple[str, dict[str, Any]]:
        if not self.summary_path.exists():
            return "", {}
        raw_content = self.summary_path.read_text(encoding="utf-8")
        lines = raw_content.splitlines()
        metadata: dict[str, Any] = {}
        if lines and lines[0].startswith("<!-- sorrow-memory-summary: ") and lines[0].endswith(" -->"):
            raw_metadata = lines[0][len("<!-- sorrow-memory-summary: ") : -len(" -->")]
            try:
                parsed = json.loads(raw_metadata)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                metadata = parsed
            lines = lines[1:]
        return _normalize_memory_summary("\n".join(lines)), metadata

    def _write_summary_locked(self, summary: str, metadata: dict[str, Any]) -> None:
        content = (
            "<!-- sorrow-memory-summary: "
            f"{json.dumps(metadata, ensure_ascii=False, sort_keys=True)} -->\n{summary}\n"
        )
        _atomic_write_text(self.summary_path, content)

    def _summary_is_valid_locked(
        self,
        records: list[dict[str, Any]],
        state: dict[str, Any],
        summary: str,
        metadata: dict[str, Any],
    ) -> bool:
        covered = _nonnegative_int(state.get("summary_covered_sequence"))
        if not summary or covered == 0 or covered > len(records):
            return False
        if _nonnegative_int(metadata.get("covered_through_sequence")) != covered:
            return False
        return metadata.get("source_digest") == _memory_records_digest(records[:covered])


class MemoryRefreshScheduler:
    """Coalesces automatic refreshes without making reads or failures retry."""

    def __init__(
        self,
        refresh: Callable[[ContextMemory, int], ContextMemorySummaryUpdate] | None = None,
    ) -> None:
        self._refresh = refresh or _default_refresh
        self._lock = threading.RLock()
        self._targets: dict[Path, int] = {}
        self._running: set[Path] = set()
        self._idle = threading.Condition(self._lock)

    def schedule(
        self,
        memory: ContextMemory,
        *,
        on_failure: Callable[[Exception], None] | None = None,
    ) -> bool:
        target = memory.reserve_auto_refresh()
        if target is None:
            return False
        key = memory.path.resolve(strict=False)
        with self._lock:
            self._targets[key] = max(target, self._targets.get(key, 0))
            if key in self._running:
                return True
            self._running.add(key)
            thread = threading.Thread(
                target=self._run,
                args=(key, memory, on_failure),
                name=f"memory-refresh-{key.parent.name}",
                daemon=True,
            )
            thread.start()
        return True

    def wait_for_idle(self, timeout: float = 5.0) -> bool:
        # Condition.wait requires a duration; monotonic makes this test helper deterministic.
        import time

        end = time.monotonic() + timeout
        with self._lock:
            while self._running:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return False
                self._idle.wait(remaining)
            return True

    def _run(
        self,
        key: Path,
        memory: ContextMemory,
        on_failure: Callable[[Exception], None] | None,
    ) -> None:
        while True:
            with self._lock:
                target = self._targets.pop(key, 0)
            try:
                memory.mark_refreshing(target)
                self._refresh(memory, target)
            except Exception as exc:
                memory.mark_refresh_failed(target)
                if on_failure is not None:
                    on_failure(exc)
                with self._lock:
                    next_target = self._targets.get(key, 0)
                    # Only a record written after the failed target may unlock one retry.
                    if next_target > target:
                        continue
                    self._targets.pop(key, None)
                    self._running.discard(key)
                    self._idle.notify_all()
                    return
            with self._lock:
                if key not in self._targets:
                    self._running.discard(key)
                    self._idle.notify_all()
                    return


def _default_refresh(memory: ContextMemory, target_sequence: int) -> ContextMemorySummaryUpdate:
    from llm.Agent.memory_summary import refresh_context_memory_summary

    return refresh_context_memory_summary(memory, target_sequence=target_sequence)


def _locks_for(path: Path) -> tuple[threading.RLock, threading.Lock]:
    key = path.resolve(strict=False)
    with _MEMORY_LOCKS_GUARD:
        existing = _MEMORY_LOCKS.get(key)
        if existing is None:
            existing = (threading.RLock(), threading.Lock())
            _MEMORY_LOCKS[key] = existing
        return existing


def _default_state(raw_sequence: int) -> dict[str, Any]:
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "raw_sequence": raw_sequence,
        "summary_covered_sequence": 0,
        "last_auto_attempted_sequence": 0,
        "failed_at_sequence": None,
        "refresh_status": "idle" if raw_sequence == 0 else "dirty",
    }


def _legacy_records(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"question": str(record["question"]), "final_answer": str(record["final_answer"])}
        for record in records
    ]


def _format_record_fallback(records: list[dict[str, Any]]) -> str:
    remaining = CONTEXT_MEMORY_FALLBACK_MAX_CHARS
    entries: list[str] = []
    for record in reversed(records[-CONTEXT_MEMORY_SUMMARY_SOURCE_LIMIT:]):
        question = str(record["question"]).strip()[:CONTEXT_MEMORY_RECORD_FIELD_MAX_CHARS]
        answer = str(record["final_answer"]).strip()[:CONTEXT_MEMORY_RECORD_FIELD_MAX_CHARS]
        entry = f"用户：{question}\n助手：{answer}"
        if len(entry) + (2 if entries else 0) > remaining:
            entry = entry[: max(0, remaining - (2 if entries else 0))]
        if entry:
            entries.append(entry)
            remaining -= len(entry) + (2 if len(entries) > 1 else 0)
        if remaining <= 0:
            break
    return "\n\n".join(reversed(entries))


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _nonnegative_int(value: object) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _optional_nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _memory_records_digest(records: list[dict[str, Any]]) -> str:
    content = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _normalize_memory_summary(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("memory summary must be a string")
    return value.strip()[:CONTEXT_MEMORY_SUMMARY_MAX_CHARS]
