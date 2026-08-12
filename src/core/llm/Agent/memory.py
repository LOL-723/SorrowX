import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from llm.Agent.state import AgentLoopSignal, AgentState


DEFAULT_CONTEXT_MEMORY_PATH = (
    Path(__file__).resolve().parents[2] / "storage" / "context_memory.jsonl"
)
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_VISUALIZATION_PATH = WORKSPACE_ROOT / "plan.md"
VISUALIZATION_SEPARATOR = "#$#"
CONTEXT_MEMORY_SUMMARY_FILENAME = "context_memory_summary.md"
CONTEXT_MEMORY_SUMMARY_MAX_CHARS = 300
CONTEXT_MEMORY_SUMMARY_SOURCE_LIMIT = 10


@dataclass(frozen=True)
class ContextMemorySummaryUpdate:
    status: str
    path: Path
    summary: str
    record_count: int


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
        document_id: str | None,
        logs: list[dict[str, Any]] | None = None,
    ) -> AgentState:
        return {
            "question": question,
            "document_id": document_id,
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
        updated = dict(state)
        updated.update(self.state_fields())
        return updated

    def subagent_state(
        self,
        *,
        parent_state: AgentState,
        question: str,
        step_id: str,
        task: str,
        subagent_step: dict[str, Any],
    ) -> AgentState:
        return {
            "question": question,
            "document_id": parent_state.get("document_id"),
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
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_CONTEXT_MEMORY_PATH

    def remember(self, *, question: str, final_answer: str) -> list[dict[str, str]]:
        if not question.strip():
            return self.load()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "question": question,
            "final_answer": final_answer,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return self.load()

    @property
    def summary_path(self) -> Path:
        return self.path.with_name(CONTEXT_MEMORY_SUMMARY_FILENAME)

    def load_summary(self) -> str:
        summary, _ = self._read_summary_file()
        return summary

    def refresh_summary(
        self,
        *,
        summarize: Callable[[list[dict[str, str]]], str],
        force: bool = False,
    ) -> ContextMemorySummaryUpdate:
        records = self.load()[-CONTEXT_MEMORY_SUMMARY_SOURCE_LIMIT:]
        if not records:
            return ContextMemorySummaryUpdate(
                status="empty",
                path=self.summary_path,
                summary="",
                record_count=0,
            )

        source_digest = _memory_records_digest(records)
        existing_summary, existing_metadata = self._read_summary_file()
        if not force and existing_metadata.get("source_digest") == source_digest:
            return ContextMemorySummaryUpdate(
                status="unchanged",
                path=self.summary_path,
                summary=existing_summary,
                record_count=len(records),
            )

        generated_summary = _normalize_memory_summary(summarize(records))
        if not generated_summary:
            raise ValueError("memory summary cannot be empty")

        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "source_digest": source_digest,
            "record_count": len(records),
        }
        self.summary_path.write_text(
            "<!-- sorrow-memory-summary: "
            f"{json.dumps(metadata, ensure_ascii=False, sort_keys=True)} -->\n"
            f"{generated_summary}\n",
            encoding="utf-8",
        )
        return ContextMemorySummaryUpdate(
            status="updated",
            path=self.summary_path,
            summary=generated_summary,
            record_count=len(records),
        )

    def load(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []

        records: list[dict[str, str]] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                question = data.get("question")
                final_answer = data.get("final_answer")
                if isinstance(question, str) and isinstance(final_answer, str):
                    records.append(
                        {
                            "question": question,
                            "final_answer": final_answer,
                        }
                    )
        return records

    def _read_summary_file(self) -> tuple[str, dict[str, Any]]:
        if not self.summary_path.exists():
            return "", {}

        raw_content = self.summary_path.read_text(encoding="utf-8")
        lines = raw_content.splitlines()
        metadata: dict[str, Any] = {}
        if lines and lines[0].startswith("<!-- sorrow-memory-summary: ") and lines[0].endswith(" -->"):
            raw_metadata = lines[0][len("<!-- sorrow-memory-summary: ") : -len(" -->")]
            try:
                parsed_metadata = json.loads(raw_metadata)
            except json.JSONDecodeError:
                parsed_metadata = {}
            if isinstance(parsed_metadata, dict):
                metadata = parsed_metadata
            lines = lines[1:]
        return _normalize_memory_summary("\n".join(lines)), metadata


def _memory_records_digest(records: list[dict[str, str]]) -> str:
    content = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _normalize_memory_summary(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("memory summary must be a string")
    return value.strip()[:CONTEXT_MEMORY_SUMMARY_MAX_CHARS]
