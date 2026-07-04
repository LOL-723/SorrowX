import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .recorder import DEFAULT_RECORDS_DIR


@dataclass(frozen=True)
class TraceSummary:
    run_id: str
    entry_count: int
    started_at: str | None
    finished_at: str | None


def list_traces(records_dir: str | Path | None = None) -> list[TraceSummary]:
    directory = Path(records_dir) if records_dir is not None else DEFAULT_RECORDS_DIR
    if not directory.exists():
        return []

    summaries: list[TraceSummary] = []
    for path in sorted(directory.glob("*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True):
        entries = _read_entries(path)
        if not entries:
            continue
        summaries.append(
            TraceSummary(
                run_id=path.stem,
                entry_count=len(entries),
                started_at=entries[0].get("ts"),
                finished_at=_finished_at(entries) or entries[-1].get("ts"),
            )
        )
    return summaries


def read_trace(run_id: str, records_dir: str | Path | None = None) -> list[dict[str, Any]]:
    if Path(run_id).name != run_id:
        raise ValueError("run_id cannot contain path separators")
    directory = Path(records_dir) if records_dir is not None else DEFAULT_RECORDS_DIR
    path = directory / f"{run_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"trace not found for run_id: {run_id}")
    entries = _read_entries(path)
    return sorted(entries, key=lambda entry: (str(entry.get("ts", "")), int(entry.get("seq", 0) or 0)))


def _read_entries(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return sorted(entries, key=lambda entry: (str(entry.get("ts", "")), int(entry.get("seq", 0) or 0)))


def _finished_at(entries: list[dict[str, Any]]) -> str | None:
    for entry in reversed(entries):
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("event_type") == "run.finished":
            return entry.get("ts") if isinstance(entry.get("ts"), str) else None
    return None
