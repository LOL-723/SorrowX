import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import RLock


CORE_ROOT = Path(__file__).resolve().parents[1]
CURRENT_SESSION_PATH = CORE_ROOT / "session" / "current_session.json"
SESSION_MEMORY_ROOT = CORE_ROOT / "storage" / "session_memory"
SESSION_TRACE_ROOT = CORE_ROOT / "trace" / "session_trace"
SESSION_ID_PREFIX = "session_"
SESSION_ID_RE = re.compile(r"^session_(\d+)$")


@dataclass(frozen=True)
class SessionPaths:
    session_id: str
    memory_dir: Path
    memory_path: Path
    trace_dir: Path


class SessionManager:
    def __init__(
        self,
        *,
        current_session_path: str | Path | None = None,
        memory_root: str | Path | None = None,
        trace_root: str | Path | None = None,
    ) -> None:
        self.current_session_path = (
            Path(current_session_path)
            if current_session_path is not None
            else CURRENT_SESSION_PATH
        )
        self.memory_root = Path(memory_root) if memory_root is not None else SESSION_MEMORY_ROOT
        self.trace_root = Path(trace_root) if trace_root is not None else SESSION_TRACE_ROOT
        self._lock = RLock()

    def new_session(self) -> str:
        with self._lock:
            session_id = self._next_session_id()
            self.ensure_session_dirs(session_id)
            self.set_current_session(session_id)
            return session_id

    def switch_session(self, session_id: str) -> str:
        self._raise_if_invalid(session_id)
        with self._lock:
            if not self.session_exists(session_id):
                raise ValueError(f"session does not exist: {session_id}")
            self.ensure_session_dirs(session_id)
            self.set_current_session(session_id)
            return session_id

    def delete_session(self, session_id: str) -> str:
        self._raise_if_invalid(session_id)
        with self._lock:
            if not self.session_exists(session_id):
                raise ValueError(f"session does not exist: {session_id}")
            if self.current_session() == session_id:
                raise ValueError(
                    f"cannot delete current session: {session_id}; "
                    "switch to another session before deleting it"
                )

            _remove_session_dir(self.memory_root, session_id)
            _remove_session_dir(self.trace_root, session_id)
            return session_id

    def ensure_current_session(self) -> str:
        with self._lock:
            current = self.current_session()
            if current is not None:
                self.ensure_session_dirs(current)
                return current
            return self.new_session()

    def current_session(self) -> str | None:
        if not self.current_session_path.exists():
            return None
        try:
            data = json.loads(self.current_session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        session_id = data.get("current_session") if isinstance(data, dict) else None
        return session_id if is_valid_session_id(session_id) else None

    def set_current_session(self, session_id: str) -> None:
        self._raise_if_invalid(session_id)
        self.current_session_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_session_path.write_text(
            json.dumps({"current_session": session_id}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear_current_session(self) -> None:
        try:
            self.current_session_path.unlink()
        except FileNotFoundError:
            return

    def list_sessions(self) -> list[str]:
        session_ids: set[str] = set()
        session_ids.update(_valid_child_dir_names(self.memory_root))
        session_ids.update(_valid_child_dir_names(self.trace_root))
        current = self.current_session()
        if current is not None:
            session_ids.add(current)
        return sorted(session_ids, key=_session_sort_key)

    def session_exists(self, session_id: str) -> bool:
        self._raise_if_invalid(session_id)
        return session_id in self.list_sessions()

    def ensure_session_dirs(self, session_id: str) -> SessionPaths:
        paths = self.paths_for(session_id)
        paths.memory_dir.mkdir(parents=True, exist_ok=True)
        paths.trace_dir.mkdir(parents=True, exist_ok=True)
        return paths

    def paths_for(self, session_id: str) -> SessionPaths:
        self._raise_if_invalid(session_id)
        memory_dir = self.memory_root / session_id
        return SessionPaths(
            session_id=session_id,
            memory_dir=memory_dir,
            memory_path=memory_dir / "context_memory.jsonl",
            trace_dir=self.trace_root / session_id,
        )

    def memory_path(self, session_id: str) -> Path:
        return self.paths_for(session_id).memory_path

    def trace_dir(self, session_id: str) -> Path:
        return self.paths_for(session_id).trace_dir

    def _next_session_id(self) -> str:
        highest = 0
        for session_id in self.list_sessions():
            match = SESSION_ID_RE.match(session_id)
            if match is not None:
                highest = max(highest, int(match.group(1)))

        next_index = highest + 1
        while True:
            session_id = f"{SESSION_ID_PREFIX}{next_index}"
            if not self.session_exists(session_id):
                return session_id
            next_index += 1

    def _raise_if_invalid(self, session_id: object) -> None:
        if not is_valid_session_id(session_id):
            raise ValueError("session_id must be a non-empty name without path separators")


_default_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = SessionManager()
    return _default_manager


def is_valid_session_id(session_id: object) -> bool:
    return (
        isinstance(session_id, str)
        and bool(session_id.strip())
        and session_id not in {".", ".."}
        and Path(session_id).name == session_id
    )


def _valid_child_dir_names(root: Path) -> list[str]:
    if not root.exists():
        return []
    names: list[str] = []
    for path in root.iterdir():
        if path.is_dir() and is_valid_session_id(path.name):
            names.append(path.name)
    return names


def _remove_session_dir(root: Path, session_id: str) -> None:
    target = root / session_id
    if not target.exists():
        return

    resolved_root = root.resolve(strict=False)
    resolved_target = target.resolve(strict=False)
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"refusing to delete path outside session root: {target}") from exc

    if resolved_target == resolved_root or target.name != session_id:
        raise ValueError(f"refusing to delete invalid session path: {target}")
    shutil.rmtree(target)


def _session_sort_key(session_id: str) -> tuple[int, int | str]:
    match = SESSION_ID_RE.match(session_id)
    if match is not None:
        return (0, int(match.group(1)))
    return (1, session_id)
