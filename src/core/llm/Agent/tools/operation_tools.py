from pathlib import Path
from typing import Any


DEFAULT_MAX_ENTRIES = 20
MAX_CHILD_FILES_BEFORE_COLLAPSE = 5
MAX_CHILD_DIRS_BEFORE_COLLAPSE = 5
MAX_READ_FILE_LINES = 200
MAX_KEYWORD_MATCHES = 20
MAX_MATCH_PREVIEW_CHARS = 160
READ_FILE_SINGLE_MATCH_CONTEXT_LINES = 10

SYSTEM_OR_CACHE_DIRS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "runs",
}

SYSTEM_CONFIG_FILES = {
    ".env",
    ".gitignore",
    "pyproject.toml",
    "pytest.ini",
    "requirements.txt",
    "setup.cfg",
    "tox.ini",
}


# === Operation tool entrypoints begin ===========================================
def list_dir_tool(
    path: str | None = None,
    max_entries: int | None = DEFAULT_MAX_ENTRIES,
    **kwargs: Any,
) -> dict[str, Any]:
    """Return a compact, structured view of the workspace tree."""
    workspace_root = _resolve_workspace_root(kwargs.get("workspace_root"))
    entry_limit = _normalize_max_entries(max_entries)

    try:
        target = _resolve_target_path(workspace_root, path)
    except ValueError as exc:
        return _error_result("list_dir", str(exc), root=workspace_root)

    if _has_filtered_path_segment(target, workspace_root):
        return _error_result(
            "list_dir",
            f"path is filtered: {_display_path(target, workspace_root)}",
            root=workspace_root,
        )
    if not target.exists():
        return _error_result(
            "list_dir",
            f"path does not exist: {_display_path(target, workspace_root)}",
            root=workspace_root,
        )
    if not target.is_dir():
        return _error_result(
            "list_dir",
            f"path is not a directory: {_display_path(target, workspace_root)}",
            root=workspace_root,
        )

    budget = _EntryBudget(entry_limit)
    entries, truncated = _directory_entries(
        target,
        workspace_root=workspace_root,
        requested_root=target,
        budget=budget,
    )

    return {
        "tool_name": "list_dir",
        "root": str(workspace_root),
        "target": _display_path(target, workspace_root),
        "entries": entries,
        "truncated": truncated or budget.truncated,
    }


def read_file_tool(
    path: str | None = None,
    keyword: str | None = None,
    start_line: int | str | None = None,
    end_line: int | str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    workspace_root = _resolve_workspace_root(kwargs.get("workspace_root"))
    if path is None or not path.strip():
        return _read_file_error_result(
            "path is required",
            root=workspace_root,
            path=path or "",
        )

    try:
        target = _resolve_target_path(workspace_root, path)
    except ValueError as exc:
        return _read_file_error_result(
            str(exc),
            root=workspace_root,
            path=path,
        )

    display_path = _display_path(target, workspace_root)
    if _has_read_filtered_path_segment(target, workspace_root):
        return _read_file_error_result(
            f"path is filtered: {display_path}",
            root=workspace_root,
            path=display_path,
        )
    if not target.exists():
        return _read_file_error_result(
            f"path does not exist: {display_path}",
            root=workspace_root,
            path=display_path,
        )
    if not target.is_file():
        return _read_file_error_result(
            f"path is not a file: {display_path}",
            root=workspace_root,
            path=display_path,
        )

    try:
        lines = _read_utf8_lines(target)
        selected_start, selected_end, selected_lines = _select_line_range(
            lines,
            start_line=start_line,
            end_line=end_line,
        )
    except ValueError as exc:
        return _read_file_error_result(
            str(exc),
            root=workspace_root,
            path=display_path,
            total_lines=len(lines) if "lines" in locals() else 0,
        )

    normalized_keyword = _normalize_keyword(keyword)
    if normalized_keyword is None:
        returned_lines, truncated = _limit_numbered_lines(selected_lines)
        return _read_file_success_result(
            path=display_path,
            total_lines=len(lines),
            returned_lines=returned_lines,
            content=_format_numbered_lines(returned_lines),
            matched=True,
            matched_count=None,
            matches=[],
            truncated=truncated,
        )

    matches = _find_keyword_matches(selected_lines, normalized_keyword)
    if not matches:
        return _read_file_success_result(
            path=display_path,
            total_lines=len(lines),
            returned_lines=[],
            content="",
            matched=False,
            matched_count=0,
            matches=[],
            truncated=False,
            error="keyword not found",
            fallback_start_line=selected_start,
            fallback_end_line=selected_end,
        )

    if len(matches) > 1:
        returned_matches, truncated = _limit_keyword_matches(matches)
        return _read_file_success_result(
            path=display_path,
            total_lines=len(lines),
            returned_lines=[],
            content="",
            matched=True,
            matched_count=len(matches),
            matches=[
                _match_metadata(line_number, line_text)
                for line_number, line_text in returned_matches
            ],
            truncated=truncated,
            needs_followup=True,
            followup_hint="Use start_line/end_line around the relevant match.",
            fallback_start_line=selected_start,
            fallback_end_line=selected_end,
        )

    match_line_number = matches[0][0]
    context_lines = _keyword_context_lines(
        selected_lines,
        match_line_number=match_line_number,
    )
    returned_lines, truncated = _limit_numbered_lines(context_lines)
    return _read_file_success_result(
        path=display_path,
        total_lines=len(lines),
        returned_lines=returned_lines,
        content=_format_numbered_lines(returned_lines),
        matched=True,
        matched_count=1,
        matches=[_match_metadata(matches[0][0], matches[0][1])],
        truncated=truncated,
    )


def patch_file_tool(**_: Any) -> dict[str, Any]:
    return _not_implemented_result("patch_file")


def run_tests_tool(**_: Any) -> dict[str, Any]:
    return _not_implemented_result("run_tests")
# === Operation tool entrypoints end ========================================


# === list_dir helpers begin ===========================================
class _EntryBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.count = 0
        self.truncated = False

    def claim(self) -> bool:
        if self.count >= self.limit:
            self.truncated = True
            return False
        self.count += 1
        return True


def _normalize_max_entries(value: int | None) -> int:
    if value is None:
        return DEFAULT_MAX_ENTRIES
    if isinstance(value, bool) or not isinstance(value, int):
        return DEFAULT_MAX_ENTRIES
    if value < 1:
        return 1
    return min(value, DEFAULT_MAX_ENTRIES)


def _directory_entries(
    directory: Path,
    *,
    workspace_root: Path,
    requested_root: Path,
    budget: _EntryBudget,
) -> tuple[list[dict[str, Any]], bool]:
    try:
        children = [child for child in directory.iterdir() if _should_include_path(child)]
    except OSError as exc:
        return (
            [
                {
                    "path": _display_path(directory, workspace_root),
                    "type": "directory",
                    "error": str(exc),
                }
            ],
            False,
        )

    ordered_children = _sort_children(children)
    entries: list[dict[str, Any]] = []
    truncated = False

    for child in ordered_children:
        if not budget.claim():
            truncated = True
            break

        if child.is_dir():
            child_entry: dict[str, Any] = {
                "path": _display_path(child, workspace_root),
                "type": "directory",
            }
            if not _should_collapse_directory(child, requested_root):
                nested_entries, nested_truncated = _directory_entries(
                    child,
                    workspace_root=workspace_root,
                    requested_root=requested_root,
                    budget=budget,
                )
                if nested_entries:
                    child_entry["entries"] = nested_entries
                if nested_truncated:
                    child_entry["truncated"] = True
                truncated = truncated or nested_truncated
            else:
                child_entry["truncated"] = True
                truncated = True
            entries.append(child_entry)
            continue

        if child.is_file():
            entries.append(
                {
                    "path": _display_path(child, workspace_root),
                    "type": "file",
                }
            )

    return entries, truncated


def _should_collapse_directory(directory: Path, requested_root: Path) -> bool:
    if directory == requested_root:
        return False
    try:
        children = [child for child in directory.iterdir() if _should_include_path(child)]
    except OSError:
        return False

    file_count = sum(1 for child in children if child.is_file())
    directory_count = sum(1 for child in children if child.is_dir())
    return (
        file_count > MAX_CHILD_FILES_BEFORE_COLLAPSE
        or directory_count > MAX_CHILD_DIRS_BEFORE_COLLAPSE
    )


def _should_include_path(path: Path) -> bool:
    name = path.name
    if path.is_dir():
        return name not in SYSTEM_OR_CACHE_DIRS and not name.startswith(".")
    if name in SYSTEM_CONFIG_FILES:
        return True
    return not name.startswith(".")


def _has_filtered_path_segment(path: Path, workspace_root: Path) -> bool:
    try:
        relative_parts = path.relative_to(workspace_root).parts
    except ValueError:
        return True

    for part in relative_parts:
        if part in SYSTEM_OR_CACHE_DIRS:
            return True
        if part.startswith(".") and part not in SYSTEM_CONFIG_FILES:
            return True
    return False


def _sort_children(children: list[Path]) -> list[Path]:
    return sorted(children, key=_sort_key)


def _sort_key(path: Path) -> tuple[int, int, str]:
    if path.name == "src" and path.is_dir():
        priority = 0
    elif path.is_file():
        priority = 1
    else:
        priority = 2
    return (priority, 0 if path.is_file() else 1, path.name.lower())
# === list_dir helpers end ===========================================


# === read_file helpers begin ===========================================
def _read_utf8_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8: {path.name}") from exc
    except OSError as exc:
        raise ValueError(f"file read failed: {exc}") from exc


def _select_line_range(
    lines: list[str],
    *,
    start_line: int | str | None,
    end_line: int | str | None,
) -> tuple[int, int, list[tuple[int, str]]]:
    total_lines = len(lines)
    start = _coerce_line_number(start_line, "start_line") or 1
    end = _coerce_line_number(end_line, "end_line")
    if end is None:
        end = total_lines

    if start < 1:
        raise ValueError("start_line must be greater than or equal to 1")
    if end < 1 and total_lines > 0:
        raise ValueError("end_line must be greater than or equal to 1")
    if end_line is not None and start > end:
        raise ValueError("start_line must be less than or equal to end_line")

    if total_lines == 0:
        return 1, 0, []
    if start > total_lines:
        return start, total_lines, []

    clamped_end = min(end, total_lines)
    selected = [
        (line_number, lines[line_number - 1])
        for line_number in range(start, clamped_end + 1)
    ]
    return start, clamped_end, selected


def _coerce_line_number(value: int | str | None, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped_value = value.strip()
        if stripped_value.isdecimal():
            return int(stripped_value)
    raise ValueError(f"{field} must be an integer")


def _normalize_keyword(keyword: str | None) -> str | None:
    if keyword is None:
        return None
    stripped_keyword = keyword.strip()
    return stripped_keyword or None


def _find_keyword_matches(
    lines: list[tuple[int, str]],
    keyword: str,
) -> list[tuple[int, str]]:
    return [
        (line_number, line_text)
        for line_number, line_text in lines
        if keyword in line_text
    ]


def _keyword_context_lines(
    lines: list[tuple[int, str]],
    *,
    match_line_number: int,
) -> list[tuple[int, str]]:
    lower_bound = match_line_number - READ_FILE_SINGLE_MATCH_CONTEXT_LINES
    upper_bound = match_line_number + READ_FILE_SINGLE_MATCH_CONTEXT_LINES
    return [
        (line_number, line_text)
        for line_number, line_text in lines
        if lower_bound <= line_number <= upper_bound
    ]


def _limit_numbered_lines(
    lines: list[tuple[int, str]],
) -> tuple[list[tuple[int, str]], bool]:
    if len(lines) <= MAX_READ_FILE_LINES:
        return lines, False
    return lines[:MAX_READ_FILE_LINES], True


def _limit_keyword_matches(
    matches: list[tuple[int, str]],
) -> tuple[list[tuple[int, str]], bool]:
    if len(matches) <= MAX_KEYWORD_MATCHES:
        return matches, False
    return matches[:MAX_KEYWORD_MATCHES], True


def _format_numbered_lines(lines: list[tuple[int, str]]) -> str:
    return "\n".join(f"{line_number}: {line_text}" for line_number, line_text in lines)


def _match_metadata(line_number: int, line_text: str) -> dict[str, Any]:
    preview = line_text.strip()
    if len(preview) > MAX_MATCH_PREVIEW_CHARS:
        preview = preview[: MAX_MATCH_PREVIEW_CHARS - 3] + "..."
    return {
        "line": line_number,
        "preview": preview,
    }


def _has_read_filtered_path_segment(path: Path, workspace_root: Path) -> bool:
    try:
        relative_parts = path.relative_to(workspace_root).parts
    except ValueError:
        return True

    return any(
        part in SYSTEM_OR_CACHE_DIRS or part.startswith(".")
        for part in relative_parts
    )


def _read_file_success_result(
    *,
    path: str,
    total_lines: int,
    returned_lines: list[tuple[int, str]],
    content: str,
    matched: bool,
    matched_count: int | None,
    matches: list[dict[str, Any]],
    truncated: bool,
    error: str | None = None,
    needs_followup: bool = False,
    followup_hint: str | None = None,
    fallback_start_line: int | None = None,
    fallback_end_line: int | None = None,
) -> dict[str, Any]:
    if returned_lines:
        start_line = returned_lines[0][0]
        end_line = returned_lines[-1][0]
    else:
        start_line = fallback_start_line
        end_line = fallback_end_line

    return {
        "tool_name": "read_file",
        "path": path,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": total_lines,
        "content": content,
        "matched": matched,
        "matched_count": matched_count,
        "matches": matches,
        "needs_followup": needs_followup,
        "followup_hint": followup_hint,
        "truncated": truncated,
        "error": error,
    }


def _read_file_error_result(
    error: str,
    *,
    root: Path,
    path: str,
    total_lines: int = 0,
) -> dict[str, Any]:
    return {
        "tool_name": "read_file",
        "root": str(root),
        "path": path,
        "start_line": None,
        "end_line": None,
        "total_lines": total_lines,
        "content": "",
        "matched": False,
        "matched_count": 0,
        "matches": [],
        "needs_followup": False,
        "followup_hint": None,
        "truncated": False,
        "error": error,
    }
# === read_file helpers end ===========================================


# === shared path and result helpers begin ===========================================
def _resolve_workspace_root(value: Any) -> Path:
    if value is None:
        return Path.cwd().resolve()
    return Path(str(value)).resolve()


def _resolve_target_path(workspace_root: Path, path: str | None) -> Path:
    if path is None or not path.strip():
        target = workspace_root
    else:
        raw_path = Path(path)
        target = raw_path if raw_path.is_absolute() else workspace_root / raw_path
    target = target.resolve()
    if not _is_relative_to(target, workspace_root):
        raise ValueError("path must stay inside the workspace root")
    return target


def _display_path(path: Path, workspace_root: Path) -> str:
    try:
        relative = path.relative_to(workspace_root)
    except ValueError:
        return path.as_posix()
    if not relative.parts:
        return "."
    return relative.as_posix()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _error_result(tool_name: str, error: str, *, root: Path) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "root": str(root),
        "entries": [],
        "truncated": False,
        "error": error,
    }


def _not_implemented_result(tool_name: str) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "status": "not_implemented",
        "result": None,
        "error": f"{tool_name} is registered but not implemented yet",
    }
# === shared path and result helpers end ===========================================
