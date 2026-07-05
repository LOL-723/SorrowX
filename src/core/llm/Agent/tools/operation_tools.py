import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_MAX_ENTRIES = 40
MAX_CHILD_FILES_BEFORE_COLLAPSE = 5
MAX_CHILD_DIRS_BEFORE_COLLAPSE = 5
MAX_READ_FILE_LINES = 200
MAX_KEYWORD_MATCHES = 20
MAX_MATCH_PREVIEW_CHARS = 160
READ_FILE_SINGLE_MATCH_CONTEXT_LINES = 10
MAX_WRITE_FILE_BYTES = 1_048_576
MAX_RUN_TESTS_OUTPUT_CHARS = 12_000
DEFAULT_RUN_TESTS_TIMEOUT_SECONDS = 30
SUPPORTED_WRITE_FILE_SUFFIXES = {".cpp", ".md", ".py", ".txt"}

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
    keyword: str | None = None,
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

    normalized_keyword = _normalize_list_dir_keyword(keyword)
    if normalized_keyword is not None:
        matches, search_truncated = _search_paths_by_name(
            target,
            workspace_root=workspace_root,
            keyword=normalized_keyword,
            max_matches=entry_limit,
        )
        return {
            "tool_name": "list_dir",
            "root": str(workspace_root),
            "target": _display_path(target, workspace_root),
            "keyword": normalized_keyword,
            "matches": matches,
            "matched_count": len(matches),
            "search_truncated": search_truncated,
            "entries": [],
            "direct_entries_complete": True,
            "tree_truncated": False,
            "truncated": search_truncated,
            "completion_notes": _list_dir_search_notes(
                keyword=normalized_keyword,
                search_truncated=search_truncated,
            ),
        }

    entries, direct_entries_complete, tree_truncated = _directory_entries(
        target,
        workspace_root=workspace_root,
        requested_root=target,
        entry_limit=entry_limit,
    )
    truncated = not direct_entries_complete or tree_truncated

    return {
        "tool_name": "list_dir",
        "root": str(workspace_root),
        "target": _display_path(target, workspace_root),
        "entries": entries,
        "direct_entries_complete": direct_entries_complete,
        "tree_truncated": tree_truncated,
        "truncated": truncated,
        "completion_notes": _list_dir_completion_notes(
            direct_entries_complete=direct_entries_complete,
            tree_truncated=tree_truncated,
        ),
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


def write_file_tool(
    path: str | None = None,
    content: str | None = None,
    old_content: str | None = None,
    new_create_confirm: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    workspace_root = _resolve_workspace_root(kwargs.get("workspace_root"))
    try:
        target = _prepare_write_target(
            workspace_root=workspace_root,
            path=path,
            content=content,
            new_create_confirm=new_create_confirm,
        )
        target_exists = target.exists()

        if target_exists:
            return _modify_existing_file(
                target=target,
                workspace_root=workspace_root,
                content=content,
                old_content=old_content,
            )

        return _create_new_file(
            target=target,
            workspace_root=workspace_root,
            content=content,
        )
    except ValueError as exc:
        return _write_file_error_result(
            str(exc),
            root=workspace_root,
            path=path or "",
        )


def run_tests_tool(
    command: str | None = None,
    timeout_seconds: int | str | None = DEFAULT_RUN_TESTS_TIMEOUT_SECONDS,
    **kwargs: Any,
) -> dict[str, Any]:
    workspace_root = _resolve_workspace_root(kwargs.get("workspace_root"))
    try:
        argv, display_command = _prepare_run_tests_command(
            command=command,
            workspace_root=workspace_root,
        )
        timeout = _normalize_timeout_seconds(timeout_seconds)
        completed = subprocess.run(
            argv,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except ValueError as exc:
        return _run_tests_result(
            command=command or "",
            cwd=workspace_root,
            status="failed",
            returncode=None,
            stdout="",
            stderr="",
            error=str(exc),
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _run_tests_result(
            command=display_command if "display_command" in locals() else (command or ""),
            cwd=workspace_root,
            status="failed",
            returncode=None,
            stdout=_coerce_process_output(exc.stdout),
            stderr=_coerce_process_output(exc.stderr),
            error=f"command timed out after {timeout} seconds",
            timed_out=True,
        )
    except OSError as exc:
        return _run_tests_result(
            command=display_command if "display_command" in locals() else (command or ""),
            cwd=workspace_root,
            status="failed",
            returncode=None,
            stdout="",
            stderr="",
            error=str(exc),
            timed_out=False,
        )

    status = "success" if completed.returncode == 0 else "failed"
    error = None if completed.returncode == 0 else f"command exited with code {completed.returncode}"
    return _run_tests_result(
        command=display_command,
        cwd=workspace_root,
        status=status,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        error=error,
        timed_out=False,
    )
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

    @property
    def remaining(self) -> int:
        return max(self.limit - self.count, 0)


def _normalize_max_entries(value: int | None) -> int:
    return DEFAULT_MAX_ENTRIES


def _normalize_list_dir_keyword(keyword: str | None) -> str | None:
    if keyword is None:
        return None
    stripped_keyword = keyword.strip()
    return stripped_keyword or None


def _search_paths_by_name(
    directory: Path,
    *,
    workspace_root: Path,
    keyword: str,
    max_matches: int,
) -> tuple[list[dict[str, Any]], bool]:
    matches: list[dict[str, Any]] = []
    search_truncated = False
    lowered_keyword = keyword.lower()

    for child in _walk_included_paths(directory):
        if lowered_keyword not in child.name.lower():
            continue
        if len(matches) >= max_matches:
            search_truncated = True
            break
        matches.append(_path_entry(child, workspace_root))

    return matches, search_truncated


def _walk_included_paths(directory: Path):
    try:
        children = [child for child in directory.iterdir() if _should_include_path(child)]
    except OSError:
        return
    for child in _sort_children(children):
        yield child
        if child.is_dir():
            yield from _walk_included_paths(child)


def _directory_entries(
    directory: Path,
    *,
    workspace_root: Path,
    requested_root: Path,
    entry_limit: int,
) -> tuple[list[dict[str, Any]], bool, bool]:
    children, error_entry = _included_children(directory, workspace_root)
    if error_entry is not None:
        return [error_entry], True, False

    ordered_children = _sort_children(children)
    direct_children = ordered_children[:entry_limit]
    direct_entries_complete = len(ordered_children) <= entry_limit
    entries = [
        _path_entry(child, workspace_root)
        for child in direct_children
    ]

    if not direct_entries_complete:
        return entries, False, True

    budget = _EntryBudget(entry_limit - len(entries))
    tree_truncated = _expand_child_directories(
        entries=entries,
        children=direct_children,
        workspace_root=workspace_root,
        requested_root=requested_root,
        budget=budget,
    )
    return entries, True, tree_truncated or budget.truncated


def _nested_directory_entries(
    directory: Path,
    *,
    workspace_root: Path,
    requested_root: Path,
    budget: _EntryBudget,
) -> tuple[list[dict[str, Any]], bool]:
    children, error_entry = _included_children(directory, workspace_root)
    if error_entry is not None:
        return [error_entry], False

    entries: list[dict[str, Any]] = []
    ordered_children = _sort_children(children)
    direct_entries_truncated = False
    for child in ordered_children:
        if not budget.claim():
            direct_entries_truncated = True
            break
        entries.append(_path_entry(child, workspace_root))

    if direct_entries_truncated:
        return entries, True

    tree_truncated = _expand_child_directories(
        entries=entries,
        children=ordered_children,
        workspace_root=workspace_root,
        requested_root=requested_root,
        budget=budget,
    )
    return entries, tree_truncated or budget.truncated


def _expand_child_directories(
    *,
    entries: list[dict[str, Any]],
    children: list[Path],
    workspace_root: Path,
    requested_root: Path,
    budget: _EntryBudget,
) -> bool:
    tree_truncated = False
    for entry, child in zip(entries, children, strict=False):
        if not child.is_dir():
            continue
        if _should_collapse_directory(child, requested_root):
            entry["truncated"] = True
            tree_truncated = True
            continue
        if budget.remaining <= 0:
            if _has_included_children(child):
                entry["truncated"] = True
                tree_truncated = True
            continue

        nested_entries, nested_truncated = _nested_directory_entries(
            child,
            workspace_root=workspace_root,
            requested_root=requested_root,
            budget=budget,
        )
        if nested_entries:
            entry["entries"] = nested_entries
        if nested_truncated:
            entry["truncated"] = True
        tree_truncated = tree_truncated or nested_truncated
    return tree_truncated


def _included_children(
    directory: Path,
    workspace_root: Path,
) -> tuple[list[Path], dict[str, Any] | None]:
    try:
        return [child for child in directory.iterdir() if _should_include_path(child)], None
    except OSError as exc:
        return [], {
            "path": _display_path(directory, workspace_root),
            "type": "directory",
            "error": str(exc),
        }


def _path_entry(path: Path, workspace_root: Path) -> dict[str, Any]:
    return {
        "path": _display_path(path, workspace_root),
        "type": "directory" if path.is_dir() else "file",
    }


def _has_included_children(directory: Path) -> bool:
    try:
        return any(_should_include_path(child) for child in directory.iterdir())
    except OSError:
        return False


def _list_dir_completion_notes(
    *,
    direct_entries_complete: bool,
    tree_truncated: bool,
) -> list[str]:
    notes: list[str] = []
    if not direct_entries_complete:
        notes.append(
            "direct_entries_complete=false: the target directory's direct files/directories are incomplete; call list_dir with a narrower path or larger max_entries."
        )
    if tree_truncated:
        notes.append(
            "tree_truncated=true: some descendant files/directories are not shown; call list_dir on a truncated directory path to inspect it."
        )
    return notes


def _list_dir_search_notes(
    *,
    keyword: str,
    search_truncated: bool,
) -> list[str]:
    if search_truncated:
        return [
            f"search_truncated=true: more paths matching {keyword!r} exist but were not returned; call list_dir with a narrower path or larger max_entries."
        ]
    return []


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


# === write_file helpers begin ===========================================
def _prepare_write_target(
    *,
    workspace_root: Path,
    path: str | None,
    content: str | None,
    new_create_confirm: bool,
) -> Path:
    if path is None or not path.strip():
        raise ValueError("path is required")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    if not isinstance(new_create_confirm, bool):
        raise ValueError("new_create_confirm must be a boolean")

    target = _resolve_target_path(workspace_root, path)
    display_path = _display_path(target, workspace_root)
    if _has_write_filtered_path_segment(target, workspace_root):
        raise ValueError(f"path is filtered: {display_path}")
    if target.exists() and not target.is_file():
        raise ValueError(f"path is not a file: {display_path}")
    if target.suffix.lower() not in SUPPORTED_WRITE_FILE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_WRITE_FILE_SUFFIXES))
        raise ValueError(f"unsupported file extension: {target.suffix}; supported: {supported}")
    if not target.parent.exists():
        raise ValueError(f"parent directory does not exist: {_display_path(target.parent, workspace_root)}")
    if target.parent.is_file():
        raise ValueError(f"parent path is not a directory: {_display_path(target.parent, workspace_root)}")
    if not target.exists() and new_create_confirm is not True:
        raise ValueError("new_create_confirm must be true to create a new file")

    _validate_write_size(content)
    return target


def _create_new_file(
    *,
    target: Path,
    workspace_root: Path,
    content: str,
) -> dict[str, Any]:
    display_path = _display_path(target, workspace_root)
    new_end_line = _end_line_for_content(start_line=1, content=content)
    _atomic_write_text(target, content)
    return _write_file_success_result(
        path=display_path,
        created=True,
        old_start_line=None,
        old_end_line=None,
        old_content=None,
        new_start_line=1,
        new_end_line=new_end_line,
        new_content=content,
        bytes_written=_content_size(content),
    )


def _modify_existing_file(
    *,
    target: Path,
    workspace_root: Path,
    content: str,
    old_content: str | None,
) -> dict[str, Any]:
    display_path = _display_path(target, workspace_root)
    current_text = _read_utf8_text(target)
    if current_text == "":
        new_end_line = _end_line_for_content(start_line=1, content=content)
        _validate_write_size(content)
        _atomic_write_text(target, content)
        return _write_file_success_result(
            path=display_path,
            created=False,
            old_start_line=1,
            old_end_line=0,
            old_content="",
            new_start_line=1,
            new_end_line=new_end_line,
            new_content=content,
            bytes_written=_content_size(content),
        )

    if old_content is None or not old_content:
        raise ValueError("old_content is required when modifying an existing file")

    match_positions = _find_old_content_positions(current_text, old_content)
    if not match_positions:
        raise ValueError("old_content does not match current file; reread before writing")
    if len(match_positions) > 1:
        raise ValueError("old_content is ambiguous; provide a more specific old_content")

    old_start_index = match_positions[0]
    old_end_index = old_start_index + len(old_content)
    old_start_line = _line_number_at_index(current_text, old_start_index)
    old_end_line = _end_line_for_content(
        start_line=old_start_line,
        content=old_content,
    )
    new_start_line = old_start_line
    new_end_line = _end_line_for_content(
        start_line=new_start_line,
        content=content,
    )
    updated_text = current_text[:old_start_index] + content + current_text[old_end_index:]

    _validate_write_size(updated_text)
    _atomic_write_text(target, updated_text)
    return _write_file_success_result(
        path=display_path,
        created=False,
        old_start_line=old_start_line,
        old_end_line=old_end_line,
        old_content=old_content,
        new_start_line=new_start_line,
        new_end_line=new_end_line,
        new_content=content,
        bytes_written=_content_size(updated_text),
    )


def _read_utf8_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8: {path.name}") from exc
    except OSError as exc:
        raise ValueError(f"file read failed: {exc}") from exc


def _validate_write_size(content: str) -> None:
    if _content_size(content) > MAX_WRITE_FILE_BYTES:
        raise ValueError("content is too large; maximum write size is 1MB")


def _content_size(content: str) -> int:
    return len(content.encode("utf-8"))


def _has_write_filtered_path_segment(path: Path, workspace_root: Path) -> bool:
    try:
        relative_parts = path.relative_to(workspace_root).parts
    except ValueError:
        return True

    return any(
        part in SYSTEM_OR_CACHE_DIRS or part.startswith(".")
        for part in relative_parts
    )


def _find_old_content_positions(text: str, old_content: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        index = text.find(old_content, start)
        if index == -1:
            break
        positions.append(index)
        start = index + len(old_content)
    return positions


def _line_number_at_index(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _end_line_for_content(*, start_line: int, content: str) -> int:
    line_count = _content_line_count(content)
    if line_count == 0:
        return start_line - 1
    return start_line + line_count - 1


def _content_line_count(content: str) -> int:
    if content == "":
        return 0
    return len(content.splitlines()) or 1


def _atomic_write_text(path: Path, content: str) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=path.parent,
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        temp_path.replace(path)
    except OSError as exc:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ValueError(f"atomic write failed: {exc}") from exc


def _write_file_success_result(
    *,
    path: str,
    created: bool,
    old_start_line: int | None,
    old_end_line: int | None,
    old_content: str | None,
    new_start_line: int,
    new_end_line: int,
    new_content: str,
    bytes_written: int,
) -> dict[str, Any]:
    return {
        "tool_name": "write_file",
        "path": path,
        "status": "success",
        "created": created,
        "old_start_line": old_start_line,
        "old_end_line": old_end_line,
        "old_content": old_content,
        "new_start_line": new_start_line,
        "new_end_line": new_end_line,
        "new_content": new_content,
        "bytes_written": bytes_written,
        "error": None,
    }


def _write_file_error_result(
    error: str,
    *,
    root: Path,
    path: str,
) -> dict[str, Any]:
    return {
        "tool_name": "write_file",
        "root": str(root),
        "path": path,
        "status": "failed",
        "created": False,
        "old_start_line": None,
        "old_end_line": None,
        "old_content": None,
        "new_start_line": None,
        "new_end_line": None,
        "new_content": None,
        "bytes_written": 0,
        "error": error,
    }
# === write_file helpers end ===========================================


# === run_tests helpers begin ===========================================
def _prepare_run_tests_command(
    *,
    command: str | None,
    workspace_root: Path,
) -> tuple[list[str], str]:
    if command is None or not command.strip():
        argv = [sys.executable, "-m", "pytest"]
        return argv, _display_command(argv, workspace_root)

    if _contains_shell_control(command):
        raise ValueError("shell control operators are not supported")

    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"command parse failed: {exc}") from exc
    if not parts:
        raise ValueError("command is required")

    executable = parts[0].lower()
    args = parts[1:]
    if executable in {"python", "python.exe"}:
        argv = [sys.executable, *_route_run_tests_args(args, workspace_root=workspace_root)]
    elif executable in {"pytest", "pytest.exe"}:
        argv = [
            sys.executable,
            "-m",
            "pytest",
            *_route_run_tests_args(args, workspace_root=workspace_root),
        ]
    else:
        raise ValueError("unsupported command; use python ... or pytest ...")

    return argv, _display_command(argv, workspace_root)


def _contains_shell_control(command: str) -> bool:
    return any(token in command for token in ["&&", "||", "|", ";", ">", "<", "`"])


def _route_run_tests_args(args: list[str], *, workspace_root: Path) -> list[str]:
    routed_args: list[str] = []
    for arg in args:
        if arg.startswith("-") or not _looks_like_path(arg):
            routed_args.append(arg)
            continue
        target = _resolve_target_path(workspace_root, arg)
        routed_args.append(_display_path(target, workspace_root))
    return routed_args


def _looks_like_path(value: str) -> bool:
    return (
        "/" in value
        or "\\" in value
        or Path(value).suffix.lower() in SUPPORTED_WRITE_FILE_SUFFIXES
    )


def _normalize_timeout_seconds(value: int | str | None) -> int:
    if value is None:
        return DEFAULT_RUN_TESTS_TIMEOUT_SECONDS
    if isinstance(value, bool):
        raise ValueError("timeout_seconds must be an integer")
    if isinstance(value, int):
        timeout = value
    elif isinstance(value, str) and value.strip().isdecimal():
        timeout = int(value.strip())
    else:
        raise ValueError("timeout_seconds must be an integer")
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout_seconds must be between 1 and 120")
    return timeout


def _display_command(argv: list[str], workspace_root: Path) -> str:
    display_parts: list[str] = []
    for index, part in enumerate(argv):
        if index == 0 and Path(part).resolve() == Path(sys.executable).resolve():
            display_parts.append("python")
        else:
            display_parts.append(part)
    return " ".join(shlex.quote(part) for part in display_parts)


def _coerce_process_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _run_tests_result(
    *,
    command: str,
    cwd: Path,
    status: str,
    returncode: int | None,
    stdout: str,
    stderr: str,
    error: str | None,
    timed_out: bool,
) -> dict[str, Any]:
    limited_stdout, stdout_truncated = _limit_output(stdout)
    limited_stderr, stderr_truncated = _limit_output(stderr)
    return {
        "tool_name": "run_tests",
        "command": command,
        "cwd": str(cwd),
        "status": status,
        "returncode": returncode,
        "stdout": limited_stdout,
        "stderr": limited_stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "timed_out": timed_out,
        "error": error,
    }


def _limit_output(value: str) -> tuple[str, bool]:
    if len(value) <= MAX_RUN_TESTS_OUTPUT_CHARS:
        return value, False
    return value[:MAX_RUN_TESTS_OUTPUT_CHARS], True
# === run_tests helpers end ===========================================


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
        if raw_path.is_absolute():
            target = raw_path
        elif raw_path.root and not raw_path.drive:
            target = workspace_root / Path(*raw_path.parts[1:])
        else:
            target = workspace_root / raw_path
            if not target.exists():
                routed_target = _route_relative_path_by_nearest_directory(
                    workspace_root=workspace_root,
                    raw_path=raw_path,
                )
                if routed_target is not None:
                    target = routed_target
    target = target.resolve()
    if not _is_relative_to(target, workspace_root):
        raise ValueError("path must stay inside the workspace root")
    return target


def _route_relative_path_by_nearest_directory(
    *,
    workspace_root: Path,
    raw_path: Path,
) -> Path | None:
    parts = raw_path.parts
    if not parts:
        return None
    first_part = parts[0]
    if first_part in {".", ".."} or first_part.startswith("."):
        return None

    matched_directory = _nearest_directory_named(
        workspace_root=workspace_root,
        directory_name=first_part,
    )
    if matched_directory is None:
        return None
    return matched_directory.joinpath(*parts[1:])


def _nearest_directory_named(
    *,
    workspace_root: Path,
    directory_name: str,
) -> Path | None:
    lowered_name = directory_name.lower()
    matches = [
        path
        for path in _walk_included_paths(workspace_root)
        if path.is_dir() and path.name.lower() == lowered_name
    ]
    if not matches:
        return None
    return min(matches, key=lambda path: _directory_route_key(path, workspace_root))


def _directory_route_key(path: Path, workspace_root: Path) -> tuple[int, tuple[int, ...], str]:
    relative = path.relative_to(workspace_root)
    depth = len(relative.parts)
    priority_path = tuple(_route_part_priority(part) for part in relative.parts)
    return (depth, priority_path, relative.as_posix().lower())


def _route_part_priority(part: str) -> int:
    if part == "src":
        return 0
    if part == "core":
        return 1
    return 2


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
