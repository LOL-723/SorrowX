import tempfile
import unittest
from pathlib import Path

from llm.Agent.tools.operation_tools import (
    list_dir_tool,
    patch_file_tool,
    read_file_tool,
    run_tests_tool,
)
from llm.tools import TOOL_REGISTRY


class OperationToolsTest(unittest.TestCase):
    def test_operation_tools_are_registered(self) -> None:
        self.assertIs(TOOL_REGISTRY["list_dir"], list_dir_tool)
        self.assertIs(TOOL_REGISTRY["read_file"], read_file_tool)
        self.assertIs(TOOL_REGISTRY["patch_file"], patch_file_tool)
        self.assertIs(TOOL_REGISTRY["run_tests"], run_tests_tool)

    def test_list_dir_prioritizes_src_and_preserves_source_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_workspace(workspace)

            result = list_dir_tool(workspace_root=workspace)
            entries = result["entries"]
            flattened = self._flatten(entries)
            paths = [entry["path"] for entry in flattened]

        self.assertEqual(result["tool_name"], "list_dir")
        self.assertEqual(entries[0]["path"], "src")
        self.assertLessEqual(len(flattened), 20)
        self.assertIn("src/core/agent_loop.py", paths)
        self.assertIn("src/core/test_agent_runner.py", paths)
        self.assertIn("src/core/tools.py", paths)
        for entry in flattened:
            self.assertIn("path", entry)
            self.assertIn("type", entry)

    def test_list_dir_filters_hidden_and_cache_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_workspace(workspace)

            result = list_dir_tool(workspace_root=workspace)
            flattened = self._flatten(result["entries"])
            paths = [entry["path"] for entry in flattened]

        self.assertNotIn(".git", paths)
        self.assertNotIn(".venv", paths)
        self.assertNotIn("node_modules", paths)
        self.assertNotIn("__pycache__", paths)
        self.assertNotIn(".secret", paths)

    def test_list_dir_returns_config_files_as_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_workspace(workspace)

            result = list_dir_tool(workspace_root=workspace)
            entries_by_path = {
                entry["path"]: entry for entry in self._flatten(result["entries"])
            }

        self.assertEqual(entries_by_path[".env"], {"path": ".env", "type": "file"})
        self.assertEqual(
            entries_by_path[".gitignore"],
            {"path": ".gitignore", "type": "file"},
        )

    def test_list_dir_collapses_large_child_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_workspace(workspace)

            result = list_dir_tool(workspace_root=workspace)
            entries_by_path = {
                entry["path"]: entry for entry in self._flatten(result["entries"])
            }

        self.assertEqual(entries_by_path["src/large_files"]["type"], "directory")
        self.assertTrue(entries_by_path["src/large_files"]["truncated"])
        self.assertNotIn("entries", entries_by_path["src/large_files"])
        self.assertEqual(entries_by_path["src/large_dirs"]["type"], "directory")
        self.assertTrue(entries_by_path["src/large_dirs"]["truncated"])
        self.assertNotIn("entries", entries_by_path["src/large_dirs"])

    def test_list_dir_respects_max_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_workspace(workspace)

            result = list_dir_tool(max_entries=3, workspace_root=workspace)
            flattened = self._flatten(result["entries"])

        self.assertLessEqual(len(flattened), 3)
        self.assertTrue(result["truncated"])

    def test_list_dir_rejects_paths_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir()
            outside = Path(temp_dir) / "outside"
            outside.mkdir()

            result = list_dir_tool(str(outside), workspace_root=workspace)

        self.assertEqual(result["tool_name"], "list_dir")
        self.assertEqual(result["entries"], [])
        self.assertIn("workspace root", result["error"])

    def test_list_dir_rejects_filtered_target_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_workspace(workspace)

            result = list_dir_tool(".git", workspace_root=workspace)

        self.assertEqual(result["entries"], [])
        self.assertIn("filtered", result["error"])

    def test_read_file_reads_line_range_with_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_read_workspace(workspace)

            result = read_file_tool(
                "src/core/read_target.py",
                start_line=2,
                end_line=4,
                workspace_root=workspace,
            )

        self.assertEqual(result["tool_name"], "read_file")
        self.assertEqual(result["path"], "src/core/read_target.py")
        self.assertEqual(result["start_line"], 2)
        self.assertEqual(result["end_line"], 4)
        self.assertEqual(result["total_lines"], 30)
        self.assertEqual(
            result["content"],
            "2: line 2\n3: line 3\n4: line 4",
        )
        self.assertTrue(result["matched"])
        self.assertIsNone(result["matched_count"])
        self.assertFalse(result["truncated"])
        self.assertIsNone(result["error"])

    def test_read_file_single_keyword_match_returns_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_read_workspace(workspace)

            result = read_file_tool(
                "src/core/read_target.py",
                keyword="unique_loop_marker",
                workspace_root=workspace,
            )

        self.assertTrue(result["matched"])
        self.assertEqual(result["matched_count"], 1)
        self.assertFalse(result["needs_followup"])
        self.assertEqual(result["matches"], [{"line": 15, "preview": "unique_loop_marker"}])
        self.assertIn("5: line 5", result["content"])
        self.assertIn("15: unique_loop_marker", result["content"])
        self.assertIn("25: line 25", result["content"])

    def test_read_file_multiple_keyword_matches_return_candidates_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_read_workspace(workspace)

            result = read_file_tool(
                "src/core/read_target.py",
                keyword="repeat_marker",
                workspace_root=workspace,
            )

        self.assertTrue(result["matched"])
        self.assertEqual(result["matched_count"], 3)
        self.assertEqual(result["content"], "")
        self.assertTrue(result["needs_followup"])
        self.assertEqual(
            [match["line"] for match in result["matches"]],
            [7, 18, 27],
        )
        self.assertIn("start_line/end_line", result["followup_hint"])

    def test_read_file_keyword_search_respects_line_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_read_workspace(workspace)

            result = read_file_tool(
                "src/core/read_target.py",
                keyword="repeat_marker",
                start_line=1,
                end_line=10,
                workspace_root=workspace,
            )

        self.assertTrue(result["matched"])
        self.assertEqual(result["matched_count"], 1)
        self.assertFalse(result["needs_followup"])
        self.assertEqual(result["matches"], [{"line": 7, "preview": "repeat_marker first"}])
        self.assertIn("7: repeat_marker first", result["content"])

    def test_read_file_keyword_not_found_returns_structured_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_read_workspace(workspace)

            result = read_file_tool(
                "src/core/read_target.py",
                keyword="missing_keyword",
                workspace_root=workspace,
            )

        self.assertFalse(result["matched"])
        self.assertEqual(result["matched_count"], 0)
        self.assertEqual(result["content"], "")
        self.assertEqual(result["error"], "keyword not found")

    def test_read_file_rejects_invalid_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            outside = Path(temp_dir) / "outside.py"
            workspace.mkdir()
            outside.write_text("outside", encoding="utf-8")
            self._make_read_workspace(workspace)

            outside_result = read_file_tool(str(outside), workspace_root=workspace)
            directory_result = read_file_tool("src/core", workspace_root=workspace)
            missing_result = read_file_tool("missing.py", workspace_root=workspace)
            hidden_result = read_file_tool(".env", workspace_root=workspace)
            cache_result = read_file_tool(
                "__pycache__/ignored.py",
                workspace_root=workspace,
            )

        self.assertIn("workspace root", outside_result["error"])
        self.assertIn("not a file", directory_result["error"])
        self.assertIn("does not exist", missing_result["error"])
        self.assertIn("filtered", hidden_result["error"])
        self.assertIn("filtered", cache_result["error"])

    def test_read_file_truncates_large_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            target = workspace / "src" / "core" / "large.py"
            target.parent.mkdir(parents=True)
            target.write_text(
                "\n".join(f"line {index}" for index in range(1, 251)),
                encoding="utf-8",
            )

            result = read_file_tool("src/core/large.py", workspace_root=workspace)

        self.assertTrue(result["truncated"])
        self.assertEqual(result["start_line"], 1)
        self.assertEqual(result["end_line"], 200)
        self.assertEqual(len(result["content"].splitlines()), 200)

    def test_unimplemented_tools_return_stable_result(self) -> None:
        self.assertEqual(patch_file_tool()["status"], "not_implemented")
        self.assertEqual(run_tests_tool()["status"], "not_implemented")

    @staticmethod
    def _make_workspace(workspace: Path) -> None:
        src_core = workspace / "src" / "core"
        src_core.mkdir(parents=True)
        (src_core / "agent_loop.py").write_text("print('agent')", encoding="utf-8")
        (src_core / "test_agent_runner.py").write_text("print('test')", encoding="utf-8")
        (src_core / "tools.py").write_text("print('tools')", encoding="utf-8")

        large_files = workspace / "src" / "large_files"
        large_files.mkdir()
        for index in range(6):
            (large_files / f"file_{index}.py").write_text("", encoding="utf-8")

        large_dirs = workspace / "src" / "large_dirs"
        large_dirs.mkdir()
        for index in range(6):
            (large_dirs / f"dir_{index}").mkdir()

        (workspace / ".env").write_text("SECRET=value", encoding="utf-8")
        (workspace / ".gitignore").write_text(".venv", encoding="utf-8")
        (workspace / ".secret").write_text("hidden", encoding="utf-8")
        (workspace / "README.md").write_text("readme", encoding="utf-8")

        for directory_name in [".git", ".venv", "__pycache__", "node_modules"]:
            directory = workspace / directory_name
            directory.mkdir()
            (directory / "ignored.py").write_text("", encoding="utf-8")

    @staticmethod
    def _make_read_workspace(workspace: Path) -> None:
        src_core = workspace / "src" / "core"
        src_core.mkdir(parents=True)
        lines = [f"line {index}" for index in range(1, 31)]
        lines[6] = "repeat_marker first"
        lines[14] = "unique_loop_marker"
        lines[17] = "repeat_marker second"
        lines[26] = "repeat_marker third"
        (src_core / "read_target.py").write_text("\n".join(lines), encoding="utf-8")
        (workspace / ".env").write_text("SECRET=value", encoding="utf-8")
        cache_dir = workspace / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "ignored.py").write_text("ignored", encoding="utf-8")

    @classmethod
    def _flatten(cls, entries: list[dict]) -> list[dict]:
        flattened: list[dict] = []
        for entry in entries:
            flattened.append(entry)
            child_entries = entry.get("entries", [])
            if child_entries:
                flattened.extend(cls._flatten(child_entries))
        return flattened


if __name__ == "__main__":
    unittest.main()
