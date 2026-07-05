import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm.Agent.tools.operation_tools import (
    list_dir_tool,
    read_file_tool,
    run_tests_tool,
    write_file_tool,
)
from llm.tools import TOOL_REGISTRY


class OperationToolsTest(unittest.TestCase):
    def test_operation_tools_are_registered(self) -> None:
        self.assertIs(TOOL_REGISTRY["list_dir"], list_dir_tool)
        self.assertIs(TOOL_REGISTRY["read_file"], read_file_tool)
        self.assertIs(TOOL_REGISTRY["write_file"], write_file_tool)
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

    def test_list_dir_ignores_lower_max_entries_and_uses_fixed_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            for index in range(45):
                (workspace / f"file_{index:02d}.txt").write_text("", encoding="utf-8")

            result = list_dir_tool(max_entries=20, workspace_root=workspace)
            flattened = self._flatten(result["entries"])

        self.assertEqual(len(flattened), 40)
        self.assertTrue(result["truncated"])
        self.assertFalse(result["direct_entries_complete"])
        self.assertTrue(result["tree_truncated"])
        self.assertTrue(
            any("direct_entries_complete=false" in note for note in result["completion_notes"])
        )

    def test_list_dir_preserves_all_direct_entries_before_expanding_children(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_core_like_workspace(workspace)

            result = list_dir_tool("src/core", workspace_root=workspace)
            direct_paths = [entry["path"] for entry in result["entries"]]

        self.assertTrue(result["direct_entries_complete"])
        self.assertTrue(result["tree_truncated"])
        self.assertEqual(
            direct_paths,
            [
                "src/core/plan.md",
                "src/core/api",
                "src/core/daemon",
                "src/core/ipc",
                "src/core/llm",
                "src/core/practice",
                "src/core/schemas",
                "src/core/scripts",
                "src/core/set",
                "src/core/storage",
                "src/core/tests",
                "src/core/trace",
            ],
        )
        self.assertTrue(
            any("tree_truncated=true" in note for note in result["completion_notes"])
        )

    def test_list_dir_treats_windows_rooted_path_as_workspace_relative(self) -> None:
        rooted_path = Path("/src/core/tests")
        if rooted_path.is_absolute() or not rooted_path.root or rooted_path.drive:
            self.skipTest("platform does not treat /src as drive-rooted relative path")

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_core_like_workspace(workspace)

            result = list_dir_tool("/src/core/tests", workspace_root=workspace)
            direct_paths = [entry["path"] for entry in result["entries"]]

        self.assertEqual(result["target"], "src/core/tests")
        self.assertEqual(
            direct_paths,
            [
                "src/core/tests/file_0.py",
                "src/core/tests/file_1.py",
                "src/core/tests/file_2.py",
            ],
        )
        self.assertNotIn("error", result)

    def test_list_dir_routes_missing_top_level_directory_to_nearest_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_core_like_workspace(workspace)

            result = list_dir_tool("tests", workspace_root=workspace)
            direct_paths = [entry["path"] for entry in result["entries"]]

        self.assertEqual(result["target"], "src/core/tests")
        self.assertEqual(
            direct_paths,
            [
                "src/core/tests/file_0.py",
                "src/core/tests/file_1.py",
                "src/core/tests/file_2.py",
            ],
        )
        self.assertNotIn("error", result)

    def test_list_dir_route_prefers_shallowest_matching_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_core_like_workspace(workspace)
            deeper_tests = workspace / "src" / "core" / "llm" / "deep" / "tests"
            deeper_tests.mkdir(parents=True)
            (deeper_tests / "wrong.py").write_text("", encoding="utf-8")

            result = list_dir_tool("tests", workspace_root=workspace)

        self.assertEqual(result["target"], "src/core/tests")

    def test_list_dir_keyword_search_finds_file_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_workspace(workspace)

            result = list_dir_tool(
                keyword="test_agent_runner.py",
                workspace_root=workspace,
            )

        self.assertEqual(result["keyword"], "test_agent_runner.py")
        self.assertEqual(result["matched_count"], 1)
        self.assertFalse(result["search_truncated"])
        self.assertEqual(
            result["matches"],
            [{"path": "src/core/test_agent_runner.py", "type": "file"}],
        )
        self.assertEqual(result["entries"], [])
        self.assertFalse(result["tree_truncated"])

    def test_list_dir_keyword_search_respects_limits_and_filters(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            src_core = workspace / "src" / "core"
            src_core.mkdir(parents=True)
            for index in range(3):
                (src_core / f"target_{index}.py").write_text("", encoding="utf-8")
            hidden_dir = workspace / ".hidden"
            hidden_dir.mkdir()
            (hidden_dir / "target_hidden.py").write_text("", encoding="utf-8")
            cache_dir = workspace / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "target_cache.py").write_text("", encoding="utf-8")

            result = list_dir_tool(
                keyword="target",
                max_entries=2,
                workspace_root=workspace,
            )
            paths = [match["path"] for match in result["matches"]]

        self.assertEqual(result["matched_count"], 3)
        self.assertFalse(result["search_truncated"])
        self.assertEqual(
            paths,
            [
                "src/core/target_0.py",
                "src/core/target_1.py",
                "src/core/target_2.py",
            ],
        )
        self.assertNotIn(".hidden/target_hidden.py", paths)
        self.assertNotIn("__pycache__/target_cache.py", paths)

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

    def test_write_file_requires_confirm_for_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            result = write_file_tool(
                "notes.txt",
                content="hello",
                workspace_root=workspace,
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("new_create_confirm", result["error"])
        self.assertFalse((workspace / "notes.txt").exists())

    def test_write_file_creates_supported_text_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            cases = {
                "notes.md": "# Title\nbody",
                "notes.txt": "plain text",
                "main.cpp": "int main() { return 0; }\n",
                "script.py": "print('ok')\n",
            }

            results = {
                path: write_file_tool(
                    path,
                    content=content,
                    new_create_confirm=True,
                    workspace_root=workspace,
                )
                for path, content in cases.items()
            }

            written = {
                path: (workspace / path).read_text(encoding="utf-8")
                for path in cases
            }

        for path, content in cases.items():
            self.assertEqual(results[path]["status"], "success")
            self.assertTrue(results[path]["created"])
            self.assertEqual(results[path]["old_content"], None)
            self.assertEqual(results[path]["new_start_line"], 1)
            self.assertEqual(results[path]["new_end_line"], len(content.splitlines()))
            self.assertEqual(results[path]["new_content"], content)
            self.assertEqual(written[path], content)

    def test_write_file_routes_missing_top_level_parent_to_nearest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            self._make_core_like_workspace(workspace)

            result = write_file_tool(
                "tests/demo1.py",
                content="print(1 + 1)\n",
                new_create_confirm=True,
                workspace_root=workspace,
            )
            routed_target_exists = (workspace / "src" / "core" / "tests" / "demo1.py").exists()
            root_target_exists = (workspace / "tests" / "demo1.py").exists()

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["path"], "src/core/tests/demo1.py")
        self.assertTrue(routed_target_exists)
        self.assertFalse(root_target_exists)

    def test_write_file_modifies_existing_file_with_unique_old_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            target = workspace / "src" / "core" / "change.py"
            target.parent.mkdir(parents=True)
            target.write_text("alpha\nold block\nomega\n", encoding="utf-8")

            result = write_file_tool(
                "src/core/change.py",
                old_content="old block",
                content="new block\nsecond new",
                workspace_root=workspace,
            )
            final_content = target.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["created"])
        self.assertEqual(result["old_start_line"], 2)
        self.assertEqual(result["old_end_line"], 2)
        self.assertEqual(result["old_content"], "old block")
        self.assertEqual(result["new_start_line"], 2)
        self.assertEqual(result["new_end_line"], 3)
        self.assertEqual(result["new_content"], "new block\nsecond new")
        self.assertEqual(final_content, "alpha\nnew block\nsecond new\nomega\n")

    def test_write_file_fills_empty_existing_file_without_old_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            target_without_old = workspace / "empty_without_old.py"
            target_with_empty_old = workspace / "empty_with_old.py"
            target_with_wrong_old = workspace / "empty_with_wrong_old.py"
            target_without_old.write_text("", encoding="utf-8")
            target_with_empty_old.write_text("", encoding="utf-8")
            target_with_wrong_old.write_text("", encoding="utf-8")

            result_without_old = write_file_tool(
                "empty_without_old.py",
                content="print(1 + 1)\n",
                workspace_root=workspace,
            )
            result_with_empty_old = write_file_tool(
                "empty_with_old.py",
                old_content="",
                content="print(2 + 2)\n",
                workspace_root=workspace,
            )
            result_with_wrong_old = write_file_tool(
                "empty_with_wrong_old.py",
                old_content="stale text",
                content="print(3 + 3)\n",
                workspace_root=workspace,
            )
            final_without_old = target_without_old.read_text(encoding="utf-8")
            final_with_empty_old = target_with_empty_old.read_text(encoding="utf-8")
            final_with_wrong_old = target_with_wrong_old.read_text(encoding="utf-8")

        self.assertEqual(result_without_old["status"], "success")
        self.assertFalse(result_without_old["created"])
        self.assertEqual(result_without_old["old_content"], "")
        self.assertEqual(result_without_old["old_start_line"], 1)
        self.assertEqual(result_without_old["old_end_line"], 0)
        self.assertEqual(final_without_old, "print(1 + 1)\n")
        self.assertEqual(result_with_empty_old["status"], "success")
        self.assertEqual(result_with_empty_old["old_content"], "")
        self.assertEqual(final_with_empty_old, "print(2 + 2)\n")
        self.assertEqual(result_with_wrong_old["status"], "success")
        self.assertEqual(result_with_wrong_old["old_content"], "")
        self.assertEqual(final_with_wrong_old, "print(3 + 3)\n")

    def test_write_file_rejects_old_content_mismatch_without_changing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            target = workspace / "target.py"
            target.write_text("current text", encoding="utf-8")

            result = write_file_tool(
                "target.py",
                old_content="stale text",
                content="new text",
                workspace_root=workspace,
            )
            final_content = target.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "failed")
        self.assertIn("reread", result["error"])
        self.assertEqual(final_content, "current text")

    def test_write_file_rejects_ambiguous_old_content_without_changing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            target = workspace / "target.py"
            target.write_text("dup\nkeep\ndup\n", encoding="utf-8")

            result = write_file_tool(
                "target.py",
                old_content="dup",
                content="new",
                workspace_root=workspace,
            )
            final_content = target.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "failed")
        self.assertIn("ambiguous", result["error"])
        self.assertEqual(final_content, "dup\nkeep\ndup\n")

    def test_write_file_rejects_unsafe_paths_and_unsupported_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            outside = Path(temp_dir) / "outside.py"
            workspace.mkdir()
            outside.write_text("outside", encoding="utf-8")
            cache_dir = workspace / "__pycache__"
            cache_dir.mkdir()
            (workspace / "directory").mkdir()

            outside_result = write_file_tool(
                str(outside),
                content="x",
                new_create_confirm=True,
                workspace_root=workspace,
            )
            hidden_result = write_file_tool(
                ".hidden.py",
                content="x",
                new_create_confirm=True,
                workspace_root=workspace,
            )
            cache_result = write_file_tool(
                "__pycache__/cached.py",
                content="x",
                new_create_confirm=True,
                workspace_root=workspace,
            )
            directory_result = write_file_tool(
                "directory",
                content="x",
                workspace_root=workspace,
            )
            unsupported_result = write_file_tool(
                "data.json",
                content="{}",
                new_create_confirm=True,
                workspace_root=workspace,
            )
            missing_parent_result = write_file_tool(
                "missing/child.py",
                content="x",
                new_create_confirm=True,
                workspace_root=workspace,
            )

        self.assertIn("workspace root", outside_result["error"])
        self.assertIn("filtered", hidden_result["error"])
        self.assertIn("filtered", cache_result["error"])
        self.assertIn("not a file", directory_result["error"])
        self.assertIn("unsupported", unsupported_result["error"])
        self.assertIn("parent directory", missing_parent_result["error"])

    def test_write_file_rejects_large_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            result = write_file_tool(
                "large.txt",
                content="x" * 1_048_577,
                new_create_confirm=True,
                workspace_root=workspace,
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("1MB", result["error"])
        self.assertFalse((workspace / "large.txt").exists())

    def test_write_file_rolls_back_when_atomic_write_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            target = workspace / "target.py"
            target.write_text("before", encoding="utf-8")

            with patch(
                "llm.Agent.tools.operation_tools._atomic_write_text",
                side_effect=ValueError("atomic write failed: disk full"),
            ):
                result = write_file_tool(
                    "target.py",
                    old_content="before",
                    content="after",
                    workspace_root=workspace,
                )
            final_content = target.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "failed")
        self.assertIn("atomic write failed", result["error"])
        self.assertEqual(final_content, "before")

    def test_run_tests_executes_python_file_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            target = workspace / "src" / "core" / "tests" / "demo1.py"
            target.parent.mkdir(parents=True)
            target.write_text("print(1 + 1)\n", encoding="utf-8")

            result = run_tests_tool(
                "python src/core/tests/demo1.py",
                workspace_root=workspace,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["stdout"].strip(), "2")
        self.assertEqual(result["stderr"], "")
        self.assertIsNone(result["error"])

    def test_run_tests_routes_python_script_path_to_nearest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            target = workspace / "src" / "core" / "tests" / "demo1.py"
            target.parent.mkdir(parents=True)
            target.write_text("print(1 + 1)\n", encoding="utf-8")

            result = run_tests_tool(
                "python tests/demo1.py",
                workspace_root=workspace,
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["command"], "python src/core/tests/demo1.py")
        self.assertEqual(result["stdout"].strip(), "2")

    def test_run_tests_rejects_shell_control_operators(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            result = run_tests_tool(
                "python ok.py && python other.py",
                workspace_root=workspace,
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("shell control", result["error"])

    def test_run_tests_rejects_unsupported_command(self) -> None:
        self.assertEqual(run_tests_tool(command="node app.js")["status"], "failed")

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
    def _make_core_like_workspace(workspace: Path) -> None:
        core = workspace / "src" / "core"
        core.mkdir(parents=True)
        (core / "plan.md").write_text("plan", encoding="utf-8")
        for directory_name in [
            "api",
            "daemon",
            "ipc",
            "llm",
            "practice",
            "schemas",
            "scripts",
            "set",
            "storage",
            "tests",
            "trace",
        ]:
            directory = core / directory_name
            directory.mkdir()
            for index in range(3):
                (directory / f"file_{index}.py").write_text("", encoding="utf-8")

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
