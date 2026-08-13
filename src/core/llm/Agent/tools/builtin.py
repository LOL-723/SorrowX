"""Built-in Agent tools and their Pydantic contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from llm.Agent.tools.common_tools import calculate_expression, get_current_time, get_today_weather
from llm.Agent.tools.contracts import FILESYSTEM_READ, FILESYSTEM_WRITE, NETWORK_HTTP, PROCESS_EXECUTE, ToolSpec
from llm.Agent.tools.operation_tools import list_dir_tool, read_file_tool, run_tests_tool, write_file_tool
from llm.Agent.tools.registry import ToolRegistry


class _ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CurrentTimeInput(_ToolInput):
    location: str | None = None
    timezone_name: str | None = None

class CalculateInput(_ToolInput):
    expression: str = Field(min_length=1, max_length=200)

class WeatherInput(_ToolInput):
    location: str | None = None
    timezone_name: str | None = None

class ListDirInput(_ToolInput):
    path: str | None = None
    keyword: str | None = None
    max_entries: int | None = Field(default=40, ge=1, le=40)

class ReadFileInput(_ToolInput):
    path: str = Field(min_length=1)
    keyword: str | None = None
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)

class WriteFileInput(_ToolInput):
    path: str = Field(min_length=1)
    content: str
    old_content: str | None = None
    new_create_confirm: bool = False

class RunTestsInput(_ToolInput):
    command: str | None = None
    timeout_seconds: int = Field(default=30, ge=1, le=120)


def create_builtin_registry() -> ToolRegistry:
    return ToolRegistry([
        ToolSpec("get_current_time", "获取当前时间；可指定地点或 IANA 时区。", CurrentTimeInput, frozenset(), False, None, get_current_time),
        ToolSpec("calculate_expression", "安全计算基础数学表达式。", CalculateInput, frozenset(), False, None, calculate_expression),
        ToolSpec("get_today_weather", "查询指定地点当天的天气预报。", WeatherInput, frozenset({NETWORK_HTTP}), False, 10, get_today_weather),
        ToolSpec("list_dir", "列出工作区目录或按名称搜索路径。", ListDirInput, frozenset({FILESYSTEM_READ}), False, None, list_dir_tool),
        ToolSpec("read_file", "读取工作区内的文本文件或指定行范围。", ReadFileInput, frozenset({FILESYSTEM_READ}), False, None, read_file_tool),
        ToolSpec("write_file", "创建或安全修改工作区内的文本文件。", WriteFileInput, frozenset({FILESYSTEM_WRITE}), True, None, write_file_tool),
        ToolSpec("run_tests", "运行受限的 Python 或 pytest 验证命令。", RunTestsInput, frozenset({PROCESS_EXECUTE}), True, 120, run_tests_tool),
    ])


BUILTIN_TOOL_REGISTRY = create_builtin_registry()
