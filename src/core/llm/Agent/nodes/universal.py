from typing import Any

from openai import OpenAI

from set.config import settings
from llm.Agent.state import AgentState
from llm.tools import TOOL_ARGUMENTS, TOOL_DESCRIPTIONS
from trace.recorder import current_run_id, get_trace_recorder


def add_log(
    state: AgentState,
    node: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    logs = state.get("logs", [])
    log_item: dict[str, Any] = {
        "node": node,
        "message": message,
    }
    if extra:
        log_item.update(extra)
    return logs + [log_item]


def _available_tools(excluded_tools: list[str] | None = None) -> list[dict[str, Any]]:
    excluded_tool_names = set(excluded_tools or [])
    return [
        {
            "name": name,
            "description": description,
            "arguments": TOOL_ARGUMENTS.get(name, {}),
        }
        for name, description in TOOL_DESCRIPTIONS.items()
        if name not in excluded_tool_names
    ]


def _chat_completion(
    system_prompt: str,
    user_message: str,
    response_format: dict[str, str] | None = None,
    *,
    tool_count: int = 0,
) -> str:
    request: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": settings.LLM_TEMPERATURE,
    }
    if response_format is not None:
        request["response_format"] = response_format

    run_id = current_run_id()
    recorder = get_trace_recorder()
    call_id = recorder.record_core_to_llm(
        run_id,
        model=settings.LLM_MODEL,
        message_count=len(request["messages"]),
        tool_count=tool_count,
    )
    try:
        response = _openai_client().chat.completions.create(**request)
    except Exception as exc:
        recorder.record_llm_to_core(run_id, call_id=call_id, error=str(exc))
        raise
    recorder.record_llm_to_core(run_id, call_id=call_id, usage=getattr(response, "usage", None))
    return response.choices[0].message.content or ""


def _openai_client() -> OpenAI:
    return OpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        timeout=settings.LLM_TIMEOUT,
    )
