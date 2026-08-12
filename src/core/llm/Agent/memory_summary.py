import json

from llm.Agent.memory import ContextMemory, ContextMemorySummaryUpdate
from llm.Agent.nodes.universal import _chat_completion


MEMORY_SUMMARY_PROMPT = """
You summarize a session's prior user questions and final answers for an Agent.

Create a concise Chinese memory summary that may be used as prior-conversation
context. Include only information useful for answering later requests. Do not
invent facts or treat answers as verified external evidence.

The input records are ordered from oldest to newest. When the same or similar
question appears more than once, preserve its sequence. If the answers differ,
clearly distinguish each result and identify the latest one. If the conclusion
is the same, state how many times it was repeated and keep the stable result.

Return exactly one JSON object: {"summary":"..."}. The summary must contain
at most 300 Chinese characters and no markdown heading.
""".strip()


def refresh_context_memory_summary(
    context_memory: ContextMemory,
    *,
    force: bool = False,
) -> ContextMemorySummaryUpdate:
    return context_memory.refresh_summary(summarize=_summarize_records, force=force)


def _summarize_records(records: list[dict[str, str]]) -> str:
    payload = {
        "records": [
            {
                "sequence": index,
                "question": record["question"],
                "final_answer": record["final_answer"],
            }
            for index, record in enumerate(records, start=1)
        ],
        "max_summary_characters": 300,
    }
    content = _chat_completion(
        system_prompt=MEMORY_SUMMARY_PROMPT,
        user_message=json.dumps(payload, ensure_ascii=False),
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("memory summary returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("memory summary returned non-object JSON")
    summary = data.get("summary")
    if not isinstance(summary, str):
        raise ValueError("memory summary must include a string summary")
    return summary
