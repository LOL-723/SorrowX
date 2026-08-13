import json

from llm.Agent.memory import ContextMemory, ContextMemorySummaryUpdate
from llm.Agent.nodes.universal import _chat_completion


MEMORY_SUMMARY_PROMPT = """
You summarize a session's prior user questions and final answers for an Agent.

Create a concise Chinese rolling memory summary that may be used as
prior-conversation context. Merge the previous summary with the new records.
Keep stable user preferences, unfinished work, latest conclusions, and explicit
corrections; discard superseded or low-value detail. Do not invent facts or
treat answers as verified external evidence.

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
    target_sequence: int | None = None,
) -> ContextMemorySummaryUpdate:
    return context_memory.refresh_rolling_summary(
        summarize=_summarize_records,
        force=force,
        target_sequence=target_sequence,
    )


def _summarize_records(previous_summary: str, records: list[dict[str, object]]) -> str:
    payload = {
        "previous_summary": previous_summary,
        "records": [
            {
                "sequence": int(record.get("sequence", index)),
                "question": str(record["question"]),
                "final_answer": str(record["final_answer"]),
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
