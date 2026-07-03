import sys
import time
from typing import Any


Event = dict[str, Any]


class CliEventPrinter:
    def __init__(self) -> None:
        self._started_at = 0.0

    def handle(self, event: Event) -> None:
        event_type = event.get("type")
        if event_type == "run.started":
            self._started_at = time.monotonic()
            self._print(f"[run] {event.get('run_id')} {event.get('goal')}")
        elif event_type == "agent.node.started":
            self._print(f"[node] {event.get('node')} started")
        elif event_type == "agent.node.finished":
            status = event.get("status") or "unknown"
            self._print(f"[node] {event.get('node')} {status}")
        elif event_type == "agent.step.started":
            self._print(f"[step] {event.get('step_id')} {event.get('task')}")
        elif event_type == "agent.loop.thought":
            self._print(
                "[Agent Thought] "
                f"step={event.get('step_id')} "
                f"turn={event.get('turn')} "
                f"decision={event.get('decision')} "
                f"signal={event.get('signal')} "
                f"tool={event.get('tool')}\n"
                f"{event.get('thought', '')}",
            )
        elif event_type == "agent.log":
            self._print(f"[log] {event.get('source')}: {event.get('message')}")
        elif event_type == "agent.answer":
            self._print(str(event.get("answer", "")))
        elif event_type == "run.finished":
            elapsed = time.monotonic() - self._started_at if self._started_at else 0.0
            self._print(f"[run] {event.get('status')} {elapsed:.1f}s")

    def _print(self, text: str) -> None:
        try:
            print(text, flush=True)
        except UnicodeEncodeError:
            encoding = sys.stdout.encoding or "utf-8"
            safe_text = text.encode(encoding, errors="replace").decode(encoding)
            print(safe_text, flush=True)
