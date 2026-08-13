import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TextIO

from daemon.events import Event, EventBus, EventHandler
from llm.Agent.memory import ContextMemory, MemoryRefreshScheduler
from session.manager import SessionManager, get_session_manager
from trace.recorder import trace_run


RUNS_DIR = Path("runs")


@dataclass(frozen=True)
class AgentRequest:
    goal: str
    session_id: str


@dataclass(frozen=True)
class AgentRunContext:
    run_id: str
    session_id: str
    context_memory_path: Path
    context_memory: str


@dataclass(frozen=True)
class EngineResult:
    answer: str


@dataclass(frozen=True)
class AgentResult:
    run_id: str
    status: Literal["finished", "failed"]
    answer: str
    error: str | None
    events_path: Path


if TYPE_CHECKING:
    from llm.Agent.AgentEngine import AgentEngine


class EventWriter:
    def __init__(self, path: str | Path, *, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self._file: TextIO | None = None

    def __enter__(self) -> "EventWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *args: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def subscribe(self, bus: EventBus) -> None:
        bus.subscribe(self.handle)

    def handle(self, event: Event) -> None:
        if self._file is None:
            return
        if self.run_id is not None and event.get("run_id") != self.run_id:
            return
        self._file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._file.flush()


class AgentRuntime:
    def __init__(
        self,
        engine: "AgentEngine",
        *,
        runs_dir: str | Path | None = None,
        extra_handlers: list[EventHandler] | None = None,
        session_manager: SessionManager | None = None,
        memory_scheduler: MemoryRefreshScheduler | None = None,
    ) -> None:
        self.engine = engine
        self.runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
        self.extra_handlers = list(extra_handlers or [])
        self.session_manager = session_manager or get_session_manager()
        self.memory_scheduler = memory_scheduler or MemoryRefreshScheduler()

    def new_run_id(self) -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return f"{timestamp}-{uuid.uuid4().hex[:6]}"

    def run(
        self,
        request: AgentRequest,
        *,
        run_id: str | None = None,
        event_bus: EventBus | None = None,
    ) -> AgentResult:
        context = self._create_context(request, run_id=run_id)
        events_path = self.runs_dir / context.run_id / "events.jsonl"
        bus = event_bus or EventBus()
        subscribed_handlers: list[EventHandler] = []
        for handler in self.extra_handlers:
            bus.subscribe(handler)
            subscribed_handlers.append(handler)

        status: Literal["finished", "failed"] = "failed"
        answer = ""
        error: str | None = None

        try:
            with trace_run(context.run_id, session_id=context.session_id):
                with EventWriter(events_path, run_id=context.run_id) as writer:
                    writer.subscribe(bus)
                    subscribed_handlers.append(writer.handle)
                    self._publish(bus, "run.started", context, goal=request.goal)
                    try:
                        engine_result = self.engine.execute(
                            request,
                            context,
                            lambda event: self._emit_engine_event(bus, context, event),
                        )
                        answer = engine_result.answer
                        status = "finished"
                        self._remember_success(request, context, bus, answer)
                        self._publish(bus, "agent.answer", context, answer=answer)
                    except Exception as exc:
                        error = str(exc)
                        self._publish(bus, "agent.error", context, error=error)
                    finally:
                        self._publish(
                            bus,
                            "run.finished",
                            context,
                            status=status,
                            answer=answer,
                            error=error,
                        )
        finally:
            for handler in reversed(subscribed_handlers):
                bus.unsubscribe(handler)

        return AgentResult(
            run_id=context.run_id,
            status=status,
            answer=answer,
            error=error,
            events_path=events_path,
        )

    def _create_context(
        self,
        request: AgentRequest,
        *,
        run_id: str | None,
    ) -> AgentRunContext:
        if not request.goal or not request.goal.strip():
            raise ValueError("goal cannot be empty")
        if not self.session_manager.session_exists(request.session_id):
            raise ValueError(f"session does not exist: {request.session_id}")

        paths = self.session_manager.ensure_session_dirs(request.session_id)
        memory = ContextMemory(paths.memory_path)
        return AgentRunContext(
            run_id=run_id or self.new_run_id(),
            session_id=request.session_id,
            context_memory_path=paths.memory_path,
            context_memory=memory.load_context(),
        )

    def _remember_success(
        self,
        request: AgentRequest,
        context: AgentRunContext,
        bus: EventBus,
        answer: str,
    ) -> None:
        memory = ContextMemory(context.context_memory_path)
        try:
            memory.remember(
                question=request.goal,
                final_answer=answer,
                run_id=context.run_id,
            )
        except Exception as exc:
            self._publish(bus, "agent.memory.failed", context, error=str(exc), phase="persist")
            return

        def report_refresh_failure(exc: Exception) -> None:
            self._publish(bus, "agent.memory.failed", context, error=str(exc), phase="refresh")

        try:
            self.memory_scheduler.schedule(memory, on_failure=report_refresh_failure)
        except Exception as exc:
            self._publish(bus, "agent.memory.failed", context, error=str(exc), phase="schedule")

    def _emit_engine_event(
        self,
        bus: EventBus,
        context: AgentRunContext,
        event: Event,
    ) -> None:
        event_type = str(event.get("type", "agent.event"))
        payload = {
            key: value
            for key, value in event.items()
            if key not in {"type", "run_id", "session_id", "ts"}
        }
        self._publish(bus, event_type, context, **payload)

    def _publish(
        self,
        bus: EventBus,
        event_type: str,
        context: AgentRunContext,
        **payload: Any,
    ) -> None:
        bus.publish(
            {
                "type": event_type,
                "run_id": context.run_id,
                "session_id": context.session_id,
                "ts": datetime.now(UTC).isoformat(),
                **payload,
            }
        )


_default_runtime: AgentRuntime | None = None


def get_agent_runtime() -> AgentRuntime:
    global _default_runtime
    if _default_runtime is None:
        from llm.Agent.AgentEngine import AgentLoopEngine

        _default_runtime = AgentRuntime(engine=AgentLoopEngine())
    return _default_runtime
