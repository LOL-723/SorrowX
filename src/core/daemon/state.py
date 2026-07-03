import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from daemon.events import EventBus, EventHub


DAEMON_VERSION = "0.1.0"


@dataclass
class DaemonState:
    host: str
    port: int
    version: str = DAEMON_VERSION
    started_monotonic: float = field(default_factory=time.monotonic)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_bus: EventBus = field(default_factory=EventBus)
    event_hub: EventHub = field(init=False)
    active_runs: dict[str, asyncio.Task[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.event_hub = EventHub()
        self.event_bus.subscribe(self.event_hub.publish)

    def uptime_ms(self) -> int:
        return int((time.monotonic() - self.started_monotonic) * 1000)

    def iso_started_at(self) -> str:
        return self.started_at.isoformat()
