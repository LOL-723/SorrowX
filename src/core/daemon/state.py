import time
from dataclasses import dataclass, field
from datetime import UTC, datetime


DAEMON_VERSION = "0.1.0"


@dataclass
class DaemonState:
    host: str
    port: int
    version: str = DAEMON_VERSION
    started_monotonic: float = field(default_factory=time.monotonic)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def uptime_ms(self) -> int:
        return int((time.monotonic() - self.started_monotonic) * 1000)

    def iso_started_at(self) -> str:
        return self.started_at.isoformat()
