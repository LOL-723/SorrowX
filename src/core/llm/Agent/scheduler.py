"""Run-scoped background execution primitives for Agent Runtime."""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal, TypeVar, cast


T = TypeVar("T")
WorkerStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


@dataclass(frozen=True)
class WorkerHandle(Generic[T]):
    """A stable worker identifier associated with its execution future."""

    worker_id: str
    future: Future[T]


class AgentScheduler:
    """Own the bounded worker pool and handles for one Agent run."""

    def __init__(self, *, run_id: str, max_workers: int) -> None:
        if not run_id or not run_id.strip():
            raise ValueError("run_id cannot be empty")
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        self.run_id = run_id
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"agent-worker-{run_id}",
        )
        self._handles: dict[str, WorkerHandle[Any]] = {}
        self._lock = threading.Lock()
        self._shutdown = False

    @property
    def is_shutdown(self) -> bool:
        with self._lock:
            return self._shutdown

    def submit(
        self,
        task: Callable[..., T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> WorkerHandle[T]:
        """Submit a callable and return immediately with its worker handle."""
        if not callable(task):
            raise TypeError("task must be callable")

        with self._lock:
            if self._shutdown:
                raise RuntimeError("scheduler is shut down")
            worker_id = f"{self.run_id}:worker-{uuid.uuid4().hex}"
            future = self._executor.submit(task, *args, **kwargs)
            handle = WorkerHandle(worker_id=worker_id, future=future)
            self._handles[worker_id] = cast(WorkerHandle[Any], handle)
        return handle

    def get(self, worker_id: str) -> WorkerHandle[Any] | None:
        """Return this run's handle for ``worker_id``, if it exists."""
        with self._lock:
            return self._handles.get(worker_id)

    def status(self, worker_id: str) -> WorkerStatus:
        future = self._require_handle(worker_id).future
        if future.running():
            return "running"
        if not future.done():
            return "pending"
        if future.cancelled():
            return "cancelled"
        return "failed" if future.exception() is not None else "completed"

    def wait(self, worker_id: str, *, timeout: float | None = None) -> Any:
        """Wait for a worker and return its result, propagating its exception."""
        return self._require_handle(worker_id).future.result(timeout=timeout)

    def cancel(self, worker_id: str) -> bool:
        """Try to cancel a worker that has not started running."""
        return self._require_handle(worker_id).future.cancel()

    def shutdown(self) -> None:
        """Stop submission, wait for submitted workers, and release handles."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True

        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._lock:
            self._handles.clear()

    def _require_handle(self, worker_id: str) -> WorkerHandle[Any]:
        handle = self.get(worker_id)
        if handle is None:
            raise KeyError(f"unknown worker_id for run {self.run_id}: {worker_id}")
        return handle

    def __enter__(self) -> "AgentScheduler":
        return self

    def __exit__(self, *args: object) -> None:
        self.shutdown()
