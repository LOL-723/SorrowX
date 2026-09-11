import threading
import time
import unittest

from llm.Agent.scheduler import AgentScheduler


class AgentSchedulerTests(unittest.TestCase):
    def test_worker_runs_on_background_thread_and_returns_result(self) -> None:
        root_thread_id = threading.get_ident()
        scheduler = AgentScheduler(run_id="run-thread", max_workers=1)
        try:
            handle = scheduler.submit(lambda: (threading.get_ident(), "result"))
            worker_thread_id, result = scheduler.wait(handle.worker_id)
        finally:
            scheduler.shutdown()

        self.assertNotEqual(worker_thread_id, root_thread_id)
        self.assertEqual(result, "result")

    def test_submit_is_non_blocking(self) -> None:
        release = threading.Event()
        scheduler = AgentScheduler(run_id="run-non-blocking", max_workers=1)
        try:
            started_at = time.monotonic()
            handle = scheduler.submit(release.wait)
            elapsed = time.monotonic() - started_at
            self.assertLess(elapsed, 0.2)
            self.assertFalse(handle.future.done())
        finally:
            release.set()
            scheduler.shutdown()

    def test_handles_are_unique_and_link_the_correct_futures(self) -> None:
        scheduler = AgentScheduler(run_id="run-handles", max_workers=2)
        try:
            first = scheduler.submit(lambda: "first")
            second = scheduler.submit(lambda: "second")
            self.assertNotEqual(first.worker_id, second.worker_id)
            self.assertIs(scheduler.get(first.worker_id), first)
            self.assertIs(scheduler.get(second.worker_id), second)
            self.assertEqual(scheduler.wait(first.worker_id), "first")
            self.assertEqual(scheduler.wait(second.worker_id), "second")
        finally:
            scheduler.shutdown()

    def test_status_covers_pending_running_completed_failed_and_cancelled(self) -> None:
        running = threading.Event()
        release = threading.Event()

        def blocker() -> str:
            running.set()
            release.wait()
            return "done"

        scheduler = AgentScheduler(run_id="run-status", max_workers=1)
        try:
            active = scheduler.submit(blocker)
            self.assertTrue(running.wait(1.0))
            self.assertEqual(scheduler.status(active.worker_id), "running")

            pending = scheduler.submit(lambda: "queued")
            self.assertEqual(scheduler.status(pending.worker_id), "pending")
            cancelled = scheduler.submit(lambda: "cancelled")
            self.assertTrue(scheduler.cancel(cancelled.worker_id))
            self.assertEqual(scheduler.status(cancelled.worker_id), "cancelled")

            release.set()
            self.assertEqual(scheduler.wait(active.worker_id), "done")
            self.assertEqual(scheduler.status(active.worker_id), "completed")
            self.assertEqual(scheduler.wait(pending.worker_id), "queued")

            failed = scheduler.submit(_raise_worker_error)
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                scheduler.wait(failed.worker_id)
            self.assertEqual(scheduler.status(failed.worker_id), "failed")
        finally:
            release.set()
            scheduler.shutdown()

    def test_wait_blocks_until_worker_finishes_and_propagates_result(self) -> None:
        release = threading.Event()
        waiter_started = threading.Event()
        waiter_finished = threading.Event()
        results: list[str] = []
        scheduler = AgentScheduler(run_id="run-wait", max_workers=1)
        handle = scheduler.submit(release.wait)

        def wait_for_result() -> None:
            waiter_started.set()
            results.append(str(scheduler.wait(handle.worker_id)))
            waiter_finished.set()

        waiter = threading.Thread(target=wait_for_result)
        waiter.start()
        try:
            self.assertTrue(waiter_started.wait(1.0))
            self.assertFalse(waiter_finished.wait(0.05))
            release.set()
            self.assertTrue(waiter_finished.wait(1.0))
            self.assertEqual(results, ["True"])
        finally:
            release.set()
            waiter.join(1.0)
            scheduler.shutdown()

    def test_pool_never_exceeds_max_workers_and_queues_excess_tasks(self) -> None:
        release = threading.Event()
        two_workers_started = threading.Event()
        lock = threading.Lock()
        running = 0
        maximum_running = 0

        def tracked_worker() -> str:
            nonlocal running, maximum_running
            with lock:
                running += 1
                maximum_running = max(maximum_running, running)
                if running == 2:
                    two_workers_started.set()
            release.wait()
            with lock:
                running -= 1
            return "done"

        scheduler = AgentScheduler(run_id="run-limit", max_workers=2)
        handles = [scheduler.submit(tracked_worker) for _ in range(4)]
        try:
            self.assertTrue(two_workers_started.wait(1.0))
            self.assertEqual(sum(handle.future.running() for handle in handles), 2)
            self.assertEqual(sum(not handle.future.running() for handle in handles), 2)
            release.set()
            self.assertEqual(
                [scheduler.wait(handle.worker_id) for handle in handles],
                ["done"] * 4,
            )
            self.assertEqual(maximum_running, 2)
        finally:
            release.set()
            scheduler.shutdown()

    def test_schedulers_are_isolated_by_run(self) -> None:
        first = AgentScheduler(run_id="run-a", max_workers=1)
        second = AgentScheduler(run_id="run-b", max_workers=1)
        try:
            first_handle = first.submit(lambda: "a")
            self.assertIsNone(second.get(first_handle.worker_id))
            with self.assertRaises(KeyError):
                second.wait(first_handle.worker_id)

            first.shutdown()
            second_handle = second.submit(lambda: "b")
            self.assertEqual(second.wait(second_handle.worker_id), "b")
        finally:
            first.shutdown()
            second.shutdown()

    def test_shutdown_waits_for_workers_releases_handles_and_rejects_submit(self) -> None:
        release = threading.Event()
        scheduler = AgentScheduler(run_id="run-shutdown", max_workers=1)
        handle = scheduler.submit(release.wait)
        release.set()
        scheduler.shutdown()

        self.assertTrue(handle.future.done())
        self.assertTrue(scheduler.is_shutdown)
        self.assertIsNone(scheduler.get(handle.worker_id))
        with self.assertRaisesRegex(RuntimeError, "shut down"):
            scheduler.submit(lambda: None)


def _raise_worker_error() -> None:
    raise RuntimeError("worker failed")


if __name__ == "__main__":
    unittest.main()
