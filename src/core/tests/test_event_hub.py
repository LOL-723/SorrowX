import asyncio
import unittest

from daemon.events import EventHub
from ipc.protocol import decode_message, read_event_push


class FakeWriter:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        await asyncio.sleep(0)


class EventHubTests(unittest.IsolatedAsyncioTestCase):
    async def test_publish_filters_by_run_id_and_topic(self) -> None:
        hub = EventHub()
        writer_a = FakeWriter()
        writer_b = FakeWriter()
        client_a = hub.register_client(writer_a)
        client_b = hub.register_client(writer_b)
        hub.subscribe(client_id=client_a, topics=["agent.*"], run_id="run-a")
        hub.subscribe(client_id=client_b, topics=["run.*"], run_id="run-b")

        hub.publish({"type": "agent.log", "run_id": "run-a", "message": "matched"})
        hub.publish({"type": "agent.log", "run_id": "run-b", "message": "ignored"})
        hub.publish({"type": "run.started", "run_id": "run-b"})
        await asyncio.sleep(0.05)

        self.assertEqual(len(writer_a.writes), 1)
        self.assertEqual(len(writer_b.writes), 1)
        event_a = read_event_push(decode_message(writer_a.writes[0]))
        event_b = read_event_push(decode_message(writer_b.writes[0]))
        self.assertEqual(event_a["message"], "matched")
        self.assertEqual(event_b["type"], "run.started")

        hub.remove_client(client_a)
        hub.remove_client(client_b)
        await asyncio.sleep(0.05)
        self.assertEqual(hub.client_count(), 0)
        self.assertEqual(hub.subscription_count(), 0)


if __name__ == "__main__":
    unittest.main()
