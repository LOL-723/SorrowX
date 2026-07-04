import asyncio
import uuid
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

from ipc.protocol import encode_message, make_event_push
from trace.recorder import TraceRecorder


Event = dict[str, Any]
EventHandler = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[EventHandler] = []
        self._lock = RLock()

    def subscribe(self, handler: EventHandler) -> None:
        with self._lock:
            if handler not in self._subscribers:
                self._subscribers.append(handler)

    def unsubscribe(self, handler: EventHandler) -> None:
        with self._lock:
            if handler in self._subscribers:
                self._subscribers.remove(handler)

    def publish(self, event: Event) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for handler in subscribers:
            handler(event)


@dataclass(frozen=True)
class Subscription:
    subscription_id: str
    client_id: str
    topics: tuple[str, ...]
    run_id: str | None = None
    client_name: str | None = None


@dataclass
class ClientConnection:
    client_id: str
    writer: asyncio.StreamWriter
    queue: asyncio.Queue[dict[str, Any] | None]
    send_task: asyncio.Task[None] | None = None


class EventHub:
    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        max_queue_size: int = 256,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        self._loop = loop
        self._max_queue_size = max_queue_size
        self._trace_recorder = trace_recorder
        self._clients: dict[str, ClientConnection] = {}
        self._subscriptions: dict[str, Subscription] = {}

    def register_client(self, writer: asyncio.StreamWriter) -> str:
        loop = self._ensure_loop()
        client_id = uuid.uuid4().hex
        connection = ClientConnection(
            client_id=client_id,
            writer=writer,
            queue=asyncio.Queue(maxsize=self._max_queue_size),
        )
        connection.send_task = loop.create_task(self._send_loop(connection))
        self._clients[client_id] = connection
        return client_id

    def remove_client(self, client_id: str) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        if loop.is_running():
            loop.call_soon_threadsafe(self._remove_client_on_loop, client_id)
        else:
            self._remove_client_on_loop(client_id)

    def subscribe(
        self,
        *,
        client_id: str,
        topics: list[str] | tuple[str, ...],
        run_id: str | None = None,
        client_name: str | None = None,
    ) -> Subscription:
        if client_id not in self._clients:
            raise ValueError(f"unknown client_id: {client_id}")
        subscription = Subscription(
            subscription_id=uuid.uuid4().hex,
            client_id=client_id,
            topics=tuple(topics),
            run_id=run_id,
            client_name=client_name,
        )
        self._subscriptions[subscription.subscription_id] = subscription
        return subscription

    def unsubscribe(self, subscription_id: str) -> None:
        self._subscriptions.pop(subscription_id, None)

    def publish(self, event: Event) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._publish_on_loop, dict(event))

    def client_count(self) -> int:
        return len(self._clients)

    def subscription_count(self) -> int:
        return len(self._subscriptions)

    async def _send_loop(self, connection: ClientConnection) -> None:
        try:
            while True:
                message = await connection.queue.get()
                if message is None:
                    break
                connection.writer.write(encode_message(message))
                await connection.writer.drain()
        except (ConnectionError, OSError):
            self._remove_client_on_loop(connection.client_id)
        except asyncio.CancelledError:
            raise

    def _publish_on_loop(self, event: Event) -> None:
        envelope = make_event_push(event)
        sent_clients: set[str] = set()
        for subscription in list(self._subscriptions.values()):
            if subscription.client_id in sent_clients:
                continue
            if not _subscription_matches(subscription, event):
                continue
            connection = self._clients.get(subscription.client_id)
            if connection is None:
                continue
            try:
                connection.queue.put_nowait(envelope)
                if self._trace_recorder is not None:
                    self._trace_recorder.record_core_to_client_event(
                        event,
                        subscription_id=subscription.subscription_id,
                        client_id=subscription.client_id,
                    )
                sent_clients.add(subscription.client_id)
            except asyncio.QueueFull:
                self._remove_client_on_loop(subscription.client_id)

    def _remove_client_on_loop(self, client_id: str) -> None:
        connection = self._clients.pop(client_id, None)
        if connection is not None and connection.send_task is not None:
            connection.send_task.cancel()
        for subscription_id, subscription in list(self._subscriptions.items()):
            if subscription.client_id == client_id:
                self._subscriptions.pop(subscription_id, None)

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.get_running_loop()
        return self._loop


def normalize_topics(value: Any) -> list[str]:
    if value is None:
        return ["*"]
    if isinstance(value, str):
        topics = [value]
    elif isinstance(value, list):
        topics = value
    else:
        raise ValueError("topics must be a string or a list of strings")

    normalized: list[str] = []
    for topic in topics:
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topics must contain non-empty strings")
        normalized.append(topic.strip())
    return normalized or ["*"]


def _subscription_matches(subscription: Subscription, event: Event) -> bool:
    if subscription.run_id is not None and event.get("run_id") != subscription.run_id:
        return False
    event_type = event.get("type")
    if not isinstance(event_type, str) or not event_type:
        return False
    return any(_topic_matches(topic, event_type) for topic in subscription.topics)


def _topic_matches(topic: str, event_type: str) -> bool:
    if topic == "*":
        return True
    if topic.endswith("*"):
        return event_type.startswith(topic[:-1])
    return topic == event_type
