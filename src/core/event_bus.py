"""
NOCTURNA – Event Bus (Architecture Foundation)

Thread-safe in-process pub/sub message bus.
Designed as a drop-in foundation that can later be swapped for
Redis Streams, Kafka, or NATS without changing publisher/subscriber code.

Topics (convention):
  engine.state
  market.bar
  market.tick
  signal.generated
  signal.rejected
  order.submitted
  order.filled
  order.cancelled
  order.rejected
  risk.event
  risk.critical
  portfolio.update
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Event:
    """Immutable event envelope."""
    topic: str
    payload: dict[str, Any]
    event_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    source: str = "system"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
        }


Handler = Callable[[Event], None]


class EventBus:
    """
    Lightweight, thread-safe event bus.

    Usage:
        bus = EventBus()
        bus.subscribe("order.filled", my_handler)
        bus.publish("order.filled", {"order_id": "abc", "symbol": "AAPL"})
    """

    def __init__(self, name: str = "nocturna") -> None:
        self.name = name
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._lock = threading.RLock()
        self._event_count = 0
        self._history: list[Event] = []
        self._max_history = 1000
        self._enabled = True
        logger.info("EventBus '%s' initialized", name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register a handler for a topic. Idempotent for the same callable."""
        with self._lock:
            if handler not in self._subscribers[topic]:
                self._subscribers[topic].append(handler)
                logger.debug("Subscribed %s to topic '%s'", getattr(handler, "__name__", handler), topic)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        """Remove a handler from a topic."""
        with self._lock:
            if handler in self._subscribers[topic]:
                self._subscribers[topic].remove(handler)

    def publish(
        self,
        topic: str,
        payload: dict[str, Any] | None = None,
        source: str = "system",
    ) -> Event | None:
        """
        Publish an event to all subscribers of the topic.
        Returns the Event object (or None if bus is disabled).
        Handlers are called synchronously in the publishing thread.
        Exceptions in handlers are logged and do not stop other handlers.
        """
        if not self._enabled:
            return None

        event = Event(
            topic=topic,
            payload=payload or {},
            source=source,
        )

        with self._lock:
            self._event_count += 1
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]

            handlers = list(self._subscribers.get(topic, []))
            # Also notify wildcard subscribers ("*")
            handlers.extend(self._subscribers.get("*", []))

        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.error(
                    "EventBus handler error on topic '%s' (event_id=%s): %s",
                    topic,
                    event.event_id,
                    exc,
                    exc_info=True,
                )

        return event

    def clear(self) -> None:
        """Remove all subscribers (useful in tests)."""
        with self._lock:
            self._subscribers.clear()

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "enabled": self._enabled,
                "total_events": self._event_count,
                "topics": {t: len(h) for t, h in self._subscribers.items()},
                "history_size": len(self._history),
            }

    def recent_events(self, topic: str | None = None, limit: int = 50) -> list[dict]:
        with self._lock:
            events = self._history
            if topic:
                events = [e for e in events if e.topic == topic]
            return [e.to_dict() for e in events[-limit:]]


# Singleton used by the trading engine (can be replaced later by a Redis-backed implementation)
default_bus = EventBus(name="nocturna-core")
