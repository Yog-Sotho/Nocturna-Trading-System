"""
NOCTURNA – Order State Machine (Architecture Foundation)

Formal, idempotent order lifecycle with validated transitions.
Emits events via the EventBus on every successful state change.

States follow industry convention and map 1:1 to the existing
OrderStatus enum used by OrderExecutionManager.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.core.event_bus import EventBus, default_bus

logger = logging.getLogger(__name__)


class OrderState(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    ERROR = "ERROR"


# Valid directed transitions (from → allowed targets)
_TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.PENDING: {
        OrderState.SUBMITTED,
        OrderState.REJECTED,
        OrderState.CANCELLED,
        OrderState.ERROR,
    },
    OrderState.SUBMITTED: {
        OrderState.PARTIALLY_FILLED,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
        OrderState.ERROR,
    },
    OrderState.PARTIALLY_FILLED: {
        OrderState.PARTIALLY_FILLED,  # further partial fills
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
        OrderState.ERROR,
    },
    # Terminal states – no further transitions
    OrderState.FILLED: set(),
    OrderState.CANCELLED: set(),
    OrderState.REJECTED: set(),
    OrderState.EXPIRED: set(),
    OrderState.ERROR: set(),
}


@dataclass
class OrderStateRecord:
    order_id: str
    symbol: str
    side: str
    quantity: float
    state: OrderState = OrderState.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    client_order_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        return self.state in {
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.ERROR,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "state": self.state.value,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "client_order_id": self.client_order_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_terminal": self.is_terminal(),
            "metadata": self.metadata,
        }


class OrderStateMachine:
    """
    Manages the lifecycle of orders with strict transition rules
    and automatic event emission.
    """

    def __init__(self, bus: EventBus | None = None) -> None:
        self.bus = bus or default_bus
        self._orders: dict[str, OrderStateRecord] = {}
        self._lock = threading.RLock()

    def create(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        client_order_id: str = "",
        metadata: dict | None = None,
    ) -> OrderStateRecord:
        """Create a new order in PENDING state."""
        with self._lock:
            if order_id in self._orders:
                existing = self._orders[order_id]
                logger.warning("Order %s already exists in state %s", order_id, existing.state.value)
                return existing

            record = OrderStateRecord(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                client_order_id=client_order_id,
                metadata=metadata or {},
            )
            record.history.append(
                {
                    "from": None,
                    "to": OrderState.PENDING.value,
                    "at": record.created_at.isoformat(),
                    "reason": "created",
                }
            )
            self._orders[order_id] = record

            self.bus.publish(
                "order.created",
                record.to_dict(),
                source="order_state_machine",
            )
            return record

    def transition(
        self,
        order_id: str,
        new_state: OrderState | str,
        reason: str = "",
        filled_quantity: float | None = None,
        avg_fill_price: float | None = None,
        extra: dict | None = None,
    ) -> OrderStateRecord | None:
        """
        Attempt a state transition.
        Returns the updated record on success, None on invalid transition.
        Idempotent: transitioning to the same state is a no-op success.
        """
        if isinstance(new_state, str):
            new_state = OrderState(new_state)

        with self._lock:
            record = self._orders.get(order_id)
            if record is None:
                logger.error("Cannot transition unknown order %s", order_id)
                return None

            current = record.state

            # Idempotent: already in target state
            if current == new_state:
                if filled_quantity is not None:
                    record.filled_quantity = filled_quantity
                if avg_fill_price is not None:
                    record.avg_fill_price = avg_fill_price
                record.updated_at = datetime.now(UTC)
                return record

            allowed = _TRANSITIONS.get(current, set())
            if new_state not in allowed:
                logger.warning(
                    "Invalid transition for order %s: %s → %s (allowed: %s)",
                    order_id,
                    current.value,
                    new_state.value,
                    [s.value for s in allowed],
                )
                return None

            # Apply transition
            old_state = current
            record.state = new_state
            record.updated_at = datetime.now(UTC)
            if filled_quantity is not None:
                record.filled_quantity = filled_quantity
            if avg_fill_price is not None:
                record.avg_fill_price = avg_fill_price

            history_entry = {
                "from": old_state.value,
                "to": new_state.value,
                "at": record.updated_at.isoformat(),
                "reason": reason or "",
            }
            if extra:
                history_entry["extra"] = extra
            record.history.append(history_entry)

            # Emit domain event
            topic = self._topic_for_state(new_state)
            payload = record.to_dict()
            if extra:
                payload["extra"] = extra
            self.bus.publish(topic, payload, source="order_state_machine")

            logger.info(
                "Order %s transitioned %s → %s (%s)",
                order_id,
                old_state.value,
                new_state.value,
                reason,
            )
            return record

    def get(self, order_id: str) -> OrderStateRecord | None:
        with self._lock:
            return self._orders.get(order_id)

    def get_active(self) -> list[OrderStateRecord]:
        with self._lock:
            return [r for r in self._orders.values() if not r.is_terminal()]

    def _topic_for_state(self, state: OrderState) -> str:
        mapping = {
            OrderState.SUBMITTED: "order.submitted",
            OrderState.PARTIALLY_FILLED: "order.partial",
            OrderState.FILLED: "order.filled",
            OrderState.CANCELLED: "order.cancelled",
            OrderState.REJECTED: "order.rejected",
            OrderState.EXPIRED: "order.expired",
            OrderState.ERROR: "order.error",
        }
        return mapping.get(state, "order.state_changed")
