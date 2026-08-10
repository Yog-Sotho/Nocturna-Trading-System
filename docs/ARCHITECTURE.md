# NOCTURNA v2.0 – Architecture Foundation

## Current Status (this PR)

This change introduces the **first surgical step** toward a fully event-driven, service-oriented architecture without breaking the existing threaded trading loop.

### What was added

1. **`src/core/event_bus.py`** – Thread-safe in-process EventBus
   - Pub/sub with topic-based routing
   - Immutable `Event` envelopes (id, timestamp, source, payload)
   - Wildcard (`*`) subscribers for observability
   - Bounded history for debugging
   - Zero external dependencies; designed so a Redis Streams / Kafka / NATS backend can be swapped later behind the same interface

2. **`src/core/order_state_machine.py`** – Formal Order State Machine
   - Explicit, validated state transitions (PENDING → SUBMITTED → PARTIAL/FILLED/CANCELLED/…)
   - Idempotent transitions
   - Automatic event emission on every successful change
   - Full transition history per order

3. **TradingEngine integration**
   - Engine now owns an `EventBus` instance
   - Key lifecycle points publish events (`engine.state`, `signal.generated`, `order.filled`, `risk.event`, etc.)
   - Existing callback system is preserved and also bridged to the bus → **zero breaking changes**

### Event Topics (convention)

| Topic              | Emitted when                          |
|--------------------|---------------------------------------|
| `engine.state`      | Engine start / stop / pause / resume / emergency |
| `signal.generated` | Strategy produces a signal            |
| `signal.rejected`  | Risk manager rejects a signal         |
| `order.submitted`  | Order accepted by broker / sim        |
| `order.filled`     | Order fully filled                    |
| `order.cancelled`  | Order cancelled                       |
| `order.rejected`   | Broker / validation rejection         |
| `risk.event`       | Any risk event                        |
| `risk.critical`    | Drawdown / daily-loss limit hit       |

## Roadmap to Full Event-Driven Architecture

### Phase 2 – Service boundaries (next recommended step)
- Extract MarketDataHandler, StrategyManager, RiskManager, OrderExecutionManager into independent processes or asyncio tasks that only communicate via the EventBus.
- Replace the polling main loop with event-driven reaction to `market.bar` / `market.tick` events.

### Phase 3 – Durable bus
- Swap the in-process EventBus for Redis Streams (or Kafka / NATS).
- Guaranteed delivery, consumer groups, replay capability.

### Phase 4 – Async execution
- Convert the remaining synchronous I/O (broker calls, data fetches) to `asyncio` + `httpx` / native async SDKs.
- Use `asyncio.TaskGroup` for concurrent symbol analysis.

### Phase 5 – Horizontal scaling
- Stateless signal generators, stateful risk service, dedicated execution service.
- Shared portfolio state via Redis / PostgreSQL.

## Design Principles Applied

- **Surgical**: Existing public APIs and the main trading loop continue to work unchanged.
- **Backward compatible**: Callbacks still fire; new event listeners are additive.
- **Observable**: Every important state change is now an event that can be logged, metered, or reacted to.
- **Idempotent**: Order transitions and duplicate-signal protection already present in OrderExecutionManager are reinforced by the state machine.

## How to use the new components

```python
from src.core.event_bus import default_bus
from src.core.order_state_machine import OrderStateMachine, OrderState

# Subscribe to events
def on_fill(event):
    print("Filled:", event.payload)

default_bus.subscribe("order.filled", on_fill)

# Use the state machine
osm = OrderStateMachine()
rec = osm.create("ord-123", "AAPL", "buy", 10)
osm.transition("ord-123", OrderState.SUBMITTED, reason="broker_ack")
osm.transition("ord-123", OrderState.FILLED, filled_quantity=10, avg_fill_price=182.5)
```

This foundation is the necessary first step before larger refactors (full asyncio, multi-process, durable messaging) can be applied safely.
