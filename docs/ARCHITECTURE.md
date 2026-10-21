# NOCTURNA architecture

NOCTURNA has one trading core, `src/trading`, used unchanged for backtests and for paper
trading. Only three collaborators are swapped between the two:

| Collaborator | Backtest | Paper trading |
|---|---|---|
| Clock | `SimulatedClock`, advanced to the moment each bar became final | `SystemClock` |
| Bar source | `InMemoryBarSource` over historical bars | `AlpacaBarSource` (Alpaca market data, free plan) |
| Broker | `SimulatedBroker` | `AlpacaBroker` (Alpaca paper Trading API) |

A parity test runs the paper-trading code path (polling a bar source on a simulated clock)
over the same data as a backtest and requires identical fills and final equity.

## Data flow

```
bar source ──(closed bars)──> PollingFeed ──> Engine.on_bars
                                                 │
                     ┌───────────────────────────┼───────────────────────────────┐
                     ▼                           ▼                               ▼
            broker.on_bars              strategy.target_weights          (history kept per symbol)
       (simulator fills working              per strategy
        orders; Alpaca: no-op)                    │
                                                  ▼
                                  net targets x allocation -> order sizes
                                  (sells first, buys within cash, rounded
                                   to the instrument's increment)
                                                  │
                                                  ▼
                                      RiskGate.check ──> broker.submit
                                                               │
              broker reports (simulator: immediately; Alpaca: trade-update websocket)
                                                               │
                                                               ▼
                                     Engine.on_order_update ──> Order lifecycle + Ledger
                                                               │
                                                               ▼
                                        listeners: EngineRecorder (database, audit log)
```

### Closed bars only

A bar covers `[start, end)` and is *closed* once `end + feed delay <= now`. The delay is
15 minutes for equities (the free Alpaca plan serves SIP data only once it is 15 minutes old)
and 1 minute for crypto. Every source filters on this rule against its clock, so a strategy
can never see a bar that had not finished. Daily equity bars follow the New York calendar day
(DST-aware); crypto days are 24 hours from UTC midnight, as Alpaca's data shows.

### Orders fill only on prices they could have had

The simulator fills an order only on a bar whose *trading* started after the order was placed:
for daily equity bars that is the 09:30 session open, not New York midnight. Market and
opening-auction (OPG) orders fill at the open, close (CLS) orders at the close, both with
adverse slippage; limits need the price to trade through them; stops fill at the stop or at a
gapped open. Fees and slippage are per asset class (`DEFAULT_COSTS`); crypto purchases pay the
fee in the asset received, as on Alpaca.

## Accounting

`Ledger` derives cash, positions and P&L from fills only. Positions store a signed cost basis
(not an average price) and arithmetic runs at 60 significant digits with the single division
rounded to 1e-12, so

```
equity == starting cash + realized P&L + unrealized P&L - fees + adjustments
```

holds exactly; property tests check it over thousands of random fill sequences. Fills are
idempotent by `fill_id`, so a replayed broker report is never booked twice.

## Live engine service

`EngineService` runs one engine on its own thread. Nothing else touches the engine: HTTP
handlers and the Alpaca stream put commands and order updates on a queue, executed in order on
the engine thread, and read state from the snapshot it publishes.

- **Start-up:** the ledger is built from the broker account (cash and cost basis), working
  orders are adopted, warm-up history is loaded and the engine acts on the newest bars. A
  restart therefore submits nothing that is already working or filled.
- **Loop:** the service wakes when the next bar becomes final (`PollingFeed.next_poll`), polls
  the bar source, applies broker updates as they arrive, and records an equity snapshot.
- **Reconciliation:** the broker is the source of truth. When no order is working and the
  stream has been quiet for two minutes, positions, cash and orders are compared with the
  broker; differences are booked as ledger adjustments and audited.
- **Kill switch:** `halt` cancels the engine's orders and, optionally, flattens through the
  risk gate (only position-reducing orders pass while halted). The emergency-stop endpoint
  also works when the engine is not running, by asking Alpaca to cancel and close everything.

The broker is the source of current state; the database holds history (orders, fills, equity
snapshots, audit log) and is not replayed to rebuild state.

## Safety

`src/trading/safety.py` refuses to start the app or build the Alpaca adapter unless
`TRADING_MODE=PAPER` and `ALPACA_BASE_URL` (if set) is the paper endpoint. The adapter only
connects to Alpaca's paper API.

## What comes next

Phase 3 extends the risk gate (loss limits, drawdown kill switch, price collars, duplicate
detection, volatility-targeted sizing), Phase 4 adds walk-forward validation on top of the
backtester, and Phase 5 adds the strategies; see [ROADMAP.md](ROADMAP.md).
