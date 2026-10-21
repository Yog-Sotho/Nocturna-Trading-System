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
                                  net targets x allocation -> PortfolioSizer
                                  (caps, volatility target, drawdown ladder)
                                  -> orders outside the rebalance band
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
- **Kill switch:** `halt` cancels the engine's orders and, with flatten, asks the broker to
  liquidate (Alpaca's close-all cancels and closes in the right order, which separate engine
  orders cannot while cancels are in flight); the liquidation orders are adopted so their
  fills are booked. While halted only position-reducing orders pass the gate. The
  emergency-stop endpoint also works when the engine is not running, by asking Alpaca to
  cancel and close everything.
- **Restarts:** the risk state is restored from the `risk_state` table before anything
  trades, so a halt, the high-water mark and the day's counters survive a restart.

The broker is the source of current state; the database holds history (orders, fills, equity
snapshots, audit log) and is not replayed to rebuild state. The exception is the risk state,
which no broker knows.

## Risk

Two layers, both on the engine thread and identical in backtests and paper trading.

**Sizing** (`sizing.py`) turns the strategies' combined targets into the weights traded, on
every new bar:

1. *Caps:* each weight is clipped to the position limit, each asset class (crypto) is scaled
   into its limit, and the book into the gross limit (100%: no leverage).
2. *Volatility target:* the book's volatility is predicted from an EWMA covariance of daily
   log returns (RiskMetrics, lambda = 0.94, about 17 days of memory) on the dates every held
   instrument traded, so crypto's weekends do not distort equity correlations. When the
   prediction exceeds the target (8%) the book is scaled down; it is never levered up.
   Volatility scaling raises the Sharpe ratio of risk assets and balanced portfolios and
   thins their left tail (Harvey, Hoyle, Korgaonkar, Rattray, Sargaison and Van Hemert,
   "The Impact of Volatility Targeting", Journal of Portfolio Management, 2018).
3. *Drawdown ladder:* from 5% below the high-water mark the book shrinks linearly with the
   cushion above the 10% limit, as in Grossman and Zhou, "Optimal Investment Strategies for
   Controlling Drawdowns" (Mathematical Finance, 1993), down to 25% at the limit. The textbook
   rule goes to zero, which parks the book in cash just short of the limit, where it can
   never earn its way back and nobody is alerted ("cash lock"); a crash backtest showed
   exactly that. With the floor, losses that continue reach the limit and trip the kill
   switch.

Positions are traded only when they are more than the rebalance band (2% of equity) from
target, or to close them. Strategies with protective stops size entries with
`stop_risk_weight`: the weight that loses 0.5% of equity if the stop is hit.

**The gate** (`risk.py`) checks every order: halt, reference price, limit collar,
duplicates; then, for orders that add risk, the cool-down, market-data age and one-bar
jumps, daily order and turnover breakers, single-order size, and the position, asset-class
and gross caps on exposure including working orders. After each mark-to-market it updates
the high-water mark, starts a cool-down for the rest of the New York day once the day's loss
reaches 3%, and trips the kill switch (halt and flatten) at a 10% drawdown. A reconciliation
adjustment that raises equity (a deposit, a dividend) moves the high-water mark with it so it
cannot hide a drawdown; one that lowers equity counts as a loss.

## Safety

`src/trading/safety.py` refuses to start the app or build the Alpaca adapter unless
`TRADING_MODE=PAPER` and `ALPACA_BASE_URL` (if set) is the paper endpoint. The adapter only
connects to Alpaca's paper API.

## What comes next

Phase 4 adds walk-forward validation on top of the backtester and Phase 5 adds the
strategies; see [ROADMAP.md](ROADMAP.md).
