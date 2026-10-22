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

### When equity orders go out

Market orders sent between 19:00 ET and two minutes before the open go to the opening
auction (OPG), and the rest are DAY orders. A strategy whose dates are known in advance (a
calendar) can set `execution = CLOSE`. The engine then sends its equity orders to the next
closing auction (CLS), as soon as the previous session's bar is final. Alpaca queues a CLS
order sent after 19:00 ET for the next session's close, and rejects one sent from 10 minutes
before the close (15:50 on a full day) until 19:00, so the engine sends nothing then and the
next bar retries. Strategies trading at the open and at the close cannot share an engine.

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

## Processes

`ENGINE_MODE` decides where the engine service lives:

- **embedded** (development): inside the single gunicorn worker, next to the API.
- **external** (the Compose stack): in its own process, `flask engine run`
  (`src/trading/worker.py`), so redeploying the API never restarts trading. The two meet only
  in the database (`src/trading/channel.py`). Every second the worker runs the queued
  commands in order and publishes the engine's views with a heartbeat to the single
  `engine_status` row; the API's `RemoteRuntime` answers reads from that row and sends
  commands through `engine_commands`, waiting for each result. A heartbeat older than 30
  seconds means the engine is unreachable: the API reports it and its emergency stop goes to
  Alpaca directly. A command the engine has not started when it expires, or when the API
  stops waiting, is never executed (an order must not fill minutes late); halts never expire.
  A second worker refuses to start while another's heartbeat is fresh.

Both processes run the migrations at start-up; on PostgreSQL an advisory lock serialises them.

## Monitoring

`src/trading/alerts.py` turns what needs a person into alerts (Telegram, ntfy, a JSON webhook),
sent from their own thread with retries and a 15-minute dedup window: the engine's events
(kill switch, halts, broker and strategy errors, fills it did not place, reconciliation
corrections) through an engine listener, and state changes (failed, an error, the daily loss
limit) through a monitor that watches the published views where the engine runs. The API's
monitor reports an engine process that stops answering, the one failure the engine cannot
report itself. A dead host cannot report anything, so `HEARTBEAT_URL` (healthchecks.io) is
pinged while the engine runs and alerts from outside when the pings stop.

`/health` is liveness; `/ready` adds the database; `/metrics` serves Prometheus gauges behind
`METRICS_TOKEN`. Logs are JSON lines on stdout in the Compose stack (`LOG_JSON`,
`LOG_TO_FILE`). Operations are in [RUNBOOK.md](RUNBOOK.md).

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

**The cash sleeve.** With a T-bill ETF as cash sleeve (`Engine(cash_sleeve=...)`; the live
lineup uses BIL), the equity the sized weights leave unused, minus the 0.5% cash buffer, is
held in it instead of in cash.
- It is bought after the strategies' positions are funded, and sold first when they need cash.
- It counts toward gross exposure but not toward the position or single-order caps.
- It may be bought during a cool-down, since it adds no market risk, but never while trading
  is halted.
- The kill switch sells it like every position.

**The gate** (`risk.py`) checks every order: halt, reference price, limit collar,
duplicates; then, for orders that add risk, the cool-down, market-data age and one-bar
jumps, daily order and turnover breakers, single-order size, and the position, asset-class
and gross caps on exposure including working orders. After each mark-to-market it updates
the high-water mark, starts a cool-down for the rest of the New York day once the day's loss
reaches 3%, and trips the kill switch (halt and flatten) at a 10% drawdown. A reconciliation
adjustment that raises equity (a deposit, a dividend) moves the high-water mark with it so it
cannot hide a drawdown; one that lowers equity counts as a loss.

## Validation

`src/trading/validation` decides whether a candidate (a strategy family and its parameter
grid) may paper trade. Every number in it comes from `run_backtest`, so it already includes
costs, the risk gate, sizing and the kill switch.

- **Trials:** each parameter set is backtested over the whole period. A trial's return on a
  day depends only on data before that day, so any slice of it is what that parameter set
  would have earned; selections below use only slices that ended before they are made.
- **Walk-forward** (`experiment.walk_forward`): calendar windows (36 months of training,
  12 of testing by default). At each test start the set with the best training Sharpe is
  chosen; a `SwitchingStrategy` then trades the chosen sets through the engine, consulting
  every set on every bar so a set switched in is current. The engine pays for every switch.
- **Selection bias** (`stats`): the Deflated Sharpe ratio of the best set, with the number of
  effectively independent trials N = rho + (1 - rho) M from their average correlation. Checked
  against the paper's own example (DSR 0.9004 for N = 100, 0.9505 for N = 46).
- **Overfitting** (`overfitting`): PBO by CSCV over C(16, 8) = 12,870 splits, computed from
  per-block sums so it stays fast; combinatorial purged cross-validation with an embargo after
  each test group, stitched into C(N-1, k-1) out-of-sample paths.
- **Robustness:** cost stress re-trades the walk-forward at 2x and 3x costs; neighbours are the
  sets one grid step away; regimes classify each day by the reference symbol's 6-month trend
  and 1-month volatility known when the day began.
- **Gates and report** (`gates`, `report`, `pipeline`): `validate()` returns a report with
  every gate decided, a Markdown tear sheet and a JSON form. The paper track record gate stays
  open until paper trading decides it.
- **The T-bill hurdle:** `flask validate` holds idle cash in BIL, as the live engine does. It
  measures every Sharpe-based number on daily returns minus BIL's daily total return
  (`ValidationConfig.hurdle`; `Returns.excess`). That covers the selection, walk-forward,
  deflated and probabilistic Sharpe, overfitting, cross-validation, neighbours and regimes.
  The cost-stress gate needs a total return above BIL's. Returns, volatility and drawdowns
  stay as earned. `--no-tbills` runs the earlier rules: no sleeve, Sharpe against zero.

The backtester's `resume_after_kill` simulates the operator: a kill halts and flattens, and
trading resumes that long after, once the book is flat. Without it a long backtest would stop
measuring the strategy at its first kill.

## Strategies

Strategies (`src/trading/strategies/`) decide from bar history alone and return target weights;
sizing, risk and execution belong to the engine. Each follows its published rules, and where a
paper is silent the module says what was assumed.

| Module | Rules | Source |
|---|---|---|
| `benchmark.BuyAndHold` | fixed weights, rebalanced by the engine | the yardstick |
| `rebalancing.RebalancingFlows` | simulate threshold (0-2.5%) and month-end 60/40 rebalancers; trade SPY against IEF ahead of their flows (`spread`), or tilt a 60/40 (`tilt`) | Harvey, Mazzoleni and Melone, NBER w33554 (2025) |
| `trend.TimeSeriesMomentum` | each month, sign of the 12-month return (or the clipped t-statistic of daily returns) times 40% / EWMA volatility (60-day centre of mass), averaged over the basket | Moskowitz, Ooi and Pedersen, JFE 2012; Baltas and Kosowski 2013 |
| `crypto_trend.DonchianEnsemble` | nine breakout models (5-360 days) with trailing stops at the channel midpoint, each sized to 25% volatility, averaged; long only | Zarattini, Pagani and Barbon, SFI 25-80 (2025) |
| `month_end.MonthEndCycle` | SPY from the closing auction of T-k to that of T+3 around each month's last session T; k follows the settlement cycle (4, 3 from 5 September 2017, 2 from 28 May 2024), or 4, or 1; trades at the close | Etula, Rinne, Suominen and Vaittinen, RFS 2020 |
| `blend.Blend` | fixed-weight sum of strategies, so a lineup validates and trades as one book | - |

A strategy held inside another (a blend, a walk-forward switch) is asked for targets through
`strategy.consult`, exactly as the engine would ask it on its own: once it has the history it
needs, seeing only its instruments and lookback. `Strategy.min_history` lets a blend be called
before its slowest component is ready, so a component whose data starts later (Alpaca's crypto
history begins in 2021) does not hold back the others.

## Safety

`src/trading/safety.py` refuses to start the app or build the Alpaca adapter unless
`TRADING_MODE=PAPER` and `ALPACA_BASE_URL` (if set) is the paper endpoint. The adapter only
connects to Alpaca's paper API.

## Dashboard and Telegram

The dashboard (`src/static/`) is plain HTML, CSS and one JavaScript module, served by the API
with no build step, so the source is what runs. It signs in with the JWT endpoints, keeps the
access token in memory and the single-use refresh token in `sessionStorage` (or
`localStorage` with "Stay signed in"), and rotates it one request at a time per tab, under a
Web Lock across tabs, since presenting a used refresh token revokes every session. It reads
the same endpoints as any API client and builds every element with DOM calls, never
`innerHTML`, under the API's `script-src 'self'` policy. Its files are served with
`Cache-Control: no-cache`, so an upgrade reaches phones at once. `tools/dashboard/` runs the
app with a simulated engine and drives it in Chromium at phone size.

Telegram commands (`src/trading/telegram.py`) are a second front end on the same runtime: a
long-polling bot in the API's serving process, answering only the operator's private chat,
with every change confirmed by a one-time code.

## API contract

The API is versioned under `/api/v1`. Each route declares its contract with `@documented`
(`src/routes/openapi.py`): a summary, who may call it, the request body and query parameters,
and the `data` of each response as a pydantic model (`src/routes/schemas.py`, unknown fields
forbidden). 500 and, unless exempt, 429 are implied, as are 401 for signed-in access and 403
for trader and admin access. `build_spec` walks the URL map, so an undeclared route cannot
reach the document: it fails the build. `src/tests/test_openapi.py` keeps the declarations
true: every signed-in route answers 401 without credentials, admin routes refuse a trader and
trader routes a reader, and a tour of the whole API checks every response against the
declared model for its status (an undeclared status fails) and that every operation was
exercised.

## What comes next

Phase 9 is the paper soak: at least 60 trading days of the engine trading the Alpaca paper account
unattended, then a staged live ramp only if the promotion gates hold; see
[ROADMAP.md](ROADMAP.md).
