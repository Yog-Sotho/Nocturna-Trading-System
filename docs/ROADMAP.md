# NOCTURNA Rebuild Roadmap

This file records the agreed plan and the decisions behind it. It is updated as phases land.

## Objective

Maximise after-cost, risk-adjusted growth of a single operator's account within hard risk
limits. No strategy trades live unless it passes the promotion gates below.

## Operator decisions

| Topic | Decision |
|---|---|
| Architecture | One in-repo event-driven core; identical strategy code in backtest, paper and live; Alpaca via `alpaca-py` |
| Market data budget | $0 — free Alpaca data only (full-market SIP history ending ≥ 15 min ago, live IEX stream ≤ 30 symbols, crypto data) |
| Asset scope | US equities/ETFs and crypto, each admitted only if it passes the gates |
| Risk profile | Conservative / low risk |
| Users | Single operator (public registration disabled) |

### Conservative risk limits (confirmed by the operator after the real-data run below)

- Maximum drawdown kill switch: 10% below the equity high-water mark (halt and flatten)
- Drawdown ladder: exposure shrinks linearly from 5% drawdown to 25% at the limit
- Daily loss limit: 3% (no new risk for the rest of the day)
- Risk per trade for stop-based strategies: 0.5% of equity
- Portfolio volatility target: 8% annualised (scale down only, no leverage)
- Caps: 60% per position, 15% crypto in total, 100% gross

## Promotion gates

A strategy moves research → paper → small live → scaled live only if:

- Out-of-sample Sharpe ≥ 1.0 net of costs (portfolio level), in excess of T-bills (BIL) since
  22 October 2026: every Sharpe-based gate uses daily returns minus BIL's, and idle cash is held
  in BIL (see [STRATEGY_SHORTLIST.md](STRATEGY_SHORTLIST.md#t-bills))
- Deflated Sharpe ratio ≥ 0.95 and probability of backtest overfitting ≤ 0.20
- Still profitable at 2–3× assumed slippage; stable across neighbouring parameters
- Maximum drawdown within the risk limits, with an automatic kill switch
- ≥ 60 trading days of paper trading tracking the backtest within tolerance (below)
- Live capital ramp: 10% → 25% → 50% → 100%

### Paper soak tolerance (agreed with the operator)

`flask soak report` compares paper trading with a backtest of the same days, started on the
bar the engine first acted on, with the same equity, lineup, sizing and risk limits. The
soak passes when all five hold:

- At least 60 trading days recorded (a day counts when the engine recorded its equity)
- Every engine error, refused order, unexpected fill, halt and kill-switch trip reviewed
- Every reconciliation difference with Alpaca explained; a cash difference counts as
  explained when the account's own entries (dividends, interest, fees) add up to it within
  a cent
- Tracking error, the annualised spread of daily return differences, below 2%
- Total return within 1 percentage point of the backtest

Returns are compared from the first session at whose close both had traded. Expect small
differences around dividends: the backtest's bars are dividend-adjusted, while Alpaca pays
the dividend as cash. The report also measures opening-auction fills against the open as
traded, next to the 5 bps the backtest assumes.

## Strategy lineup

| Mode | Strategy | Data | Status |
|---|---|---|---|
| SENTINEL | Time-series momentum on 12 liquid ETFs across asset classes (`tsmom`): sign of the 12-month return, or the clipped t-statistic of daily returns (statistically significant trends), each sized 40% / EWMA volatility, monthly (Moskowitz, Ooi & Pedersen 2012) | Daily bars (free) | Built; alone 2017–2026: walk-forward Sharpe 0.32, long-only full period 0.71; a lineup component |
| LUCIFER | **Replaced.** Month-end pension-rebalancing flows (Harvey, Mazzoleni & Melone, NBER w33554, 2025): simulate a 60/40 SPY/IEF portfolio; Threshold signal (drift > ~2 pp) and Calendar signal (last week of month, reversal on the first business day); trade the equity-vs-bond spread against the expected flows. Two variants to test: long/short spread scaled to ~4–5% vol, and long-only SPY/IEF switch | Daily bars (free) | Built (`rebalancing`); rejected for now: the signal works close-to-close (Sharpe 0.64 on 2017–2026 SPY/IEF, before costs) but only 0.24 at the next open, and it turns over about 43 times a year. Needs close-of-day execution |
| CRYPTO BREAKOUT | Donchian ensemble (Zarattini, Pagani & Barbon 2025): 9 lookbacks 5–360 days, trailing stop at channel midpoint, volatility-targeted; long-only, no leverage, BTC/ETH/SOL; the engine trades the stops at the next bar | Alpaca crypto data (free, from 2021) | Built (`crypto-trend`); alone 2023–2026: Sharpe 0.85, walk-forward 2024–2026 0.45; a lineup component at the 15% crypto limit |
| MONTH-END | Month-end payment cycle (Etula, Rinne, Suominen & Vaittinen, RFS 2020): SPY from the close of T−k to the close of T+3 around each month's last session, T-bills otherwise; k follows the settlement cycle | Daily bars (free) | Picked 22 October 2026; rules fixed in [STRATEGY_SHORTLIST.md](STRATEGY_SHORTLIST.md#fixed-rules-month-end) before any backtest |
| REAPER | Large-cap/ETF short-term reversal with a trend filter, low-turnover construction | Daily bars (free) | Deferred: net-of-cost reversal evidence is for individual large-cap stocks (residual reversal), whose honest test needs point-in-time index membership (not free); no peer-reviewed ETF evidence was found |
| EVE | Grid trading — expected return ≈ 0 and ruin risk in trends | — | Retired from live; research only |
| Overlays | Regime model used only to scale exposure; crypto Fear & Greed and x70 signals removed from equity decisions | — | Planned |

The next candidates, with their sources and the evidence for and against them, are in
[STRATEGY_SHORTLIST.md](STRATEGY_SHORTLIST.md); none is built until the operator picks one and
its rules are fixed there.

**Phase 5 outcome.** The lineup (60/40 core, ETF trend, crypto trend at 15%; two mixes and
two trend variants fixed before any lineup backtest) was validated as one candidate on Alpaca
data from 2017 to October 2026 and rejected: walk-forward Sharpe 0.76 (gate 1.0), 4.6% a year,
maximum drawdown -11.0%, probability of backtest overfitting 0.24 (gate 0.20); deflated Sharpe
0.999, 3x costs, stability and drawdown passed. Out of sample it did not beat the 60/40
benchmark (Sharpe 0.81, 5.5% a year, drawdown -14.1%); its best full-period mix (Sharpe 1.01)
was in-sample selection. No candidate is promoted, so paper trading keeps the benchmark.

The original LUCIFER (opening-range breakout on "stocks in play") was dropped because its
stock selection needs full-market real-time volume, which requires a $99/month data plan.

### Known risks of the selected strategies

- **Rebalancing flows:** paper sample ends 2023-03-17, so 2023–2026 must be tested out of
  sample; published in 2025 and may be crowded; at least one large fund (Korea's NPS)
  suspended rebalancing in H1 2026; paper used futures, we use ETFs (IEF is 7–10y, not the
  10-year note).
- **Crypto breakout:** published Sharpe > 1.5 comes from a 2015–2025 crypto bull sample;
  independent recent tests show mainly drawdown reduction, not extra return.

## Phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Safety freeze: paper-only guard, closed registration, no fabricated data, truthful README | Done |
| 1 | Platform foundations: dependency clean-up with hashed lockfile, validated settings, auth fixes, DB-backed scoped API keys, token revocation, enforced rate limits, proxy-aware client IPs, Alembic migrations + PostgreSQL, persisted audit log | Done |
| 2 | Event-driven core: domain model, event flow, one broker interface (simulated + Alpaca via `alpaca-py`), data layer with explicit ranges and closed bars, removal of the legacy engine and `alpaca-trade-api` | Done |
| 3 | Risk engine: pre-trade controls (price collar, size/exposure caps, duplicate detection, kill switch), equity-based loss limits, volatility-targeted sizing, PDT logic removed | Done |
| 4 | Validation pipeline: walk-forward, combinatorial purged cross-validation, deflated Sharpe, overfitting probability, cost stress, regime splits, tear sheets | Done |
| 5 | Strategy rebuild per the lineup above, each gated by Phase 4 | Done: four candidates built and validated on Alpaca data; none passed, so the engine keeps the 60/40 benchmark |
| 6 | Operations: engine as its own service, structured logs, metrics, alerts, health/readiness, Docker/Compose fixes, backups, runbooks | Done: see [RUNBOOK.md](RUNBOOK.md) |
| 7 | Versioned API with OpenAPI docs; dashboard rebuilt with source and auth | Done: the API under `/api/v1` with an OpenAPI 3.1 document kept true by contract tests; the dashboard (`src/static/`) and Telegram commands |
| 8 | CI quality gates: strict typing on core, coverage targets, mutation testing on risk, blocking security scans | Done: mypy `--strict` on the trading core, coverage floors (94% overall, 95% per core module), weekly Cosmic Ray mutation testing of `risk.py` and `sizing.py`, blocking bandit and pip-audit, dashboard and Compose-stack jobs; see the README's quality gates |
| 9 | ≥ 60-day paper soak, then staged live ramp | In progress: the soak report (`flask soak report`) checks the tolerance above; the soak starts when the operator starts the engine on the paper account |

### Sequencing notes

- Phase 2 removed `alpaca-trade-api` with the legacy engine, the paper toy engine
  (`/api/paper`), the ML tuner and the sentiment analyser, and moved to Python 3.13. The
  dependency audit in CI now ignores nothing.
- The live engine runs only the 60/40 SPY/IEF benchmark until a strategy passes the Phase 4
  gates. It exercises orders, fills, reconciliation and history in paper trading meanwhile.
- Phase 3 widened the rebalance band to 2% of equity and rebalances on every bar, because
  the volatility and drawdown scales move with the market.
- The PDT logic went with the legacy engine in Phase 2 and needs no replacement: FINRA
  eliminated the pattern-day-trader rule and its $25,000 minimum (Rule 4210 amendments, SEC
  approval 14 April 2026, effective 4 June 2026, Regulatory Notice 26-10), and Alpaca now
  applies its intraday margin rule instead. The engine uses no leverage (gross exposure is
  capped at 100% of equity).
- The risk limits were confirmed on real data (Alpaca, SPY/IEF 60/40, January 2017 to
  October 2026, a single history with one large drawdown in 2022):

  | Settings | Return/yr | Volatility | Sharpe | Max drawdown |
  |---|---|---|---|---|
  | Kill switch 10%, ladder, 8% volatility target (chosen) | 6.6% | 6.9% | 0.96 | -12.1% |
  | No kill switch or ladder, 8% target | 7.1% | 7.3% | 0.98 | -14.1% |
  | No kill switch or ladder, 10% target | 7.9% | 8.2% | 0.96 | -16.8% |
  | Plain 60/40 | 9.5% | 10.7% | 0.90 | -21.2% |

  The kill switch and ladder cost about 0.5% a year for 2 points less drawdown; the volatility
  target sets the return. The validated benchmark family was rejected (walk-forward Sharpe
  0.81, PBO 0.74, out-of-sample drawdown -14.1% after a kill on 2022-06-11), as a passive
  yardstick should be.
- Phase 4 validates one candidate at a time. The 1.0 Sharpe gate is for the portfolio; a
  component can be validated with `--min-sharpe`, and Phase 5 validates the combined lineup as
  its own candidate.
- Alpaca's free plan serves equity history from 2016: with 36 months of training, about six
  years are out of sample.
- The Alpaca adapter is tested against fakes with alpaca-py's real request models; its first
  run against a live paper account should be watched (order acknowledgements, fill events,
  crypto fee postings, position symbol formats).
- In the Compose stack the engine runs as its own process (`ENGINE_MODE=external`), so an API
  redeploy never restarts it; an engine restart rebuilds from the broker's state and the
  persisted risk state. The stack was verified end to end without Alpaca keys (commands
  across processes, crash restart, alerts when the engine stops answering, the emergency
  stop's broker fallback, backup and in-place restore, the HTTPS proxy); its first run
  against the paper account should be watched.
