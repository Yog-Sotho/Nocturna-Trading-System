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

- Out-of-sample Sharpe ≥ 1.0 net of costs (portfolio level)
- Deflated Sharpe ratio ≥ 0.95 and probability of backtest overfitting ≤ 0.20
- Still profitable at 2–3× assumed slippage; stable across neighbouring parameters
- Maximum drawdown within the risk limits, with an automatic kill switch
- ≥ 60 trading days of paper trading tracking the backtest within tolerance
- Live capital ramp: 10% → 25% → 50% → 100%

## Strategy lineup

| Mode | Strategy | Data | Status |
|---|---|---|---|
| SENTINEL | Trend following on a diversified basket of ~10–15 liquid ETFs (crypto optional), daily bars, entries only on statistically significant trends, volatility-scaled, weekly rebalance | Daily bars (free) | Candidate |
| LUCIFER | **Replaced.** Month-end pension-rebalancing flows (Harvey, Mazzoleni & Melone, NBER w33554, 2025): simulate a 60/40 SPY/IEF portfolio; Threshold signal (drift > ~2 pp) and Calendar signal (last week of month, reversal on the first business day); trade the equity-vs-bond spread against the expected flows. Two variants to test: long/short spread scaled to ~4–5% vol, and long-only SPY/IEF switch | Daily bars (free) | Candidate |
| CRYPTO BREAKOUT | Donchian ensemble (Zarattini, Pagani & Barbon 2025): 9 lookbacks 5–360 days, trailing stop at channel midpoint, volatility-targeted; long-only, no leverage, Alpaca's supported coins; client-side stops | Alpaca crypto data (free) | Candidate, small risk budget |
| REAPER | Large-cap/ETF short-term reversal with a trend filter, low-turnover construction | Daily bars (free) | Candidate |
| EVE | Grid trading — expected return ≈ 0 and ruin risk in trends | — | Retired from live; research only |
| Overlays | Regime model used only to scale exposure; crypto Fear & Greed and x70 signals removed from equity decisions | — | Planned |

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
| 5 | Strategy rebuild per the lineup above, each gated by Phase 4 | Planned |
| 6 | Operations: engine as its own service, structured logs, metrics, alerts, health/readiness, Docker/Compose fixes, backups, runbooks | Planned |
| 7 | Versioned API with OpenAPI docs; dashboard rebuilt with source and auth | Planned |
| 8 | CI quality gates: strict typing on core, coverage targets, mutation testing on risk, blocking security scans | Planned |
| 9 | ≥ 60-day paper soak, then staged live ramp | Planned |

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
- The engine runs inside the single web worker; a crash or redeploy restarts it from the
  broker's state. Phase 6 moves it into its own service.
