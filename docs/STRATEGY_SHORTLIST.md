# Strategy shortlist (research, 22 October 2026)

Phase 5 built four candidates (60/40 benchmark, month-end rebalancing flows, ETF time-series
momentum, crypto Donchian ensemble) and none passed the promotion gates. This file lists what
could come next. The operator picks, the rules are fixed in this file, and only then is a
candidate built and run through `flask validate`. The pick and its fixed rules are under
[Decisions](#decisions-22-october-2026).

What a candidate had to be:

- **Price-only**, plus public calendars at most: free Alpaca daily bars (US stocks and ETFs from
  2016, crypto from 2021). No paid data, no fundamentals, no options data.
- **Different from the Phase 5 four.**
- **Conservative:** long-only, no leverage, within the risk limits in [ROADMAP.md](ROADMAP.md).
- **Still working after it was published.** Published anomalies lose about half their returns
  after publication on average (McLean and Pontiff, *Journal of Finance* 2016: 26% out of
  sample, 58% after publication, across 97 predictors;
  [paper](https://ivey.uwo.ca/media/3775549/pontiff.pdf)). So for each idea I looked for the
  newest evidence, not only the original paper.

Most of the well-known price-only timing effects have published evidence that they faded (see
[Checked and rejected](#checked-and-rejected)). Three candidates remain, none with a strong
prior.

## A. Macroeconomic announcement days

**Source.** Savor and Wilson, "How Much Do Investors Care About Macroeconomic Risk? Evidence
from Scheduled Economic Announcements", *Journal of Financial and Quantitative Analysis* 2013.
Survey and update: Ai, Bansal and Guo, "Macroeconomic Announcement Premium", NBER working
paper 31923, 2023 ([pdf](https://www.nber.org/system/files/working_papers/w31923/w31923.pdf)).

**Published rule.** The stock market earns most of its premium on the days with scheduled
macroeconomic announcements. Ai, Bansal and Guo count FOMC statements, the employment report
(non-farm payrolls), GDP (first and last estimates), the ISM manufacturing index, and the earlier
of CPI and PPI each month. Savor and Wilson used employment, the earlier of CPI/PPI, and FOMC.

**Evidence for.**

| Sample | Announcement days | Other days |
|---|---|---|
| 1958–2009 (Savor and Wilson) | 11.4 bps a day | 1.1 bps a day |
| 1961–2023 (Ai, Bansal and Guo), 44 days a year | 10.68 bps a day, 4.65% a year, t = 5.36 | 0.93 bps a day, 1.94% a year |
| January 2020 to August 2023 | 16.33 bps a day | — |

The 1961–2023 figures are market returns in excess of T-bills; the announcement days carry
about 71% of the 6.59% yearly premium. A t-statistic of 5.36 over 63 years is an annual Sharpe
ratio of about 0.68 on those days alone (my calculation: 5.36 / √63).

**Evidence against.** Ernst, Gilbert and Hrdlicka (2019) argue that choosing the announcement
types after the fact, and not controlling for the day of the month, could bias the result;
FOMC days stay significant in their tests. Knox and Vissing-Jorgensen (Federal Reserve FEDS
2026-023, [pdf](https://www.federalreserve.gov/econres/feds/files/2026023pap.pdf)) report that
the equity premium implied by S&P 500 options is only 1–2 bps higher on FOMC days (August 2016
to December 2023). That suggests high FOMC-day returns were partly good news, not a reliable
premium.

**What it could add.** Stocks on about 44 days a year and cash or bonds on the rest: low
volatility, and a source of return unlike 60/40.

**Cost check.** Each event is a round trip at the close. At the 5 bps per fill the backtest
assumes, a round trip from cash costs 10 bps, about the whole average premium per event. It
only pays if closing-auction fills cost less than modelled. Some events fall on consecutive
days and share one round trip. The soak report measures real auction fills.

**Data and engine work.**

- The dates: a checked-in file of actual release dates, built from the Federal Reserve, BLS,
  BEA and ISM calendars (free). Use the dates releases actually came out; scheduled releases
  are sometimes postponed.
- Closing-auction orders. Alpaca queues a CLS order sent after 19:00 ET for the next day's
  closing auction ([Alpaca docs](https://docs.alpaca.markets/docs/orders-at-alpaca)). So the
  engine can send it at its 00:15 poll, because the dates are known in advance. The simulated
  broker already fills CLS orders at the close; the engine never sends them yet.

**Draft rules (to fix if chosen).**

- On the session before each announcement day, buy SPY at the close, up to the 60% position cap.
- Sell it at the close of the announcement day, unless the next session is also an
  announcement day.
- Grid: Savor and Wilson's three types against Ai, Bansal and Guo's five; cash or IEF on other
  days.

## B. Month-end payment cycle

**Source.** Etula, Rinne, Suominen and Vaittinen, "Dash for Cash: Monthly Market Impact of
Institutional Liquidity Needs", *Review of Financial Studies* 2020
([open access](https://acris.aalto.fi/ws/portalfiles/portal/40965155/hhz054.pdf)).

**Published rule.** T is the last trading day of the month. Institutions sell before month-end
payments, which depresses prices from T−8 to T−4. Prices recover from T−3 to T−1, and new money
lifts them from T to T+3.

**Evidence for.**

- On the CRSP value-weighted index from July 1995 to December 2013, the market returned −17 bps
  from T−8 to T−4 and +77 bps from T−3 to T+3.
- Above the average day, that is −37 bps and +48 bps in an average month. The same pattern
  shows in the paper's international sample of 22 equity markets.
- The paper's sample ends in 2013. So a test on 2017–2026, the period our data allows, is fully
  out of sample.

**Evidence against.** The classic turn-of-the-month window (T to T+3; McConnell and Xu,
*Financial Analysts Journal* 2008) looks faded in the last decade. A retest on S&P 500 futures
up to 2026 ([post](https://atbresearch.substack.com/p/the-turn-of-the-month-effect-tested),
not peer reviewed) finds the edge near zero and then negative in the latest ten years. An Aalto
thesis on 15 international indices reportedly finds it almost gone over 2014–2022 (as
summarized in search results; the thesis itself could not be opened).

**What it could add.** Stocks about 7 days a month, timed by a known calendar. The signal
differs from Phase 5's rebalancing flows, which traded the stock-bond spread depending on how
far a 60/40 portfolio had drifted.

**Data and engine work.**

- Data: only the NYSE calendar the engine already has.
- Engine: closing-auction orders, as for A.
- Cost: one round trip a month, 10 bps at the modelled 5 bps a fill.

**Draft rules (to fix if chosen).**

- Buy SPY at the close of T−4, up to the 60% position cap, and sell it at the close of T+3.
- Grid: T−3 to T+3 (the paper) against T to T+3 (McConnell and Xu); cash or IEF on other days.

## C. Industry trend on sector ETFs

**Source.** Zarattini and Antonacci, "A Century of Profitable Industry Trends" (2024; Charles H.
Dow Award 2025) ([summary](https://concretumgroup.com/a-century-of-profitable-industry-trends/)).
The rules below are as summarized by
[CXO Advisory](https://www.cxoadvisory.com/?p=45684).

**Published rule.** Long-only, daily, across 48 US industry portfolios:

- Buy an industry when its close crosses above the upper band of its 20-day Keltner channel
  (2 × the high-low range) or its 20-day Donchian channel.
- Exit below a trailing stop at the lower band of the 40-day Keltner or Donchian channel. The
  stop is never lowered.
- Resize daily so each position contributes the same volatility (14-day), with gross exposure
  up to 200%. Cash goes into 1-month T-bills.
- The authors also ran it on 31 SPDR sector and industry ETFs (launched 2005–2018) up to March
  2024.

**Evidence for.** From 1926 to 2024 it returned 18.5% a year at 12.1% volatility, a Sharpe
ratio of 1.46. The US market returned 9.7% a year at 17.1%, a Sharpe ratio of 0.63, and the
strategy cut its worst drawdown by almost 60%.

**Evidence against.**

- An independent replication with walk-forward tests, a Columbia student project
  ([arXiv 2412.14361](https://arxiv.org/abs/2412.14361)), fell short of the market out of sample.
- The paper allows 200% exposure; we cap it at 100%.
- Daily resizing adds turnover.
- It is the same family as Phase 5's ETF trend (time-series momentum), which failed, on a
  different universe.

**What it could add.** Sector rotation with crash protection, close to the existing Donchian
code (`crypto_trend.py`). Trades at the next open, so no engine change.

**Draft rules (to fix if chosen).**

- The paper's rules on the SPDR sector and industry ETFs with Alpaca history from 2016.
- Total exposure capped at 100% and each position at 60%; cash otherwise.
- Grid: the paper's 20/40-day channels and neighbouring lengths (for the stability gate).

## Checked and rejected

| Idea | Source | Why not |
|---|---|---|
| Overnight drift | Boyarchenko, Larsen and Whelan, *Review of Financial Studies* 2023 | The 2:00–3:00 a.m. window in S&P 500 futures has averaged close to zero since 2021, by the same authors ([Liberty Street, July 2026](https://libertystreeteconomics.newyorkfed.org/2026/07/the-disappearing-overnight-drift/)); it also needs futures overnight |
| Pre-FOMC drift | Lucca and Moench, *Journal of Finance* 2015 | Essentially disappeared after 2015 (Kurov, Wolfe and Gilbert, *Finance Research Letters* 2021) |
| FOMC-cycle even weeks | Cieslak, Morse and Vissing-Jorgensen, *Journal of Finance* 2019 | 2017–2021 shows a statistically significant opposite cycle (Knox and Vissing-Jorgensen, FEDS 2026-023) |
| Intraday momentum (last half hour) | Gao, Han, Li and Zhou, *JFE* 2018; Baltussen, Da, Lammers and Martens, *JFE* 2021 | A replication of a published SPY intraday momentum strategy (Zarattini, Aziz and Barbon) returned 9.4% out of sample, May 2024 to March 2026, against 29.5% for buy-and-hold (Paz, SSRN 2026); needs intraday data |
| Treasury auction cycle | Lou, Yan and Zhang, *Review of Financial Studies* 2013 | After 2015 the effect is less than half its earlier size; for the 10-year note 1.2 bps of yield and not significant (Fleming, Liu and Nguyen, New York Fed Staff Report 1188, 2026) |
| Turn of the month (T to T+3) | McConnell and Xu, *Financial Analysts Journal* 2008 | Faded in recent retests (see B); B is the variant with a mechanism and an untouched test period |
| Sell in May / Halloween | Bouman and Jacobsen, *American Economic Review* 2002 | No longer evident after January 2003 in any of the five indices tested, the S&P 500 included (Dichtl and Drobetz, as summarized by [CXO Advisory](https://www.cxoadvisory.com/calendar-effects/halloween-indicator-out-of-sample-test)); our data holds only nine summers |
| Pre-holiday days | Ariel, *Journal of Finance* 1990 | No longer significant for large firms after 1990 (Ko and Yang, *Critical Finance Review* 2024) |
| Volatility-managed portfolios | Moreira and Muir, *Journal of Finance* 2017 | Out of sample, across 103 equity strategies, generally no better than the unmanaged portfolios (Cederburg, O'Doherty, Wang and Yan, *JFE* 2020); the engine already targets volatility |
| Same-calendar-month seasonality | Keloharju, Linnainmaa and Nyberg, *Journal of Finance* 2016 | The signal averages each calendar month's returns over many past years (up to 20); Alpaca's history starts in 2016 |
| Factor momentum with factor ETFs | Ehsani and Linnainmaa, *Journal of Finance* 2022 | Shown for long-short factors; no evidence found for long-only factor ETFs |
| Dual momentum | Antonacci, working paper 2013 | Little independent evidence after publication; overlaps the ETF trend candidate that failed |

## Decisions (22 October 2026)

The operator left the pick to me, allowed any free data source, and allowed T-bills.

**The pick: B, the month-end payment cycle.** These rules were fixed and committed before any
backtest of it.

- **C is out on its own paper's numbers.** The authors' ETF test (31 SPDR ETFs, January 2005 to
  March 2024, with costs) returned 7.7% a year at 13.7% volatility. Its Sharpe ratio was 0.61
  against 0.59 for SPY, with a 24% maximum drawdown and exposure up to 200%. The authors note the
  strategy weakened over the last 20 years.
- **A loses money under our cost model.** Each announcement means moving between T-bills and SPY
  and back. That is four fills, or 20 bps at 5 bps a fill, against an average gain of 10.7 bps
  per announcement. It cannot pass the 3× cost gate even if the premium holds.
- **B pays its costs in the paper's sample.** It needs one round trip a month, 20 bps at the
  modelled cost, against +48 bps above the average day in 1995–2013 (+77 bps raw).
  - It has a mechanism with a causal test: when 13 European markets moved from three-day to
    two-day settlement in 2014, the reversal moved closer to month-end, as the mechanism predicts.
  - All of 2017–2026 is after the paper's sample.
  - It needs nothing beyond the NYSE calendar.
- **Expectation, stated before the test.** The paper's magnitudes imply a Sharpe ratio of about
  0.6 in excess of T-bills, below the 1.0 gate, and the classic turn of the month has faded. B
  is the cheapest honest test of the three, not a likely pass.

### Fixed rules: `month-end`

- **Instruments.** SPY in the window; the T-bill sleeve (below) at all other times.
- **Days.** T is the last NYSE session of a month. T−k and T+k count NYSE sessions.
- **Window.** Buy SPY in the closing auction of T−k and sell it in the closing auction of T+3.
  The position is held at the closes of T−k through T+2, so it earns the returns of T−k+1
  through T+3.
- **k, three variants (the whole grid):**
  - `settlement`, the paper's mechanism. An institution that needs cash on T must sell by the
    close of T−(s+1), where s is the settlement period. So k = 4 before 5 September 2017
    (T+3), 3 from 5 September 2017 (T+2) and 2 from 28 May 2024 (T+1). The rule in force on the
    entry day applies.
  - `paper`: k = 4 throughout, the paper's US window as published.
  - `turn`: k = 1, the classic turn of the month from T to T+3 (McConnell and Xu 2008).
- **Size.** The strategy targets 100% SPY in the window. The engine's standard sizing applies
  unchanged: the 60% position cap, the 8% volatility target and the drawdown ladder. So the SPY
  weight actually held is lower, and the rest sits in the T-bill sleeve.
- **Orders.** Market-on-close orders, sent when the previous session's bar is final (00:15 New
  York). Alpaca queues a closing-auction order sent after 19:00 ET for the next session's close.
  It rejects one sent after 15:50 and before 19:00, so the engine sends nothing in that window.
- **Costs.** The standard model: 5 bps per equity fill, closing-auction and BIL fills included.
- **Test.** `flask validate month-end --start 2017-01-01`. It uses the standard walk-forward
  (36-month training, 12-month test) and the ROADMAP gates, measured in excess of T-bills as
  below.
- **No other variant is tested on this data.** If B fails, it is recorded as rejected.

### Result: rejected (22 October 2026)

`flask validate month-end --start 2017-01-01`, on Alpaca data to 22 October 2026. Re-run on 23
October with the corrected T-bill sleeve (below); the verdict did not change:

| Gate | Value | Threshold | |
|---|---|---|---|
| Walk-forward Sharpe over T-bills | 0.05 | ≥ 1.0 | fail |
| Deflated Sharpe | 0.84 | ≥ 0.95 | fail |
| Probability of backtest overfitting | 0.68 | ≤ 0.20 | fail |
| At 3× costs | 2.1% against T-bills' 20.3% | above T-bills | fail |
| Parameter stability | 0.72 | ≥ 0.5 | pass |
| Maximum drawdown | −6.0% | ≥ −12% | pass |

- **Out of sample.** From January 2020 to October 2026 (the walk-forward) it returned 21.4%
  against 20.3% for T-bills. Over the whole period the best variant, `paper` (k = 4), returned
  42.7% against 25.7%, an excess Sharpe ratio of 0.32.
- **The effect.** On the days the `paper` window earns, SPY averaged 9.4 bps a day in 2017–2026,
  against 4.7 bps on the other days. That is the paper's direction at about two-thirds of its
  size, but it is not statistically significant (t = 1.0). After costs and a position the 8%
  volatility target holds to roughly 45%, it adds little over T-bills.
- **As fixed in advance,** no other variant is tested on this data. The code stays as the
  `month-end` candidate.

### T-bills

- **The sleeve.** The engine holds the T-bill ETF BIL (SPDR Bloomberg 1-3 Month T-Bill ETF,
  listed May 2007, expense ratio 0.1356%). It takes the equity not allocated to positions, minus
  the 0.5% cash buffer, for every strategy, the live 60/40 included.
- **How the risk rules treat it.**
  - It is exempt from the 60% position cap and the single-order size limit.
  - It counts toward gross exposure.
  - It may be bought during a daily-loss cool-down, since it adds no market risk.
  - It is never bought while trading is halted, and the kill switch sells it like any position.
- **Sized on holdings.** The sleeve takes what the positions leave once this rebalance's trades
  are done, not what their targets would leave. A target moving inside the rebalance band then
  trades nothing. The first version followed the targets: two positions each moving inside the
  band sold T-bills for cash. The dashboard's fills showed it on 23 October, and it was fixed.
- **The gates.** Every Sharpe-based number is computed on daily returns minus BIL's daily total
  return: the walk-forward Sharpe, its selection of parameter sets, the deflated and
  probabilistic Sharpe, the overfitting test, the cross-validation, the neighbours and the
  regimes. The cost-stress gate needs a total return above BIL's over the same days. Drawdown
  stays on total returns.

**Effect on the live 60/40.** Validated the same way from 2017, the 60/40 with its idle cash in
BIL returned 6.26% a year out of sample (2020 to October 2026), against 5.48% with the cash
earning nothing. Its maximum drawdown stayed at 14%. Its Sharpe ratio over T-bills is 0.52; the
0.81 reported before was against zero. `flask validate benchmark --no-tbills` reproduces the
earlier report exactly.

### Other decisions

- **Free data.** Any free source is allowed. B needs none beyond the NYSE calendar.
- **Auction costs.** These stay at the modelled 5 bps until the soak report's measurements of
  real auction fills justify a change.

## Round 2: free long-history data (23 October 2026)

The operator ruled out taking more market risk (option B) and paying for data (option E), and
chose to research ideas built on free data from outside Alpaca. Both sources work from here:

- **Kenneth French's Data Library** has daily US market and 1-month T-bill returns from 1 July
  1926 to August 2026 (from CRSP). Most of its portfolio sorts are monthly; the
  variance-sorted portfolios, for example, are monthly from July 1963.
- **FRED** has the 10-year Treasury yield, daily from 2 January 1962.

**What the long history adds.** Tests over 60 to 100 years instead of 9, through 1973–74, 1987,
2000–02 and 2008 rather than one or two crashes.

**Its limits:**
- Before ETFs existed these are index returns, not tradable funds.
- Bond returns have to be computed from yields, which is an approximation.
- Trading costs were higher in the past.

The promotion gates still come from `flask validate` on Alpaca data. The long history is an
additional test.

### D1. Trend filter on the equity sleeve

**Source.** Faber, "A Quantitative Approach to Tactical Asset Allocation" (*Journal of Wealth
Management* 2007; revisited in the *Journal of Portfolio Management* 2018).

**Published rule.** Hold stocks while the index is above its 10-month moving average and
T-bills otherwise, checked at each month end.

**Evidence for.**
- In Faber's 2018 update the filter raised the Sharpe ratio of US stocks from 0.45 to 0.80, per
  [Swedroe's summary](https://www.etf.com/sections/index-investor-corner/swedroe-why-financial-trends-persist).
- Hurst, Ooi and Pedersen, "A Century of Evidence on Trend-Following Investing" (AQR, 2017),
  find trend following positive in every decade since 1880. That is long-short across 67
  futures markets, though, not this rule.

**Evidence against.**
- Faber's own update: after helping in 2008–09, the filter lagged stocks in six of the next
  eight years.
- Zakamulin ("The real-life performance of market timing with moving average and time-series
  momentum rules", *Journal of Asset Management*; SSRN 2013) argues that earlier results were
  inflated by data mining. Out of sample, the rules beat buy-and-hold only in short episodes
  (per secondary summaries).
- Huang, Li, Wang and Zhou, "Time Series Momentum: Is It There?" (*JFE* 2020), find little
  evidence, market by market, that past returns predict future returns.

**What it could add.** The engine already cuts exposure by volatility and by drawdown. The
question this test answers is narrow: does a trend filter add anything on top of that? And
does it let the portfolio hold more equity within the same 10% kill-switch limit? Option B
failed because the drawdown limits cut in during crashes.

**Overlap.** It belongs to the same family as the rejected ETF time-series momentum, but uses
a single asset and a century of data.

**Data.** French's daily market returns (1926–), FRED's 10-year yield for bonds (1962–), and
Alpaca for 2017–2026.

### D2. Defensive (low-volatility) stocks instead of SPY

**Sources.** Blitz and van Vliet, "The Volatility Effect" (*JPM* 2007); Baker, Bradley and
Wurgler, "Benchmarks as Limits to Arbitrage: Understanding the Low-Volatility Anomaly" (*FAJ*
2011).

**Rule.** Hold the least volatile stocks (through an ETF such as USMV or SPLV) in place of SPY
in the 60/40.

**Evidence for.** Baltussen, Blitz and van Vliet, "The Volatility Effect Revisited" (*JPM*
2020), find the effect persistent across periods and markets, with no sign it has been
arbitraged away (per summaries).

**Evidence against.**
- USMV, the largest such ETF, returned 10.5% a year over 2016–2025 against 14.8% for MSCI USA.
  It beat the index only in 2018 and 2022. These are BlackRock's annual figures, compounded by
  me.
- It fell about as far as the S&P 500 in the 2020 crash (33%, per one analysis).

**What it could add.** Each dollar in it carries less volatility, so the 8% volatility target
would hold more equity. The test is whether its Sharpe ratio beats SPY's inside the engine.

**Data.** French's variance-sorted portfolios are monthly only (from July 1963), so the
long-history check would be monthly and outside the engine. The engine test runs on Alpaca
data with USMV or SPLV (2017–2026).

### D3. Economic trend (macro momentum)

**Sources.** Brooks, "A Half Century of Macro Momentum" (AQR, 2017); Brooks, Feilbogen, Ooi and
Akant, "Economic Trend" (AQR, 2023–24).

**Rule.** Position each market by trends in its economic fundamentals: growth forecasts,
inflation, monetary policy, trade and risk sentiment. It goes long and short across equity,
bond, currency and commodity futures.

**Evidence for.** AQR reports attractive returns over more than 50 years, with gains in the
largest equity drawdowns.

**Evidence against.**
- The evidence comes from the manager that sells the strategy.
- It is long-short in futures, not long-only in ETFs.
- It needs data as first released (FRED's ALFRED vintages) and growth forecasts, which have no
  free history.

**Verdict.** It can't be tested faithfully with free data.

### Checked and rejected (round 2)

| Idea | Source | Why not |
|---|---|---|
| Timing stocks with valuation and macro predictors (dividend yield, earnings yield, term spread, and others) | Goyal and Welch, *RFS* 2008 | Re-tested by Goyal, Welch and Zafirov (*RFS* 2024; 46 variables): more than a third are no longer significant even in-sample, and half of the rest do poorly out of sample |
| Timing bonds with the yield curve (forward rates) | Cochrane and Piazzesi, *AER* 2005 | No economic value to investors out of sample, worsening over time (Thornton and Valente, *RFS* 2012) |
| Volatility timing | Moreira and Muir, *JF* 2017 | Fails out of sample (Cederburg et al., *JFE* 2020); the engine already targets volatility |

### Recommendation (round 2)

D1, though not because its evidence is strong: it is mixed. It is the one idea where a century
of data answers a precise question about this engine.
- It is conservative and cheap to build, and the answer is useful either way.
- If it adds nothing, the current risk setup has been checked through 1973–74, 1987, 2000–02 and
  2008 instead of one crash.

D2 is the alternative. D3 is not recommended.

## Decision (23 October 2026): D1

The operator picked D1. These rules were fixed and committed before any backtest of it.

**Primary source checked.** Faber's paper (SSRN 962461, February 2013 update):
- "Buy when monthly price > 10-month SMA. Sell and move to cash when monthly price < 10-month
  SMA."
- Prices are total returns including dividends, at the close of the last day of the month;
  the rest of the month is ignored.
- Cash earns 90-day T-bills. Costs and taxes are excluded.
- "There is broad parameter stability from 3 months on out to over 12 months." He chose 10
  months as the one closest to the 200-day average.

**Expectation, stated before the test.**
- A 60/40 held to an 8% volatility target earned a Sharpe ratio of 0.52 over T-bills out of
  sample on Alpaca data. A filter on its equity sleeve is unlikely to double that, so D1 is not
  a likely pass of the 1.0 gate.
- The real question is whether it beats the 60/40 (below).

### Fixed rules: `trend-filter`

- **Portfolio.** The live 60/40: SPY 60% and IEF 40%. The idle equity goes to the T-bill
  sleeve (BIL), as for every strategy.
- **Signal.** At each month end (the month's last NYSE session), compare SPY's adjusted close
  (total return) with the mean of the last n month-end closes, that one included.
  - Above the mean: target SPY 60%.
  - Otherwise: SPY 0%, so that 60% goes to T-bills.
  - IEF stays at 40% throughout. The filter applies only to the equity sleeve, as picked.
  - Before n month ends exist, the strategy makes no decision.
- **n, three variants (the whole grid):** 8, 10 and 12 months. That is Faber's 10 and the
  neighbours inside the range he reports as stable.
- **Size.** The engine's standard sizing applies unchanged: the 60% position cap, the 8%
  volatility target and the drawdown ladder.
  - The cap rules out a higher equity target. The filter can add only by avoiding the
    drawdowns that make the ladder cut exposure.
  - The 8% target stays, since the operator ruled out more risk (option B).
- **Orders.** Each order goes at the opening auction of the next session, as for the live
  60/40, so a month-end signal trades on the first session of the next month.
- **Costs.** The standard model: 5 bps per equity fill.

**Test 1, the gates: Alpaca data.**
- `flask validate trend-filter --start 2017-01-01`, with the standard walk-forward (36-month
  training, 12-month test) and the ROADMAP gates, in excess of T-bills.
- This decides promotion, as for every candidate.

**Test 2, the long history: 1970 to August 2026.** Proxies built from free data, run through
the same engine and the same `flask validate`, with `--data history --start 1971-01-01 --end
2026-08-31`.
- **The proxies:**
  - **SPY:** Kenneth French's daily US market return (Mkt-RF plus RF: all NYSE, AMEX and
    NASDAQ stocks, value-weighted, dividends included, from CRSP).
  - **BIL:** French's daily 1-month T-bill return (RF).
  - **IEF:** the mean daily return of a 7-year and a 10-year Treasury par bond from FRED's
    constant-maturity yields (DGS7 and DGS10). IEF holds 7- to 10-year Treasuries.
    - Each day's return buys a par bond at the previous day's yield (coupon equal to that
      yield, paid twice a year). It reprices that bond at today's yield, with its maturity
      shortened by the calendar days elapsed, and adds the coupon accrued over them.
    - On a day without a FRED value, the last earlier yield applies.
- **How well the proxies track,** on daily returns from January 2016 to August 2026 against
  Alpaca's adjusted bars (2,679 days). The bond mix was chosen on this comparison, which uses no
  strategy results.

  | Proxy | Its return | ETF's return | Its volatility | ETF's volatility | Correlation | Tracking error |
  |---|---|---|---|---|---|---|
  | US market for SPY | 15.35% | 15.16% | 18.4% | 17.5% | 0.992 | 2.4% |
  | 7/10-year par bonds for IEF | 0.94% | 1.01% | 6.3% | 6.6% | 0.971 | 1.6% |
  | 10-year par bond alone | 0.72% | 1.01% | 7.2% | 6.6% | 0.966 | 1.9% |
  | 7-year par bond alone | 1.15% | 1.01% | 5.5% | 6.6% | 0.963 | 1.9% |
  | 1-month T-bills for BIL | 2.31% | 2.14% | 0.1% | 0.3% | 0.44 | 0.2% |

- **The bars.** One per NYSE session from 2 January 1970.
  - Open, high, low and close all equal the cumulative total-return level. An order sent after
    a session's bar therefore fills at the next session's close: a day later than Faber, and
    later than the live open.
  - Quantities are fractional, since the levels grow more than a hundredfold and whole units
    would be too coarse.
- **Why 1970.** From 1970 the NYSE calendar the engine uses matches French's trading days
  exactly (14,287 days). Before 1970 it doesn't: it lacks the Saturday sessions (until 1952)
  and many holidays (431 days in all). So the data from 1926 and FRED's 10-year yield from
  1962 go unused. The test still spans 1973–74, the 1970s bond losses, 1987, 2000–02, 2008,
  2020 and 2022.
- **Costs.** 5 bps per fill, as above. Real costs were higher before ETFs and cheap trading;
  the 3× cost gate (15 bps per fill) is the check.

**The comparison.** The 60/40 (`benchmark` candidate) is run through `flask validate` on the
same data and dates in both tests.

**What decides.**
- **Promotion:** only the gates of test 1, unchanged.
- **If D1 fails the gates but beats the 60/40 in both tests,** I report that and the operator
  decides whether it replaces the 60/40 as the live benchmark. "Beats" means a higher
  walk-forward Sharpe ratio over T-bills and a maximum drawdown no deeper. It would still not
  be a promoted strategy.
- **Otherwise** D1 is recorded as rejected, and the live engine keeps the 60/40.
- **No other variant is tested on this data,** whatever the result: no other filter length,
  no filter on bonds, no other volatility target.

### Result: rejected (23 October 2026)

Both tests were run as fixed above. The commands were `flask validate trend-filter --start
2017-01-01 --end 2026-10-23` and `flask validate trend-filter --data history --start 1971-01-01
--end 2026-08-31`, and the same for `benchmark`. The downloaded files' SHA-256 digests begin
`2f29e225` (French), `8e6ff8f4` (DGS7) and `e6bd270b` (DGS10).

**The gates (test 1, Alpaca, walk-forward 2020 to October 2026):**

| Gate | Value | Threshold | |
|---|---|---|---|
| Walk-forward Sharpe over T-bills | 0.53 | ≥ 1.0 | fail |
| Deflated Sharpe | 0.98 | ≥ 0.95 | pass |
| Probability of backtest overfitting | 0.84 | ≤ 0.20 | fail |
| At 3× costs | 41.7% against T-bills' 20.3% | above T-bills | pass |
| Parameter stability | 0.72 | ≥ 0.5 | pass |
| Maximum drawdown | −11.2% | ≥ −12% | pass |

**Against the 60/40, out of sample (walk-forward):**

| | D1, Alpaca | 60/40, Alpaca | D1, 1974–2026 | 60/40, 1974–2026 |
|---|---|---|---|---|
| Sharpe over T-bills | 0.529 | 0.528 | 0.557 | 0.582 |
| Return a year | 6.31% | 6.34% | 7.91% | 8.18% |
| Volatility | 6.9% | 7.0% | 6.4% | 6.6% |
| Maximum drawdown | −11.2% | −14.0% | −12.2% | −14.1% |
| Kill-switch trips | 1 | 1 | 2 | 5 |
| Gates failed | Sharpe, overfitting | Sharpe, overfitting, drawdown | Sharpe, overfitting, drawdown | Sharpe, overfitting, drawdown |

**The verdict.** D1 fails the gates. It also doesn't beat the 60/40 in both tests:
- On Alpaca data it ties: a Sharpe ratio of 0.529 against 0.528, with a shallower drawdown.
- Over 1974–2026 its Sharpe ratio is lower, 0.557 against 0.582.

So it is rejected, and the live engine keeps the 60/40.

**What the long history shows:**
- **The filter trades drawdowns for return.** It was in stocks at 75% of month ends and switched
  0.8 times a year, in line with Faber's "less than one round-trip trade per asset class per
  year".
  - Held fixed over 1971–2026 (10 months against the 60/40), it cut the maximum drawdown from
    −15.8% (1973–74) to −12.2% (1987).
  - It also cut the kill-switch trips from five (1974, 1987, 2001, 2008, 2022) to two (1987,
    1990).
  - It earned less: 7.61% a year against 7.92%, an identical Sharpe ratio of 0.51.
- **The month-end check can't react to a fast crash.** October 1987 fell inside one month,
  and both portfolios tripped the kill switch on 20 October.
- **No length stands out.** Over the full periods, 8, 10 and 12 months gave Sharpe ratios of
  0.65, 0.46 and 0.56 on Alpaca data, and 0.49, 0.51 and 0.55 since 1971. The fixed 60/40
  gave 0.68 and 0.51. The walk-forward's choices moved between lengths, and the overfitting
  probability (0.84 and 0.65) says the in-sample best length tells little about the next year.
- **The answer to the question asked.** The engine's volatility target and drawdown ladder
  already do most of what the filter does, so the filter adds no risk-adjusted return. It
  lowers drawdowns and kill-switch trips at a small cost in return, but not enough to pass the
  12% drawdown gate over 55 years.

**The current setup through five crashes.** Validated over 1974–2026, the 60/40 earned 8.18%
a year out of sample with a Sharpe ratio of 0.58 over T-bills. Its maximum drawdown was −14.1%
despite the 10% kill switch:
- The kill switch tripped five times in 55 years, about once a decade.
- The gap past 10% comes from moves between one session and the next fill, which in these
  proxies lands a day later than live.

**As fixed in advance,** no other variant is tested on this data. The code stays as the
`trend-filter` candidate, and `--data history` stays available to every candidate trading SPY
and IEF.

## Decision (23 October 2026): D2

The operator asked for D2 next. These rules were fixed and committed before any backtest of it.

**The two funds, from their providers:**
- **SPLV** (Invesco S&P 500 Low Volatility ETF; launched 5 May 2011; expense ratio 0.25%) tracks
  the S&P 500 Low Volatility Index.
  - The index holds the 100 S&P 500 stocks with the lowest standard deviation of their trailing
    252 daily returns, weighted by the inverse of that volatility.
  - It is rebalanced quarterly, in February, May, August and November, with no sector limits.
  - Sources: [S&P DJI research](https://www.spglobal.com/spdji/en/documents/research/research-sp-500-low-volatility-index-five-decades-of-history.pdf)
    and a [2012 SEC filing](https://www.sec.gov/Archives/edgar/data/83246/000114420412016831/v306941_fwp.htm).
    S&P's methodology PDF was not reachable from here (HTTP 403).
- **USMV** (iShares MSCI USA Min Vol Factor ETF; launched 18 October 2011; expense ratio
  0.15%; 174 holdings) tracks the MSCI USA Minimum Volatility Index.
  - That index is the minimum-variance portfolio of MSCI USA under a Barra risk model. Its
    stock, sector and factor weights are kept near the parent index's, and it is re-optimized
    each May and November ([MSCI methodology](https://www.msci.com/eqb/methodology/meth_docs/MSCI_Minimum_Volatility_Methodology_May2018.pdf)).

**What Alpaca has.** Both are active and tradable, with daily bars from 4 January 2016.
Checking that also showed their buy-and-hold returns from 4 January 2016 to 22 October 2026:

| | Return a year | Volatility |
|---|---|---|
| USMV | 10.85% | 14.0% |
| SPLV | 8.93% | 14.8% |
| SPY | 14.89% | 17.4% |

These figures were seen before the rules were written. They don't affect the rules, whose
only choice (which fund) the round-2 text had already named.

**No daily long history.** French's variance-sorted portfolios are monthly only, so D2 can't
go through the engine on long-history data. The decision rests on Alpaca data alone.

**Expectation, stated before the test.** Over 2016–2026 both funds gave up more return than
volatility against SPY. So D2 is unlikely to beat the 60/40, and very unlikely to reach the 1.0
gate.

### Fixed rules: `low-vol`

- **Portfolio.** The live 60/40 with the low-volatility fund in place of SPY: the fund at 60%
  and IEF at 40%. Idle equity goes to the T-bill sleeve (BIL).
- **The grid (two variants):** USMV and SPLV.
- **Size.** The engine's standard sizing applies unchanged: the 60% position cap, the 8%
  volatility target and the drawdown ladder.
  - The fund's lower volatility means the target scales the portfolio down less, so it holds
    more of both funds than the 60/40 holds of SPY and IEF. That is the mechanism under test.
- **Orders.** At the next session's opening auction, as for the live 60/40.
- **Costs.** The standard 5 bps per equity fill. Both funds trade about $200 million a day
  (median over the last 250 sessions).
- **Test.** `flask validate low-vol --start 2017-01-01 --end 2026-10-23`, with the standard
  walk-forward and the ROADMAP gates, in excess of T-bills.
- **The comparison.** `flask validate benchmark` on the same dates.
- **What decides:** the same rule as for D1.
  - Promotion only through the gates.
  - If D2 fails them but beats the 60/40 (a higher walk-forward Sharpe ratio over T-bills and a
    maximum drawdown no deeper), the operator decides whether it replaces the 60/40 as the
    live benchmark.
  - Otherwise D2 is recorded as rejected.
- **Monthly long-history check, descriptive only.** It runs outside the engine on French's
  "Portfolios Formed on Variance" (value-weighted; variance of the last 60 daily returns;
  formed monthly on NYSE breakpoints; file SHA-256 beginning `232fdeac`).
  - It compares the lowest-variance quintile with the market (French's daily Mkt-RF plus RF,
    compounded by month).
  - Over three periods: July 1963 to August 2026, July 1963 to December 2010 (before the
    funds), and January 2012 to August 2026 (both funds trading).
  - Measures: return a year, volatility, Sharpe ratio over the T-bill rate, and maximum
    drawdown on monthly returns.
  - It decides nothing.
- **No other variant is tested on this data,** whatever the result: no other fund (SPHD,
  LGLV, EFAV and the like), no other weights, no switching between the funds and SPY.
