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
