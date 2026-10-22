# Strategy shortlist (research, 22 October 2026)

Phase 5 built four candidates (60/40 benchmark, month-end rebalancing flows, ETF time-series
momentum, crypto Donchian ensemble) and none passed the promotion gates. This file lists what
could come next. Nothing here has been backtested: the operator picks, the rules are fixed in
this file, and only then is a candidate built and run through `flask validate`.

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

## Decisions for the operator

1. **Which candidates to fix and test.** A and B need closing-auction orders in the engine; C
   does not.
2. **For A: may a checked-in calendar of official release dates be used?** It is free, but it
   is not Alpaca data.
3. **Idle cash.** A and B hold no stocks on most days, and the 60/40 also leaves cash idle when
   its volatility target or drawdown ladder cuts exposure. Cash earns nothing in the backtest.
   The 3-month T-bill rate was 3.86% in late August 2026 (FRED, as reported by
   [MarketXLS](https://marketxls.com/indicators/3-month-treasury-bill-rate)). Holding a T-bill ETF
   instead (for example BIL, once confirmed in Alpaca's asset list) would add that income with
   little risk. But the Sharpe gate measures returns against zero, so T-bill income alone would
   raise every Sharpe ratio. If a T-bill ETF is allowed, the gate should measure returns in
   excess of T-bills.
4. **The cost of auction orders.** The backtest charges 5 bps for every equity fill. A depends on
   that figure. The soak report measures real opening-auction fills; any change to the cost model
   should come from those measurements, not from an assumption.
