"""
NOCTURNA Trading System - Long-history proxies for validation runs.

Free data from outside Alpaca, turned into daily bars the engine trades like
any other (docs/STRATEGY_SHORTLIST.md, "Fixed rules: trend-filter"):

* ``SPY``: Kenneth French's daily US market return, Mkt-RF plus RF (CRSP: all
  NYSE, AMEX and NASDAQ stocks, value-weighted, dividends included),
* ``BIL``: French's daily 1-month T-bill return, RF,
* ``IEF``: the mean daily return of a 7-year and a 10-year Treasury par bond
  priced from FRED's constant-maturity yields (DGS7 and DGS10). Each day buys
  a par bond at the previous day's yield (coupon equal to that yield, paid
  twice a year), reprices it at today's yield with its maturity shortened by
  the calendar days elapsed, and adds the coupon accrued over them. A day
  without a FRED value keeps the last earlier yield.

Open, high, low and close all equal the cumulative total-return level, so an
order sent after a session's bar fills at the next session's close. The bars
start on 2 January 1970: from then on the NYSE calendar matches CRSP's
trading days exactly, which ``load`` checks. Quantities are fractional, since
the levels grow more than a hundredfold over the period.
"""

from __future__ import annotations

import csv
import hashlib
import io
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal

from ..calendar import TradingCalendar
from ..domain import DAILY, NEW_YORK, AssetClass, Bar, Instrument

FRENCH_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
              "F-F_Research_Data_Factors_daily_CSV.zip")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
#: Bond maturities (years) averaged for the IEF proxy, and their FRED series
BOND_SERIES = {7: "DGS7", 10: "DGS10"}
FIRST_DAY = date(1970, 1, 2)
START_LEVEL = 100.0
INCREMENT = Decimal("0.0001")
SYMBOLS = ("SPY", "IEF", "BIL")
_TIMEOUT = 60  # seconds per download
#: FRED drops requests whose User-Agent it does not like ("nocturna-validation" alone, for one)
USER_AGENT = "nocturna-validation/1.0 (+https://github.com/Yog-Sotho/Nocturna-Trading-System)"


@dataclass(frozen=True)
class Source:
    """One downloaded file, for the record: what it was and what it covered."""

    url: str
    sha256: str
    first: date
    last: date


def download(url: str) -> bytes:
    # Only the fixed https URLs above are fetched
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # nosec B310
        data: bytes = response.read()
    return data


def parse_french(archive: bytes) -> dict[date, tuple[float, float]]:
    """(market total return, T-bill return) per day from French's daily factors archive."""
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"expected one CSV in French's archive, found {names}")
        text = z.read(names[0]).decode("latin-1")
    days: dict[date, tuple[float, float]] = {}
    for row in csv.reader(io.StringIO(text)):
        key = row[0].strip() if row else ""
        if len(key) != 8 or not key.isdigit():
            continue  # titles, headers and the copyright line
        mkt_rf, rf = float(row[1]) / 100, float(row[4]) / 100
        days[datetime.strptime(key, "%Y%m%d").date()] = (mkt_rf + rf, rf)
    if not days:
        raise ValueError("no daily rows in French's archive")
    return days


def parse_fred(text: str) -> dict[date, float]:
    """Yields (decimal, not percent) per day from a FRED CSV; days without a value are left out."""
    yields = {}
    lines = text.strip().splitlines()
    for line in lines[1:]:
        day, _, value = line.partition(",")
        value = value.strip()
        if value and value != ".":
            yields[date.fromisoformat(day.strip())] = float(value) / 100
    if not yields:
        raise ValueError("no values in the FRED series")
    return yields


def par_bond_return(previous_yield: float, current_yield: float, years: float, maturity: float) -> float:
    """
    The return over ``years`` of a ``maturity``-year par bond bought at
    ``previous_yield`` (its coupon, paid twice a year) and repriced at
    ``current_yield`` with the maturity left, coupon accrued included.
    """
    left = maturity - years
    periods = 2 * left
    if current_yield == 0:
        price = 1 + previous_yield * left
    else:
        discount = (1 + current_yield / 2) ** -periods
        price = previous_yield / current_yield * (1 - discount) + discount
    return price - 1 + previous_yield * years


def bond_returns(days: Sequence[date], yields: Mapping[date, float], maturity: float) -> dict[date, float]:
    """Par-bond returns for each of ``days`` after the first that has a yield on or before it."""
    observed = sorted(yields)
    returns: dict[date, float] = {}
    j, current = 0, None
    previous: tuple[date, float] | None = None
    for day in days:
        while j < len(observed) and observed[j] <= day:
            current = yields[observed[j]]
            j += 1
        if current is None:
            continue
        if previous is not None:
            returns[day] = par_bond_return(previous[1], current, (day - previous[0]).days / 365.25, maturity)
        previous = (day, current)
    return returns


def _bar_start(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=NEW_YORK).astimezone(UTC)


def to_bars(symbol: str, days: Sequence[date], returns: Mapping[date, float]) -> list[Bar]:
    """One bar per day at the cumulative level: START_LEVEL on the first day, then compounded."""
    bars, level = [], START_LEVEL
    for i, day in enumerate(days):
        if i:
            level *= 1 + returns[day]
        start = _bar_start(day)
        bars.append(Bar(symbol, start, DAILY.bar_end(start, AssetClass.EQUITY), level, level, level, level, 0.0))
    return bars


def _check_calendar(days: Sequence[date]) -> None:
    sessions = TradingCalendar(AssetClass.EQUITY).sessions(days[0], days[-1])
    if list(days) != sessions:
        extra = sorted(set(days) - set(sessions))[:3]
        missing = sorted(set(sessions) - set(days))[:3]
        raise ValueError(f"French's days differ from NYSE sessions (not sessions: {extra}, missing: {missing})")


def load(start: datetime, end: datetime, fetch: Callable[[str], bytes] = download
         ) -> tuple[dict[str, Instrument], dict[str, list[Bar]], list[Source]]:
    """
    Proxy bars for SPY, IEF and BIL that start in [``start``, ``end``) and
    not before FIRST_DAY, as Alpaca's bars are selected, with the sources
    downloaded.

    Raises:
        ValueError: If the data stops before ``end`` or does not match the NYSE calendar.
    """
    archive = fetch(FRENCH_URL)
    french = parse_french(archive)
    sources = [Source(FRENCH_URL, hashlib.sha256(archive).hexdigest(), min(french), max(french))]
    yields = {}
    for maturity, series in BOND_SERIES.items():
        url = FRED_URL.format(series=series)
        raw = fetch(url)
        yields[maturity] = parse_fred(raw.decode("utf-8"))
        sources.append(Source(url, hashlib.sha256(raw).hexdigest(), min(yields[maturity]),
                              max(yields[maturity])))

    calendar = TradingCalendar(AssetClass.EQUITY)
    for source in sources:
        if _bar_start(calendar.next_session(source.last)) < end:
            raise ValueError(f"{source.url} ends on {source.last}: pass an end no later than "
                             f"{calendar.next_session(source.last)}")
    all_days = sorted(french)
    days = [d for d in all_days if d >= FIRST_DAY and start <= _bar_start(d) < end]
    if len(days) < 2:
        raise ValueError(f"fewer than two sessions from {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    _check_calendar(days)
    # A day's bond return needs the yield of the day before, so they run over all of French's days
    bonds = [bond_returns(all_days, y, maturity) for maturity, y in yields.items()]
    if any(days[1] not in b for b in bonds):
        raise ValueError(f"FRED's yields start after {days[0]}")
    returns = {
        "SPY": {d: french[d][0] for d in days},
        "IEF": {d: sum(b[d] for b in bonds) / len(bonds) for d in days[1:]},
        "BIL": {d: french[d][1] for d in days},
    }
    instruments = {s: Instrument(s, AssetClass.EQUITY, qty_increment=INCREMENT, min_qty=INCREMENT) for s in SYMBOLS}
    return instruments, {s: to_bars(s, days, returns[s]) for s in SYMBOLS}, sources
