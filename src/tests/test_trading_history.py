"""
Tests for the long-history proxies: parsing French's and FRED's files, the
par-bond return against an explicit cash-flow price, and the bars the engine
receives.
"""

import hashlib
import io
import zipfile
from datetime import UTC, date, datetime, timedelta

import pytest

from src.trading.calendar import TradingCalendar
from src.trading.domain import DAILY, NEW_YORK, AssetClass
from src.trading.validation import history

SESSIONS = TradingCalendar(AssetClass.EQUITY)


def _french_zip(rows):
    text = ("This file was created using the 202608 CRSP database.\n"
            "The Tbill return is the simple daily rate.\n\n"
            ",Mkt-RF,SMB,HML,RF\n"
            + "".join(f"{d:%Y%m%d},{m:8.2f},{0.1:8.2f},{-0.1:8.2f},{rf:8.3f}\n" for d, m, rf in rows)
            + "\nCopyright 2026 Eugene F. Fama and Kenneth R. French\n")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("F-F_Research_Data_Factors_daily.csv", text)
    return buffer.getvalue()


def _fred_csv(series, values):
    return (f"observation_date,{series}\n" + "".join(f"{d.isoformat()},{v}\n" for d, v in values)).encode()


def _pv(coupon, yield_, years):
    """A bond's price from its cash flows: semiannual coupons, whole years to maturity."""
    periods = 2 * years
    flows = sum(coupon / 2 / (1 + yield_ / 2) ** k for k in range(1, periods + 1))
    return flows + 1 / (1 + yield_ / 2) ** periods


class TestParsing:
    def test_french_market_is_excess_return_plus_the_t_bill(self):
        rows = [(date(2020, 1, 2), 0.86, 0.006), (date(2020, 1, 3), -0.67, 0.006)]
        parsed = history.parse_french(_french_zip(rows))
        assert parsed == {date(2020, 1, 2): pytest.approx((0.0086 + 0.00006, 0.00006)),
                          date(2020, 1, 3): pytest.approx((-0.0067 + 0.00006, 0.00006))}

    def test_french_archive_without_rows(self):
        with pytest.raises(ValueError, match="no daily rows"):
            history.parse_french(_french_zip([]))

    def test_fred_yields_in_decimal_with_gaps_left_out(self):
        text = _fred_csv("DGS10", [(date(2020, 1, 2), "1.88"), (date(2020, 1, 3), ""),
                                   (date(2020, 1, 6), "."), (date(2020, 1, 7), "1.83")]).decode()
        assert history.parse_fred(text) == {date(2020, 1, 2): pytest.approx(0.0188),
                                            date(2020, 1, 7): pytest.approx(0.0183)}


class TestParBond:
    def test_an_unchanged_yield_earns_the_coupon(self):
        # Repriced at its own coupon rate a bond is at par, whatever the maturity left
        assert history.par_bond_return(0.05, 0.05, 3 / 365.25, 10) == pytest.approx(0.05 * 3 / 365.25, rel=1e-12)

    @pytest.mark.parametrize("new_yield", [0.0, 0.03, 0.0501, 0.07])
    def test_matches_the_price_from_cash_flows(self, new_yield):
        # Whole years elapsed, so the remaining coupons fall on whole periods
        expected = _pv(0.05, new_yield, 9) - 1 + 0.05
        assert history.par_bond_return(0.05, new_yield, 1.0, 10) == pytest.approx(expected, rel=1e-9)

    def test_a_basis_point_costs_about_the_duration(self):
        # A 10-year par bond at 5% has a modified duration of 7.8
        loss = history.par_bond_return(0.05, 0.0501, 0.0, 10)
        assert loss == pytest.approx(-0.00078, abs=0.000005)

    def test_daily_returns_accrue_over_calendar_days_and_keep_the_last_yield(self):
        days = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6), date(2020, 1, 7)]
        yields = {date(2019, 12, 31): 0.02, date(2020, 1, 3): 0.021, date(2020, 1, 7): 0.021}
        returns = history.bond_returns(days, yields, 10)
        assert set(returns) == set(days[1:])  # the first day has no day before it in ``days``
        assert returns[date(2020, 1, 3)] == pytest.approx(history.par_bond_return(0.02, 0.021, 1 / 365.25, 10))
        # Monday the 6th has no yield: the 3rd's carries over the weekend's three days
        assert returns[date(2020, 1, 6)] == pytest.approx(0.021 * 3 / 365.25)

    def test_days_before_the_first_yield_are_skipped(self):
        returns = history.bond_returns([date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)],
                                       {date(2020, 1, 3): 0.02}, 7)
        assert list(returns) == [date(2020, 1, 6)]


# -- loading ----------------------------------------------------------------------------------

FIRST, LAST = date(1969, 12, 1), date(1970, 3, 31)


def _sources(first=FIRST, last=LAST, drop=None):
    days = [d for d in SESSIONS.sessions(first, last) if d != drop]
    french = _french_zip([(d, 0.1 if i % 2 else -0.05, 0.03) for i, d in enumerate(days)])
    fred = {s: _fred_csv(s, [(d, f"{6 + 0.01 * (i % 3) + (0.5 if s == 'DGS7' else 0):.2f}")
                             for i, d in enumerate(days)]) for s in history.BOND_SERIES.values()}
    files = {history.FRENCH_URL: french, **{history.FRED_URL.format(series=s): v for s, v in fred.items()}}
    return files


def _load(files, start=datetime(1969, 11, 1, tzinfo=UTC), end=datetime(1970, 4, 1, tzinfo=UTC)):
    return history.load(start, end, fetch=files.__getitem__)


class TestLoad:
    def test_bars_for_the_three_proxies_from_1970(self):
        files = _sources()
        instruments, bars, sources = _load(files)
        days = SESSIONS.sessions(date(1970, 1, 2), date(1970, 3, 31))
        assert set(bars) == set(instruments) == {"SPY", "IEF", "BIL"}
        for symbol, series in bars.items():
            assert [b.start.astimezone(NEW_YORK).date() for b in series] == days
            assert series[0].close == history.START_LEVEL
            assert all(b.open == b.high == b.low == b.close and b.end == DAILY.bar_end(b.start, AssetClass.EQUITY)
                       for b in series)
            assert instruments[symbol].qty_increment == history.INCREMENT
        spy = bars["SPY"]
        french = history.parse_french(files[history.FRENCH_URL])
        assert spy[1].close / spy[0].close - 1 == pytest.approx(french[days[1]][0])
        assert bars["BIL"][1].close / bars["BIL"][0].close - 1 == pytest.approx(0.0003)
        assert [s.sha256 for s in sources] == [hashlib.sha256(files[s.url]).hexdigest() for s in sources]
        assert sources[0].first == SESSIONS.sessions(FIRST, LAST)[0]

    def test_bonds_average_the_seven_and_ten_year_par_bonds(self):
        files = _sources()
        _, bars, _ = _load(files)
        days = SESSIONS.sessions(FIRST, LAST)
        legs = []
        for maturity, series in history.BOND_SERIES.items():
            yields = history.parse_fred(files[history.FRED_URL.format(series=series)].decode())
            legs.append(history.bond_returns(days, yields, maturity))
        ief = bars["IEF"]
        first_day = ief[0].start.astimezone(NEW_YORK).date()
        second_day = ief[1].start.astimezone(NEW_YORK).date()
        assert first_day == date(1970, 1, 2)
        # The first bar's return is not used; the second's needs the 2 January yield
        assert ief[1].close / ief[0].close - 1 == pytest.approx((legs[0][second_day] + legs[1][second_day]) / 2)

    def test_bars_start_before_end_as_alpacas_do(self):
        # A bar starts at midnight in New York, 05:00 UTC in winter
        _, bars, _ = _load(_sources(), start=datetime(1970, 2, 2, 6, tzinfo=UTC),
                           end=datetime(1970, 3, 2, 6, tzinfo=UTC))
        days = [b.start.astimezone(NEW_YORK).date() for b in bars["SPY"]]
        assert days[0] == date(1970, 2, 3) and days[-1] == date(1970, 3, 2)

    def test_refuses_an_end_after_the_data(self):
        with pytest.raises(ValueError, match="ends on 1970-03-31"):
            _load(_sources(), end=datetime(1970, 4, 3, tzinfo=UTC))

    def test_refuses_data_that_misses_an_nyse_session(self):
        with pytest.raises(ValueError, match="differ from NYSE sessions"):
            _load(_sources(drop=date(1970, 2, 10)))

    def test_refuses_yields_that_start_too_late(self):
        files = _sources()
        late = _fred_csv("DGS7", [(d, "6.00") for d in SESSIONS.sessions(date(1970, 2, 2), LAST)])
        files[history.FRED_URL.format(series="DGS7")] = late
        with pytest.raises(ValueError, match="FRED's yields start after"):
            _load(files)

    def test_downloads_with_a_timeout(self, monkeypatch):
        calls = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def urlopen(request, timeout):
            calls.append((request.full_url, request.get_header("User-agent"), timeout))
            return Response(b"payload")

        monkeypatch.setattr(history.urllib.request, "urlopen", urlopen)
        assert history.download(history.FRENCH_URL) == b"payload"
        assert calls == [(history.FRENCH_URL, history.USER_AGENT, 60)]
        assert history.USER_AGENT.startswith("nocturna-validation/1.0 (+https://")


def test_the_calendar_reaches_back_to_1970():
    # The long history needs NYSE sessions from 2 January 1970 (Friday)
    assert SESSIONS.sessions(date(1970, 1, 1), date(1970, 1, 9)) == [date(1970, 1, 2), date(1970, 1, 5),
                                                                       date(1970, 1, 6), date(1970, 1, 7),
                                                                       date(1970, 1, 8), date(1970, 1, 9)]
    assert SESSIONS.next_session(date(1970, 1, 30)) == date(1970, 2, 2)
    assert timedelta(0) < SESSIONS.session_close(date(1970, 1, 2)) - SESSIONS.session_open(date(1970, 1, 2))
