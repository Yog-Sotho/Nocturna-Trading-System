"""
Tests for the trading core's domain model, ledger, clocks and calendars.
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, localcontext

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.trading.calendar import TradingCalendar
from src.trading.clock import SimulatedClock
from src.trading.domain import (
    DAILY,
    AssetClass,
    Bar,
    Fill,
    Instrument,
    InvalidTransitionError,
    Order,
    OrderRequest,
    OrderStatus,
    OrderType,
    OrderUpdate,
    Side,
    Timeframe,
)
from src.trading.ledger import LEDGER_CONTEXT, Ledger

T0 = datetime(2026, 3, 2, 14, 30, tzinfo=UTC)
D = Decimal


def _fill(n, side, qty, price, fee="0", symbol="SPY", order="o1"):
    return Fill(f"f{n}", order, symbol, side, D(qty), D(price), D(fee), T0 + timedelta(minutes=n))


class TestBar:
    def _bar(self, **kw):
        values = dict(symbol="SPY", start=T0, end=T0 + timedelta(days=1), open=10.0, high=11.0,
                      low=9.0, close=10.5, volume=100.0)
        values.update(kw)
        return Bar(**values)

    def test_valid_bar(self):
        assert self._bar().close == 10.5

    @pytest.mark.parametrize("kw", [
        {"high": 10.2},                         # below close
        {"low": 10.1},                          # above open
        {"open": 0.0, "low": 0.0},              # non-positive price
        {"volume": -1.0},
        {"end": T0},                            # empty interval
        {"start": datetime(2026, 3, 2)},        # naive datetime
    ])
    def test_invalid_bars_rejected(self, kw):
        with pytest.raises(ValueError):
            self._bar(**kw)


class TestTimeframe:
    def test_daily_equity_bar_ends_at_next_new_york_midnight_across_dst(self):
        # 2026-03-07 midnight New York is 05:00Z; DST starts on 03-08, so the next
        # midnight is 04:00Z: the bar is 23 hours long
        start = datetime(2026, 3, 8, 5, 0, tzinfo=UTC)
        assert DAILY.bar_end(start, AssetClass.EQUITY) == datetime(2026, 3, 9, 4, 0, tzinfo=UTC)

    def test_daily_crypto_bar_is_24_hours(self):
        start = datetime(2026, 3, 8, tzinfo=UTC)
        assert DAILY.bar_end(start, AssetClass.CRYPTO) == datetime(2026, 3, 9, tzinfo=UTC)

    def test_intraday(self):
        assert Timeframe(15, "minute").bar_end(T0, AssetClass.EQUITY) == T0 + timedelta(minutes=15)

    @pytest.mark.parametrize("amount,unit", [(0, "day"), (1, "week")])
    def test_invalid(self, amount, unit):
        with pytest.raises(ValueError):
            Timeframe(amount, unit)


class TestInstrument:
    @pytest.mark.parametrize("qty,expected", [("1.9", "1"), ("-1.9", "-1"), ("0.4", "0")])
    def test_round_qty_whole_shares(self, qty, expected):
        assert Instrument("SPY", AssetClass.EQUITY).round_qty(D(qty)) == D(expected)

    def test_round_qty_crypto_increment(self):
        btc = Instrument("BTC/USD", AssetClass.CRYPTO, qty_increment=D("0.0001"), min_qty=D("0.0001"))
        assert btc.round_qty(D("0.123456")) == D("0.1234")


class TestOrderRequest:
    @pytest.mark.parametrize("kw", [
        {"qty": D("0")},
        {"order_type": OrderType.LIMIT},
        {"order_type": OrderType.STOP, "stop_price": None},
        {"limit_price": D("10")},
        {"client_order_id": ""},
        {"client_order_id": "x" * 49},
        {"order_type": OrderType.LIMIT, "limit_price": D("-1")},
    ])
    def test_invalid(self, kw):
        values = dict(client_order_id="o1", symbol="SPY", side=Side.BUY, qty=D("1"))
        values.update(kw)
        with pytest.raises(ValueError):
            OrderRequest(**values)


class TestOrderLifecycle:
    def _order(self, qty="10"):
        return Order.new(OrderRequest("o1", "SPY", Side.BUY, D(qty)), T0)

    def _update(self, status, fill=None):
        return OrderUpdate("o1", status, T0 + timedelta(minutes=1), fill=fill, broker_order_id="b1")

    def test_partial_then_full_fill_tracks_average_price(self):
        order = self._order()
        order = order.apply(self._update(OrderStatus.NEW))
        order = order.apply(self._update(OrderStatus.PARTIALLY_FILLED, _fill(1, Side.BUY, "4", "100")))
        order = order.apply(self._update(OrderStatus.FILLED, _fill(2, Side.BUY, "6", "110")))
        assert order.status is OrderStatus.FILLED
        assert order.filled_qty == D("10")
        assert order.avg_fill_price == D("106")
        assert order.broker_order_id == "b1"
        assert order.remaining_qty == 0

    def test_replayed_fill_is_ignored(self):
        fill = _fill(1, Side.BUY, "4", "100")
        order = self._order().apply(self._update(OrderStatus.PARTIALLY_FILLED, fill))
        assert order.apply(self._update(OrderStatus.PARTIALLY_FILLED, fill)) is order

    @pytest.mark.parametrize("path", [
        [OrderStatus.FILLED, OrderStatus.NEW],
        [OrderStatus.CANCELED, OrderStatus.FILLED],
        [OrderStatus.REJECTED, OrderStatus.CANCELED],
        [OrderStatus.NEW, OrderStatus.PENDING_NEW],
    ])
    def test_impossible_transitions_rejected(self, path):
        order = self._order()
        with pytest.raises(InvalidTransitionError):
            for status in path:
                order = order.apply(self._update(status))

    def test_fill_after_cancel_request_is_accepted(self):
        order = self._order().apply(self._update(OrderStatus.NEW)).apply(self._update(OrderStatus.PENDING_CANCEL))
        order = order.apply(self._update(OrderStatus.FILLED, _fill(1, Side.BUY, "10", "100")))
        assert order.status is OrderStatus.FILLED

    def test_overfill_rejected(self):
        with pytest.raises(InvalidTransitionError, match="exceeds"):
            self._order("5").apply(self._update(OrderStatus.FILLED, _fill(1, Side.BUY, "6", "100")))

    def test_fill_for_wrong_side_rejected(self):
        with pytest.raises(InvalidTransitionError, match="does not match"):
            self._order().apply(self._update(OrderStatus.FILLED, _fill(1, Side.SELL, "10", "100")))

    def test_terminal_statuses(self):
        assert {s for s in OrderStatus if s.is_terminal} == {
            OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED}


class TestLedger:
    def test_round_trip_realizes_pnl_and_fees(self):
        ledger = Ledger(D("10000"))
        ledger.apply_fill(_fill(1, Side.BUY, "10", "100", fee="1"))
        ledger.apply_fill(_fill(2, Side.SELL, "10", "110", fee="1"))
        assert ledger.realized_pnl == D("100")
        assert ledger.fees == D("2")
        assert ledger.cash == D("10098")
        assert ledger.positions == {}

    def test_average_cost_and_partial_close(self):
        ledger = Ledger(D("10000"))
        ledger.apply_fill(_fill(1, Side.BUY, "10", "100"))
        ledger.apply_fill(_fill(2, Side.BUY, "10", "120"))
        assert ledger.position("SPY").avg_price == D("110")
        ledger.apply_fill(_fill(3, Side.SELL, "5", "130"))
        assert ledger.realized_pnl == D("100")
        assert ledger.position("SPY").qty == D("15")
        assert ledger.position("SPY").avg_price == D("110")

    def test_flip_from_long_to_short(self):
        ledger = Ledger(D("10000"))
        ledger.apply_fill(_fill(1, Side.BUY, "10", "100"))
        ledger.apply_fill(_fill(2, Side.SELL, "15", "90"))
        assert ledger.realized_pnl == D("-100")
        position = ledger.position("SPY")
        assert position.qty == D("-5") and position.avg_price == D("90")
        assert ledger.unrealized_pnl({"SPY": D("80")}) == D("50")

    def test_fee_taken_from_the_asset_received(self):
        ledger = Ledger(D("100000"))
        ledger.apply_fill(Fill("f1", "o1", "BTC/USD", Side.BUY, D("1"), D("50000"), D("0"), T0,
                               fee_qty=D("0.0025")))
        position = ledger.position("BTC/USD")
        assert position.qty == D("0.9975") and position.avg_price == D("50000")
        assert ledger.cash == D("50000") and ledger.fees == D("125")
        assert ledger.equity({"BTC/USD": D("50000")}) == D("99875")

    @pytest.mark.parametrize("kw", [{"fee_qty": D("1")}, {"fee_qty": D("-0.1")},
                                    {"side": Side.SELL, "fee_qty": D("0.1")}])
    def test_invalid_asset_fees(self, kw):
        values = dict(fill_id="f", client_order_id="o", symbol="BTC/USD", side=Side.BUY, qty=D("1"),
                      price=D("1"), fee=D("0"), timestamp=T0)
        values.update(kw)
        with pytest.raises(ValueError):
            Fill(**values)

    def test_duplicate_fill_is_not_booked_twice(self):
        ledger = Ledger(D("1000"))
        fill = _fill(1, Side.BUY, "1", "100")
        assert ledger.apply_fill(fill) is True
        assert ledger.apply_fill(fill) is False
        assert ledger.position("SPY").qty == D("1")

    def test_equity_needs_a_price_for_every_position(self):
        ledger = Ledger(D("1000"))
        ledger.apply_fill(_fill(1, Side.BUY, "1", "100"))
        with pytest.raises(KeyError):
            ledger.equity({})

    @given(st.lists(st.tuples(st.sampled_from([Side.BUY, Side.SELL]),
                              st.integers(min_value=1, max_value=50),
                              st.integers(min_value=1, max_value=500),
                              st.integers(min_value=0, max_value=5)),
                    min_size=1, max_size=40),
           st.integers(min_value=1, max_value=500))
    def test_accounting_identity(self, trades, mark):
        """equity == starting cash + realized + unrealized - fees, for any fill sequence."""
        ledger = Ledger(D("100000"))
        net_qty = D("0")
        for n, (side, qty, price, fee) in enumerate(trades):
            ledger.apply_fill(_fill(n, side, str(qty), str(price), fee=str(fee)))
            net_qty += D(qty) * side.sign
        prices = {"SPY": D(mark)}
        assert ledger.position("SPY").qty == net_qty
        with localcontext(LEDGER_CONTEXT):  # the check itself must not round
            expected = ledger.starting_cash + ledger.realized_pnl + ledger.unrealized_pnl(prices) - ledger.fees
        assert ledger.equity(prices) == expected


class TestLedgerFractional:
    @given(st.lists(st.tuples(st.sampled_from([Side.BUY, Side.SELL]),
                              st.decimals(min_value=D("0.000000001"), max_value=D("5"), places=9),
                              st.decimals(min_value=D("0.00000001"), max_value=D("150000"), places=8),
                              st.decimals(min_value=D("0"), max_value=D("10"), places=6)),
                    min_size=1, max_size=40),
           st.decimals(min_value=D("0.00000001"), max_value=D("150000"), places=8))
    def test_accounting_identity_with_crypto_sized_decimals(self, trades, mark):
        ledger = Ledger(D("100000"))
        for n, (side, qty, price, fee) in enumerate(trades):
            fee_qty = (qty * D("0.0025")).quantize(D("1e-9")) if side is Side.BUY else D("0")
            ledger.apply_fill(Fill(f"f{n}", "o1", "BTC/USD", side, qty, price, fee, T0 + timedelta(minutes=n),
                                   fee_qty=fee_qty if fee_qty < qty else D("0")))
        prices = {"BTC/USD": mark}
        with localcontext(LEDGER_CONTEXT):
            expected = ledger.starting_cash + ledger.realized_pnl + ledger.unrealized_pnl(prices) - ledger.fees
        assert ledger.equity(prices) == expected


class TestLedgerAdjustments:
    @given(st.lists(st.one_of(
        st.tuples(st.just("fill"), st.sampled_from([Side.BUY, Side.SELL]),
                  st.integers(min_value=1, max_value=50), st.integers(min_value=1, max_value=500)),
        st.tuples(st.just("qty"), st.integers(min_value=-60, max_value=60), st.integers(min_value=1, max_value=500),
                  st.just(0)),
        st.tuples(st.just("cash"), st.integers(min_value=-5000, max_value=5000), st.just(0), st.just(0)),
    ), min_size=1, max_size=40), st.integers(min_value=1, max_value=500))
    def test_identity_with_adjustments(self, steps, mark):
        """equity == start + realized + unrealized - fees + adjustments, for fills mixed with adjustments."""
        ledger = Ledger(D("100000"))
        for n, (kind, a, b, _) in enumerate(steps):
            if kind == "fill":
                ledger.apply_fill(_fill(n, a, str(b), "100"))
            elif kind == "qty":
                ledger.adjust_position("SPY", D(a), D(b))
            else:
                ledger.adjust_cash(D(a))
        prices = {"SPY": D(mark)}
        with localcontext(LEDGER_CONTEXT):
            expected = (ledger.starting_cash + ledger.realized_pnl + ledger.unrealized_pnl(prices)
                        - ledger.fees + ledger.adjustments)
        assert ledger.equity(prices) == expected

    def test_removed_units_leave_at_average_cost(self):
        ledger = Ledger(D("10000"))
        ledger.apply_fill(_fill(1, Side.BUY, "10", "100"))
        ledger.adjust_position("SPY", D("-4"), D("150"))
        position = ledger.position("SPY")
        assert position.qty == D("6") and position.avg_price == D("100")
        assert ledger.adjustments == D("-400") and ledger.realized_pnl == 0


class TestSimulatedClock:
    def test_only_moves_forward(self):
        clock = SimulatedClock(T0)
        clock.advance_to(T0 + timedelta(hours=1))
        assert clock.now() == T0 + timedelta(hours=1)
        with pytest.raises(ValueError):
            clock.advance_to(T0)

    def test_rejects_naive_times(self):
        with pytest.raises(ValueError):
            SimulatedClock(datetime(2026, 1, 1))


class TestCalendar:
    nyse = TradingCalendar(AssetClass.EQUITY)
    crypto = TradingCalendar(AssetClass.CRYPTO)

    def test_holidays_and_weekends_are_skipped(self):
        assert self.nyse.sessions(date(2026, 12, 31), date(2027, 1, 5)) == [
            date(2026, 12, 31), date(2027, 1, 4), date(2027, 1, 5)]
        assert self.nyse.is_session(date(2026, 11, 26)) is False  # Thanksgiving

    def test_next_and_previous_sessions(self):
        assert self.nyse.next_session(date(2026, 11, 25)) == date(2026, 11, 27)
        assert self.nyse.previous_session(date(2026, 11, 27)) == date(2026, 11, 25)
        assert self.nyse.next_session(date(2026, 11, 28)) == date(2026, 11, 30)

    def test_early_close(self):
        assert self.nyse.session_close(date(2026, 11, 27)) == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)
        assert self.nyse.session_close(date(2026, 11, 25)) == datetime(2026, 11, 25, 21, 0, tzinfo=UTC)

    def test_crypto_trades_every_day(self):
        assert len(self.crypto.sessions(date(2026, 11, 26), date(2026, 11, 29))) == 4
        assert self.crypto.session_close(date(2026, 11, 26)) == datetime(2026, 11, 27, tzinfo=UTC)
