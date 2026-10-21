"""
NOCTURNA Trading System - Account ledger.

Cash, positions and profit and loss derived purely from fills, so backtest,
paper and live accounting are the same code. Positions use average-cost
accounting; a fill that crosses zero closes the old position and opens the new
one at the fill price. The identity

    equity == starting cash + realized P&L + unrealized P&L - fees

holds exactly for any sequence of fills. Differences found when reconciling
with the broker (deposits, withdrawals, fees charged later, trades made outside
the engine) are booked as adjustments, which extends it to

    equity == starting cash + realized + unrealized - fees + adjustments
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from functools import wraps

from .domain import Fill

ZERO = Decimal("0")

#: Ledger arithmetic runs at this precision, and the one division (the share of
#: cost removed by a partial close) is rounded to COST_SCALE. Every other
#: operation is then an exact addition or multiplication of bounded decimals.
LEDGER_CONTEXT = Context(prec=60, rounding=ROUND_HALF_EVEN)
COST_SCALE = Decimal("1e-12")

def _exact[F: Callable](method: F) -> F:
    @wraps(method)
    def wrapper(*args, **kwargs):
        with localcontext(LEDGER_CONTEXT):
            return method(*args, **kwargs)
    return wrapper  # type: ignore[return-value]


@dataclass(frozen=True)
class Position:
    """
    Signed quantity (negative is short) and its signed cost basis.

    The cost basis, not an average price, is stored: it changes by exact
    amounts, so cash + positions always reconcile to realized and unrealized
    P&L without rounding drift.
    """

    symbol: str
    qty: Decimal = ZERO
    cost: Decimal = ZERO

    @property
    @_exact
    def avg_price(self) -> Decimal:
        return self.cost / self.qty if self.qty else ZERO

    @_exact
    def market_value(self, price: Decimal) -> Decimal:
        return self.qty * price

    @_exact
    def unrealized_pnl(self, price: Decimal) -> Decimal:
        return self.qty * price - self.cost


@dataclass(frozen=True)
class LedgerSnapshot:
    cash: Decimal
    positions: dict[str, Position]
    realized_pnl: Decimal
    fees: Decimal
    adjustments: Decimal


class Ledger:
    """
    Applies fills and reports account state.

    Applying the same fill twice (same ``fill_id``) is a no-op, so broker
    replays and restarts cannot double-count executions.
    """

    def __init__(self, starting_cash: Decimal) -> None:
        if starting_cash < 0:
            raise ValueError("starting_cash must not be negative")
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.realized_pnl = ZERO  # before fees
        self.fees = ZERO
        self.adjustments = ZERO
        self._positions: dict[str, Position] = {}
        self._fill_ids: set[str] = set()

    @classmethod
    def from_account(cls, cash: Decimal, positions: Mapping[str, tuple[Decimal, Decimal]]) -> Ledger:
        """
        Start from a broker account: cash plus (qty, cost basis) per symbol. The
        starting capital is cash plus the cost of the positions.
        """
        with localcontext(LEDGER_CONTEXT):
            ledger = cls(cash + sum((cost for _, cost in positions.values()), ZERO))
        ledger.cash = cash
        for symbol, (qty, cost) in positions.items():
            if qty:
                ledger._positions[symbol] = Position(symbol, qty, cost)
        return ledger

    def has_fill(self, fill_id: str) -> bool:
        return fill_id in self._fill_ids

    @_exact
    def adjust_cash(self, amount: Decimal) -> None:
        """Book a cash movement that did not come from a fill."""
        self.cash += amount
        self.adjustments += amount

    @_exact
    def adjust_position(self, symbol: str, qty_delta: Decimal, price: Decimal) -> None:
        """
        Book a quantity change that did not come from a fill. Removed units leave
        at their average cost; added units arrive at ``price``.
        """
        if not qty_delta:
            return
        current = self.position(symbol)
        if current.qty and (current.qty > 0) != (qty_delta > 0):
            # Reduce the existing position first, at its average cost
            reduced = min(abs(qty_delta), abs(current.qty))
            if reduced == abs(current.qty):
                removed = current.cost
            else:
                removed = (current.cost * reduced / abs(current.qty)).quantize(COST_SCALE)
            step = reduced if qty_delta > 0 else -reduced
            current = Position(symbol, current.qty + step, current.cost - removed)
            self.adjustments -= removed
            qty_delta -= step
        if qty_delta:
            # New units (or the part past zero) arrive at the given price
            current = Position(symbol, current.qty + qty_delta, current.cost + qty_delta * price)
            self.adjustments += qty_delta * price
        if current.qty == 0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = current

    @property
    def positions(self) -> dict[str, Position]:
        """Open positions (non-zero quantity), keyed by symbol."""
        return dict(self._positions)

    def position(self, symbol: str) -> Position:
        return self._positions.get(symbol, Position(symbol))

    @_exact
    def apply_fill(self, fill: Fill) -> bool:
        """
        Book one execution.

        Returns:
            True if the fill was booked, False if it had already been applied.
        """
        if fill.fill_id in self._fill_ids:
            return False
        self._fill_ids.add(fill.fill_id)

        # A fee taken from the asset received is the same as receiving fewer units
        # and paying their value as a fee: cash still leaves for the full quantity
        fee = fill.fee + fill.fee_qty * fill.price
        signed = fill.signed_qty - fill.fee_qty
        self.cash -= signed * fill.price + fee
        self.fees += fee

        current = self.position(fill.symbol)
        if current.qty == 0 or (current.qty > 0) == (signed > 0):
            # Opening or adding
            updated = Position(fill.symbol, current.qty + signed, current.cost + signed * fill.price)
        else:
            closed = min(abs(current.qty), abs(signed))
            # Share of the cost basis that leaves with the closed quantity; exact on a full close
            if closed == abs(current.qty):
                removed = current.cost
            else:
                removed = (current.cost * closed / abs(current.qty)).quantize(COST_SCALE)
            direction = 1 if current.qty > 0 else -1
            self.realized_pnl += fill.price * closed * direction - removed
            remaining = current.qty + signed
            if remaining == 0 or (remaining > 0) == (current.qty > 0):
                updated = Position(fill.symbol, remaining, current.cost - removed)
            else:
                # Flipped: the excess opens a new position at the fill price
                updated = Position(fill.symbol, remaining, remaining * fill.price)

        if updated.qty == 0:
            self._positions.pop(fill.symbol, None)
        else:
            self._positions[fill.symbol] = updated
        return True

    @_exact
    def market_value(self, prices: Mapping[str, Decimal]) -> Decimal:
        """Signed value of all positions. Raises KeyError if a held symbol has no price."""
        return sum((p.market_value(prices[s]) for s, p in self._positions.items()), ZERO)

    @_exact
    def equity(self, prices: Mapping[str, Decimal]) -> Decimal:
        return self.cash + self.market_value(prices)

    @_exact
    def gross_exposure(self, prices: Mapping[str, Decimal]) -> Decimal:
        return sum((abs(p.market_value(prices[s])) for s, p in self._positions.items()), ZERO)

    @_exact
    def unrealized_pnl(self, prices: Mapping[str, Decimal]) -> Decimal:
        return sum((p.unrealized_pnl(prices[s]) for s, p in self._positions.items()), ZERO)

    def snapshot(self) -> LedgerSnapshot:
        return LedgerSnapshot(self.cash, self.positions, self.realized_pnl, self.fees, self.adjustments)
