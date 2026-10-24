"""
E1's monthly long-history check (docs/STRATEGY_SHORTLIST.md, "Fixed rules: gold").

Stocks, bonds and T-bills are the round-2 daily proxies (src/trading/validation/history.py),
compounded by calendar month. Gold is the change in the World Bank's monthly average price
(the Pink Sheet workbook), less GLD's 0.40% a year. Portfolios are rebalanced to fixed weights
every month. For each half (January 1975 to December 2000, January 2001 to August 2026) it
prints the Sharpe ratio over T-bills and the maximum drawdown of the 60/40 and of each gold
weight.

    pip install openpyxl   # not a dependency of the app
    python tools/research/gold_long_history.py <CMO-Historical-Data-Monthly.xlsx>
"""

import hashlib
import math
import sys
from datetime import UTC, date, datetime
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.trading.validation import history  # noqa: E402

#: The release the rules were fixed on (2 October 2026)
WORKBOOK_SHA256_PREFIX = "ea1c3508"
GLD_FEE = 0.0040
HALVES = ((date(1975, 1, 1), date(2000, 12, 31)), (date(2001, 1, 1), date(2026, 8, 31)))
GOLD_WEIGHTS = (0.10, 0.15, 0.20)


def previous(month: tuple[int, int]) -> tuple[int, int]:
    year, number = month
    return (year, number - 1) if number > 1 else (year - 1, 12)


def gold_prices(path: Path) -> dict[tuple[int, int], float]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith(WORKBOOK_SHA256_PREFIX):
        raise SystemExit(f"{path} is not the release the rules were fixed on (SHA-256 {digest})")
    rows = list(openpyxl.load_workbook(path, read_only=True, data_only=True)["Monthly Prices"].iter_rows(
        values_only=True))
    column = rows[4].index("Gold")
    return {(int(r[0][:4]), int(r[0][5:])): float(r[column]) for r in rows[6:]
            if isinstance(r[0], str) and isinstance(r[column], (int, float))}


def monthly(bars: list) -> dict[tuple[int, int], float]:
    """The return of each calendar month, from the last close of the month before."""
    last: dict[tuple[int, int], float] = {}
    for bar in bars:
        day = bar.start.astimezone(UTC).date()
        last[(day.year, day.month)] = bar.close
    months = sorted(last)
    return {m: last[m] / last[p] - 1 for p, m in zip(months, months[1:], strict=False)}


def stats(returns: list[float], bills: list[float]) -> tuple[float, float, float]:
    excess = [r - b for r, b in zip(returns, bills, strict=True)]
    mean = sum(excess) / len(excess)
    sd = math.sqrt(sum((x - mean) ** 2 for x in excess) / (len(excess) - 1))
    wealth, peak, worst = 1.0, 1.0, 0.0
    for r in returns:
        wealth *= 1 + r
        peak = max(peak, wealth)
        worst = min(worst, wealth / peak - 1)
    growth = math.prod(1 + r for r in returns) ** (12 / len(returns)) - 1
    return mean / sd * math.sqrt(12), worst, growth


def main(path: Path) -> None:
    gold = gold_prices(path)
    _, bars, sources = history.load(datetime(1974, 11, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC))
    for source in sources:
        print(f"Data: {source.url} ({source.first} to {source.last}, sha256 {source.sha256})")
    print(f"Data: World Bank Pink Sheet ({path.name}, sha256 {hashlib.sha256(path.read_bytes()).hexdigest()})")
    stocks, bonds, bills = (monthly(bars[s]) for s in ("SPY", "IEF", "BIL"))
    months = sorted(m for m in stocks if m in bonds and m in bills and m in gold and previous(m) in gold)

    def gold_return(m: tuple[int, int]) -> float:
        return gold[m] / gold[previous(m)] - 1 - GLD_FEE / 12

    mixes = {"60/40": (0.6, 0.4, 0.0)} | {f"gold {g:.0%}": (0.6 - g / 2, 0.4 - g / 2, g) for g in GOLD_WEIGHTS}
    for first, last in HALVES:
        span = [m for m in months if (first.year, first.month) <= m <= (last.year, last.month)]
        print(f"\n{span[0][0]}-{span[0][1]:02d} to {span[-1][0]}-{span[-1][1]:02d} ({len(span)} months)")
        print(f"{'':10} {'Sharpe':>7} {'Max DD':>8} {'A year':>7}")
        cash = [bills[m] for m in span]
        for name, (s, b, g) in mixes.items():
            returns = [s * stocks[m] + b * bonds[m] + g * gold_return(m) for m in span]
            sharpe, drawdown, growth = stats(returns, cash)
            print(f"{name:10} {sharpe:7.2f} {drawdown:8.1%} {growth:7.2%}")
        sharpe, drawdown, growth = stats([gold_return(m) for m in span], cash)
        print(f"{'gold only':10} {sharpe:7.2f} {drawdown:8.1%} {growth:7.2%}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
