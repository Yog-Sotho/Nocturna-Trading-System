"""
The monthly long-history checks of E3 and E2 (docs/STRATEGY_SHORTLIST.md, "Fixed rules:
commodities" and "Fixed rules: international").

Stocks, bonds and T-bills are the round-2 daily proxies (src/trading/validation/history.py),
compounded by calendar month, as in gold_long_history.py. Portfolios are rebalanced to fixed
weights every month. For each half it prints the Sharpe ratio over T-bills, the maximum drawdown
and the return a year of the 60/40 and of each variant.

    pip install openpyxl   # not a dependency of the app
    python tools/research/sleeve_long_history.py commodities <Commodities-for-the-Long-Run-Index-Level-Data-Monthly.xlsx>
    python tools/research/sleeve_long_history.py international <Developed_ex_US_3_Factors_CSV.zip>
"""

import csv
import hashlib
import io
import sys
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gold_long_history import monthly, stats  # noqa: E402
from src.trading.validation import history  # noqa: E402

Month = tuple[int, int]

#: The files the rules were fixed on (24 October 2026)
SHA256_PREFIX = {"commodities": "eeb89c65", "international": "83eda24f"}
FEE = {"commodities": 0.0085, "international": 0.0003}
HALVES = {
    "commodities": ((date(1975, 1, 1), date(2000, 12, 31)), (date(2001, 1, 1), date(2025, 5, 31))),
    "international": ((date(1990, 7, 1), date(2007, 12, 31)), (date(2008, 1, 1), date(2026, 8, 31))),
}
#: (stocks at home, the sleeve, bonds) for each variant, as in candidates.py
MIXES = {
    "commodities": {f"commodities {c:.0%}": (0.6 - c / 2, c, 0.4 - c / 2) for c in (0.05, 0.10, 0.15)},
    "international": {f"abroad {name}": (0.6 * (1 - f), 0.6 * f, 0.4)
                      for name, f in (("1/3", 1 / 3), ("1/2", 1 / 2), ("2/3", 2 / 3))},
}


def commodity_excess(path: Path) -> dict[Month, float]:
    """AQR's equal-weight commodity futures portfolio: its excess return over T-bills, by month."""
    rows = openpyxl.load_workbook(path, read_only=True, data_only=True)["Commodities for the Long Run"].iter_rows(
        values_only=True)
    return {(r[0].year, r[0].month): float(r[1]) for r in rows if isinstance(r[0], datetime)}


def developed_ex_us(path: Path) -> dict[Month, float]:
    """French's developed ex-US market return in dollars (Mkt-RF plus RF), by month."""
    with zipfile.ZipFile(path) as archive:
        text = archive.read(archive.namelist()[0]).decode("latin-1")
    out = {}
    for row in csv.reader(io.StringIO(text)):
        if not row or not row[0].strip().isdigit() or len(row[0].strip()) != 6:
            if out:
                break  # the annual table follows the monthly one
            continue
        key = row[0].strip()
        out[(int(key[:4]), int(key[4:]))] = (float(row[1]) + float(row[4])) / 100
    return out


def main(kind: str, path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith(SHA256_PREFIX[kind]):
        raise SystemExit(f"{path} is not the file the rules were fixed on (SHA-256 {digest})")
    _, bars, sources = history.load(datetime(1974, 11, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC))
    for source in sources:
        print(f"Data: {source.url} ({source.first} to {source.last}, sha256 {source.sha256})")
    print(f"Data: {path.name} (sha256 {digest})")
    stocks, bonds, bills = (monthly(bars[s]) for s in ("SPY", "IEF", "BIL"))
    if kind == "commodities":
        excess = commodity_excess(path)
        sleeve = {m: excess[m] + bills[m] - FEE[kind] / 12 for m in excess if m in bills}
    else:
        sleeve = {m: r - FEE[kind] / 12 for m, r in developed_ex_us(path).items()}
    months = sorted(m for m in stocks if m in bonds and m in bills and m in sleeve)
    mixes = {"60/40": (0.6, 0.0, 0.4)} | MIXES[kind]
    for first, last in HALVES[kind]:
        span = [m for m in months if (first.year, first.month) <= m <= (last.year, last.month)]
        print(f"\n{span[0][0]}-{span[0][1]:02d} to {span[-1][0]}-{span[-1][1]:02d} ({len(span)} months)")
        print(f"{'':18} {'Sharpe':>7} {'Max DD':>8} {'A year':>7}")
        cash = [bills[m] for m in span]
        for name, (home, other, bond) in mixes.items():
            returns = [home * stocks[m] + other * sleeve[m] + bond * bonds[m] for m in span]
            sharpe, drawdown, growth = stats(returns, cash)
            print(f"{name:18} {sharpe:7.2f} {drawdown:8.1%} {growth:7.2%}")
        sharpe, drawdown, growth = stats([sleeve[m] for m in span], cash)
        print(f"{kind + ' alone':18} {sharpe:7.2f} {drawdown:8.1%} {growth:7.2%}")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
