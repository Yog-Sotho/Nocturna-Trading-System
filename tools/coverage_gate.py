"""
Fail CI when test coverage of the trading core falls below its floor.

pytest's --cov-fail-under guards the total; this guards the modules that
move money or decide risk, where a gap matters most. Reads the JSON report:

    pytest src/tests --cov=src --cov-report=json:coverage.json
    python tools/coverage_gate.py coverage.json
"""

import json
import sys

#: Line coverage each of these must keep, in percent
CORE_FLOOR = 95.0
CORE = (
    "src/trading/domain.py",
    "src/trading/ledger.py",
    "src/trading/engine.py",
    "src/trading/risk.py",
    "src/trading/sizing.py",
    "src/trading/broker.py",
    "src/trading/reconcile.py",
    "src/trading/backtest.py",
    "src/trading/operator.py",
    "src/trading/channel.py",
)


def main(path: str) -> int:
    with open(path) as f:
        files = json.load(f)["files"]
    failures = []
    for name in CORE:
        if name not in files:
            failures.append(f"{name}: not in the coverage report")
            continue
        covered = files[name]["summary"]["percent_covered"]
        mark = "ok " if covered >= CORE_FLOOR else "LOW"
        print(f"{mark} {covered:6.2f}%  {name}")
        if covered < CORE_FLOOR:
            failures.append(f"{name}: {covered:.2f}% < {CORE_FLOOR}%")
    if failures:
        print("\nCore coverage below the floor:\n  " + "\n  ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "coverage.json"))
