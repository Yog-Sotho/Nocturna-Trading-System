"""
The real app and dashboard with a simulated engine, for trying the dashboard
and for tools/dashboard/e2e.cjs. The engine trades the saved trading profile
(the 60/40 until another mix is saved, idle cash in the T-bill ETF BIL) on
synthetic prices for every symbol a strategy can trade. The simulated clock
advances one trading day every three seconds while the engine runs, so orders
fill and the equity curve grows. The strategy lab runs in this process, on the
same synthetic prices. No broker account is used.

    python tools/dashboard/serve_sim.py . /tmp/dashboard.db 5055

Sign in as ops / Harness-Passw0rd! (created on first start).
"""
import os
import sys
import threading
import time
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

repo, db_path, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ENGINE_AUTOSTART"):
    os.environ.pop(name, None)
os.environ.update(FLASK_ENV="development", TRADING_MODE="PAPER", DATABASE_URL=f"sqlite:///{db_path}",
                  FLASK_SECRET_KEY="dashboard-harness-secret-key-0123456789abcdef", LOG_TO_FILE="false",
                  SECURE_COOKIES="false")
sys.path.insert(0, repo)

from pydantic import SecretStr  # noqa: E402
from src.main import create_app  # noqa: E402
from src.models.user import User, create_admin_user  # noqa: E402
from src.tests.trading_fixtures import daily_bars, ny  # noqa: E402
from src.trading import lab  # noqa: E402
from src.trading.broker import SimulatedBroker  # noqa: E402
from src.trading.clock import SimulatedClock  # noqa: E402
from src.trading.data import InMemoryBarSource  # noqa: E402
from src.trading.domain import AssetClass, Instrument  # noqa: E402
from src.trading.lineup import CASH_SLEEVE  # noqa: E402
from src.trading.persistence import EngineRecorder  # noqa: E402
from src.trading.profile import allocations, effective_settings, warm_up_days  # noqa: E402
from src.trading.runtime import TradingRuntime, held_over, risk_limits, sizing_policy  # noqa: E402
from src.trading.service import EngineService  # noqa: E402
from src.trading.validation import history, market  # noqa: E402
from src.trading.validation.candidates import CANDIDATES  # noqa: E402

app = create_app()
clock = SimulatedClock(ny(2026, 3, 2, 12))

# Every symbol a strategy trades, with made-up prices: equities from 2010, crypto from 2018
EQUITY, CRYPTO = AssetClass.EQUITY, AssetClass.CRYPTO
UNIVERSE = sorted({s for spec in CANDIDATES.values() for s in spec.symbols} | {(CASH_SLEEVE, EQUITY)})
INSTRUMENTS = {s: Instrument(s, a, qty_increment=Decimal("1") if a is EQUITY else Decimal("0.0001"),
                             min_qty=Decimal("1") if a is EQUITY else Decimal("0.0001"), shortable=a is EQUITY)
               for s, a in UNIVERSE}
BARS = {}
for seed, (symbol, asset_class) in enumerate(UNIVERSE, start=1):
    tbills = symbol == CASH_SLEEVE
    BARS[symbol] = daily_bars(INSTRUMENTS[symbol], date(2010, 1, 4) if asset_class is EQUITY else date(2018, 1, 1),
                              date(2026, 12, 31), 91.5 if tbills else 100.0 + 10 * seed, seed=seed,
                              drift=0.00015 if tbills else 0.0003, vol=0.0002 if tbills else 0.012)
broker = SimulatedBroker(clock, INSTRUMENTS, Decimal("100000"))


def factory(flask_app, settings, profile):
    effective = effective_settings(settings, profile)
    close_out = held_over(list(broker.positions()), profile)
    recorder = EngineRecorder(flask_app)
    return EngineService(clock=clock, source=InMemoryBarSource(BARS, clock), broker=broker,
                         allocations=allocations(profile, INSTRUMENTS), cash_sleeve=INSTRUMENTS[CASH_SLEEVE],
                         close_out=[INSTRUMENTS[s] for s, _ in close_out], risk_limits=risk_limits(effective),
                         sizing=sizing_policy(effective), warm_up=timedelta(days=warm_up_days(profile)),
                         risk_state=recorder.load_risk_state, listeners=[recorder],
                         snapshot_recorder=recorder.record_snapshot)


def sim_market(settings, wanted, start, end):
    """The lab's "Alpaca" bars: the same synthetic prices."""
    return ({s: INSTRUMENTS[s] for s, _ in wanted},
            {s: [b for b in BARS[s] if start <= b.start < end] for s, _ in wanted})


LONG = {s: daily_bars(INSTRUMENTS[s], date(1970, 1, 2), date(2026, 8, 31), 50.0 + 5 * i, seed=40 + i,
                      vol=0.0002 if s == CASH_SLEEVE else 0.01) for i, s in enumerate(history.SYMBOLS)}


def sim_history(start, end, fetch=None):
    """The lab's long history: synthetic too, and nothing is downloaded."""
    source = history.Source("https://example.invalid/synthetic", "0" * 64, date(1970, 1, 2), date(2026, 8, 31))
    return ({s: INSTRUMENTS[s] for s in history.SYMBOLS},
            {s: [b for b in bars if start <= b.start < end] for s, bars in LONG.items()}, [source])


market.load = sim_market
history.load = sim_history


class InProcessLab:
    """The lab's runs on a thread of this process instead of a process of their own."""

    def __init__(self):
        self._thread = None
        self._lock = threading.Lock()

    def launch(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=lab.drain, args=(app, settings, app.extensions["alerts"]),
                                            daemon=True)
            self._thread.start()


settings = app.extensions["settings"].model_copy(
    update={"alpaca_api_key": SecretStr("sim"), "alpaca_secret_key": SecretStr("sim")})
runtime = TradingRuntime(app, settings, factory=factory)
app.extensions["trading"] = runtime
app.extensions["lab"] = InProcessLab()
with app.app_context():
    if User.query.filter_by(username="ops").first() is None:
        create_admin_user("ops", "ops@nocturna.example", "Harness-Passw0rd!")


def advance():
    while True:
        time.sleep(3)
        service = runtime.service
        if service is not None and service.running:
            clock.advance_to(clock.now() + timedelta(days=1))
            service._queue.put(None)  # wake the engine loop


threading.Thread(target=advance, daemon=True).start()
print(f"Simulated dashboard on http://127.0.0.1:{port} (started {datetime.now(UTC):%H:%M:%S} UTC)", flush=True)
app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)
