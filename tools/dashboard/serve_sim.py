"""
The real app and dashboard with a simulated engine, for trying the dashboard
and for tools/dashboard/e2e.cjs. The engine runs the live lineup (the 60/40,
idle cash in the T-bill ETF BIL) on synthetic prices. The simulated clock
advances one trading day every three seconds while the engine runs, so orders
fill and the equity curve grows. No broker account is used.

    python tools/dashboard/serve_sim.py . /tmp/dashboard.db 5055

Sign in as ops / Harness-Passw0rd! (created on first start).
"""
import os
import sys
import threading
import time
from datetime import date, timedelta
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
from src.tests.trading_fixtures import BIL, IEF, SPY, daily_bars, ny  # noqa: E402
from src.trading.broker import SimulatedBroker  # noqa: E402
from src.trading.clock import SimulatedClock  # noqa: E402
from src.trading.data import InMemoryBarSource  # noqa: E402
from src.trading.lineup import CASH_SLEEVE, build_lineup  # noqa: E402
from src.trading.persistence import EngineRecorder  # noqa: E402
from src.trading.runtime import TradingRuntime, risk_limits, sizing_policy  # noqa: E402
from src.trading.service import EngineService  # noqa: E402

app = create_app()
clock = SimulatedClock(ny(2026, 3, 2, 12))


def factory(flask_app, settings):
    instruments = {"SPY": SPY, "IEF": IEF, CASH_SLEEVE: BIL}
    history = {"SPY": daily_bars(SPY, date(2025, 1, 2), date(2026, 12, 31), 500.0, seed=1),
               "IEF": daily_bars(IEF, date(2025, 1, 2), date(2026, 12, 31), 95.0, seed=2),
               CASH_SLEEVE: daily_bars(BIL, date(2025, 1, 2), date(2026, 12, 31), 91.5, seed=5, drift=0.00015,
                                       vol=0.0002)}
    recorder = EngineRecorder(flask_app)
    return EngineService(clock=clock, source=InMemoryBarSource(history, clock),
                         broker=SimulatedBroker(clock, instruments, Decimal("100000")),
                         allocations=build_lineup(instruments), cash_sleeve=BIL, risk_limits=risk_limits(settings),
                         sizing=sizing_policy(settings), risk_state=recorder.load_risk_state, listeners=[recorder],
                         snapshot_recorder=recorder.record_snapshot)


settings = app.extensions["settings"].model_copy(
    update={"alpaca_api_key": SecretStr("sim"), "alpaca_secret_key": SecretStr("sim")})
runtime = TradingRuntime(app, settings, factory=factory)
app.extensions["trading"] = runtime
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
app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)
