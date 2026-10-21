<div align="center">
  <img src="src/assets/banner.png" alt="Nocturna Banner" width="420" style="margin-bottom: 20px;">

# NOCTURNA — Algorithmic Trading Framework

</div>

> [!WARNING]
> **Status: prototype under active rebuild. Paper trading only.**
> Live trading is disabled in code: the app refuses to start unless `TRADING_MODE=PAPER`
> and `ALPACA_BASE_URL` points at the Alpaca paper API. No strategy in this repository has
> passed out-of-sample validation, and none should be trusted with real money.

NOCTURNA is a Python/Flask service for researching, backtesting and paper-trading
rule-based strategies on US equities, ETFs and crypto through [Alpaca](https://alpaca.markets).
Its core is event-driven: the same strategy, ledger and risk code runs in backtests and in paper
trading; only the clock, the market-data source and the broker are swapped. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and the [Roadmap](#roadmap).

## What exists today

| Component | Location | State |
|---|---|---|
| Trading core: domain model, ledger, engine, strategy interface | `src/trading/` | Working; property-tested accounting |
| Risk engine: pre-trade gate, drawdown kill switch, daily loss limit, volatility-targeted sizing with a drawdown ladder | `src/trading/risk.py`, `sizing.py` | Working; limits are configurable, defaults below |
| Backtester (simulated clock and broker, cost model, no look-ahead) | `src/trading/backtest.py` | Working; validation statistics arrive in Phase 4 |
| Paper trading on Alpaca (`alpaca-py`): broker adapter, trade-update stream, polling data feed, reconciliation | `src/trading/alpaca.py`, `service.py` | Working against fakes and Alpaca's real request models; not yet run against a live paper account |
| Strategies | `src/trading/strategies/` | Only a 60/40 SPY/IEF benchmark; the validated lineup arrives in Phase 5 |
| Trading history (orders, fills, equity snapshots) and audit log | `src/models/` | Working |
| REST API with JWT and API-key auth (single operator) | `src/main.py`, `src/routes/` | Working; public registration is disabled |
| Dashboard | `src/static/` | Compiled bundle only (no source); it targets the old API and does not send auth headers. Rebuilt in Phase 7 |

### Known limitations

- No strategy has an edge yet: the engine runs a benchmark portfolio until strategies pass
  the Phase 4 validation gates.
- The engine runs inside the web worker; Phase 6 moves it into its own service.

## Quick start (local, paper only)

Requires Python 3.13+.

```bash
git clone https://github.com/Yog-Sotho/Nocturna-Trading-System.git
cd Nocturna-Trading-System
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set FLASK_SECRET_KEY, JWT_SECRET_KEY and API_KEY_PEPPER
#   python -c "import secrets; print(secrets.token_hex(32))"
# For paper trading set ALPACA_API_KEY and ALPACA_SECRET_KEY (paper account keys).
# Keep TRADING_MODE=PAPER and the paper ALPACA_BASE_URL.

# Create the operator account (prompts for a password). The database schema is
# created and upgraded automatically whenever the app starts.
flask --app src.main create-admin --email you@example.com

# Development server
python main.py
```

Log in to get a token, then pass it as `Authorization: Bearer <token>`:

```bash
curl -s -X POST http://localhost:5000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username": "admin", "password": "<your password>"}'
```

Start the engine with `POST /api/trading/control/start` (admin), or set `ENGINE_AUTOSTART=true`.
On start it loads the paper account from Alpaca (cash, positions, working orders), warms up
history, and acts on the latest closed bars; after that it wakes each time a new bar becomes
final.

In the Docker image the production entry point is
`gunicorn -c gunicorn.conf.py 'src.main:create_app()'`. It runs a single, never-recycled
worker because the engine lives in that process.

## Risk controls

Every order, from a strategy or an operator, passes one risk gate before it reaches the
broker, and the book is sized by the same rules in backtests and in paper trading
(details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#risk)). Defaults, all fractions of
account equity and configurable through `RISK_*` / `SIZING_*` variables:

| Control | Default |
|---|---|
| Drawdown kill switch: halt and close every position | 10% below the equity high-water mark |
| Drawdown ladder: exposure shrinks linearly from the start to 25% at the kill limit | from 5% drawdown |
| Daily loss limit: no new risk for the rest of the New York day | 3% |
| Portfolio volatility target (EWMA, lambda 0.94; scales down only, never levers up) | 8% a year |
| Largest position / crypto in total / gross exposure (no leverage) | 60% / 15% / 100% |
| Single order (fat finger) | 60% of equity, and `RISK_MAX_ORDER_NOTIONAL` dollars if set |
| Limit price collar around the last close | 5% |
| The same symbol, side and quantity again | refused for 5 minutes |
| Orders / turnover per New York day | 50 / 2x equity |
| Market data: newest bar older than / one-bar move larger than | equities 5 days / 25%, crypto 2 days / 50% |
| Rebalance band: positions are adjusted only when this far from target | 2% |

Orders that only reduce a position skip the daily breakers and data checks, so the book can
always be reduced. The gate's state (high-water mark, halt, daily counters) is stored in the
database and restored on restart, so a halt survives a redeploy. After a drawdown halt,
`POST /api/trading/control/resume` restarts the high-water mark at the current equity.
Strategies with protective stops size positions with `stop_risk_weight` (0.5% of equity at
risk per trade by default).

## API

All endpoints except `/health`, `/api/version`, `/api/trading/health` and the login and
refresh endpoints require a bearer token or an API key (`X-API-Key` header). Roles and account
status are read from the database on every request, so changes apply immediately. Endpoints marked
*admin* also require the admin role; order and control endpoints require the trader role and
an account whose trading is not disabled.

**System:** `GET /health`, `GET /api/version`

Requests are rate limited per client IP: 300 per minute by default, with tighter limits on
login (10/min), token refresh (30/min), password changes (5/min), user creation (5/min) and
API key creation (10/hour); see `ROUTE_RATE_LIMITS` in `src/middleware/security.py`.
`/health` and the emergency stop are never limited. Five failed logins from one address
within 15 minutes block logins from it until the failures are 15 minutes old. Request bodies
are limited to 1 MB.

**Auth and users** (`/api`)

| Method | Path | Notes |
|---|---|---|
| POST | `/auth/login` | returns an access token (1 h) and a refresh token (7 days) |
| POST | `/auth/refresh` | body `{"refresh_token": "..."}`; returns a new pair. Refresh tokens are single use: reusing one revokes all of the user's sessions |
| POST | `/auth/logout` | revokes the bearer token, and the refresh token if sent in the body |
| POST | `/auth/logout-all` | revokes every token of the caller (API keys are separate) |
| GET | `/auth/me` | |
| GET, POST | `/users` | admin; `POST` creates a user |
| GET, PUT | `/users/<id>` | admin; `roles` may contain `trader` and `reader`, admin rights come only from `is_admin` |
| PUT | `/users/<id>/password` | own account, or admin; revokes all of that user's tokens |
| GET, POST | `/api-keys` | own keys; `POST` body `{"name": "...", "permissions": ["read", "trade"], "expires_in_days": 90}` returns the key (`ntr_<id>_<secret>`) once. Only the id and a salted, peppered scrypt hash of the secret are stored. Keys never carry admin rights, and `trade` needs an owner with the trader role |
| DELETE | `/api-keys/<id>` | revokes one of your keys |
| GET | `/audit-logs` | admin; logins, logouts, password, user and API key changes, operator orders, engine start/stop/halt, emergency stops, risk rejections, reconciliations and engine errors. Filters: `event_type`, `user_id`, `page`, `per_page` |

**Trading** (`/api/trading`)

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | public; engine state only |
| GET | `/status` | engine state, equity, cash, P&L, fees, halt, last reconciliation; `risk`: drawdown, high-water mark, cool-down, today's orders and turnover, predicted volatility and the volatility and drawdown scales |
| GET | `/positions`, `/targets` | live, from the running engine (409 when it is not running) |
| GET | `/orders` | working orders and order history (`?limit=`, newest first) |
| GET | `/fills`, `/equity-curve` | trading history (`?limit=`) |
| POST | `/orders` | trader; body `{"symbol": "SPY", "side": "buy", "qty": "10", "order_type": "market", "time_in_force": "day", "limit_price": null, "stop_price": null}`; passes the same risk gate as strategy orders (422 with the reason when refused) |
| POST | `/orders/<client_order_id>/cancel` | trader |
| POST | `/control/start`, `/control/stop`, `/control/resume` | admin; resume lifts a halt (after a drawdown halt it also restarts the high-water mark) |
| POST | `/emergency-stop` | admin; body `{"reason": "...", "close_all_positions": true}`. Halts the engine and cancels its orders; with `close_all_positions` also closes every position. Works when the engine is not running by asking Alpaca directly |

## Configuration

All settings come from environment variables; `.env.example` lists them with comments.
They are validated at startup (`src/config.py`); with `FLASK_ENV=production` the app refuses
to start and lists every problem if the configuration is unsafe. The ones that matter most:

| Variable | Purpose |
|---|---|
| `FLASK_SECRET_KEY`, `JWT_SECRET_KEY` | Required in production: 32+ characters each, and different from each other |
| `API_KEY_PEPPER` | Required in production (32+ characters); secret mixed into API key hashes. Changing it invalidates every API key |
| `TRADING_MODE` | Must be `PAPER` |
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | Alpaca paper account; without them the API works but the engine cannot start |
| `ALPACA_BASE_URL` | Optional; must be the paper endpoint |
| `ENGINE_AUTOSTART` | Start the engine with the server (default false) |
| `RISK_*`, `SIZING_*` | Risk limits and sizing; see [Risk controls](#risk-controls) and `.env.example` |
| `DATABASE_URL` | Defaults to SQLite at `src/database/nocturna.db`; PostgreSQL is required in production (`postgresql://...` URLs use the psycopg 3 driver) |
| `ALLOWED_ORIGINS` | Required in production (CORS allowlist) |
| `TRUSTED_PROXY_COUNT` | Number of reverse proxies in front of the app (default 0). Set it behind nginx or a load balancer, or every client shares the proxy's IP for rate limits and login lockout |
| `REDIS_URL` | Rate-limit storage; in-process memory when unset |
| `ADMIN_USERNAME`, `ADMIN_EMAIL`, `ADMIN_PASSWORD` | Read by `flask create-admin` |

## Development

```bash
pip install --require-hashes -r requirements-dev.txt   # runtime + dev tools, hashed lockfile
pytest src/tests                                        # test suite
ruff check src/                                         # lint
```

### Database migrations

The schema is owned by the Alembic revisions in `src/migrations/versions/`, and the app
upgrades the database to the latest revision at startup. The same operations are available
as commands:

```bash
flask --app src.main db current                    # database revision vs latest
flask --app src.main db upgrade                    # upgrade explicitly
flask --app src.main db revision -m "add x"        # after changing src/models: autogenerate, then review
flask --app src.main db check                      # fails if models changed without a revision
flask --app src.main db downgrade <revision>       # asks for confirmation; back up first
```

The test suite fails if the models and the revisions disagree, on SQLite and, when
`TEST_POSTGRES_URL` is set (as in CI), on PostgreSQL.

A database created before migrations existed (by an older version of this app) is refused at
startup. It holds only the operator account, so back it up, delete `src/database/nocturna.db`
(or drop the tables on PostgreSQL), restart, and run `flask create-admin` again.

Dependencies are declared in `requirements.in` / `requirements-dev.in` and compiled into
hashed lockfiles with `uv pip compile ... --generate-hashes` (commands are in the `.in`
headers). CI (`.github/workflows/ci.yml`) runs ruff, bandit (report only), a dependency
audit with pip-audit, and pytest.

```
src/
├── main.py          # Flask app factory
├── cli.py           # operator commands (create-admin, db)
├── db_migrations.py # Alembic helpers; revisions live in migrations/
├── trading/         # the trading core (see docs/ARCHITECTURE.md)
│   ├── domain.py    #   instruments, bars, orders and their lifecycle, fills
│   ├── ledger.py    #   cash, positions, P&L
│   ├── engine.py    #   targets -> orders -> risk gate -> broker
│   ├── risk.py      #   pre-trade risk gate, loss limits and the kill switch
│   ├── sizing.py    #   caps, volatility target, drawdown ladder, stop-risk sizing
│   ├── strategy.py  #   strategy interface; strategies/ holds the strategies
│   ├── data.py      #   bar sources (in-memory, Alpaca) and the closed-bar rule
│   ├── broker.py    #   broker interface and the simulated broker
│   ├── alpaca.py    #   Alpaca paper broker and trade-update stream
│   ├── backtest.py  #   backtest runner
│   ├── service.py   #   live engine thread; runtime.py wires it into the app
│   └── ...          #   clock, calendar, feed, reconcile, persistence, lineup, safety
├── routes/          # REST endpoints
├── middleware/      # auth, security headers, rate limits
├── models/          # SQLAlchemy models (users, keys, audit, trading history)
├── migrations/      # Alembic revisions
├── utils/           # validators, logging, audit
├── static/          # compiled dashboard bundle
└── tests/
```

## Roadmap

| Phase | Scope |
|---|---|
| 0 | Safety freeze: paper-only guard, closed registration, no fabricated data, truthful docs |
| 1 | Platform foundations: auth fixes, scoped API keys, token revocation, enforced rate limits, PostgreSQL + migrations, persisted audit log, dependency clean-up |
| 2 | Event-driven core with identical backtest / paper / live code paths and broker reconciliation |
| 3 | Risk engine: pre-trade controls, equity-based loss limits, kill switch, volatility-targeted sizing (done) |
| 4 | Validation pipeline: walk-forward, purged cross-validation, deflated Sharpe ratio, overfitting tests |
| 5 | Strategy rebuild (trend, month-end rebalancing flows, crypto breakout, reversal), each admitted only if it passes the validation gates |
| 6–8 | Operations, API and dashboard, CI quality gates |
| 9 | Paper-trading soak, then a staged live ramp |

The full plan, operator decisions, strategy lineup and promotion gates are in
[docs/ROADMAP.md](docs/ROADMAP.md).

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

This software is provided as-is, without warranty. Trading involves substantial risk of
loss. Nothing in this repository is investment advice.
