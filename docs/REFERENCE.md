# NOCTURNA technical reference

Everything the [README](https://github.com/Yog-Sotho/Nocturna-Trading-System#readme) leaves out: what each part does and how well it is
tested, the full risk limits, the validation pipeline, the API, every setting, and how to
develop on the code. The design is in [ARCHITECTURE.md](ARCHITECTURE.md), the plan and its
decisions in [ROADMAP.md](ROADMAP.md), operations in [RUNBOOK.md](RUNBOOK.md) and the home
server in [HOME_SERVER.md](HOME_SERVER.md).

> ⚠️ **Paper trading only.** Live trading is disabled in code: the app refuses to start unless
> `TRADING_MODE=PAPER` and `ALPACA_BASE_URL` points at the Alpaca paper API. No strategy in this
> repository has passed out-of-sample validation, and none should be trusted with real money.

NOCTURNA is a Python/Flask service for researching, backtesting and paper-trading
rule-based strategies on US equities, ETFs and crypto through [Alpaca](https://alpaca.markets).
Its core is event-driven: the same strategy, ledger and risk code runs in backtests and in paper
trading; only the clock, the market-data source and the broker are swapped. See
[ARCHITECTURE.md](ARCHITECTURE.md) and [ROADMAP.md](ROADMAP.md).

## What exists today

| Component | Location | State |
|---|---|---|
| Trading core: domain model, ledger, engine, strategy interface | `src/trading/` | Working; property-tested accounting |
| Risk engine: pre-trade gate, drawdown kill switch, daily loss limit, volatility-targeted sizing with a drawdown ladder | `src/trading/risk.py`, `sizing.py` | Working; limits are configurable, defaults below |
| Backtester (simulated clock and broker, cost model, no look-ahead) | `src/trading/backtest.py` | Working |
| Validation pipeline: walk-forward traded through the engine, deflated Sharpe, probability of backtest overfitting, purged cross-validation, cost stress, parameter stability, regimes, promotion gates | `src/trading/validation/`, `flask validate` | Working; no candidate has passed the gates yet |
| Paper soak report: paper trading against a backtest of the same days, the account's dividends and interest, the soak tolerance | `src/trading/soak.py`, `flask soak report` | Working against recorded fakes; the soak has not started |
| Paper trading on Alpaca (`alpaca-py`): broker adapter, trade-update stream, polling data feed, reconciliation | `src/trading/alpaca.py`, `service.py` | Working against fakes and Alpaca's real request models; not yet run against a live paper account |
| Strategies | `src/trading/strategies/` | 60/40 benchmark (traded, idle cash in T-bills), plus month-end rebalancing flows, ETF time-series momentum, a crypto Donchian ensemble, the month-end payment cycle, a trend filter on the 60/40 and a blend; none has passed the promotion gates yet |
| Trading history (orders, fills, equity snapshots) and audit log | `src/models/` | Working |
| REST API with JWT and API-key auth (single operator) | `src/main.py`, `src/routes/` | Working; public registration is disabled |
| Dashboard: sign-in, account and risk figures, equity curve, positions and targets, working orders (cancel), fills and order history, start / stop / resume, emergency stop; phone-first, light and dark | `src/static/` (plain HTML, CSS and JavaScript; no build step) | Working; checked in Chromium at phone size by `tools/dashboard/e2e.cjs` |
| Telegram commands: `/status`, `/positions`, `/start_engine`, `/stop_engine`, `/resume`, `/halt`, `/flatten`, each change confirmed with a code | `src/trading/telegram.py` | Working against a fake Bot API; not yet run against Telegram |
| Operations: the engine as its own process, alerts (Telegram, ntfy, webhook), a dead man's switch, readiness and Prometheus endpoints, JSON logs, a Compose stack with backups and an HTTPS proxy | `src/trading/worker.py`, `alerts.py`, `docker-compose.yml`, [RUNBOOK.md](RUNBOOK.md) | Working; verified as a full stack without Alpaca keys |

### Known limitations

- No strategy has a validated edge: on Alpaca data from 2017 to 2026 none of the Phase 5
  candidates, alone or combined, beat the 60/40 benchmark out of sample, and neither did the
  month-end payment cycle, a trend filter on the 60/40 (also over 1974–2026 on long-history
  proxies) or low-volatility funds in place of SPY (see [ROADMAP.md](ROADMAP.md)).
  The engine trades the benchmark, with its idle cash in T-bills.
- The Compose stack has not yet run against the paper account (see the runbook before the
  first run).

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
curl -s -X POST http://localhost:5000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username": "admin", "password": "<your password>"}'
```

Start the engine with `POST /api/v1/trading/control/start` (admin), or set `ENGINE_AUTOSTART=true`.
On start it loads the paper account from Alpaca (cash, positions, working orders), warms up
history, and acts on the latest closed bars; after that it wakes each time a new bar becomes
final.

## Running it 24/7

`docker compose up -d --build` runs the API, the engine as its own process, PostgreSQL,
Redis and daily backups; `--profile proxy` adds HTTPS through Caddy. Setup, alerts, upgrades,
incidents and restores are in [RUNBOOK.md](RUNBOOK.md).

On an Ubuntu machine, `sudo ./setup.sh` does all of it. It first checks the machine and lists
the changes it would make, then asks. It installs what is missing (Docker, Compose, buildx,
Tailscale), serves the dashboard on a private Tailscale port, and writes `.env`, asking only
for what you must supply. It then starts the stack, creates your dashboard account and sends
a test alert. It never adds anyone to the `docker` group, and never changes what Tailscale
already serves. `sudo ./setup.sh --check` only reports. See
[HOME_SERVER.md](HOME_SERVER.md).

The API's entry point is `gunicorn -c gunicorn.conf.py 'src.main:create_app()'`, a single,
never-recycled worker (in development the engine lives in that process); the engine's is
`flask --app src.main engine run` with `ENGINE_MODE=external` for both.

## Risk controls

Every order, from a strategy or an operator, passes one risk gate before it reaches the
broker, and the book is sized by the same rules in backtests and in paper trading
(details in [ARCHITECTURE.md](ARCHITECTURE.md#risk)). Defaults, all fractions of
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
| Idle cash: the equity the sized targets leave unused is held in T-bills (BIL) | all of it, less a 0.5% cash buffer |

The T-bill sleeve is exempt from the position and single-order caps and may be bought during
a cool-down; it counts toward gross exposure, is never bought while trading is halted, and is
sold by the kill switch like everything else. Orders that only reduce a position skip the
daily breakers and data checks, so the book can always be reduced. The gate's state (high-water mark, halt, daily counters) is stored in the
database and restored on restart, so a halt survives a redeploy. After a drawdown halt,
`POST /api/v1/trading/control/resume` restarts the high-water mark at the current equity.
Strategies with protective stops size positions with `stop_risk_weight` (0.5% of equity at
risk per trade by default).

## Validating a strategy

No strategy trades until it passes the promotion gates in [ROADMAP.md](ROADMAP.md).
`flask validate` decides the gates a backtest can decide, on Alpaca data, with the engine, costs,
risk gate and sizing of paper trading:

```bash
flask --app src.main validate benchmark --start 2017-01-01 --output reports/benchmark.md
```

It writes a Markdown tear sheet and a JSON report, prints the verdict, and exits with code 1
when the candidate is rejected. Idle cash is held in T-bills (BIL), as in paper trading, and
every Sharpe ratio is measured in excess of BIL's return; `--no-tbills` runs the earlier rules
(cash earns nothing, Sharpe against zero). `--data history` runs a candidate that trades only
SPY and IEF on long-history proxies instead: French's daily US market and T-bill returns and
par bonds priced from FRED's 7- and 10-year Treasury yields, from 1970, with no Alpaca account
(for example `--data history --start 1971-01-01 --end 2026-08-31`). It is an additional test;
promotion is decided on Alpaca data. The command prints each downloaded file's SHA-256, and
[STRATEGY_SHORTLIST.md](STRATEGY_SHORTLIST.md#fixed-rules-trend-filter) shows how
closely the proxies track the ETFs. For each candidate (a strategy family and its parameter
grid):

1. Every parameter set is backtested over the whole period.
2. **Walk-forward:** at the start of each test year the parameter set with the best Sharpe ratio
   over the previous 36 months is chosen, and the chosen sets are traded through the engine, so
   switching parameters costs what it would live. This is the out-of-sample record.
3. **Selection bias:** the best set's Sharpe ratio is deflated for the number of effectively
   independent trials and for skewness and kurtosis (Bailey and López de Prado 2014); the
   probability of backtest overfitting comes from combinatorially symmetric cross-validation
   (Bailey, Borwein, López de Prado and Zhu 2017); combinatorial purged cross-validation gives a
   distribution of out-of-sample Sharpe ratios.
4. **Robustness:** the walk-forward again at 2x and 3x costs, the Sharpe ratio of neighbouring
   parameter sets, and returns by year and by market regime (trend and volatility of a reference
   symbol).

| Gate (backtest) | Threshold |
|---|---|
| Walk-forward Sharpe ratio, net of costs, in excess of T-bills | ≥ 1.0 (portfolio level; `--min-sharpe` for a component) |
| Deflated Sharpe ratio | ≥ 0.95 |
| Probability of backtest overfitting | ≤ 0.20 |
| At 3x costs | a total return above T-bills' |
| Median neighbouring parameter set's Sharpe / selected Sharpe | ≥ 0.5 |
| Out-of-sample maximum drawdown | ≤ 12% (the 10% kill switch plus gaps) |

Registered candidates: `benchmark`, `rebalancing`, `tsmom`, `crypto-trend` (use
`--start 2023-01-01 --train-months 12`: crypto data starts in 2021 and the longest channel needs
two years), `month-end`, `trend-filter`, `low-vol` and `lineup`.

Notes: Alpaca's free plan serves equity history from 2016, so the default first trading day is
2017-01-01 (2016 warms up indicators). Sharpe ratios are in excess of T-bills, which idle cash
earns in the engine. A backtest has no operator, so a drawdown kill is followed by a
simulated resume after `--resume-after-kill-days` (30); every kill is counted in the report.
Candidates are defined in `src/trading/validation/candidates.py`.

### The paper soak

The last gate is paper trading: at least 60 trading days that track a backtest of the same
days. `flask --app src.main soak report` reads what the engine recorded, backtests those days
with the same equity, lineup, sizing and risk limits, reads the paper account's dividends and
interest from Alpaca, and prints `PASS`, `NOT YET`, `REVIEW` or `FAIL` (exit code 1) with the
details. It passes when every engine error, halt and reconciliation difference is reviewed or
explained, the annualised tracking error of daily returns is below 2% and the total return is
within 1 percentage point. [HOME_SERVER.md](HOME_SERVER.md) shows how to run it on
the server.

## API

Version 1 of the API is served under `/api/v1`; `GET /api/version` lists the versions. The
OpenAPI 3.1 document is at `GET /api/v1/openapi.json` (or `flask --app src.main openapi
--output openapi.json`), generated from each route's declaration (`src/routes/openapi.py`,
response models in `src/routes/schemas.py`); contract tests check that every route is declared,
that its declared access holds and that real responses match its models. Money and quantities
are decimal strings.

All endpoints except `/health`, `/ready`, `/api/version`, `/api/v1/openapi.json`,
`/api/v1/trading/health` and the login and refresh endpoints require a bearer token or an API key (`X-API-Key` header). Roles and account
status are read from the database on every request, so changes apply immediately. Endpoints marked
*admin* also require the admin role; order and control endpoints require the trader role and
an account whose trading is not disabled.

**System:** `GET /health` (liveness), `GET /ready` (database), `GET /metrics` (Prometheus, with
`METRICS_TOKEN`), `GET /api/version`, `GET /api/v1/openapi.json`

Requests are rate limited per client IP: 300 per minute by default, with tighter limits on
login (10/min), token refresh (30/min), password changes (5/min), user creation (5/min) and
API key creation (10/hour); see `ROUTE_RATE_LIMITS` in `src/middleware/security.py`.
`/health` and the emergency stop are never limited. Five failed logins from one address
within 15 minutes block logins from it until the failures are 15 minutes old. Request bodies
are limited to 1 MB.

**Auth and users** (`/api/v1`)

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

**Trading** (`/api/v1/trading`)

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
| `ENGINE_MODE` | `embedded` (default: the engine runs in the web worker) or `external` (its own process, `flask engine run`; set it for both) |
| `ALERT_*`, `HEARTBEAT_URL` | Alert channels and the healthchecks.io dead man's switch; see [RUNBOOK.md](RUNBOOK.md) |
| `METRICS_TOKEN` | Bearer token for `GET /metrics`; unset, the endpoint does not exist |
| `LOG_JSON`, `LOG_TO_FILE` | JSON log lines; no log files (stdout only) |
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
ruff check src/ tools/ setup/                           # lint
mypy                                                    # strict typing of the trading core
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
headers).

The documentation site (<https://yog-sotho.github.io/Nocturna-Trading-System/>) is built from
`docs/` with MkDocs Material (`mkdocs.yml`, locked in `requirements-docs.txt`) by
`.github/workflows/docs.yml`: a strict build on pull requests, a deployment to GitHub Pages on
every push to `main`. To preview it:

```bash
pip install --require-hashes -r requirements-docs.txt
mkdocs serve
```

### Quality gates

CI (`.github/workflows/ci.yml`) blocks a merge on any of:

| Gate | What it checks |
|---|---|
| ruff | lint of `src/`, `tools/` and `setup/` |
| mypy | `--strict` typing of the trading core (`src/trading/`, `src/models/trading.py`; settings in `pyproject.toml`) |
| bandit | security findings of medium severity and confidence or higher; a reviewed exception carries `# nosec <id>` with the reason on the line above |
| pip-audit | known vulnerabilities in the lockfiles (development, mutation testing, docs site) |
| pytest | the suite on SQLite and PostgreSQL; at least 94% line coverage overall and 95% for each core module (`tools/coverage_gate.py`) |
| dashboard | the dashboard against a simulated engine in Chromium at phone size (`tools/dashboard/`) |
| image | the Docker image builds and the Compose stack starts healthy with a throwaway `.env` |

Mutation testing (`.github/workflows/mutation.yml`, weekly, on demand, and on pull requests
that touch the risk code) changes `risk.py` and `sizing.py` one operator at a time with
[Cosmic Ray](https://github.com/sixty-north/cosmic-ray) and runs the risk tests against each
change; a change no test notices survives, and the job fails when more than 3% of the tested
changes survive. To run it locally (it takes a while):

```bash
pip install --require-hashes -r requirements-mutation.txt
cosmic-ray init tools/mutation/cosmic-ray.toml mutation.sqlite
cr-filter-operators mutation.sqlite tools/mutation/cosmic-ray.toml
cr-filter-pragma mutation.sqlite
cosmic-ray exec tools/mutation/cosmic-ray.toml mutation.sqlite
cosmic-ray dump mutation.sqlite | python tools/mutation/gate.py --max-survival 3
```

### Source layout

```
src/
├── main.py          # Flask app factory
├── cli.py           # operator commands (create-admin, db, validate, soak, engine, openapi, send-test-alert)
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
│   ├── validation/  #   statistics, overfitting tests, walk-forward, gates, tear sheet
│   ├── service.py   #   live engine thread; runtime.py wires it into the app
│   └── ...          #   clock, calendar, feed, reconcile, persistence, lineup, safety
├── routes/          # REST endpoints
├── middleware/      # auth, security headers, rate limits
├── models/          # SQLAlchemy models (users, keys, audit, trading history)
├── migrations/      # Alembic revisions
├── utils/           # validators, logging, audit
├── static/          # the dashboard (index.html, app.js, app.css)
└── tests/
```
