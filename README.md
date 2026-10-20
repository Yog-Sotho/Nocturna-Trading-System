<div align="center">
  <img src="src/assets/banner.png" alt="Nocturna Banner" width="420" style="margin-bottom: 20px;">

# NOCTURNA — Algorithmic Trading Framework

</div>

> [!WARNING]
> **Status: prototype under active rebuild. Paper trading only.**
> Live trading is disabled in code: the app refuses to start unless `TRADING_MODE=PAPER`
> and `ALPACA_BASE_URL` points at the Alpaca paper API. No strategy in this repository has
> passed out-of-sample validation, and none should be trusted with real money.

NOCTURNA is a Python/Flask service for researching and paper-trading rule-based strategies
on US equities through [Alpaca](https://alpaca.markets). It is being rebuilt into an
event-driven framework in which the same strategy code runs in backtest, paper and live
trading. See [Roadmap](#roadmap).

## What exists today

| Component | Location | State |
|---|---|---|
| REST API with JWT auth (single operator) | `src/main.py`, `src/routes/` | Working; public registration is disabled |
| Trading engine prototype with four modes — EVE (grid), LUCIFER (breakout), REAPER (reversal), SENTINEL (trend) | `src/core/` | Experimental, with known defects (below) |
| Paper trading engine (prices supplied through the API) | `src/core/paper_trading.py` | Working, simplified fill model |
| Backtester | `src/advanced/backtester.py` | Single-symbol prototype |
| ML parameter tuner, sentiment analyser | `src/advanced/` | Experimental, not validated |
| Event bus and order state machine | `src/core/event_bus.py`, `src/core/order_state_machine.py` | Foundation for the rebuild; not yet driving orders |
| Dashboard | `src/static/` | Compiled bundle only (no source); it does not send auth headers, so its API calls are rejected |

### Known limitations

These are documented in the codebase audit and are scheduled in the roadmap:

- The engine's drawdown check uses position value instead of account equity, so it
  triggers an emergency stop on its first risk check.
- Orders submitted through `POST /api/trading/orders` are rejected (field-name mismatch
  between the API schema and the order manager).
- Rate limits are configured but not attached to any route.
- Positions, orders and risk state live in memory and are lost on restart.
- Market data is fetched without a start date, so Alpaca returns only the current day's bars.
- Realized P&L is never computed, so win rate and loss-based limits do not work.

## Quick start (local, paper only)

Requires Python 3.11+.

```bash
git clone https://github.com/Yog-Sotho/Nocturna-Trading-System.git
cd Nocturna-Trading-System
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set FLASK_SECRET_KEY and JWT_SECRET_KEY
#   python -c "import secrets; print(secrets.token_hex(32))"
# Keep TRADING_MODE=PAPER and the paper ALPACA_BASE_URL.

# Create the operator account (prompts for a password)
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

In the Docker image the production entry point is
`gunicorn -c gunicorn.conf.py 'src.main:create_app()'`. It must run a single worker
because the engine keeps its state in memory.

## API

All endpoints except `/health`, `/api/version`, `/api/trading/health` and the login and
refresh endpoints require a bearer token. Endpoints marked *admin* also require the admin role.

**System:** `GET /health`, `GET /api/version`

**Auth and users** (`/api`)

| Method | Path | Notes |
|---|---|---|
| POST | `/auth/login`, `/auth/refresh`, `/auth/logout` | |
| GET | `/auth/me` | |
| GET, POST | `/users` | admin; `POST` creates a user |
| GET, PUT | `/users/<id>` | admin |
| PUT | `/users/<id>/password` | own account, or admin |
| GET, POST | `/api-keys` | key creation works, but keys are not yet accepted by auth |
| DELETE | `/api-keys/<id>` | |
| GET | `/audit-logs` | admin; the audit table is not yet written to |

**Trading engine** (`/api/trading`)

| Method | Path | Notes |
|---|---|---|
| GET | `/health`, `/status`, `/status/detailed`, `/positions`, `/orders`, `/performance`, `/strategy`, `/risk`, `/config`, `/trades`, `/equity-curve` | |
| POST | `/control/start`, `/control/stop`, `/control/pause`, `/control/resume` | admin |
| POST | `/emergency-stop` | admin; body `{"reason": "...", "close_all_positions": true}` |
| POST | `/orders`, `/orders/<id>/cancel` | |
| PUT | `/config` | admin |
| POST | `/optimize` | admin; returns 501 (not implemented) |

**Paper trading** (`/api/paper`)

| Method | Path |
|---|---|
| POST | `/start`, `/stop`, `/reset`, `/orders`, `/prices` |
| GET | `/orders`, `/positions`, `/trades`, `/portfolio`, `/equity-curve` |
| DELETE | `/orders/<id>` |

## Configuration

All settings come from environment variables; `.env.example` lists them with comments.
The ones that matter most:

| Variable | Purpose |
|---|---|
| `FLASK_SECRET_KEY`, `JWT_SECRET_KEY` | Required in production |
| `TRADING_MODE` | Must be `PAPER` |
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ALPACA_BASE_URL` | Alpaca paper account; without keys the engine runs in simulation |
| `DATABASE_URL` | Defaults to SQLite at `src/database/nocturna.db` |
| `ALLOWED_ORIGINS` | Required in production (CORS allowlist) |
| `ADMIN_USERNAME`, `ADMIN_EMAIL`, `ADMIN_PASSWORD` | Read by `flask create-admin` |

## Development

```bash
pip install --require-hashes -r requirements-dev.txt   # runtime + dev tools, hashed lockfile
pytest src/tests                                        # test suite
ruff check src/                                         # lint
```

Dependencies are declared in `requirements.in` / `requirements-dev.in` and compiled into
hashed lockfiles with `uv pip compile ... --generate-hashes` (commands are in the `.in`
headers). CI (`.github/workflows/ci.yml`) runs ruff, bandit (report only), a dependency
audit with pip-audit, and pytest.

```
src/
├── main.py          # Flask app factory
├── cli.py           # operator commands (create-admin)
├── core/            # engine, strategies, risk, orders, market data, paper trading, safety guard
├── advanced/        # backtester, ML tuner, sentiment
├── routes/          # REST endpoints
├── middleware/      # auth, security headers
├── models/          # SQLAlchemy models
├── utils/           # validators, logging
├── static/          # compiled dashboard bundle
└── tests/
```

## Roadmap

| Phase | Scope |
|---|---|
| 0 | Safety freeze: paper-only guard, closed registration, no fabricated data, truthful docs |
| 1 | Platform foundations: auth fixes, enforced rate limits, PostgreSQL + migrations, dependency clean-up, migration to `alpaca-py` |
| 2 | Event-driven core with identical backtest / paper / live code paths and broker reconciliation |
| 3 | Risk engine: pre-trade controls, equity-based loss limits, kill switch, volatility-targeted sizing |
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
