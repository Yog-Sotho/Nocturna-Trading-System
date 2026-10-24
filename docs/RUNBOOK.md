# NOCTURNA operations runbook

Setting up a home server (Ubuntu, Docker, Tailscale, Telegram, healthchecks.io): run
`sudo ./setup.sh` ([HOME_SERVER.md](HOME_SERVER.md)). `sudo ./setup.sh --check` reports what
the machine has and what is missing at any time, and shows no secrets. On a machine set up
this way Docker runs through `sudo`: put it in front of the `docker` commands below.

How to run the paper-trading stack around the clock on one host with Docker
Compose, what the alerts mean, and what to do when something goes wrong.
Everything here is paper trading on Alpaca; the app refuses any live
configuration.

## The stack

`docker-compose.yml` runs:

| Service | What it does | State it keeps |
|---|---|---|
| `api` | Web API and dashboard (gunicorn, one worker) | none |
| `engine` | The trading engine, in a process of its own (`flask --app src.main engine run`) | in the database |
| `postgres` | The database both share: orders, fills, equity, risk state, audit log, the engine's status and command queue | volume `postgres-data` |
| `redis` | Rate-limit counters | none (not persisted) |
| `backup` | A `pg_dump` into `./backups` every day | `./backups` on the host |
| `caddy` | Optional HTTPS reverse proxy (`--profile proxy`) | volumes `caddy-data`, `caddy-config` (certificates) |

The API and the engine meet only in the database: the engine publishes its
state every second, and the API queues commands (start, stop, orders,
emergency stop) that the engine runs. So restarting or redeploying the API
never touches trading, and the engine keeps trading while the API is down.

Every container restarts on failure (`restart: unless-stopped`), logs to
`docker compose logs` (rotated, 5 × 10 MB each), and has a memory limit.

## First deployment

1. Install Docker Engine with the Compose plugin on the host.
2. Clone the repository and create the configuration:

   ```bash
   git clone https://github.com/Yog-Sotho/Nocturna-Trading-System.git
   cd Nocturna-Trading-System
   cp .env.example .env
   chmod 600 .env
   ```

3. Edit `.env`. Generate every secret with
   `python3 -c "import secrets; print(secrets.token_hex(32))"`:
   - `FLASK_SECRET_KEY`, `JWT_SECRET_KEY` (different), `API_KEY_PEPPER`;
   - `DB_PASSWORD`, `REDIS_PASSWORD`: hex only, since they go into URLs;
   - `ALLOWED_ORIGINS`: the address you open the dashboard from, such as
     `https://trading.example.com`;
   - `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`: the paper account's keys;
   - alerts and the heartbeat ([below](#alerts-and-the-dead-mans-switch));
   - `ENGINE_AUTOSTART=true` if a restart (a crash, a host reboot, an
     upgrade) should resume trading by itself. Without it the engine waits
     to be started after every restart: **Start engine** in the dashboard,
     `/start_engine` in Telegram, or `POST /api/v1/trading/control/start`.

   `DATABASE_URL`, `REDIS_URL`, `FLASK_ENV`, `ENGINE_MODE`, `TRADING_MODE`
   and the log settings are fixed by `docker-compose.yml`; values for them
   in `.env` are ignored.
4. Build and start, then wait until every service is `healthy`:

   ```bash
   docker compose up -d --build
   docker compose ps
   ```

5. Create the operator account (it prompts for a password):

   ```bash
   docker compose exec api flask --app src.main create-admin --username ops --email you@example.com
   ```

6. Check the alert channels: `docker compose exec api flask --app src.main send-test-alert`
   sends an urgent test alert to each one and prints `sent` or why it failed.

### Reaching the API

The API listens on `127.0.0.1:5000` of the host only. Either:

- reach it through [Tailscale](https://tailscale.com) from your own devices,
  with no port opened to the internet: `sudo tailscale serve --bg 5000` and
  `TRUSTED_PROXY_COUNT=1` (step by step in [HOME_SERVER.md](HOME_SERVER.md));
- tunnel to it: `ssh -L 5000:127.0.0.1:5000 you@host`, then use
  `http://localhost:5000`; or
- serve it over HTTPS with the `caddy` proxy. Point a DNS record for your
  domain at the host, open ports 80 and 443, and set in `.env`:

  ```bash
  DOMAIN=trading.example.com
  TRUSTED_PROXY_COUNT=1          # Caddy sets X-Forwarded-For; the API trusts one proxy
  ALLOWED_ORIGINS=https://trading.example.com
  ```

  then run `docker compose --profile proxy up -d`. Caddy obtains and renews
  the certificate itself and redirects plain HTTP to HTTPS. Without
  `TRUSTED_PROXY_COUNT=1` every client would share Caddy's address for rate
  limits and login lockouts.

## Alerts and the dead man's switch

Alerts go to any combination of Telegram, ntfy and a JSON webhook. Urgent
ones ring (ntfy priority 5; Telegram with sound), the others arrive silently.
An alert repeated within 15 minutes is sent once.

| Alert | Urgent | Meaning |
|---|---|---|
| Kill switch: trading halted | yes | Drawdown reached `RISK_MAX_DRAWDOWN`; positions are being closed (`RISK_FLATTEN_ON_KILL`) |
| Trading halted | yes | An emergency stop or another halt |
| Daily loss limit reached | yes | No new orders until the next New York trading day |
| Broker error | yes | A broker report that contradicts the order's state, or a failed kill-switch liquidation (retried on every bar) |
| Fill the engine did not place | yes | Someone or something else traded the account |
| Engine failed | yes | The engine could not start |
| Engine not answering | yes | Sent by the API: the engine process has published nothing for 30 seconds |
| Engine error | no | A recoverable error, such as market data that failed and is retried every minute |
| Strategy error | no | A strategy raised; the engine keeps its last targets until it works again |
| Positions corrected from the broker | no | Reconciliation found the engine's book differed from Alpaca's and adopted Alpaca's |
| Engine process started / stopped | no | A start without a stop before it was a crash |
| Trading resumed, Engine answering again | no | Recovery |
| Filled: buy 3 SPY | no | An order the engine placed was filled (off until you turn fills on) |
| Lab run #N finished / failed | no | A strategy-lab run is done; its result is on the dashboard's **Lab** page |

On the dashboard's **Alerts** page you choose which channels are used and
which kinds of alert are sent: warnings (engine, strategy and reconciliation
errors), comings and goings (the engine process starting and stopping,
resumes, the engine answering again), fills and lab results. Safety alerts
(the kill switch, halts, the daily loss limit) and problems (broker errors,
a fill the engine did not place, a failed or silent engine) are always sent,
and at least one configured channel is always used. Tokens and addresses
stay in `.env`; both processes follow a change within about 10 seconds.
**Send a test alert** on the same page (or `flask --app src.main
send-test-alert`) reports what each channel answered.

### Telegram

1. In Telegram, message `@BotFather` with `/newbot`, choose a name and a
   username; it answers with the bot's token (`123456789:AA...`).
2. Open a chat with your bot and press **Start** (or send it any message).
3. Open `https://api.telegram.org/bot<token>/getUpdates` in a browser; the
   number at `"chat":{"id":...}` is your chat id.
4. Set `ALERT_TELEGRAM_BOT_TOKEN` and `ALERT_TELEGRAM_CHAT_ID`.

### Commands from Telegram

With `TELEGRAM_COMMANDS=true` the same bot also takes commands, only from that
private chat (messages from anyone else, and from groups, are ignored):

| Command | What it does |
|---|---|
| `/status` | Engine state, equity, drawdown, halt, cool-down, last market data, last error |
| `/positions` | Positions and working orders |
| `/start_engine` | Start the engine (after a restart when `ENGINE_AUTOSTART=false`) |
| `/stop_engine` | Stop the engine; positions and working orders stay at Alpaca |
| `/resume` | Lift a halt; the high-water mark restarts at the current equity |
| `/halt [reason]` | Emergency stop: halt and cancel every order, keeping the positions |
| `/flatten [reason]` | Emergency stop that also closes every position |

Every command that changes something answers with a code and runs only after
`/confirm <code>` within 60 seconds, since a tap on the command menu sends at
once. A change sent while the server was down is not run when it comes back.
The bot runs in the `api` container and acts like the API's own endpoints:
with the engine not answering, `/halt` and `/flatten` go to Alpaca directly.
Each action is in the audit log with `"source": "telegram"`. Anyone who can
use your Telegram account can use these commands: protect the account with
two-step verification.

### ntfy

Install the ntfy app and subscribe to a topic whose name nobody can guess
(on the public `ntfy.sh` server anyone who knows the name can read it), for
example the output of `python3 -c "import secrets; print('nocturna-' + secrets.token_hex(12))"`.
Set `ALERT_NTFY_URL=https://ntfy.sh/<that topic>`. For a protected topic or
your own server, also set `ALERT_NTFY_TOKEN` (`tk_...`).

### The dead man's switch (healthchecks.io)

Alerts come from the stack itself, so they cannot report a dead host. A
[healthchecks.io](https://healthchecks.io) check can:

1. Create a check with a period of 1 minute and a grace time of, say,
   5 minutes; copy its ping URL (`https://hc-ping.com/<uuid>`).
2. Set `HEARTBEAT_URL` to it. The engine pings it every
   `HEARTBEAT_INTERVAL_SECONDS` (default 60) while it runs, and pings
   `/fail` at once when it fails. A stopped engine, a dead process or a dead
   host sends nothing, and healthchecks.io alerts when the grace time runs
   out. Pause the check there before you stop the engine on purpose.
3. For the backups, create a second check with a period of 1 day and a
   grace time of a few hours, and set `BACKUP_HEARTBEAT_URL`.

## Daily operation

- **State:** `GET /api/v1/trading/status` shows the engine's state, equity,
  drawdown, whether it is halted or cooling down, and the last
  reconciliation. `docker compose ps` shows each container's health.
- **BIL in the positions** is the engine's idle cash held in T-bills (the
  SPDR Bloomberg 1-3 Month T-Bill ETF), not a strategy position. Its size
  moves with the 60/40's: it takes what the volatility target and the
  drawdown ladder leave unused. The kill switch sells it with everything else.
- **Changing the mix or the limits:** the dashboard's **Strategies** and
  **Risk** pages save a new version of the trading profile (each save is in
  the audit log as `PROFILE_SAVED`). The running engine keeps trading the
  version it started with until it restarts (**Restart now**, or
  `POST /api/v1/trading/control/restart`); then it sells positions in
  symbols the new mix does not trade. Restart outside US market hours when
  you can: orders go to the next auction either way.
- **The strategy lab:** each run executes in a `flask lab run` process that
  the API starts inside its own container, one run at a time; queued runs
  wait. A run whose process dies (for instance when the `api` container is
  recreated) is marked failed within 5 minutes; queue it again. Long-history
  runs download Kenneth French's and FRED's files each time.
- **Logs:** `docker compose logs -f engine` (or `api`); one JSON object per
  line (`LOG_JSON=false` in `.env` for plain text).
- **Metrics:** with `METRICS_TOKEN` set, `GET /metrics` serves Prometheus
  metrics to `Authorization: Bearer <token>`. A Prometheus scrape job:

  ```yaml
  scrape_configs:
    - job_name: nocturna
      scheme: https
      authorization:
        credentials_file: /etc/prometheus/nocturna-token
      static_configs:
        - targets: ["trading.example.com"]
  ```

- **Health endpoints:** `/health` answers while the API process runs;
  `/ready` also needs the database (503 without it). The engine's state does
  not count, so the API stays up, and its emergency stop with it, while the
  engine is down.

### Upgrading

```bash
git pull
docker compose up -d --build api        # the API only: trading is not interrupted
docker compose up -d --build engine     # the engine: do it outside US market hours
```

Restarting the engine stops it cleanly (up to 60 seconds). On start it
reloads the account from Alpaca (cash, positions, working orders),
reconciles, restores the risk state (high-water mark, halts) from the
database, warms up and acts on the latest closed bars. A halt survives the
restart. Database migrations run on start, in whichever of the two
starts first.

## Incidents

### Emergency stop

```bash
curl -X POST https://trading.example.com/api/v1/trading/emergency-stop \
  -H "Authorization: Bearer <token>" -H 'Content-Type: application/json' \
  -d '{"reason": "what happened", "close_all_positions": true}'
```

It halts the engine and cancels its orders; `close_all_positions`, which is
the default, also closes every position. Send `false` to keep the positions.
When the engine process is not running, or has not answered within 25
seconds, the API asks Alpaca directly, and the halt stays queued for the
engine when it comes back.

When the API is unreachable too: run `docker compose stop engine` **first**,
then cancel orders and close positions on the Alpaca paper dashboard. A
running engine would buy its targets back on the next bar.

### Kill switch fired

Trading is halted and, with `RISK_FLATTEN_ON_KILL`, every position is being
closed. Check `GET /api/v1/trading/status` (`halted`, `halt_reason`, `risk`) and
the Alpaca dashboard. Understand the loss before resuming: `POST
/api/v1/trading/control/resume` lifts the halt and restarts the high-water mark
at the current equity.

### Daily loss limit reached

The engine sends no new orders until the next New York trading day and then
clears the cool-down itself. Nothing to do unless the loss needs explaining.

### Engine failed

Read `docker compose logs --tail 200 engine`. The usual causes are paper keys
that are wrong or were regenerated (update `.env`, then `docker compose up -d
engine`), or Alpaca being unreachable (wait, then start again).

### Engine not answering

The API has seen nothing from the engine for 30 seconds. Check `docker
compose ps engine` and its logs. A crashed engine is restarted by Docker by
itself; one that hangs is not (Docker does not restart unhealthy
containers): `docker compose restart engine`.

### Broker error

Read the alert and the engine's log. A failed kill-switch liquidation is
retried on every new bar; if it keeps failing, stop the engine and close the
positions on the Alpaca dashboard. A broker report that contradicts an
order's state is logged; the next reconciliation takes the order's final
state, the positions and the cash from Alpaca.

### Fill the engine did not place

Something other than the engine traded the account: a manual trade on the
dashboard, or another program using the same keys. The engine adopts the
result at its next reconciliation. Keep the paper account for the engine
alone.

### Positions corrected from the broker

Reconciliation found the engine's book different from Alpaca's and adopted
Alpaca's. Expected after a manual trade, a dividend or interest (cash) or a
fee the engine did not book; anything else is worth reading the engine's log
for.

### Database down

`/ready` answers 503 and the API cannot reach the engine. Bring PostgreSQL
back (`docker compose ps postgres`, `docker compose logs postgres`; a full
disk is the usual cause). Until then, act on the account from the Alpaca
dashboard after stopping the engine.

## Backups and restores

The `backup` service writes `./backups/nocturna-<UTC time>.dump` 10 minutes
after it starts and then every day, and deletes those older than
`BACKUP_KEEP_DAYS` (14). They are on the same disk as the database: copy
`./backups` to another machine regularly, or a lost host loses the backups too.

A backup now (named `manual-`, so retention never deletes it):

```bash
docker compose exec backup sh -c 'pg_dump --format=custom --file=/backups/manual-$(date -u +%Y%m%dT%H%M%SZ).dump'
```

Restore in place (this replaces everything in the database with the dump):

```bash
docker compose stop api engine
docker compose exec backup pg_restore --clean --if-exists --single-transaction --no-owner \
  -d nocturna /backups/nocturna-20260101T000000Z.dump
docker compose start engine api
```

On a new host: copy the repository, `.env` and the dump into `./backups`,
run `docker compose up -d postgres backup`, restore as above, then `docker
compose up -d`.

After a restore the engine reloads positions and cash from Alpaca at start,
so the account is always taken from the broker. The dump brings back the
history (orders, fills, equity), the audit log and the risk state: the
high-water mark and any halt as of the backup.
