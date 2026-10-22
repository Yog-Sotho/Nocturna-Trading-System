# Running Nocturna on your home server

The setup script runs the trading system around the clock on an Ubuntu machine at home,
headless or not. It works on a fresh machine, and on one that already runs other services.
You reach the dashboard from your phone through Tailscale, and alerts and commands come
through Telegram. Paper trading only.

```bash
git clone https://github.com/Yog-Sotho/Nocturna-Trading-System.git
cd Nocturna-Trading-System
sudo ./setup.sh --check     # what is there, what is missing, what it would change: changes nothing
sudo ./setup.sh             # the same, then it asks, and sets everything up
```

## What the setup does

It first looks at everything Nocturna needs on this machine, and changes nothing while doing
so:
- **The system:** Ubuntu version, memory, disk and clock.
- **Tools:** git, curl and ca-certificates.
- **Docker:** where it was installed from, whether it starts at boot and runs, and whether
  Compose v2 and buildx are there.
- **Your user and Docker:** whether your account is in the `docker` group.
- **Tailscale:** connected, MagicDNS, HTTPS certificates, key expiry, and everything it
  already serves, with which entries are public through Funnel.
- **Ports:** which ones are in use, and by which program.
- **The network:** whether the services the setup and the stack use answer.
- **Your settings:** `.env`, and the app's own verdict on them.
- **The stack:** each service's state, and the newest backup.
- **The dashboard account.**

Then it lists the changes it would make, in order, and asks. If something it must not decide
for you is in the way, it says what and stops before changing anything (see
[When it stops before changing anything](#when-it-stops-before-changing-anything)).

When you say yes, it does each change and verifies it:

1. **The clock:** turns on network time, if the clock is not synchronized.
2. **Missing packages:** installs whichever of git, curl and ca-certificates is missing.
3. **Docker:** installs Docker Engine, Compose and buildx from Docker's repository, if Docker is
   missing. If Docker is there, it only adds a missing Compose or buildx, and starts the service
   if it is not running.
4. **Tailscale:** installs it and connects the machine, if it is missing.
5. **The dashboard's address:** serves the dashboard on its own private Tailscale HTTPS port.
   That is **9443**, or the next free port. Funnel can only use ports 443, 8443 and 10000, so
   this port can never become public, even by mistake.
   - The setup compares Tailscale's configuration before and after. If anything other than the
     new entry changed, it removes the new entry again and stops.
6. **`.env`:** generates the passwords and secrets, and asks you only for what it cannot work out
   (see [What it asks you](#what-it-asks-you)). It tests each value as you give it.
7. **The stack:** builds and starts it, and waits until every service is healthy.
8. **Your dashboard account:** creates it.
9. **Alerts:** sends a test alert to Telegram.

It ends by checking everything again and printing your dashboard's address.

Run it again at any time, for instance after `git pull`. What is already done stays as it is,
and `.env` keeps every value unless you change it. When a step fails, the setup says why and
stops; fix that and run it again, and it carries on from there.

### What it never does

- **The `docker` group:** it never adds anyone to it. Membership works as root without a
  password, so Docker runs through `sudo` instead.
- **Packages:** it never removes one.
- **What Tailscale already serves:** it never changes those entries or their Funnel, and never
  runs `tailscale up` on a machine that is already set up.
- **Your other services:** it never touches nginx or any other service. The API listens on
  `127.0.0.1` only, on port 5000 or the next free one. PostgreSQL and Redis stay inside Docker's
  network.
- **Trading:** it never starts it. You do, when you are ready.
- **Secrets:** it never prints one, so you can share its reports when you need help.

## Before you run it

These are the things only you can do. Have them ready, because the setup asks for them.

**Alpaca paper keys.**
1. Sign in at [app.alpaca.markets](https://app.alpaca.markets).
2. Make sure the account switcher at the top left shows your **Paper** account, not Live.
3. On the paper account's home page, find the **API Keys** box and generate the keys.
4. Copy both the **Key ID** and the **Secret Key**: the secret is shown only once. Paper Key IDs
   normally start with `PK`.

Generating new keys disables the previous pair, wherever it is still in use, so if you still
have a working pair, use it.

**A Telegram bot.**
1. In Telegram, open **@BotFather**, Telegram's own bot for creating bots, and send `/newbot`.
2. Give the bot a display name.
3. Give it a username: 5 to 32 letters, digits or underscores, ending in `bot`.
4. BotFather answers with the token, which looks like
   `4839574812:AAFD39kkdpWt3ywyRZergyOLMaJhac60qc`. Anyone with the token controls the bot,
   so keep it private.

You do not need your chat id: the setup finds it.

**healthchecks.io** (recommended): it tells you when the machine goes silent.
1. Create a check `engine`, with **Period 1 minute** and **Grace Time 5 minutes**.
2. Create a check `backup`, with **Period 1 day** and **Grace Time 2 hours**.
3. Each check's page shows its ping URL, `https://hc-ping.com/<uuid>`.
4. To be told on Telegram, open **Integrations**, choose **Telegram**, send `/start` to
   **@HealthchecksBot**, open the link it answers with, and click **Connect Telegram**.

**Tailscale.** You need an account at [tailscale.com](https://tailscale.com), and the Tailscale
app signed in on your phone. If this machine is not on your tailnet yet, the setup installs
Tailscale and prints a sign-in link. In the [admin console](https://login.tailscale.com/admin):
- **DNS page:** turn on **MagicDNS** and, under **HTTPS Certificates**, select **Enable HTTPS**.
  The setup checks both and waits for you if they are off. Machine names and your tailnet's name
  then appear in a public certificate log.
- **Machines page:** open the menu at the right of the server's row and choose **Disable Key
  Expiry**. Otherwise the server drops off your tailnet when its key expires, after 180 days by
  default. The setup tells you the date if the key expires.

## What it asks you

1. **Go ahead?** It asks this after showing the list of changes.
2. **Alpaca:** the Key ID and the Secret Key. Both are hidden as you type or paste. It signs in
   to your paper account to check them, and shows the account number and equity.
3. **Telegram:**
   - The bot token, hidden; it checks the token with Telegram.
   - It then tells you to open your bot in Telegram and tap **Start**, finds your chat, and
     sends a test message.
   - Whether the bot may also take commands (`/status`, `/start_engine`, `/halt`…).
4. **healthchecks.io:** the ping URLs of the `engine` and `backup` checks. Press Enter to skip
   one.
5. **Your dashboard account:**
   - A username: letters, digits and underscores; your login name is offered.
   - An email: it only identifies the account, and nothing is sent to it.
   - A password, typed twice and hidden: at least 12 characters, with upper case, lower case,
     a digit and a symbol.
6. **Did the test alert reach your phone?**

When you run the setup again, it offers to keep each value already in `.env` (press Enter).
It tests the Alpaca keys and the bot token again.

## When it stops before changing anything

| It says | What to do |
|---|---|
| Docker comes from the snap store | It does not remove the snap for you: that would delete its containers and images. If nothing else uses it, run `sudo snap remove docker`, then run the setup again, and it installs Docker's packages. |
| Packages that conflict with Docker are installed | Other software (Kubernetes, Podman) may need them. Remove them only if nothing uses them, then run the setup again. |
| Tailscale is installed but not logged in, stopped, or its service is not running | Bring it up the way you normally do, with your usual options; the setup does not change your Tailscale settings. Then run the setup again. |
| Tailscale waits for this machine to be approved | Approve it on the Machines page of the admin console. |
| The dashboard is served with Funnel on, so it is public | Turn that off (`sudo tailscale funnel --https=<port> off`), then run the setup again: it serves the dashboard privately. |
| This machine is not Ubuntu | The setup installs software with Ubuntu's tools. |

## Docker with `sudo`

Your account stays out of the `docker` group, so Docker commands start with `sudo`:

```bash
sudo docker compose ps                     # each service's state
sudo docker compose logs api --tail 50     # what a service says (api, engine, postgres, redis, backup)
```

## What goes in `.env`

The setup writes `.env` with permissions that only root can read. If you edit it by hand
(`sudo nano .env`): one `NAME=value` per line, with no spaces around `=` and no quotes needed.
Then run `sudo ./setup.sh` to check the result and apply it.

| Setting | Needed | What it is | Where it comes from |
|---|---|---|---|
| `FLASK_SECRET_KEY` | yes | Signs the server's cookies | Generated by the setup (64 random hex characters) |
| `JWT_SECRET_KEY` | yes | Signs sign-in tokens; must differ from `FLASK_SECRET_KEY` | Generated by the setup |
| `API_KEY_PEPPER` | yes | Mixed into stored API key hashes; changing it voids every API key | Generated by the setup |
| `DB_PASSWORD` | yes | The database password, fixed when the database is first created; letters and digits only | Generated by the setup; never change it afterwards |
| `REDIS_PASSWORD` | yes | Password of the rate-limit store; letters and digits only | Generated by the setup |
| `ALPACA_API_KEY` | yes | The Key ID of your Alpaca **paper** account | [Before you run it](#before-you-run-it) |
| `ALPACA_SECRET_KEY` | yes | Its Secret Key | [Before you run it](#before-you-run-it) |
| `ALLOWED_ORIGINS` | yes | The dashboard's address, such as `https://<machine>.<tailnet>.ts.net:9443`, with no trailing `/`. Only requests from other websites are checked against it, but the stack refuses to start without it. | Set by the setup from Tailscale |
| `TRUSTED_PROXY_COUNT` | yes | `1`: Tailscale Serve forwards visitors to the API. With `0`, every visitor looks the same, and a few failed logins lock everyone out. | Set by the setup |
| `API_PORT` | no | The API's port on this machine, on `127.0.0.1` only (default `5000`) | Set by the setup: 5000, or the next free port |
| `ENGINE_AUTOSTART` | no | `false`: after any restart, trading waits for you | Set by the setup to `false` |
| `ALERT_TELEGRAM_BOT_TOKEN` | recommended | Your bot's token | [Before you run it](#before-you-run-it) |
| `ALERT_TELEGRAM_CHAT_ID` | recommended | The id of your private chat with the bot (a number) | Found by the setup after you tap Start |
| `TELEGRAM_COMMANDS` | recommended | `true`: the bot also takes `/status`, `/start_engine`, `/halt` and the other commands, each change confirmed with a code | Your answer in the setup |
| `HEARTBEAT_URL` | recommended | Ping URL of the `engine` check | healthchecks.io |
| `BACKUP_HEARTBEAT_URL` | recommended | Ping URL of the `backup` check | healthchecks.io |

`docker-compose.yml` sets `FLASK_ENV`, `FLASK_DEBUG`, `TRADING_MODE`, `ENGINE_MODE`,
`DATABASE_URL`, `REDIS_URL`, `LOG_TO_FILE` and `ENGINE_ALIVE_FILE` itself, so values for them
in `.env` are ignored, and the setup drops them. Leave `API_BIND` out, so that the API listens
on this machine only. The risk limits (`RISK_*`, `SIZING_*`) are optional. Their defaults are
in the README's risk controls and in [`.env.example`](../.env.example), and the setup keeps any
you set.

## After the setup

1. **Open the dashboard.** On your phone, with Tailscale connected, open the address the setup
   printed and sign in. Tick **Stay signed in on this device**, and add the page to the home
   screen. Try it over mobile data, with Wi-Fi off.
2. **Start trading.** Send `/start_engine` to the bot, then the `/confirm <code>` it answers
   with, or press **Start engine** in the dashboard. `/status` shows `running` once the engine
   has loaded the account and the price history. Until trading runs, the healthchecks.io
   `engine` check stays silent and reports it after its grace time: pause the check if you start
   later.
3. **Test a restart.** Run `sudo reboot`. The "Engine process started … trading waits" alert
   arrives, `/status` says `stopped`, and `/start_engine` starts trading again.
4. **Keep a copy of the backups.** After the first day, `sudo ./setup.sh --check` shows
   `newest backup … hours old`. Copy `./backups` to another machine from time to time: a backup
   on the same disk is lost with the disk.

## Day to day

| You want to | Telegram | Dashboard |
|---|---|---|
| See how it is doing | `/status`, `/positions` | Account, Equity, Positions |
| Start after a restart | `/start_engine` + `/confirm` | **Start engine** |
| Lift a halt | `/resume` + `/confirm` | **Resume after halt** |
| Stop everything now | `/halt` or `/flatten` + `/confirm` | **Emergency stop** |

Pause the `engine` check on healthchecks.io before you stop trading on purpose.

To update to a new version, do it outside US market hours. The engine restarts too, and then
waits for you to start trading:

```bash
cd ~/Nocturna-Trading-System
git pull
sudo ./setup.sh
```

[RUNBOOK.md](RUNBOOK.md) covers what each alert means, incidents and restores. Its commands
are written without `sudo`, so add it in front of each `docker` command.
