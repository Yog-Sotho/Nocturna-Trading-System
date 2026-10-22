# Running Nocturna on your home server

How to run the trading system around the clock on a headless Ubuntu machine at home. You
reach the dashboard from your phone through Tailscale, and alerts and commands come through
Telegram. Paper trading only.

**Check first, then act.** The guide does not assume what your machine has. Step 1 runs a
check that reports what is already installed and configured, and what is missing:

```bash
python3 ops/homeserver.py check
```

- Each finding is marked `ok`, `WARN` or `FIX`. Those that need action name the step of this
  guide that deals with them.
- The check ends with a numbered to-do list. Skip every step it does not mention.
- Run the check again after each step.

The check changes nothing. To test your settings, it starts one throwaway container and
removes it again. It never shows a password, key or token, so you can paste its whole report
into a chat when you need help.

What you end up with:

- **The stack restarts itself.** It comes back by itself after a crash, a reboot or a power
  cut.
- **Trading does not restart itself.** After a restart the engine waits for you. Telegram
  tells you so, and you start trading again with `/start_engine` or the dashboard.
- **A private dashboard.** It is at `https://<machine>.<tailnet>.ts.net`, and only your own
  devices can reach it. Nothing is opened to the internet, and your router stays as it is.
- **One Telegram chat** with your own bot, for alerts and commands.
- **A warning when the machine goes silent.** healthchecks.io messages you after a power cut
  or a dead internet line, which the machine cannot report itself.

[RUNBOOK.md](RUNBOOK.md) covers what each alert means, upgrades, incidents and restores.

## What you need

All of these are free:

| Account | What it is for | Where |
|---|---|---|
| Alpaca, paper trading | The play-money account the engine trades | [app.alpaca.markets](https://app.alpaca.markets) |
| Telegram | Alerts and commands on your phone | The Telegram app |
| Tailscale | The dashboard on your phone, from anywhere | [tailscale.com](https://tailscale.com); you sign in with an account you already have, such as Google or Microsoft |
| healthchecks.io | A message when the machine goes silent (optional, recommended) | [healthchecks.io](https://healthchecks.io) |

## 1. Get the code and run the check

On the server, first check that Python 3 and git are there:

```bash
python3 --version && git --version
```

If either answers `command not found`, install them:

```bash
sudo apt update && sudo apt install -y python3 git
```

If you cloned the repository before, go into its folder and update it:

```bash
cd ~/Nocturna-Trading-System && git pull
```

Otherwise, get it now:

```bash
cd ~ && git clone https://github.com/Yog-Sotho/Nocturna-Trading-System.git && cd Nocturna-Trading-System
```

Then run the check from that folder:

```bash
python3 ops/homeserver.py check
```

All the commands below are run from this folder.

## 2. Machine basics

Do only what the check flagged.

- **"clock not synchronized"**: sign-ins and market times depend on the clock. Turn on
  network time:

  ```bash
  sudo timedatectl set-ntp true
  ```

  `timedatectl` then shows `System clock synchronized: yes` within a few minutes.
- **"git is not installed" or "curl is not installed"**: `sudo apt install -y git curl`.
- **"laptop: closing the lid suspends it"**: create the file
  `/etc/systemd/logind.conf.d/60-lid.conf` (`sudo mkdir -p /etc/systemd/logind.conf.d`,
  then `sudo nano /etc/systemd/logind.conf.d/60-lid.conf`) with these lines, then reboot:

  ```ini
  [Login]
  HandleLidSwitch=ignore
  HandleLidSwitchExternalPower=ignore
  ```

- **A setting no program can check**: in the BIOS/UEFI setup, turn on the option that
  powers the machine on when power comes back. It is often called "Restore on AC power
  loss" or "AC back". Without it, a power cut keeps the machine off until you press the
  button.

## 3. Docker

Skip this step if the check shows these four lines under Docker:
- `Docker Engine … answers this user`
- `Docker starts at boot`
- `Docker Compose 2` or later
- no `snap store` warning

Do what matches the finding:

- **"Docker is not installed"**: install it from Docker's own repository, which supports
  Ubuntu 26.04 ([Docker's instructions](https://docs.docker.com/engine/install/ubuntu/)).

  ```bash
  sudo apt update
  sudo apt install -y ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
  Types: deb
  URIs: https://download.docker.com/linux/ubuntu
  Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
  Components: stable
  Architectures: $(dpkg --print-architecture)
  Signed-By: /etc/apt/keyrings/docker.asc
  EOF
  sudo apt update
  sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  ```

  Then let your user run Docker without `sudo`. The `docker` group gives root-level control
  of this machine, so add only your own account:

  ```bash
  sudo usermod -aG docker $USER
  ```

  Log out and back in (or reconnect over SSH) for the group to apply.
- **"your user cannot use Docker"**: run the `usermod` line above, then log out and back
  in.
- **"Docker is installed but not running"** or **"does not start at boot"**:

  ```bash
  sudo systemctl enable --now docker
  ```

- **"the Docker Compose v2 plugin is missing"**: if you use Docker's packages, run
  `sudo apt install -y docker-compose-plugin`. If you use Ubuntu's `docker.io`, run
  `sudo apt install -y docker-compose-v2`.
- **"Docker comes from the snap store"**: this guide uses Docker's apt packages. If the
  stack later cannot write `./backups`, remove the snap (`sudo snap remove docker`) and
  install from the repository as above.

## 4. Tailscale

Tailscale gives the dashboard a private HTTPS address that only your own devices can open.

Skip this step if the check shows these lines under Tailscale:
- `connected as …`
- `serves the dashboard at …`
- no `FIX` or `WARN` line

1. **"Tailscale is not installed"**: install it on the server
   ([Tailscale's instructions](https://tailscale.com/kb/1031/install-linux)):

   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   ```

2. **"not connected"**: run `sudo tailscale up`. It prints a link: open it on your phone and
   sign in.
3. On your phone, install the Tailscale app from the Play Store and sign in with the same
   account.
4. **"MagicDNS is off" or "HTTPS certificates are off"**: in the
   [admin console](https://login.tailscale.com/admin), open the **DNS** page.
   - Enable **MagicDNS** if it is off.
   - Then, under **HTTPS Certificates**, select **Enable HTTPS**.
   - Machine names and your tailnet's name are then published in the public certificate
     log, so keep anything private out of the server's name.
5. **"this machine's key expires"**: on the **Machines** page of the admin console, open the
   menu at the right of the server's row and choose **Disable Key Expiry**. Otherwise the
   server drops off your tailnet after 180 days.
6. **"does not serve the dashboard yet"**:

   ```bash
   sudo tailscale serve --bg 5000
   ```

   This forwards `https://<machine>.<tailnet>.ts.net` to the API on this machine. It
   persists across reboots. It shows an error page until the stack runs in step 7.

## 5. The values only you can provide

You need two of these, and the third is recommended. The next step asks for each of them
and tests it as you enter it, so collect them first.

**Alpaca paper keys.**
1. Sign in at [app.alpaca.markets](https://app.alpaca.markets).
2. Make sure the account switcher at the top left shows your **Paper** account, not Live.
3. On the paper account's home page, find the **API Keys** box and generate the keys.
4. Copy both the **Key ID** and the **Secret Key** at once, because the secret is shown only
   once. Paper Key IDs normally start with `PK`.

Generating new keys disables the previous pair, wherever it is still in use. If you still
have a working pair, use it.

**A Telegram bot.**
1. In Telegram, search for **@BotFather**, Telegram's own bot for creating bots, and open it.
2. Tap **Start**, then send `/newbot`.
3. Give the bot a display name.
4. Give it a username: 5 to 32 letters, digits or underscores, ending in `bot`, for example
   `nocturna_alerts_bot`.
5. BotFather answers with the bot's token, which looks like
   `4839574812:AAFD39kkdpWt3ywyRZergyOLMaJhac60qc`. Anyone with the token controls the bot,
   so keep it private.

You do not need to look up your chat id: the next step finds it.

**healthchecks.io checks** (recommended).
1. Sign up at [healthchecks.io](https://healthchecks.io).
2. Create a check named `engine` with **Period 1 minute** and **Grace Time 5 minutes**. The
   engine pings it every minute while it trades. After a restart, it stays silent until you
   start trading, so the check also reminds you that trading is stopped.
3. Create a check named `backup` with **Period 1 day** and **Grace Time 2 hours**.
4. Each check's page shows its ping URL, `https://hc-ping.com/<uuid>`.
5. To be told on Telegram, open **Integrations**, choose **Telegram**, and send `/start` to
   **@HealthchecksBot**. Open the link it answers with, choose your project, then click
   **Connect Telegram**.

## 6. Create `.env`

`.env` holds the settings and secrets of the stack. Create it with:

```bash
python3 ops/homeserver.py env
```

It asks only for what it cannot work out itself:

- **It generates the five passwords and secrets.** It keeps the ones an existing `.env`
  already has. It never replaces `DB_PASSWORD` once the database exists, because PostgreSQL
  keeps the password it was created with.
- **It asks for the Alpaca Key ID and Secret Key.** It hides them as you type or paste, and
  signs in to your paper account to check them.
- **It asks for the Telegram bot token, then finds your chat.** You tap **Start** in your
  bot's chat when it tells you to. It then sends a test message there and asks whether the
  bot may also take commands.
- **It reads the dashboard's address from Tailscale.** If Tailscale is not set up yet, it
  uses a temporary value. Run it again after step 4.
- **It asks for the two healthchecks.io ping URLs.** Press Enter to skip them, or to keep
  what is set.

It writes `.env` so that only your user can read it, and copies an existing one to
`.env.backup-<date>` first. Settings it does not manage, such as risk limits you changed,
are kept. Run it again whenever you want to change something, then run the check.

### What goes in `.env`

If you prefer to edit it by hand (`nano .env`): one `NAME=value` per line, with no spaces
around `=` and no quotes needed.

| Setting | Needed | What it is | Where it comes from |
|---|---|---|---|
| `FLASK_SECRET_KEY` | yes | Signs the server's cookies | Generated (64 random hex characters, `openssl rand -hex 32`) |
| `JWT_SECRET_KEY` | yes | Signs sign-in tokens; must differ from `FLASK_SECRET_KEY` | Generated, as above |
| `API_KEY_PEPPER` | yes | Mixed into stored API key hashes; changing it voids every API key | Generated, as above |
| `DB_PASSWORD` | yes | The database password, fixed when the database is first created; letters and digits only | Generated (`openssl rand -hex 16`); never change it afterwards |
| `REDIS_PASSWORD` | yes | Password of the rate-limit store; letters and digits only | Generated, as above |
| `ALPACA_API_KEY` | yes | The Key ID of your Alpaca **paper** account (starts with `PK`) | Step 5 |
| `ALPACA_SECRET_KEY` | yes | Its Secret Key | Step 5 |
| `ALLOWED_ORIGINS` | yes | The dashboard's address, `https://<machine>.<tailnet>.ts.net`, with no trailing `/`. Only requests from other websites are checked against it, but the stack refuses to start without it. | `tailscale status` shows the name; `env` reads it for you |
| `TRUSTED_PROXY_COUNT` | yes | `1`: Tailscale Serve forwards visitors to the API. With `0`, every visitor looks the same, and a few failed logins lock everyone out. | Always `1` in this setup |
| `ENGINE_AUTOSTART` | no | `false` (the default): after any restart, trading waits for you | Your choice; this guide uses `false` |
| `ALERT_TELEGRAM_BOT_TOKEN` | recommended | Your bot's token | Step 5, from @BotFather |
| `ALERT_TELEGRAM_CHAT_ID` | recommended | The id of your private chat with the bot (a number) | Found by `env` after you tap Start in the bot's chat |
| `TELEGRAM_COMMANDS` | recommended | `true`: the bot also takes `/status`, `/start_engine`, `/halt` and the other commands, each change confirmed with a code | Your choice |
| `HEARTBEAT_URL` | recommended | Ping URL of the `engine` check | Step 5, from healthchecks.io |
| `BACKUP_HEARTBEAT_URL` | recommended | Ping URL of the `backup` check | Step 5, from healthchecks.io |

`docker-compose.yml` sets `FLASK_ENV`, `FLASK_DEBUG`, `TRADING_MODE`, `ENGINE_MODE`,
`DATABASE_URL`, `REDIS_URL`, `LOG_TO_FILE` and `ENGINE_ALIVE_FILE` itself, so values for them
in `.env` are ignored. Leave `API_BIND` out: the API then listens on this machine only, and
Tailscale forwards to it. The risk limits (`RISK_*`, `SIZING_*`) are optional; their defaults
and meaning are in the README's risk controls and in [`.env.example`](../.env.example).

## 7. Start the stack

```bash
docker compose up -d --build
```

The first build takes several minutes. Then run the check again. Under **Stack**, wait until
`api`, `engine`, `postgres` and `redis` show `running, healthy`. If one does not, look at
what it says:

```bash
docker compose logs <service> --tail 50
```

Create your account on the dashboard. Use your own username and email:

```bash
docker compose exec api flask --app src.main create-admin --username ops --email you@example.com
```

It asks for a password twice. The password needs at least 12 characters, with upper case,
lower case, a digit and a symbol. The username may use letters, digits and underscores.

Test the alerts:

```bash
docker compose exec api flask --app src.main send-test-alert
```

It should print `telegram: sent`, and the message should reach your phone. In Telegram, send
`/status` to your bot: it answers with the engine's state (`stopped` until you start it).

## 8. The dashboard on your phone

With the Tailscale app connected on the phone, open the address the check shows after
`serves the dashboard at`, and sign in with the account from step 7. Tick **Stay signed in on
this device**, and add the page to the home screen from the browser's menu.

Try it with Wi-Fi off, over mobile data.

## 9. Start trading

Send `/start_engine` to the bot, then the `/confirm <code>` it answers with. Or press **Start
engine** in the dashboard. `/status` shows `running` once the engine has loaded the account
and the price history.

## Checklist

- [ ] `python3 ops/homeserver.py check` ends with `Nothing to fix`. It may still list
      warnings: read them, and accept those that do not apply to you.
- [ ] `send-test-alert` reached your phone, and `/status` answers.
- [ ] The dashboard opens over mobile data, and you can sign in.
- [ ] Both healthchecks.io checks are up and connected to Telegram.
- [ ] Restart test: `sudo reboot`. The "Engine process started … trading waits" alert
      arrives, `/status` says `stopped`, and `/start_engine` starts trading again.
- [ ] After the first day, the check shows `newest backup … hours old`. Copy `./backups` to
      another machine from time to time: a backup on the same disk is lost with the disk.

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
docker compose up -d --build
python3 ops/homeserver.py check
```
