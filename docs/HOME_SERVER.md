# Running Nocturna on a home server

Step by step for one setup: a headless Ubuntu 26.04 machine at home, an
Android phone, Tailscale to reach the dashboard from anywhere, and Telegram
for alerts and commands. Paper trading only.

What you end up with:

- The stack runs around the clock and comes back by itself after a crash, a
  reboot or a power cut. **Trading does not**: after any restart the engine
  waits for you (`ENGINE_AUTOSTART=false`). Telegram tells you ("Engine
  process started … trading waits until you start it"), and you start it
  with `/start_engine` in Telegram or **Start engine** in the dashboard.
- The dashboard at `https://<machine>.<tailnet>.ts.net`, reachable only from
  your own devices through Tailscale. Nothing on your home network is opened
  to the internet, and no router change is needed.
- Alerts and commands in one private Telegram chat with your bot.
- healthchecks.io messages you on Telegram when the machine goes silent (a
  power cut, a dead internet line), which the machine cannot report itself.

[RUNBOOK.md](RUNBOOK.md) explains every alert, upgrades, incidents and
restores. This page is the setup.

## 1. Prepare the machine

```bash
sudo apt update && sudo apt upgrade
timedatectl            # "System clock synchronized: yes" is needed for logins and market timing
```

If the machine is a laptop, keep it running with the lid closed: create
`/etc/systemd/logind.conf.d/60-lid.conf` with

```ini
[Login]
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
```

and reboot. In the BIOS/UEFI, turn on the setting that powers the machine on
again after a power cut (often called "Restore on AC power loss").

## 2. Install Docker

From Docker's own repository ([docs](https://docs.docker.com/engine/install/ubuntu/);
Ubuntu's `docker.io` package is a different build). Remove conflicting
packages, if any:

```bash
sudo apt remove $(dpkg --get-selections docker.io docker-compose docker-compose-v2 docker-doc docker-buildx podman-docker containerd runc | cut -f1)
```

Add the repository and install:

```bash
sudo apt update
sudo apt install ca-certificates curl
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
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo docker run hello-world
```

On Ubuntu, Docker starts at boot by default. To run `docker` without `sudo`,
add yourself to the `docker` group and log in again (the group is
equivalent to root on this machine):

```bash
sudo usermod -aG docker $USER
```

## 3. Get the code and configure it

```bash
git clone https://github.com/Yog-Sotho/Nocturna-Trading-System.git
cd Nocturna-Trading-System
cp .env.example .env
chmod 600 .env
```

Generate the secrets and paste each line over the matching line in `.env`
(`nano .env`):

```bash
for name in FLASK_SECRET_KEY JWT_SECRET_KEY API_KEY_PEPPER METRICS_TOKEN; do echo "$name=$(openssl rand -hex 32)"; done
for name in DB_PASSWORD REDIS_PASSWORD; do echo "$name=$(openssl rand -hex 16)"; done
```

Then set, in `.env`:

```bash
ALPACA_API_KEY=<paper key id>
ALPACA_SECRET_KEY=<paper secret>
ENGINE_AUTOSTART=false

# Telegram: create the bot and find your chat id as in RUNBOOK.md → "Telegram"
ALERT_TELEGRAM_BOT_TOKEN=<bot token>
ALERT_TELEGRAM_CHAT_ID=<your chat id>
TELEGRAM_COMMANDS=true

# Tailscale Serve (step 5) sits in front of the API and sets X-Forwarded-For
TRUSTED_PROXY_COUNT=1
# The dashboard's address, which step 5 prints; until then keep the example value
ALLOWED_ORIGINS=https://<machine>.<tailnet>.ts.net

# healthchecks.io (step 6); empty until then
HEARTBEAT_URL=
BACKUP_HEARTBEAT_URL=
```

Leave `API_BIND` unset: the API listens on the machine itself only, and
Tailscale Serve forwards to it.

## 4. Start the stack

```bash
docker compose up -d --build
docker compose ps                     # wait until every service is healthy
docker compose exec api flask --app src.main create-admin --username ops --email you@example.com
docker compose exec api flask --app src.main send-test-alert
```

`send-test-alert` should print `telegram: sent`, and the test alert should
reach your phone. In Telegram, send `/status` to your bot: it answers with
the engine's state (`stopped` until you start it).

## 5. Tailscale: the dashboard on your phone, from anywhere

1. On the server ([docs](https://tailscale.com/kb/1031/install-linux)):

   ```bash
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   ```

   `tailscale up` prints a login link: open it on your phone and sign in.
2. On the phone, install the Tailscale app from the Play Store and sign in
   with the same account.
3. In the [admin console](https://login.tailscale.com/admin), on the **DNS**
   page: enable **MagicDNS**, then under **HTTPS Certificates** select
   **Enable HTTPS**. Machine names and the tailnet name are then published in
   a public certificate log, so keep anything private out of the machine's
   name.
4. On the **Machines** page, open the server's menu and select **Disable key
   expiry**; otherwise it drops off your tailnet after 180 days.
5. Serve the API over HTTPS inside your tailnet, in the background and
   across reboots:

   ```bash
   sudo tailscale serve --bg 5000
   ```

   It prints the address, `https://<machine>.<tailnet>.ts.net`. Put it in
   `ALLOWED_ORIGINS` in `.env`, then apply it with `docker compose up -d api`.
6. On the phone, with Tailscale connected, open that address and sign in as
   `ops`. Tick **Stay signed in on this device**, and add the page to the
   home screen from the browser's menu.

Test it with Wi-Fi off, over mobile data. `tailscale serve status` shows what
is served; `sudo tailscale serve reset` stops it.

## 6. healthchecks.io: an alert when the machine goes silent

1. Sign up at [healthchecks.io](https://healthchecks.io) (the free plan
   monitors 20 checks).
2. Add a check for the engine: **period 1 minute, grace 5 minutes**. The
   engine pings it every minute while it runs and pings `/fail` if it fails.
   After a restart it stays silent until you start trading, so the check
   reminds you that trading is stopped.
3. Add a check for the backups: **period 1 day, grace 2 hours**.
4. Under **Integrations**, add Telegram: send `/start` to `@HealthchecksBot`,
   open the link it answers with, and connect it to your project.
5. Copy the two ping URLs into `HEARTBEAT_URL` and `BACKUP_HEARTBEAT_URL`
   in `.env`, then `docker compose up -d`.

Pause the engine check on healthchecks.io before you stop the engine on
purpose.

## 7. Start trading

Send `/start_engine` to the bot, then the `/confirm <code>` it answers with,
or press **Start engine** in the dashboard. `/status` shows `running` once
the engine has loaded the account and the history.

## Checklist

- [ ] `docker compose ps`: every service healthy.
- [ ] `send-test-alert` reached your phone; `/status` answers.
- [ ] The dashboard opens over mobile data with Tailscale on, and you can sign in.
- [ ] Both healthchecks.io checks are up, and connected to Telegram.
- [ ] Restart test: `sudo reboot`. The "Engine process started … trading
      waits" alert arrives, `/status` says `stopped`, and `/start_engine`
      starts it again.
- [ ] Backups: after the first day, `ls backups/` shows a dump. Copy
      `./backups` to another machine from time to time: a backup on the
      same disk is lost with the disk.

## Day to day

| You want to | Telegram | Dashboard |
|---|---|---|
| See how it is doing | `/status`, `/positions` | Account, Equity, Positions |
| Start after a restart | `/start_engine` + `/confirm` | **Start engine** |
| Lift a halt | `/resume` + `/confirm` | **Resume after halt** |
| Stop everything now | `/halt` or `/flatten` + `/confirm` | **Emergency stop** |

Updating to a new version, outside US market hours:

```bash
cd Nocturna-Trading-System
git pull
docker compose up -d --build
```

The engine restarts too, and then waits for you to start it.
