#!/usr/bin/env python3
"""
Nocturna setup. Run it with `sudo ./setup.sh` (docs/HOME_SERVER.md).

    sudo ./setup.sh            # check this machine, show what will change, ask, then set it up
    sudo ./setup.sh --check    # report only: what is there, what is missing, what would change

It looks at everything Nocturna needs on this machine, lists the changes it
would make and asks before making any. Then, step by step, verifying each:
it installs what is missing (base tools, Docker, Docker Compose and buildx,
Tailscale), serves the dashboard on a private Tailscale HTTPS port, writes
.env (asking only for what you must supply, and testing each value), builds
and starts the stack, creates your dashboard account and sends a test alert.
Run it again at any time: what is already done is kept.

It never adds anyone to the docker group, never removes a package, never
changes what Tailscale already serves or its Funnel, never starts trading,
and never prints a secret. Standard library only, Python 3.10 or later.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUIDE = "docs/HOME_SERVER.md"
ALPACA_PAPER = "https://paper-api.alpaca.markets"
TELEGRAM = "https://api.telegram.org"
DEFAULT_API_PORT = 5000
IMAGE = "nocturna:latest"
#: Funnel can only expose these ports; the dashboard never uses one
FUNNEL_PORTS = (443, 8443, 10000)
#: Where the dashboard's private Tailscale HTTPS port is looked for
SERVE_PORTS = range(9443, 9500)
API_PORTS = range(5000, 5100)
#: Hosts the setup and the stack must reach
HOSTS = {
    "registry-1.docker.io": "Docker images (the base images of the stack)",
    "pypi.org": "Python packages for the image",
    "files.pythonhosted.org": "Python packages for the image",
    "paper-api.alpaca.markets": "the Alpaca paper account",
    "data.alpaca.markets": "Alpaca market data",
    "api.telegram.org": "Telegram alerts and commands",
    "hc-ping.com": "healthchecks.io pings",
}


# -- running commands and HTTP ------------------------------------------------------------

@dataclass(frozen=True)
class Result:
    code: int
    out: str = ""
    err: str = ""


Runner = Callable[..., Result]


def run(args: Sequence[str], *, stream: bool = False, timeout: float = 120, cwd: Path = ROOT,
        env: dict[str, str] | None = None) -> Result:
    """
    Run a command. With ``stream`` its output goes straight to the terminal (and
    it can ask questions there). A missing program gives code 127, a hung one 124.
    """
    environment = {**os.environ, **(env or {})}
    try:
        if stream:
            return Result(subprocess.run(list(args), cwd=cwd, env=environment, check=False).returncode)
        done = subprocess.run(list(args), cwd=cwd, env=environment, capture_output=True, text=True,
                              timeout=timeout, check=False, stdin=subprocess.DEVNULL)
    except FileNotFoundError:
        return Result(127, "", f"{args[0]}: not found")
    except subprocess.TimeoutExpired:
        return Result(124, "", f"{args[0]}: timed out")
    return Result(done.returncode, done.stdout, done.stderr)


@dataclass(frozen=True)
class Response:
    status: int  # 0 when there was no answer
    body: bytes = b""
    error: str = ""

    def json(self) -> dict:
        try:
            value = json.loads(self.body)
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}


Http = Callable[..., Response]


def http(method: str, url: str, headers: dict[str, str] | None = None, data: dict | None = None,
         timeout: float = 15) -> Response:
    """One request. Errors carry the reason only: URLs can hold a token."""
    body = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(url, data=body, method=method, headers=dict(headers or {}))
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        # The URLs are this file's own constants, or the local API
        with urllib.request.urlopen(request, timeout=timeout) as answer:  # nosec B310
            return Response(answer.status, answer.read())
    except urllib.error.HTTPError as e:
        return Response(e.code, e.read(), str(e.reason))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return Response(0, b"", str(getattr(e, "reason", e)))


# -- .env ----------------------------------------------------------------------------------

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse_env(text: str) -> list[tuple[str, str, str]]:
    """(name, value, the line as written) for each setting, as Docker Compose reads them."""
    entries = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            continue
        name, raw = match.groups()
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
            value = raw[1:-1]
        else:
            value = re.split(r"\s+#", raw, maxsplit=1)[0]
        entries.append((name, value, line))
    return entries


def env_values(path: Path) -> dict[str, str]:
    return {name: value for name, value, _ in parse_env(path.read_text())} if path.exists() else {}


def env_line(name: str, value: str) -> str:
    """A line Compose reads back as exactly this value (single quotes are literal)."""
    if re.fullmatch(r"[A-Za-z0-9_:/.,@%+=-]*", value):
        return f"{name}={value}"
    if "'" in value or "\n" in value:
        raise ValueError(f"{name}: the value cannot contain quotes or line breaks")
    return f"{name}='{value}'"


@dataclass(frozen=True)
class Var:
    name: str
    need: str  # required, recommended or optional
    what: str


GENERATED = ("FLASK_SECRET_KEY", "JWT_SECRET_KEY", "API_KEY_PEPPER", "DB_PASSWORD", "REDIS_PASSWORD")

#: What this setup puts in .env; docs/HOME_SERVER.md explains each one
VARS = (
    Var("FLASK_SECRET_KEY", "required", "signs the server's cookies"),
    Var("JWT_SECRET_KEY", "required", "signs sign-in tokens; differs from FLASK_SECRET_KEY"),
    Var("API_KEY_PEPPER", "required", "mixed into stored API key hashes"),
    Var("DB_PASSWORD", "required", "the PostgreSQL password, fixed when the database is created"),
    Var("REDIS_PASSWORD", "required", "the Redis password"),
    Var("ALPACA_API_KEY", "required", "the Key ID of your Alpaca paper account"),
    Var("ALPACA_SECRET_KEY", "required", "its Secret Key"),
    Var("ALLOWED_ORIGINS", "required", "the dashboard's address"),
    Var("TRUSTED_PROXY_COUNT", "required", "1: Tailscale Serve forwards to the API"),
    Var("API_PORT", "optional", "the API's port on this machine (127.0.0.1 only)"),
    Var("ENGINE_AUTOSTART", "optional", "false: after a restart trading waits for you"),
    Var("ALERT_TELEGRAM_BOT_TOKEN", "recommended", "your bot's token"),
    Var("ALERT_TELEGRAM_CHAT_ID", "recommended", "your private chat with the bot"),
    Var("TELEGRAM_COMMANDS", "recommended", "true: the bot also takes commands"),
    Var("HEARTBEAT_URL", "recommended", "ping URL of the engine check"),
    Var("BACKUP_HEARTBEAT_URL", "recommended", "ping URL of the backup check"),
)

#: docker-compose.yml sets these itself: a value in .env is ignored
STACK_SETS = ("FLASK_ENV", "FLASK_DEBUG", "TRADING_MODE", "ENGINE_MODE", "DATABASE_URL", "REDIS_URL",
              "LOG_TO_FILE", "ENGINE_ALIVE_FILE")

_PLACEHOLDER = re.compile(r"your-|yourdomain|example\.com|<|>|changeme", re.IGNORECASE)
_URL_SAFE = re.compile(r"[A-Za-z0-9]+")
_TOKEN = re.compile(r"\d+:[A-Za-z0-9_-]{20,}")
_CHAT = re.compile(r"-?\d+")
_ORIGIN = re.compile(r"https?://[^/\s,]+")
_SECRET_NAME = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|PEPPER|HEARTBEAT|NTFY|WEBHOOK|DATABASE_URL|REDIS_URL")
_USERNAME = re.compile(r"[A-Za-z0-9_]{3,50}")
_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PASSWORD_RULES = ("at least 12 characters, with an upper-case letter, a lower-case letter, a digit and a symbol "
                  "(any character that is not a letter, a digit or a space, such as ! - _ # or @)")


def password_problem(password: str) -> str:
    """What the password lacks, or "": the rules of src/utils/validators.py (RegisterSchema)."""
    for broken, problem in ((len(password) < 12, f"it has {len(password)} characters, not 12 or more"),
                            (len(password) > 128, "it has more than 128 characters"),
                            (not re.search(r"[A-Z]", password), "it has no upper-case letter"),
                            (not re.search(r"[a-z]", password), "it has no lower-case letter"),
                            (not re.search(r"\d", password), "it has no digit"),
                            (not re.search(r"[^A-Za-z0-9\s]", password), "it has no symbol")):
        if broken:
            return problem
    return ""


def is_placeholder(value: str) -> bool:
    return bool(_PLACEHOLDER.search(value))


def secret_values(values: dict[str, str]) -> set[str]:
    """The values never to show: secrets, and URLs that work as one (a ping URL, an ntfy topic)."""
    return {value for name, value in values.items() if _SECRET_NAME.search(name) and len(value) >= 4}


def env_problems(values: dict[str, str]) -> dict[str, tuple[str, str]]:
    """Problems per setting: name -> (fix or warn, message). Messages never contain the value."""
    found: dict[str, tuple[str, str]] = {}

    def fix(name: str, message: str) -> None:
        found[name] = ("fix", message)

    def warn(name: str, message: str) -> None:
        found.setdefault(name, ("warn", message))

    for var in VARS:
        value = values.get(var.name, "").strip()
        if not value:
            if var.need == "required" and var.name != "TRUSTED_PROXY_COUNT":
                fix(var.name, "missing")
            elif var.need == "recommended" and var.name != "TELEGRAM_COMMANDS":
                warn(var.name, "not set")
        elif is_placeholder(value):
            fix(var.name, "still the example value")
    for name in ("FLASK_SECRET_KEY", "JWT_SECRET_KEY", "API_KEY_PEPPER"):
        if name not in found and len(values[name]) < 32:
            fix(name, "shorter than 32 characters")
    if "JWT_SECRET_KEY" not in found and values["JWT_SECRET_KEY"] == values.get("FLASK_SECRET_KEY"):
        fix("JWT_SECRET_KEY", "the same as FLASK_SECRET_KEY")
    for name in ("DB_PASSWORD", "REDIS_PASSWORD"):
        if name not in found and not _URL_SAFE.fullmatch(values[name]):
            fix(name, "use letters and digits only: it goes into a URL")
    key = values.get("ALPACA_API_KEY", "")
    if "ALPACA_API_KEY" not in found and not key.upper().startswith("PK"):
        warn("ALPACA_API_KEY", "paper Key IDs start with PK: is this a live key?")
    if "ALLOWED_ORIGINS" not in found:
        origins = [o.strip() for o in values["ALLOWED_ORIGINS"].split(",") if o.strip()]
        if not origins or not all(_ORIGIN.fullmatch(o) for o in origins):
            fix("ALLOWED_ORIGINS", "not an address like https://name.tailnet.ts.net:9443 (no path, no trailing /)")
    if values.get("TRUSTED_PROXY_COUNT", "").strip() != "1":
        warn("TRUSTED_PROXY_COUNT", "should be 1 behind Tailscale Serve, or every visitor looks the same "
                                    "and a few failed logins lock everyone out")
    port = values.get("API_PORT", "").strip()
    if port and not (port.isdigit() and 1024 <= int(port) <= 65535):
        fix("API_PORT", "must be a port number from 1024 to 65535")
    token, chat = values.get("ALERT_TELEGRAM_BOT_TOKEN", "").strip(), values.get("ALERT_TELEGRAM_CHAT_ID", "").strip()
    if token and "ALERT_TELEGRAM_BOT_TOKEN" not in found and not _TOKEN.fullmatch(token):
        fix("ALERT_TELEGRAM_BOT_TOKEN", "not a bot token (digits, a colon, then letters and digits)")
    if chat and "ALERT_TELEGRAM_CHAT_ID" not in found and not _CHAT.fullmatch(chat):
        fix("ALERT_TELEGRAM_CHAT_ID", "not a chat id (a number)")
    if bool(token) != bool(chat):
        fix("ALERT_TELEGRAM_CHAT_ID" if token else "ALERT_TELEGRAM_BOT_TOKEN",
            "set the bot token and the chat id together, or neither")
    commands = values.get("TELEGRAM_COMMANDS", "").strip().lower()
    if commands not in ("", "true", "false"):
        fix("TELEGRAM_COMMANDS", "must be true or false")
    elif commands == "true" and not (token and chat):
        fix("TELEGRAM_COMMANDS", "needs the bot token and the chat id")
    for name in ("HEARTBEAT_URL", "BACKUP_HEARTBEAT_URL"):
        url = values.get(name, "").strip()
        if url and name not in found:
            if not url.startswith("https://"):
                fix(name, "must start with https://")
            elif not url.startswith("https://hc-ping.com/"):
                warn(name, "healthchecks.io ping URLs start with https://hc-ping.com/")
    if values.get("HEARTBEAT_URL") and values.get("HEARTBEAT_URL") == values.get("BACKUP_HEARTBEAT_URL"):
        fix("BACKUP_HEARTBEAT_URL", "the same as HEARTBEAT_URL: give each its own check")
    autostart = values.get("ENGINE_AUTOSTART", "").strip().lower()
    if autostart not in ("", "true", "false"):
        fix("ENGINE_AUTOSTART", "must be true or false")
    elif autostart == "true":
        warn("ENGINE_AUTOSTART", "true: trading restarts by itself after a crash or a reboot")
    return found


# -- what is on this machine ----------------------------------------------------------------

@dataclass
class Finding:
    level: str  # ok, warn, fix, info
    area: str
    text: str


@dataclass(frozen=True)
class Listener:
    address: str
    port: int
    program: str


def parse_listeners(text: str) -> list[Listener]:
    """The TCP listeners in `ss -ltnpH` output."""
    found = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or ":" not in parts[3]:
            continue
        address, _, port = parts[3].rpartition(":")
        program = re.search(r'users:\(\("([^"]+)"', line)
        if port.isdigit():
            found.append(Listener(address.strip("[]"), int(port), program.group(1) if program else "?"))
    return found


@dataclass
class Machine:
    """What the setup knows about this machine; filled by inspect()."""

    run: Runner
    http: Http
    root: Path = ROOT
    etc: Path = Path("/etc")
    os_release: Path = Path("/etc/os-release")
    findings: list[Finding] = field(default_factory=list)
    hide: set[str] = field(default_factory=set)
    operator: str = ""
    os_id: str = ""
    codename: str = ""
    clock_synced: bool | None = None
    missing_tools: list[str] = field(default_factory=list)
    docker_package: str = ""  # docker-ce, docker.io, snap, other, or "" when Docker is not installed
    conflicts: list[str] = field(default_factory=list)
    docker_running: bool = False
    docker_enabled: bool = False
    compose: bool = False
    buildx: bool = False
    tailscale: bool = False
    ts_state: str = ""
    ts_name: str = ""
    magicdns: bool = False
    certs: bool = False
    key_expiry: str = ""
    serve: dict | None = None  # None: not readable
    dashboard_port: int | None = None  # the Serve port that forwards to the API
    dashboard_public: bool = False
    listeners: list[Listener] = field(default_factory=list)
    api_port: int = DEFAULT_API_PORT
    chosen_api_port: int | None = None  # picked by the setup because the configured one is taken
    api_port_taken_by: str = ""
    env: dict[str, str] = field(default_factory=dict)
    db_volume: bool = False
    services: list[dict] = field(default_factory=list)
    admins: list[str] | None = None  # None: unknown (the API is not running)
    unreachable: list[str] = field(default_factory=list)

    @property
    def env_file(self) -> Path:
        return self.root / ".env"

    def add(self, level: str, area: str, text: str) -> None:
        self.findings.append(Finding(level, area, text))

    @property
    def docker_ok(self) -> bool:
        return self.docker_running and self.compose

    @property
    def origin(self) -> str | None:
        if not (self.ts_name and self.dashboard_port):
            return None
        return f"https://{self.ts_name}" + ("" if self.dashboard_port == 443 else f":{self.dashboard_port}")


def _first_line(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[0] if lines else ""


def _installed(m: Machine, package: str) -> bool:
    return m.run(["dpkg-query", "-W", "-f=${Status}", package]).out.startswith("install ok installed")


def inspect_system(m: Machine, meminfo: Path = Path("/proc/meminfo")) -> None:
    area = "System"
    info = {name: value for name, value, _ in parse_env(m.os_release.read_text())} if m.os_release.exists() else {}
    m.os_id = info.get("ID", "").lower()
    m.codename = info.get("UBUNTU_CODENAME") or info.get("VERSION_CODENAME", "")
    memory = ""
    if meminfo.exists():
        kb = re.search(r"MemTotal:\s+(\d+)", meminfo.read_text())
        if kb:
            gb = int(kb.group(1)) / 1024 ** 2
            memory = f", {gb:.1f} GB RAM"
            if gb < 2:
                m.add("warn", area, f"{gb:.1f} GB RAM: building the image and running the stack may run out")
    free = shutil.disk_usage(m.root).free / 1024 ** 3
    m.add("info", area, f"{info.get('PRETTY_NAME', platform.system())}"
                        f"{' (' + m.codename + ')' if m.codename else ''}, {platform.machine()}, "
                        f"{os.cpu_count()} CPUs{memory}, {free:.0f} GB free")
    if m.os_id != "ubuntu":
        m.add("fix", area, "this setup is for Ubuntu")
    if free < 5:
        m.add("warn", area, f"{free:.1f} GB free disk: the image and the backups need about 5 GB")
    synced = m.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"])
    if synced.code == 0:
        m.clock_synced = synced.out.strip() == "yes"
        m.add("ok" if m.clock_synced else "fix", area,
              "clock synchronized" if m.clock_synced else "clock not synchronized: sign-ins and market times "
                                                         "depend on it")
    else:
        m.add("info", area, "could not read the clock state (timedatectl)")


def inspect_tools(m: Machine) -> None:
    m.missing_tools = [p for p in ("git", "curl", "ca-certificates") if not _installed(m, p)]
    m.add("fix" if m.missing_tools else "ok", "Tools",
          f"missing: {', '.join(m.missing_tools)}" if m.missing_tools else "git, curl and ca-certificates installed")


def inspect_docker(m: Machine) -> None:
    area = "Docker"
    if m.run(["snap", "list", "docker"]).code == 0:
        m.docker_package = "snap"
    elif _installed(m, "docker-ce"):
        m.docker_package = "docker-ce"
    elif _installed(m, "docker.io"):
        m.docker_package = "docker.io"
    elif m.run(["docker", "--version"]).code == 0:
        m.docker_package = "other"
    if not m.docker_package:
        m.conflicts = [p for p in ("docker-compose", "docker-compose-v2", "docker-doc", "podman-docker",
                                   "containerd", "runc") if _installed(m, p)]
        m.add("fix", area, "Docker is not installed" + (
            f" (but these conflicting packages are: {', '.join(m.conflicts)})" if m.conflicts else ""))
        return
    source = {"docker-ce": "Docker's own packages", "docker.io": "Ubuntu's docker.io package",
              "snap": "the snap store", "other": "installed by other means"}[m.docker_package]
    m.add("warn" if m.docker_package == "snap" else "ok", area,
          f"{_first_line(m.run(['docker', '--version']).out) or 'Docker'} ({source})")
    if m.docker_package != "snap":
        m.docker_enabled = m.run(["systemctl", "is-enabled", "docker"]).out.strip() == "enabled"
        m.add("ok" if m.docker_enabled else "fix", area,
              "starts at boot" if m.docker_enabled else "does not start at boot")
    m.docker_running = m.run(["docker", "info", "--format", "{{.ServerVersion}}"]).code == 0
    m.add("ok" if m.docker_running else "fix", area, "the Docker service runs" if m.docker_running
          else "the Docker service is not running")
    version = m.run(["docker", "compose", "version", "--short"])
    major = version.out.strip().lstrip("v").split(".")[0]
    m.compose = version.code == 0 and major.isdigit() and int(major) >= 2
    m.add("ok" if m.compose else "fix", area, f"Docker Compose {version.out.strip().lstrip('v')}" if m.compose
          else "Docker Compose v2 is missing")
    m.buildx = m.run(["docker", "buildx", "version"]).code == 0
    m.add("ok" if m.buildx else "fix", area, "buildx installed (builds the image)" if m.buildx
          else "buildx is missing (Compose needs it to build the image)")
    if m.operator and m.operator != "root":
        groups = m.run(["id", "-nG", m.operator]).out.split()
        if "docker" in groups:
            m.add("warn", area, f"{m.operator} is in the docker group, which equals root without a password: "
                                f"sudo gpasswd -d {m.operator} docker removes it")
        else:
            m.add("ok", area, f"{m.operator} is not in the docker group: Docker needs sudo")


def inspect_tailscale(m: Machine) -> None:
    area = "Tailscale"
    version = m.run(["tailscale", "version"])
    m.tailscale = version.code == 0
    if not m.tailscale:
        m.add("fix", area, "Tailscale is not installed")
        return
    m.add("ok", area, f"Tailscale {_first_line(version.out)}")
    status = m.run(["tailscale", "status", "--json"])
    try:
        state = json.loads(status.out) if status.out.strip() else {}
    except ValueError:
        state = {}
    m.ts_state = state.get("BackendState", "") or ("not running" if "tailscaled" in status.err else "")
    if m.ts_state != "Running":
        m.add("fix", area, f"not connected ({m.ts_state or 'no status'})")
        return
    me = state.get("Self") or {}
    m.ts_name = (me.get("DNSName") or "").rstrip(".")
    m.magicdns = bool((state.get("CurrentTailnet") or {}).get("MagicDNSEnabled"))
    m.certs = bool(state.get("CertDomains"))
    m.key_expiry = (me.get("KeyExpiry") or "")[:10]
    m.add("ok", area, f"connected as {m.ts_name}")
    m.add("ok" if m.magicdns else "fix", area, "MagicDNS on" if m.magicdns else "MagicDNS is off")
    m.add("ok" if m.certs else "fix", area, "HTTPS certificates on" if m.certs else "HTTPS certificates are off")
    if m.key_expiry:
        m.add("warn", area, f"this machine's key expires on {m.key_expiry}: then it drops off your tailnet")
    served = m.run(["tailscale", "serve", "status", "--json"])
    try:
        m.serve = json.loads(served.out) if served.code == 0 and served.out.strip() else ({} if served.code == 0
                                                                                           else None)
    except ValueError:
        m.serve = None
    if m.serve is None:
        m.add("fix", area, "could not read what Tailscale serves")
        return
    funnel = m.serve.get("AllowFunnel") or {}
    for hostport, site in sorted((m.serve.get("Web") or {}).items()):
        for mount, handler in sorted(((site or {}).get("Handlers") or {}).items()):
            target = (handler or {}).get("Proxy") or (handler or {}).get("Path") or "text"
            public = bool(funnel.get(hostport))
            m.add("info", area, f"serves https://{hostport}{mount} -> {target} "
                                f"({'public, Funnel' if public else 'tailnet only'}); left as it is")


def _api_target(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def find_dashboard(serve: dict, api_port: int) -> tuple[int | None, bool]:
    """The Serve port that forwards to the API, and whether Funnel makes it public."""
    pattern = re.compile(rf"^(https?://)?(127\.0\.0\.1|localhost):{api_port}/?$|^{api_port}$")
    funnel = serve.get("AllowFunnel") or {}
    for hostport, site in (serve.get("Web") or {}).items():
        for handler in ((site or {}).get("Handlers") or {}).values():
            if pattern.match((handler or {}).get("Proxy", "")):
                return int(hostport.rsplit(":", 1)[1]), bool(funnel.get(hostport))
    return None, False


def choose_serve_port(serve: dict, listeners: list[Listener]) -> int | None:
    """The first private HTTPS port that nothing uses: Serve's ports, and anything listening on it."""
    used = {int(port) for port in (serve.get("TCP") or {})}
    used |= {int(hp.rsplit(":", 1)[1]) for hp in (serve.get("Web") or {})}
    used |= {listener.port for listener in listeners}
    return next((p for p in SERVE_PORTS if p not in used and p not in FUNNEL_PORTS), None)


def choose_api_port(listeners: list[Listener], current: int, ours: bool) -> int | None:
    taken = {listener.port for listener in listeners}
    if ours or current not in taken:
        return current
    return next((p for p in API_PORTS if p not in taken), None)


def inspect_ports(m: Machine) -> None:
    area = "Ports"
    found = m.run(["ss", "-ltnpH"])
    m.listeners = parse_listeners(found.out) if found.code == 0 else []
    configured = m.env.get("API_PORT", "").strip()
    m.api_port = m.chosen_api_port or (int(configured) if configured.isdigit() else DEFAULT_API_PORT)
    users = [listener for listener in m.listeners if listener.port == m.api_port]
    ours = False
    if m.docker_ok and m.env_file.exists():  # Docker may publish the port without a listening process
        published = m.run(["docker", "compose", "port", "api", "5000"])
        ours = published.code == 0 and published.out.strip().endswith(f":{m.api_port}")
    if ours:
        m.add("ok", area, f"port {m.api_port}: the Nocturna API")
    elif not users:
        m.add("ok", area, f"port {m.api_port} is free for the API (it listens on 127.0.0.1 only)")
    else:
        m.api_port_taken_by = ", ".join(sorted({listener.program for listener in users}))
        m.add("fix", area, f"port {m.api_port} is used by {m.api_port_taken_by}")
    if m.tailscale and m.ts_state == "Running" and m.serve is not None:
        m.dashboard_port, m.dashboard_public = find_dashboard(m.serve, m.api_port)
        if m.dashboard_port and m.dashboard_public:
            m.add("fix", area, f"the dashboard's Serve port {m.dashboard_port} is on Funnel: it is public")
        elif m.dashboard_port:
            m.add("ok", area, f"the dashboard is served privately at {m.origin}")
        else:
            m.add("info", area, "the dashboard is not served yet")


def inspect_network(m: Machine, extra: dict[str, str]) -> None:
    for host, why in {**HOSTS, **extra}.items():
        if m.http("HEAD", f"https://{host}/", timeout=10).status == 0:
            m.unreachable.append(host)
            m.add("warn", "Network", f"cannot reach {host} ({why})")
    if not m.unreachable:
        m.add("ok", "Network", "every service the setup and the stack use answers")


def _project(root: Path) -> str:
    """Compose's default project name: the folder's name, lower case, a-z 0-9 - _ only."""
    return re.sub(r"[^a-z0-9_-]", "", root.name.lower())


def postgres_volume_exists(r: Runner, root: Path) -> bool:
    found = r(["docker", "volume", "ls", "-q", "--filter", f"label=com.docker.compose.project={_project(root)}",
               "--filter", "label=com.docker.compose.volume=postgres-data"])
    return found.code == 0 and bool(found.out.strip())


def inspect_env(m: Machine) -> None:
    area = ".env"
    if not m.env_file.exists():
        m.add("fix", area, "there is no .env yet")
        return
    mode = m.env_file.stat().st_mode & 0o777
    if mode & 0o077:
        m.add("warn", area, f"readable by other users (mode {mode:o})")
    m.env = env_values(m.env_file)
    m.hide.update(secret_values(m.env))
    problems = env_problems(m.env)
    for var in VARS:
        if var.name in problems:
            level, message = problems[var.name]
            m.add(level, area, f"{var.name}: {message}")
        elif m.env.get(var.name, "").strip():
            m.add("ok", area, f"{var.name}: set")
    ignored = [name for name in STACK_SETS if name in m.env]
    if ignored:
        m.add("info", area, f"ignored, the stack sets them itself: {', '.join(ignored)}")
    bind = m.env.get("API_BIND", "").strip()
    if bind and bind not in ("127.0.0.1", "localhost"):
        m.add("warn", area, f"API_BIND={bind} opens the API to your network without HTTPS; Tailscale does not "
                            "need it")


_SETTINGS_CHECK = """
import sys
from pydantic import ValidationError
from src.config import Settings
try:
    settings = Settings()
except ValidationError as e:
    for error in e.errors(include_url=False, include_input=False, include_context=False):
        where = ".".join(str(part) for part in error["loc"]).upper()
        print((where + ": " if where else "") + error["msg"])
    sys.exit(1)
problems = settings.production_problems()
print("\\n".join(problems))
sys.exit(1 if problems else 0)
"""


def _compose_services(m: Machine) -> list[dict]:
    listed = m.run(["docker", "compose", "ps", "--all", "--format", "json"])
    text = listed.out.strip()
    if listed.code != 0 or not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except ValueError:
        services = []
        for line in text.splitlines():
            try:
                services.append(json.loads(line))
            except ValueError:
                continue
        return services


def inspect_stack(m: Machine) -> None:
    area = "Stack"
    if not m.docker_ok:
        return
    m.db_volume = postgres_volume_exists(m.run, m.root)
    if m.db_volume and not m.env.get("DB_PASSWORD"):
        m.add("fix", area, "the database exists, but .env has no DB_PASSWORD: it needs the one the database "
                           "was created with")
    if not m.env_file.exists():
        return
    config = m.run(["docker", "compose", "config", "--quiet"])
    if config.code != 0:
        m.add("fix", area, f"docker compose rejects the configuration: {_first_line(config.err)}")
        return
    if m.run(["docker", "image", "inspect", IMAGE]).code == 0:
        verdict = m.run(["docker", "compose", "run", "--rm", "--no-deps", "-T", "api", "python", "-c",
                         _SETTINGS_CHECK])
        if verdict.code == 0:
            m.add("ok", area, "the app accepts the settings")
        else:
            for line in (verdict.out.strip() or _first_line(verdict.err)).splitlines():
                m.add("fix", area, f"the app refuses the settings: {line.strip()}")
    m.services = _compose_services(m)
    for service in m.services:
        name, state, health = service.get("Service", "?"), service.get("State", "?"), service.get("Health", "")
        if state == "running" and health in ("", "healthy"):
            m.add("ok", area, f"{name}: running{', healthy' if health else ''}")
        elif state == "running" and health == "starting":
            m.add("info", area, f"{name}: starting")
        else:
            m.add("fix", area, f"{name}: {state}{', ' + health if health else ''}")
    if not m.services:
        m.add("info", area, "not started yet")
    dumps = sorted((m.root / "backups").glob("nocturna-*.dump"))
    if dumps:
        age = (time.time() - dumps[-1].stat().st_mtime) / 3600
        m.add("ok" if age < 26 else "warn", area, f"newest backup {age:.0f} hours old ({len(dumps)} kept)")


def api_running(m: Machine) -> bool:
    return any(s.get("Service") == "api" and s.get("State") == "running" for s in m.services)


def inspect_admin(m: Machine) -> None:
    if not api_running(m):
        return
    listed = m.run(["docker", "compose", "exec", "-T", "api", "flask", "--app", "src.main", "list-admins"])
    if listed.code != 0:
        return
    m.admins = listed.out.split()
    m.add("ok" if m.admins else "fix", "Account", f"dashboard account: {', '.join(m.admins)}" if m.admins
          else "no dashboard account yet")


def inspect(m: Machine) -> Machine:
    """Everything, read only. Secrets of .env are scrubbed from every message."""
    m.findings.clear()
    m.unreachable.clear()
    inspect_system(m)
    inspect_tools(m)
    inspect_docker(m)
    inspect_tailscale(m)
    inspect_env(m)
    inspect_ports(m)
    extra = {}
    if not m.docker_package:
        extra["download.docker.com"] = "Docker's packages"
    if not m.tailscale:
        extra["pkgs.tailscale.com"] = "Tailscale's packages"
    inspect_network(m, extra)
    inspect_stack(m)
    inspect_admin(m)
    for finding in m.findings:
        for value in sorted(m.hide, key=len, reverse=True):
            finding.text = finding.text.replace(value, "***")
    return m


_MARK = {"ok": "ok  ", "info": "    ", "warn": "WARN", "fix": "TODO"}


def report(findings: list[Finding]) -> str:
    lines = []
    area = None
    for f in findings:
        if f.area != area:
            area = f.area
            lines.append(area)
        lines.append(f"  {_MARK[f.level]}  {f.text}")
    return "\n".join(lines)


# -- the plan -----------------------------------------------------------------------------

class StepFailedError(RuntimeError):
    """A step could not be done; the message says why and what to do."""


@dataclass
class Step:
    title: str
    do: Callable[[Setup], None]


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)


def plan(m: Machine) -> Plan:
    """What the setup would change, in order, and what stops it."""
    p = Plan()
    if m.os_id != "ubuntu":
        p.blockers.append("This setup installs software with Ubuntu's tools; this machine is not Ubuntu.")
        return p
    if m.clock_synced is False:
        p.steps.append(Step("Turn on network time (timedatectl set-ntp true)", Setup.sync_clock))
    if m.missing_tools:
        p.steps.append(Step(f"Install {', '.join(m.missing_tools)} (apt)", Setup.install_tools))

    if m.docker_package == "snap":
        p.blockers.append("Docker comes from the snap store. The setup does not remove it for you: removing it "
                          "deletes its containers and images. If nothing else uses it, run "
                          "`sudo snap remove docker`, then run the setup again: it then installs Docker's "
                          "packages.")
    elif not m.docker_package and m.conflicts:
        p.blockers.append(f"Docker is not installed, but packages that conflict with it are: "
                          f"{', '.join(m.conflicts)}. Other software (Kubernetes, Podman) may need them, so the "
                          "setup leaves them. If nothing uses them, `sudo apt remove` them and run the setup "
                          "again.")
    elif not m.docker_package:
        p.steps.append(Step("Install Docker Engine, Compose and buildx from Docker's repository",
                            Setup.install_docker))
    else:
        if not (m.docker_enabled and m.docker_running):
            p.steps.append(Step("Start Docker, and start it at boot (systemctl enable --now docker)",
                                Setup.start_docker))
        if not m.compose:
            p.steps.append(Step("Install Docker Compose v2 (apt)", Setup.install_compose))
        if not m.buildx:
            p.steps.append(Step("Install Docker buildx (apt)", Setup.install_buildx))

    if not m.tailscale:
        p.steps.append(Step("Install Tailscale (Tailscale's install script; it adds Tailscale's apt repository)",
                            Setup.install_tailscale))
        p.steps.append(Step("Connect this machine to your tailnet (tailscale up prints a link: open it and "
                            "sign in)", Setup.connect_tailscale))
    elif m.ts_state == "not running":
        p.blockers.append("Tailscale is installed but its service is not running. Start it the way you normally "
                          "do (sudo systemctl enable --now tailscaled), then run the setup again.")
    elif m.ts_state in ("NeedsLogin", "NoState", ""):
        p.blockers.append("Tailscale is installed but this machine is not logged in. Log in the way you normally "
                          "do (sudo tailscale up, with your usual options), then run the setup again: the setup "
                          "does not change your Tailscale settings.")
    elif m.ts_state == "Stopped":
        p.blockers.append("Tailscale is stopped on this machine (tailscale down). Start it the way you normally "
                          "do, then run the setup again: the setup does not change your Tailscale settings.")
    elif m.ts_state == "NeedsMachineAuth":
        p.blockers.append("Tailscale waits for this machine to be approved: approve it on the Machines page of "
                          "the admin console, then run the setup again.")
    elif m.ts_state != "Running":
        p.blockers.append(f"Tailscale is in an unexpected state ({m.ts_state}).")
    if m.ts_state == "Running" and not (m.magicdns and m.certs):
        p.steps.append(Step("Turn on MagicDNS and HTTPS certificates in the Tailscale admin console (you do it; "
                            "the setup waits and checks)", Setup.tailscale_console))
    if m.ts_state == "Running" and m.serve is None:
        p.blockers.append("The setup could not read what Tailscale serves (tailscale serve status --json).")

    api_port = choose_api_port(m.listeners, m.api_port, ours=not m.api_port_taken_by)
    if api_port is None:
        p.blockers.append(f"Port {m.api_port} is used by {m.api_port_taken_by}, and so is every port from "
                          f"{API_PORTS.start} to {API_PORTS.stop - 1}.")
    elif api_port != m.api_port:
        p.steps.append(Step(f"Use port {api_port} on 127.0.0.1 for the API ({m.api_port} is used by "
                            f"{m.api_port_taken_by})", lambda s, port=api_port: s.use_api_port(port)))

    if m.dashboard_public:
        p.blockers.append(f"The dashboard is served on port {m.dashboard_port} with Funnel on, so it is public. "
                          f"Turn that off (sudo tailscale funnel --https={m.dashboard_port} off), then run the "
                          "setup again: it serves it privately.")
    elif not m.dashboard_port:
        port = choose_serve_port(m.serve or {}, m.listeners) if m.serve is not None else None
        where = (f"https://{m.ts_name}:{port}" if port and m.ts_name else "a private HTTPS port from 9443 up")
        p.steps.append(Step(f"Serve the dashboard at {where}: tailnet only (Funnel cannot use this port), and "
                            "nothing Tailscale serves now changes", Setup.serve_dashboard))

    problems = env_problems(m.env)
    if not m.env_file.exists():
        p.steps.append(Step("Create .env: the passwords are generated; you give the Alpaca keys, the Telegram "
                            "bot and the healthchecks.io URLs, and each is tested", Setup.write_env))
    else:
        p.steps.append(Step("Check .env with you: keep what works, fill in what is missing" + (
            f" ({', '.join(sorted(problems))})" if problems else ""), Setup.write_env))
    p.steps.append(Step("Build and start Nocturna (docker compose up -d --build), and wait until every service "
                        "is healthy", Setup.start_stack))
    if not m.admins:
        p.steps.append(Step("Create your dashboard account (you choose the username and password)",
                            Setup.create_account))
    p.steps.append(Step("Send a test alert to Telegram", Setup.test_alert))

    if m.key_expiry:
        p.advice.append(f"Tailscale: this machine's key expires on {m.key_expiry}. On the Machines page of the "
                        "admin console, open the menu at the right of its row and choose Disable Key Expiry.")
    if m.unreachable:
        p.advice.append(f"Unreachable now: {', '.join(m.unreachable)}. The steps that need them will fail until "
                        "they answer.")
    return p


# -- the steps --------------------------------------------------------------------------------

Ask = Callable[[str], str]


@dataclass
class Dialog:
    ask: Ask = input
    ask_secret: Ask = getpass.getpass
    say: Callable[[str], None] = print

    def yes(self, question: str, default: bool = True) -> bool:
        answer = self.ask(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
        return default if not answer else answer.startswith("y")


APT = ["apt-get", "-o", "DPkg::Lock::Timeout=300"]


class Setup:
    """Carries the machine and the dialog through the steps."""

    def __init__(self, m: Machine, d: Dialog) -> None:
        self.m, self.d = m, d
        self.apt_updated = False

    # helpers

    def must(self, args: Sequence[str], what: str, stream: bool = True) -> Result:
        result = self.m.run(args, stream=stream)
        if result.code != 0:
            detail = "" if stream else f": {_first_line(result.err) or _first_line(result.out)}"
            raise StepFailedError(f"{what} failed (exit {result.code}){detail}")
        return result

    def apt_update(self, force: bool = False) -> None:
        if force or not self.apt_updated:
            self.must([*APT, "update"], "apt-get update")
            self.apt_updated = True

    def apt_install(self, packages: Sequence[str]) -> None:
        self.apt_update()
        self.must(["env", "DEBIAN_FRONTEND=noninteractive", *APT, "install", "-y", *packages],
                  f"installing {', '.join(packages)}")

    def available(self, package: str) -> bool:
        policy = self.m.run(["apt-cache", "policy", package]).out
        candidate = re.search(r"Candidate:\s*(\S+)", policy)
        return bool(candidate) and candidate.group(1) != "(none)"

    def install_first_available(self, packages: Sequence[str], what: str) -> None:
        self.apt_update()
        for package in packages:
            if self.available(package):
                self.apt_install([package])
                return
        raise StepFailedError(f"no apt package provides {what} ({' or '.join(packages)})")

    def refresh(self) -> None:
        inspect(self.m)

    # steps

    def sync_clock(self) -> None:
        self.must(["timedatectl", "set-ntp", "true"], "timedatectl set-ntp true", stream=False)
        for _ in range(18):
            if self.m.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"]).out.strip() == "yes":
                self.d.say("The clock is synchronized.")
                return
            time.sleep(5)
        self.d.say("Network time is on; the clock is not synchronized yet. It usually takes a few minutes.")

    def install_tools(self) -> None:
        self.apt_install(self.m.missing_tools)

    def install_docker(self) -> None:
        codename = self.m.codename
        if self.m.http("GET", f"https://download.docker.com/linux/ubuntu/dists/{codename}/Release").status != 200:
            raise StepFailedError(f"Docker's repository has no packages for Ubuntu {codename}")
        self.apt_install(["ca-certificates", "curl"])
        keyrings = self.m.etc / "apt" / "keyrings"
        self.must(["install", "-m", "0755", "-d", str(keyrings)], "creating the apt keyring folder", stream=False)
        key = keyrings / "docker.asc"
        self.must(["curl", "-fsSL", "https://download.docker.com/linux/ubuntu/gpg", "-o", str(key)],
                  "downloading Docker's signing key", stream=False)
        key.chmod(0o644)
        architecture = self.must(["dpkg", "--print-architecture"], "dpkg --print-architecture",
                                 stream=False).out.strip()
        (self.m.etc / "apt" / "sources.list.d" / "docker.sources").write_text(
            "Types: deb\nURIs: https://download.docker.com/linux/ubuntu\n"
            f"Suites: {codename}\nComponents: stable\nArchitectures: {architecture}\nSigned-By: {key}\n")
        self.apt_update(force=True)
        self.apt_install(["docker-ce", "docker-ce-cli", "containerd.io", "docker-buildx-plugin",
                          "docker-compose-plugin"])
        self.start_docker()

    def start_docker(self) -> None:
        self.must(["systemctl", "enable", "--now", "docker"], "systemctl enable --now docker", stream=False)
        if self.m.run(["docker", "info", "--format", "{{.ServerVersion}}"]).code != 0:
            raise StepFailedError("Docker was started but does not answer (systemctl status docker shows why)")

    def install_compose(self) -> None:
        packages = ["docker-compose-plugin", "docker-compose-v2"]
        self.install_first_available(packages if self.m.docker_package == "docker-ce" else packages[::-1],
                                     "Docker Compose v2")

    def install_buildx(self) -> None:
        packages = ["docker-buildx-plugin", "docker-buildx"]
        self.install_first_available(packages if self.m.docker_package == "docker-ce" else packages[::-1],
                                     "Docker buildx")

    def install_tailscale(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / "install.sh"
            self.must(["curl", "-fsSL", "https://tailscale.com/install.sh", "-o", str(script)],
                      "downloading Tailscale's install script", stream=False)
            self.must(["sh", str(script)], "Tailscale's install script")
        self.must(["systemctl", "enable", "--now", "tailscaled"], "systemctl enable --now tailscaled", stream=False)

    def connect_tailscale(self) -> None:
        self.d.say("tailscale up prints a link: open it (on your phone is fine) and sign in.")
        self.must(["tailscale", "up"], "tailscale up")
        self.refresh()
        if self.m.ts_state != "Running":
            raise StepFailedError(f"Tailscale is not connected ({self.m.ts_state})")

    def tailscale_console(self) -> None:
        while True:
            self.refresh()
            if self.m.magicdns and self.m.certs:
                self.d.say("MagicDNS and HTTPS certificates are on.")
                return
            self.d.say("In the Tailscale admin console (https://login.tailscale.com/admin/dns), on the DNS page:\n"
                       "  - turn on MagicDNS, if it is off;\n"
                       "  - under HTTPS Certificates, select Enable HTTPS.\n"
                       "Machine names and the tailnet's name then appear in a public certificate log.")
            if self.d.ask("Press Enter when done (or type q to stop): ").strip().lower() == "q":
                raise StepFailedError("MagicDNS and HTTPS certificates are needed for the dashboard's address")

    def use_api_port(self, port: int) -> None:
        self.m.api_port = self.m.chosen_api_port = port
        self.m.api_port_taken_by = ""

    def serve_dashboard(self) -> None:
        if not self.m.ts_name or self.m.serve is None:
            self.refresh()
        if not (self.m.magicdns and self.m.certs):
            self.tailscale_console()
        before = self.m.serve
        if before is None or not self.m.ts_name:
            raise StepFailedError("Tailscale is not ready to serve (see the Tailscale lines above)")
        port = choose_serve_port(before, self.m.listeners)
        if port is None:
            raise StepFailedError(f"every port from {SERVE_PORTS.start} to {SERVE_PORTS.stop - 1} is in use")
        target = _api_target(self.m.api_port)
        self.must(["tailscale", "serve", "--bg", f"--https={port}", target], "tailscale serve")
        after = self.read_serve()
        changes = serve_changes(before, after, f"{self.m.ts_name}:{port}", target)
        if changes:
            self.m.run(["tailscale", "serve", f"--https={port}", "off"])
            raise StepFailedError("Tailscale's configuration did not change as expected, so the new entry was removed "
                             "again: " + "; ".join(changes))
        self.m.serve, self.m.dashboard_port = after, port
        self.d.say(f"The dashboard will be at {self.m.origin} (your devices only).")

    def read_serve(self) -> dict:
        served = self.m.run(["tailscale", "serve", "status", "--json"])
        try:
            return json.loads(served.out) if served.out.strip() else {}
        except ValueError as e:
            raise StepFailedError("could not read Tailscale's Serve configuration back") from e

    def write_env(self) -> None:
        if not self.m.origin:
            self.refresh()
        code = build_env(self.m.env_file, self.m.run, self.m.http, self.d, origin=self.m.origin,
                         api_port=self.m.api_port, docker=self.m.docker_ok)
        if code != 0:
            raise StepFailedError(".env was not written")
        self.m.env = env_values(self.m.env_file)
        self.m.hide.update(secret_values(self.m.env))
        fixes = [f"{name}: {message}" for name, (level, message) in env_problems(self.m.env).items()
                 if level == "fix"]
        if fixes:
            raise StepFailedError("these settings still need a value: " + "; ".join(fixes))

    def start_stack(self) -> None:
        self.d.say("The first build takes a few minutes.")
        result = self.m.run(["docker", "compose", "up", "-d", "--build", "--wait", "--wait-timeout", "600"],
                            stream=True)
        if result.code != 0:
            raise StepFailedError("the stack did not become healthy: `sudo docker compose ps` shows which service, and "
                             "`sudo docker compose logs <service> --tail 50` why")
        ready = self.m.http("GET", f"http://127.0.0.1:{self.m.api_port}/ready", timeout=10)
        if ready.status != 200:
            raise StepFailedError(f"the API does not answer on 127.0.0.1:{self.m.api_port} ({ready.status or ready.error})")
        self.m.services = _compose_services(self.m)
        self.d.say("Every service is running and healthy.")

    def create_account(self) -> None:
        listed = self.m.run(["docker", "compose", "exec", "-T", "api", "flask", "--app", "src.main", "list-admins"])
        if listed.code == 0 and listed.out.split():
            self.d.say(f"The dashboard account exists: {', '.join(listed.out.split())}.")
            return
        username = ""
        while not _USERNAME.fullmatch(username):
            default = self.m.operator if _USERNAME.fullmatch(self.m.operator or "") else "operator"
            username = self.d.ask(f"Dashboard username (3-50 letters, digits or _) [{default}]: ").strip() or default
        email = ""
        while not _EMAIL.fullmatch(email):
            email = self.d.ask("Email for the account (it identifies the account; nothing is sent to it): ").strip()
        password = self.choose_password()
        self.m.hide.add(password)
        # The password goes in the environment, not on the command line; the app's startup logs stay quiet
        done = self.m.run(["docker", "compose", "exec", "-T", "-e", "LOG_LEVEL=WARNING", "-e", "ADMIN_PASSWORD",
                           "api", "flask", "--app", "src.main", "create-admin", "--username", username,
                           "--email", email], env={"ADMIN_PASSWORD": password})
        if done.code != 0:
            lines = [line.strip() for line in (done.err + done.out).splitlines() if line.strip()]
            reason = next((line for line in lines if line.startswith("Error:")), lines[-1] if lines else "")
            raise StepFailedError(f"the dashboard account was not created ({reason or f'exit {done.code}'})")
        self.m.admins = [username.lower()]
        self.d.say(f"The dashboard account {username.lower()} is created.")

    def choose_password(self) -> str:
        self.d.say(f"Now the password, typed twice and hidden: {PASSWORD_RULES}.")
        while True:
            password = self.d.ask_secret("Password (hidden): ")
            problem = password_problem(password)
            if problem:
                self.d.say(f"Not accepted: {problem}. The rules: {PASSWORD_RULES}.")
            elif self.d.ask_secret("The same password again (hidden): ") != password:
                self.d.say("The two did not match. Type it again.")
            else:
                return password

    def test_alert(self) -> None:
        if not (self.m.env.get("ALERT_TELEGRAM_BOT_TOKEN") and self.m.env.get("ALERT_TELEGRAM_CHAT_ID")):
            self.d.say("Telegram is not set up: no test alert.")
            return
        sent = self.m.run(["docker", "compose", "exec", "-T", "api", "flask", "--app", "src.main",
                           "send-test-alert"])
        for line in sent.out.strip().splitlines():
            self.d.say(f"  {line}")
        if sent.code != 0:
            raise StepFailedError("the test alert was not sent (the reason is above)")
        if not self.d.yes("Did the test alert reach your phone?"):
            self.d.say("Check that you opened the bot's chat and tapped Start; then run the setup again.")


def serve_changes(before: dict, after: dict, hostport: str, target: str) -> list[str]:
    """What is wrong with Serve's new configuration: anything else changed, or the new entry not as asked."""
    problems = []
    port = hostport.rsplit(":", 1)[1]
    for section, skip in (("TCP", port), ("Web", hostport), ("AllowFunnel", hostport)):
        old, new = before.get(section) or {}, after.get(section) or {}
        for key in sorted(set(old) | set(new)):
            if key != skip and old.get(key) != new.get(key):
                problems.append(f"{section} {key} changed")
    handler = (((after.get("Web") or {}).get(hostport) or {}).get("Handlers") or {}).get("/") or {}
    if handler.get("Proxy", "").rstrip("/") != target:
        problems.append(f"{hostport} does not forward to {target}")
    if (after.get("AllowFunnel") or {}).get(hostport):
        problems.append(f"{hostport} is public (Funnel)")
    return problems


# -- writing .env with the operator ------------------------------------------------------------

def alpaca_account(h: Http, key: str, secret: str) -> tuple[bool, str]:
    """Whether the keys open an Alpaca paper account, and what to tell the operator."""
    answer = h("GET", f"{ALPACA_PAPER}/v2/account", headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    if answer.status == 200:
        account = answer.json()
        return True, (f"Alpaca paper account {account.get('account_number', '?')}: {account.get('status', '?')}, "
                      f"equity {account.get('equity', '?')} {account.get('currency', '')}".rstrip())
    if answer.status in (401, 403):
        return False, "Alpaca refused these keys: copy them again from the paper account (the Key ID starts with PK)"
    return False, f"could not reach Alpaca ({answer.status or answer.error})"


def telegram(h: Http, token: str, method: str, data: dict | None = None) -> tuple[bool, object]:
    answer = h("POST" if data is not None else "GET", f"{TELEGRAM}/bot{token}/{method}", data=data)
    body = answer.json()
    if answer.status == 200 and body.get("ok"):
        return True, body.get("result")
    return False, body.get("description") or answer.error or f"HTTP {answer.status}"


def private_chats(updates: object) -> dict[str, str]:
    """chat id -> who, for the private chats that wrote to the bot."""
    chats: dict[str, str] = {}
    for update in updates if isinstance(updates, list) else []:
        for kind in ("message", "edited_message", "my_chat_member"):
            chat = (update.get(kind) or {}).get("chat") or {}
            if chat.get("type") == "private" and "id" in chat:
                who = " ".join(filter(None, (chat.get("first_name"), chat.get("last_name"))))
                handle = f" (@{chat['username']})" if chat.get("username") else ""
                chats[str(chat["id"])] = (who or "?") + handle
    return chats


ALPACA_HELP = """
Alpaca paper keys: sign in at https://app.alpaca.markets and make sure the account
switcher at the top left shows your Paper account, not Live. On its home page, in the
API Keys box, generate the keys and copy both the Key ID and the Secret Key: the secret
is shown only once. Generating new keys disables the old pair.
"""

TELEGRAM_HELP = """
Telegram bot: in Telegram, open @BotFather, send /newbot, give the bot a name and then a
username ending in "bot". BotFather answers with the token (digits, a colon, letters).
"""

HEALTHCHECKS_HELP = """
healthchecks.io: the ping URL of each check is on its page, https://hc-ping.com/<uuid>.
The "engine" check: period 1 minute, grace 5 minutes; the "backup" check: period 1 day,
grace 2 hours. Press Enter to skip, or to keep what is set.
"""


def build_env(env_file: Path, r: Runner, h: Http, d: Dialog, *, origin: str | None,
              api_port: int = DEFAULT_API_PORT, docker: bool = True) -> int:
    """Create or update .env with the operator. Returns 0, or 1 when it must not be written."""
    previous = parse_env(env_file.read_text()) if env_file.exists() else []
    values = {name: value for name, value, _ in previous}
    done: dict[str, str] = {}
    problems = env_problems(values)

    # Secrets: generated once and then never changed
    db_set = bool(values.get("DB_PASSWORD")) and not is_placeholder(values["DB_PASSWORD"])
    if not db_set and docker and postgres_volume_exists(r, env_file.parent):
        d.say("The database already exists, but .env has no DB_PASSWORD: PostgreSQL keeps the password it was\n"
              "created with. Put the old .env back (look for .env.backup-* files), or, only if the database\n"
              "holds nothing you need, delete it with `sudo docker compose down -v` and run the setup again.")
        return 1
    for name, size in (("FLASK_SECRET_KEY", 32), ("JWT_SECRET_KEY", 32), ("API_KEY_PEPPER", 32),
                       ("DB_PASSWORD", 16), ("REDIS_PASSWORD", 16)):
        if name == "DB_PASSWORD" and db_set:
            done[name] = "kept"
            if name in problems:
                d.say(f"DB_PASSWORD: {problems[name][1]}. Left unchanged, since the database uses it.")
        elif name in problems and problems[name][0] == "fix" or not values.get(name):
            values[name], done[name] = secrets.token_hex(size), "generated"
        else:
            done[name] = "kept"
    if values["JWT_SECRET_KEY"] == values["FLASK_SECRET_KEY"]:
        values["JWT_SECRET_KEY"], done["JWT_SECRET_KEY"] = secrets.token_hex(32), "generated"

    # Alpaca
    d.say(ALPACA_HELP)
    key, secret = values.get("ALPACA_API_KEY", ""), values.get("ALPACA_SECRET_KEY", "")
    have = bool(key and secret) and not (is_placeholder(key) or is_placeholder(secret))
    if have and d.yes("Keep the Alpaca keys already in .env?"):
        ok, message = alpaca_account(h, key, secret)
        d.say(message)
        have = ok or not d.yes("Enter them again?")
    else:
        have = False
    while not have:
        key = d.ask_secret("Alpaca paper Key ID (hidden as you type or paste): ").strip()
        secret = d.ask_secret("Alpaca paper Secret Key (hidden): ").strip()
        ok, message = alpaca_account(h, key, secret)
        d.say(message)
        if ok or d.yes("Try again?"):
            have = ok
        elif d.yes("Save them anyway?", default=False):
            have = True
        else:
            key, secret = values.get("ALPACA_API_KEY", ""), values.get("ALPACA_SECRET_KEY", "")
            d.say("Left the Alpaca keys as they were.")
            have = True
    if (key, secret) != (values.get("ALPACA_API_KEY"), values.get("ALPACA_SECRET_KEY")):
        done["ALPACA_API_KEY"] = done["ALPACA_SECRET_KEY"] = "set"
    values["ALPACA_API_KEY"], values["ALPACA_SECRET_KEY"] = key, secret

    _telegram_step(values, done, h, d)

    # The dashboard's address, the proxy in front of it, the API's port
    if origin and values.get("ALLOWED_ORIGINS") != origin:
        values["ALLOWED_ORIGINS"], done["ALLOWED_ORIGINS"] = origin, "set"
    elif not origin and (not values.get("ALLOWED_ORIGINS") or "ALLOWED_ORIGINS" in problems):
        d.say(f"The dashboard is not served yet: its address is set to http://127.0.0.1:{api_port} for now.\n"
              "It only matters for requests from other sites; the setup sets it once Tailscale serves it.")
        values["ALLOWED_ORIGINS"], done["ALLOWED_ORIGINS"] = f"http://127.0.0.1:{api_port}", "set"
    for name, value in (("TRUSTED_PROXY_COUNT", "1"), ("API_PORT", str(api_port))):
        if values.get(name) != value:
            values[name], done[name] = value, "set"
    if values.get("ENGINE_AUTOSTART", "").strip().lower() not in ("true", "false"):
        values["ENGINE_AUTOSTART"], done["ENGINE_AUTOSTART"] = "false", "set"

    # healthchecks.io
    d.say(HEALTHCHECKS_HELP)
    for name, label in (("HEARTBEAT_URL", "engine"), ("BACKUP_HEARTBEAT_URL", "backup")):
        while True:
            url = d.ask(f"Ping URL of the {label} check: ").strip()
            if not url:
                break
            if url.startswith("https://"):
                values[name], done[name] = url, "set"
                break
            d.say("A ping URL starts with https://hc-ping.com/")

    write_env(env_file, values, previous)
    d.say(f"\nWrote {env_file} (only root can read it).")
    for var in VARS:
        state = done.get(var.name) or ("kept" if values.get(var.name) else "not set")
        d.say(f"  {var.name:26} {state}")
    return 0


def _telegram_step(values: dict[str, str], done: dict[str, str], h: Http, d: Dialog) -> None:
    token, chat = values.get("ALERT_TELEGRAM_BOT_TOKEN", ""), values.get("ALERT_TELEGRAM_CHAT_ID", "")
    if not d.yes("Set up Telegram alerts (and commands)?"):
        if not (token and chat):
            values["TELEGRAM_COMMANDS"] = "false"
        return
    d.say(TELEGRAM_HELP)
    bot = None
    if token and not is_placeholder(token):
        ok, result = telegram(h, token, "getMe")
        bot = result if ok and isinstance(result, dict) else None
        if bot is None:
            d.say(f"The bot token in .env does not work ({result}).")
    while bot is None:
        token = d.ask_secret("Bot token (hidden): ").strip()
        ok, result = telegram(h, token, "getMe")
        if ok and isinstance(result, dict):
            bot = result
            done["ALERT_TELEGRAM_BOT_TOKEN"] = "set"
        else:
            d.say(f"Telegram refused this token ({result}).")
            if not d.yes("Try again?"):
                return
    username = bot.get("username", "")
    d.say(f"The bot is @{username}.")
    if chat and _CHAT.fullmatch(chat) and d.yes("Keep the chat id already in .env?"):
        done.setdefault("ALERT_TELEGRAM_CHAT_ID", "kept")
    else:
        chat = ""
        while not chat:
            d.say(f"In Telegram, open https://t.me/{username} and tap Start (or send it any message).")
            typed = d.ask("Then press Enter here (or type the chat id if you know it, or skip): ").strip()
            if typed.lower() == "skip":
                d.say("Telegram left as it was.")
                return
            if _CHAT.fullmatch(typed):
                chat = typed
                break
            ok, result = telegram(h, token, "getUpdates", {"timeout": 0})
            if not ok:
                d.say(f"Could not read the bot's messages ({result}). If Nocturna already runs with\n"
                      "TELEGRAM_COMMANDS=true it reads them first: sudo docker compose stop api, then try again.")
                if not d.yes("Try again?"):
                    return
                continue
            chats = private_chats(result)
            if not chats:
                d.say("No message reached the bot yet.")
                continue
            choices = list(chats.items())
            if len(choices) == 1:
                chat = choices[0][0]
                d.say(f"Found your chat with {choices[0][1]}.")
            else:
                for n, (_, who) in enumerate(choices, start=1):
                    d.say(f"  {n}. {who}")
                pick = d.ask("Which one is you? ").strip()
                if pick.isdigit() and 1 <= int(pick) <= len(choices):
                    chat = choices[int(pick) - 1][0]
        done["ALERT_TELEGRAM_CHAT_ID"] = "set"
    ok, result = telegram(h, token, "sendMessage", {"chat_id": chat, "text": "Nocturna: alerts will come here."})
    d.say("Sent a test message to that chat." if ok else f"Could not send a test message ({result}).")
    values["ALERT_TELEGRAM_BOT_TOKEN"], values["ALERT_TELEGRAM_CHAT_ID"] = token, chat
    commands = d.yes("Let the bot also take commands from that chat (/status, /start_engine, /halt...)?")
    wanted = "true" if commands else "false"
    if values.get("TELEGRAM_COMMANDS") != wanted:
        values["TELEGRAM_COMMANDS"], done["TELEGRAM_COMMANDS"] = wanted, "set"


_SECTIONS = (
    ("Generated by the setup. Never change them once the stack has run: PostgreSQL keeps the\n"
     "# DB_PASSWORD it was created with, and new secrets sign everyone out and void the API keys.", GENERATED),
    ("Alpaca paper account (app.alpaca.markets, Paper account, API Keys)", ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")),
    ("The dashboard's address (Tailscale Serve, tailnet only), the proxy in front of the API, and the\n"
     "# API's port on this machine (127.0.0.1 only)", ("ALLOWED_ORIGINS", "TRUSTED_PROXY_COUNT", "API_PORT")),
    ("After a restart trading waits for you to start it", ("ENGINE_AUTOSTART",)),
    ("Telegram: alerts, and commands when TELEGRAM_COMMANDS=true",
     ("ALERT_TELEGRAM_BOT_TOKEN", "ALERT_TELEGRAM_CHAT_ID", "TELEGRAM_COMMANDS")),
    ("healthchecks.io ping URLs: the engine check and the backup check", ("HEARTBEAT_URL", "BACKUP_HEARTBEAT_URL")),
)


def write_env(env_file: Path, values: dict[str, str], previous: list[tuple[str, str, str]]) -> None:
    """Write .env (mode 600, replaced in one step), after copying the old one aside."""
    known = {name for _, names in _SECTIONS for name in names}
    lines = [f"# Nocturna settings, written by sudo ./setup.sh on {datetime.now():%Y-%m-%d %H:%M}.",
             f"# {GUIDE} explains each one. Keep this file private; run the setup again to change it."]
    for title, names in _SECTIONS:
        lines += ["", f"# {title}"]
        lines += [env_line(name, values.get(name, "")) for name in names]
    kept = [raw for name, value, raw in previous
            if name not in known and name not in STACK_SETS and not is_placeholder(value)]
    if kept:
        lines += ["", "# Kept from the previous .env"] + kept
    if env_file.exists():
        backup = env_file.with_name(f".env.backup-{datetime.now():%Y%m%d-%H%M%S}")
        _write_private(backup, env_file.read_text())
    _write_private(env_file, "\n".join(lines) + "\n")


def _write_private(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(text)
    os.replace(temporary, path)


# -- running it -------------------------------------------------------------------------------

def summary(m: Machine) -> str:
    account = ", ".join(m.admins) if m.admins else "not created"
    lines = [
        "Nocturna is set up.",
        f"  Dashboard:  {m.origin or 'not served yet'} (your Tailscale devices only; account: {account})",
        "  Trading:    runs only when you start it: /start_engine in Telegram, or Start engine in the dashboard.",
        "              After any restart of the engine it waits for you again.",
        "  healthchecks.io: the engine check stays silent while trading is stopped; pause it while you wait.",
        "  Run `sudo ./setup.sh` again at any time; `sudo ./setup.sh --check` only reports.",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, d: Dialog | None = None, m: Machine | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check this machine and set Nocturna up (docs/HOME_SERVER.md).")
    parser.add_argument("--check", action="store_true", help="report only; change nothing")
    args = parser.parse_args(argv)
    d = d or Dialog()
    if m is None:
        if os.geteuid() != 0:
            d.say("Run it with sudo: sudo ./setup.sh" + (" --check" if args.check else ""))
            return 2
        m = Machine(run, http)
    m.operator = m.operator or os.environ.get("SUDO_USER", "root")

    d.say("Nocturna setup: looking at this machine (this changes nothing)...\n")
    inspect(m)
    d.say(report(m.findings))
    p = plan(m)
    d.say("")
    if p.blockers:
        d.say("Stopped before changing anything:")
        for blocker in p.blockers:
            d.say(f"  - {blocker}")
        return 1
    d.say("To do:" if p.steps else "Nothing to do.")
    for n, step in enumerate(p.steps, start=1):
        d.say(f"  {n}. {step.title}")
    for advice in p.advice:
        d.say(f"  Also: {advice}")
    if args.check:
        return 1 if p.steps else 0
    if not p.steps or not d.yes("\nGo ahead?"):
        return 0

    setup = Setup(m, d)
    for n, step in enumerate(p.steps, start=1):
        d.say(f"\n==> {n}/{len(p.steps)}: {step.title}")
        try:
            step.do(setup)
        except StepFailedError as e:
            d.say(f"\nStopped at step {n}: {e}.\nThe steps before it are done. Fix this, then run "
                  "`sudo ./setup.sh` again: it carries on from here.")
            return 1
    d.say("\nChecking everything again...\n")
    inspect(m)
    d.say(report(m.findings))
    d.say("")
    d.say(summary(m))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Nothing more was changed; run `sudo ./setup.sh` again to carry on.")
        sys.exit(130)
