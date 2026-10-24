"""
Tests for setup.sh's installer (setup/installer.py): what it finds on a
machine, the plan it makes from that, each step it takes, the .env it writes,
and that it never shows a secret or changes what it must leave alone.
"""

import importlib.util
import json
import re
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("installer", ROOT / "setup" / "installer.py")
inst = importlib.util.module_from_spec(_spec)
sys.modules["installer"] = inst
_spec.loader.exec_module(inst)

# Has the shape setup checks (digits, a colon, 20+ characters), not a real token's
TOKEN = "123:fake-token-for-tests-only"
NAME = "gibliz.taile5d4a8.ts.net"
GOOD = {
    "FLASK_SECRET_KEY": "a" * 64, "JWT_SECRET_KEY": "b" * 64, "API_KEY_PEPPER": "c" * 64,
    "DB_PASSWORD": "d" * 32, "REDIS_PASSWORD": "e" * 32,
    "ALPACA_API_KEY": "PKABCDEFGH123456", "ALPACA_SECRET_KEY": "s3cr3tAlpacaValue0123456789",
    "ALLOWED_ORIGINS": f"https://{NAME}:9443", "TRUSTED_PROXY_COUNT": "1", "API_PORT": "5000",
    "ENGINE_AUTOSTART": "false",
    "ALERT_TELEGRAM_BOT_TOKEN": TOKEN, "ALERT_TELEGRAM_CHAT_ID": "424242", "TELEGRAM_COMMANDS": "true",
    "HEARTBEAT_URL": "https://hc-ping.com/11111111-aaaa", "BACKUP_HEARTBEAT_URL": "https://hc-ping.com/22222222-bbbb",
}

#: The operator's Tailscale Serve configuration: nginx and another service, both on Funnel
OPERATOR_SERVE = {
    "TCP": {"443": {"HTTPS": True}, "8443": {"HTTPS": True}},
    "Web": {f"{NAME}:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:80"}}},
            f"{NAME}:8443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:3000"}}}},
    "AllowFunnel": {f"{NAME}:443": True, f"{NAME}:8443": True},
}
SS = "\n".join([
    'LISTEN 0 511 0.0.0.0:80 0.0.0.0:* users:(("nginx",pid=812,fd=6))',
    'LISTEN 0 4096 127.0.0.1:3000 0.0.0.0:* users:(("node",pid=901,fd=20))',
    'LISTEN 0 4096 [::]:22 [::]:* users:(("sshd",pid=700,fd=4))',
    'LISTEN 0 4096 100.101.102.103:443 0.0.0.0:* users:(("tailscaled",pid=650,fd=30))',
    'LISTEN 0 4096 127.0.0.53%lo:53 0.0.0.0:* users:(("systemd-resolve",pid=600,fd=15))',
])


def write(path, values, mode=0o600):
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    path.chmod(mode)
    return path


def ok(out=""):
    return inst.Result(0, out)


def missing(name="x"):
    return inst.Result(127, "", f"{name}: not found")


INSTALLED = ok("install ok installed")


class Runner:
    """Answers commands by their leading words; a value may be a function of the command. Anything else is
    not installed."""

    def __init__(self, answers=None):
        self.answers = dict(answers or {})
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        for n in range(len(args), 0, -1):
            answer = self.answers.get(tuple(args[:n]))
            if answer is not None:
                return answer(list(args)) if callable(answer) else answer
        return missing(args[0])

    def ran(self, *words):
        return [c for c in self.calls if c[:len(words)] == list(words)]


def tailscale_status(state="Running", magic=True, certs=True, expiry=None):
    me = {"DNSName": NAME + "."}
    if expiry:
        me["KeyExpiry"] = expiry
    return ok(json.dumps({"BackendState": state, "Self": me, "CurrentTailnet": {"MagicDNSEnabled": magic},
                          "CertDomains": [NAME] if certs else None}))


def operator_machine(**changes):
    """The operator's server as the setup sees it: Ubuntu 26.04, Ubuntu's docker.io, their user outside the
    docker group, Tailscale up with nginx and another service on Funnel, no .env yet."""
    answers = {
        ("timedatectl",): ok("yes\n"),
        ("dpkg-query", "-W", "-f=${Status}", "git"): INSTALLED, ("dpkg-query", "-W", "-f=${Status}", "curl"): INSTALLED,
        ("dpkg-query", "-W", "-f=${Status}", "ca-certificates"): INSTALLED,
        ("dpkg-query", "-W", "-f=${Status}", "docker.io"): INSTALLED,
        ("docker", "--version"): ok("Docker version 29.1.3, build 29.1.3-0ubuntu4.1\n"),
        ("systemctl", "is-enabled", "docker"): ok("enabled\n"), ("docker", "info"): ok("29.1.3\n"),
        ("docker", "compose", "version"): ok("5.6.1\n"), ("docker", "buildx", "version"): ok("github.com/docker/buildx 0.30\n"),
        ("id", "-nG"): ok("yogsotho adm sudo\n"),
        ("tailscale", "version"): ok("1.104.2\n"), ("tailscale", "status"): tailscale_status(expiry="2026-11-09T10:00:00Z"),
        ("tailscale", "serve", "status"): ok(json.dumps(OPERATOR_SERVE)),
        ("ss", "-ltnpH"): ok(SS),
    }
    answers.update(changes)
    return answers


def machine(tmp_path, answers, http=None, **fields):
    release = tmp_path / "os-release"
    if not release.exists():
        release.write_text('ID=ubuntu\nVERSION_ID="26.04"\nUBUNTU_CODENAME=resolute\n'
                           'PRETTY_NAME="Ubuntu 26.04.1 LTS"\n')
    root = tmp_path / "Nocturna-Trading-System"
    root.mkdir(exist_ok=True)
    return inst.Machine(Runner(answers), http or (lambda *a, **k: inst.Response(200)), root=root,
                        etc=tmp_path / "etc", os_release=release, operator="yogsotho", **fields)


def texts(findings, level=None):
    return [f.text for f in findings if level is None or f.level == level]


def titles(p):
    return [step.title for step in p.steps]


# -- reading and writing .env ---------------------------------------------------------------

def test_env_is_read_as_docker_compose_reads_it():
    text = "\n".join([
        "# a comment", "", "PLAIN=value", "export EXPORTED=1", "SPACED = padded ",
        "COMMENTED=abc # note", "HASH_IN_VALUE=abc#def", "SINGLE='a b # c'", 'DOUBLE="x y"', "EMPTY=",
        "not a setting"])
    assert {k: v for k, v, _ in inst.parse_env(text)} == {
        "PLAIN": "value", "EXPORTED": "1", "SPACED": "padded", "COMMENTED": "abc", "HASH_IN_VALUE": "abc#def",
        "SINGLE": "a b # c", "DOUBLE": "x y", "EMPTY": ""}


@pytest.mark.parametrize("value", ["abc123", TOKEN, "https://hc-ping.com/x-y", "with space", "a$b", "x#y", ""])
def test_written_lines_read_back_as_the_same_value(value):
    (name, read, _), = inst.parse_env(inst.env_line("NAME", value))
    assert (name, read) == ("NAME", value)


def test_a_value_with_a_quote_is_refused():
    with pytest.raises(ValueError, match="NAME"):
        inst.env_line("NAME", "it's")


# -- what is wrong in .env ------------------------------------------------------------------

def test_a_complete_configuration_has_no_problem():
    assert inst.env_problems(GOOD) == {}


def test_the_example_file_is_all_placeholders():
    example = {k: v for k, v, _ in inst.parse_env((ROOT / ".env.example").read_text())}
    problems = inst.env_problems(example)
    for name in ("FLASK_SECRET_KEY", "JWT_SECRET_KEY", "API_KEY_PEPPER", "DB_PASSWORD", "REDIS_PASSWORD",
                 "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALLOWED_ORIGINS"):
        assert problems[name] == ("fix", "still the example value"), name


@pytest.mark.parametrize("change, name, text", [
    ({"FLASK_SECRET_KEY": ""}, "FLASK_SECRET_KEY", "missing"),
    ({"API_KEY_PEPPER": "short"}, "API_KEY_PEPPER", "shorter than 32"),
    ({"JWT_SECRET_KEY": "a" * 64}, "JWT_SECRET_KEY", "same as FLASK_SECRET_KEY"),
    ({"DB_PASSWORD": "abc@def:1234567890"}, "DB_PASSWORD", "letters and digits only"),
    ({"ALLOWED_ORIGINS": f"https://{NAME}:9443/"}, "ALLOWED_ORIGINS", "no trailing /"),
    ({"API_PORT": "80"}, "API_PORT", "1024 to 65535"),
    ({"ALERT_TELEGRAM_CHAT_ID": ""}, "ALERT_TELEGRAM_CHAT_ID", "together"),
    ({"ALERT_TELEGRAM_BOT_TOKEN": "nonsense"}, "ALERT_TELEGRAM_BOT_TOKEN", "not a bot token"),
    ({"ALERT_TELEGRAM_BOT_TOKEN": "", "ALERT_TELEGRAM_CHAT_ID": ""}, "TELEGRAM_COMMANDS", "needs the bot token"),
    ({"TELEGRAM_COMMANDS": "yes"}, "TELEGRAM_COMMANDS", "true or false"),
    ({"HEARTBEAT_URL": "http://hc-ping.com/x"}, "HEARTBEAT_URL", "https://"),
    ({"BACKUP_HEARTBEAT_URL": GOOD["HEARTBEAT_URL"]}, "BACKUP_HEARTBEAT_URL", "own check"),
    ({"ALPACA_SECRET_KEY": "<paste it>"}, "ALPACA_SECRET_KEY", "example value"),
])
def test_problems_that_stop_the_stack_or_a_feature(change, name, text):
    level, message = inst.env_problems({**GOOD, **change})[name]
    assert level == "fix" and text in message


@pytest.mark.parametrize("change, name, text", [
    ({"ALPACA_API_KEY": "AKLIVEKEY123"}, "ALPACA_API_KEY", "live key"),
    ({"TRUSTED_PROXY_COUNT": "0"}, "TRUSTED_PROXY_COUNT", "lock everyone out"),
    ({"ENGINE_AUTOSTART": "true"}, "ENGINE_AUTOSTART", "restarts by itself"),
    ({"HEARTBEAT_URL": ""}, "HEARTBEAT_URL", "not set"),
    ({"HEARTBEAT_URL": "https://example.org/ping"}, "HEARTBEAT_URL", "hc-ping.com"),
])
def test_warnings(change, name, text):
    level, message = inst.env_problems({**GOOD, **change})[name]
    assert level == "warn" and text in message


def test_problem_messages_never_contain_the_value():
    broken = {name: f"{value}@x" for name, value in GOOD.items()}
    for _, message in inst.env_problems(broken).values():
        assert not any(value in message for value in broken.values() if len(value) > 5)


# -- ports and Tailscale Serve ---------------------------------------------------------------

def test_listeners_are_read_from_ss():
    found = inst.parse_listeners(SS)
    assert inst.Listener("0.0.0.0", 80, "nginx") in found
    assert inst.Listener("::", 22, "sshd") in found
    assert inst.Listener("127.0.0.53%lo", 53, "systemd-resolve") in found
    assert len(found) == 5


def test_the_dashboard_gets_a_port_funnel_cannot_expose():
    listeners = inst.parse_listeners(SS)
    assert inst.choose_serve_port(OPERATOR_SERVE, listeners) == 9443
    taken = {**OPERATOR_SERVE, "TCP": {**OPERATOR_SERVE["TCP"], "9443": {"HTTPS": True}}}
    assert inst.choose_serve_port(taken, listeners) == 9444
    assert inst.choose_serve_port(OPERATOR_SERVE, listeners + [inst.Listener("0.0.0.0", 9443, "nginx")]) == 9444
    assert not set(inst.SERVE_PORTS) & set(inst.FUNNEL_PORTS)


def test_the_dashboard_is_found_in_serve_and_funnel_is_noticed():
    served = {**OPERATOR_SERVE, "Web": {**OPERATOR_SERVE["Web"],
                                        f"{NAME}:9443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:5000"}}}}}
    assert inst.find_dashboard(served, 5000) == (9443, False)
    assert inst.find_dashboard({**served, "AllowFunnel": {f"{NAME}:9443": True}}, 5000) == (9443, True)
    assert inst.find_dashboard(OPERATOR_SERVE, 5000) == (None, False)
    assert inst.find_dashboard(OPERATOR_SERVE, 3000) == (8443, True)  # the operator's own service, not ours


def test_the_api_port_moves_only_when_something_else_has_it():
    nginx_on_5000 = [inst.Listener("0.0.0.0", 5000, "nginx"), inst.Listener("127.0.0.1", 5001, "x")]
    assert inst.choose_api_port([], 5000, ours=False) == 5000
    assert inst.choose_api_port(nginx_on_5000, 5000, ours=True) == 5000
    assert inst.choose_api_port(nginx_on_5000, 5000, ours=False) == 5002


def _after(port=9443, target="http://127.0.0.1:5000", **edits):
    after = json.loads(json.dumps(OPERATOR_SERVE))
    after["TCP"][str(port)] = {"HTTPS": True}
    after["Web"][f"{NAME}:{port}"] = {"Handlers": {"/": {"Proxy": target}}}
    for section, value in edits.items():
        after[section] = value
    return after


def test_serve_changes_accepts_only_the_new_private_entry():
    hostport = f"{NAME}:9443"
    assert inst.serve_changes(OPERATOR_SERVE, _after(), hostport, "http://127.0.0.1:5000") == []
    unfunnelled = _after(AllowFunnel={f"{NAME}:8443": True})
    assert inst.serve_changes(OPERATOR_SERVE, unfunnelled, hostport, "http://127.0.0.1:5000") == [
        f"AllowFunnel {NAME}:443 changed"]
    replaced = _after()
    replaced["Web"][f"{NAME}:443"] = {"Handlers": {"/": {"Proxy": "http://127.0.0.1:5000"}}}
    assert f"Web {NAME}:443 changed" in inst.serve_changes(OPERATOR_SERVE, replaced, hostport, "http://127.0.0.1:5000")
    public = _after(AllowFunnel={**OPERATOR_SERVE["AllowFunnel"], hostport: True})
    assert f"{hostport} is public (Funnel)" in inst.serve_changes(OPERATOR_SERVE, public, hostport,
                                                                 "http://127.0.0.1:5000")


# -- what the setup finds, and the plan it makes ------------------------------------------------

def test_the_operators_server(tmp_path):
    m = inst.inspect(machine(tmp_path, operator_machine()))
    found = texts(m.findings)
    assert "Docker version 29.1.3, build 29.1.3-0ubuntu4.1 (Ubuntu's docker.io package)" in found
    assert "yogsotho is not in the docker group: Docker needs sudo" in found
    assert f"serves https://{NAME}:443/ -> http://127.0.0.1:80 (public, Funnel); left as it is" in found
    assert f"serves https://{NAME}:8443/ -> http://127.0.0.1:3000 (public, Funnel); left as it is" in found
    assert "port 5000 is free for the API (it listens on 127.0.0.1 only)" in found
    p = inst.plan(m)
    assert p.blockers == []
    assert titles(p) == [
        f"Serve the dashboard at https://{NAME}:9443: tailnet only (Funnel cannot use this port), and nothing "
        "Tailscale serves now changes",
        "Create .env: the passwords are generated; you give the Alpaca keys, the Telegram bot and the "
        "healthchecks.io URLs, and each is tested",
        "Build and start Nocturna (docker compose up -d --build), and wait until every service is healthy",
        "Create your dashboard account (you choose the username and password)",
        "Send a test alert to Telegram",
    ]
    assert any("Disable Key Expiry" in advice for advice in p.advice)
    assert not any("usermod" in call or "gpasswd" in call for call in m.run.calls)


def test_a_bare_ubuntu_machine(tmp_path):
    answers = {("timedatectl",): ok("no\n"), ("ss", "-ltnpH"): ok("")}
    m = inst.inspect(machine(tmp_path, answers))
    p = inst.plan(m)
    assert p.blockers == []
    assert titles(p)[:6] == [
        "Turn on network time (timedatectl set-ntp true)",
        "Install git, curl, ca-certificates (apt)",
        "Install Docker Engine, Compose and buildx from Docker's repository",
        "Install Tailscale (Tailscale's install script; it adds Tailscale's apt repository)",
        "Connect this machine to your tailnet (tailscale up prints a link: open it and sign in)",
        "Serve the dashboard at a private HTTPS port from 9443 up: tailnet only (Funnel cannot use this port), "
        "and nothing Tailscale serves now changes",
    ]


@pytest.mark.parametrize("changes, text", [
    ({("snap", "list", "docker"): ok("docker 28\n")}, "snap store"),
    ({("dpkg-query", "-W", "-f=${Status}", "docker.io"): missing(), ("docker", "--version"): missing(),
      ("dpkg-query", "-W", "-f=${Status}", "containerd"): INSTALLED}, "conflict with it are: containerd"),
    ({("tailscale", "status"): tailscale_status(state="NeedsLogin")}, "not logged in"),
    ({("tailscale", "status"): tailscale_status(state="Stopped")}, "stopped on this machine"),
    ({("tailscale", "status"): inst.Result(1, "", "failed to connect to local tailscaled")}, "service is not running"),
    ({("tailscale", "serve", "status"): ok(json.dumps(
        {"TCP": {"9443": {"HTTPS": True}}, "Web": {f"{NAME}:9443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:5000"}}}},
         "AllowFunnel": {f"{NAME}:9443": True}}))}, "Funnel on, so it is public"),
])
def test_what_stops_the_setup_before_any_change(tmp_path, changes, text):
    m = inst.inspect(machine(tmp_path, operator_machine(**{"_": None}) | changes))
    p = inst.plan(m)
    assert any(text in blocker for blocker in p.blockers), p.blockers


def test_not_ubuntu(tmp_path):
    (tmp_path / "os-release").write_text("ID=fedora\nPRETTY_NAME=Fedora\n")
    p = inst.plan(inst.inspect(machine(tmp_path, operator_machine())))
    assert p.blockers and p.steps == []


@pytest.mark.parametrize("changes, title", [
    ({("systemctl", "is-enabled", "docker"): ok("disabled\n"), ("docker", "info"): inst.Result(1)},
     "Start Docker, and start it at boot (systemctl enable --now docker)"),
    ({("docker", "compose", "version"): inst.Result(1, "", "unknown command")}, "Install Docker Compose v2 (apt)"),
    ({("docker", "buildx", "version"): inst.Result(1)}, "Install Docker buildx (apt)"),
    ({("tailscale", "status"): tailscale_status(certs=False)},
     "Turn on MagicDNS and HTTPS certificates in the Tailscale admin console (you do it; the setup waits and checks)"),
])
def test_docker_and_tailscale_gaps_become_steps(tmp_path, changes, title):
    p = inst.plan(inst.inspect(machine(tmp_path, operator_machine() | changes)))
    assert title in titles(p)


def test_a_port_5000_already_in_use_moves_the_api(tmp_path):
    ss = ok(SS + '\nLISTEN 0 4096 0.0.0.0:5000 0.0.0.0:* users:(("python3",pid=999,fd=3))')
    m = inst.inspect(machine(tmp_path, operator_machine() | {("ss", "-ltnpH"): ss}))
    assert "port 5000 is used by python3" in texts(m.findings, "fix")
    p = inst.plan(m)
    assert "Use port 5001 on 127.0.0.1 for the API (5000 is used by python3)" in titles(p)


def test_a_finished_setup(tmp_path):
    served = _after()
    answers = operator_machine() | {
        ("tailscale", "status"): tailscale_status(), ("tailscale", "serve", "status"): ok(json.dumps(served)),
        ("docker", "volume", "ls"): ok("nocturna-trading-system_postgres-data\n"), ("docker", "compose", "config"): ok(),
        ("docker", "image", "inspect"): ok("[]"), ("docker", "compose", "run"): ok(),
        ("docker", "compose", "port"): ok("127.0.0.1:5000\n"),
        ("docker", "compose", "ps"): ok("\n".join(json.dumps({"Service": s, "State": "running", "Health": "healthy"})
                                                  for s in ("api", "engine", "postgres", "redis"))),
        ("docker", "compose", "exec", "-T", "api", "flask", "--app", "src.main", "list-admins"): ok("yogsotho\n"),
        ("ss", "-ltnpH"): ok(SS + '\nLISTEN 0 4096 127.0.0.1:5000 0.0.0.0:* users:(("docker-proxy",pid=5,fd=4))'),
    }
    m = machine(tmp_path, answers)
    write(m.env_file, GOOD)
    inspect = inst.inspect(m)
    found = texts(inspect.findings)
    assert texts(inspect.findings, "fix") == []
    assert f"the dashboard is served privately at https://{NAME}:9443" in found
    assert "port 5000: the Nocturna API" in found and "the app accepts the settings" in found
    assert "dashboard account: yogsotho" in found
    p = inst.plan(m)
    assert "Create your dashboard account (you choose the username and password)" not in titles(p)
    assert not any(t.startswith("Serve the dashboard") for t in titles(p))
    assert not any(value in inst.report(inspect.findings) for value in inst.secret_values(GOOD))


def test_the_report_never_shows_a_secret(tmp_path):
    leaked = f"ALERT_NTFY_URL: bad value {GOOD['ALPACA_SECRET_KEY']}"
    answers = operator_machine() | {("docker", "compose", "config"): ok(), ("docker", "image", "inspect"): ok("[]"),
                                    ("docker", "compose", "run"): inst.Result(1, leaked + "\n")}
    m = machine(tmp_path, answers)
    write(m.env_file, GOOD, mode=0o644)
    report = inst.report(inst.inspect(m).findings)
    assert "the app refuses the settings: ALERT_NTFY_URL: bad value ***" in report
    assert "readable by other users (mode 644)" in report
    assert not any(value in report for value in inst.secret_values(GOOD))


# -- the steps ------------------------------------------------------------------------------------

class Script:
    """A dialog that answers from lists and records what it said."""

    def __init__(self, answers=(), secrets=()):
        self.answers, self.secrets, self.said = list(answers), list(secrets), []

    def dialog(self):
        return inst.Dialog(ask=self._ask, ask_secret=self._secret, say=self.said.append)

    def _ask(self, question):
        self.said.append(question)
        return self.answers.pop(0)

    def _secret(self, question):
        self.said.append(question)
        return self.secrets.pop(0)

    @property
    def text(self):
        return "\n".join(self.said)


def setup_for(tmp_path, answers, script=None, http=None):
    m = inst.inspect(machine(tmp_path, answers, http=http))
    return inst.Setup(m, (script or Script()).dialog()), m


def stateful_serve(before, after):
    """`tailscale serve status` answers `before` until `tailscale serve --bg` ran, then `after`."""
    state = {"config": before}

    def serve(args):
        if "--bg" in args:
            state["config"] = after
            return ok()
        if args[2:3] == ["status"]:
            return ok(json.dumps(state["config"]))
        return ok()
    return {("tailscale", "serve"): serve, ("tailscale", "serve", "status"): serve}


def test_serving_the_dashboard_adds_one_private_entry(tmp_path):
    s, m = setup_for(tmp_path, operator_machine() | stateful_serve(OPERATOR_SERVE, _after()))
    s.serve_dashboard()
    assert m.run.ran("tailscale", "serve", "--bg") == [["tailscale", "serve", "--bg", "--https=9443",
                                                        "http://127.0.0.1:5000"]]
    assert m.origin == f"https://{NAME}:9443"


def test_a_serve_change_that_touches_anything_else_is_undone(tmp_path):
    broken = _after(AllowFunnel={f"{NAME}:8443": True})  # Funnel removed from 443
    s, m = setup_for(tmp_path, operator_machine() | stateful_serve(OPERATOR_SERVE, broken))
    with pytest.raises(inst.StepFailedError, match=f"AllowFunnel {NAME}:443 changed"):
        s.serve_dashboard()
    assert ["tailscale", "serve", "--https=9443", "off"] in m.run.calls


def test_compose_and_buildx_come_from_the_engines_family(tmp_path):
    policy = {("apt-cache", "policy", "docker-compose-v2"): ok("  Candidate: 2.40.0\n"),
              ("apt-cache", "policy", "docker-compose-plugin"): ok("  Candidate: (none)\n"),
              ("apt-cache", "policy", "docker-buildx"): ok("  Candidate: 0.30\n"),
              ("apt-get",): ok(), ("env",): ok()}
    s, m = setup_for(tmp_path, operator_machine() | policy)
    s.install_compose()
    s.install_buildx()
    installs = [c[-1] for c in m.run.ran("env", "DEBIAN_FRONTEND=noninteractive")]
    assert installs == ["docker-compose-v2", "docker-buildx"]
    assert len(m.run.ran("apt-get", "-o", "DPkg::Lock::Timeout=300", "update")) == 1


def test_installing_docker_uses_dockers_repository(tmp_path):
    answers = {("timedatectl",): ok("yes\n"), ("install",): ok(), ("curl",): ok(), ("dpkg",): ok("amd64\n"),
               ("apt-get",): ok(), ("env",): ok(), ("systemctl",): ok(), ("docker", "info"): ok("29\n")}
    s, m = setup_for(tmp_path, answers, http=lambda *a, **k: inst.Response(200))
    keyrings = m.etc / "apt" / "keyrings"
    keyrings.mkdir(parents=True)
    (keyrings / "docker.asc").write_text("key")
    (m.etc / "apt" / "sources.list.d").mkdir()
    s.install_docker()
    sources = (m.etc / "apt" / "sources.list.d" / "docker.sources").read_text()
    assert "Suites: resolute" in sources and "Architectures: amd64" in sources and "docker.asc" in sources
    assert m.run.ran("env")[-1][-5:] == ["docker-ce", "docker-ce-cli", "containerd.io", "docker-buildx-plugin",
                                         "docker-compose-plugin"]
    assert ["systemctl", "enable", "--now", "docker"] in m.run.calls


def test_docker_is_not_installed_for_a_release_dockers_repository_lacks(tmp_path):
    s, _ = setup_for(tmp_path, {("timedatectl",): ok("yes\n")}, http=lambda *a, **k: inst.Response(404))
    with pytest.raises(inst.StepFailedError, match="no packages for Ubuntu resolute"):
        s.install_docker()


def test_starting_the_stack(tmp_path):
    answers = operator_machine() | {("docker", "compose", "up"): ok()}
    s, m = setup_for(tmp_path, answers, http=lambda *a, **k: inst.Response(200, b'{"status": "ready"}'))
    s.start_stack()
    assert m.run.ran("docker", "compose", "up") == [["docker", "compose", "up", "-d", "--build", "--wait",
                                                     "--wait-timeout", "600"]]
    s, m = setup_for(tmp_path, operator_machine() | {("docker", "compose", "up"): inst.Result(1)})
    with pytest.raises(inst.StepFailedError, match="did not become healthy"):
        s.start_stack()


CREATE = ("docker", "compose", "exec", "-T", "-e", "LOG_LEVEL=WARNING")


def test_creating_the_account(tmp_path):
    answers = operator_machine() | {("docker", "compose", "exec", "-T", "api"): ok(""), CREATE: ok()}
    script = Script(answers=["bad name!", "", "not-an-email", "ops@home.lan"],
                    secrets=["short1!A", "nosymbolsHere12", "Night-wolf-2026", "Night-wolf-2025",
                             "Night-wolf-2026", "Night-wolf-2026"])
    s, m = setup_for(tmp_path, answers, script)
    kwargs = []
    run = m.run
    m.run = lambda args, **k: (kwargs.append(k), run(args, **k))[1]
    s.create_account()
    assert run.ran(*CREATE) == [
        [*CREATE, "-e", "ADMIN_PASSWORD", "api", "flask", "--app", "src.main", "create-admin", "--username",
         "yogsotho", "--email", "ops@home.lan"]]
    assert kwargs[-1] == {"env": {"ADMIN_PASSWORD": "Night-wolf-2026"}}  # never on the command line
    assert "Night-wolf-2026" in m.hide and m.admins == ["yogsotho"]
    assert "Not accepted: it has 8 characters, not 12 or more." in script.text.replace(" The rules", "\n")
    assert "Not accepted: it has no symbol." in script.text.replace(" The rules", "\n")
    assert "The two did not match. Type it again." in script.said
    assert "(any character that is not a letter, a digit or a space" in script.said[script.said.index(
        "Password (hidden): ") - 1]


def test_a_refused_account_says_why(tmp_path):
    answers = operator_machine() | {("docker", "compose", "exec", "-T", "api"): ok(""), CREATE: inst.Result(
        1, "", '{"level": "WARNING", "message": "x"}\nError: A user with username \'yogsotho\' already exists\n')}
    script = Script(answers=["", "ops@home.lan"], secrets=["Night-wolf-2026", "Night-wolf-2026"])
    s, _ = setup_for(tmp_path, answers, script)
    with pytest.raises(inst.StepFailedError, match=r"not created \(Error: A user with username"):
        s.create_account()


@pytest.mark.parametrize("password, problem", [
    ("Aa1!" * 3, ""), ("Night_wolf_2026", ""), ("Night wolf 2026", "it has no symbol"),
    ("night-wolf-2026", "it has no upper-case letter"), ("NIGHT-WOLF-2026", "it has no lower-case letter"),
    ("Night-wolf-two", "it has no digit"), ("A1!a" * 33, "it has more than 128 characters"),
])
def test_the_setup_checks_the_password_as_the_app_does(password, problem):
    from src.utils.validators import validate_registration
    assert inst.password_problem(password) == problem
    valid, _, _ = validate_registration({"username": "yogsotho", "email": "ops@home.lan", "password": password,
                                         "password_confirm": password})
    assert valid == (problem == "")


def test_an_existing_account_is_left_alone(tmp_path):
    answers = operator_machine() | {("docker", "compose", "exec", "-T", "api"): ok("yogsotho\n")}
    s, m = setup_for(tmp_path, answers)
    s.create_account()
    assert not m.run.ran(*CREATE)


def test_the_test_alert(tmp_path):
    answers = operator_machine() | {("docker", "compose", "exec", "-T", "api"): ok("telegram: sent\n")}
    script = Script(answers=[""])
    s, m = setup_for(tmp_path, answers, script)
    m.env = dict(GOOD)
    s.test_alert()
    assert "  telegram: sent" in script.said
    s, m = setup_for(tmp_path, operator_machine() | {("docker", "compose", "exec", "-T", "api"):
                                                      inst.Result(1, "telegram: FAILED (Unauthorized)\n")})
    m.env = dict(GOOD)
    with pytest.raises(inst.StepFailedError, match="not sent"):
        s.test_alert()


def changes(calls):
    """The commands that change the machine; everything else the setup runs only reads."""
    return [c for c in calls if c[0] in ("apt-get", "env", "install", "curl", "sh", "timedatectl") and
            c[:2] != ["timedatectl", "show"] or c[:2] == ["systemctl", "enable"] or
            c[:3] in (["tailscale", "serve", "--bg"], ["tailscale", "up"], ["docker", "compose", "up"]) or
            "create-admin" in c]


def test_check_mode_changes_nothing(tmp_path):
    m = machine(tmp_path, operator_machine())
    script = Script()
    assert inst.main(["--check"], d=script.dialog(), m=m) == 1
    assert "To do:" in script.text and "  1. Serve the dashboard at" in script.text
    assert changes(m.run.calls) == []


def test_a_blocked_setup_changes_nothing(tmp_path):
    m = machine(tmp_path, operator_machine() | {("snap", "list", "docker"): ok("docker\n")})
    script = Script()
    assert inst.main([], d=script.dialog(), m=m) == 1
    assert "Stopped before changing anything" in script.text
    assert changes(m.run.calls) == []


def test_declining_the_plan_changes_nothing(tmp_path):
    m = machine(tmp_path, operator_machine())
    assert inst.main([], d=Script(answers=["n"]).dialog(), m=m) == 0
    assert not m.run.ran("tailscale", "serve", "--bg")


def test_a_failed_step_stops_the_run_and_says_where(tmp_path):
    broken = _after(AllowFunnel={})
    m = machine(tmp_path, operator_machine() | stateful_serve(OPERATOR_SERVE, broken))
    script = Script(answers=[""])
    assert inst.main([], d=script.dialog(), m=m) == 1
    assert "Stopped at step 1:" in script.text and "carries on from here" in script.text
    assert not m.run.ran("docker", "compose", "up")


# -- writing .env with the operator ---------------------------------------------------------------

class Services:
    """Alpaca and Telegram as the setup sees them."""

    def __init__(self, alpaca_key="PKABCDEFGH123456", chats=({"id": 424242, "type": "private", "first_name": "Op"},)):
        self.alpaca_key, self.chats = alpaca_key, list(chats)

    def __call__(self, method, url, headers=None, data=None, timeout=15):
        if "alpaca" in url:
            if headers["APCA-API-KEY-ID"] == self.alpaca_key:
                return inst.Response(200, json.dumps({"account_number": "PA1", "status": "ACTIVE",
                                                      "equity": "100000", "currency": "USD"}).encode())
            return inst.Response(401, b"{}")
        method_name = url.rsplit("/", 1)[-1]
        if method_name == "getMe":
            good = url.split("/bot", 1)[1].startswith(TOKEN)
            return inst.Response(200 if good else 401, json.dumps(
                {"ok": good, "result": {"username": "nocturna_bot"}} if good else
                {"ok": False, "description": "Unauthorized"}).encode())
        if method_name == "getUpdates":
            updates = [{"update_id": n, "message": {"chat": chat}} for n, chat in enumerate(self.chats)]
            return inst.Response(200, json.dumps({"ok": True, "result": updates}).encode())
        return inst.Response(200, b'{"ok": true, "result": {}}')


ORIGIN = f"https://{NAME}:9443"


def build(tmp_path, script, services=None, answers=None, origin=ORIGIN, api_port=5000):
    env = tmp_path / ".env"
    code = inst.build_env(env, Runner(answers or {}), services or Services(), script.dialog(), origin=origin,
                          api_port=api_port)
    return code, env


def test_a_first_run(tmp_path):
    script = Script(answers=["", "", "", "", "https://hc-ping.com/1111", "https://hc-ping.com/2222"],
                    secrets=["PKWRONG", "wrong", "PKABCDEFGH123456", "s3cr3tAlpacaValue0123456789", TOKEN])
    code, env = build(tmp_path, script)
    assert code == 0 and stat.S_IMODE(env.stat().st_mode) == 0o600
    values = inst.env_values(env)
    for name in ("FLASK_SECRET_KEY", "JWT_SECRET_KEY", "API_KEY_PEPPER"):
        assert re.fullmatch(r"[0-9a-f]{64}", values[name])
    assert re.fullmatch(r"[0-9a-f]{32}", values["DB_PASSWORD"]) and len(values["REDIS_PASSWORD"]) == 32
    assert values["JWT_SECRET_KEY"] != values["FLASK_SECRET_KEY"]
    assert values["ALPACA_API_KEY"] == "PKABCDEFGH123456"
    assert values["ALERT_TELEGRAM_BOT_TOKEN"] == TOKEN and values["ALERT_TELEGRAM_CHAT_ID"] == "424242"
    assert values["TELEGRAM_COMMANDS"] == "true" and values["ENGINE_AUTOSTART"] == "false"
    assert values["ALLOWED_ORIGINS"] == ORIGIN and values["TRUSTED_PROXY_COUNT"] == "1"
    assert values["API_PORT"] == "5000"
    assert values["HEARTBEAT_URL"].endswith("1111") and values["BACKUP_HEARTBEAT_URL"].endswith("2222")
    assert inst.env_problems(values) == {}
    assert "Alpaca refused these keys" in script.text and "Found your chat with Op." in script.text
    assert not any(value in script.text for value in inst.secret_values(values) if "hc-ping" not in value)


def test_a_second_run_keeps_every_secret_and_backs_up_the_file(tmp_path):
    env = write(tmp_path / ".env", {**GOOD, "RISK_MAX_DRAWDOWN": "0.08", "DATABASE_URL": "postgresql://x",
                                    "ADMIN_EMAIL": "admin@yourdomain.com"})
    code, _ = build(tmp_path, Script(answers=["", "", "", "", "", ""]))
    assert code == 0
    values = inst.env_values(env)
    assert {k: values[k] for k in GOOD} == GOOD
    assert values["RISK_MAX_DRAWDOWN"] == "0.08"
    assert "DATABASE_URL" not in values and "ADMIN_EMAIL" not in values
    backups = list(tmp_path.glob(".env.backup-*"))
    assert len(backups) == 1 and stat.S_IMODE(backups[0].stat().st_mode) == 0o600


def test_the_example_file_becomes_a_working_configuration(tmp_path):
    (tmp_path / ".env").write_text((ROOT / ".env.example").read_text())
    script = Script(answers=["", "", "", "", ""], secrets=["PKABCDEFGH123456", "s3cr3tAlpacaValue0123456789", TOKEN])
    code, env = build(tmp_path, script)
    values = inst.env_values(env)
    assert code == 0 and [name for name, (level, _) in inst.env_problems(values).items() if level == "fix"] == []
    assert not any(name in values for name in inst.STACK_SETS)


def test_the_database_password_is_never_replaced(tmp_path):
    write(tmp_path / ".env", {**GOOD, "DB_PASSWORD": "not-url-safe@1"})
    script = Script(answers=["", "", "", "", "", ""])
    _, env = build(tmp_path, script)
    assert inst.env_values(env)["DB_PASSWORD"] == "not-url-safe@1"
    assert "Left unchanged, since the database uses it" in script.text


def test_an_existing_database_without_its_password_stops_it(tmp_path):
    script = Script()
    code, env = build(tmp_path, script, answers={("docker", "volume", "ls"): ok("x_postgres-data\n")})
    assert code == 1 and not env.exists() and "PostgreSQL keeps the password" in script.text


def test_alpaca_keys_that_do_not_work_can_be_left_as_they_were(tmp_path):
    script = Script(answers=["n", "n", "n", "", ""], secrets=["PKBAD", "bad"])
    code, env = build(tmp_path, script, services=Services(alpaca_key="PKOTHER"))
    assert code == 0 and inst.env_values(env)["ALPACA_API_KEY"] == ""
    assert "Left the Alpaca keys as they were." in script.text


def test_telegram_can_be_skipped_and_several_chats_are_offered(tmp_path):
    script = Script(answers=["", "skip", "", ""], secrets=["PKABCDEFGH123456", "secret", TOKEN])
    _, env = build(tmp_path, script)
    values = inst.env_values(env)
    assert values["ALERT_TELEGRAM_BOT_TOKEN"] == "" and "Telegram left as it was." in script.text
    two = Services(chats=[{"id": 1, "type": "private", "first_name": "A"},
                          {"id": 2, "type": "private", "first_name": "B", "username": "bee"}])
    script = Script(answers=["", "", "2", "n", "", ""], secrets=["PKABCDEFGH123456", "secret", TOKEN])
    (tmp_path / "two").mkdir()
    _, env = build(tmp_path / "two", script, services=two)
    values = inst.env_values(env)
    assert values["ALERT_TELEGRAM_CHAT_ID"] == "2" and values["TELEGRAM_COMMANDS"] == "false"
    assert "B (@bee)" in script.text


def test_before_tailscale_serves_it_the_address_waits(tmp_path):
    script = Script(answers=["", "", "", "", ""], secrets=["PKABCDEFGH123456", "secret", TOKEN])
    _, env = build(tmp_path, script, origin=None, api_port=5001)
    values = inst.env_values(env)
    assert values["ALLOWED_ORIGINS"] == "http://127.0.0.1:5001" and values["API_PORT"] == "5001"
    assert "The dashboard is not served yet" in script.text


# -- the files around it ------------------------------------------------------------------------

def test_the_api_port_is_configurable_in_compose():
    assert '"${API_BIND:-127.0.0.1}:${API_PORT:-5000}:5000"' in (ROOT / "docker-compose.yml").read_text()


def test_setup_sh_runs_the_installer_as_root():
    script = (ROOT / "setup.sh").read_text()
    assert script.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in script
    assert 'exec sudo bash "${here}/setup.sh" "$@"' in script
    assert 'exec python3 -I "${here}/setup/installer.py" "$@"' in script
    assert (ROOT / "setup.sh").stat().st_mode & 0o111


def test_the_guide_explains_every_setting_the_setup_writes():
    guide = (ROOT / "docs" / "HOME_SERVER.md").read_text()
    missing_names = [var.name for var in inst.VARS if f"`{var.name}`" not in guide]
    assert not missing_names, missing_names
