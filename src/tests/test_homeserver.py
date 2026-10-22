"""
Tests for ops/homeserver.py, the home server check and .env builder: what it
reports for each machine state, the .env it writes, and that it never shows a
secret.
"""

import importlib.util
import json
import re
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("homeserver", ROOT / "ops" / "homeserver.py")
hs = importlib.util.module_from_spec(_spec)
sys.modules["homeserver"] = hs
_spec.loader.exec_module(hs)

TOKEN = "123456789:AAFD39kkdpWt3ywyRZergyOLMaJhac60qc"
GOOD = {
    "FLASK_SECRET_KEY": "a" * 64, "JWT_SECRET_KEY": "b" * 64, "API_KEY_PEPPER": "c" * 64,
    "DB_PASSWORD": "d" * 32, "REDIS_PASSWORD": "e" * 32,
    "ALPACA_API_KEY": "PKABCDEFGH123456", "ALPACA_SECRET_KEY": "s3cr3tAlpacaValue0123456789",
    "ALLOWED_ORIGINS": "https://box.tail1234.ts.net", "TRUSTED_PROXY_COUNT": "1", "ENGINE_AUTOSTART": "false",
    "ALERT_TELEGRAM_BOT_TOKEN": TOKEN, "ALERT_TELEGRAM_CHAT_ID": "424242", "TELEGRAM_COMMANDS": "true",
    "HEARTBEAT_URL": "https://hc-ping.com/11111111-aaaa", "BACKUP_HEARTBEAT_URL": "https://hc-ping.com/22222222-bbbb",
}


def write(path, values):
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    path.chmod(0o600)
    return path


class Runner:
    """Answers commands by their leading words; anything else is not installed."""

    def __init__(self, answers=None):
        self.answers = dict(answers or {})
        self.calls = []

    def __call__(self, args):
        self.calls.append(list(args))
        for n in range(len(args), 0, -1):
            if tuple(args[:n]) in self.answers:
                return self.answers[tuple(args[:n])]
        return hs.Result(127, "", f"{args[0]}: not found")


def ok(out=""):
    return hs.Result(0, out)


def tailscale_status(state="Running", name="box.tail1234.ts.net.", magic=True, certs=True, expiry=None):
    me = {"DNSName": name}
    if expiry:
        me["KeyExpiry"] = expiry
    return ok(json.dumps({"BackendState": state, "Self": me, "CurrentTailnet": {"MagicDNSEnabled": magic},
                          "CertDomains": [name.rstrip(".")] if certs else None}))


SERVING = ok(json.dumps({"TCP": {"443": {"HTTPS": True}},
                         "Web": {"box.tail1234.ts.net:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:5000"}}}}}))

HEALTHY_MACHINE = {
    ("timedatectl",): ok("yes\n"), ("git", "--version"): ok("git version 2.51.0\n"),
    ("curl", "--version"): ok("curl 8.14.1 (x86_64-pc-linux-gnu) libcurl/8.14.1\n"),
    ("docker", "--version"): ok("Docker version 29.0.0, build x\n"), ("docker", "info"): ok("29.0.0\n"),
    ("systemctl", "is-enabled", "docker"): ok("enabled\n"), ("docker", "compose", "version"): ok("v2.40.0\n"),
    ("tailscale", "version"): ok("1.90.0\n"), ("tailscale", "status"): tailscale_status(),
    ("tailscale", "serve"): SERVING,
    ("docker", "compose", "config"): ok(), ("docker", "image", "inspect"): ok("[]"),
    ("docker", "compose", "run"): ok(),
    ("docker", "compose", "ps"): ok('{"Service":"api","State":"running","Health":"healthy"}\n'
                                    '{"Service":"engine","State":"running","Health":"healthy"}\n'),
}


def machine(tmp_path, answers=HEALTHY_MACHINE, listening=False, http=None):
    return hs.Machine(Runner(answers), http or (lambda *a, **k: hs.Response(0)), root=tmp_path,
                      listening=lambda port: listening)


def texts(findings, level=None):
    return [f.text for f in findings if level is None or f.level == level]


# -- reading and writing .env ---------------------------------------------------------------

def test_env_is_read_as_docker_compose_reads_it():
    text = "\n".join([
        "# a comment", "", "PLAIN=value", "export EXPORTED=1", "SPACED = padded ",
        "COMMENTED=abc # note", "HASH_IN_VALUE=abc#def", "SINGLE='a b # c'", 'DOUBLE="x y"', "EMPTY=",
        "not a setting"])
    assert {k: v for k, v, _ in hs.parse_env(text)} == {
        "PLAIN": "value", "EXPORTED": "1", "SPACED": "padded", "COMMENTED": "abc", "HASH_IN_VALUE": "abc#def",
        "SINGLE": "a b # c", "DOUBLE": "x y", "EMPTY": ""}


@pytest.mark.parametrize("value", ["abc123", TOKEN, "https://hc-ping.com/x-y", "with space", "a$b", "x#y", ""])
def test_written_lines_read_back_as_the_same_value(value):
    (name, read, _), = hs.parse_env(hs.env_line("NAME", value))
    assert (name, read) == ("NAME", value)


def test_a_value_with_a_quote_is_refused():
    with pytest.raises(ValueError, match="NAME"):
        hs.env_line("NAME", "it's")


# -- what is wrong in .env ------------------------------------------------------------------

def test_a_complete_configuration_has_no_problem():
    assert hs.env_problems(GOOD) == {}


def test_the_example_file_is_all_placeholders():
    example = {k: v for k, v, _ in hs.parse_env((ROOT / ".env.example").read_text())}
    problems = hs.env_problems(example)
    for name in ("FLASK_SECRET_KEY", "JWT_SECRET_KEY", "API_KEY_PEPPER", "DB_PASSWORD", "REDIS_PASSWORD",
                 "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALLOWED_ORIGINS"):
        assert problems[name] == ("fix", "still the example value"), name


@pytest.mark.parametrize("change, name, text", [
    ({"FLASK_SECRET_KEY": ""}, "FLASK_SECRET_KEY", "missing"),
    ({"API_KEY_PEPPER": "short"}, "API_KEY_PEPPER", "shorter than 32"),
    ({"JWT_SECRET_KEY": "a" * 64}, "JWT_SECRET_KEY", "same as FLASK_SECRET_KEY"),
    ({"DB_PASSWORD": "abc@def:1234567890"}, "DB_PASSWORD", "letters and digits only"),
    ({"ALLOWED_ORIGINS": "https://box.tail1234.ts.net/"}, "ALLOWED_ORIGINS", "no trailing /"),
    ({"ALERT_TELEGRAM_CHAT_ID": ""}, "ALERT_TELEGRAM_CHAT_ID", "together"),
    ({"ALERT_TELEGRAM_BOT_TOKEN": "nonsense"}, "ALERT_TELEGRAM_BOT_TOKEN", "not a bot token"),
    ({"ALERT_TELEGRAM_BOT_TOKEN": "", "ALERT_TELEGRAM_CHAT_ID": ""}, "TELEGRAM_COMMANDS", "needs the bot token"),
    ({"TELEGRAM_COMMANDS": "yes"}, "TELEGRAM_COMMANDS", "true or false"),
    ({"HEARTBEAT_URL": "http://hc-ping.com/x"}, "HEARTBEAT_URL", "https://"),
    ({"BACKUP_HEARTBEAT_URL": GOOD["HEARTBEAT_URL"]}, "BACKUP_HEARTBEAT_URL", "own check"),
    ({"ALPACA_SECRET_KEY": "<paste it>"}, "ALPACA_SECRET_KEY", "example value"),
])
def test_problems_that_stop_the_stack_or_a_feature(change, name, text):
    level, message = hs.env_problems({**GOOD, **change})[name]
    assert level == "fix" and text in message


@pytest.mark.parametrize("change, name, text", [
    ({"ALPACA_API_KEY": "AKLIVEKEY123"}, "ALPACA_API_KEY", "live key"),
    ({"TRUSTED_PROXY_COUNT": "0"}, "TRUSTED_PROXY_COUNT", "lock everyone out"),
    ({"ENGINE_AUTOSTART": "true"}, "ENGINE_AUTOSTART", "restarts by itself"),
    ({"HEARTBEAT_URL": ""}, "HEARTBEAT_URL", "not set"),
    ({"HEARTBEAT_URL": "https://example.org/ping"}, "HEARTBEAT_URL", "hc-ping.com"),
])
def test_warnings(change, name, text):
    level, message = hs.env_problems({**GOOD, **change})[name]
    assert level == "warn" and text in message


def test_problem_messages_never_contain_the_value():
    broken = {name: f"{value}@x" for name, value in GOOD.items()}
    for _, message in hs.env_problems(broken).values():
        assert not any(value in message for value in broken.values() if len(value) > 5)


# -- the machine check --------------------------------------------------------------------

def test_a_ready_machine(tmp_path):
    env = write(tmp_path / ".env", GOOD)
    m = machine(tmp_path)
    findings = hs.check(m, env)
    assert texts(findings, "fix") == []
    assert "Tailscale 1.90.0" in texts(findings) and "serves the dashboard at https://box.tail1234.ts.net" in \
        texts(findings)
    assert "the app accepts this configuration" in texts(findings)
    assert {"api: running, healthy", "engine: running, healthy"} <= set(texts(findings))
    assert "curl 8.14.1" in texts(findings)  # not the whole version banner


def test_nothing_installed(tmp_path):
    findings = hs.check(machine(tmp_path, answers={}), tmp_path / ".env")
    fixes = {(f.text, f.step) for f in findings if f.level == "fix"}
    assert ("Docker is not installed", hs.STEP_DOCKER) in fixes
    assert ("Tailscale is not installed", hs.STEP_TAILSCALE) in fixes
    assert ("there is no .env yet: python3 ops/homeserver.py env", hs.STEP_ENV) in fixes
    assert any(t.startswith("git is not installed") for t, _ in fixes)
    assert not any(f.area == "Stack" and f.level == "fix" for f in findings)


@pytest.mark.parametrize("answer, text", [
    (hs.Result(1, "", "permission denied while trying to connect to the Docker daemon socket"),
     "your user cannot use Docker"),
    (hs.Result(1, "", "Cannot connect to the Docker daemon. Is the docker daemon running?"),
     "installed but not running"),
])
def test_docker_installed_but_not_usable(tmp_path, answer, text):
    findings = hs.check(machine(tmp_path, answers={**HEALTHY_MACHINE, ("docker", "info"): answer}), tmp_path / ".env")
    assert any(text in t for t in texts(findings, "fix"))


def test_docker_from_the_snap_store_and_old_compose(tmp_path):
    answers = {**HEALTHY_MACHINE, ("snap", "list", "docker"): ok("docker 28\n"),
               ("docker", "compose", "version"): hs.Result(1, "", "unknown command")}
    findings = hs.check(machine(tmp_path, answers=answers), tmp_path / ".env")
    assert any("snap store" in t for t in texts(findings, "warn"))
    assert any("Compose v2 plugin is missing" in t for t in texts(findings, "fix"))


@pytest.mark.parametrize("status, text", [
    (tailscale_status(state="NeedsLogin"), "not connected (NeedsLogin): sudo tailscale up"),
    (hs.Result(1, "", "failed to connect to local tailscaled; it doesn't appear to be running"),
     "service is not running"),
    (tailscale_status(magic=False), "MagicDNS is off"),
    (tailscale_status(certs=False), "HTTPS certificates are off"),
])
def test_tailscale_not_ready(tmp_path, status, text):
    findings = hs.check(machine(tmp_path, answers={**HEALTHY_MACHINE, ("tailscale", "status"): status}),
                        tmp_path / ".env")
    assert any(text in t for t in texts(findings, "fix"))


def test_tailscale_key_expiry_and_no_serve(tmp_path):
    answers = {**HEALTHY_MACHINE, ("tailscale", "status"): tailscale_status(expiry="2027-04-20T10:00:00Z"),
               ("tailscale", "serve"): ok("{}")}
    findings = hs.check(machine(tmp_path, answers=answers), tmp_path / ".env")
    assert any("expires (2027-04-20)" in t for t in texts(findings, "warn"))
    assert "does not serve the dashboard yet: sudo tailscale serve --bg 5000" in texts(findings, "fix")


def test_the_dashboard_address_must_be_this_machine(tmp_path):
    env = write(tmp_path / ".env", {**GOOD, "ALLOWED_ORIGINS": "https://other.tail1234.ts.net"})
    findings = hs.check(machine(tmp_path), env)
    assert any("does not list this machine's address, https://box.tail1234.ts.net" in t
               for t in texts(findings, "fix"))


def test_env_file_permissions_and_ignored_settings(tmp_path):
    env = write(tmp_path / ".env", {**GOOD, "DATABASE_URL": "postgresql://x", "API_BIND": "0.0.0.0"})
    env.chmod(0o644)
    findings = hs.check(machine(tmp_path), env)
    assert any("chmod 600" in t for t in texts(findings, "warn"))
    assert any("API_BIND=0.0.0.0" in t for t in texts(findings, "warn"))
    assert "ignored, the stack sets them itself: DATABASE_URL" in texts(findings, "info")


def test_an_existing_database_without_a_password_is_flagged(tmp_path):
    answers = {**HEALTHY_MACHINE, ("docker", "volume", "ls"): ok(f"{tmp_path.name}_postgres-data\n")}
    findings = hs.check(machine(tmp_path, answers=answers), tmp_path / ".env")
    assert any("needs the DB_PASSWORD it was created with" in t for t in texts(findings, "fix"))


def test_the_app_refusing_the_configuration_is_reported_without_values(tmp_path):
    env = write(tmp_path / ".env", GOOD)
    leaked = f"ALERT_NTFY_URL: bad value {GOOD['ALPACA_SECRET_KEY']}"
    answers = {**HEALTHY_MACHINE, ("docker", "compose", "run"): hs.Result(1, leaked + "\n")}
    findings = hs.check(machine(tmp_path, answers=answers), env)
    report = hs.report(findings)
    assert "the app refuses the configuration: ALERT_NTFY_URL: bad value ***" in report
    assert not any(value in report for value in hs.secret_values(GOOD))
    assert "serves the dashboard at https://box.tail1234.ts.net" in report  # an address is not a secret


def test_services_that_are_down_or_unhealthy(tmp_path):
    env = write(tmp_path / ".env", GOOD)
    ps = ok(json.dumps([{"Service": "api", "State": "running", "Health": "unhealthy"},
                        {"Service": "engine", "State": "exited", "Health": ""},
                        {"Service": "postgres", "State": "running", "Health": "starting"}]))
    findings = hs.check(machine(tmp_path, answers={**HEALTHY_MACHINE, ("docker", "compose", "ps"): ps}), env)
    assert "api: running, unhealthy (docker compose logs api)" in texts(findings, "fix")
    assert "engine: exited (docker compose logs engine)" in texts(findings, "fix")
    assert "postgres: starting" in texts(findings, "info")


def test_something_else_on_the_api_port(tmp_path):
    m = machine(tmp_path, listening=True, http=lambda *a, **k: hs.Response(200, b"<html>other</html>"))
    hs.check_port(m)
    assert texts(m.findings, "fix") == ["another program uses port 5000: the API cannot start"]
    m = machine(tmp_path, listening=True, http=lambda *a, **k: hs.Response(200, b'{"status": "healthy"}'))
    hs.check_port(m)
    assert texts(m.findings, "ok") == ["the Nocturna API answers on 127.0.0.1:5000"]


def test_a_laptop_lid(tmp_path):
    batteries = tmp_path / "power"
    (batteries / "BAT0").mkdir(parents=True)
    for config, level in (("#HandleLidSwitch=suspend\n", "warn"), ("[Login]\nHandleLidSwitch=ignore\n", "ok")):
        m = machine(tmp_path, answers={("systemd-analyze",): ok(config)})
        hs.check_system(m, batteries=batteries)
        assert any("lid" in t for t in texts(m.findings, level))


def test_the_report_lists_what_to_do_in_step_order(tmp_path):
    findings = [hs.Finding("warn", "A", "later", 6), hs.Finding("fix", "B", "first", 3),
                hs.Finding("ok", "B", "fine"), hs.Finding("fix", "C", "second", 4)]
    report = hs.report(findings)
    assert report.index("1. FIX step 3: first") < report.index("2. FIX step 4: second") \
        < report.index("3. WARN step 6: later")
    assert "Nothing to fix" not in report
    assert hs.report([hs.Finding("ok", "A", "fine")]).endswith("Nothing to fix.")
    assert hs.report([hs.Finding("warn", "A", "odd", 2)]).endswith("Nothing to fix. Read the warnings above.")


# -- building .env ---------------------------------------------------------------------------

class Script:
    """A dialog that answers from lists and records what it said."""

    def __init__(self, answers=(), secrets=()):
        self.answers, self.secrets, self.said = list(answers), list(secrets), []

    def dialog(self):
        return hs.Dialog(ask=self._ask, ask_secret=self._secret, say=self.said.append)

    def _ask(self, question):
        self.said.append(question)
        return self.answers.pop(0)

    def _secret(self, question):
        self.said.append(question)
        return self.secrets.pop(0)

    @property
    def text(self):
        return "\n".join(self.said)


class Services:
    """Alpaca and Telegram as the builder sees them."""

    def __init__(self, alpaca_key="PKABCDEFGH123456", chats=({"id": 424242, "type": "private", "first_name": "Op"},)):
        self.alpaca_key, self.chats, self.calls = alpaca_key, list(chats), []

    def __call__(self, method, url, headers=None, data=None, timeout=15):
        self.calls.append((method, url.split("/")[-1], data))
        if "alpaca" in url:
            if headers["APCA-API-KEY-ID"] == self.alpaca_key:
                return hs.Response(200, json.dumps({"account_number": "PA1", "status": "ACTIVE", "equity": "100000",
                                                    "currency": "USD"}).encode())
            return hs.Response(401, b"{}")
        method_name = url.rsplit("/", 1)[-1]
        if method_name == "getMe":
            good = url.split("/bot", 1)[1].startswith(TOKEN)
            return hs.Response(200 if good else 401, json.dumps(
                {"ok": good, "result": {"username": "nocturna_bot"}} if good else
                {"ok": False, "description": "Unauthorized"}).encode())
        if method_name == "getUpdates":
            updates = [{"update_id": n, "message": {"chat": chat}} for n, chat in enumerate(self.chats)]
            return hs.Response(200, json.dumps({"ok": True, "result": updates}).encode())
        return hs.Response(200, b'{"ok": true, "result": {}}')


TAILSCALE = {("tailscale", "status"): tailscale_status(), ("docker", "info"): ok("29\n")}


def build(tmp_path, script, services=None, answers=TAILSCALE):
    env = tmp_path / ".env"
    code = hs.build_env(env, Runner(answers), services or Services(), script.dialog())
    return code, env


def test_a_first_run(tmp_path):
    script = Script(
        answers=["", "", "", "", "", "https://hc-ping.com/1111", "https://hc-ping.com/2222"],
        secrets=["PKWRONG", "wrong", "PKABCDEFGH123456", "s3cr3tAlpacaValue0123456789", TOKEN])
    code, env = build(tmp_path, script)
    assert code == 0
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    values = hs.env_values(env)
    for name in ("FLASK_SECRET_KEY", "JWT_SECRET_KEY", "API_KEY_PEPPER"):
        assert re.fullmatch(r"[0-9a-f]{64}", values[name])
    assert re.fullmatch(r"[0-9a-f]{32}", values["DB_PASSWORD"]) and len(values["REDIS_PASSWORD"]) == 32
    assert values["JWT_SECRET_KEY"] != values["FLASK_SECRET_KEY"]
    assert values["ALPACA_API_KEY"] == "PKABCDEFGH123456"
    assert values["ALERT_TELEGRAM_BOT_TOKEN"] == TOKEN and values["ALERT_TELEGRAM_CHAT_ID"] == "424242"
    assert values["TELEGRAM_COMMANDS"] == "true" and values["ENGINE_AUTOSTART"] == "false"
    assert values["ALLOWED_ORIGINS"] == "https://box.tail1234.ts.net" and values["TRUSTED_PROXY_COUNT"] == "1"
    assert values["HEARTBEAT_URL"].endswith("1111") and values["BACKUP_HEARTBEAT_URL"].endswith("2222")
    assert hs.env_problems(values) == {}
    assert "Alpaca refused these keys" in script.text and "Found your chat with Op." in script.text
    assert not any(value in script.text for value in hs.secret_values(values) if "hc-ping" not in value)


def test_a_second_run_keeps_every_secret_and_backs_up_the_file(tmp_path):
    env = write(tmp_path / ".env", {**GOOD, "RISK_MAX_DRAWDOWN": "0.08", "DATABASE_URL": "postgresql://x",
                                    "ADMIN_EMAIL": "admin@yourdomain.com"})
    script = Script(answers=["", "", "", "", "", "", ""])  # keep everything
    code, _ = build(tmp_path, script)
    assert code == 0
    values = hs.env_values(env)
    assert {k: values[k] for k in GOOD} == GOOD
    assert values["RISK_MAX_DRAWDOWN"] == "0.08"
    assert "DATABASE_URL" not in values and "ADMIN_EMAIL" not in values  # the stack sets it; an example value
    backups = list(tmp_path.glob(".env.backup-*"))
    assert len(backups) == 1 and stat.S_IMODE(backups[0].stat().st_mode) == 0o600
    assert hs.env_values(backups[0])["DATABASE_URL"] == "postgresql://x"


def test_the_example_file_becomes_a_working_configuration(tmp_path):
    (tmp_path / ".env").write_text((ROOT / ".env.example").read_text())
    script = Script(answers=["", "", "", "", "", ""], secrets=["PKABCDEFGH123456", "s3cr3tAlpacaValue0123456789", TOKEN])
    code, env = build(tmp_path, script)
    values = hs.env_values(env)
    assert code == 0 and {name for name, (level, _) in hs.env_problems(values).items() if level == "fix"} == set()
    assert not any(name in values for name in hs.STACK_SETS)


def test_the_database_password_is_never_replaced(tmp_path):
    write(tmp_path / ".env", {**GOOD, "DB_PASSWORD": "not-url-safe@1"})
    script = Script(answers=["", "", "", "", "", "", ""])
    _, env = build(tmp_path, script)
    assert hs.env_values(env)["DB_PASSWORD"] == "not-url-safe@1"
    assert "Left unchanged, since the database uses it" in script.text


def test_an_existing_database_without_its_password_stops_the_builder(tmp_path):
    answers = {**TAILSCALE, ("docker", "volume", "ls"): ok("x_postgres-data\n")}
    script = Script()
    code, env = build(tmp_path, script, answers=answers)
    assert code == 1 and not env.exists() and "PostgreSQL keeps the password" in script.text


def test_alpaca_keys_that_do_not_work_can_be_left_as_they_were(tmp_path):
    script = Script(answers=["n", "n", "n", "", "", ""], secrets=["PKBAD", "bad"])
    code, env = build(tmp_path, script, services=Services(alpaca_key="PKOTHER"))
    assert code == 0 and hs.env_values(env)["ALPACA_API_KEY"] == ""
    assert "Left the Alpaca keys as they were." in script.text


def test_telegram_can_be_skipped_and_several_chats_are_offered(tmp_path):
    script = Script(answers=["", "skip", "", "", ""], secrets=["PKABCDEFGH123456", "secret", TOKEN])
    _, env = build(tmp_path, script)
    values = hs.env_values(env)
    assert values["ALERT_TELEGRAM_BOT_TOKEN"] == "" and values["TELEGRAM_COMMANDS"] in ("", "false")
    assert "Telegram left as it was." in script.text
    two = Services(chats=[{"id": 1, "type": "private", "first_name": "A"},
                          {"id": 2, "type": "private", "first_name": "B", "username": "bee"}])
    script = Script(answers=["", "", "2", "n", "", "", ""], secrets=["PKABCDEFGH123456", "secret", TOKEN])
    (tmp_path / "two").mkdir()
    _, env = build(tmp_path / "two", script, services=two)
    values = hs.env_values(env)
    assert values["ALERT_TELEGRAM_CHAT_ID"] == "2" and values["TELEGRAM_COMMANDS"] == "false"
    assert "B (@bee)" in script.text


def test_without_tailscale_the_address_waits_for_it(tmp_path):
    script = Script(answers=["", "", "", "", ""], secrets=["PKABCDEFGH123456", "secret", TOKEN])
    _, env = build(tmp_path, script, answers={("docker", "info"): ok("29\n")})
    assert hs.env_values(env)["ALLOWED_ORIGINS"] == "http://127.0.0.1:5000"
    assert "Tailscale is not set up yet" in script.text


# -- the guide -------------------------------------------------------------------------------

def test_the_guide_explains_every_setting_the_tool_writes():
    guide = (ROOT / "docs" / "HOME_SERVER.md").read_text()
    missing = [var.name for var in hs.VARS if f"`{var.name}`" not in guide]
    assert not missing, missing
