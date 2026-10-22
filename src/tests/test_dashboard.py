"""
Tests for how the dashboard (src/static) is served: it must run under the
app's Content-Security-Policy, and browsers must pick up a new version.
The dashboard's behaviour is checked in a browser: tools/dashboard/e2e.cjs.
"""

import re
from html.parser import HTMLParser
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static"


class _Scripts(HTMLParser):
    """The page's script tags, inline code and event-handler attributes, as a browser reads them (any case)."""

    def __init__(self, page):
        super().__init__()
        self.scripts, self.inline, self.handlers, self._in_script = [], [], [], False
        self.feed(page)
        self.close()

    def handle_starttag(self, tag, attrs):
        self.handlers += [name for name, _ in attrs if name.startswith("on")]
        if tag == "script":
            self.scripts.append(dict(attrs))
            self._in_script = True

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False

    def handle_data(self, data):
        if self._in_script and data.strip():
            self.inline.append(data)


def test_index_is_served_for_the_root_and_unknown_pages(client):
    for path in ("/", "/some/client/route"):
        response = client.get(path)
        assert response.status_code == 200 and b'<main id="login"' in response.data
        assert response.headers["Cache-Control"] == "no-cache"


def test_scripts_and_styles_load_under_the_csp(client):
    page = client.get("/").get_data(as_text=True)
    csp = client.get("/").headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp.split("script-src")[1].split(";")[0]
    # No inline script or handler: the CSP would block it
    found = _Scripts(page)
    assert found.scripts == [{"type": "module", "src": "/app.js"}]
    assert found.inline == [] and found.handlers == []
    script = client.get("/app.js")
    assert script.status_code == 200 and script.mimetype == "text/javascript"
    assert script.headers["Cache-Control"] == "no-cache" and script.headers["X-Content-Type-Options"] == "nosniff"
    assert client.get("/app.css").mimetype == "text/css"
    assert client.get("/favicon.svg").mimetype == "image/svg+xml"


def test_the_page_never_inserts_markup():
    """Everything is built with DOM calls, so nothing the API returns can inject markup."""
    code = "\n".join(line for line in (STATIC / "app.js").read_text().splitlines()
                     if not line.strip().startswith("//"))
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert sink not in code, sink


def test_the_dashboard_uses_endpoints_that_exist(app):
    source = (STATIC / "app.js").read_text()
    used = set(re.findall(r'"(/api/[a-z0-9/-]+)', source))
    rules = {rule.rule for rule in app.url_map.iter_rules()}
    # The cancel path is built from the order's id
    assert used and used <= rules | {"/api/v1/trading/orders/"}, used - rules


def test_the_compiled_bundle_is_gone():
    assert not (STATIC / "assets").exists()
