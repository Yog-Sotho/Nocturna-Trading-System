// Nocturna dashboard.
//
// Talks to the API of this origin with a bearer token. The access token lives
// in memory only; the single-use refresh token in sessionStorage, or in
// localStorage with "Stay signed in". Every element is built with DOM calls,
// never innerHTML, so nothing the API returns can inject markup.

const REFRESH_EVERY_MS = 15000;
const REMEMBER_KEY = "nocturna.remember";
const REFRESH_KEY = "nocturna.refresh";

class AuthError extends Error {}

const $ = (id) => document.getElementById(id);

// -- tokens ----------------------------------------------------------------------

const session = { access: null, expiresAt: 0 };
let refreshing = null;

function tokenStore() {
  try {
    return localStorage.getItem(REMEMBER_KEY) === "1" ? localStorage : sessionStorage;
  } catch {
    return null;
  }
}

function readRefreshToken() {
  try {
    return tokenStore()?.getItem(REFRESH_KEY) ?? null;
  } catch {
    return null;
  }
}

function saveTokens(data) {
  session.access = data.access_token;
  session.expiresAt = Date.now() + (data.expires_in ?? 3600) * 1000;
  try {
    tokenStore()?.setItem(REFRESH_KEY, data.refresh_token);
  } catch {
    // Storage blocked: the session lasts as long as the page
  }
}

function clearTokens() {
  session.access = null;
  session.expiresAt = 0;
  for (const store of [localStorage, sessionStorage]) {
    try {
      store.removeItem(REFRESH_KEY);
    } catch {
      // nothing stored
    }
  }
  try {
    localStorage.removeItem(REMEMBER_KEY);
  } catch {
    // nothing stored
  }
}

async function rotateTokens() {
  // A refresh token works once; presenting a used one ends every session. Tabs
  // that share localStorage take turns, and each reads the token inside the lock.
  const rotate = async () => {
    const token = readRefreshToken();
    if (!token) throw new AuthError("signed out");
    const response = await fetch("/api/auth/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: token }),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok || !body.success) {
      clearTokens();
      throw new AuthError("session ended");
    }
    saveTokens(body.data);
  };
  return navigator.locks ? navigator.locks.request("nocturna-refresh", rotate) : rotate();
}

function freshTokens() {
  // One rotation at a time in this tab, however many requests need it
  if (!refreshing) refreshing = rotateTokens().finally(() => { refreshing = null; });
  return refreshing;
}

async function api(method, path, body) {
  if (!session.access || Date.now() > session.expiresAt - 60000) await freshTokens();
  const send = () => fetch(path, {
    method,
    headers: {
      Authorization: `Bearer ${session.access}`,
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const used = session.access;
  let response = await send();
  if (response.status === 401) {
    if (session.access === used) await freshTokens();
    response = await send();
    if (response.status === 401) throw new AuthError("unauthorized");
  }
  const json = await response.json().catch(() => ({}));
  return {
    ok: response.ok && json.success !== false,
    status: response.status,
    data: json.data,
    error: json.error || json.message || `HTTP ${response.status}`,
    message: json.message,
  };
}

// -- formatting --------------------------------------------------------------------

const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const money = (v) => (v === null || v === undefined ? "–" : usd.format(Number(v)));
const pct = (v) => (v === null || v === undefined ? "–" : `${(Number(v) * 100).toFixed(2)}%`);
const qty = (v) => (v === null || v === undefined ? "–" : String(Number(v)));
const when = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" }) : "–");
// Tables are narrow on a phone: month/day and time only
const brief = (iso) => (iso ? new Date(iso).toLocaleString(undefined,
  { month: "numeric", day: "numeric", hour: "numeric", minute: "2-digit" }) : "–");
const weight = (v) => (v === null || v === undefined ? "–" : `${(Number(v) * 100).toFixed(1)}%`);

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined) node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

function table(headers, rows, empty) {
  if (!rows.length) return el("p", { class: "muted" }, empty);
  return el("div", { class: "scroll" },
    el("table", {},
      el("thead", {}, el("tr", {}, headers.map((h) => el("th", {}, h)))),
      el("tbody", {}, rows.map((cells) => el("tr", {}, cells.map((c) => el("td", {}, c)))))));
}

const signed = (value, text) => el("span", { class: Number(value) >= 0 ? "up" : "down" }, text);

// -- rendering ---------------------------------------------------------------------

function renderStatus(status) {
  const risk = status.risk || {};
  const halted = status.halted === true;
  const badge = $("engine-badge");
  const label = halted ? "halted" : risk.cooling_down ? "cooling down" : status.state;
  badge.textContent = label;
  badge.className = `badge ${halted ? "halted" : risk.cooling_down ? "cooling" : status.state}`;

  const problems = [];
  if (!status.configured) problems.push("Alpaca keys are not set: the engine cannot start.");
  if (status.state === "unreachable") {
    problems.push("The engine process is not answering. The emergency stop still reaches Alpaca.");
  }
  if (halted) problems.push(`Trading halted: ${status.halt_reason || "no reason given"}`);
  if (status.state === "failed") problems.push(`The engine failed: ${status.last_error || "see the logs"}`);
  const banner = $("banner");
  banner.textContent = problems.join(" ");
  banner.hidden = problems.length === 0;

  const report = status.last_reconciliation;
  const figures = [
    ["Equity", money(status.equity)],
    ["Cash", money(status.cash)],
    ["Realized P&L", status.realized_pnl === undefined ? "–" : signed(status.realized_pnl, money(status.realized_pnl))],
    ["Fees", money(status.fees)],
    ["Drawdown", pct(risk.drawdown)],
    ["High-water mark", money(risk.high_water_mark)],
    ["Exposure scale", exposureScale(risk)],
    ["Orders today", risk.orders_today ?? "–"],
    ["Last market data", when(status.last_poll)],
    ["Next poll", when(status.next_poll)],
    ["Reconciliation", report ? (report.in_sync ? `in sync (${when(report.at)})`
      : `${report.discrepancies.length} corrected (${when(report.at)})`) : "–"],
    ["Running since", when(status.started_at)],
  ];
  if (status.last_error && status.state !== "failed") figures.push(["Last error", status.last_error, "wide"]);
  $("account").replaceChildren(...figures.map(([name, value, wide]) =>
    el("div", wide ? { class: "wide" } : {}, el("dt", {}, name), el("dd", {}, value))));

  const state = status.state;
  setEnabled("start", ["stopped", "failed"].includes(state) && status.configured);
  setEnabled("stop", ["running", "starting"].includes(state));
  setEnabled("resume", state === "running" && halted);
  setEnabled("halt", true);
}

function exposureScale(risk) {
  // Volatility targeting times the drawdown ladder; either is null until it applies
  const parts = [risk.volatility_scale, risk.drawdown_scale].filter((v) => v !== null && v !== undefined);
  return parts.length ? pct(parts.reduce((product, v) => product * Number(v), 1)) : "–";
}

function setEnabled(action, enabled) {
  document.querySelector(`[data-action="${action}"]`).disabled = !enabled;
}

function renderPositions(positions, targets) {
  if (!positions) {
    $("positions").replaceChildren(el("p", { class: "muted" }, "Shown while the engine runs."));
    return;
  }
  const rows = positions.positions.map((p) => [
    p.symbol, qty(p.qty), money(p.market_value), signed(p.unrealized_pnl, money(p.unrealized_pnl)),
    targets && targets[p.symbol] !== undefined ? weight(targets[p.symbol]) : "–",
  ]);
  for (const [symbol, target] of Object.entries(targets || {})) {
    if (!positions.positions.some((p) => p.symbol === symbol)) rows.push([symbol, "0", "–", "–", weight(target)]);
  }
  $("positions").replaceChildren(table(["Symbol", "Qty", "Value", "P&L", "Target"], rows, "No positions."));
}

function renderWorking(working) {
  const rows = working.map((o) => [
    `${o.side} ${qty(o.qty)} ${o.symbol}`, o.order_type, o.status, brief(o.created_at),
    el("button", { class: "link", onclick: () => confirmCancel(o) }, "Cancel"),
  ]);
  $("working").replaceChildren(table(["Order", "Type", "Status", "Placed", ""], rows, "No working orders."));
}

function renderFills(fills) {
  const rows = fills.map((f) => [brief(f.timestamp), `${f.side} ${qty(f.qty)} ${f.symbol}`, money(f.price), money(f.fee)]);
  $("fills").replaceChildren(table(["Time", "Fill", "Price", "Fee"], rows, "No fills yet."));
}

function renderHistory(history) {
  const rows = history.map((o) => [
    brief(o.created_at), `${o.side} ${qty(o.qty)} ${o.symbol}`, o.status,
    o.avg_fill_price ? `${qty(o.filled_qty)} @ ${money(o.avg_fill_price)}` : "–",
  ]);
  $("history").replaceChildren(table(["Time", "Order", "Status", "Filled"], rows, "No orders yet."));
}

function renderEquity(points) {
  const box = $("equity-chart");
  if (points.length < 2) {
    box.replaceChildren(el("p", { class: "muted" }, "The curve appears after two recorded days."));
    return;
  }
  const values = points.map((p) => Number(p.equity));
  const low = Math.min(...values);
  const high = Math.max(...values);
  const span = high - low || 1;
  const width = 1000;
  const height = 300;
  const coords = values.map((v, i) => `${(i / (values.length - 1)) * width},${height - ((v - low) / span) * height}`);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `Equity from ${money(values[0])} to ${money(values.at(-1))}`);
  const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
  line.setAttribute("class", "line");
  line.setAttribute("points", coords.join(" "));
  svg.append(line);
  const change = values.at(-1) - values[0];
  box.replaceChildren(
    svg,
    el("p", { class: "muted small" },
      `${when(points[0].timestamp)} → ${when(points.at(-1).timestamp)} · low ${money(low)} · high ${money(high)} · `,
      signed(change, `${change >= 0 ? "+" : ""}${money(change)} (${pct(change / values[0])})`)),
  );
}

// -- loading -----------------------------------------------------------------------

async function load() {
  try {
    const status = await api("GET", "/api/trading/status");
    if (!status.ok) throw new Error(status.error);
    // Positions and targets exist only while the engine has a snapshot
    const live = status.data.state === "running";
    const none = Promise.resolve({ ok: false });
    const [positions, targets, orders, fills, equity] = await Promise.all([
      live ? api("GET", "/api/trading/positions") : none,
      live ? api("GET", "/api/trading/targets") : none,
      api("GET", "/api/trading/orders?limit=20"),
      api("GET", "/api/trading/fills?limit=20"),
      api("GET", "/api/trading/equity-curve?limit=500"),
    ]);
    renderStatus(status.data);
    renderPositions(positions.ok ? positions.data : null, targets.ok ? targets.data.targets : null);
    if (orders.ok) {
      renderWorking(orders.data.working);
      renderHistory(orders.data.history);
    }
    if (fills.ok) renderFills(fills.data.fills);
    if (equity.ok) renderEquity(equity.data.equity_curve);
    $("updated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    if (error instanceof AuthError) return showLogin();
    $("updated").textContent = `Update failed: ${error.message}`;
  }
}

// -- actions -----------------------------------------------------------------------

const ACTIONS = {
  start: {
    title: "Start the engine?",
    text: "It loads the account from Alpaca, then trades on the latest closed bars.",
    path: "/api/trading/control/start",
  },
  stop: {
    title: "Stop the engine?",
    text: "Trading stops. Positions and working orders stay at Alpaca.",
    path: "/api/trading/control/stop",
  },
  resume: {
    title: "Lift the halt?",
    text: "Trading resumes and the high-water mark restarts at the current equity.",
    path: "/api/trading/control/resume",
  },
};

function ask(title, text, extra = [], ready = () => true) {
  const dialog = $("confirm");
  $("confirm-title").textContent = title;
  $("confirm-text").textContent = text;
  $("confirm-extra").replaceChildren(...extra);
  const ok = $("confirm-ok");
  const update = () => { ok.disabled = !ready(); };
  dialog.oninput = update;
  dialog.onchange = update;
  update();
  dialog.returnValue = "";
  dialog.showModal();
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "ok"), { once: true });
  });
}

async function run(path, body) {
  const result = await api("POST", path, body);
  notify(result.ok ? (result.message || "Done.") : result.error, !result.ok);
  await load();
}

async function onAction(action) {
  try {
    if (action === "halt") return await confirmEmergencyStop();
    const spec = ACTIONS[action];
    if (await ask(spec.title, spec.text)) await run(spec.path);
  } catch (error) {
    if (error instanceof AuthError) return showLogin();
    notify(error.message, true);
  }
}

async function confirmEmergencyStop() {
  const reason = el("input", { name: "reason", maxlength: "500", placeholder: "What happened?" });
  const keep = el("input", { type: "radio", name: "close", value: "keep" });
  const close = el("input", { type: "radio", name: "close", value: "close" });
  const extra = [
    el("label", {}, "Reason", reason),
    el("label", { class: "check" }, keep, "Keep the positions (cancel every order)"),
    el("label", { class: "check" }, close, "Close every position too"),
  ];
  const confirmed = await ask(
    "Emergency stop",
    "Trading halts at once. If the engine does not answer, Alpaca is asked directly.",
    extra,
    () => keep.checked || close.checked,
  );
  if (!confirmed) return;
  await run("/api/trading/emergency-stop", {
    reason: reason.value.trim() || "Emergency stop from the dashboard",
    close_all_positions: close.checked,
  });
}

async function confirmCancel(order) {
  try {
    const text = `${order.side} ${qty(order.qty)} ${order.symbol} (${order.client_order_id})`;
    if (await ask("Cancel this order?", text)) {
      await run(`/api/trading/orders/${encodeURIComponent(order.client_order_id)}/cancel`);
    }
  } catch (error) {
    if (error instanceof AuthError) return showLogin();
    notify(error.message, true);
  }
}

let noticeTimer = null;

function notify(text, failed) {
  const notice = $("notice");
  notice.textContent = text;
  notice.className = failed ? "notice failed" : "notice";
  notice.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { notice.hidden = true; }, failed ? 15000 : 6000);
}

// -- screens -----------------------------------------------------------------------

let timer = null;

function showLogin() {
  clearInterval(timer);
  timer = null;
  $("app").hidden = true;
  $("login").hidden = false;
}

function showApp() {
  $("login").hidden = true;
  $("app").hidden = false;
  load();
  clearInterval(timer);
  timer = setInterval(() => {
    if (document.visibilityState === "visible") load();
  }, REFRESH_EVERY_MS);
}

async function onLogin(event) {
  event.preventDefault();
  const form = event.target;
  const error = $("login-error");
  error.hidden = true;
  const button = form.querySelector("button");
  button.disabled = true;
  try {
    const response = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: form.username.value.trim(), password: form.password.value }),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok || !body.success) throw new Error(body.error || `HTTP ${response.status}`);
    clearTokens();
    try {
      if (form.remember.checked) localStorage.setItem(REMEMBER_KEY, "1");
    } catch {
      // Storage blocked: signed in for this page only
    }
    saveTokens(body.data);
    form.password.value = "";
    showApp();
  } catch (e) {
    error.textContent = e.message;
    error.hidden = false;
  } finally {
    button.disabled = false;
  }
}

async function onLogout() {
  const refreshToken = readRefreshToken();
  try {
    if (session.access) {
      await fetch("/api/auth/logout", {
        method: "POST",
        headers: { Authorization: `Bearer ${session.access}`, "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
    }
  } finally {
    clearTokens();
    showLogin();
  }
}

async function start() {
  $("login-form").addEventListener("submit", onLogin);
  $("logout").addEventListener("click", onLogout);
  for (const button of document.querySelectorAll("[data-action]")) {
    button.addEventListener("click", () => onAction(button.dataset.action));
  }
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && timer) load();
  });
  if (!readRefreshToken()) return showLogin();
  try {
    await freshTokens();
    showApp();
  } catch {
    showLogin();
  }
}

start();
