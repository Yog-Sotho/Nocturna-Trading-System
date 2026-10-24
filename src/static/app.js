// Nocturna dashboard: the shell (sign-in, sections, theme), the overview and the controls.
//
// Talks to the API of this origin (js/api.js). Every element is built with DOM
// calls, never innerHTML, so nothing the API returns can inject markup.

import { AuthError, api, freshTokens, readRefreshToken, signIn, signOut } from "./js/api.js";
import { labView } from "./js/lab.js";
import { alertsView, settingsView } from "./js/settings.js";
import { riskView, strategiesView } from "./js/strategies.js";
import {
  $, ask, brief, decorateIcons, el, figures, lineChart, mixBar, money, notify, pct, qty, signed, table, weight,
  when,
} from "./js/ui.js";

const REFRESH_EVERY_MS = 15000;
const LOOK_KEY = "nocturna.look";

// -- what every section shares ----------------------------------------------------------------

const ctx = {
  me: null,
  admin: false,
  prefs: null,
  status: null,
  profile: null,
  catalog: null,
  // The mix being edited on the Strategies page; the lab can backtest it before it is saved
  draft: null,

  async strategies(force = false) {
    if (!this.catalog || force) {
      const result = await api("GET", "/api/v1/trading/strategies");
      if (!result.ok) throw new Error(result.error);
      this.catalog = result.data.strategies;
    }
    return this.catalog;
  },

  async loadProfile() {
    const result = await api("GET", "/api/v1/trading/profile");
    if (!result.ok) throw new Error(result.error);
    this.profile = result.data;
    return this.profile;
  },

  // The strategy's place in the catalog: its colour everywhere
  colour(name) {
    const index = (this.catalog || []).findIndex((s) => s.name === name);
    return index < 0 ? 7 : index;
  },

  title(name) {
    return (this.catalog || []).find((s) => s.name === name)?.title || name;
  },

  // Runs an action; a lost session shows the sign-in, anything else is reported
  async guard(action) {
    try {
      return await action();
    } catch (error) {
      if (error instanceof AuthError) return showLogin();
      notify(error.message, true);
      return undefined;
    }
  },

  applyLook(prefs) {
    const root = document.documentElement;
    root.dataset.theme = prefs.theme;
    root.dataset.accent = prefs.accent;
    try {
      localStorage.setItem(LOOK_KEY, JSON.stringify({ theme: prefs.theme, accent: prefs.accent }));
    } catch {
      // Storage blocked: the look applies to this page
    }
  },

  applyCards(prefs) {
    const box = $("cards");
    for (const name of prefs.cards) {
      const node = box.querySelector(`[data-card="${name}"]`);
      if (!node) continue;
      node.hidden = prefs.hidden.includes(name);
      box.append(node);
    }
  },

  greet() {
    const hour = new Date().getHours();
    const part = hour < 5 ? "Night watch" : hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
    const name = this.prefs?.display_name || this.me?.username || "";
    const status = this.status;
    const line = !status ? "" : status.state === "running"
      ? `The engine is running${status.halted ? ", halted" : ""}.`
      : `The engine is ${status.state}.`;
    const waiting = this.profile?.pending ? " A newer mix waits for a restart." : "";
    $("greeting").replaceChildren(el("strong", {}, `${part}${name ? `, ${name}` : ""}.`),
      el("span", { class: "muted small" }, line + waiting));
  },
};

function restoreLook() {
  try {
    const look = JSON.parse(localStorage.getItem(LOOK_KEY) || "null");
    if (look?.theme) document.documentElement.dataset.theme = look.theme;
    if (look?.accent) document.documentElement.dataset.accent = look.accent;
  } catch {
    // Nothing stored, or storage blocked: the night theme
  }
}

// -- the engine, on every page ----------------------------------------------------------------

function renderHeader(status) {
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
}

async function refreshStatus() {
  const status = await api("GET", "/api/v1/trading/status");
  if (!status.ok) throw new Error(status.error);
  ctx.status = status.data;
  renderHeader(status.data);
  ctx.greet();
  return status.data;
}

// -- the overview -------------------------------------------------------------------------------

function exposureScale(risk) {
  // Volatility targeting times the drawdown ladder; either is null until it applies
  const parts = [risk.volatility_scale, risk.drawdown_scale].filter((v) => v !== null && v !== undefined);
  return parts.length ? pct(parts.reduce((product, v) => product * Number(v), 1)) : "–";
}

function renderAccount(status) {
  const risk = status.risk || {};
  const report = status.last_reconciliation;
  const items = [
    ["Equity", money(status.equity), "big"],
    ["Cash", money(status.cash)],
    ["Realized P&L", status.realized_pnl === undefined ? "–" : signed(status.realized_pnl, money(status.realized_pnl))],
    ["Fees", money(status.fees)],
    ["High-water mark", money(risk.high_water_mark)],
    ["Last market data", when(status.last_poll)],
    ["Next poll", when(status.next_poll)],
    ["Reconciliation", report ? (report.in_sync ? `in sync (${when(report.at)})`
      : `${report.discrepancies.length} corrected (${when(report.at)})`) : "–"],
    ["Running since", when(status.started_at)],
  ];
  if (status.last_error && status.state !== "failed") items.push(["Last error", status.last_error, "wide"]);
  $("account").replaceWith(Object.assign(figures(items), { id: "account" }));
}

function renderControls(status) {
  const state = status.state;
  const halted = status.halted === true;
  setEnabled("start", ["stopped", "failed"].includes(state) && status.configured);
  setEnabled("stop", ["running", "starting"].includes(state));
  setEnabled("resume", state === "running" && halted);
  setEnabled("halt", true);
  const restart = document.querySelector('[data-action="restart"]');
  restart.hidden = !(ctx.profile?.pending && state === "running");
}

function setEnabled(action, enabled) {
  document.querySelector(`[data-action="${action}"]`).disabled = !enabled;
}

function renderMix() {
  const state = ctx.profile;
  if (!state) return;
  const shown = state.running || state.saved;
  const parts = [
    ...mixBar(shown.mix.map((e) => ({ label: ctx.title(e.strategy), share: Number(e.weight),
      index: ctx.colour(e.strategy) }))),
    el("p", { class: "muted small" }, state.running
      ? `Trading ${shown.id === null ? "the default profile" : `profile #${shown.id}`}`
        + (shown.activated_at ? ` since ${when(shown.activated_at)}.` : ".")
      : `${shown.id === null ? "The default profile" : `Profile #${shown.id}`}: the engine trades it when it starts.`),
  ];
  if (shown.note) parts.push(el("p", { class: "small" }, `“${shown.note}”`));
  if (state.pending) {
    parts.push(el("p", { class: "verdict" }, `A newer mix is saved (profile #${state.saved.id}). `
      + "It is traded from the engine's next start: restart to trade it now."));
  }
  $("mix").replaceChildren(...parts);
}

function renderRiskNow(status) {
  const risk = status.risk || {};
  const rows = Object.fromEntries((ctx.profile?.risk || []).map((r) => [r.name, r]));
  const kill = Number(rows.risk_max_drawdown?.value ?? 0.1);
  const drawdown = Math.abs(Number(risk.drawdown ?? 0));
  const used = kill > 0 ? Math.min(1, drawdown / kill) : 0;
  $("risk-now").replaceChildren(
    el("p", { class: "small muted" }, `Drawdown ${pct(risk.drawdown)} of the ${pct(kill, 0)} kill switch`),
    el("div", { class: "meter", role: "meter", "aria-valuemin": "0", "aria-valuemax": String(kill),
      "aria-valuenow": String(drawdown), "aria-label": "Drawdown towards the kill switch" },
    el("span", { style: { width: `${(used * 100).toFixed(1)}%` } })),
    figures([
      ["Exposure scale", exposureScale(risk)],
      ["Daily loss limit", rows.risk_daily_loss_limit ? pct(rows.risk_daily_loss_limit.value, 1) : "–"],
      ["Orders today", `${risk.orders_today ?? "–"} of ${rows.risk_max_orders_per_day?.value ?? "–"}`],
      ["Today", risk.cooling_down ? "loss limit reached" : "trading"],
    ]));
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

function renderHistory(history) {
  const rows = history.map((o) => [
    brief(o.created_at), `${o.side} ${qty(o.qty)} ${o.symbol}`, o.status,
    o.avg_fill_price ? `${qty(o.filled_qty)} @ ${money(o.avg_fill_price)}` : "–",
  ]);
  $("history").replaceChildren(table(["Time", "Order", "Status", "Filled"], rows, "No orders yet."));
}

function renderFills(fills) {
  const rows = fills.map((f) => [brief(f.timestamp), `${f.side} ${qty(f.qty)} ${f.symbol}`, money(f.price), money(f.fee)]);
  $("fills").replaceChildren(table(["Time", "Fill", "Price", "Fee"], rows, "No fills yet."));
}

function renderEquity(points) {
  $("equity-chart").replaceChildren(lineChart(points.map((p) => ({ t: p.timestamp, v: Number(p.equity) })),
    { empty: "The curve appears after two recorded days." }));
}

async function loadOverview() {
  const status = await refreshStatus();
  // Positions and targets exist only while the engine has a snapshot
  const live = status.state === "running";
  const none = Promise.resolve({ ok: false });
  const [positions, targets, orders, fills, equity] = await Promise.all([
    live ? api("GET", "/api/v1/trading/positions") : none,
    live ? api("GET", "/api/v1/trading/targets") : none,
    api("GET", "/api/v1/trading/orders?limit=20"),
    api("GET", "/api/v1/trading/fills?limit=20"),
    api("GET", "/api/v1/trading/equity-curve?limit=500"),
    ctx.strategies().catch(() => null),
    ctx.loadProfile().catch(() => null),
  ]);
  renderAccount(status);
  renderPositions(positions.ok ? positions.data : null, targets.ok ? targets.data.targets : null);
  if (orders.ok) {
    renderWorking(orders.data.working);
    renderHistory(orders.data.history);
  }
  if (fills.ok) renderFills(fills.data.fills);
  if (equity.ok) renderEquity(equity.data.equity_curve);
  renderMix();
  renderRiskNow(status);
  renderControls(status);
  ctx.greet();
  $("updated").textContent = `Updated ${new Date().toLocaleTimeString()}`;
}

const overviewView = {
  async show() {
    try {
      await loadOverview();
    } catch (error) {
      if (error instanceof AuthError) return showLogin();
      $("updated").textContent = `Update failed: ${error.message}`;
    }
    return undefined;
  },
};

// -- the controls ---------------------------------------------------------------------------------

const ACTIONS = {
  start: () => ({
    title: "Start the engine?",
    text: "It loads the account from Alpaca, then trades the saved mix on the latest closed bars.",
    path: "/api/v1/trading/control/start",
  }),
  stop: () => ({
    title: "Stop the engine?",
    text: "Trading stops. Positions and working orders stay at Alpaca.",
    path: "/api/v1/trading/control/stop",
  }),
  restart: () => ({
    title: "Restart with the new mix?",
    text: `The engine stops and starts again trading profile #${ctx.profile?.saved?.id}. Positions in symbols `
      + "the new mix does not trade are sold.",
    path: "/api/v1/trading/control/restart",
  }),
  resume: () => ({
    title: "Lift the halt?",
    text: "Trading resumes and the high-water mark restarts at the current equity.",
    path: "/api/v1/trading/control/resume",
  }),
};

async function control(path, body) {
  const result = await api("POST", path, body);
  notify(result.ok ? (result.message || "Done.") : result.error, !result.ok);
  await overviewView.show();
  return result.ok;
}

async function onAction(action) {
  await ctx.guard(async () => {
    if (action === "halt") return confirmEmergencyStop();
    const spec = ACTIONS[action]();
    if (await ask(spec.title, spec.text)) await control(spec.path);
    return undefined;
  });
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
    "Stop now",
    "danger",
  );
  if (!confirmed) return;
  await control("/api/v1/trading/emergency-stop", {
    reason: reason.value.trim() || "Emergency stop from the dashboard",
    close_all_positions: close.checked,
  });
}

async function confirmCancel(order) {
  await ctx.guard(async () => {
    const text = `${order.side} ${qty(order.qty)} ${order.symbol} (${order.client_order_id})`;
    if (await ask("Cancel this order?", text)) {
      await control(`/api/v1/trading/orders/${encodeURIComponent(order.client_order_id)}/cancel`);
    }
  });
}

// -- sections --------------------------------------------------------------------------------------

const VIEWS = {
  overview: overviewView,
  strategies: strategiesView,
  lab: labView,
  risk: riskView,
  alerts: alertsView,
  settings: settingsView,
};

let current = "overview";

function route() {
  const [name, query] = location.hash.slice(1).split("?");
  current = VIEWS[name] ? name : "overview";
  for (const key of Object.keys(VIEWS)) $(`view-${key}`).hidden = key !== current;
  for (const link of document.querySelectorAll("#tabs a")) {
    if (link.dataset.view === current) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  // A look tried on the You page and not saved goes away
  if (ctx.prefs) {
    ctx.applyLook(ctx.prefs);
    ctx.applyCards(ctx.prefs);
  }
  const params = new URLSearchParams(query || "");
  ctx.guard(() => VIEWS[current].show(ctx, $(`view-${current}`), params));
}

// -- screens --------------------------------------------------------------------------------------

let timer = null;

function showLogin() {
  clearInterval(timer);
  timer = null;
  document.body.classList.remove("signed-in");
  $("app").hidden = true;
  $("login").hidden = false;
}

function tick() {
  if (document.visibilityState !== "visible") return;
  if (current === "overview") overviewView.show();
  else ctx.guard(refreshStatus);
}

async function showApp() {
  $("login").hidden = true;
  $("app").hidden = false;
  document.body.classList.add("signed-in");
  await ctx.guard(async () => {
    const [me, prefs] = await Promise.all([api("GET", "/api/v1/auth/me"), api("GET", "/api/v1/me/preferences")]);
    if (me.ok) {
      ctx.me = me.data.user;
      ctx.admin = me.data.user.is_admin === true;
      $("who").textContent = ctx.me.username;
    }
    if (prefs.ok) {
      ctx.prefs = prefs.data;
      ctx.applyLook(prefs.data);
      ctx.applyCards(prefs.data);
    }
  });
  route();
  clearInterval(timer);
  timer = setInterval(tick, REFRESH_EVERY_MS);
}

async function onLogin(event) {
  event.preventDefault();
  const form = event.target;
  const error = $("login-error");
  error.hidden = true;
  const button = form.querySelector("button");
  button.disabled = true;
  try {
    await signIn(form.username.value.trim(), form.password.value, form.remember.checked);
    form.password.value = "";
    await showApp();
  } catch (e) {
    error.textContent = e.message;
    error.hidden = false;
  } finally {
    button.disabled = false;
  }
}

async function logout() {
  try {
    await signOut();
  } finally {
    ctx.me = null;
    ctx.profile = null;
    ctx.draft = null;
    showLogin();
  }
}

ctx.control = control;
ctx.logout = logout;

async function start() {
  restoreLook();
  decorateIcons();
  $("login-form").addEventListener("submit", onLogin);
  $("logout").addEventListener("click", logout);
  for (const button of document.querySelectorAll("[data-action]")) {
    button.addEventListener("click", () => onAction(button.dataset.action));
  }
  window.addEventListener("hashchange", () => { if (timer) route(); });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && timer) tick();
  });
  if (!readRefreshToken()) return showLogin();
  try {
    await freshTokens();
    await showApp();
  } catch {
    showLogin();
  }
  return undefined;
}

start();
