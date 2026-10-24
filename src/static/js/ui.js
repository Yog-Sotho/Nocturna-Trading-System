// Building blocks: elements, icons, numbers, dialogs and charts.
//
// Every element is built with DOM calls, never innerHTML, so nothing the API
// returns can inject markup.

export const $ = (id) => document.getElementById(id);

const PROPERTIES = new Set(["value", "checked", "disabled", "selected", "hidden", "open", "indeterminate"]);

export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) {
      if (PROPERTIES.has(key)) node[key] = false;
      continue;
    }
    if (key === "class") node.className = value;
    else if (key === "style") Object.assign(node.style, value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (PROPERTIES.has(key)) node[key] = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  append(node, children);
  return node;
}

function append(node, children) {
  for (const child of children.flat()) {
    if (child !== null && child !== undefined && child !== false) {
      node.append(child instanceof Node ? child : String(child));
    }
  }
}

const SVG = "http://www.w3.org/2000/svg";

export function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  append(node, children);
  return node;
}

// -- icons (24 by 24, drawn with the current colour) --------------------------------------

const ICONS = {
  moon: ["M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"],
  layers: ["M12 2 2 7l10 5 10-5-10-5z", "M2 17l10 5 10-5", "M2 12l10 5 10-5"],
  flask: ["M9 3h6", "M10 3v6.5L4.5 19a2 2 0 0 0 1.7 3h11.6a2 2 0 0 0 1.7-3L14 9.5V3", "M7 15h10"],
  shield: ["M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"],
  bell: ["M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9", "M13.7 21a2 2 0 0 1-3.4 0"],
  sliders: ["M4 21v-7", "M4 10V3", "M12 21v-9", "M12 8V3", "M20 21v-5", "M20 12V3", "M1 14h6", "M9 8h6",
    "M17 16h6"],
  power: ["M18.4 6.6a9 9 0 1 1-12.8 0", "M12 2v10"],
  wallet: ["M3 7h15a3 3 0 0 1 3 3v8a3 3 0 0 1-3 3H6a3 3 0 0 1-3-3V7z", "M3 7l12-4 2 4", "M16 14h2"],
  chart: ["M3 3v18h18", "M7 15l4-4 3 3 6-7"],
  stack: ["M3 6h18v5H3z", "M3 13h18v5H3z"],
  list: ["M8 6h13", "M8 12h13", "M8 18h13", "M3 6h.01", "M3 12h.01", "M3 18h.01"],
  check: ["M20 6 9 17l-5-5"],
  lock: ["M5 11h14v10H5z", "M8 11V7a4 4 0 0 1 8 0v4"],
  plus: ["M12 5v14", "M5 12h14"],
  x: ["M18 6 6 18", "M6 6l12 12"],
  up: ["M12 19V5", "M5 12l7-7 7 7"],
  down: ["M12 5v14", "M19 12l-7 7-7-7"],
  send: ["M22 2 11 13", "M22 2 15 22l-4-9-9-4 20-7z"],
};

export function icon(name) {
  return svg("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": "2",
    "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true", focusable: "false" },
  ...(ICONS[name] || []).map((d) => svg("path", { d })));
}

export function decorateIcons(root = document) {
  for (const node of root.querySelectorAll("[data-icon]")) {
    if (!node.querySelector(":scope > svg")) node.prepend(icon(node.dataset.icon));
  }
}

// -- numbers and times ------------------------------------------------------------------------

const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });
const usd0 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const missing = (v) => v === null || v === undefined || v === "" || Number.isNaN(Number(v));

export const money = (v) => (missing(v) ? "–" : usd.format(Number(v)));
export const money0 = (v) => (missing(v) ? "–" : usd0.format(Number(v)));
export const pct = (v, digits = 2) => (missing(v) ? "–" : `${(Number(v) * 100).toFixed(digits)}%`);
export const signedPct = (v, digits = 2) => (missing(v) ? "–" : `${Number(v) >= 0 ? "+" : ""}${pct(v, digits)}`);
export const qty = (v) => (missing(v) ? "–" : String(Number(v)));
export const ratio = (v) => (missing(v) ? "–" : Number(v).toFixed(2));
export const weight = (v) => (missing(v) ? "–" : `${+(Number(v) * 100).toFixed(2)}%`);
export const when = (iso) => (iso ? new Date(iso).toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" }) : "–");
export const day = (iso) => (iso ? new Date(iso).toLocaleDateString(undefined, { dateStyle: "medium" }) : "–");
// Tables are narrow on a phone: month/day and time only
export const brief = (iso) => (iso ? new Date(iso).toLocaleString(undefined,
  { month: "numeric", day: "numeric", hour: "numeric", minute: "2-digit" }) : "–");

export const signed = (value, text) => el("span", { class: Number(value) >= 0 ? "up" : "down" }, text);

export function today() {
  const now = new Date();
  const local = new Date(now.getTime() - now.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 10);
}

// -- building blocks ----------------------------------------------------------------------------

export function table(headers, rows, empty) {
  if (!rows.length) return el("p", { class: "muted" }, empty);
  return el("div", { class: "scroll" },
    el("table", {},
      el("thead", {}, el("tr", {}, headers.map((h) => el("th", {}, h)))),
      el("tbody", {}, rows.map((cells) => el("tr", {}, cells.map((c) => el("td", {}, c)))))));
}

export function figures(items) {
  return el("dl", { class: "figures" }, items.map(([name, value, cls]) =>
    el("div", cls ? { class: cls } : {}, el("dt", {}, name), el("dd", {}, value))));
}

export function card(title, iconName, ...children) {
  return el("section", { class: "card" },
    el("div", { class: "card-head" }, el("h2", {}, icon(iconName), title)), ...children);
}

export function chip(text, cls = "") {
  return el("span", { class: `chip ${cls}`.trim() }, text);
}

let noticeTimer = null;

export function notify(text, failed = false) {
  const notice = $("notice");
  notice.textContent = text;
  notice.className = failed ? "notice failed" : "notice";
  notice.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { notice.hidden = true; }, failed ? 15000 : 6000);
}

export function ask(title, text, extra = [], ready = () => true, okLabel = "Confirm", tone = "primary") {
  const dialog = $("confirm");
  $("confirm-title").textContent = title;
  $("confirm-text").textContent = text;
  $("confirm-extra").replaceChildren(...extra);
  const ok = $("confirm-ok");
  ok.textContent = okLabel;
  ok.className = tone;
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

export function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = el("a", { href: url, download: name });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// -- the mix as a bar ---------------------------------------------------------------------------

// entries: [{label, share (0 to 1), index}]; what is left is held in T-bills
export function mixBar(entries) {
  const used = entries.reduce((sum, e) => sum + e.share, 0);
  const rest = Math.max(0, 1 - used);
  const parts = [...entries.map((e) => ({ ...e, cls: `c${e.index % 8}` })),
    ...(rest > 0.00005 ? [{ label: "T-bills (BIL)", share: rest, cls: "ctb" }] : [])];
  return [
    el("div", { class: "mixbar", role: "img",
      "aria-label": parts.map((p) => `${weight(p.share)} ${p.label}`).join(", ") },
    parts.map((p) => el("span", { class: p.cls, style: { flexGrow: String(p.share) } }))),
    el("ul", { class: "legend" }, parts.map((p) =>
      el("li", {}, el("span", { class: `swatch ${p.cls}` }), el("span", {}, p.label),
        el("span", { class: "share" }, weight(p.share))))),
  ];
}

// -- charts ---------------------------------------------------------------------------------

let gradients = 0;
const W = 1000;
const H = 300;

// points: [{t: ISO time, v: number}]
export function lineChart(points, { format = money, empty = "Not enough data yet.", label = "Equity" } = {}) {
  if (points.length < 2) return el("p", { class: "muted" }, empty);
  const values = points.map((p) => p.v);
  let low = Math.min(...values);
  let high = Math.max(...values);
  const pad = (high - low) * 0.06 || Math.abs(high) * 0.01 || 1;
  low -= pad;
  high += pad;
  const x = (i) => (i / (points.length - 1)) * W;
  const y = (v) => H - ((v - low) / (high - low)) * H;
  const coords = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
  const id = `area-${++gradients}`;
  const cursor = svg("line", { class: "cursor", x1: "0", x2: "0", y1: "0", y2: String(H), visibility: "hidden" });
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", role: "img",
    "aria-label": `${label} from ${format(values[0])} to ${format(values.at(-1))}` },
  svg("defs", {}, svg("linearGradient", { id, x1: "0", y1: "0", x2: "0", y2: "1" },
    svg("stop", { offset: "0", class: "stop-top" }), svg("stop", { offset: "1", class: "stop-bottom" }))),
  [0.25, 0.5, 0.75].map((f) => svg("line", { class: "grid", x1: "0", x2: String(W), y1: String(H * f),
    y2: String(H * f) })),
  svg("path", { d: `M0,${H} L${coords.join(" L")} L${W},${H} Z`, fill: `url(#${id})` }),
  svg("polyline", { class: "line", points: coords.join(" ") }),
  cursor);
  const dot = el("div", { class: "chart-dot", hidden: true });
  const tip = el("div", { class: "tip", hidden: true });
  const box = el("div", { class: "chart" }, chart, dot, tip);
  const hide = () => { cursor.setAttribute("visibility", "hidden"); dot.hidden = true; tip.hidden = true; };
  chart.addEventListener("pointermove", (event) => {
    const rect = chart.getBoundingClientRect();
    const i = Math.round(Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)) * (points.length - 1));
    const px = (x(i) / W) * rect.width;
    const py = (y(values[i]) / H) * rect.height;
    cursor.setAttribute("x1", String(x(i)));
    cursor.setAttribute("x2", String(x(i)));
    cursor.setAttribute("visibility", "visible");
    Object.assign(dot.style, { left: `${px}px`, top: `${py}px` });
    dot.hidden = false;
    tip.textContent = `${day(points[i].t)} · ${format(values[i])}`;
    Object.assign(tip.style, { left: `${Math.min(Math.max(px, 70), rect.width - 70)}px` });
    tip.hidden = false;
  });
  chart.addEventListener("pointerleave", hide);
  const change = values.at(-1) / values[0] - 1;
  return el("div", {}, box,
    el("div", { class: "chart-foot muted small" },
      el("span", {}, day(points[0].t)),
      el("span", {}, `low ${format(Math.min(...values))} · high ${format(Math.max(...values))} · `,
        signed(change, signedPct(change))),
      el("span", {}, day(points.at(-1).t))));
}

// rows: [{label, value}] around zero, e.g. the return of each year
export function barChart(rows, { format = signedPct } = {}) {
  if (!rows.length) return el("p", { class: "muted" }, "No full year yet.");
  const top = Math.max(0, ...rows.map((r) => r.value));
  const bottom = Math.min(0, ...rows.map((r) => r.value));
  const span = top - bottom || 1;
  const zero = (top / span) * H;
  const slot = W / rows.length;
  const tip = el("div", { class: "tip", hidden: true });
  const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", role: "img",
    "aria-label": rows.map((r) => `${r.label} ${format(r.value)}`).join(", ") },
  svg("line", { class: "grid", x1: "0", x2: String(W), y1: String(zero), y2: String(zero) }),
  rows.map((r, i) => {
    const h = (Math.abs(r.value) / span) * H;
    return svg("rect", { class: `bar ${r.value >= 0 ? "up" : "down"}`, x: String(i * slot + slot * 0.15),
      width: String(slot * 0.7), y: String(r.value >= 0 ? zero - h : zero), height: String(Math.max(h, 1)) });
  }));
  const box = el("div", { class: "chart bars" }, chart, tip);
  chart.addEventListener("pointermove", (event) => {
    const rect = chart.getBoundingClientRect();
    const i = Math.min(rows.length - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * rows.length)));
    tip.textContent = `${rows[i].label}: ${format(rows[i].value)}`;
    const px = ((i + 0.5) / rows.length) * rect.width;
    Object.assign(tip.style, { left: `${Math.min(Math.max(px, 60), rect.width - 60)}px` });
    tip.hidden = false;
  });
  chart.addEventListener("pointerleave", () => { tip.hidden = true; });
  const best = rows.reduce((a, b) => (b.value > a.value ? b : a));
  const worst = rows.reduce((a, b) => (b.value < a.value ? b : a));
  return el("div", {}, box,
    el("div", { class: "chart-foot muted small" },
      el("span", {}, rows[0].label),
      el("span", {}, `best ${best.label} `, signed(best.value, format(best.value)), ` · worst ${worst.label} `,
        signed(worst.value, format(worst.value))),
      el("span", {}, rows.at(-1).label)));
}
