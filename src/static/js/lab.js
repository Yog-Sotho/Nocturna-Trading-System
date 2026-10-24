// The strategy lab: backtest a mix, or validate a strategy against the promotion
// gates, and read the result. Runs execute one at a time on the server; nothing
// here trades or changes the saved mix.

import { api } from "./api.js";
import { mixForApi } from "./strategies.js";
import {
  barChart, card, chip, download, el, figures, icon, lineChart, money, notify, pct, ratio, signedPct, table, today,
  weight, when,
} from "./ui.js";

const POLL_MS = 4000;
const START = { alpaca: "2017-01-03", history: "1972-01-03" };
const LONG_HISTORY = ["SPY", "IEF", "BIL"];

let poll = null;
let selected = null;

function statusChip(run) {
  return run.status === "running" ? el("span", { class: "chip running" }, el("span", { class: "spinner" }), " running")
    : chip(run.status, run.status);
}

export const labView = {
  async show(ctx, root, params = new URLSearchParams()) {
    clearTimeout(poll);
    const [catalog, state] = await Promise.all([ctx.strategies(), ctx.loadProfile()]);
    const form = newRunForm(ctx, catalog, state, params, async (run) => {
      selected = run.id;
      await refresh();
    });
    const runs = el("ul", { class: "runs" });
    const detail = el("div");
    root.replaceChildren(form,
      card("Runs", "list", el("p", { class: "muted small" }, "Newest first. A run takes from seconds to many "
        + "minutes; this page follows it."), runs),
      detail);

    async function refresh() {
      clearTimeout(poll);
      if (root.hidden) return;
      const list = await api("GET", "/api/v1/lab/runs?limit=30");
      if (!list.ok) throw new Error(list.error);
      const all = list.data.runs;
      if (selected === null && all.length) selected = all[0].id;
      runs.replaceChildren(...(all.length ? all.map((run) => el("li", {},
        el("button", { "aria-current": String(run.id === selected), onclick: () => ctx.guard(async () => {
          selected = run.id;
          await refresh();
        }) },
        statusChip(run),
        el("span", {}, el("strong", {}, `#${run.id} `), run.title,
          run.headline ? el("span", { class: "muted small" }, ` · ${run.headline}`) : null,
          run.error ? el("span", { class: "error small" }, ` · ${run.error}`) : null),
        el("span", { class: "when" }, `${when(run.created_at)}${run.created_by ? ` by ${run.created_by}` : ""}`))))
        : [el("li", { class: "muted" }, "No run yet.")]));
      if (selected !== null) {
        const one = await api("GET", `/api/v1/lab/runs/${selected}`);
        detail.replaceChildren(one.ok ? runDetail(ctx, one.data.run) : el("p", { class: "error" }, one.error));
      }
      if (all.some((r) => r.status === "queued" || r.status === "running")) {
        poll = setTimeout(() => ctx.guard(refresh), POLL_MS);
      }
    }

    await refresh();
  },
};

function newRunForm(ctx, catalog, state, params, onQueued) {
  const wanted = params.get("strategy");
  const kind = wanted ? "validate" : "backtest";
  const backtest = el("input", { type: "radio", name: "kind", value: "backtest", checked: kind === "backtest" });
  const validate = el("input", { type: "radio", name: "kind", value: "validate", checked: kind === "validate" });
  const strategy = el("select", { name: "strategy" }, catalog.map((s) =>
    el("option", { value: s.name, selected: s.name === wanted }, `${s.title} (${s.verdict})`)));
  const data = el("select", { name: "data" },
    el("option", { value: "alpaca" }, "Alpaca daily bars (from 2016; crypto from 2021)"),
    el("option", { value: "history" }, "Long history from 1970 (SPY, IEF, BIL only)"));
  const start = el("input", { type: "date", name: "start", value: START.alpaca, min: "1970-01-01", max: today() });
  const end = el("input", { type: "date", name: "end", value: today(), min: "1970-01-02", max: today() });
  const mixNote = el("p", { class: "small" });
  const hint = el("p", { class: "faint small" });
  const submit = el("button", { class: "primary", disabled: !ctx.admin }, icon("flask"), " Run");

  const mix = () => (ctx.draft ? mixForApi(ctx.draft) : state.saved.mix);
  const symbolsOf = (names) => names.flatMap((n) => catalog.find((s) => s.name === n)?.symbols || []);
  const update = () => {
    const isValidate = validate.checked;
    strategy.closest("label").hidden = !isValidate;
    mixNote.hidden = isValidate;
    const entries = mix();
    mixNote.replaceChildren(el("strong", {}, "The mix: "),
      entries.map((e) => `${weight(e.weight)} ${ctx.title(e.strategy)}`).join(", ") || "empty",
      ctx.draft ? el("span", { class: "muted" }, " (as on the Strategies page, saved or not)") : null,
      el("span", { class: "muted" }, ". Risk limits: the saved ones."));
    const symbols = isValidate ? symbolsOf([strategy.value]) : symbolsOf(entries.map((e) => e.strategy));
    const historyFits = symbols.every((s) => LONG_HISTORY.includes(s));
    data.options[1].disabled = !historyFits;
    if (!historyFits && data.value === "history") data.value = "alpaca";
    hint.textContent = isValidate
      ? "A validation needs at least 4 years: 3 to choose the parameters, then at least 1 to test them out of sample."
      : "A backtest trades the mix through the same engine, costs and risk limits as paper trading.";
  };
  for (const input of [backtest, validate, strategy]) input.addEventListener("change", update);
  data.addEventListener("change", () => {
    if (start.value === START.alpaca || start.value === START.history) start.value = START[data.value];
  });

  submit.addEventListener("click", (event) => ctx.guard(async () => {
    event.preventDefault();
    const body = validate.checked
      ? { kind: "validate", strategy: strategy.value, start: start.value, end: end.value || null, data: data.value }
      : { kind: "backtest", mix: mix(), risk: state.saved.risk, start: start.value, end: end.value || null,
        data: data.value };
    const result = await api("POST", "/api/v1/lab/runs", body);
    if (!result.ok) {
      notify(result.error, true);
      return;
    }
    notify(`Queued run #${result.data.run.id}.`);
    await onQueued(result.data.run);
  }));

  const node = card("New run", "flask",
    el("div", { class: "segmented", role: "radiogroup", "aria-label": "What to run" },
      el("label", {}, backtest, el("span", {}, "Backtest a mix")),
      el("label", {}, validate, el("span", {}, "Validate a strategy"))),
    el("div", { class: "form-grid" },
      el("label", {}, "Strategy", strategy),
      el("label", {}, "Data", data),
      el("label", {}, "From", start),
      el("label", {}, "To", end)),
    mixNote, hint,
    el("div", { class: "buttons" }, submit),
    ctx.admin ? null : el("p", { class: "muted small" }, "Only an admin can start a run."));
  update();
  return node;
}

function runDetail(ctx, run) {
  const parts = [el("p", { class: "muted small" }, `${run.title}. Asked ${when(run.created_at)}`
    + `${run.created_by ? ` by ${run.created_by}` : ""}${run.finished_at ? `; finished ${when(run.finished_at)}` : ""}.`)];
  if (run.status === "queued" || run.status === "running") {
    parts.push(el("p", {}, el("span", { class: "spinner" }), run.status === "queued"
      ? " Waiting for the runs before it." : " Computing. This page updates by itself."));
  } else if (run.status === "failed") {
    parts.push(el("p", { class: "verdict rejected" }, run.error || "Failed."));
  } else if (run.kind === "backtest") {
    parts.push(...backtestResult(run.result));
  } else {
    parts.push(...validationResult(run));
  }
  return card(`Run #${run.id}`, run.kind === "backtest" ? "chart" : "check", ...parts);
}

function backtestResult(result) {
  const s = result.summary;
  return [
    el("p", { class: "verdict" }, result.headline),
    figures([
      ["A year", signedPct(s.cagr), "big"],
      ["In all", signedPct(s.total_return)],
      ["T-bills in all", signedPct(s.hurdle_return)],
      ["Sharpe", ratio(s.sharpe)],
      ["Sortino", ratio(s.sortino)],
      ["Volatility", pct(s.volatility)],
      ["Worst drawdown", pct(s.max_drawdown)],
      ["Kill switch trips", String(result.kill_switch_trips)],
      ["Fills", String(result.fills)],
      ["Fees", money(result.fees)],
      ["Start", money(result.starting_cash)],
      ["End", money(result.final_equity)],
    ]),
    el("h3", {}, "Equity"),
    lineChart(result.curve.map((p) => ({ t: p.timestamp, v: Number(p.equity) }))),
    el("h3", {}, "Each year"),
    barChart(result.years.map((y) => ({ label: String(y.year), value: y.return }))),
    el("p", { class: "faint small" }, `Data: ${result.data}. Sharpe and Sortino in excess of ${result.hurdle}. `
      + "A drawdown kill is followed by a simulated resume 30 days later."),
  ];
}

function gateRows(gates) {
  return gates.map((g) => [
    g.name,
    g.passed === true ? chip("pass", "passed") : g.passed === false ? chip("fail", "failed") : chip("n/a"),
    g.value === null || g.value === undefined ? "–" : Math.abs(g.value) < 10 ? ratio(g.value) : String(g.value),
    g.threshold,
    el("span", { class: "small muted" }, g.note || ""),
  ]);
}

function validationResult(run) {
  const result = run.result;
  const report = result.report;
  const oos = report.out_of_sample;
  return [
    el("p", { class: `verdict ${result.passed ? "passed" : "rejected"}` }, result.headline),
    figures([
      ["Out of sample, a year", signedPct(oos.cagr), "big"],
      ["Sharpe out of sample", ratio(oos.sharpe)],
      ["Worst drawdown", pct(oos.max_drawdown)],
      ["Deflated Sharpe", ratio(report.deflated_sharpe?.dsr)],
      ["Overfitting (PBO)", report.overfitting ? pct(report.overfitting.pbo, 0) : "–"],
      ["Parameter sets", String(report.trials.length)],
    ]),
    el("h3", {}, "The gates"),
    table(["Gate", "", "Value", "Needs", ""], gateRows(report.gates), "No gates."),
    el("h3", {}, "Equity traded out of sample"),
    lineChart(result.curve.map((p) => ({ t: p.timestamp, v: Number(p.equity) }))),
    el("p", { class: "small" }, `Best parameters in the whole period: ${JSON.stringify(report.selected.params)}. `
      + `Data: ${result.data}.`),
    el("details", {}, el("summary", {}, "The full tear sheet"), el("pre", { class: "tearsheet" }, run.report || "")),
    el("div", { class: "buttons" },
      el("button", { onclick: () => download(`validation-${report.candidate}-${run.id}.md`, run.report || "",
        "text/markdown") }, "Download the tear sheet"),
      el("button", { onclick: () => download(`validation-${report.candidate}-${run.id}.json`,
        JSON.stringify(report, null, 2), "application/json") }, "Download the JSON")),
  ];
}
