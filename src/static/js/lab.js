// The strategy lab: backtest a mix, or validate a strategy against the promotion
// gates, and read the result next to the 60/40. Runs execute one at a time on the
// server; they can be cancelled, run again or copied into the form, and their
// equity downloaded. Nothing here trades or changes the saved mix.

import { api } from "./api.js";
import { closingConflict, editableMix, mixForApi, mixRow, round2 } from "./strategies.js";
import {
  ask, barChart, card, chip, download, el, figures, icon, lineChart, mixBar, money, notify, pct, ratio, signedPct,
  table, today, when,
} from "./ui.js";

const POLL_MS = 4000;
const START = { alpaca: "2017-01-03", history: "1972-01-03" };
const LONG_HISTORY = ["SPY", "IEF", "BIL"];
const WAITING = ["queued", "running"];

let poll = null;
let selected = null;

function statusChip(run) {
  return run.status === "running" ? el("span", { class: "chip running" }, el("span", { class: "spinner" }), " running")
    : chip(run.status, run.status);
}

// "4 s", "2 min 5 s", "1 h 3 min"
function duration(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min${s % 60 ? ` ${s % 60} s` : ""}`;
  return `${Math.floor(s / 3600)} h${Math.floor((s % 3600) / 60) ? ` ${Math.floor((s % 3600) / 60)} min` : ""}`;
}

// How long a run took, or has been running
function elapsed(run) {
  if (!run.started_at) return null;
  const end = run.finished_at ? new Date(run.finished_at) : new Date();
  const took = duration(end - new Date(run.started_at));
  return run.status === "running" ? `running for ${took}` : `took ${took}`;
}

function csv(rows) {
  return rows.map((row) => row.map((v) => (/[",\n]/.test(String(v)) ? `"${String(v).replaceAll('"', '""')}"` : v))
    .join(",")).join("\n") + "\n";
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
    root.replaceChildren(form.node,
      card("Runs", "list", el("p", { class: "muted small" }, "Newest first. A run takes from seconds to many "
        + "minutes; this page follows it."), runs),
      detail);

    const actions = {
      refresh: () => refresh(),
      edit: (run) => {
        form.load(run);
        form.node.scrollIntoView({ behavior: "smooth", block: "start" });
        notify(`The form now holds run #${run.id}: change what you like, then Run.`);
      },
      queued: async (run) => {
        selected = run.id;
        await refresh();
      },
    };

    async function refresh() {
      clearTimeout(poll);
      if (root.hidden) return;
      const list = await api("GET", "/api/v1/lab/runs?limit=30");
      if (!list.ok) throw new Error(list.error);
      const all = list.data.runs;
      if (selected !== null && !all.some((r) => r.id === selected)) selected = null;
      if (selected === null && all.length) selected = all[0].id;
      runs.replaceChildren(...(all.length ? all.map((run) => el("li", {},
        el("button", { "aria-current": String(run.id === selected), onclick: () => ctx.guard(async () => {
          selected = run.id;
          await refresh();
        }) },
        statusChip(run),
        el("span", {}, el("strong", {}, `#${run.id} `), run.title,
          run.headline ? el("span", { class: "muted small" }, ` · ${run.headline}`) : null,
          run.error && run.status !== "cancelled" ? el("span", { class: "error small" }, ` · ${run.error}`) : null),
        el("span", { class: "when" }, `${when(run.created_at)}${run.created_by ? ` by ${run.created_by}` : ""}`
          + `${elapsed(run) ? ` · ${elapsed(run)}` : ""}`))))
        : [el("li", { class: "muted" }, "No run yet.")]));
      if (selected !== null) {
        const one = await api("GET", `/api/v1/lab/runs/${selected}`);
        detail.replaceChildren(one.ok ? runDetail(ctx, one.data.run, actions) : el("p", { class: "error" }, one.error));
      } else {
        detail.replaceChildren();
      }
      if (all.some((r) => WAITING.includes(r.status))) {
        poll = setTimeout(() => ctx.guard(refresh), POLL_MS);
      }
    }

    await refresh();
  },
};

// -- the form ---------------------------------------------------------------------------------

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
  const dataNote = el("p", { class: "small muted" });
  const hint = el("p", { class: "faint small" });
  const submit = el("button", { class: "primary", disabled: !ctx.admin }, icon("flask"), " Run");

  // The lab's own mix, apart from the one the engine trades: it starts as the Strategies page has it
  const lab = {
    mix: editableMix(ctx.draft ? mixForApi(ctx.draft) : state.saved.mix),
    // The risk limits a backtest runs with: the saved ones, or a past run's when it was copied here
    risk: state.saved.risk,
    riskFrom: null,
  };
  const mixBox = el("div", { class: "lab-mix" });

  // The strategies the long history cannot serve, each with the symbols it lacks
  const outsideHistory = (names) => names.map((n) => {
    const missing = (catalog.find((s) => s.name === n)?.symbols || []).filter((s) => !LONG_HISTORY.includes(s));
    return missing.length ? `${ctx.title(n)} (${missing.join(", ")})` : null;
  }).filter(Boolean);
  // The first day follows the data, unless the operator chose one
  const followData = () => {
    if (start.value === START.alpaca || start.value === START.history) start.value = START[data.value];
    describeData();
  };
  let blocked = [];
  const describeData = () => {
    dataNote.textContent = blocked.length
      ? `The long history is not offered here: it has SPY, IEF and T-bills only, and ${blocked.join(", ")} `
        + `${blocked.length === 1 ? "trades" : "trade"} more than that.`
      : data.value === "history"
        ? "The long history is published every month or two, so it ends some weeks back: a run stops on its last "
          + "day, whatever the To date, and says so."
        : "";
    dataNote.hidden = !dataNote.textContent;
  };

  const problems = () => {
    if (validate.checked) return [];
    const used = round2(lab.mix.reduce((sum, e) => sum + e.pct, 0));
    const issues = [];
    if (!lab.mix.length) issues.push("Add at least one strategy.");
    if (used > 100) issues.push(`The shares add up to ${used}%: at most 100% (the rest is held in T-bills).`);
    if (lab.mix.some((e) => !(e.pct > 0))) issues.push("Each strategy needs a share above 0%.");
    const conflict = closingConflict(ctx, lab.mix);
    if (conflict) issues.push(conflict);
    return issues;
  };

  const renderMix = () => {
    const used = round2(lab.mix.reduce((sum, e) => sum + e.pct, 0));
    const missing = catalog.filter((s) => !lab.mix.some((e) => e.strategy === s.name));
    const adder = el("select", { "aria-label": "Add a strategy", disabled: !ctx.admin || !missing.length },
      el("option", { value: "" }, missing.length ? "Add a strategy…" : "Every strategy is in the mix"),
      missing.map((s) => el("option", { value: s.name }, `${s.title} (${s.verdict})`)));
    adder.addEventListener("change", () => {
      const s = catalog.find((c) => c.name === adder.value);
      if (!s) return;
      lab.mix.push({ strategy: s.name, params: { ...s.defaults }, pct: used < 100 ? round2(100 - used) : 10 });
      update();
    });
    const issues = problems();
    mixBox.replaceChildren(
      el("div", { class: "card-head" }, el("h3", {}, "The mix to backtest"),
        el("span", { class: "muted small" }, "Here only: the engine keeps trading its own")),
      mixBar(lab.mix.map((e) => ({ label: ctx.title(e.strategy), share: e.pct / 100, index: ctx.colour(e.strategy) })))[0],
      ...lab.mix.map((entry, i) => mixRow(ctx, entry, {
        editable: ctx.admin, onChange: () => { refreshProblems(); },
        onRemove: () => { lab.mix.splice(i, 1); update(); } })),
      el("div", { class: "total" }, el("span", {}, `Strategies ${used}%`),
        el("span", { class: "muted" }, `T-bills (BIL) ${round2(Math.max(0, 100 - used))}%`)),
      el("div", { class: "buttons" }, adder,
        el("button", { class: "link", disabled: !ctx.admin, onclick: () => {
          lab.mix = editableMix(state.saved.mix);
          update();
        } }, "Use your saved mix"),
        ctx.draft ? el("button", { class: "link", disabled: !ctx.admin, onclick: () => {
          lab.mix = editableMix(mixForApi(ctx.draft));
          update();
        } }, "Use the Strategies page's") : null),
      el("p", { class: "error small", role: "alert", hidden: !issues.length }, issues.join(" ")),
      el("p", { class: "small muted" }, lab.riskFrom
        ? [`Risk limits: those of run #${lab.riskFrom}. `, el("button", { class: "link", onclick: () => {
          lab.risk = state.saved.risk;
          lab.riskFrom = null;
          update();
        } }, "Use the saved ones")]
        : "Risk limits: the saved ones (Risk page)."));
  };
  const refreshProblems = () => {
    const bar = mixBox.querySelector(".mixbar");
    if (bar) bar.replaceWith(mixBar(lab.mix.map((e) => ({ label: ctx.title(e.strategy), share: e.pct / 100,
      index: ctx.colour(e.strategy) })))[0]);
    const used = round2(lab.mix.reduce((sum, e) => sum + e.pct, 0));
    const total = mixBox.querySelector(".total");
    if (total) total.replaceChildren(el("span", {}, `Strategies ${used}%`),
      el("span", { class: "muted" }, `T-bills (BIL) ${round2(Math.max(0, 100 - used))}%`));
    const issues = problems();
    const alert = mixBox.querySelector("[role=alert]");
    if (alert) {
      alert.textContent = issues.join(" ");
      alert.hidden = !issues.length;
    }
    submit.disabled = !ctx.admin || issues.length > 0;
  };

  const update = () => {
    const isValidate = validate.checked;
    strategy.closest("label").hidden = !isValidate;
    mixBox.hidden = isValidate;
    if (!isValidate) renderMix();
    blocked = outsideHistory(isValidate ? [strategy.value] : lab.mix.map((e) => e.strategy));
    data.options[1].disabled = blocked.length > 0;
    if (blocked.length && data.value === "history") data.value = "alpaca";
    followData();
    hint.textContent = isValidate
      ? "A validation needs at least 4 years: 3 to choose the parameters, then at least 1 to test them out of sample."
      : "A backtest trades the mix through the same engine, costs and risk limits as paper trading, next to the "
        + "60/40 over the same days.";
    submit.disabled = !ctx.admin || problems().length > 0;
  };
  for (const input of [backtest, validate, strategy]) input.addEventListener("change", update);
  data.addEventListener("change", followData);

  const body = () => (validate.checked
    ? { kind: "validate", strategy: strategy.value, start: start.value, end: end.value || null, data: data.value }
    : { kind: "backtest", mix: lab.mix.map((e) => ({ strategy: e.strategy, params: e.params,
      weight: (e.pct / 100).toFixed(4) })), risk: lab.risk, start: start.value, end: end.value || null,
    data: data.value });

  submit.addEventListener("click", (event) => ctx.guard(async () => {
    event.preventDefault();
    const result = await api("POST", "/api/v1/lab/runs", body());
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
    mixBox, dataNote, hint,
    el("div", { class: "buttons" }, submit),
    ctx.admin ? null : el("p", { class: "muted small" }, "Only an admin can start a run."));
  update();

  // Fill the form from a past run: what it ran, on which data and days, with which limits
  const load = (run) => {
    const spec = run.spec;
    backtest.checked = run.kind === "backtest";
    validate.checked = run.kind === "validate";
    if (run.kind === "validate") strategy.value = spec.strategy;
    else {
      lab.mix = editableMix(spec.profile.mix);
      lab.risk = spec.profile.risk || {};
      lab.riskFrom = run.id;
    }
    update();
    data.value = spec.data;
    start.value = spec.start;
    end.value = spec.end;
    describeData();
  };
  return { node, load };
}

// -- a run --------------------------------------------------------------------------------------

// The request that runs it again, exactly as it was
function again(run) {
  const spec = run.spec;
  return run.kind === "validate"
    ? { kind: "validate", strategy: spec.strategy, start: spec.start, end: spec.end, data: spec.data }
    : { kind: "backtest", mix: spec.profile.mix, risk: spec.profile.risk || {}, start: spec.start, end: spec.end,
      data: spec.data };
}

function runActions(ctx, run, actions) {
  const waiting = WAITING.includes(run.status);
  const buttons = [];
  if (ctx.admin && waiting) {
    buttons.push(el("button", { onclick: () => ctx.guard(async () => {
      const result = await api("POST", `/api/v1/lab/runs/${run.id}/cancel`);
      notify(result.ok ? `Run #${run.id} cancelled${run.status === "running" ? ": it stops within seconds" : ""}.`
        : result.error, !result.ok);
      await actions.refresh();
    }) }, icon("x"), " Cancel"));
  }
  if (ctx.admin && !waiting) {
    buttons.push(el("button", { onclick: () => ctx.guard(async () => {
      const result = await api("POST", "/api/v1/lab/runs", again(run));
      if (!result.ok) {
        notify(result.error, true);
        return;
      }
      notify(`Queued run #${result.data.run.id}, as run #${run.id}.`);
      await actions.queued(result.data.run);
    }) }, icon("flask"), " Run again"));
  }
  if (ctx.admin) buttons.push(el("button", { onclick: () => actions.edit(run) }, "Copy into the form"));
  if (run.status === "done") {
    buttons.push(el("button", { onclick: () => ctx.guard(async () => {
      const result = await api("GET", `/api/v1/lab/runs/${run.id}/equity`);
      if (!result.ok) {
        notify(result.error, true);
        return;
      }
      download(result.data.filename, result.data.csv, "text/csv");
    }) }, "Equity (CSV)"));
    if (run.kind === "backtest") {
      const other = new Map((run.result.benchmark?.years || []).map((y) => [y.year, y.return]));
      buttons.push(el("button", { onclick: () => download(`lab-${run.id}-years.csv`, csv([
        ["year", "the mix", ...(other.size ? [run.result.benchmark.title] : [])],
        ...run.result.years.map((y) => [y.year, y.return, ...(other.size ? [other.get(y.year) ?? ""] : [])]),
      ]), "text/csv") }, "Each year (CSV)"));
    }
  }
  if (ctx.admin && !waiting) {
    buttons.push(el("button", { class: "danger", onclick: () => ctx.guard(async () => {
      if (!await ask(`Delete run #${run.id}?`, "Its result is gone for good. Nothing else changes.", [], () => true,
        "Delete", "danger")) return;
      const result = await api("DELETE", `/api/v1/lab/runs/${run.id}`);
      notify(result.ok ? `Run #${run.id} deleted.` : result.error, !result.ok);
      await actions.refresh();
    }) }, "Delete"));
  }
  return buttons.length ? el("div", { class: "buttons" }, buttons) : null;
}

function runDetail(ctx, run, actions) {
  const took = elapsed(run);
  const parts = [el("p", { class: "muted small" }, `${run.title}. Asked ${when(run.created_at)}`
    + `${run.created_by ? ` by ${run.created_by}` : ""}${run.finished_at ? `; finished ${when(run.finished_at)}` : ""}`
    + `${took ? `; ${took}` : ""}.`)];
  if (run.status === "queued" || run.status === "running") {
    parts.push(el("p", {}, el("span", { class: "spinner" }), run.status === "queued"
      ? " Waiting for the runs before it." : ` Computing (${took}). This page updates by itself.`));
  } else if (run.status === "cancelled") {
    parts.push(el("p", { class: "verdict" }, "Cancelled. Nothing was stored."));
  } else if (run.status === "failed") {
    parts.push(el("p", { class: "verdict rejected" }, run.error || "Failed."));
  } else {
    const [verdict, ...rest] = run.kind === "backtest" ? backtestResult(run.result) : validationResult(run);
    // Where the bars fell short of the period asked for, right under the verdict (runs before this had none)
    const notes = (run.result.notes || []).map((note) => el("li", {}, note));
    parts.push(verdict, notes.length ? el("ul", { class: "run-notes" }, notes) : null, ...rest);
  }
  parts.push(runActions(ctx, run, actions));
  return card(`Run #${run.id}`, run.kind === "backtest" ? "chart" : "check", ...parts);
}

function backtestResult(result) {
  const s = result.summary;
  const plain = result.benchmark;
  const b = plain?.summary;
  const rows = [
    ["A year", signedPct, "cagr"], ["In all", signedPct, "total_return"], ["Sharpe", ratio, "sharpe"],
    ["Sortino", ratio, "sortino"], ["Volatility", pct, "volatility"], ["Worst drawdown", pct, "max_drawdown"],
  ];
  return [
    el("p", { class: "verdict" }, result.headline),
    plain ? el("h3", {}, `Next to the ${plain.title}`) : null,
    plain ? table(["", "The mix", plain.title], [
      ...rows.map(([label, format, key]) => [label, format(s[key]), format(b[key])]),
      ["Kill switch trips", String(result.kill_switch_trips), String(plain.kill_switch_trips)],
      ["End", money(result.final_equity), money(plain.final_equity)],
    ], "") : el("p", { class: "small muted" }, "This mix is the 60/40 itself, the yardstick every mix is compared with."),
    figures([
      ...(plain ? [] : rows.map(([label, format, key], i) => [label, format(s[key]), i === 0 ? "big" : undefined])),
      ["T-bills in all", signedPct(s.hurdle_return)],
      ...(plain ? [] : [["Kill switch trips", String(result.kill_switch_trips)]]),
      ["Fills", String(result.fills)],
      ["Fees", money(result.fees)],
      ["Start", money(result.starting_cash)],
      ...(plain ? [] : [["End", money(result.final_equity)]]),
    ]),
    el("h3", {}, "Equity"),
    lineChart(result.curve.map((p) => ({ t: p.timestamp, v: Number(p.equity) })), plain ? {
      label: "The mix",
      compare: { label: plain.title, points: plain.curve.map((p) => ({ t: p.timestamp, v: Number(p.equity) })) },
    } : {}),
    el("h3", {}, "Each year"),
    barChart(result.years.map((y) => ({ label: String(y.year), value: y.return })), plain ? {
      compare: { label: plain.title, values: new Map(plain.years.map((y) => [String(y.year), y.return])) },
    } : {}),
    el("p", { class: "faint small" }, `Data: ${result.data}. Sharpe and Sortino in excess of ${result.hurdle}. `
      + "A drawdown kill is followed by a simulated resume 30 days later."
      + (plain ? ` The ${plain.title} runs on the same days, data and risk limits.` : "")),
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
