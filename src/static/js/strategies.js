// The Strategies page (the mix and the catalog) and the Risk page (the limits).
// Both edit the trading profile: PUT /api/v1/trading/profile saves a new version,
// which the engine trades from its next start.

import { api } from "./api.js";
import { ask, card, chip, el, icon, mixBar, notify, pct, weight } from "./ui.js";

const round2 = (n) => Math.round(n * 100) / 100;

function draftFrom(profile) {
  return {
    mix: profile.mix.map((e) => ({ strategy: e.strategy, params: { ...e.params }, pct: round2(Number(e.weight) * 100) })),
    note: profile.note || "",
  };
}

export function mixForApi(draft) {
  return draft.mix.map((e) => ({ strategy: e.strategy, params: e.params, weight: (e.pct / 100).toFixed(4) }));
}

function sameValue(a, b) {
  return typeof a === typeof b && a === b;
}

function changed(draft, saved) {
  const theirs = draftFrom(saved);
  return JSON.stringify(mixForApi(draft)) !== JSON.stringify(mixForApi(theirs)) || draft.note.trim() !== theirs.note;
}

function closingConflict(ctx, mix) {
  const of = (name) => (ctx.catalog || []).find((s) => s.name === name)?.execution;
  const closing = mix.filter((e) => of(e.strategy) === "close").map((e) => e.strategy);
  if (!closing.length || closing.length === mix.length) return null;
  return `${closing.join(", ")} trades in the closing auction and cannot be mixed with strategies that trade at the open.`;
}

async function offerRestart(ctx, saved) {
  const running = ["running", "starting"].includes(ctx.status?.state);
  if (!running) return;
  if (await ask("Restart the engine now?",
    `Profile #${saved.id} is saved. The running engine keeps its mix until it restarts. Positions in symbols the `
    + "new mix does not trade are sold.", [], () => true, "Restart now")) {
    await ctx.control("/api/v1/trading/control/restart");
  }
}

// -- the Strategies page -------------------------------------------------------------------------

export const strategiesView = {
  async show(ctx, root) {
    const [catalog, state] = await Promise.all([ctx.strategies(true), ctx.loadProfile()]);
    if (!ctx.draft) ctx.draft = draftFrom(state.saved);
    const editor = el("div");
    const list = el("div", { class: "grid-2" });

    const save = async () => {
      const result = await api("PUT", "/api/v1/trading/profile",
        { mix: mixForApi(ctx.draft), risk: state.saved.risk, note: ctx.draft.note.trim() });
      if (!result.ok) {
        notify(result.error, true);
        return;
      }
      notify(`Saved as profile #${result.data.profile.id}. The engine trades it from its next start.`);
      ctx.draft = draftFrom(result.data.profile);
      await strategiesView.show(ctx, root);
      if (result.data.restart_needed) await offerRestart(ctx, result.data.profile);
    };

    const renderEditor = () => {
      editor.replaceChildren(mixEditor(ctx, state, { save, rerender: () => { renderEditor(); renderList(); } }));
    };
    const renderList = () => {
      list.replaceChildren(...catalog.map((s) => strategyCard(ctx, s, {
        add: async () => {
          if (s.verdict === "rejected") {
            const understood = el("input", { type: "checkbox" });
            const ok = await ask(`Add ${s.title}?`,
              `Validation rejected it: ${s.verdict_detail} You can still paper-trade it, with money that is not real.`,
              [el("label", { class: "check" }, understood, "I have read why it was rejected")],
              () => understood.checked, "Add it anyway");
            if (!ok) return;
          }
          const used = ctx.draft.mix.reduce((sum, e) => sum + e.pct, 0);
          ctx.draft.mix.push({ strategy: s.name, params: { ...s.defaults }, pct: used < 100 ? round2(100 - used) : 10 });
          renderEditor();
          renderList();
          editor.scrollIntoView({ behavior: "smooth", block: "start" });
        },
      })));
    };
    renderEditor();
    renderList();
    root.replaceChildren(editor,
      el("div", { class: "card-head" },
        el("h2", {}, icon("layers"), "Every strategy"),
        el("span", { class: "muted small" }, "What validation concluded about each. Paper only.")),
      list);
  },
};

function mixEditor(ctx, state, { save, rerender }) {
  const draft = ctx.draft;
  const admin = ctx.admin;
  const strategyOf = (name) => ctx.catalog.find((s) => s.name === name);
  const bar = el("div");
  const total = el("div", { class: "total" });
  const problems = el("p", { class: "error small", role: "alert" });
  const saveButton = el("button", { class: "primary", onclick: () => ctx.guard(save) }, "Save the mix");
  const dirty = el("span");

  const update = () => {
    const used = round2(draft.mix.reduce((sum, e) => sum + e.pct, 0));
    bar.replaceChildren(mixBar(draft.mix.map((e) => ({ label: ctx.title(e.strategy), share: e.pct / 100,
      index: ctx.colour(e.strategy) })))[0]);
    total.replaceChildren(el("span", {}, `Strategies ${used}%`),
      el("span", { class: "muted" }, `T-bills (BIL) ${round2(Math.max(0, 100 - used))}%`));
    const issues = [];
    if (!draft.mix.length) issues.push("Add at least one strategy.");
    if (used > 100) issues.push(`The shares add up to ${used}%: at most 100% (the rest is held in T-bills).`);
    if (draft.mix.some((e) => !(e.pct > 0))) issues.push("Each strategy needs a share above 0%.");
    const conflict = closingConflict(ctx, draft.mix);
    if (conflict) issues.push(conflict);
    problems.textContent = issues.join(" ");
    problems.hidden = !issues.length;
    const isDirty = changed(draft, state.saved);
    saveButton.disabled = !admin || issues.length > 0 || !isDirty;
    dirty.replaceChildren(isDirty ? chip("unsaved changes", "queued")
      : chip(state.saved.id === null ? "the default mix" : `saved: profile #${state.saved.id}`, "done"));
  };

  const rows = draft.mix.map((entry, i) => {
    const info = strategyOf(entry.strategy);
    const number = el("input", { type: "number", min: "0.5", max: "100", step: "0.5", value: String(entry.pct),
      disabled: !admin, "aria-label": `${ctx.title(entry.strategy)} share, percent` });
    const range = el("input", { type: "range", min: "0", max: "100", step: "0.5", value: String(entry.pct),
      disabled: !admin, "aria-label": `${ctx.title(entry.strategy)} share` });
    const set = (value) => {
      entry.pct = round2(Math.min(100, Math.max(0, Number(value) || 0)));
      number.value = String(entry.pct);
      range.value = String(entry.pct);
      range.style.setProperty("--fill", `${entry.pct}%`);
      update();
    };
    range.style.setProperty("--fill", `${entry.pct}%`);
    number.addEventListener("change", () => set(number.value));
    range.addEventListener("input", () => set(range.value));
    const params = (info?.parameters || []).map((p) => {
      const current = p.choices.findIndex((c) => sameValue(c.value, entry.params[p.name]));
      const select = el("select", { disabled: !admin, onchange: () => {
        entry.params[p.name] = p.choices[Number(select.value)].value;
        update();
      } }, p.choices.map((c, index) => el("option", { value: String(index), selected: index === current }, c.label)));
      return el("label", {}, p.label, select);
    });
    return el("div", { class: "mix-row" },
      el("div", { class: "title" }, el("span", { class: `swatch c${ctx.colour(entry.strategy) % 8}` }),
        ctx.title(entry.strategy), info?.verdict === "rejected" ? chip("rejected", "rejected") : null),
      admin ? el("button", { class: "link", "aria-label": `Remove ${ctx.title(entry.strategy)}`, onclick: () => {
        draft.mix.splice(i, 1);
        rerender();
      } }, icon("x")) : el("span"),
      el("div", { class: "weight" }, range, el("label", { class: "check" }, number, "%")),
      params.length ? el("div", { class: "params" }, params) : null);
  });

  const note = el("input", { maxlength: "200", value: draft.note, placeholder: "Why this mix? (optional)",
    disabled: !admin, oninput: () => { draft.note = note.value; update(); } });
  const discard = el("button", { disabled: !admin, onclick: () => { ctx.draft = draftFrom(state.saved); rerender(); } },
    "Discard changes");

  const head = el("div", { class: "card-head" }, el("h2", {}, icon("layers"), "Your mix"), dirty);
  const parts = [head, el("p", { class: "muted small" },
    "Choose strategies and the share of the account each one trades. What is left stays in T-bills. "
    + "The engine trades a saved mix from its next start."),
  bar, ...rows, total, problems,
  el("label", {}, "Note", note),
  el("div", { class: "buttons" }, saveButton, discard,
    el("a", { href: "#lab?mix=draft", class: "small" }, "Backtest this mix in the lab"))];
  if (!admin) parts.push(el("p", { class: "muted small" }, "Only an admin can change the mix."));
  if (state.pending) {
    parts.push(el("p", { class: "verdict" }, `Profile #${state.saved.id} is saved but the engine still trades `
      + `profile #${state.running?.id ?? "default"}.`,
    admin ? el("button", { class: "link", onclick: () => ctx.guard(() => offerRestart(ctx, state.saved)) },
      "Restart now") : null));
  }
  const node = el("section", { class: "card" }, parts);
  update();
  return node;
}

function strategyCard(ctx, s, { add }) {
  const inMix = ctx.draft.mix.some((e) => e.strategy === s.name);
  const shown = s.symbols.slice(0, 8);
  const lab = s.lab;
  return el("article", { class: "card strategy" },
    el("div", { class: "head" },
      el("div", {}, el("h3", {}, s.title), el("span", { class: "name" }, s.name)),
      chip(s.verdict, s.verdict)),
    el("p", {}, s.summary),
    el("div", { class: "chips" }, shown.map((symbol) => chip(symbol)),
      s.symbols.length > shown.length ? chip(`+${s.symbols.length - shown.length}`) : null,
      s.execution === "close" ? chip("closing auction") : null),
    lab ? el("p", { class: "small" }, chip(lab.passed ? "lab: passed" : "lab: rejected", lab.passed ? "passed" : "failed"),
      ` ${lab.verdict || ""} (${lab.data}, ${lab.start} to ${lab.end})`) : null,
    el("details", {}, el("summary", {}, "Why this verdict"),
      el("p", {}, s.verdict_detail),
      el("p", { class: "small" }, `Follows: ${s.source}`),
      el("a", { href: s.doc, target: "_blank", rel: "noopener noreferrer", class: "small" }, "Read the evidence")),
    el("div", { class: "buttons" },
      el("button", { disabled: inMix || !ctx.admin, onclick: () => ctx.guard(add) },
        icon("plus"), inMix ? " In the mix" : " Add to the mix"),
      el("a", { href: `#lab?strategy=${encodeURIComponent(s.name)}`, class: "small" }, "Validate in the lab")));
}

// -- the Risk page ---------------------------------------------------------------------------------

const percent = (v) => +(Number(v) * 100).toFixed(3);
const fraction = (p) => String(+(Number(p) / 100).toFixed(6));

function shown(row, value) {
  if (value === null || value === undefined) return "no limit";
  if (row.kind === "fraction") return Number(value) === Number(row.off ?? NaN) ? "off" : pct(value, 2);
  if (row.kind === "boolean") return value ? "yes" : "no";
  if (row.kind === "dollars") return `$${Number(value).toLocaleString()}`;
  return String(value);
}

export const riskView = {
  async show(ctx, root) {
    const state = await ctx.loadProfile();
    const overrides = { ...state.saved.risk };
    const admin = ctx.admin;
    const saveButton = el("button", { class: "primary", disabled: true }, "Save the limits");
    const dirty = () => JSON.stringify(overrides) !== JSON.stringify(state.saved.risk);
    const refresh = () => { saveButton.disabled = !admin || !dirty(); };
    const list = el("div");

    const renderRows = () => list.replaceChildren(...state.risk.map((row) => {
      const set = row.name in overrides;
      const value = set ? overrides[row.name] : row.environment;
      const change = (next) => {
        overrides[row.name] = next;
        refresh();
        renderRows();
      };
      const controls = [];
      if (row.kind === "boolean") {
        controls.push(el("input", { type: "checkbox", class: "switch", checked: value === true, disabled: !admin,
          "aria-label": row.label, onchange: (e) => change(e.target.checked) }));
      } else {
        const isOff = row.off !== null && value !== null && Number(value) === Number(row.off);
        const noLimit = row.optional && value === null;
        const scale = row.kind === "fraction" ? percent : Number;
        const input = el("input", {
          type: "number", value: isOff || noLimit ? "" : String(scale(value)), disabled: !admin || isOff || noLimit,
          min: String(scale(row.minimum)), max: String(scale(row.maximum)),
          step: row.kind === "fraction" ? "0.1" : row.kind === "dollars" ? "100" : "1", "aria-label": row.label,
        });
        input.addEventListener("change", () => {
          if (input.value === "") return;
          const number = Number(input.value);
          change(row.kind === "fraction" ? fraction(number) : row.kind === "integer" ? Math.round(number)
            : String(number));
        });
        controls.push(input, el("span", { class: "unit" }, row.kind === "fraction" ? "%" : row.kind === "dollars" ? "dollars" : ""));
        if (row.off !== null) {
          controls.push(el("label", { class: "check small" }, el("input", { type: "checkbox", class: "switch",
            checked: !isOff, disabled: !admin, onchange: (e) => change(e.target.checked
              ? String(row.environment !== null && Number(row.environment) !== Number(row.off) ? row.environment
                : row.minimum) : String(row.off)) }), "On"));
        }
        if (row.optional) {
          controls.push(el("label", { class: "check small" }, el("input", { type: "checkbox", checked: noLimit,
            disabled: !admin, onchange: (e) => change(e.target.checked ? null
              : String(row.environment ?? row.minimum)) }), "No limit"));
        }
      }
      return el("div", { class: "setting wide-layout" },
        el("div", { class: "row" }, el("span", { class: "label" }, row.label),
          chip(set ? "set here" : "from .env", set ? "done" : "")),
        el("div", { class: "controls" }, controls,
          set && admin ? el("button", { class: "link small", onclick: () => {
            delete overrides[row.name];
            refresh();
            renderRows();
          } }, "Use .env") : null),
        el("p", { class: "faint small help" }, row.kind === "boolean" ? `.env: ${shown(row, row.environment)}`
          : `Between ${shown(row, row.minimum)} and ${shown(row, row.maximum)}`
            + `${row.off !== null ? ", or off" : ""}. .env: ${shown(row, row.environment)}.`));
    }));

    saveButton.addEventListener("click", () => ctx.guard(async () => {
      const result = await api("PUT", "/api/v1/trading/profile",
        { mix: state.saved.mix, risk: overrides, note: state.saved.note });
      if (!result.ok) {
        notify(result.error, true);
        return;
      }
      notify(`Saved as profile #${result.data.profile.id}. The engine applies the limits from its next start.`);
      if (ctx.draft && !changed(ctx.draft, state.saved)) ctx.draft = null;
      await riskView.show(ctx, root);
      if (result.data.restart_needed) await offerRestart(ctx, result.data.profile);
    }));

    renderRows();
    const pending = ctx.draft && changed(ctx.draft, state.saved);
    root.replaceChildren(card("Risk limits", "shield",
      el("p", { class: "muted small" }, "The hard limits the engine trades within. Each stays inside fixed bounds; "
        + "what you do not set here comes from .env. Saved limits apply from the engine's next start."),
      pending ? el("p", { class: "verdict" }, "Your mix has unsaved changes on the Strategies page. Saving here keeps "
        + `the saved mix (${state.saved.mix.map((e) => `${weight(e.weight)} ${ctx.title(e.strategy)}`).join(", ")}).`)
        : null,
      list,
      el("div", { class: "buttons" }, saveButton),
      admin ? null : el("p", { class: "muted small" }, "Only an admin can change the limits.")));
    refresh();
  },
};
