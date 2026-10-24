// The Alerts page (which alerts are sent, and where) and the You page (how the
// dashboard looks to you, and your account). Alert tokens and addresses stay in
// .env on the server: the dashboard never sees them.

import { api } from "./api.js";
import { card, chip, el, icon, notify } from "./ui.js";

const DOCS = "https://yog-sotho.github.io/Nocturna-Trading-System/";
const ENV = {
  telegram: "ALERT_TELEGRAM_BOT_TOKEN and ALERT_TELEGRAM_CHAT_ID",
  ntfy: "ALERT_NTFY_URL",
  webhook: "ALERT_WEBHOOK_URL",
};

// -- alerts ----------------------------------------------------------------------------------------

export const alertsView = {
  async show(ctx, root) {
    const loaded = await api("GET", "/api/v1/settings/alerts");
    if (!loaded.ok) throw new Error(loaded.error);
    const settings = loaded.data;
    const admin = ctx.admin;
    const channels = Object.fromEntries(settings.channels.map((c) => [c.name, c.enabled]));
    const categories = Object.fromEntries(settings.categories.map((c) => [c.name, c.enabled]));
    const save = el("button", { class: "primary", disabled: true }, "Save");
    const results = el("div");
    const changed = () => settings.channels.some((c) => channels[c.name] !== c.enabled)
      || settings.categories.some((c) => categories[c.name] !== c.enabled);
    const refresh = () => { save.disabled = !admin || !changed(); };

    const channelRows = settings.channels.map((c) => el("div", { class: "setting wide-layout" },
      el("div", { class: "row" },
        el("span", { class: "label" }, c.label),
        el("input", { type: "checkbox", class: "switch", checked: c.enabled, disabled: !admin || !c.configured,
          "aria-label": `Send alerts by ${c.label}`, onchange: (e) => { channels[c.name] = e.target.checked; refresh(); } })),
      el("p", { class: "faint small help" }, c.configured ? "Configured in .env."
        : `Not configured: set ${ENV[c.name]} in .env, then restart the containers.`)));

    const categoryRows = settings.categories.map((c) => el("div", { class: "setting wide-layout" },
      el("div", { class: "row" },
        el("span", { class: "label" }, c.label, c.locked ? " " : null, c.locked ? chip("always sent", "locked") : null),
        el("input", { type: "checkbox", class: "switch", checked: c.enabled, disabled: !admin || c.locked,
          "aria-label": `${c.label} alerts`, onchange: (e) => { categories[c.name] = e.target.checked; refresh(); } })),
      el("p", { class: "faint small help" }, c.description)));

    save.addEventListener("click", () => ctx.guard(async () => {
      const body = {
        channels: Object.fromEntries(settings.channels.filter((c) => c.configured).map((c) => [c.name, channels[c.name]])),
        categories: Object.fromEntries(settings.categories.filter((c) => !c.locked)
          .map((c) => [c.name, categories[c.name]])),
      };
      const result = await api("PUT", "/api/v1/settings/alerts", body);
      notify(result.ok ? "Saved: every process follows it within seconds." : result.error, !result.ok);
      if (result.ok) await alertsView.show(ctx, root);
    }));

    const test = el("button", { disabled: !admin || !settings.channels.some((c) => c.configured) },
      icon("send"), " Send a test alert");
    test.addEventListener("click", () => ctx.guard(async () => {
      test.disabled = true;
      try {
        const result = await api("POST", "/api/v1/settings/alerts/test");
        if (!result.ok) {
          notify(result.error, true);
          return;
        }
        results.replaceChildren(el("ul", { class: "legend" }, result.data.results.map((r) =>
          el("li", {}, chip(r.sent ? "sent" : "failed", r.sent ? "passed" : "failed"), el("span", {}, r.channel),
            r.error ? el("span", { class: "muted small" }, r.error) : null))));
      } finally {
        test.disabled = false;
      }
    }));

    root.replaceChildren(
      card("Where alerts go", "send",
        el("p", { class: "muted small" }, "Tokens and addresses live in .env on the server, never here. "
          + "At least one configured channel always stays on."),
        ...channelRows,
        el("div", { class: "buttons" }, test), results),
      card("What is sent", "bell",
        el("p", { class: "muted small" }, "Safety and problem alerts cannot be turned off."),
        ...categoryRows,
        el("div", { class: "buttons" }, save),
        admin ? null : el("p", { class: "muted small" }, "Only an admin can change the alerts.")));
  },
};

// -- you ---------------------------------------------------------------------------------------------

const CARD_NAMES = {
  account: "Account", equity: "Equity", mix: "The mix", positions: "Positions", risk: "Risk", orders: "Orders",
  fills: "Recent fills",
};
const ACCENTS = ["violet", "indigo", "magenta", "teal", "amber"];

export const settingsView = {
  async show(ctx, root) {
    const prefs = structuredClone(ctx.prefs);
    const saved = JSON.stringify(ctx.prefs);
    const save = el("button", { class: "primary", disabled: true }, "Save");
    const preview = () => {
      ctx.applyLook(prefs);
      ctx.applyCards(prefs);
      save.disabled = JSON.stringify(prefs) === saved;
    };

    const name = el("input", { maxlength: "40", value: prefs.display_name, placeholder: ctx.me?.username || "",
      oninput: () => { prefs.display_name = name.value; preview(); } });

    const theme = el("div", { class: "segmented", role: "radiogroup", "aria-label": "Theme" },
      [["dark", "Night"], ["light", "Day"], ["auto", "Like the device"]].map(([value, label]) =>
        el("label", {}, el("input", { type: "radio", name: "theme", value, checked: prefs.theme === value,
          onchange: () => { prefs.theme = value; preview(); } }), el("span", {}, label))));

    const accents = el("div", { class: "swatches", role: "radiogroup", "aria-label": "Accent colour" },
      ACCENTS.map((accent) => el("label", { title: accent },
        el("input", { type: "radio", name: "accent", value: accent, checked: prefs.accent === accent,
          "aria-label": accent, onchange: () => { prefs.accent = accent; preview(); } }),
        el("span", { class: `sw-${accent}` }))));

    const order = el("ul", { class: "order-list" });
    const renderOrder = () => order.replaceChildren(...prefs.cards.map((cardName, i) => el("li", {},
      el("input", { type: "checkbox", checked: !prefs.hidden.includes(cardName), "aria-label": `Show ${CARD_NAMES[cardName]}`,
        onchange: (e) => {
          prefs.hidden = e.target.checked ? prefs.hidden.filter((h) => h !== cardName) : [...prefs.hidden, cardName];
          preview();
        } }),
      el("span", { class: "grow" }, CARD_NAMES[cardName] || cardName),
      el("button", { class: "icon-only", disabled: i === 0, "aria-label": `Move ${CARD_NAMES[cardName]} up`,
        onclick: () => { move(i, -1); } }, icon("up")),
      el("button", { class: "icon-only", disabled: i === prefs.cards.length - 1,
        "aria-label": `Move ${CARD_NAMES[cardName]} down`, onclick: () => { move(i, 1); } }, icon("down")))));
    const move = (i, by) => {
      const [moved] = prefs.cards.splice(i, 1);
      prefs.cards.splice(i + by, 0, moved);
      renderOrder();
      preview();
    };
    renderOrder();

    save.addEventListener("click", () => ctx.guard(async () => {
      const result = await api("PUT", "/api/v1/me/preferences", prefs);
      if (!result.ok) {
        notify(result.error, true);
        return;
      }
      ctx.prefs = result.data;
      ctx.greet();
      notify("Saved.");
      await settingsView.show(ctx, root);
    }));

    const version = await fetch("/api/version").then((r) => r.json()).catch(() => ({}));
    const me = ctx.me || {};
    root.replaceChildren(
      card("How the dashboard looks to you", "sliders",
        el("label", {}, "Your name here", name),
        el("p", { class: "faint small" }, "The greeting uses it. Empty: your username."),
        el("p", { class: "label" }, "Theme"), theme,
        el("p", { class: "label" }, "Accent"), accents,
        el("p", { class: "label" }, "Overview cards"),
        el("p", { class: "faint small" }, "Order them, and untick the ones you do not want."),
        order,
        el("div", { class: "buttons" }, save,
          el("button", { onclick: () => {
            ctx.applyLook(ctx.prefs);
            ctx.applyCards(ctx.prefs);
            ctx.guard(() => settingsView.show(ctx, root));
          } }, "Undo"))),
      card("Your account", "wallet",
        el("p", {}, el("strong", {}, me.username || ""), " · ", me.email || ""),
        el("div", { class: "chips" }, (me.roles || []).map((r) => chip(r)), me.is_admin ? chip("admin", "done") : null),
        el("p", { class: "muted small" }, "Paper trading only: Nocturna never touches real money."),
        el("div", { class: "buttons" },
          el("a", { href: DOCS, target: "_blank", rel: "noopener noreferrer" }, "Documentation"),
          el("button", { onclick: () => ctx.logout() }, "Sign out")),
        el("p", { class: "faint small" }, `Nocturna ${version.version || ""}, API ${version.api_version || ""}`)));
  },
};
