// Drives the dashboard in Chromium against tools/dashboard/serve_sim.py (started on a
// fresh database): the phone first, through every page, then a desktop. Fails on any
// page error, console error or CSP violation.
//   node tools/dashboard/e2e.cjs <screenshot dir> [base URL]
// Set PW_MODULE to Playwright's path if it is not resolvable (e.g. "$(npm root -g)/playwright").
const { chromium } = require(process.env.PW_MODULE || "playwright");
const out = process.argv[2];
const base = process.argv[3] || "http://127.0.0.1:5055";

(async () => {
  const browser = await chromium.launch();
  const problems = [];
  let signedIn = false;
  const watch = (page, label) => {
    // The deliberate wrong password answers 401; any other failed request or error is a problem
    page.on("console", (m) => {
      const text = m.text();
      if (/status of 401/.test(text) && !signedIn) return;
      if (m.type() === "error" || /Refused|Content Security/.test(text)) problems.push(`${label} console: ${text}`);
    });
    page.on("pageerror", (e) => problems.push(`${label} pageerror: ${e.message}`));
  };
  const step = (name) => console.log(`ok - ${name}`);
  const badge = (page, text, timeout = 30000) => page.waitForFunction(
    (t) => document.querySelector("#engine-badge").textContent === t, text, { timeout });
  const notice = async (page, pattern) => {
    await page.waitForFunction((p) => {
      const n = document.querySelector("#notice");
      return !n.hidden && new RegExp(p).test(n.textContent);
    }, pattern, { timeout: 30000 });
    return page.textContent("#notice");
  };
  const confirm = async (page, title) => {
    await page.waitForSelector("dialog[open]");
    if (title && !(await page.textContent("#confirm-title")).includes(title)) {
      problems.push(`dialog "${await page.textContent("#confirm-title")}" instead of "${title}"`);
    }
    await page.click("#confirm-ok");
  };
  const go = async (page, view) => {
    await page.click(`#tabs a[data-view="${view}"]`);
    await page.waitForSelector(`#view-${view}:not([hidden])`);
  };

  const context = await browser.newContext({ viewport: { width: 412, height: 915 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const page = await context.newPage();
  watch(page, "phone");

  // -- signing in ------------------------------------------------------------------------
  await page.goto(base);
  await page.waitForSelector("#login:not([hidden])");
  await page.screenshot({ path: `${out}/0-sign-in.png` });
  step("sign-in screen shown");
  await page.fill("input[name=username]", "ops");
  await page.fill("input[name=password]", "wrong-password");
  await page.click("#login-form button");
  await page.waitForSelector("#login-error:not([hidden])");
  step(`bad password rejected: "${await page.textContent("#login-error")}"`);

  await page.fill("input[name=password]", "Harness-Passw0rd!");
  await page.click("#login-form button");
  await page.waitForSelector("#app:not([hidden])");
  signedIn = true;
  await badge(page, "stopped");
  step(`signed in; engine stopped; greeting "${await page.textContent("#greeting strong")}"`);
  await page.screenshot({ path: `${out}/1-stopped.png`, fullPage: true });

  // -- the engine ------------------------------------------------------------------------
  await page.click('[data-action="start"]');
  await confirm(page, "Start the engine?");
  step(`start confirmed: "${await notice(page, ".")}"`);
  await badge(page, "running");
  await page.waitForFunction(() => document.querySelectorAll("#positions tbody tr").length >= 2
    && document.querySelector("#equity-chart svg"), null, { timeout: 90000 });
  step(`running: ${(await page.$$("#positions tbody tr")).length} positions; equity chart drawn`);
  await page.screenshot({ path: `${out}/2-running.png`, fullPage: true });

  // -- the mix ---------------------------------------------------------------------------
  await go(page, "strategies");
  const trend = page.locator("article.strategy", { hasText: "Trend filter on the 60/40" });
  await trend.getByRole("button", { name: /Add to the mix/ }).click();
  await page.waitForSelector("dialog[open]");
  if (!(await page.isDisabled("#confirm-ok"))) problems.push("a rejected strategy was addable without reading why");
  await page.check("dialog input[type=checkbox]");
  await page.click("#confirm-ok");
  await page.waitForSelector('.mix-row:has-text("Trend filter on the 60/40")');
  const share = (title) => page.locator(".mix-row", { hasText: title }).locator("input[type=number]");
  await share("60/40 stocks and bonds").fill("50");
  await share("60/40 stocks and bonds").press("Tab");
  await share("Trend filter on the 60/40").fill("30");
  await share("Trend filter on the 60/40").press("Tab");
  const total = await page.textContent("#view-strategies .total");
  if (!/Strategies 80%/.test(total) || !/T-bills \(BIL\) 20%/.test(total)) problems.push(`totals read "${total}"`);
  await page.fill('#view-strategies input[placeholder^="Why this mix"]', "half 60/40, some trend, the rest in T-bills");
  await page.screenshot({ path: `${out}/3-mix.png`, fullPage: true });
  await page.getByRole("button", { name: "Save the mix" }).click();
  step(`mix saved: "${await notice(page, "Saved as profile")}"`);
  await confirm(page, "Restart the engine now?");
  await notice(page, "Engine restarting");
  await badge(page, "running", 60000);
  await go(page, "overview");
  await page.waitForFunction(() => /Trend filter on the 60\/40/.test(document.querySelector("#mix").textContent)
    && /Trading profile #1/.test(document.querySelector("#mix").textContent), null, { timeout: 30000 });
  step(`restarted with the new mix: "${(await page.textContent("#mix .legend")).replace(/\s+/g, " ")}"`);

  // -- the lab -----------------------------------------------------------------------------
  await go(page, "lab");
  await page.click("#view-lab button.primary");
  step(`backtest queued: "${await notice(page, "Queued run")}"`);
  await page.waitForSelector("#view-lab .verdict", { timeout: 120000 });
  await page.waitForSelector("#view-lab .bars svg");
  step(`backtest done: "${await page.textContent("#view-lab .verdict")}"`);
  await page.screenshot({ path: `${out}/4-lab-backtest.png`, fullPage: true });

  await page.click("#view-lab >> text=Validate a strategy");
  await page.selectOption("#view-lab select[name=strategy]", "benchmark");
  await page.fill("#view-lab input[name=start]", "2016-01-04");
  await page.fill("#view-lab input[name=end]", "2021-01-01");
  await page.click("#view-lab button.primary");
  await notice(page, "Queued run #2");
  await page.waitForSelector("#view-lab .verdict.passed, #view-lab .verdict.rejected", { timeout: 300000 });
  const gates = await page.$$("#view-lab table tbody tr");
  step(`validation done: "${await page.textContent("#view-lab .verdict")}" (${gates.length} gates)`);
  await page.screenshot({ path: `${out}/5-lab-validation.png`, fullPage: true });

  // -- risk limits --------------------------------------------------------------------------
  await go(page, "risk");
  const kill = page.locator(".setting", { hasText: "Kill switch: halt at this drop" }).locator("input[type=number]");
  await kill.fill("12");
  await kill.press("Tab");
  await page.waitForSelector('.setting:has-text("Kill switch") .chip.done');
  await page.getByRole("button", { name: "Save the limits" }).click();
  step(`limits saved: "${await notice(page, "Saved as profile #2")}"`);
  await page.waitForSelector("dialog[open]");
  await page.click("dialog button[value=cancel]");
  await page.screenshot({ path: `${out}/6-risk.png`, fullPage: true });

  // -- alerts ----------------------------------------------------------------------------------
  await go(page, "alerts");
  if (!(await page.isDisabled('.setting:has-text("Safety") input.switch'))) problems.push("safety alerts can be turned off");
  await page.locator(".setting", { hasText: "Every order the engine filled" }).locator("input.switch").check();
  await page.getByRole("button", { name: "Save", exact: true }).click();
  step(`alert choices saved: "${await notice(page, "Saved")}"`);
  await page.screenshot({ path: `${out}/7-alerts.png`, fullPage: true });

  // -- you -------------------------------------------------------------------------------------
  await go(page, "settings");
  await page.getByLabel("Your name here").fill("Night Owl");
  await page.click('#view-settings >> text="Day"');
  await page.click("#view-settings .sw-teal");
  if ((await page.getAttribute("html", "data-theme")) !== "light") problems.push("the theme does not preview");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await notice(page, "Saved");
  await go(page, "overview");
  await page.waitForFunction(() => /Night Owl/.test(document.querySelector("#greeting").textContent));
  step(`preferences saved: "${await page.textContent("#greeting strong")}", day theme, teal`);
  await page.screenshot({ path: `${out}/8-day-teal.png`, fullPage: true });
  await go(page, "settings");
  await page.click('#view-settings >> text="Night"');
  await page.click("#view-settings .sw-violet");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await notice(page, "Saved");

  // -- the emergency stop ---------------------------------------------------------------------
  await go(page, "overview");
  await page.click('[data-action="halt"]');
  await page.waitForSelector("dialog[open]");
  if (!(await page.isDisabled("#confirm-ok"))) problems.push("emergency stop confirmable without choosing");
  await page.screenshot({ path: `${out}/9-emergency-dialog.png` });
  await page.fill("dialog input[name=reason]", "dashboard drill");
  await page.check('dialog input[value="keep"]');
  await page.click("#confirm-ok");
  await badge(page, "halted");
  step(`emergency stop: banner "${await page.textContent("#banner")}"`);

  await page.click('[data-action="resume"]');
  await confirm(page, "Lift the halt?");
  await badge(page, "running");
  step("resumed");

  await page.reload();
  await page.waitForSelector("#app:not([hidden])");
  await badge(page, "running");
  step("still signed in after a reload (refresh token rotated)");

  await page.click("#logout");
  await page.waitForSelector("#login:not([hidden])");
  await page.reload();
  await page.waitForSelector("#login:not([hidden])");
  step("signed out, and stays signed out after a reload");
  await context.close();

  // -- a desktop ---------------------------------------------------------------------------------
  const desktop = await browser.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  const wide = await desktop.newPage();
  watch(wide, "desktop");
  await wide.goto(base);
  await wide.fill("input[name=username]", "ops");
  await wide.fill("input[name=password]", "Harness-Passw0rd!");
  await wide.check("input[name=remember]");
  await wide.click("#login-form button");
  await wide.waitForSelector("#equity-chart svg", { timeout: 30000 });
  await wide.waitForSelector("#positions tbody tr");
  await wide.screenshot({ path: `${out}/10-desktop.png`, fullPage: true });
  await wide.click('#tabs a[data-view="strategies"]');
  await wide.waitForSelector("#view-strategies article.strategy");
  await wide.screenshot({ path: `${out}/11-desktop-strategies.png`, fullPage: true });
  await wide.click('#tabs a[data-view="lab"]');
  await wide.waitForSelector("#view-lab .verdict");
  await wide.screenshot({ path: `${out}/12-desktop-lab.png`, fullPage: true });
  step("desktop signed in with 'stay signed in'");

  await browser.close();
  if (problems.length) { console.log("PROBLEMS:\n" + problems.join("\n")); process.exit(1); }
  console.log("no console errors or CSP violations");
})().catch((e) => { console.error(e); process.exit(1); });
