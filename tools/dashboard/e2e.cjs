// Drives the dashboard in Chromium at phone size against tools/dashboard/serve_sim.py
// (started on a fresh database), and fails on any page error or CSP violation.
//   node tools/dashboard/e2e.cjs <screenshot dir> [base URL]
// Set PW_MODULE to Playwright's path if it is not resolvable (e.g. "$(npm root -g)/playwright").
const { chromium } = require(process.env.PW_MODULE || "playwright");
const out = process.argv[2];
const base = process.argv[3] || "http://127.0.0.1:5055";

(async () => {
  const browser = await chromium.launch();
  const problems = [];
  let signedIn = false;
  const context = await browser.newContext({ viewport: { width: 412, height: 915 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  const page = await context.newPage();
  // The deliberate wrong password answers 401; any other failed request or error is a problem
  page.on("console", (m) => {
    const text = m.text();
    if (/status of 401/.test(text) && !signedIn) return;
    if (m.type() === "error" || /Refused|Content Security/.test(text)) problems.push(`console: ${text}`);
  });
  page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  const step = (name) => console.log(`ok - ${name}`);

  await page.goto(base);
  await page.waitForSelector("#login:not([hidden])");
  step("login screen shown");
  await page.fill("input[name=username]", "ops");
  await page.fill("input[name=password]", "wrong-password");
  await page.click("#login-form button");
  await page.waitForSelector("#login-error:not([hidden])");
  step(`bad password rejected: "${await page.textContent("#login-error")}"`);

  await page.fill("input[name=password]", "Harness-Passw0rd!");
  await page.click("#login-form button");
  await page.waitForSelector("#app:not([hidden])");
  signedIn = true;
  await page.waitForFunction(() => document.querySelector("#engine-badge").textContent === "stopped");
  step("signed in; engine stopped");
  await page.screenshot({ path: `${out}/1-stopped.png`, fullPage: true });

  await page.click('[data-action="start"]');
  await page.waitForSelector("dialog[open]");
  await page.click("#confirm-ok");
  await page.waitForSelector("#notice:not([hidden])");
  step(`start confirmed: "${await page.textContent("#notice")}"`);
  await page.waitForFunction(() => document.querySelector("#engine-badge").textContent === "running", null, { timeout: 30000 });
  step("engine running");

  await page.waitForFunction(() => document.querySelectorAll("#positions tbody tr").length >= 2
    && document.querySelector("#equity-chart svg"), null, { timeout: 90000 });
  step(`positions: ${(await page.$$("#positions tbody tr")).length} rows; equity chart drawn`);
  step(`fills rows: ${(await page.$$("#fills tbody tr")).length}`);
  await page.screenshot({ path: `${out}/2-running.png`, fullPage: true });

  await page.click('[data-action="halt"]');
  await page.waitForSelector("dialog[open]");
  if (!(await page.isDisabled("#confirm-ok"))) problems.push("emergency stop confirmable without choosing");
  await page.screenshot({ path: `${out}/3-emergency-dialog.png` });
  await page.fill("dialog input[name=reason]", "dashboard drill");
  await page.check('dialog input[value="keep"]');
  await page.click("#confirm-ok");
  await page.waitForFunction(() => document.querySelector("#engine-badge").textContent === "halted", null, { timeout: 30000 });
  step(`emergency stop: banner "${await page.textContent("#banner")}"`);
  await page.screenshot({ path: `${out}/4-halted.png`, fullPage: true });

  await page.click('[data-action="resume"]');
  await page.click("#confirm-ok");
  await page.waitForFunction(() => document.querySelector("#engine-badge").textContent === "running", null, { timeout: 30000 });
  step("resumed");

  await page.reload();
  await page.waitForSelector("#app:not([hidden])");
  await page.waitForFunction(() => document.querySelector("#engine-badge").textContent === "running");
  step("still signed in after a reload (refresh token rotated)");

  await page.click("#logout");
  await page.waitForSelector("#login:not([hidden])");
  await page.reload();
  await page.waitForSelector("#login:not([hidden])");
  step("signed out, and stays signed out after a reload");

  const dark = await browser.newContext({ viewport: { width: 412, height: 915 }, deviceScaleFactor: 2, colorScheme: "dark" });
  const darkPage = await dark.newPage();
  darkPage.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  await darkPage.goto(base);
  await darkPage.fill("input[name=username]", "ops");
  await darkPage.fill("input[name=password]", "Harness-Passw0rd!");
  await darkPage.check("input[name=remember]");
  await darkPage.click("#login-form button");
  await darkPage.waitForSelector("#equity-chart svg", { timeout: 30000 });
  await darkPage.screenshot({ path: `${out}/5-dark.png`, fullPage: true });
  step("dark mode signed in with 'stay signed in'");

  await browser.close();
  if (problems.length) { console.log("PROBLEMS:\n" + problems.join("\n")); process.exit(1); }
  console.log("no console errors or CSP violations");
})().catch((e) => { console.error(e); process.exit(1); });
