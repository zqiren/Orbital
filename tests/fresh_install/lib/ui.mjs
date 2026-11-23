// Orbital — fresh-install test: the UI a new user clicks through.
//
// Selectors come from the shipped SPA (web/src): data-testids where they
// exist, otherwise the English strings from web/src/i18n/strings.ts. The
// locale is pinned to English for the driven browser only.
import { waitFor } from './orbital.mjs';

const T = {
  dialogNewProject: 'New project',
  chooseFolder: 'Choose a folder',
  manualPath: 'Type a path and press Enter...',
  useThisFolder: 'Use this folder',
  createProject: 'Create project',
  projectNamePlaceholder: 'Copenhagen Trip',
  pinListbox: 'Choose who this chat talks to',
  settings: 'Settings',
  fetchingModels: 'Fetching available models...',
  modelPlaceholder: 'Select or type a model name',
  newProjectSidebar: '+ New Project',
};

const LONG = 120_000;

export async function openApp(page, port) {
  await page.addInitScript(() => {
    try {
      if (!localStorage.getItem('orbital.locale')) localStorage.setItem('orbital.locale', 'en');
    } catch {}
  });
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'domcontentloaded', timeout: LONG });
}

/**
 * First run, keyless. Two shapes, by build:
 *  - Welcome (batch 1123 on): ONE page; Start with no key lands on the
 *    Workbench at once. Returns `{ flow: 'welcome' }`.
 *  - the four-page setup wizard (every release before it; the upgrade run
 *    still starts on one): agents -> (projects) -> skip key -> accounts.
 *    Returns `{ flow: 'wizard' }`.
 * `shot(name)` (optional) screenshots the page before Start.
 */
export async function onboardingSkipKey(page, note, shot = async () => {}) {
  const welcome = page.getByTestId('welcome-page');
  const wizard = page.getByTestId('wizard-steps');
  await welcome.or(wizard).first().waitFor({ timeout: LONG });
  if (await welcome.isVisible().catch(() => false)) return welcomeSkipKey(page, note, shot);
  return wizardSkipKey(page, note);
}

/** The installed agents as the Welcome page shows them: `slug(state)`. */
async function welcomeAgents(page) {
  return page
    .locator('li[data-testid^="welcome-agent-"][data-state]')
    .evaluateAll((els) => els.map((e) => `${e.getAttribute('data-testid').replace('welcome-agent-', '')}(${e.getAttribute('data-state')})`));
}

async function welcomeSkipKey(page, note, shot) {
  note('Welcome page shown (one page, no wizard steps)');
  const agentsError = page.getByTestId('welcome-agents-error');
  const loading = page.getByTestId('welcome-agents-loading');
  await waitFor(
    async () => {
      if (await agentsError.isVisible()) return { fail: 'the Welcome page could not load the agent list (welcome-agents-error)' };
      return (await page.getByTestId('welcome-agent-list').isVisible()) || !(await loading.isVisible());
    },
    { timeoutMs: LONG, intervalMs: 1000, label: 'Welcome agents list to load' },
  );
  // Codex is signed in by the rig (codex login --with-api-key): it must be
  // READY here, which is what opens Start without a key.
  const start = page.getByTestId('welcome-start');
  await waitFor(async () => (await page.getByTestId('welcome-agent-ready-codex').isVisible()) && (await start.isEnabled()), {
    timeoutMs: LONG,
    intervalMs: 1000,
    label: 'Codex ready and Start enabled',
  }).catch(async (e) => {
    throw new Error(`${e.message}; Welcome lists: ${(await welcomeAgents(page)).join(', ') || '(none)'}`);
  });
  const agents = await welcomeAgents(page);
  note(`Welcome lists agents: ${agents.join(', ') || '(none)'}`);

  // Projects band: shown only when the agents' history has something. Like
  // the wizard run (which pressed "skip import"), import nothing, so the
  // only projects are the ones this scenario creates. The scan never blocks
  // Start; give it a moment to settle so a late line cannot import behind us.
  const band = page.getByTestId('welcome-band-projects');
  await page.getByTestId('welcome-projects-scanning').waitFor({ state: 'hidden', timeout: 60_000 }).catch(() => {});
  if (await band.isVisible().catch(() => false)) {
    const check = page.getByTestId('welcome-import-check');
    if (await check.isVisible().catch(() => false)) {
      note(`projects band: "${(await page.getByTestId('welcome-import-label').innerText()).trim()}"; unticking it`);
      for (let i = 0; i < 3 && (await check.getAttribute('aria-checked')) !== 'false'; i++) await check.click();
      if ((await check.getAttribute('aria-checked')) !== 'false') throw new Error('could not untick the import line');
    } else if (await page.getByTestId('welcome-projects-error').isVisible().catch(() => false)) {
      note('projects band: the history scan reported an error (shown with Try again); not importing');
    }
  } else {
    note('no projects band (nothing in the agents\' history to import)');
  }
  await shot('welcome');
  const label = (await start.innerText()).trim();
  await start.click();
  note(`pressed "${label}" with no key`);
  await page.getByTestId('welcome-page').waitFor({ state: 'detached', timeout: LONG });
  await page.getByTestId('workbench-page').waitFor({ timeout: LONG });
  note('landed on the Workbench');
  return { agents, flow: 'welcome' };
}

async function wizardSkipKey(page, note) {
  note('setup wizard shown');
  const cont = page.getByTestId('wizard-agents-continue');
  await cont.waitFor({ timeout: LONG });
  await waitFor(async () => cont.isEnabled(), { timeoutMs: LONG, intervalMs: 1000, label: 'agents list to load' });
  const agents = await page.locator('[data-testid^="wizard-agent-check-"]').evaluateAll((els) =>
    els.map((e) => `${e.getAttribute('data-testid').replace('wizard-agent-check-', '')}${e.checked ? '' : '(unchecked)'}`),
  );
  note(`wizard lists agents: ${agents.join(', ') || '(none)'}`);
  await cont.click();

  // Step 2 appears only when the agents' history has folders to import.
  const skipImport = page.getByTestId('import-skip');
  const skipKey = page.getByTestId('wizard-skip-key');
  await waitFor(async () => (await skipImport.isVisible()) || (await skipKey.isVisible()), {
    timeoutMs: LONG,
    intervalMs: 1000,
    label: 'projects or key step',
  });
  if (await skipImport.isVisible()) {
    note('projects (import) step shown; skipping it');
    await skipImport.click();
  }
  await skipKey.waitFor({ timeout: LONG });
  await skipKey.click();
  note('skipped the API key ("Skip for now — use my existing agents")');
  const getStarted = page.getByTestId('wizard-get-started');
  await getStarted.waitFor({ timeout: LONG });
  await getStarted.click();
  await page.getByTestId('wizard-steps').waitFor({ state: 'detached', timeout: LONG });
  note('wizard finished');
  return { agents, flow: 'wizard' };
}

/** Either first-run surface on screen (an upgraded user must see neither). */
export async function firstRunShown(page) {
  for (const id of ['welcome-page', 'wizard-steps']) {
    if (await page.getByTestId(id).isVisible().catch(() => false)) return id;
  }
  return null;
}

/** The New project dialog: name + an existing folder typed into the folder browser. */
export async function createProject(page, { name, folder }, note) {
  let dialog = page.getByRole('dialog', { name: T.dialogNewProject });
  if (!(await dialog.isVisible().catch(() => false))) {
    await walkTourIfShown(page, () => {}, 0);
    await revealSidebar(page);
    await page.getByRole('button', { name: T.newProjectSidebar }).click();
  }
  dialog = page.getByRole('dialog', { name: T.dialogNewProject });
  await dialog.waitFor({ timeout: LONG });
  await dialog.getByPlaceholder(T.projectNamePlaceholder).fill(name);
  await dialog.getByRole('button', { name: T.chooseFolder }).click();
  const manual = dialog.getByPlaceholder(T.manualPath);
  await manual.waitFor({ timeout: LONG });
  await manual.fill(folder);
  await manual.press('Enter');
  // Wait for the browser panel to have navigated before using the folder.
  await page.waitForTimeout(1500);
  await dialog.getByRole('button', { name: T.useThisFolder }).click();
  await dialog.getByTestId('create-project-folder-row').waitFor({ timeout: LONG });
  const create = dialog.getByRole('button', { name: T.createProject });
  await waitFor(async () => create.isEnabled(), { timeoutMs: 30_000, intervalMs: 500, label: 'Create project enabled' });
  await create.click();
  await dialog.waitFor({ state: 'detached', timeout: LONG });
  note(`project "${name}" created at ${folder}`);
}

/**
 * The first-journey tour a new user sees on their first project: walk it to
 * the end. It starts only once the installed-agents probe answers, so it can
 * appear well after the chat opens; `waitMs` 0 = only if already showing.
 */
export async function walkTourIfShown(page, note = () => {}, waitMs = 45_000) {
  const tour = page.getByTestId('tour-card');
  if (waitMs > 0) {
    try {
      await tour.waitFor({ timeout: waitMs });
    } catch {
      note(`no first-journey tour within ${waitMs / 1000}s`);
      return 0;
    }
  } else if (!(await tour.isVisible().catch(() => false))) {
    return 0;
  }
  let stops = 0;
  for (; stops < 20 && (await tour.isVisible().catch(() => false)); stops++) {
    await page.getByTestId('tour-next').click({ timeout: 15_000 }).catch(() => {});
    await page.waitForTimeout(700);
  }
  if (await tour.isVisible().catch(() => false)) {
    await page.getByTestId('tour-skip').click().catch(() => {});
  }
  note(`walked the first-journey tour (${stops} stops)`);
  return stops;
}

/** Pick who the chat talks to: a worker display name ("Codex") or "Orbital". */
export async function choosePin(page, displayName, note) {
  await walkTourIfShown(page, note, 0);
  const trigger = page.getByTestId('pin-target-select').locator('button[aria-haspopup="listbox"]');
  await trigger.waitFor({ timeout: LONG });
  await trigger.click();
  const list = page.getByRole('listbox', { name: T.pinListbox });
  await list.waitFor({ timeout: 30_000 });
  const option = list.getByRole('option', { name: new RegExp(`^${displayName}`) }).first();
  await option.waitFor({ timeout: 30_000 });
  const options = await list
    .getByRole('option')
    .evaluateAll((els) => els.map((e) => (e.getAttribute('aria-label') || e.innerText || '').split('\n')[0].trim()));
  note(`pin menu offers: ${options.join(' | ')}`);
  await option.click();
  await list.waitFor({ state: 'hidden', timeout: 30_000 }).catch(() => {});
}

export function composer(page) {
  return page.getByTestId('composer-input').locator('textarea');
}

/** Type and send. Waits (long) for an idle composer: a busy one shows Stop/Queue, not Send. */
export async function send(page, message) {
  const box = composer(page);
  await box.waitFor({ timeout: LONG });
  await walkTourIfShown(page, () => {}, 0);
  await page.getByRole('button', { name: 'Stop', exact: true }).waitFor({ state: 'hidden', timeout: 600_000 });
  await box.click();
  await box.fill(message);
  await page.getByRole('button', { name: 'Send', exact: true }).click({ timeout: 600_000 });
  // The message must land as a user bubble.
  await page.getByTestId('user-message').filter({ hasText: message.slice(0, 40) }).last().waitFor({ timeout: LONG });
}

/**
 * Inside a project the app sidebar is folded into a 20px edge strip; a user
 * clicks the strip ("Projects") to dock it open. No-op when already open.
 */
export async function revealSidebar(page) {
  const strip = page.locator('[data-testid="edge-strip-wrapper"][data-pinned="false"]');
  if (await strip.count()) {
    await strip.getByRole('button', { name: 'Projects', exact: true }).first().click();
    await page.locator('[data-testid="edge-strip-wrapper"][data-pinned="true"]').waitFor({ timeout: 15_000 }).catch(() => {});
    await page.waitForTimeout(400);
  }
}

export async function openGlobalSettings(page) {
  await walkTourIfShown(page, () => {}, 0);
  await revealSidebar(page);
  await page.getByRole('button', { name: T.settings, exact: true }).first().click();
  await page.getByTestId('global-settings-title').waitFor({ timeout: LONG });
}

export async function leaveGlobalSettings(page) {
  await page.getByTestId('global-settings-back-button').click();
  await page.getByTestId('global-settings-title').waitFor({ state: 'detached', timeout: LONG });
}

/**
 * Settings -> LLM Providers -> Add provider -> card, test, save. Two shapes:
 * the connect screen (batch 1123 on: provider tiles / More search, the key
 * tests itself, the model is picked after it passes) and, for older builds,
 * the form with a provider combobox and a Test button.
 */
export async function addProviderCard(page, { providerId, key, model, providerName = 'OpenCode Go' }, note) {
  const section = page.locator('[data-settings-section="llm"]');
  await section.scrollIntoViewIfNeeded().catch(() => {});
  const add = page.getByTestId('cards-add').or(page.getByTestId('card-list-add')).first();
  await add.click();
  const modal = page.getByTestId('card-modal');
  await modal.waitFor({ timeout: LONG });
  const panel = modal.getByTestId('connect-panel');
  const combo = modal.getByTestId('provider-combobox');
  await panel.or(combo).or(modal.locator('select')).first().waitFor({ timeout: LONG });
  if (await panel.isVisible().catch(() => false)) return connectAddCard(page, modal, { providerId, key, model, providerName }, note);
  return legacyAddCard(page, modal, { providerId, key, model }, note);
}

async function connectAddCard(page, modal, { providerId, key, model, providerName }, note) {
  // The provider: a logo tile, else More -> search -> pick.
  const tile = modal.getByTestId(`connect-tile-${providerId}`);
  if (await tile.isVisible().catch(() => false)) {
    await tile.click();
  } else {
    const more = modal.getByTestId('connect-more').or(modal.getByTestId('connect-more-narrow')).filter({ visible: true }).first();
    await more.click();
    await modal.getByTestId('connect-more-search').fill(providerName);
    await modal.getByTestId(`connect-pick-${providerId}`).click();
  }
  await modal.getByTestId(`connect-provider-${providerId}`).waitFor({ timeout: LONG });
  note(`provider chosen: ${providerId}`);
  // A provider that lists no models asks for one before the test.
  const typedModel = modal.getByTestId('connect-model-typed');
  if (await typedModel.isVisible().catch(() => false)) await typedModel.fill(model);
  const keyInput = modal.getByTestId('connect-key');
  await keyInput.fill(key);
  await keyInput.press('Enter'); // the test fires by itself (paste/blur/Enter); Enter is the deterministic one
  const ok = modal.getByTestId('connect-key-ok');
  const err = modal.getByTestId('connect-key-error');
  await waitFor(async () => (await ok.isVisible()) || (await err.isVisible()), {
    timeoutMs: 180_000,
    intervalMs: 1000,
    label: 'connection test result',
  });
  if (await err.isVisible()) throw new Error(`connection test failed: ${(await err.innerText()).slice(0, 300)}`);
  note('connection test passed');
  // The model row appears after the pass: a list (toggle) or a typed field.
  const toggle = modal.getByTestId('connect-model-toggle');
  const input = modal.getByTestId('connect-model-input');
  await toggle.or(input).first().waitFor({ timeout: 30_000 });
  if (await toggle.isVisible().catch(() => false)) {
    if ((await toggle.getAttribute('aria-expanded')) !== 'true') await toggle.click();
    const search = modal.getByTestId('connect-model-search');
    if (await search.isVisible().catch(() => false)) await search.fill(model);
    const option = modal.getByTestId(`connect-model-option-${model}`);
    const typed = modal.getByTestId('connect-model-use-typed');
    if (await option.isVisible().catch(() => false)) {
      await option.click();
    } else if (await typed.isVisible().catch(() => false)) {
      await typed.click();
    } else {
      const offered = await modal.locator('[data-testid^="connect-model-option-"]').evaluateAll((els) =>
        els.map((e) => e.getAttribute('data-testid').replace('connect-model-option-', '')));
      throw new Error(`model list has no ${model}: ${offered.join(', ')}`);
    }
  } else {
    await input.fill(model);
  }
  note(`model: ${model}`);
  const save = modal.getByTestId('card-save');
  await waitFor(async () => save.isEnabled(), { timeoutMs: 30_000, intervalMs: 500, label: 'Save enabled' });
  await save.click();
  const saveErr = modal.getByTestId('card-save-error');
  await waitFor(
    async () => {
      if (await saveErr.isVisible().catch(() => false)) return { fail: (await saveErr.innerText()).slice(0, 300) };
      // Save re-tests the key server-side; a refusal comes back on the key field.
      if (await err.isVisible().catch(() => false)) return { fail: `save refused: ${(await err.innerText()).slice(0, 300)}` };
      return !(await modal.isVisible().catch(() => false));
    },
    { timeoutMs: 180_000, intervalMs: 1000, label: 'card saved (modal closed)' },
  );
  note('card saved (modal closed on its own)');
}

async function legacyAddCard(page, modal, { providerId, key, model }, note) {
  const combo = modal.getByTestId('provider-combobox');
  const select = modal.locator('select').first();
  await combo.or(select).first().waitFor({ timeout: LONG });
  if (await combo.isVisible().catch(() => false)) {
    await combo.click();
    await combo.fill('OpenCode Go').catch(() => {});
    await page.getByRole('option', { name: /^OpenCode Go$/ }).first().click();
  } else {
    await select.selectOption(providerId);
  }
  await modal.locator('input[type="password"]').first().fill(key);
  // Typing the key triggers a live /models fetch that swaps the model input out.
  await page.waitForTimeout(800);
  await modal.getByText(T.fetchingModels).waitFor({ state: 'hidden', timeout: LONG }).catch(() => {});
  const modelInput = modal.locator('input[type="text"]').filter({ hasNot: page.locator('[disabled]') });
  const byPlaceholder = modal.getByPlaceholder(/Select or type a model name|Type model name/);
  const input = (await byPlaceholder.count()) ? byPlaceholder.first() : modelInput.last();
  await input.fill(model);
  await page.waitForTimeout(300);
  const test = modal.getByTestId('card-test');
  await test.waitFor({ timeout: 30_000 });
  await test.click();
  const ok = modal.locator('p.text-success');
  const err = modal.locator('p.text-error');
  await waitFor(async () => (await ok.isVisible()) || (await err.isVisible()), {
    timeoutMs: 180_000,
    intervalMs: 1000,
    label: 'connection test result',
  });
  if (await err.isVisible()) throw new Error(`connection test failed: ${(await err.innerText()).slice(0, 300)}`);
  note(`connection test: ${(await ok.innerText()).trim()}`);
  await modal.getByTestId('card-save').click();
  await modal.waitFor({ state: 'detached', timeout: 180_000 });
  note('card saved (modal closed on its own)');
}

function subAgentCard(page, slug) {
  return page
    .locator('[data-settings-section="sub-agents"] div.surface-float-static')
    .filter({ has: page.locator('code', { hasText: new RegExp(`^${slug}$`) }) })
    .first();
}

/** The Sub-agents list can take a long time to load (it probes every CLI). */
async function waitSubAgentCard(page, slug) {
  const card = subAgentCard(page, slug);
  await page.locator('[data-settings-section="sub-agents"]').scrollIntoViewIfNeeded().catch(() => {});
  await card.waitFor({ timeout: 240_000 });
  await card.scrollIntoViewIfNeeded();
  return card;
}

/** Settings -> Sub-agents -> <slug> credential field -> Save. */
export async function setSubAgentCredential(page, slug, key, value, note) {
  const card = await waitSubAgentCard(page, slug);
  const input = card.getByTestId(`sub-agent-cred-input-${key}`);
  await input.waitFor({ timeout: LONG });
  await input.fill(value);
  await card.getByTestId(`sub-agent-cred-save-${key}`).click();
  await card.getByTestId(`sub-agent-cred-remove-${key}`).waitFor({ timeout: LONG });
  note(`${slug}: ${key} stored`);
}

/** Settings -> Sub-agents -> <slug> -> Model -> Save. Handles select or free text. */
export async function setSubAgentModel(page, slug, model, note) {
  const card = await waitSubAgentCard(page, slug);
  const field = card
    .locator('div.flex.flex-col.gap-1')
    .filter({ has: page.locator('label', { hasText: /^Model$/ }) })
    .first();
  await field.waitFor({ timeout: LONG });
  const select = field.locator('select');
  if (await select.count()) {
    await waitFor(async () => select.isEnabled(), { timeoutMs: LONG, intervalMs: 1000, label: `${slug} model select enabled` });
    const values = await select.locator('option').evaluateAll((os) => os.map((o) => o.value));
    if (!values.includes(model)) throw new Error(`${slug} model list has no ${model}: ${values.join(', ')}`);
    await select.selectOption(model);
  } else {
    await field.locator('input[type="text"]').fill(model);
  }
  const save = card.locator('button:not([data-testid])', { hasText: /^Save$/ }).last();
  await save.click();
  const saved = card.getByText('Saved', { exact: true });
  const error = card.locator('p.text-warning-text');
  await waitFor(async () => (await saved.isVisible()) || (await error.isVisible()), {
    timeoutMs: 60_000,
    intervalMs: 500,
    label: `${slug} model save`,
  });
  if (await error.isVisible()) throw new Error(`${slug} model save failed: ${(await error.innerText()).slice(0, 300)}`);
  note(`${slug}: model set to ${model}`);
}

export async function newSession(page) {
  await page.getByTestId('session-new-button').click();
  await page.waitForTimeout(1000);
}

export async function openProject(page, name) {
  await revealSidebar(page);
  await page.getByRole('button', { name: new RegExp(`^${name.slice(0, 20)}`) }).first().click();
}

/** Count of chat bubbles carrying a worker's avatar. */
export async function workerBubbleCount(page, slug) {
  return page.locator(`[data-testid="message-avatar"][data-variant="agent"][data-agent-handle="${slug}"]`).count();
}

export async function chatText(page) {
  return page.locator('main').innerText().catch(() => page.locator('body').innerText());
}
