// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Re-capture the README screenshots against a running daemon.
//
// This exists because the README image set goes stale every time the UI moves,
// and re-shooting by hand is both slow and inconsistent (different window
// sizes, different scroll offsets, whatever data happened to be on screen).
//
//   node web/scripts/capture-readme-shots.mjs
//   BASE=http://127.0.0.1:8765 LOCALE=zh node web/scripts/capture-readme-shots.mjs
//   ONLY=files,skills node web/scripts/capture-readme-shots.mjs
//
// Point BASE at a daemon serving demo-safe data — NOT the packaged app, whose
// real projects are private. See docs/screenshots/CAPTURE.md for the rig.
//
// WebKit is deliberate: the desktop app ships on pywebview -> WKWebView, so
// Chromium would render a browser the users never see.

import { webkit } from '@playwright/test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const REPO = path.resolve(HERE, '..', '..')

const BASE = process.env.BASE ?? 'http://127.0.0.1:8765'
// Separate daemon, empty data dir — see WIZARD_SHOTS.
const WIZARD_BASE = process.env.WIZARD_BASE ?? 'http://127.0.0.1:8766'
const LOCALE = process.env.LOCALE ?? 'en'
const ONLY = process.env.ONLY ? new Set(process.env.ONLY.split(',')) : null
const OUT =
  process.env.OUT ??
  path.join(REPO, 'docs', 'screenshots', LOCALE === 'en' ? '' : LOCALE)

const DESKTOP = { width: 1200, height: 800 }
const TALL = { width: 1200, height: 1250 } // setup wizard runs past 800px
const MOBILE = { width: 390, height: 844 }

// ---------------------------------------------------------------------------
// Labels
//
// Selectors that must match on-screen text are resolved through the app's own
// catalog rather than hardcoded here, so the zh pass keeps working when a
// string changes. Mirrors the runtime zh -> en -> key fallback.
// ---------------------------------------------------------------------------

function loadStrings() {
  const src = fs.readFileSync(
    path.join(REPO, 'web', 'src', 'i18n', 'strings.ts'),
    'utf8',
  )
  const table = {}
  const line = /^\s*"([^"]+)":\s*\{\s*en:\s*"((?:[^"\\]|\\.)*)"(?:\s*,\s*zh:\s*"((?:[^"\\]|\\.)*)")?/gm
  let m
  while ((m = line.exec(src))) {
    const unescape = (s) => s?.replace(/\\"/g, '"').replace(/\\\\/g, '\\')
    table[m[1]] = { en: unescape(m[2]), zh: unescape(m[3]) }
  }
  return table
}

const STRINGS = loadStrings()
const t = (key, vars) => {
  const entry = STRINGS[key]
  if (!entry) throw new Error(`capture: unknown i18n key ${key}`)
  const raw = (LOCALE === 'zh' ? entry.zh : null) ?? entry.en ?? key
  return vars
    ? raw.replace(/\{(\w+)\}/g, (all, name) => (name in vars ? String(vars[name]) : all))
    : raw
}

// ---------------------------------------------------------------------------
// Navigation helpers
//
// The SPA has no URL router — `route` is in-memory React state — so every
// surface has to be reached by clicking, exactly as a user would.
// ---------------------------------------------------------------------------

const settle = (page, ms = 900) => page.waitForTimeout(ms)

async function home(page) {
  await page.goto(BASE, { waitUntil: 'networkidle' })
  await settle(page, 1500)
}

async function openProject(page, name) {
  await page.locator('aside').getByText(name, { exact: true }).first().click()
  await settle(page)
}

async function openTab(page, key) {
  const label = t(`projectDetail.tab.${key}`)
  // Queue carries a count badge, so anchor to the start rather than exact text.
  await page
    .getByRole('button', { name: new RegExp(`^${label}`) })
    .first()
    .click()
  await settle(page)
}

async function openProjectSettings(page, section) {
  await page.getByRole('button', { name: t('settingsIcon.aria') }).first().click()
  await settle(page)
  if (section) await scrollToSection(page, section)
}

async function openGlobalSettings(page, section) {
  await page.locator('aside').getByText(t('sidebar.settings'), { exact: true }).click()
  await settle(page)
  if (section) await scrollToSection(page, section)
}

async function scrollToSection(page, section) {
  const target = page.locator(`[data-settings-section="${section}"]`)
  await target.waitFor({ state: 'visible', timeout: 10_000 })
  await target.evaluate((el) => el.scrollIntoView({ block: 'start' }))
  await settle(page, 600)
}

// With more than one trigger the strip collapses to an "N triggers" row that
// has to be opened before the individual triggers are clickable.
async function expandTriggerStrip(page) {
  const summary = page.getByRole('button', {
    name: new RegExp(t('trigger.summary.other', { n: '\\d+' })),
  })
  if (await summary.count()) {
    await summary.first().click()
    await settle(page, 600)
  }
}

// A daemon with no settings yet lands on "Custom / Self-Hosted", which expands
// the Base URL / SDK advanced fields. Picking a preset shows the short path a
// first-time user actually takes.
async function pickProviderPreset(page, name = 'Anthropic') {
  const chip = page.getByRole('button', { name, exact: true })
  if (await chip.count()) {
    await chip.first().click()
    await settle(page, 700)
  }
}

async function expandDir(page, name) {
  await page.getByText(name, { exact: true }).first().click()
  await settle(page, 600)
}

// Session titles are the user's own prompts, so they are data, not UI strings —
// the same fragment matches in both locales.
async function openSession(page, titleFragment) {
  const row = page
    .locator('[data-testid="session-list"]')
    .getByText(new RegExp(titleFragment, 'i'))
  await row.first().click()
  await settle(page, 2500)
}

async function openFile(page, name) {
  await page.getByText(name, { exact: true }).first().click()
  await settle(page)
}

// ---------------------------------------------------------------------------
// Shot registry
//
// `needsAgent` marks shots whose subject only exists while a real agent run is
// in flight (a genuinely-running queue item, a streaming reply). Those are
// captured in the live-run pass, not the static one.
// ---------------------------------------------------------------------------

const MARKETING = 'Orbital-marketing'

const SHOTS = {
  'files': async (page) => {
    await home(page)
    await openProject(page, MARKETING)
    await openTab(page, 'files')
    await expandDir(page, 'orbital')
  },

  'memory-context': async (page) => {
    await home(page)
    await openProject(page, MARKETING)
    await openTab(page, 'files')
    await expandDir(page, 'orbital')
    await openFile(page, 'CONTEXT.md')
  },

  'memory-decisions': async (page) => {
    await home(page)
    await openProject(page, MARKETING)
    await openTab(page, 'files')
    await expandDir(page, 'orbital')
    await openFile(page, 'DECISIONS.md')
  },

  'memory-lessons': async (page) => {
    await home(page)
    await openProject(page, MARKETING)
    await openTab(page, 'files')
    await expandDir(page, 'orbital')
    await openFile(page, 'LESSONS.md')
  },

  'skills': async (page) => {
    await home(page)
    await openProject(page, MARKETING)
    await openProjectSettings(page, 'skills')
  },

  'subagent-memories': async (page) => {
    await home(page)
    await openProject(page, MARKETING)
    await openProjectSettings(page, 'sub-agents')
  },

  'settings-autonomy-budget': async (page) => {
    await home(page)
    await openProject(page, MARKETING)
    await openProjectSettings(page, 'autonomy')
  },

  'settings-budget': async (page) => {
    await home(page)
    await openProject(page, MARKETING)
    await openProjectSettings(page, 'budget')
  },

  'project-setting': async (page) => {
    await home(page)
    await openProject(page, MARKETING)
    await openProjectSettings(page, 'project-goals')
  },

  'credential-store': async (page) => {
    await home(page)
    await openGlobalSettings(page, 'credentials')
  },

  'qr-code-lan-pairng': async (page) => {
    await home(page)
    await openGlobalSettings(page, 'phone-pairing')
  },

  'global-setting': async (page) => {
    await home(page)
    await openGlobalSettings(page, 'llm')
  },

  'workbench': async (page) => {
    await home(page)
    await page.locator('aside').getByText(t('workspace.workbench.nav'), { exact: true }).click()
    await page.locator('[data-testid="workbench-list"], [data-testid="workbench-empty"]')
      .first()
      .waitFor({ state: 'visible', timeout: 15_000 })
    await settle(page, 1200)
  },

  'calendar': async (page) => {
    await home(page)
    await page.locator('aside').getByText(t('workspace.calendar.nav'), { exact: true }).click()
    await page.locator(
      '[data-testid="calendar-week-grid"], [data-testid="calendar-agenda"], [data-testid="calendar-empty"], [data-testid="calendar-unavailable"]',
    )
      .first()
      .waitFor({ state: 'visible', timeout: 15_000 })
    // The current week only projects slots from today forward, so on a Sunday
    // it renders as one lonely block. Next week shows the full recurring set.
    await page.locator('[data-testid="calendar-next"]').click()
    await settle(page, 1200)
    // The grid opens at midnight; nothing is scheduled before dawn.
    await page.evaluate(() => {
      const scroller = document.querySelector('[data-testid="calendar-week-grid"]')
      const target = scroller?.scrollHeight ? scroller : scroller?.parentElement
      if (target) target.scrollTop = (target.scrollHeight * 7.5) / 24
    })
    await settle(page, 800)
  },

  'scheduled-trigger': async (page) => {
    await home(page)
    await openProject(page, MARKETING)
    // Files, not Chat: the trigger detail only covers the top of the pane, and
    // the chat behind it would publish real session titles and the handles the
    // agent was working with.
    await openTab(page, 'files')
    await expandTriggerStrip(page)
    // Schedule rows are labelled with their human cadence, not their name.
    await page.getByRole('button', { name: /Monday|周一/ }).first().click()
    await settle(page)
  },

  'file-watch-trigger': async (page) => {
    await home(page)
    await openProject(page, MARKETING)
    await openTab(page, 'files')
    await expandTriggerStrip(page)
    await page
      .getByRole('button', { name: new RegExp(t('trigger.watchingPrefix', { path: '' }).trim()) })
      .first()
      .click()
    await settle(page)
  },

  'new-project-setting': async (page) => {
    await home(page)
    await page.locator('aside').getByText(t('app.newProject'), { exact: true }).click()
    await settle(page)
  },

  // Both of these show finished work, so they come from real completed
  // sessions in the demo history rather than a live run.
  'quick-task': async (page) => {
    await home(page)
    await openProject(page, 'Quick Tasks')
    await openSession(page, 'hackernews update')
  },

  'delegation-claudecode': async (page) => {
    await home(page)
    await openProject(page, 'Quick Tasks')
    await openSession(page, 'ask claude code to check')
    // Chat opens pinned to the bottom, which is the tail of the sub-agent's
    // answer — the handoff itself has scrolled away. Bring the dispatch back
    // into frame, since that is the whole subject of this shot.
    await page
      .locator('[data-testid="sub-agent-activity"], [data-testid="agent_run"]')
      .first()
      .scrollIntoViewIfNeeded()
    await settle(page, 900)
  },

  'queue-paused': async (page) => {
    await home(page)
    await openProject(page, MARKETING)
    await openTab(page, 'queue')
  },
}

// Mobile shots reuse the same flows at phone width.
const MOBILE_SHOTS = {
  '4A-mobile-dashboard': async (page) => {
    await home(page)
  },
}

// The setup wizard only renders while `llm.api_key_set` is false, so it needs
// its own daemon with an empty data dir — the demo daemon is past setup by
// definition. Taller viewport because the wizard card runs past 800px.
const WIZARD_SHOTS = {
  'apikey-setup': async (page) => {
    await page.goto(WIZARD_BASE, { waitUntil: 'networkidle' })
    await settle(page, 2000)
    await pickProviderPreset(page)
  },

  'connect-accounts': async (page) => {
    await page.goto(WIZARD_BASE, { waitUntil: 'networkidle' })
    await settle(page, 2000)
    await pickProviderPreset(page)
    await page.locator('input[type="password"]').first().fill('sk-demo-capture-rig')
    await page.getByRole('button', { name: t('wizard.next') }).click()
    await page.locator('[data-testid="wizard-connectors-group"]').waitFor({
      state: 'visible',
      timeout: 20_000,
    })
    await settle(page, 1500)
  },
}

// ---------------------------------------------------------------------------

async function run() {
  fs.mkdirSync(OUT, { recursive: true })

  const browser = await webkit.launch()
  const context = await browser.newContext({
    viewport: DESKTOP,
    deviceScaleFactor: 2,
  })
  // The app persists the language choice per-device in localStorage; there is
  // no backend locale, so this is the only way to drive the zh pass.
  await context.addInitScript(
    (locale) => window.localStorage.setItem('orbital.locale', locale),
    LOCALE,
  )

  const page = await context.newPage()
  const failures = []

  for (const [name, shot] of Object.entries(SHOTS)) {
    if (ONLY && !ONLY.has(name)) continue
    if (typeof shot !== 'function') {
      console.log(`  skip  ${name} (${shot.needsAgent ? 'needs a live agent run' : 'not a flow'})`)
      continue
    }
    try {
      await shot(page)
      const file = path.join(OUT, `${name}.png`)
      await page.screenshot({ path: file })
      console.log(`  ok    ${name}`)
    } catch (err) {
      failures.push([name, String(err).split('\n')[0]])
      console.log(`  FAIL  ${name}: ${String(err).split('\n')[0]}`)
    }
  }

  await context.close()

  const mobileCtx = await browser.newContext({
    viewport: MOBILE,
    deviceScaleFactor: 3,
    isMobile: true,
    hasTouch: true,
  })
  await mobileCtx.addInitScript(
    (locale) => window.localStorage.setItem('orbital.locale', locale),
    LOCALE,
  )
  const mobilePage = await mobileCtx.newPage()
  for (const [name, shot] of Object.entries(MOBILE_SHOTS)) {
    if (ONLY && !ONLY.has(name)) continue
    try {
      await shot(mobilePage)
      await mobilePage.screenshot({ path: path.join(OUT, `${name}.png`) })
      console.log(`  ok    ${name} (mobile)`)
    } catch (err) {
      failures.push([name, String(err).split('\n')[0]])
      console.log(`  FAIL  ${name}: ${String(err).split('\n')[0]}`)
    }
  }

  await mobileCtx.close()

  const wizardCtx = await browser.newContext({
    viewport: TALL,
    deviceScaleFactor: 2,
  })
  await wizardCtx.addInitScript(
    (locale) => window.localStorage.setItem('orbital.locale', locale),
    LOCALE,
  )
  const wizardPage = await wizardCtx.newPage()
  // The wizard card is centred in a `min-h-screen` wrapper, so a plain viewport
  // shot surrounds it with dead space. Clip to the card plus a margin instead.
  const shootWizardCard = async (file) => {
    const card = wizardPage.locator('div.bg-card').first()
    const box = await card.boundingBox()
    const pad = 40
    await wizardPage.screenshot({
      path: file,
      clip: box
        ? {
            x: Math.max(0, box.x - pad),
            y: Math.max(0, box.y - pad),
            width: Math.min(TALL.width - Math.max(0, box.x - pad), box.width + pad * 2),
            height: Math.min(TALL.height - Math.max(0, box.y - pad), box.height + pad * 2),
          }
        : undefined,
    })
  }
  // `connect-accounts` sets a key to advance the wizard, and the capture
  // daemon holds it in memory for the rest of its life — so a second run
  // against the same daemon lands on the app, not the wizard. Fail loudly
  // rather than silently shooting the wrong screen.
  const wizardReady = await fetch(`${WIZARD_BASE}/api/v2/settings`)
    .then((r) => r.json())
    .then((d) => !d?.llm?.api_key_set)
    .catch(() => false)
  for (const [name, shot] of Object.entries(WIZARD_SHOTS)) {
    if (ONLY && !ONLY.has(name)) continue
    if (!wizardReady) {
      console.log(`  skip  ${name} — wizard daemon at ${WIZARD_BASE} is past setup; restart it`)
      failures.push([name, 'wizard daemon already has an API key set'])
      continue
    }
    try {
      await shot(wizardPage)
      await shootWizardCard(path.join(OUT, `${name}.png`))
      console.log(`  ok    ${name} (wizard)`)
    } catch (err) {
      failures.push([name, String(err).split('\n')[0]])
      console.log(`  FAIL  ${name}: ${String(err).split('\n')[0]}`)
    }
  }

  await browser.close()

  console.log(`\n${LOCALE} pass -> ${OUT}`)
  if (failures.length) {
    console.log(`${failures.length} failed:`)
    for (const [name, err] of failures) console.log(`  - ${name}: ${err}`)
    process.exitCode = 1
  }
}

await run()
