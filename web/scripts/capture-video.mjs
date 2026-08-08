// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Record demo video of the real UI against a running daemon.
//
// The sibling rig `capture-readme-shots.mjs` freezes single frames; this one
// rolls tape. Same browser, same daemon, same locale plumbing — the difference
// is `recordVideo` on the context and an ffmpeg pass at the end, which is
// exactly what docs/screenshots/CAPTURE.md §7 sketched out.
//
//   node web/scripts/capture-video.mjs
//   BASE=http://127.0.0.1:8765 OUT=brand-video/clips node web/scripts/capture-video.mjs
//   ONLY=hello LOCALE=zh node web/scripts/capture-video.mjs
//
// Point BASE at a daemon serving demo-safe data — NOT the packaged app, whose
// real projects are private.
//
// WebKit is deliberate: the desktop app ships on pywebview -> WKWebView, so
// Chromium would render a browser the users never see. It also means the WebKit
// compositor quirks the app is tuned for show up on tape the way users see them.
//
// Silent by design — no audio track, no burned-in captions. These are clean
// plates; voiceover and titles happen in the edit.

import { webkit } from '@playwright/test'
import { spawn } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const REPO = path.resolve(HERE, '..', '..')

export const BASE = process.env.BASE ?? 'http://127.0.0.1:8765'
export const LOCALE = process.env.LOCALE ?? 'en'
const ONLY = process.env.ONLY ? new Set(process.env.ONLY.split(',')) : null
const OUT = path.resolve(REPO, process.env.OUT ?? path.join('brand-video', 'clips'))
const RAW = path.join(OUT, 'raw')

// Record at 1600x900 — the layout width the UI reads best at — and let ffmpeg
// carry it to 1080p. Recording at a native 1920 viewport instead would lay the
// app out at 1920 CSS px, shrinking every control relative to the frame; this
// way the 1.2x upscale acts as a legibility zoom. 16:9 both ends, so the pad
// filter is a no-op guard rather than a letterbox.
const CAPTURE = { width: 1600, height: 900 }
const DELIVER = { width: 1920, height: 1080 }
const FPS = Number(process.env.FPS ?? 30)
const CRF = process.env.CRF ?? '18'
// The .webm is the negative. Re-transcoding is free; re-shooting a live UI is
// not, and never lands the same twice. Keep it unless told otherwise.
const KEEP_WEBM = process.env.KEEP_WEBM !== '0'

// ---------------------------------------------------------------------------
// Labels
//
// Same trick as the screenshot rig: selectors that must match on-screen text
// resolve through the app's own catalog, so a zh pass keeps working when a
// string changes. Mirrors the runtime zh -> en -> key fallback.
// ---------------------------------------------------------------------------

function loadStrings() {
  const src = fs.readFileSync(path.join(REPO, 'web', 'src', 'i18n', 'strings.ts'), 'utf8')
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

export const t = (key, vars) => {
  const entry = STRINGS[key]
  if (!entry) throw new Error(`capture-video: unknown i18n key ${key}`)
  const raw = (LOCALE === 'zh' ? entry.zh : null) ?? entry.en ?? key
  return vars
    ? raw.replace(/\{(\w+)\}/g, (all, name) => (name in vars ? String(vars[name]) : all))
    : raw
}

// ---------------------------------------------------------------------------
// Timing helpers
//
// Video needs slack that screenshots do not. A shot only has to be correct at
// the instant of capture; a clip has to stay watchable across the whole beat,
// which means dwelling long enough to read and never cutting on a half-painted
// frame.
// ---------------------------------------------------------------------------

/** Dwell on the current frame. This is the viewer's reading time — spend it. */
export const pause = (page, ms = 1500) => page.waitForTimeout(ms)

/** Short wait for a transition to finish. Not reading time — settling time. */
export const settle = (page, ms = 700) => page.waitForTimeout(ms)

/**
 * Approach a target before acting on it: scroll it into frame, hover, let the
 * hover state paint, then hand the locator back. Playwright draws no cursor, so
 * the hover state IS the only visible "the pointer is here" cue on tape.
 */
export async function approach(page, target, ms = 500) {
  const locator = typeof target === 'string' ? page.locator(target) : target
  const el = locator.first()
  await el.scrollIntoViewIfNeeded()
  await el.hover()
  await settle(page, ms)
  return el
}

/** Hover, dwell, click, then let the resulting view settle. */
export async function click(page, target, { before = 500, after = 900 } = {}) {
  const el = await approach(page, target, before)
  await el.click()
  await settle(page, after)
  return el
}

/** Type at a human cadence rather than pasting a string into existence. */
export async function type(page, target, text, { delay = 55, after = 400 } = {}) {
  const el = await approach(page, target, 300)
  await el.click()
  await el.type(text, { delay })
  await settle(page, after)
  return el
}

/** Land on the app root and wait out the first paint. */
export async function home(page) {
  await page.goto(BASE, { waitUntil: 'networkidle' })
  await settle(page, 1500)
}

// ---------------------------------------------------------------------------
// Scene registry
//
// One scene = one labelled clip. Register with `scene()`, then `run()` gives
// each its own context (fresh localStorage, fresh recording) and transcodes.
// ---------------------------------------------------------------------------

const SCENES = []

/**
 * Register a clip.
 *
 * @param {string} name       output basename — `<OUT>/<name>.mp4`
 * @param {(page) => Promise<void>} fn  the beat to film
 * @param {{viewport?: {width:number,height:number}}} [opts]
 */
export function scene(name, fn, opts = {}) {
  SCENES.push({ name, fn, opts })
}

// ---------------------------------------------------------------------------
// Transcode
// ---------------------------------------------------------------------------

function ffmpeg(args) {
  return new Promise((resolve, reject) => {
    const proc = spawn('ffmpeg', args, { stdio: ['ignore', 'ignore', 'pipe'] })
    let err = ''
    proc.stderr.on('data', (d) => (err += d))
    proc.on('error', (e) =>
      reject(new Error(e.code === 'ENOENT' ? 'ffmpeg not found on PATH' : String(e))),
    )
    proc.on('close', (code) =>
      code === 0 ? resolve() : reject(new Error(`ffmpeg exited ${code}\n${err.slice(-2000)}`)),
    )
  })
}

// Playwright writes VP8 .webm; nothing outside a browser plays that reliably.
// H.264 + yuv420p is the combination QuickTime, Premiere, Keynote and every
// social uploader all accept without re-encoding it themselves.
async function toMp4(webm, mp4) {
  const { width, height } = DELIVER
  await ffmpeg([
    '-y',
    '-i', webm,
    '-vf',
    [
      `scale=${width}:${height}:force_original_aspect_ratio=decrease:flags=lanczos`,
      `pad=${width}:${height}:(ow-iw)/2:(oh-ih)/2:color=black`,
      `fps=${FPS}`,
    ].join(','),
    '-c:v', 'libx264',
    '-preset', 'slow',
    '-crf', String(CRF),
    '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart',
    '-an', // silent plates
    mp4,
  ])
}

// ---------------------------------------------------------------------------

export async function run() {
  fs.mkdirSync(OUT, { recursive: true })
  fs.mkdirSync(RAW, { recursive: true })

  const selected = SCENES.filter((s) => !ONLY || ONLY.has(s.name))
  if (!selected.length) {
    console.log('no scenes selected')
    return
  }

  const browser = await webkit.launch()
  const failures = []

  for (const { name, fn, opts } of selected) {
    const viewport = opts.viewport ?? CAPTURE
    const context = await browser.newContext({
      viewport,
      // recordVideo lives on the CONTEXT, not the browser, and the file is only
      // flushed when the context closes — not when the page does.
      //
      // Record at the DELIVERY resolution and get legibility from `opts.zoom`
      // instead of from a small viewport. Recording 1280x720 and letting ffmpeg
      // upscale to 1080p invents no detail and leaves UI text soft. And
      // supersampling is not available as a shortcut: WebKit's recordVideo
      // ignores deviceScaleFactor — set it and the page is drawn into the
      // top-left corner of an oversized canvas rather than rasterised at 2x.
      recordVideo: { dir: RAW, size: viewport },
    })
    // The app persists the language choice per-device in localStorage; there is
    // no backend locale, so this is the only way to drive the zh pass.
    await context.addInitScript(
      (locale) => window.localStorage.setItem('orbital.locale', locale),
      LOCALE,
    )

    // Legibility zoom. A 1920x1080 viewport at zoom 1.5 lays the app out at
    // 1280 CSS px — same proportions, same breakpoints as recording at 720p —
    // but every glyph is rasterised into 1080p worth of real pixels instead of
    // being interpolated up afterwards. Applied before first paint so no frame
    // is ever recorded at the unzoomed size.
    const zoom = opts.zoom ?? 1
    if (zoom !== 1) {
      await context.addInitScript((z) => {
        const apply = () => {
          if (document.documentElement) document.documentElement.style.zoom = String(z)
        }
        apply()
        document.addEventListener('DOMContentLoaded', apply)
      }, zoom)
    }

    const page = await context.newPage()
    const video = page.video()
    let failed = null
    try {
      await fn(page)
    } catch (err) {
      failed = String(err).split('\n')[0]
    }

    // Close first, THEN reach for the file — this is the whole trick.
    await context.close()

    const webm = path.join(RAW, `${name}.webm`)
    try {
      // saveAs copies; the original keeps its random hash name, so drop it or
      // every run silts up the raw dir with unidentifiable takes.
      await video.saveAs(webm)
      await video.delete()
    } catch (err) {
      failures.push([name, `no video written: ${String(err).split('\n')[0]}`])
      console.log(`  FAIL  ${name}: no video written`)
      continue
    }

    const mp4 = path.join(OUT, `${name}.mp4`)
    try {
      await toMp4(webm, mp4)
    } catch (err) {
      failures.push([name, String(err).split('\n')[0]])
      console.log(`  FAIL  ${name}: ${String(err).split('\n')[0]}`)
      continue
    }
    if (!KEEP_WEBM) fs.rmSync(webm, { force: true })

    const mb = (fs.statSync(mp4).size / 1e6).toFixed(1)
    if (failed) {
      // The tape still rolled up to the throw, so keep it — a partial clip is
      // often the fastest way to see which selector went missing.
      failures.push([name, failed])
      console.log(`  PART  ${name} (${mb} MB) — scene threw: ${failed}`)
    } else {
      console.log(`  ok    ${name} (${mb} MB)`)
    }
  }

  await browser.close()

  console.log(`\n${LOCALE} pass -> ${OUT}`)
  if (failures.length) {
    console.log(`${failures.length} scene(s) with problems:`)
    for (const [name, err] of failures) console.log(`  - ${name}: ${err}`)
    process.exitCode = 1
  }
}

// ---------------------------------------------------------------------------
// Scenes
//
// Only drive the rig when it is the thing being run. Imported from a scene file
// elsewhere, this module is just the helpers — no recording session, and none
// of the scenes below tagging along uninvited.
// ---------------------------------------------------------------------------

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  // One trivial plate, here to prove the pipeline rather than to ship. Real
  // beats go alongside it.
  scene('hello', async (page) => {
    await home(page)
    await page.locator('aside').first().waitFor({ state: 'visible', timeout: 15_000 })
    await pause(page, 3000)
  })

  await run()
}
