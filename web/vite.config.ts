// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/// <reference types="vitest" />
import { createReadStream, readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitest/config'
import type { Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import i18nEditor from './dev/i18nEditorPlugin'

// Spec 090: pdf.js loads CMaps, standard fonts, wasm image decoders and the
// CMYK ICC profile from directory URLs at runtime (the ICC profile and the
// colour-management wasm through a synchronous worker XHR), so they cannot be
// bundled imports. Ship the pinned copies beside the app under
// /pdfjs-<version>/ (never a CDN) and serve the same files in dev.
const PDFJS_DIR = fileURLToPath(new URL('./node_modules/pdfjs-dist/', import.meta.url))
const PDFJS_ASSET_DIRS = ['cmaps', 'standard_fonts', 'wasm', 'iccs']

function pdfjsAssets(): Plugin {
  const { version } = JSON.parse(readFileSync(join(PDFJS_DIR, 'package.json'), 'utf8')) as { version: string }
  const prefix = `pdfjs-${version}`
  // quickjs-eval is pdf.js's embedded-JavaScript engine; previews never run PDF scripts.
  const files = PDFJS_ASSET_DIRS.flatMap((dir) =>
    readdirSync(join(PDFJS_DIR, dir))
      .filter((name) => !name.startsWith('quickjs'))
      .map((name) => `${dir}/${name}`),
  )
  const served = new Set(files)
  return {
    name: 'orbital-pdfjs-assets',
    configureServer(server) {
      server.middlewares.use(`/${prefix}/`, (req, res, next) => {
        let rel: string
        try {
          rel = decodeURIComponent((req.url ?? '').split('?')[0]).replace(/^\/+/, '')
        } catch {
          return next()
        }
        // Exact membership in the listing above: no traversal outside it.
        if (!served.has(rel)) return next()
        if (rel.endsWith('.wasm')) res.setHeader('Content-Type', 'application/wasm')
        else if (rel.endsWith('.js')) res.setHeader('Content-Type', 'text/javascript')
        else res.setHeader('Content-Type', 'application/octet-stream')
        createReadStream(join(PDFJS_DIR, rel)).pipe(res)
      })
    },
    generateBundle() {
      for (const rel of files) {
        this.emitFile({ type: 'asset', fileName: `${prefix}/${rel}`, source: readFileSync(join(PDFJS_DIR, rel)) })
      }
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), i18nEditor(), pdfjsAssets()],
  build: {
    outDir: 'dist',
  },
  define: {
    'import.meta.env.VITE_LOCAL_MODE': JSON.stringify('true'),
  },
  server: {
    host: '0.0.0.0',
    proxy: {
      // Default is unchanged (:8000). The override exists because :8000 is
      // normally the INSTALLED Orbital.app, not the repo you are editing — so
      // a backend change you just made appears to 404/422 and reads as a
      // frontend bug. Point the dev UI at a dev daemon with
      // `ORBITAL_DEV_API_PORT=8391 npx vite`.
      // `ws: true` because the live browser view (spec 078) opens a WebSocket
      // under /api/v2/agents/{id}/browser/live; without it the dev proxy never
      // upgrades the connection and the view sits on "Connecting".
      '/api': {
        target: `http://localhost:${process.env.ORBITAL_DEV_API_PORT || 8000}`,
        ws: true,
      },
      '/ws': {
        target: `ws://localhost:${process.env.ORBITAL_DEV_API_PORT || 8000}`,
        ws: true,
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./test-setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
