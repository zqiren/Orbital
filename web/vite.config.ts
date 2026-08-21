// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/// <reference types="vitest" />
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
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
      '/api': `http://localhost:${process.env.ORBITAL_DEV_API_PORT || 8000}`,
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
