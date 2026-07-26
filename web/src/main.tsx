// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { WebSocketProvider } from './hooks/useWebSocket'
import { LocaleProvider } from './i18n/LocaleContext'

// Window chrome mode, handed over by the macOS desktop shell as a query param.
// Stamped on <html> before the first render so the titlebar gutter is present
// in the first paint — reading `window.pywebview` instead would be a frame late
// (pywebview injects it asynchronously) and the layout would visibly shift.
// Stamping it once here also means it survives later route navigations that
// rewrite the URL and drop the param.
if (new URLSearchParams(window.location.search).get('chrome') === 'mac-inline') {
  document.documentElement.dataset.chrome = 'mac-inline'
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <LocaleProvider>
      <WebSocketProvider>
        <App />
      </WebSocketProvider>
    </LocaleProvider>
  </StrictMode>,
)
