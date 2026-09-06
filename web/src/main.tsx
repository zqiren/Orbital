// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { StrictMode, Suspense, lazy } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { WebSocketProvider } from './hooks/useWebSocket'
import { LocaleProvider } from './i18n/LocaleContext'
import { devI18nEditorEnabled } from './i18n/devEditor/enabled'

// Window chrome mode, handed over by the macOS desktop shell as a query param.
// Stamped on <html> before the first render so the titlebar gutter is present
// in the first paint — reading `window.pywebview` instead would be a frame late
// (pywebview injects it asynchronously) and the layout would visibly shift.
// Stamping it once here also means it survives later route navigations that
// rewrite the URL and drop the param.
if (new URLSearchParams(window.location.search).get('chrome') === 'mac-inline') {
  document.documentElement.dataset.chrome = 'mac-inline'
}

// Dev-only click-to-translate overlay (`?i18n=edit` to enable, `?i18n=off` to
// disable). The DEV guard makes the whole branch vanish from production builds.
const DevI18nEditor = import.meta.env.DEV && devI18nEditorEnabled()
  ? lazy(() => import('./i18n/devEditor/I18nEditor'))
  : null

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <LocaleProvider>
      <WebSocketProvider>
        <App />
      </WebSocketProvider>
      {DevI18nEditor && <Suspense fallback={null}><DevI18nEditor /></Suspense>}
    </LocaleProvider>
  </StrictMode>,
)
