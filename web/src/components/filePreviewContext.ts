// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

import { createContext } from 'react';

/**
 * Carries the "open this workspace-relative path in the preview drawer" handler
 * down to the chat's MarkdownContent (spec 002 §3.3). Provided by
 * `ProjectDetail` — which owns `setRoute` — and consumed by `ChatView`, so the
 * handler reaches the deeply-nested renderer without prop-drilling through the
 * out-of-scope `ChatTab`. Default `null` keeps non-chat consumers inert.
 */
export const OpenPathContext = createContext<((path: string) => void) | null>(null);
