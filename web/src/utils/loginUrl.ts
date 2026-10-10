// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// The sign-in link to offer from a sub-agent CLI's login progress line.
// Loopback URLs are the CLI's own OAuth callback server (codex prints
// "Starting local login server on http://localhost:1455." before the real
// auth.openai.com link), and a sentence-final "." or ")" is not part of the URL.
export function pickLoginUrl(line: string): string | null {
  for (const match of line.match(/https?:\/\/\S+/g) ?? []) {
    const url = match.replace(/[.,;:)\]'"]+$/, '');
    if (!/^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])([:/]|$)/i.test(url)) return url;
  }
  return null;
}
