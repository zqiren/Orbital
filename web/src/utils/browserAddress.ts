// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * What the live browser view's address field does with what was typed
 * (spec 078 §5.6 navigation, 2026-09-04). Same rules a browser's omnibox
 * follows, reduced to the cases that matter:
 *   - a web URL with a scheme is used as typed;
 *   - something that reads as a host (a dot and no spaces, localhost, an
 *     IP, optionally with a port and a path) gets a scheme — https, or http
 *     for localhost and bare IPs, which never serve TLS in practice;
 *   - anything else is a search.
 * Non-web schemes (file:, javascript:, …) are refused: the route refuses
 * them too, but the field shouldn't even try.
 */

const SEARCH_URL = 'https://www.google.com/search?q=';

const SCHEME = /^[a-z][a-z0-9+.-]*:\/\//i;
const WEB_SCHEME = /^https?:\/\//i;
const LOCAL_HOST = /^(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|(\d{1,3}\.){3}\d{1,3})(:\d{1,5})?(\/.*)?$/i;
const HOST_LIKE = /^[^\s/?#]+\.[^\s/?#]+(:\d{1,5})?(\/\S*)?$/;

export function resolveAddress(input: string): string | null {
  const text = input.trim();
  if (!text) return null;
  if (WEB_SCHEME.test(text)) return text;
  if (SCHEME.test(text)) return null;
  if (LOCAL_HOST.test(text)) return 'http://' + text;
  if (HOST_LIKE.test(text)) return 'https://' + text;
  return SEARCH_URL + encodeURIComponent(text);
}

/** The address field shows nothing for a page that has nothing on it. */
export function displayAddress(url: string | undefined): string {
  if (!url || url === 'about:blank') return '';
  return url;
}
