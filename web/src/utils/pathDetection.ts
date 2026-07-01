// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

/**
 * Workspace-relative path detection for clickable chat links (spec 002).
 *
 * Given a candidate string (a markdown-link href or an inline-code span) and
 * the project's absolute `workspace`, decide whether it names a file that the
 * files API can serve — i.e. a workspace-relative path, or an absolute path
 * that prefix-matches the workspace (which is then stripped to its relative
 * form). Anything outside the workspace is unreachable via the files API
 * (`_resolve_path` rejects it server-side) and must NOT be linked.
 *
 * This is detection only: the renderer decides card-vs-chip from `kind`, and a
 * lazy on-click probe (not done here) confirms existence before opening.
 */

export type PathKind = 'card' | 'chip';

export interface DetectedPath {
  /** Workspace-relative, forward-slash path to hand to the files API. */
  relativePath: string;
  /** `card` for previewable artifacts; `chip` for source/other files. */
  kind: PathKind;
}

/**
 * Previewable artifact extensions → render as a card (§0.2). Everything else
 * workspace-pathy renders as an inline chip.
 */
const CARD_EXTENSIONS = new Set([
  'html',
  'htm',
  'png',
  'jpg',
  'jpeg',
  'gif',
  'svg',
  'webp',
  'csv',
  'md',
  'json',
]);

/**
 * Conservative "looks like a workspace-relative file path" shape (§3.2): up to
 * 8 directory segments of word/dot/dash chars, then a filename with an
 * extension of 1–6 alphanumerics. No leading slash, no spaces, no `..`. Anchored
 * so the whole candidate must match — this is what keeps prose like
 * `npm install` or `useState` from being treated as a path.
 */
const RELATIVE_PATH_RE = /^(?:[\w.-]+\/){0,8}[\w.-]+\.[A-Za-z0-9]{1,6}$/;

/** A URL scheme with an authority (http://, https://, file://, …). */
const URL_SCHEME_RE = /^[a-z][a-z0-9+.-]*:\/\//i;

/** A Windows drive-absolute path after `\`→`/` normalization (e.g. `C:/...`). */
const WINDOWS_ABS_RE = /^[A-Za-z]:\//;

/** Forward-slash-normalize and trim trailing slashes for prefix comparison. */
function normalize(p: string): string {
  return p.replace(/\\/g, '/');
}

function stripTrailingSlash(p: string): string {
  return p.replace(/\/+$/, '');
}

/**
 * Decide whether `raw` names a workspace file and, if so, its kind. Returns
 * `null` when the candidate is not a linkable workspace-relative path.
 */
export function detectWorkspacePath(raw: string, workspace: string): DetectedPath | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;

  // Reject obvious non-paths up front (URLs, mailto:, anchors, protocol-relative).
  if (URL_SCHEME_RE.test(trimmed)) return null;
  if (trimmed.startsWith('#') || trimmed.startsWith('mailto:')) return null;

  let candidate = normalize(trimmed);
  if (candidate.startsWith('//')) return null; // protocol-relative

  const ws = stripTrailingSlash(normalize(workspace));
  const isAbsolute = candidate.startsWith('/') || WINDOWS_ABS_RE.test(candidate);

  if (isAbsolute) {
    // Only absolute paths INSIDE the workspace are reachable; strip the prefix.
    // Compare case-insensitively (macOS/Windows are case-preserving but
    // case-insensitive) — mirrors chatTransform.ts's workspace check.
    if (!ws) return null;
    const lc = candidate.toLowerCase();
    const lcWs = ws.toLowerCase();
    if (!lc.startsWith(lcWs + '/')) return null; // outside, or the root itself
    candidate = candidate.slice(ws.length + 1);
  } else {
    // Relative path: drop a leading "./".
    if (candidate.startsWith('./')) candidate = candidate.slice(2);
  }

  // Reject traversal and anything that doesn't look like a file path.
  if (candidate.split('/').some((seg) => seg === '..')) return null;
  // Reject all-numeric "dotted number" candidates (`3.11`, `0.6.8`, `1.0`,
  // `3.14`) — agents write version/python strings in backticks constantly, and
  // they'd otherwise match RELATIVE_PATH_RE and linkify to a chip that 404s. A
  // real file always has a non-numeric character somewhere (`2024.csv`,
  // `src/v2.py` still pass — they contain letters/slashes).
  if (/^[\d.]+$/.test(candidate)) return null;
  if (!RELATIVE_PATH_RE.test(candidate)) return null;

  const ext = candidate.slice(candidate.lastIndexOf('.') + 1).toLowerCase();
  const kind: PathKind = CARD_EXTENSIONS.has(ext) ? 'card' : 'chip';
  return { relativePath: candidate, kind };
}
