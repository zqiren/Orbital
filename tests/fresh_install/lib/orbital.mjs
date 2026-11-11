// Orbital — fresh-install test: read-only API checks against the installed
// app's daemon. The UI does the driving; these calls only confirm what the
// UI showed actually landed (session rows with the right role/source).
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export class Api {
  constructor(port) {
    this.base = `http://127.0.0.1:${port}/api/v2`;
  }

  async get(p) {
    const r = await fetch(this.base + p, { signal: AbortSignal.timeout(30_000) });
    if (!r.ok) throw new Error(`GET ${p} -> HTTP ${r.status}`);
    return r.json();
  }

  settings() {
    return this.get('/settings');
  }

  projects() {
    return this.get('/projects');
  }

  async projectByName(name) {
    const list = await this.projects();
    return list.find((p) => p.name === name) || null;
  }

  async sessions(pid) {
    return (await this.get(`/projects/${encodeURIComponent(pid)}/sessions`)).sessions || [];
  }

  /** The project's most recently used session. */
  async latestSession(pid) {
    const list = await this.sessions(pid);
    const key = (s) => s.last_activity_at || s.last_user_at || s.created_at || '';
    return [...list].sort((a, b) => String(key(b)).localeCompare(String(key(a))))[0] || null;
  }

  chat(pid, sid) {
    return this.get(`/agents/${encodeURIComponent(pid)}/chat?session_id=${encodeURIComponent(sid)}&limit=0`);
  }

  subAgents() {
    return this.get('/settings/sub-agents');
  }

  cards() {
    return this.get('/settings/cards');
  }
}

/**
 * Poll until fn() returns a truthy value; throws with the last state on
 * timeout. fn may return {fail: "reason"} to stop at once (a worker's
 * failure row is an outcome, not something to wait out).
 */
export async function waitFor(fn, { timeoutMs = 300_000, intervalMs = 3000, label = 'condition' } = {}) {
  const t0 = Date.now();
  let last;
  while (Date.now() - t0 < timeoutMs) {
    try {
      last = await fn();
    } catch (e) {
      last = `error: ${e.message}`;
    }
    if (last && typeof last === 'object' && last.fail) throw new Error(`${label}: ${last.fail}`);
    if (last && !(typeof last === 'string' && last.startsWith('error: '))) return last;
    await sleep(intervalMs);
  }
  throw new Error(`timed out after ${Math.round(timeoutMs / 1000)}s waiting for ${label}` +
    (typeof last === 'string' ? ` (${last})` : ''));
}

export const text = (row) =>
  typeof row?.content === 'string' ? row.content : JSON.stringify(row?.content ?? '');

/** Rows after the last user row whose content includes `marker`. */
export function rowsAfterUser(rows, marker) {
  let idx = -1;
  rows.forEach((r, i) => {
    if (r.role === 'user' && text(r).includes(marker)) idx = i;
  });
  return idx < 0 ? null : rows.slice(idx + 1);
}
