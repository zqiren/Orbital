// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Dev-only Vite plugin behind the two translation editors. `apply: 'serve'`
// keeps it out of every build. Routes (all under the dev server's origin):
//
//   GET  /__i18n/readme                  README proposal editor page
//   GET  /__i18n/readme/source?file=X    draft markdown  (docs/i18n/pending/X)
//   PUT  /__i18n/readme/source?file=X    save draft markdown
//   GET  /__i18n/overrides               pending UI-string overrides (JSON)
//   POST /__i18n/overrides               merge {key, en?, zh?} — null removes a field
//   GET  /__i18n/asset/docs/...          repo assets for the README preview
//
// Everything lands as plain files under docs/i18n/pending/ so the edits can be
// reviewed and folded into strings.ts / the READMEs in a later step.
import type { IncomingMessage, ServerResponse } from 'node:http';
import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Plugin } from 'vite';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..', '..');
const pendingDir = path.join(repoRoot, 'docs', 'i18n', 'pending');
const overridesFile = path.join(pendingDir, 'ui-overrides.json');
const editorPage = path.join(here, 'readme-editor.html');

const DRAFT_FILES = new Set(['README.md', 'README.en.md']);
const MIME: Record<string, string> = {
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.gif': 'image/gif',
  '.svg': 'image/svg+xml', '.webp': 'image/webp', '.mp4': 'video/mp4',
};

interface Override { en?: string; zh?: string }

async function readOverrides(): Promise<Record<string, Override>> {
  try {
    return JSON.parse(await fs.readFile(overridesFile, 'utf8')) as Record<string, Override>;
  } catch {
    return {};
  }
}

async function writeOverrides(map: Record<string, Override>): Promise<void> {
  await fs.mkdir(pendingDir, { recursive: true });
  const sorted = Object.fromEntries(Object.keys(map).sort().map((k) => [k, map[k]]));
  await fs.writeFile(overridesFile, JSON.stringify(sorted, null, 2) + '\n', 'utf8');
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    req.setEncoding('utf8');
    req.on('data', (chunk: string) => { data += chunk; });
    req.on('end', () => resolve(data));
    req.on('error', reject);
  });
}

function send(res: ServerResponse, status: number, body: string | Buffer, type = 'application/json'): void {
  res.statusCode = status;
  res.setHeader('Content-Type', `${type}; charset=utf-8`);
  res.end(body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<boolean> {
  const url = new URL(req.url ?? '/', 'http://localhost');
  const p = url.pathname;

  if (p === '/__i18n/readme' && req.method === 'GET') {
    send(res, 200, await fs.readFile(editorPage, 'utf8'), 'text/html');
    return true;
  }

  if (p === '/__i18n/readme/source') {
    const file = url.searchParams.get('file') ?? '';
    if (!DRAFT_FILES.has(file)) { send(res, 400, 'unknown draft file', 'text/plain'); return true; }
    const target = path.join(pendingDir, file);
    if (req.method === 'GET') {
      try { send(res, 200, await fs.readFile(target, 'utf8'), 'text/markdown'); }
      catch { send(res, 404, 'no draft yet', 'text/plain'); }
      return true;
    }
    if (req.method === 'PUT') {
      await fs.mkdir(pendingDir, { recursive: true });
      await fs.writeFile(target, await readBody(req), 'utf8');
      send(res, 200, '{"ok":true}');
      return true;
    }
  }

  if (p === '/__i18n/overrides') {
    if (req.method === 'GET') { send(res, 200, JSON.stringify(await readOverrides())); return true; }
    if (req.method === 'POST') {
      const body = JSON.parse((await readBody(req)) || '{}') as { key?: unknown; en?: unknown; zh?: unknown };
      const key = typeof body.key === 'string' ? body.key : '';
      if (!key) { send(res, 400, '{"error":"key required"}'); return true; }
      const map = await readOverrides();
      const entry: Override = { ...(map[key] ?? {}) };
      for (const field of ['en', 'zh'] as const) {
        const value = body[field];
        if (value === undefined) continue;
        if (value === null) delete entry[field];
        else entry[field] = String(value);
      }
      if (Object.keys(entry).length) map[key] = entry; else delete map[key];
      await writeOverrides(map);
      send(res, 200, JSON.stringify(map));
      return true;
    }
  }

  if (p.startsWith('/__i18n/asset/') && req.method === 'GET') {
    const rel = decodeURIComponent(p.slice('/__i18n/asset/'.length));
    const abs = path.resolve(repoRoot, rel);
    if (!abs.startsWith(path.join(repoRoot, 'docs') + path.sep)) { send(res, 403, 'forbidden', 'text/plain'); return true; }
    try {
      const data = await fs.readFile(abs);
      res.statusCode = 200;
      res.setHeader('Content-Type', MIME[path.extname(abs).toLowerCase()] ?? 'application/octet-stream');
      res.end(data);
    } catch {
      send(res, 404, 'not found', 'text/plain');
    }
    return true;
  }

  return false;
}

export default function i18nEditor(): Plugin {
  return {
    name: 'orbital-i18n-editor',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith('/__i18n/')) return next();
        handle(req, res)
          .then((done) => { if (!done) next(); })
          .catch((err: unknown) => send(res, 500, String(err), 'text/plain'));
      });
    },
  };
}
