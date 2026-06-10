// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Print catalog rows whose `location` column mentions any of the given filenames.
// Usage from repo root: node web/scripts/keys-for.mjs GlobalSettings.tsx LLMProviderSettings.tsx
// Output: one row per line as  key \t english \t zh
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const CSV = join(here, '..', '..', 'docs', 'i18n', 'ui-terms.zh-Hans.csv');

function parseCsv(text) {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const rows = [];
  let row = [], field = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; }
      else field += c;
    } else if (c === '"') q = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\r') { /* skip */ }
    else if (c === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
    else field += c;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows;
}

const files = process.argv.slice(2);
if (!files.length) { console.error('usage: node web/scripts/keys-for.mjs <File.tsx> [...]'); process.exit(2); }

const rows = parseCsv(readFileSync(CSV, 'utf8'));
const h = rows[0].map((s) => s.trim());
const kI = h.indexOf('key'), eI = h.indexOf('english'), lI = h.indexOf('location'), zI = h.indexOf('proposed_zh_hans');

let n = 0;
for (const r of rows.slice(1)) {
  const loc = r[lI] || '';
  if (files.some((f) => loc.includes(f))) {
    console.log(`${r[kI]}\t${r[eI]}\t${r[zI] || ''}`);
    n++;
  }
}
console.error(`# ${n} rows for: ${files.join(', ')}`);
