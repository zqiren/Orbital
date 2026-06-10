// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// One-time generator: docs/i18n/ui-terms.zh-Hans.csv -> web/src/i18n/strings.ts
// Run from repo root: node web/scripts/gen-i18n.mjs
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, '..', '..');
const CSV = join(repoRoot, 'docs', 'i18n', 'ui-terms.zh-Hans.csv');
const OUT = join(repoRoot, 'web', 'src', 'i18n', 'strings.ts');

// RFC-4180 parser: handles quoted fields, embedded commas/newlines, "" escapes, BOM.
function parseCsv(text) {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1); // strip BOM
  const rows = [];
  let row = [], field = '', inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else inQuotes = false;
      } else field += c;
    } else if (c === '"') inQuotes = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\r') { /* ignore, handle on \n */ }
    else if (c === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
    else field += c;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows;
}

// Recover the 3 rows Excel corrupted into "#NAME?" by treating leading "+" as a formula.
const CORRECTIONS = {
  'app.newProject':            { en: '+ New Project', zh: '+ 新建项目空间' },
  'credStore.form.addField':   { en: '+ Add field',   zh: '+ 添加字段' },
  'sessionSidebar.newSession': { en: '+ new session', zh: '+ 新建会话' },
};

const rows = parseCsv(readFileSync(CSV, 'utf8'));
const header = rows[0].map((h) => h.trim());
const idx = (name) => header.indexOf(name);
const kI = idx('key'), eI = idx('english'), zI = idx('proposed_zh_hans');
if (kI < 0 || eI < 0 || zI < 0) throw new Error('CSV missing required columns');

const catalog = {};
for (const r of rows.slice(1)) {
  const key = (r[kI] || '').trim();
  if (!key) continue;
  let en = r[eI] ?? '';
  let zh = r[zI] ?? '';
  if (CORRECTIONS[key]) { en = CORRECTIONS[key].en; zh = CORRECTIONS[key].zh; }
  if (key in catalog) throw new Error(`Duplicate key in CSV: ${key}`);
  catalog[key] = zh && zh !== en ? { en, zh } : { en };
}

const entries = Object.keys(catalog).sort().map((k) => {
  const e = catalog[k];
  const en = JSON.stringify(e.en);
  const zh = e.zh !== undefined ? `, zh: ${JSON.stringify(e.zh)}` : '';
  return `  ${JSON.stringify(k)}: { en: ${en}${zh} },`;
}).join('\n');

const out = `// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// GENERATED from docs/i18n/ui-terms.zh-Hans.csv by web/scripts/gen-i18n.mjs.
// This file is the SOURCE OF TRUTH at runtime. Re-running the generator
// regenerates it; hand-edits are allowed but prefer round-tripping via the CSV.

export interface StringEntry {
  en: string;
  zh?: string;
}

export const STRINGS = {
${entries}
} as const satisfies Record<string, StringEntry>;

export type StringKey = keyof typeof STRINGS;
`;

writeFileSync(OUT, out);
console.log(`Wrote ${Object.keys(catalog).length} strings to ${OUT}`);
