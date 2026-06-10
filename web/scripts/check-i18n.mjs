// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Validates web/src/i18n/strings.ts. Run from repo root: node web/scripts/check-i18n.mjs
// Exit 1 on STRUCTURAL problems (missing en, placeholder mismatch). Missing zh => WARN only.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, '..', '..');
const SRC = join(repoRoot, 'web', 'src');

// strings.ts has no runtime imports, so we can extract the object literal safely.
const tsText = readFileSync(join(SRC, 'i18n', 'strings.ts'), 'utf8');
const start = tsText.indexOf('export const STRINGS = {');
const objStart = tsText.indexOf('{', start);
const objEnd = tsText.indexOf('} as const');
const literal = tsText.slice(objStart, objEnd + 1);
// eslint-disable-next-line no-new-func
const STRINGS = new Function(`return (${literal});`)();

const placeholders = (s) => new Set([...String(s).matchAll(/\{(\w+)\}/g)].map((m) => m[1]));
const eq = (a, b) => a.size === b.size && [...a].every((x) => b.has(x));

let errors = 0, missingZh = 0;
for (const [key, entry] of Object.entries(STRINGS)) {
  if (!entry.en) { console.error(`ERROR ${key}: missing en`); errors++; }
  if (entry.zh === undefined || entry.zh === '') { missingZh++; continue; }
  if (!eq(placeholders(entry.en), placeholders(entry.zh))) {
    console.error(`ERROR ${key}: placeholder mismatch en=${[...placeholders(entry.en)]} zh=${[...placeholders(entry.zh)]}`);
    errors++;
  }
}

console.log(`Checked ${Object.keys(STRINGS).length} keys. ${missingZh} missing zh (warning).`);
if (errors) { console.error(`${errors} structural error(s).`); process.exit(1); }
