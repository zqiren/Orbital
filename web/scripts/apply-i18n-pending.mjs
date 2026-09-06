// Orbital — An operating system for AI agents
// Copyright (C) 2026 Orbital Contributors
// SPDX-License-Identifier: GPL-3.0-or-later
//
// Fold the translation editor's pending UI overrides into the catalog.
//   node web/scripts/apply-i18n-pending.mjs            # apply + empty the pending file
//   node web/scripts/apply-i18n-pending.mjs --dry-run  # show what would change
//
// Reads docs/i18n/pending/ui-overrides.json ({key: {en?, zh?}}; zh "" means
// "drop zh, fall back to en"), rewrites each matching entry line in
// web/src/i18n/strings.ts in the generator's canonical single-line form, and
// resets the pending file to {}. Unknown keys abort before anything is written.
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, '..', '..');
const STRINGS = join(repoRoot, 'web', 'src', 'i18n', 'strings.ts');
const PENDING = join(repoRoot, 'docs', 'i18n', 'pending', 'ui-overrides.json');
const dryRun = process.argv.includes('--dry-run');

if (!existsSync(PENDING)) { console.log('No pending file; nothing to apply.'); process.exit(0); }
const overrides = JSON.parse(readFileSync(PENDING, 'utf8'));
const keys = Object.keys(overrides);
if (keys.length === 0) { console.log('Pending file is empty; nothing to apply.'); process.exit(0); }

let ts = readFileSync(STRINGS, 'utf8');
const JSTR = '"(?:[^"\\\\]|\\\\.)*"';
const entryRe = (key) => new RegExp(
  `^  ${JSON.stringify(key).replace(/[.*+?^${}()|[\\]\\\\]/g, '\\$&')}: \\{(?: en: (${JSTR})(?:, zh: (${JSTR}))? \\},|\\n    en: (${JSTR}),\\n(?:    zh: (${JSTR}),\\n)?  \\},)$`,
  'm',
);

const changes = [];
for (const key of keys) {
  const m = ts.match(entryRe(key));
  if (!m) { console.error(`ERROR: ${key} not found in strings.ts (or in an unexpected shape)`); process.exit(1); }
  const cur = { en: JSON.parse(m[1] ?? m[3]), zh: m[2] ?? m[4] ? JSON.parse(m[2] ?? m[4]) : undefined };
  const o = overrides[key];
  const next = { en: cur.en, zh: cur.zh };
  if (typeof o.en === 'string' && o.en) next.en = o.en;
  if (typeof o.zh === 'string') next.zh = o.zh === '' ? undefined : o.zh;
  const line = `  ${JSON.stringify(key)}: { en: ${JSON.stringify(next.en)}${next.zh !== undefined ? `, zh: ${JSON.stringify(next.zh)}` : ''} },`;
  changes.push({ key, from: cur, to: next });
  ts = ts.replace(m[0], line);
}

for (const c of changes) {
  console.log(`${c.key}`);
  if (c.from.en !== c.to.en) console.log(`  en: ${JSON.stringify(c.from.en)} -> ${JSON.stringify(c.to.en)}`);
  if (c.from.zh !== c.to.zh) console.log(`  zh: ${JSON.stringify(c.from.zh)} -> ${JSON.stringify(c.to.zh)}`);
}
if (dryRun) { console.log(`\n(dry run) ${changes.length} entr${changes.length === 1 ? 'y' : 'ies'} would change.`); process.exit(0); }
writeFileSync(STRINGS, ts, 'utf8');
writeFileSync(PENDING, '{}\n', 'utf8');
console.log(`\nApplied ${changes.length} entr${changes.length === 1 ? 'y' : 'ies'} to strings.ts; pending file reset. Run: node web/scripts/check-i18n.mjs`);
