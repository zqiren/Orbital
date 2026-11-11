#!/usr/bin/env node
// Scrub the test key out of an evidence directory, then fail (exit 1) if any
// copy of it is still there. The key comes from ORBITAL_TEST_OPENCODE_KEY /
// OPENCODE_GO_TEST_KEY in the environment and is never printed.
//
//   node scrub.mjs <evidence-dir> [--require-key]
import { scrubAndVerify } from './lib/scrub.mjs';

const args = process.argv.slice(2);
const dir = args.find((a) => !a.startsWith('--'));
if (!dir) {
  console.error('usage: node scrub.mjs <evidence-dir> [--require-key]');
  process.exit(2);
}
const report = scrubAndVerify(dir, { requireSecret: args.includes('--require-key') });
console.log(
  `scrub: secrets_checked=${report.secrets_checked} rewritten=${(report.rewritten || []).length} ` +
    `deleted=${(report.deleted || []).length} still_found=${(report.still_found || []).length} ok=${report.ok}`,
);
for (const f of report.still_found || []) console.log(`  key still present in: ${f}`);
process.exit(report.ok ? 0 : 1);
