// Orbital — fresh-install test: key scrubbing for evidence.
//
// Evidence leaves the machine (CI artifacts on a public repo, or a copy out
// of a throwaway VM), so every file is rewritten with each secret replaced,
// then the whole tree is scanned again. Any secret still present (in any
// file, text or binary, under any common encoding) fails the check.
import fs from 'node:fs';
import path from 'node:path';

const REDACTED = '[REDACTED-TEST-KEY]';

/** The secrets to scrub: the test key, from env only. */
export function secretsFromEnv(env = process.env) {
  const names = ['ORBITAL_TEST_OPENCODE_KEY', 'OPENCODE_GO_TEST_KEY'];
  const out = new Set();
  for (const n of names) {
    const v = (env[n] || '').trim();
    if (v.length >= 8) out.add(v);
  }
  return [...out];
}

/** Encodings a key could be written in by a log, a URL or a JSON dump. */
function variants(secret) {
  const v = new Set([secret]);
  v.add(encodeURIComponent(secret));
  v.add(JSON.stringify(secret).slice(1, -1));
  v.add(Buffer.from(secret, 'utf8').toString('base64').replace(/=+$/, ''));
  return [...v].filter((s) => s.length >= 8);
}

function* walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(p);
    else if (entry.isFile()) yield p;
  }
}

const TEXT_EXT = new Set([
  '.log', '.txt', '.json', '.jsonl', '.md', '.yml', '.yaml', '.toml', '.html', '.csv', '.xml', '',
]);

/**
 * Rewrite text files with secrets replaced; delete binary files that contain
 * a secret (a screenshot cannot be rewritten). Returns {rewritten, deleted}.
 */
export function scrubDir(dir, secrets) {
  const result = { rewritten: [], deleted: [] };
  if (!fs.existsSync(dir) || secrets.length === 0) return result;
  const all = secrets.flatMap(variants);
  for (const file of walk(dir)) {
    const buf = fs.readFileSync(file);
    const hit = all.some((s) => buf.includes(Buffer.from(s, 'utf8')));
    if (!hit) continue;
    if (TEXT_EXT.has(path.extname(file).toLowerCase())) {
      let text = buf.toString('utf8');
      for (const s of all) text = text.split(s).join(REDACTED);
      fs.writeFileSync(file, text);
      result.rewritten.push(path.relative(dir, file));
    } else {
      fs.rmSync(file);
      result.deleted.push(path.relative(dir, file));
    }
  }
  return result;
}

/** Files under dir that still contain a secret (any variant). */
export function findSecrets(dir, secrets) {
  const found = [];
  if (!fs.existsSync(dir) || secrets.length === 0) return found;
  const all = secrets.flatMap(variants);
  for (const file of walk(dir)) {
    const buf = fs.readFileSync(file);
    if (all.some((s) => buf.includes(Buffer.from(s, 'utf8')))) found.push(path.relative(dir, file));
  }
  return found;
}

/**
 * Scrub, re-check, and write `scrub-report.json`. `ok` is false when a
 * secret survived, or when no secret was available to check against while
 * one was required.
 */
export function scrubAndVerify(dir, { secrets = secretsFromEnv(), requireSecret = false } = {}) {
  fs.mkdirSync(dir, { recursive: true });
  const report = { checked_at: new Date().toISOString(), secrets_checked: secrets.length };
  if (secrets.length === 0) {
    report.ok = !requireSecret;
    report.note = 'no test key in env; nothing to scrub';
  } else {
    Object.assign(report, scrubDir(dir, secrets));
    report.still_found = findSecrets(dir, secrets);
    report.ok = report.still_found.length === 0;
  }
  fs.writeFileSync(path.join(dir, 'scrub-report.json'), JSON.stringify(report, null, 2));
  if (report.ok) fs.writeFileSync(path.join(dir, 'scrub-ok'), report.checked_at + '\n');
  else fs.rmSync(path.join(dir, 'scrub-ok'), { force: true });
  return report;
}
