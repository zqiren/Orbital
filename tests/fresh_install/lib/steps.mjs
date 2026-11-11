// Orbital — fresh-install test: step runner and evidence.
//
// Every step ends as passed, failed or not_verified (with the reason), and
// the result table is rewritten after each step so a crash or a killed VM
// still leaves an honest record.
import fs from 'node:fs';
import path from 'node:path';

export class NotVerified extends Error {
  constructor(reason) {
    super(reason);
    this.name = 'NotVerified';
  }
}

const ts = () => new Date().toISOString();

export class Run {
  constructor(evidenceDir, meta = {}) {
    this.dir = evidenceDir;
    this.meta = { started_at: ts(), ...meta };
    this.steps = [];
    this.page = null; // set once a browser page exists, for failure screenshots
    this.shotIndex = 0;
    fs.mkdirSync(path.join(this.dir, 'screenshots'), { recursive: true });
    this.logFile = path.join(this.dir, 'scenario.log');
  }

  log(...parts) {
    const line = `${ts()} ${parts.join(' ')}`;
    console.log(line);
    fs.appendFileSync(this.logFile, line + '\n');
  }

  status(id) {
    return this.steps.find((s) => s.id === id)?.status;
  }

  /** Screenshot of the driven page, numbered in step order. */
  async shot(name) {
    if (!this.page) return null;
    const file = path.join(
      'screenshots',
      `${String(++this.shotIndex).padStart(2, '0')}-${name.replace(/[^a-z0-9._-]+/gi, '_')}.png`,
    );
    try {
      await this.page.screenshot({ path: path.join(this.dir, file), fullPage: false, timeout: 15_000 });
      return file;
    } catch (e) {
      this.log(`screenshot ${name} failed: ${e.message}`);
      return null;
    }
  }

  /**
   * Run one step. `needs` lists step ids that must have passed; otherwise
   * the step is recorded not_verified without running.
   */
  async step(id, title, fn, { needs = [] } = {}) {
    const rec = { id, title, status: 'running', started_at: ts(), details: [] };
    this.steps.push(rec);
    const blocked = needs.filter((n) => this.status(n) !== 'passed');
    if (blocked.length) {
      rec.status = 'not_verified';
      rec.reason = `blocked: needs ${blocked.join(', ')} to pass first`;
      this.log(`[${id}] NOT VERIFIED — ${rec.reason}`);
      this.write();
      return rec;
    }
    this.log(`[${id}] ${title} ...`);
    const t0 = Date.now();
    const note = (msg) => {
      rec.details.push(msg);
      this.log(`[${id}]   ${msg}`);
    };
    try {
      const out = await fn(note);
      rec.status = 'passed';
      if (out !== undefined) rec.result = out;
    } catch (e) {
      if (e instanceof NotVerified) {
        rec.status = 'not_verified';
        rec.reason = e.message;
      } else {
        rec.status = 'failed';
        rec.reason = (e && e.message) || String(e);
        rec.stack = e && e.stack ? String(e.stack).split('\n').slice(0, 8).join('\n') : undefined;
      }
    }
    rec.seconds = Math.round((Date.now() - t0) / 100) / 10;
    const s = await this.shot(`${id}-${rec.status}`);
    if (s) rec.screenshot = s;
    this.log(`[${id}] ${rec.status.toUpperCase()}${rec.reason ? ' — ' + rec.reason : ''} (${rec.seconds}s)`);
    this.write();
    return rec;
  }

  write() {
    const results = { ...this.meta, updated_at: ts(), steps: this.steps };
    fs.writeFileSync(path.join(this.dir, 'results.json'), JSON.stringify(results, null, 2));
    const lines = [
      `# Fresh-install scenario results`,
      '',
      ...Object.entries(this.meta).map(([k, v]) => `- ${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`),
      '',
      '| step | status | seconds | reason / details |',
      '|---|---|---|---|',
      ...this.steps.map(
        (s) =>
          `| ${s.id} ${s.title} | ${s.status} | ${s.seconds ?? ''} | ${[s.reason, ...(s.details || [])]
            .filter(Boolean)
            .join('; ')
            .replace(/\|/g, '/')
            .replace(/\n/g, ' ')} |`,
      ),
      '',
    ];
    fs.writeFileSync(path.join(this.dir, 'results.md'), lines.join('\n'));
  }

  get failed() {
    return this.steps.some((s) => s.status === 'failed');
  }
}
