#!/usr/bin/env node
// Test-rig only (never product code): a localhost forwarder that adds the
// `x-opencode-session` header OpenCode Go requires on every request, for a
// worker CLI that cannot send custom headers itself (dsh's DEEPSEEK_BASE_URL
// route). Streams both directions; logs method, path and status only (never
// headers or bodies, so the key cannot reach a log).
//
//   node go-header-proxy.mjs --port 18787 [--upstream https://opencode.ai/zen/go]
import http from 'node:http';
import https from 'node:https';

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
};
const port = Number(opt('port', '18787'));
const upstream = new URL(opt('upstream', 'https://opencode.ai/zen/go'));
const session = opt('session', `orbital-fresh-install-${Date.now()}`);
const basePath = upstream.pathname.replace(/\/$/, '');

const server = http.createServer((req, res) => {
  const headers = { ...req.headers };
  delete headers.host;
  delete headers.connection;
  headers['x-opencode-session'] = session;
  const up = https.request(
    {
      protocol: upstream.protocol,
      hostname: upstream.hostname,
      port: upstream.port || 443,
      method: req.method,
      path: basePath + req.url,
      headers,
    },
    (upRes) => {
      console.log(`${new Date().toISOString()} ${req.method} ${req.url} -> ${upRes.statusCode}`);
      res.writeHead(upRes.statusCode || 502, upRes.headers);
      upRes.pipe(res);
    },
  );
  up.on('error', (err) => {
    console.log(`${new Date().toISOString()} ${req.method} ${req.url} -> upstream error ${err.code || err.message}`);
    if (!res.headersSent) res.writeHead(502, { 'content-type': 'text/plain' });
    res.end('upstream error');
  });
  req.pipe(up);
});

server.listen(port, '127.0.0.1', () => {
  console.log(`go-header-proxy listening on http://127.0.0.1:${port} -> ${upstream.href}`);
});
